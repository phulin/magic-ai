"""Opponent pool with TrueSkill ratings and periodic evaluation."""

from __future__ import annotations

import copy
import itertools
import math
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import trueskill

from magic_ai.model_state import is_opponent_policy_state_key
from magic_ai.native.sharded import (
    ShardedNativeBatchEncoder,
    ShardedNativeRolloutDriver,
)
from magic_ai.slot_encoder.model import PPOPolicy

if TYPE_CHECKING:
    from magic_ai.text_encoder.actor_critic import TextActorCritic


@dataclass
class OpponentEntry:
    path: Path
    tag: str
    rating: trueskill.Rating
    n_games: int = 0
    snapshot_games: int = 0
    cached_policy: dict[str, torch.Tensor] | None = None


@dataclass
class OpponentPool:
    env: trueskill.TrueSkill = field(default_factory=trueskill.TrueSkill)
    entries: list[OpponentEntry] = field(default_factory=list)

    def add_snapshot(self, path: Path, tag: str, *, snapshot_games: int = 0) -> OpponentEntry:
        # Use the previous checkpoint's mean as a temporal prior, but reset
        # uncertainty because each checkpoint is a separate fixed player.
        env = cast(Any, self.env)
        if self.entries:
            prev = self.entries[-1].rating
            seed_rating = env.create_rating(mu=prev.mu)
        else:
            seed_rating = env.create_rating()
        entry = OpponentEntry(
            path=path, tag=tag, rating=seed_rating, snapshot_games=int(snapshot_games)
        )
        self.entries.append(entry)
        return entry

    def sample(self, rng: random.Random) -> OpponentEntry | None:
        """Sample weighted toward opponents whose μ is close to the current
        (most-recent) entry's μ. Bandwidth is tied to combined rating
        uncertainty; falls back to uniform if all weights collapse.
        """
        if not self.entries:
            return None
        current = self.entries[-1].rating
        cur_mu = current.mu
        cur_sigma = current.sigma
        weights: list[float] = []
        for entry in self.entries:
            bandwidth_sq = cur_sigma * cur_sigma + entry.rating.sigma * entry.rating.sigma
            bandwidth_sq = max(bandwidth_sq, 1e-6)
            delta = entry.rating.mu - cur_mu
            weights.append(math.exp(-0.5 * delta * delta / bandwidth_sq))
        total = sum(weights)
        if total <= 0.0:
            return rng.choice(self.entries)
        return rng.choices(self.entries, weights=weights, k=1)[0]

    def record_match(
        self,
        rated: OpponentEntry,
        opponent: OpponentEntry,
        rated_won: bool | None,
    ) -> None:
        env = cast(Any, self.env)
        if rated_won is None:
            new_r, new_o = env.rate_1vs1(rated.rating, opponent.rating, drawn=True)
        elif rated_won:
            new_r, new_o = env.rate_1vs1(rated.rating, opponent.rating)
        else:
            new_o, new_r = env.rate_1vs1(opponent.rating, rated.rating)
        rated.rating = new_r
        opponent.rating = new_o
        rated.n_games += 1
        opponent.n_games += 1

    def current_rating_metrics(self) -> dict[str, float]:
        if not self.entries:
            return {"opponent_pool_size": 0.0}
        current = self.entries[-1].rating
        mu = float(current.mu)
        sigma = float(current.sigma)
        return {
            "trueskill/mu": mu,
            "trueskill/sigma": sigma,
            "trueskill/conservative": mu - 3.0 * sigma,
            "opponent_pool_size": float(len(self.entries)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {
                    "path": str(entry.path),
                    "tag": entry.tag,
                    "mu": float(entry.rating.mu),
                    "sigma": float(entry.rating.sigma),
                    "n_games": int(entry.n_games),
                    "snapshot_games": int(entry.snapshot_games),
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> OpponentPool:
        pool = cls()
        env = cast(Any, pool.env)
        entries = state.get("entries", [])
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                tag = str(item.get("tag", ""))
                snapshot_games_raw = item.get("snapshot_games")
                if snapshot_games_raw is None:
                    parsed = snapshot_games_from_tag(tag)
                    snapshot_games = int(parsed) if parsed is not None else 0
                else:
                    snapshot_games = int(snapshot_games_raw)
                entry = OpponentEntry(
                    path=Path(str(item.get("path", ""))),
                    tag=tag,
                    rating=env.create_rating(
                        mu=float(item.get("mu", 25.0)),
                        sigma=float(item.get("sigma", 25.0 / 3.0)),
                    ),
                    n_games=int(item.get("n_games", 0)),
                    snapshot_games=snapshot_games,
                )
                pool.entries.append(entry)
        return pool


@dataclass
class SnapshotSchedule:
    """Fires at 1%, 2%, then every 2% of the planned episode count."""

    total_episodes: int
    thresholds: list[int]  # required completed-game counts, ascending
    next_idx: int = 0

    @classmethod
    def build(cls, total_episodes: int) -> SnapshotSchedule:
        if total_episodes < 1:
            raise ValueError("total_episodes must be at least 1")
        fractions = [0.01, 0.02]
        pct = 4
        while pct <= 100:
            fractions.append(pct / 100.0)
            pct += 2
        thresholds: list[int] = []
        for frac in fractions:
            t = max(1, int(round(frac * total_episodes)))
            if thresholds and t <= thresholds[-1]:
                t = thresholds[-1] + 1
            if t > total_episodes:
                break
            thresholds.append(t)
        return cls(total_episodes=total_episodes, thresholds=thresholds)

    def fire(self, completed_games: int) -> list[int]:
        """Return thresholds that have been crossed since the last call."""
        fired: list[int] = []
        while (
            self.next_idx < len(self.thresholds)
            and completed_games >= self.thresholds[self.next_idx]
        ):
            fired.append(self.thresholds[self.next_idx])
            self.next_idx += 1
        return fired


def snapshot_tag(threshold: int, total_episodes: int) -> str:
    """Tag a snapshot by absolute games count plus a (then-current) pct hint.

    The leading ``g{games}`` segment is the canonical, schedule-independent
    identifier — it survives extending a run by raising ``--episodes``.
    The trailing ``p{pct}`` is purely a human-readable hint at the moment
    the snapshot was taken; it must not be parsed for scheduling.
    """
    pct = 100.0 * threshold / max(1, total_episodes)
    return f"g{int(threshold):06d}_p{pct:05.1f}"


def snapshot_games_from_tag(tag: str) -> int | None:
    """Parse the absolute game count from a ``g{games}_p{pct}`` snapshot tag."""
    import re

    match = re.match(r"g(\d+)", tag)
    return int(match.group(1)) if match is not None else None


def opponent_policy_state_dict(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a clone of the policy's weights suitable for opponent use.

    Tensors stay on their current device (typically the training GPU) so that
    subsequent ``load_opponent_weights`` calls do a cheap D2D copy rather than
    a host→device transfer.
    """
    state_dict = policy.state_dict()
    return {
        key: value.detach().clone()
        for key, value in state_dict.items()
        if is_opponent_policy_state_key(key)
    }


def save_snapshot(policy: torch.nn.Module, directory: Path, tag: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"snapshot_{tag}.pt"
    torch.save({"policy": opponent_policy_state_dict(policy)}, path)
    return path


def build_opponent_policy(main_policy: PPOPolicy, device: torch.device) -> PPOPolicy:
    """Create a lightweight inference-only clone of the main policy's architecture."""
    encoder = copy.deepcopy(main_policy.game_state_encoder)
    opponent = PPOPolicy(
        encoder,
        hidden_dim=main_policy.hidden_dim,
        hidden_layers=main_policy.hidden_layers,
        max_options=main_policy.max_options,
        max_targets_per_option=main_policy.max_targets_per_option,
        rollout_capacity=1,
        decision_capacity=8,
        use_lstm=main_policy.use_lstm,
        spr_enabled=False,
        validate=main_policy.validate,
        compile_forward=main_policy.compile_forward,
    ).to(device)
    opponent.eval()
    for p in opponent.parameters():
        p.requires_grad_(False)
    return opponent


def build_text_opponent_policy(
    main_policy: TextActorCritic, device: torch.device
) -> TextActorCritic:
    """Inference-only clone of a TextActorCritic that does not share replay state.

    Constructs a fresh ``TextActorCritic`` with the same config and copies
    weights via ``load_state_dict`` rather than ``copy.deepcopy``. Deepcopy
    walks every submodule and ends up cloning the ``torch.compile`` wrappers
    on the encoder forward; the cloned wrappers don't share dynamo's
    in-memory graph cache with the original, so the opponent's first call
    re-traces from scratch even though the inductor on-disk cache would
    otherwise have hit. State-dict copy avoids that — the new instance's
    compile wrappers are constructed at ``__init__`` time and pick up the
    on-disk cache normally.
    """

    from magic_ai.text_encoder.actor_critic import TextActorCritic

    # Skip the HF warm-init for the opponent: the trained main policy's
    # state_dict is loaded over it on the next line, so re-downloading and
    # copying the Ettin weights here would just be thrown away (and would
    # emit a confusing second LOAD REPORT after MLM/RL boundaries).
    cfg = main_policy.policy.cfg
    if cfg.encoder.hf_model_name is not None:
        cfg = replace(cfg, encoder=replace(cfg.encoder, hf_model_name=None))
    opponent = TextActorCritic(cfg).to(device)

    # ``live_lstm_h``/``live_lstm_c`` are per-env recurrence buffers sized
    # by the main policy's ``num_envs``; the opponent runs with 0 envs so
    # those shapes won't match. Skip them — they're re-initialized below.
    state = {
        k: v
        for k, v in main_policy.state_dict().items()
        if not k.startswith(("live_lstm_h", "live_lstm_c"))
    }
    opponent.load_state_dict(state, strict=False)

    opponent.rollout_buffer = None
    opponent.spr_enabled = False
    opponent.eval()
    for p in opponent.parameters():
        p.requires_grad_(False)
    opponent._num_envs = 0
    return opponent


def ensure_opponent_cached(entry: OpponentEntry, device: torch.device) -> None:
    """Pin this entry's weights on ``device``, loading from disk if needed."""
    if entry.cached_policy is not None:
        first = next(iter(entry.cached_policy.values()), None)
        if first is None or first.device == device:
            return
        entry.cached_policy = {k: v.to(device) for k, v in entry.cached_policy.items()}
        return
    checkpoint = torch.load(entry.path, map_location=device)
    state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint
    entry.cached_policy = {
        key: value.detach().clone()
        for key, value in state_dict.items()
        if is_opponent_policy_state_key(key)
    }


def load_opponent_weights(
    opponent: PPOPolicy,
    entry: OpponentEntry,
    device: torch.device,
) -> None:
    ensure_opponent_cached(entry, device)
    assert entry.cached_policy is not None
    opponent.load_state_dict(entry.cached_policy, strict=False)


@contextmanager
def _disable_text_replay_capture(*policies: Any) -> Iterator[None]:
    """Temporarily run text policies in inference mode without replay writes."""

    from magic_ai.text_encoder.actor_critic import TextActorCritic

    saved: list[tuple[TextActorCritic, Any]] = []
    try:
        for policy in policies:
            if isinstance(policy, TextActorCritic):
                saved.append((policy, policy.rollout_buffer))
                policy.rollout_buffer = None
        yield
    finally:
        for policy, rollout_buffer in saved:
            policy.rollout_buffer = rollout_buffer


def distribute_games_by_recency(
    entries: list[OpponentEntry],
    total_games: int,
    tau: float,
) -> list[OpponentEntry]:
    """Assign ``total_games`` across ``entries`` with an exp(-age/tau) bias.

    Age is measured in positions from the newest entry (age=0 for the last
    element in ``entries``). ``tau <= 0`` produces a uniform split. The return
    order groups all games for the same entry together (oldest-first), which
    keeps opponent-weight swaps bounded during evaluation.
    """
    if not entries or total_games <= 0:
        return []
    n = len(entries)
    if tau <= 0:
        weights = [1.0] * n
    else:
        weights = [math.exp(-(n - 1 - i) / tau) for i in range(n)]
    total_w = sum(weights)
    raw = [w / total_w * total_games for w in weights]
    counts = [int(math.floor(x)) for x in raw]
    remainder = total_games - sum(counts)
    if remainder > 0:
        order = sorted(range(n), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:remainder]:
            counts[i] += 1
    out: list[OpponentEntry] = []
    for entry, c in zip(entries, counts):
        out.extend([entry] * c)
    return out


@dataclass
class _EvalGame:
    game: Any
    slot_idx: int
    game_idx: int  # deterministic index, used to order deferred rating updates
    main_player_idx: int  # 0 or 1 — which side the main policy plays
    opponent: OpponentEntry
    action_count: int = 0
    winner_idx: int = -2  # -2 unresolved, -1 draw, 0/1 player


def run_eval_matches(
    *,
    main_policy: Any,
    opponent_policy: Any,
    game_opponents: list[OpponentEntry],
    pool: OpponentPool,
    current_entry: OpponentEntry,
    native_encoder: ShardedNativeBatchEncoder,
    native_rollout: ShardedNativeRolloutDriver,
    mage: Any,
    deck_pool: list[dict[str, Any]],
    num_envs: int,
    max_steps_per_game: int,
    max_options: int,
    max_targets_per_option: int,
    hand_size: int,
    name_a: str,
    name_b: str,
    no_shuffle: bool,
    seed_base: int,
    rng: random.Random,
    text_max_tokens: int | None = None,
) -> dict[str, float]:
    """Play one eval game per entry in ``game_opponents``; update ratings; return metrics."""
    total_games_target = len(game_opponents)
    if total_games_target == 0:
        return {}

    unique_opponents: list[OpponentEntry] = []
    seen_paths: set[Path] = set()
    for opp in game_opponents:
        if opp.path not in seen_paths:
            seen_paths.add(opp.path)
            unique_opponents.append(opp)
    for opponent in unique_opponents:
        ensure_opponent_cached(opponent, main_policy.device)

    num_envs = max(1, min(num_envs, total_games_target))

    saved_main_h: torch.Tensor | None = None
    saved_main_c: torch.Tensor | None = None
    saved_main_num_envs: int = 0
    saved_main_players_per_env: int = 2
    main_uses_lstm = bool(getattr(main_policy, "use_lstm", True))
    opp_uses_lstm = bool(getattr(opponent_policy, "use_lstm", True))
    if main_uses_lstm:
        saved_main_h = main_policy.live_lstm_h.clone()
        saved_main_c = main_policy.live_lstm_c.clone()
        saved_main_num_envs = int(getattr(main_policy, "_num_envs", 0))
        saved_main_players_per_env = int(getattr(main_policy, "_players_per_env", 2))
        main_policy.init_lstm_env_states(num_envs)
    if opp_uses_lstm:
        opponent_policy.init_lstm_env_states(num_envs)

    main_wins = 0
    opp_wins = 0
    draws = 0
    per_opp_main_wins: dict[str, int] = {opp.tag: 0 for opp in unique_opponents}
    per_opp_opp_wins: dict[str, int] = {opp.tag: 0 for opp in unique_opponents}
    per_opp_draws: dict[str, int] = {opp.tag: 0 for opp in unique_opponents}
    # Deferred outcomes: (game_idx, opponent, main_won). Applied after the
    # eval loop in game_idx order so rating updates are reproducible and
    # independent of non-deterministic game-completion timing.
    outcomes: list[tuple[int, OpponentEntry, bool | None]] = []

    free_slots = list(range(num_envs - 1, -1, -1))
    live: list[_EvalGame] = []
    next_game_idx = 0

    def start_game(slot_idx: int) -> _EvalGame:
        nonlocal next_game_idx
        game_idx = next_game_idx
        next_game_idx += 1
        seed = seed_base + game_idx
        deck_rng = random.Random(seed)
        deck_a = deck_rng.choice(deck_pool)
        deck_b = deck_rng.choice(deck_pool)
        main_player_idx = rng.randrange(2)
        opponent = game_opponents[game_idx]
        if main_uses_lstm:
            main_policy.reset_lstm_env_states([slot_idx])
        if opp_uses_lstm:
            opponent_policy.reset_lstm_env_states([slot_idx])
        return _EvalGame(
            game=mage.new_game(
                deck_a,
                deck_b,
                name_a=name_a,
                name_b=name_b,
                seed=seed,
                shuffle=not no_shuffle,
                hand_size=hand_size,
            ),
            slot_idx=slot_idx,
            game_idx=game_idx,
            main_player_idx=main_player_idx,
            opponent=opponent,
        )

    def fill_slots() -> None:
        while free_slots and next_game_idx < total_games_target:
            live.append(start_game(free_slots.pop()))

    from magic_ai.text_encoder.actor_critic import TextActorCritic

    def run_policy(
        policy: Any,
        envs: list[_EvalGame],
        players: list[int],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        slot_indices = [env.slot_idx for env in envs]
        games = [env.game for env in envs]
        if isinstance(policy, TextActorCritic):
            if text_max_tokens is None:
                raise ValueError("run_eval_matches: text_max_tokens is required for text policies")
            native_batch, nat_outputs = native_encoder.encode_tokens_packed(
                games,
                perspective_player_indices=players,
                max_tokens=text_max_tokens,
                max_options=max_options,
                max_targets=max_targets_per_option,
                max_card_refs=256,
            )
            packed_text_batch = nat_outputs.to_packed_text_batch(
                trim=False, derive_token_metadata=False
            )
            with torch.no_grad():
                sample = policy.sample_native_tensor_batch(
                    native_batch=native_batch,
                    env_indices=slot_indices,
                    perspective_player_indices=players,
                    packed_batch=packed_text_batch,
                    deterministic=False,
                )
            counts = list(sample.decision_counts)
            selected_cols = list(sample.selected_choice_cols)
            may_selected = list(sample.may_selected)
            starts = list(itertools.accumulate(counts, initial=0))[:-1]
        else:
            parsed = native_encoder.encode_handles(
                games,
                perspective_player_indices=players,
            )
            with torch.no_grad():
                steps = policy.sample_native_batch(
                    parsed,
                    env_indices=slot_indices,
                    deterministic=False,
                )
            starts = []
            counts = []
            selected_cols = []
            may_selected = []
            cursor = 0
            for step in steps:
                cols = list(step.selected_choice_cols)
                starts.append(cursor)
                counts.append(len(cols))
                selected_cols.extend(cols)
                may_selected.append(step.may_selected)
                cursor += len(cols)
        native_rollout.step_by_choice(
            games,
            decision_starts=starts,
            decision_counts=counts,
            selected_choice_cols=selected_cols,
            may_selected=may_selected,
            max_options=max_options,
            max_targets_per_option=max_targets_per_option,
        )
        for env in envs:
            env.action_count += 1
        return starts, counts, selected_cols, may_selected

    with _disable_text_replay_capture(main_policy, opponent_policy):
        fill_slots()
        while live:
            ready_t, over_t, player_t, winner_t = native_rollout.poll([env.game for env in live])
            still_live: list[_EvalGame] = []
            main_envs: list[_EvalGame] = []
            main_players: list[int] = []
            opp_groups: dict[Path, tuple[OpponentEntry, list[_EvalGame], list[int]]] = {}

            for idx, env in enumerate(live):
                over = bool(int(over_t[idx]))
                hit_cap = env.action_count >= max_steps_per_game
                if over or hit_cap:
                    env.winner_idx = int(winner_t[idx]) if over else -1
                    env.game.close()
                    if env.winner_idx == env.main_player_idx:
                        main_wins += 1
                        per_opp_main_wins[env.opponent.tag] += 1
                        outcomes.append((env.game_idx, env.opponent, True))
                    elif env.winner_idx == -1:
                        draws += 1
                        per_opp_draws[env.opponent.tag] += 1
                        outcomes.append((env.game_idx, env.opponent, None))
                    else:
                        opp_wins += 1
                        per_opp_opp_wins[env.opponent.tag] += 1
                        outcomes.append((env.game_idx, env.opponent, False))
                    free_slots.append(env.slot_idx)
                    continue
                still_live.append(env)
                if not bool(int(ready_t[idx])):
                    continue
                player = int(player_t[idx])
                if player == env.main_player_idx:
                    main_envs.append(env)
                    main_players.append(player)
                else:
                    group = opp_groups.get(env.opponent.path)
                    if group is None:
                        opp_groups[env.opponent.path] = (env.opponent, [env], [player])
                    else:
                        group[1].append(env)
                        group[2].append(player)
            live = still_live

            if main_envs:
                run_policy(main_policy, main_envs, main_players)
            for opponent, opp_envs, opp_players in opp_groups.values():
                load_opponent_weights(opponent_policy, opponent, main_policy.device)
                run_policy(opponent_policy, opp_envs, opp_players)

            fill_slots()

    if saved_main_h is not None and saved_main_c is not None:
        main_policy.live_lstm_h = saved_main_h
        main_policy.live_lstm_c = saved_main_c
        main_policy._num_envs = saved_main_num_envs
        main_policy._players_per_env = saved_main_players_per_env

    outcomes.sort(key=lambda t: t[0])
    for _, opp_entry, main_won in outcomes:
        pool.record_match(current_entry, opp_entry, rated_won=main_won)

    total = main_wins + opp_wins + draws
    denom = float(total) if total else 1.0
    metrics = {
        "eval/games": float(total),
        "eval/main_win_fraction": main_wins / denom,
        "eval/opp_win_fraction": opp_wins / denom,
        "eval/draw_fraction": draws / denom,
    }
    for opponent in unique_opponents:
        tag = opponent.tag
        opp_total = per_opp_main_wins[tag] + per_opp_opp_wins[tag] + per_opp_draws[tag]
        opp_denom = float(opp_total) if opp_total else 1.0
        metrics[f"eval/opp_{tag}_games"] = float(opp_total)
        metrics[f"eval/opp_{tag}_main_win_fraction"] = per_opp_main_wins[tag] / opp_denom
        metrics[f"eval/opp_{tag}_opp_win_fraction"] = per_opp_opp_wins[tag] / opp_denom
        metrics[f"eval/opp_{tag}_draw_fraction"] = per_opp_draws[tag] / opp_denom
        metrics[f"eval/opp_{tag}_rating_mu"] = float(opponent.rating.mu)
        metrics[f"eval/opp_{tag}_rating_sigma"] = float(opponent.rating.sigma)
    return metrics
