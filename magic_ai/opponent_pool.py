"""Opponent pool with TrueSkill ratings and periodic evaluation."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import trueskill

from magic_ai.model import PPOPolicy
from magic_ai.sharded_native import ShardedNativeBatchEncoder, ShardedNativeRolloutDriver


@dataclass
class OpponentEntry:
    path: Path
    tag: str
    rating: trueskill.Rating
    cached_policy: dict[str, torch.Tensor] | None = None


@dataclass
class OpponentPool:
    env: trueskill.TrueSkill = field(default_factory=trueskill.TrueSkill)
    entries: list[OpponentEntry] = field(default_factory=list)
    main_rating: trueskill.Rating | None = None

    def __post_init__(self) -> None:
        if self.main_rating is None:
            self.main_rating = self.env.create_rating()

    def add_snapshot(self, path: Path, tag: str) -> OpponentEntry:
        # Seed a new snapshot at main's current rating: at the moment of snapshot
        # they ARE main, so this is a better prior than a default fresh rating.
        assert self.main_rating is not None
        env = cast(Any, self.env)
        seed_rating = env.create_rating(mu=self.main_rating.mu, sigma=self.main_rating.sigma)
        entry = OpponentEntry(path=path, tag=tag, rating=seed_rating)
        self.entries.append(entry)
        return entry

    def sample(self, rng: random.Random) -> OpponentEntry | None:
        """Sample weighted toward opponents whose μ is close to main's μ.

        Uses a Gaussian weight over |Δμ| with bandwidth tied to the combined
        rating uncertainty; falls back to uniform if all weights collapse.
        """
        if not self.entries:
            return None
        assert self.main_rating is not None
        main_mu = self.main_rating.mu
        main_sigma = self.main_rating.sigma
        weights: list[float] = []
        for entry in self.entries:
            bandwidth_sq = main_sigma * main_sigma + entry.rating.sigma * entry.rating.sigma
            bandwidth_sq = max(bandwidth_sq, 1e-6)
            delta = entry.rating.mu - main_mu
            weights.append(math.exp(-0.5 * delta * delta / bandwidth_sq))
        total = sum(weights)
        if total <= 0.0:
            return rng.choice(self.entries)
        return rng.choices(self.entries, weights=weights, k=1)[0]

    def record_match(self, opponent: OpponentEntry, main_won: bool | None) -> None:
        assert self.main_rating is not None
        env = cast(Any, self.env)
        if main_won is None:
            new_main, new_opp = env.rate_1vs1(self.main_rating, opponent.rating, drawn=True)
        elif main_won:
            new_main, new_opp = env.rate_1vs1(self.main_rating, opponent.rating)
        else:
            new_opp, new_main = env.rate_1vs1(opponent.rating, self.main_rating)
        self.main_rating = new_main
        opponent.rating = new_opp

    def main_rating_metrics(self) -> dict[str, float]:
        assert self.main_rating is not None
        mu = float(self.main_rating.mu)
        sigma = float(self.main_rating.sigma)
        return {
            "trueskill/mu": mu,
            "trueskill/sigma": sigma,
            "trueskill/conservative": mu - 3.0 * sigma,
            "opponent_pool_size": float(len(self.entries)),
        }

    def state_dict(self) -> dict[str, Any]:
        assert self.main_rating is not None
        return {
            "main_rating": {
                "mu": float(self.main_rating.mu),
                "sigma": float(self.main_rating.sigma),
            },
            "entries": [
                {
                    "path": str(entry.path),
                    "tag": entry.tag,
                    "mu": float(entry.rating.mu),
                    "sigma": float(entry.rating.sigma),
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> OpponentPool:
        pool = cls()
        env = cast(Any, pool.env)
        main_rating = state.get("main_rating")
        if isinstance(main_rating, dict):
            pool.main_rating = env.create_rating(
                mu=float(main_rating.get("mu", 25.0)),
                sigma=float(main_rating.get("sigma", 25.0 / 3.0)),
            )
        entries = state.get("entries", [])
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                entry = OpponentEntry(
                    path=Path(str(item.get("path", ""))),
                    tag=str(item.get("tag", "")),
                    rating=env.create_rating(
                        mu=float(item.get("mu", 25.0)),
                        sigma=float(item.get("sigma", 25.0 / 3.0)),
                    ),
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
    pct = 100.0 * threshold / max(1, total_episodes)
    return f"g{threshold:06d}_p{pct:05.1f}"


def opponent_policy_state_dict(policy: PPOPolicy) -> dict[str, torch.Tensor]:
    state_dict = policy.state_dict()
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if not key.startswith(("target_", "spr_"))
    }


def save_snapshot(policy: PPOPolicy, directory: Path, tag: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"snapshot_{tag}.pt"
    torch.save({"policy": policy.state_dict()}, path)
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


def ensure_opponent_cached(entry: OpponentEntry) -> None:
    if entry.cached_policy is not None:
        return
    checkpoint = torch.load(entry.path, map_location="cpu")
    state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint
    entry.cached_policy = {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if not key.startswith(("target_", "spr_"))
    }


def load_opponent_weights(
    opponent: PPOPolicy,
    entry: OpponentEntry,
    device: torch.device,
) -> None:
    ensure_opponent_cached(entry)
    assert entry.cached_policy is not None
    opponent.load_state_dict(entry.cached_policy, strict=False)
    opponent.to(device)


@dataclass
class _EvalGame:
    game: Any
    slot_idx: int
    main_player_idx: int  # 0 or 1 — which side the main policy plays
    opponent: OpponentEntry
    round_idx: int
    action_count: int = 0
    winner_idx: int = -2  # -2 unresolved, -1 draw, 0/1 player


def run_eval_matches(
    *,
    main_policy: PPOPolicy,
    opponent_policy: PPOPolicy,
    opponents: list[OpponentEntry],
    pool: OpponentPool,
    native_encoder: ShardedNativeBatchEncoder,
    native_rollout: ShardedNativeRolloutDriver,
    mage: Any,
    deck_pool: list[dict[str, Any]],
    num_games_per_opponent: int,
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
) -> dict[str, float]:
    """Play eval games against sampled opponents; update ratings; return metrics."""
    if not opponents or num_games_per_opponent < 1:
        return {}

    for opponent in opponents:
        ensure_opponent_cached(opponent)

    total_games_target = len(opponents) * num_games_per_opponent
    num_envs = max(1, min(num_envs, total_games_target))

    saved_main_h: torch.Tensor | None = None
    saved_main_c: torch.Tensor | None = None
    if main_policy.use_lstm:
        saved_main_h = main_policy.live_lstm_h.clone()
        saved_main_c = main_policy.live_lstm_c.clone()
        main_policy.init_lstm_env_states(num_envs)
    if opponent_policy.use_lstm:
        opponent_policy.init_lstm_env_states(num_envs)

    main_wins = 0
    opp_wins = 0
    draws = 0
    round_main_wins = [0 for _ in opponents]
    round_opp_wins = [0 for _ in opponents]
    round_draws = [0 for _ in opponents]

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
        round_idx = game_idx // num_games_per_opponent
        opponent = opponents[round_idx]
        if main_policy.use_lstm:
            main_policy.reset_lstm_env_states([slot_idx])
        if opponent_policy.use_lstm:
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
            main_player_idx=main_player_idx,
            opponent=opponent,
            round_idx=round_idx,
        )

    def fill_slots() -> None:
        while free_slots and next_game_idx < total_games_target:
            live.append(start_game(free_slots.pop()))

    def run_policy(
        policy: PPOPolicy,
        envs: list[_EvalGame],
        players: list[int],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        slot_indices = [env.slot_idx for env in envs]
        parsed = native_encoder.encode_handles(
            [env.game for env in envs],
            perspective_player_indices=players,
        )
        with torch.no_grad():
            steps = policy.sample_native_batch(
                parsed,
                env_indices=slot_indices,
                deterministic=False,
            )
        starts: list[int] = []
        counts: list[int] = []
        selected_cols: list[int] = []
        may_selected: list[int] = []
        cursor = 0
        for step in steps:
            cols = list(step.selected_choice_cols)
            starts.append(cursor)
            counts.append(len(cols))
            selected_cols.extend(cols)
            may_selected.append(step.may_selected)
            cursor += len(cols)
        native_rollout.step_by_choice(
            [env.game for env in envs],
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
                    round_main_wins[env.round_idx] += 1
                    pool.record_match(env.opponent, main_won=True)
                elif env.winner_idx == -1:
                    draws += 1
                    round_draws[env.round_idx] += 1
                    pool.record_match(env.opponent, main_won=None)
                else:
                    opp_wins += 1
                    round_opp_wins[env.round_idx] += 1
                    pool.record_match(env.opponent, main_won=False)
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

    total = main_wins + opp_wins + draws
    denom = float(total) if total else 1.0
    metrics = {
        "eval/games": float(total),
        "eval/main_win_fraction": main_wins / denom,
        "eval/opp_win_fraction": opp_wins / denom,
        "eval/draw_fraction": draws / denom,
    }
    for round_idx, opponent in enumerate(opponents):
        round_total = (
            round_main_wins[round_idx] + round_opp_wins[round_idx] + round_draws[round_idx]
        )
        round_denom = float(round_total) if round_total else 1.0
        metrics[f"eval/round_{round_idx}_main_win_fraction"] = (
            round_main_wins[round_idx] / round_denom
        )
        metrics[f"eval/round_{round_idx}_opp_win_fraction"] = (
            round_opp_wins[round_idx] / round_denom
        )
        metrics[f"eval/round_{round_idx}_draw_fraction"] = round_draws[round_idx] / round_denom
        metrics[f"eval/opp_{opponent.tag}_rating_mu"] = float(opponent.rating.mu)
        metrics[f"eval/opp_{opponent.tag}_rating_sigma"] = float(opponent.rating.sigma)
    return metrics
