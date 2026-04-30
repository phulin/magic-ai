"""Text-only Python rollout worker (PR 13-D / §10 PR #4).

End-to-end pipeline that plays real Magic games using the text encoder
(cache + render-plan emitter + assembler + RecurrentTextPolicy) without
touching ``magic_ai/model.py`` or the slot-encoder ``PPOPolicy``.

This is the slow Python path that proves the pipeline. It produces
``(snapshot, action, reward)`` tuples for distillation experiments and is
deliberately not performance tuned — that's the next step.

Per-player LSTM state
---------------------

Each game has *two* private LSTM hidden states, one per perspective player.
When player ``p`` has priority, the policy is run with state ``p`` and only
state ``p`` is updated. This matches how RnaD's two-player v-trace already
treats per-player recurrence.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import mage
import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.actions import (
    ActionRequest,
    action_from_attackers,
    action_from_choice_accepted,
    action_from_choice_color,
    action_from_choice_index,
    action_from_priority_candidate,
    build_priority_candidates,
)
from magic_ai.game_state import (
    GAME_INFO_DIM,
    ZONE_SLOT_COUNT,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
)
from magic_ai.slot_encoder.native_encoder import NativeBatchEncoder, NativeEncodingError
from magic_ai.text_encoder.assembler import assemble_batch
from magic_ai.text_encoder.card_cache import CardTokenCache
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy
from magic_ai.text_encoder.render import OracleEntry
from magic_ai.text_encoder.render_plan import emit_render_plan

OPTION_SCALAR_DIM = 14
TARGET_SCALAR_DIM = 2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TextRolloutStep:
    snapshot: GameStateSnapshot
    legal_options: list[PendingOptionState]
    chosen_option_idx: int
    chosen_target_idx: int | None
    reward: float
    perspective_player_idx: int


@dataclass
class TextRolloutEpisode:
    steps: list[TextRolloutStep] = field(default_factory=list)
    winner_player_idx: int | None = None
    turns: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_card_row_lookup(cache: CardTokenCache):
    """Return a callable mapping card name -> 1-indexed cache row id (0 = unknown)."""

    name_to_row: dict[str, int] = {}
    for idx, name in enumerate(cache.row_to_name):
        if idx == 0:
            continue  # row 0 = UNKNOWN_NAME sentinel
        if name and name not in name_to_row:
            name_to_row[name] = idx

    def lookup(name: str) -> int:
        return name_to_row.get(name, 0)

    return lookup


def _make_tokenize_fn(tokenizer: PreTrainedTokenizerFast):
    def tokenize(s: str) -> list[int]:
        return list(tokenizer.encode(s, add_special_tokens=False))

    return tokenize


def _build_card_name_to_row(cache: CardTokenCache) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, name in enumerate(cache.row_to_name):
        if idx == 0 or not name:
            continue
        out.setdefault(name, idx)
    return out


def _load_mage_ffi() -> tuple[Any, Any]:
    if getattr(mage, "_lib", None) is None or getattr(mage, "_ffi", None) is None:
        mage.load()
    lib = getattr(mage, "_lib", None)
    ffi = getattr(mage, "_ffi", None)
    if lib is None or ffi is None:
        raise NativeEncodingError("mage native library is not loaded")
    return lib, ffi


def _categorical_sample(
    logits: Tensor,
    mask: Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> int:
    """Mask-aware categorical sample. Returns int index.

    ``logits`` and ``mask`` are 1-D tensors of equal length. Masked-out
    entries are ignored. Falls back to argmax if ``temperature <= 0``.
    Returns -1 if no entry is unmasked.
    """

    valid = mask.bool()
    if not bool(valid.any()):
        return -1
    logits = logits.detach().clone()
    logits[~valid] = float("-inf")
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    scaled = logits / float(temperature)
    # Softmax with -inf entries is well-defined as long as at least one is finite.
    probs = torch.softmax(scaled, dim=-1)
    # Guard against numerical zeros across all valid entries.
    if not bool(torch.isfinite(probs).any()) or float(probs.sum().item()) <= 0.0:
        # Uniform over valid.
        probs = valid.to(torch.float32)
        probs = probs / probs.sum()
    idx = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
    return idx


# ---------------------------------------------------------------------------
# Action translation
# ---------------------------------------------------------------------------


def _default_action_for(pending: PendingState) -> ActionRequest:
    """Best-effort default action for a pending we can't or won't decide.

    Falls back to whatever the engine is most likely to accept to advance
    the game without crashing.
    """

    kind = pending.get("kind", "") or ""
    if kind == "priority":
        return cast(ActionRequest, {"kind": "pass"})
    if kind == "attackers":
        return cast(ActionRequest, {"attackers": []})
    if kind == "blockers":
        return cast(ActionRequest, {"blockers": []})
    if kind == "may":
        return action_from_choice_accepted(False)
    # mode / number / cards_from_hand / mana_color / permanent / card_from_library:
    # selecting the first option index is the safest engine-accepted fallback.
    return action_from_choice_index(0)


def _translate_action(
    pending: PendingState,
    chosen_option_idx: int,
    chosen_target_idx: int | None,
) -> ActionRequest:
    """Turn (option_idx, target_idx) into an ``ActionRequest`` for ``game.step``.

    For ``priority`` pendings we use ``build_priority_candidates`` to find the
    flattened (option, target) entry whose payload exactly matches the
    selection. For non-priority pendings we route to the matching
    ``action_from_*`` helper.
    """

    options = pending.get("options", []) or []
    if not options or not (0 <= chosen_option_idx < len(options)):
        return _default_action_for(pending)
    option = options[chosen_option_idx]
    kind = pending.get("kind", "") or ""

    if kind == "priority":
        candidates = build_priority_candidates(pending)
        # Find a candidate matching the chosen (option_idx, target_idx).
        for cand in candidates:
            if cand.option_index != chosen_option_idx:
                continue
            if chosen_target_idx is None:
                if cand.target_index is None:
                    return action_from_priority_candidate(cand)
            else:
                if cand.target_index == chosen_target_idx:
                    return action_from_priority_candidate(cand)
        # Fallback: any candidate with this option_idx, else default.
        for cand in candidates:
            if cand.option_index == chosen_option_idx:
                return action_from_priority_candidate(cand)
        return _default_action_for(pending)

    if kind == "attackers":
        # Single-option attacker selection: declare just this one attacker.
        permanent_id = option.get("permanent_id", "") or ""
        if permanent_id:
            return action_from_attackers(
                pending, [True if i == chosen_option_idx else False for i in range(len(options))]
            )
        return cast(ActionRequest, {"attackers": []})

    if kind == "blockers":
        # Assign this blocker to its first valid target if any.
        targets = option.get("valid_targets", []) or []
        if targets and chosen_target_idx is not None and 0 <= chosen_target_idx < len(targets):
            blocker_id = option.get("permanent_id", "") or ""
            attacker_id = targets[chosen_target_idx].get("id", "") or ""
            if blocker_id and attacker_id:
                return cast(
                    ActionRequest,
                    {"blockers": [{"blocker": blocker_id, "attacker": attacker_id}]},
                )
        return cast(ActionRequest, {"blockers": []})

    if kind == "may":
        # ``may`` is a two-option (accept/decline) choice. Map option idx 0 ->
        # accept, 1 -> decline; safer than guessing labels.
        return action_from_choice_accepted(chosen_option_idx == 0)

    if kind == "mana_color":
        color = option.get("color") or option.get("label") or ""
        if color:
            return action_from_choice_color(str(color))
        return action_from_choice_index(chosen_option_idx)

    # mode / number / cards_from_hand / permanent / card_from_library / unknown:
    # selected_index by option position is the broadest engine-accepted shape.
    return action_from_choice_index(chosen_option_idx)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclass
class _PlayerLSTM:
    h: Tensor
    c: Tensor


class TextRolloutWorker:
    """Plays text-encoder-driven episodes against the mage engine."""

    def __init__(
        self,
        policy: RecurrentTextPolicy,
        cache: CardTokenCache,
        tokenizer: PreTrainedTokenizerFast,
        *,
        max_tokens: int = 2048,
        device: torch.device | str = "cpu",
        sampling_temperature: float = 1.0,
        oracle: dict[str, OracleEntry] | None = None,
        seed: int | None = None,
        use_native_render_plan: bool = False,
        render_plan_capacity: int = 4096,
        max_options: int = 64,
        max_targets_per_option: int = 4,
        dedup_card_bodies: bool = False,
    ) -> None:
        self.policy = policy
        self.cache = cache
        self.tokenizer = tokenizer
        self.max_tokens = int(max_tokens)
        self.device = torch.device(device)
        self.sampling_temperature = float(sampling_temperature)
        self.oracle = oracle
        self.use_native_render_plan = bool(use_native_render_plan)
        self.render_plan_capacity = int(render_plan_capacity)
        self.max_options = int(max_options)
        self.max_targets_per_option = int(max_targets_per_option)
        self.dedup_card_bodies = bool(dedup_card_bodies)
        self.policy.to(self.device)
        self.policy.eval()

        self._card_row_lookup = _build_card_row_lookup(cache)
        self._tokenize_fn = _make_tokenize_fn(tokenizer)
        self._native_encoder: NativeBatchEncoder | None = None
        if self.use_native_render_plan:
            lib, ffi = _load_mage_ffi()
            self._native_encoder = NativeBatchEncoder(
                max_options=self.max_options,
                max_targets_per_option=self.max_targets_per_option,
                max_cached_choices=max(
                    self.max_options,
                    self.max_options * max(1, self.max_targets_per_option),
                ),
                zone_slot_count=ZONE_SLOT_COUNT,
                game_info_dim=GAME_INFO_DIM,
                option_scalar_dim=OPTION_SCALAR_DIM,
                target_scalar_dim=TARGET_SCALAR_DIM,
                lib=lib,
                ffi=ffi,
                card_name_to_row=_build_card_name_to_row(cache),
                emit_render_plan=True,
                render_plan_capacity=self.render_plan_capacity,
                dedup_card_bodies=self.dedup_card_bodies,
            )
            if not self._native_encoder.is_available:
                raise NativeEncodingError("native render-plan encoder is unavailable")

        if seed is not None:
            self._gen: torch.Generator | None = torch.Generator(device="cpu")
            self._gen.manual_seed(int(seed))
        else:
            self._gen = None

    # --- internals --------------------------------------------------------

    def _init_states(self) -> list[_PlayerLSTM]:
        states: list[_PlayerLSTM] = []
        for _ in range(2):
            h, c = self.policy.init_state(batch_size=1, device=self.device)
            states.append(_PlayerLSTM(h=h, c=c))
        return states

    def _score_step(
        self,
        game: Any,
        snapshot: GameStateSnapshot,
        legal_options: Sequence[PendingOptionState],
        state: _PlayerLSTM,
        perspective_player_idx: int,
    ) -> tuple[int, int | None, _PlayerLSTM] | None:
        """Run policy for one priority step. Returns (opt_idx, tgt_idx, new_state) or None.

        ``None`` means the assembler / emitter could not produce a usable
        batch (overflow, missing cards, etc.) and the caller should punt.
        """

        if self._native_encoder is not None:
            try:
                native = self._native_encoder.encode_handles(
                    [game],
                    perspective_player_indices=[perspective_player_idx],
                )
                if native.render_plan is None or native.render_plan_lengths is None:
                    logger.warning("native encoder returned no render plan; punting to default")
                    return None
                if (
                    native.render_plan_overflow is not None
                    and int(native.render_plan_overflow[0]) != 0
                ):
                    logger.warning("native render plan overflowed; punting to default")
                    return None
                length = int(native.render_plan_lengths[0])
                plan = native.render_plan[0, :length].clone()
            except Exception as exc:
                logger.warning("native render-plan encode failed: %s; punting to default", exc)
                return None
        else:
            try:
                plan = emit_render_plan(
                    snapshot,
                    list(legal_options),
                    card_row_lookup=self._card_row_lookup,
                    tokenize=self._tokenize_fn,
                    oracle=self.oracle,
                    dedup_card_bodies=self.dedup_card_bodies,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("emit_render_plan failed: %s; punting to default", exc)
                return None

        try:
            batch = assemble_batch(
                [plan],
                self.cache,
                self.tokenizer,
                max_tokens=self.max_tokens,
                on_overflow="truncate",
            )
        except Exception as exc:
            logger.warning("assemble_batch failed: %s; punting to default", exc)
            return None

        # Move tensors to device.
        moved = type(batch)(
            token_ids=batch.token_ids.to(self.device),
            attention_mask=batch.attention_mask.to(self.device),
            card_ref_positions=batch.card_ref_positions.to(self.device),
            option_positions=batch.option_positions.to(self.device),
            option_mask=batch.option_mask.to(self.device),
            target_positions=batch.target_positions.to(self.device),
            target_mask=batch.target_mask.to(self.device),
            seq_lengths=batch.seq_lengths.to(self.device),
        )

        with torch.no_grad():
            out, (h_out, c_out) = self.policy(moved, h_in=state.h, c_in=state.c)

        # Sample option idx.
        opt_logits = out.policy_logits[0]  # [O]
        opt_mask = out.option_mask[0]  # [O]
        if opt_mask.numel() == 0 or not bool(opt_mask.any()):
            return None
        chosen_opt = _categorical_sample(
            opt_logits, opt_mask, temperature=self.sampling_temperature, generator=self._gen
        )
        if chosen_opt < 0:
            return None

        # Sample target idx among targets of the chosen option, if any.
        chosen_tgt: int | None = None
        if out.target_mask.shape[-1] > 0:
            tgt_mask = out.target_mask[0, chosen_opt]
            if bool(tgt_mask.any()):
                tgt_logits = out.target_logits[0, chosen_opt]
                tidx = _categorical_sample(
                    tgt_logits,
                    tgt_mask,
                    temperature=self.sampling_temperature,
                    generator=self._gen,
                )
                chosen_tgt = tidx if tidx >= 0 else None

        return (
            chosen_opt,
            chosen_tgt,
            _PlayerLSTM(h=h_out.detach().clone(), c=c_out.detach().clone()),
        )

    # --- public API -------------------------------------------------------

    def play_episode(
        self,
        game_cfg: dict[str, Any],
        max_turns: int = 100,
    ) -> TextRolloutEpisode:
        """Play one game to completion (or ``max_turns``) and return the trajectory."""

        # Accept either ``{deck_a, deck_b, ...}`` or ``{player_a, player_b, ...}``.
        deck_a = game_cfg.get("deck_a") or game_cfg.get("player_a")
        deck_b = game_cfg.get("deck_b") or game_cfg.get("player_b")
        if deck_a is None or deck_b is None:
            raise ValueError("game_cfg must contain deck_a/deck_b (or player_a/player_b)")
        kwargs = {
            k: v
            for k, v in game_cfg.items()
            if k not in ("deck_a", "deck_b", "player_a", "player_b") and v is not None
        }
        game = mage.new_game(deck_a, deck_b, **kwargs)

        episode = TextRolloutEpisode()
        states = self._init_states()
        last_turn = 0

        try:
            step_budget = 0
            # Hard cap on raw engine steps so a stuck game can't loop forever.
            # ``max_turns`` is a *game-turn* cap; allow a generous amount of
            # priority passes per turn before giving up.
            hard_step_cap = int(max_turns) * 200 + 200

            while step_budget < hard_step_cap:
                step_budget += 1
                game.refresh_state()
                if game.is_over:
                    break
                snapshot = cast(GameStateSnapshot, game.state)
                turn = int(snapshot.get("turn", 0) or 0)
                last_turn = turn
                if turn > max_turns:
                    break

                pending = cast(PendingState | None, game.pending or game.legal())
                if pending is None:
                    # No pending request: try a no-op pass to advance.
                    try:
                        game.step({"kind": "pass"})
                    except Exception as exc:
                        logger.warning("game.step(pass) failed with no pending: %s", exc)
                        break
                    continue

                options = list(pending.get("options", []) or [])
                player_idx = int(pending.get("player_idx", 0) or 0)
                if player_idx not in (0, 1):
                    player_idx = 0

                if not options:
                    # Some pending kinds carry no options (e.g. forced choice).
                    try:
                        game.step(dict(_default_action_for(pending)))
                    except Exception as exc:
                        logger.warning("game.step(default) failed (no options): %s", exc)
                        break
                    continue

                pending_kind = pending.get("kind", "") or ""
                # We only run the policy on priority decisions; other kinds use
                # a deterministic default. Keeps the smoke harness simple and
                # avoids feeding the renderer non-priority pendings it does
                # not have grammar for.
                if pending_kind != "priority":
                    action = _default_action_for(pending)
                    try:
                        game.step(dict(action))
                    except Exception as exc:
                        logger.warning(
                            "game.step(default %s) failed: %s; aborting episode",
                            pending_kind,
                            exc,
                        )
                        break
                    continue

                state = states[player_idx]
                scored = self._score_step(game, snapshot, options, state, player_idx)
                if scored is None:
                    action = _default_action_for(pending)
                    chosen_opt, chosen_tgt = -1, None
                else:
                    chosen_opt, chosen_tgt, new_state = scored
                    states[player_idx] = new_state
                    action = _translate_action(pending, chosen_opt, chosen_tgt)

                # Sanity: action's option must round-trip through the
                # candidate list (priority only). We already mapped via
                # build_priority_candidates so this is essentially asserting
                # 0 <= chosen_opt < len(options).
                if chosen_opt >= 0:
                    if not (0 <= chosen_opt < len(options)):
                        logger.warning(
                            "chosen option idx %d out of range (n=%d); using default",
                            chosen_opt,
                            len(options),
                        )
                        action = _default_action_for(pending)

                episode.steps.append(
                    TextRolloutStep(
                        snapshot=snapshot,
                        legal_options=options,
                        chosen_option_idx=int(chosen_opt) if chosen_opt >= 0 else 0,
                        chosen_target_idx=chosen_tgt,
                        reward=0.0,
                        perspective_player_idx=player_idx,
                    )
                )

                try:
                    game.step(dict(action))
                except Exception as exc:
                    logger.warning(
                        "game.step failed (kind=%s opt=%d tgt=%s): %s; aborting",
                        pending_kind,
                        chosen_opt,
                        chosen_tgt,
                        exc,
                    )
                    break

            # Determine winner.
            game.refresh_state()
            episode.turns = last_turn
            if game.is_over:
                winner_id = game.winner or ""
                players = (game.state or {}).get("players", []) or []
                winner_idx: int | None = None
                for idx, p in enumerate(players):
                    if p.get("ID", "") == winner_id or p.get("Name", "") == winner_id:
                        winner_idx = idx
                        break
                episode.winner_player_idx = winner_idx
                # Assign terminal rewards backward.
                if winner_idx is not None:
                    for step in episode.steps:
                        if step.perspective_player_idx == winner_idx:
                            step.reward = 1.0
                        else:
                            step.reward = -1.0
                # else: draw -> all rewards stay 0
            else:
                episode.winner_player_idx = None
        finally:
            try:
                game.close()
            except Exception:
                pass

        return episode

    def play_episodes(
        self,
        game_cfg: dict[str, Any],
        n: int,
    ) -> list[TextRolloutEpisode]:
        return [self.play_episode(game_cfg) for _ in range(int(n))]


__all__ = [
    "TextRolloutEpisode",
    "TextRolloutStep",
    "TextRolloutWorker",
]


# Numpy used only inside emit_render_plan; ensure it's imported (silences
# "unused import" if a static checker doesn't see the transitive use).
_ = np
