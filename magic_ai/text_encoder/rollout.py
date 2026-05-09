"""Text-only Python rollout worker (PR 13-D / §10 PR #4).

End-to-end pipeline that plays real Magic games using the text encoder
(renderer + tokenizer + RecurrentTextPolicy) without
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
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
)
from magic_ai.text_encoder.actor_critic import (
    DecoderDecisionLayout,
    decode_decoder_action,
    decoder_sample,
)
from magic_ai.text_encoder.card_cache import CardTokenCache
from magic_ai.text_encoder.policy import TextPolicy
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy
from magic_ai.text_encoder.render import OracleEntry

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
        max_options: int = 64,
        max_targets_per_option: int = 4,
    ) -> None:
        self.policy = policy
        self.cache = cache
        self.tokenizer = tokenizer
        self.max_tokens = int(max_tokens)
        self.device = torch.device(device)
        self.sampling_temperature = float(sampling_temperature)
        self.oracle = oracle
        self.max_options = int(max_options)
        self.max_targets_per_option = int(max_targets_per_option)
        self.policy.to(self.device)
        self.policy.eval()

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
    ) -> tuple[ActionRequest, _PlayerLSTM, DecoderDecisionLayout] | None:
        """Run the encoder + LSTM + grammar decoder for one priority step.

        Returns ``(engine_action, new_state, layout)`` so the caller can both
        send the engine action and record the decoded layout. ``None`` means
        the policy refused to score this snapshot and the caller should fall
        back to a deterministic default.
        """

        del legal_options, perspective_player_idx
        text_policy: TextPolicy = self.policy.text_policy
        if text_policy.grammar_decoder is None:
            logger.warning("text rollout requires text_policy.grammar_decoder")
            return None
        try:
            batch = TextPolicy.encode_snapshots([snapshot], self.oracle, self.tokenizer)
        except Exception as exc:
            logger.warning("encode_snapshots failed: %s", exc)
            return None
        device = self.device
        token_ids = batch.token_ids.to(device=device, dtype=torch.long)
        attn_mask = batch.attention_mask.to(device=device, dtype=torch.long)
        moved = type(batch)(
            token_ids=token_ids,
            attention_mask=attn_mask,
            card_ref_positions=batch.card_ref_positions.to(device=device, dtype=torch.long),
            seq_lengths=batch.seq_lengths.to(device=device, dtype=torch.long),
            spec_tokens=batch.spec_tokens.to(device=device),
            spec_lens=batch.spec_lens.to(device=device),
            decision_type=batch.decision_type.to(device=device, dtype=torch.long),
            pointer_anchor_positions=batch.pointer_anchor_positions.to(device=device),
            pointer_anchor_kinds=batch.pointer_anchor_kinds.to(device=device),
            pointer_anchor_subjects=batch.pointer_anchor_subjects.to(device=device),
            pointer_anchor_handles=batch.pointer_anchor_handles.to(device=device),
            legal_edge_bitmap=(
                batch.legal_edge_bitmap.to(device=device)
                if batch.legal_edge_bitmap is not None
                else None
            ),
        )
        h_in = state.h.to(device=device)
        c_in = state.c.to(device=device)
        with torch.no_grad():
            encoded = text_policy.encoder(moved)
            _, (h_out, c_out) = self.policy(moved, h_in=h_in, c_in=c_in)
            sample = decoder_sample(
                text_policy,
                encoded,
                moved.attention_mask.to(dtype=torch.bool),
                moved.decision_type.to(dtype=torch.long),
                moved.pointer_anchor_positions.to(dtype=torch.long),
                moved.pointer_anchor_kinds.to(dtype=torch.long),
                moved.pointer_anchor_subjects.to(dtype=torch.long),
                moved.pointer_anchor_handles.to(dtype=torch.long),
                legal_edge_bitmap=moved.legal_edge_bitmap,
                greedy=self.sampling_temperature <= 0.0,
                temperature=max(self.sampling_temperature, 1e-6),
            )
        layout = DecoderDecisionLayout(
            output_token_ids=sample.output_token_ids[0],
            output_pointer_pos=sample.output_pointer_pos[0],
            output_pointer_subjects=sample.output_pointer_subjects[0],
            output_is_pointer=sample.output_is_pointer[0],
            output_pad_mask=sample.output_pad_mask[0],
            decision_type=int(sample.decision_type[0].item()),
            pointer_anchor_handles=sample.pointer_anchor_handles[0],
            pointer_anchor_count=int(sample.pointer_anchor_count[0].item()),
        )
        pending = cast(PendingState, game.pending or game.legal() or {})
        action = decode_decoder_action(pending, layout)
        return action, _PlayerLSTM(h=h_out, c=c_out), layout

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
                # The decoder handles every grammar-supported decision type
                # (priority, attackers, blockers, may, choose-targets/mode/X).
                # Anything else falls through to the deterministic default
                # path so we don't feed the renderer pendings it has no
                # grammar for.
                if pending_kind not in (
                    "priority",
                    "attackers",
                    "blockers",
                    "may",
                ):
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
                    chosen_opt: int = 0
                else:
                    action, new_state, _layout = scored
                    states[player_idx] = new_state
                    # We don't currently round-trip the chosen option index
                    # back from the engine action; record 0 as a placeholder
                    # since downstream consumers ignore it for non-priority.
                    chosen_opt = 0

                episode.steps.append(
                    TextRolloutStep(
                        snapshot=snapshot,
                        legal_options=options,
                        chosen_option_idx=int(chosen_opt),
                        chosen_target_idx=None,
                        reward=0.0,
                        perspective_player_idx=player_idx,
                    )
                )

                try:
                    game.step(dict(action))
                except Exception as exc:
                    logger.warning(
                        "game.step failed (kind=%s opt=%d): %s; aborting",
                        pending_kind,
                        chosen_opt,
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
