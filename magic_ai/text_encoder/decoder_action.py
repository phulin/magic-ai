"""Translate a decoded grammar token sequence back into an engine ``ActionRequest``."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from magic_ai.actions import (
    ActionRequest,
    BlockerAssignState,
    action_from_attackers,
    action_from_choice_accepted,
    action_from_choice_index,
    action_from_priority_candidate,
    build_priority_candidates,
)
from magic_ai.game_state import PendingState
from magic_ai.text_encoder.decision_spec import DecisionType
from magic_ai.text_encoder.decoder_batch import DecoderDecisionLayout
from magic_ai.text_encoder.grammar import DIGIT_0_ID, DIGIT_9_ID, GrammarVocab


def _digit_seq_to_int(token_ids: Sequence[int]) -> int:
    value = 0
    saw_digit = False
    for tok in token_ids:
        t = int(tok)
        if not (DIGIT_0_ID <= t <= DIGIT_9_ID):
            continue
        value = value * 10 + (t - DIGIT_0_ID)
        saw_digit = True
    return value if saw_digit else 0


def decode_decoder_action(
    pending: PendingState,
    layout: DecoderDecisionLayout,
) -> ActionRequest:
    """Translate a decoded token sequence into an engine ``ActionRequest``.

    The :class:`DecoderDecisionLayout` carries pointer-anchor handles for
    each pointer step; the handle is the engine option index (PRIORITY) or
    object id (combat / targets / mode). Decoding routes per
    :class:`DecisionType` and falls back to a safe default if the sequence
    is malformed.
    """

    dt_val = int(layout.decision_type)
    if dt_val < 0:
        # No pending decision spec was attached to this row; fall back to a
        # pass / no-op action that the engine will accept for the current pending.
        return action_from_choice_index(0)
    dt = DecisionType(dt_val)
    tokens = layout.output_token_ids.detach().to("cpu", dtype=torch.long).tolist()
    is_ptr = layout.output_is_pointer.detach().to("cpu").tolist()
    pad = layout.output_pad_mask.detach().to("cpu").tolist()
    handles = layout.pointer_anchor_handles.detach().to("cpu", dtype=torch.long).tolist()

    # Walk pointer steps in order; the i-th valid pointer step maps to
    # handles[i] (anchor list ordering). This requires the caller to pass
    # the anchor handles in subject_index order — which is how the renderer
    # builds them.
    ptr_step_indices: list[int] = []
    for i in range(len(tokens)):
        if not bool(pad[i]):
            break
        if bool(is_ptr[i]):
            ptr_step_indices.append(i)

    # Look up handles by anchor subject_index. ``pointer_pos`` here holds
    # the encoder absolute position chosen by the pointer head; we need the
    # subject index to index into ``handles``. Without the spec we don't
    # have the position→subject map at decode time, but the layout above
    # was built with handles in subject_index order and ``pointer_pos[i]``
    # points at one specific anchor's encoder position. We assume the caller
    # tracks the subject ordering (renderer guarantees it) and resolves
    # handles by the anchor at the chosen position. Fallback below uses
    # ptr_step ordinal as the pointer-step index into handles (matches the
    # PRIORITY case where there is exactly one pointer step).
    def _handle_for_pointer(step_idx: int, ordinal: int) -> int:
        del step_idx
        if ordinal < len(handles):
            return int(handles[ordinal])
        return 0

    if dt == DecisionType.PRIORITY:
        # Single pointer step → engine option index.
        if not ptr_step_indices:
            candidates = build_priority_candidates(pending)
            return action_from_priority_candidate(candidates[0]) if candidates else {"kind": "pass"}
        opt_idx = _handle_for_pointer(ptr_step_indices[0], 0)
        candidates = build_priority_candidates(pending)
        for cand in candidates:
            if cand.option_index == opt_idx:
                return action_from_priority_candidate(cand)
        if candidates:
            return action_from_priority_candidate(candidates[0])
        return {"kind": "pass"}

    if dt == DecisionType.DECLARE_ATTACKERS:
        # Pairs of (LEGAL_ATTACKER ptr, DEFENDER ptr). Build a binary attacker
        # selection by attacker handle.
        options = pending.get("options", []) or []
        n = len(options)
        selected = [False] * n
        for ord_i in range(0, len(ptr_step_indices), 2):
            attacker_handle = _handle_for_pointer(ptr_step_indices[ord_i], ord_i)
            if 0 <= attacker_handle < n:
                selected[attacker_handle] = True
        return action_from_attackers(pending, selected)

    if dt == DecisionType.DECLARE_BLOCKERS:
        # Pairs of (LEGAL_BLOCKER ptr, ATTACKER ptr). Engine wants
        # ``{blockers: [{blocker, attacker}, ...]}`` keyed by id.
        options = pending.get("options", []) or []
        assignments: list[BlockerAssignState] = []
        for ord_i in range(0, len(ptr_step_indices) - 1, 2):
            blk_h = _handle_for_pointer(ptr_step_indices[ord_i], ord_i)
            atk_h = _handle_for_pointer(ptr_step_indices[ord_i + 1], ord_i + 1)
            if not (0 <= blk_h < len(options)):
                continue
            opt = options[blk_h]
            blocker_id = str(opt.get("permanent_id", "") or "")
            targets = opt.get("valid_targets", []) or []
            if not (0 <= atk_h < len(targets)):
                continue
            attacker_id = str(targets[atk_h].get("id", "") or "")
            if blocker_id and attacker_id:
                assignments.append(BlockerAssignState(blocker=blocker_id, attacker=attacker_id))
        return ActionRequest(blockers=assignments)

    if dt == DecisionType.CHOOSE_TARGETS:
        if not ptr_step_indices:
            return action_from_choice_index(0)
        h = _handle_for_pointer(ptr_step_indices[0], 0)
        return action_from_choice_index(int(h))

    if dt == DecisionType.MAY:
        # Look for YES / NO grammar tokens.
        for i, t in enumerate(tokens):
            if not bool(pad[i]):
                break
            if int(t) == int(GrammarVocab.YES):
                return action_from_choice_accepted(True)
            if int(t) == int(GrammarVocab.NO):
                return action_from_choice_accepted(False)
        return action_from_choice_accepted(False)

    if dt in (DecisionType.CHOOSE_MODE, DecisionType.CHOOSE_X):
        digits = [tokens[i] for i in range(len(tokens)) if bool(pad[i])]
        return action_from_choice_index(_digit_seq_to_int(digits))

    return action_from_choice_index(0)


__all__ = ["decode_decoder_action"]
