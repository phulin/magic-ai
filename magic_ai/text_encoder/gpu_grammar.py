"""On-device grammar mask state machine for the autoregressive decoder.

Mirrors the per-row finite-state machines in :mod:`grammar` but holds all
state as device tensors and computes ``(vocab_mask, pointer_pos_mask)``
each step without any host sync. Used by
:func:`decoder_inference.decoder_sample` and the replay-mask builder so
the inner sample loop runs entirely on GPU.

The CPU implementation in :mod:`grammar` remains the source of truth and
is still used by tests and by the inline-blank pretraining loop; this
module is the vectorized equivalent.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from magic_ai.text_encoder.decision_spec import AnchorKind, DecisionType
from magic_ai.text_encoder.grammar import (
    DIGIT_0_ID,
    DIGIT_9_ID,
    GRAMMAR_VOCAB_SIZE,
    GrammarVocab,
)

N_ANCHOR_KINDS = 5  # AnchorKind enum width


def _grammar_update_body(
    sampled_vocab: Tensor,  # [B] long
    sampled_pointer_idx: Tensor,  # [B] long
    is_pointer_step: Tensor,  # [B] bool
    *,
    # static lookup tables (treated as immutable across the decode loop)
    anchor_subjects: Tensor,  # [B, N_anchor] long
    anchor_kinds: Tensor,  # [B, N_anchor] long
    b_arange: Tensor,  # [B] long
    # mutable state
    chosen_per_kind: Tensor,  # [B, N_ANCHOR_KINDS, MAX_K] bool
    last_chosen_blk_subj: Tensor,  # [B] long
    current_int: Tensor,  # [B] long
    n_digits: Tensor,  # [B] long
    ended: Tensor,  # [B] bool
    step: Tensor,  # [B] long
    max_k: int,
    n_anchor: int,
    atk_kind: int,
    blk_kind: int,
    end_id: int,
    digit_0_id: int,
    digit_9_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pure-functional body of :meth:`GrammarMaskState.update`.

    Returns ``(chosen_per_kind, last_chosen_blk_subj, current_int, n_digits,
    ended, step)`` so the wrapper can be ``torch.compile``-d as a single
    fused graph instead of paying ~20 op dispatches per decoder step.
    """
    active = ~ended
    safe_idx = sampled_pointer_idx.clamp(min=0, max=max(n_anchor - 1, 0))
    safe_idx_2d = safe_idx.unsqueeze(-1)
    subj_at_chosen = anchor_subjects.gather(1, safe_idx_2d).squeeze(-1)
    kind_at_chosen = anchor_kinds.gather(1, safe_idx_2d).squeeze(-1)
    valid_choice = is_pointer_step & active & (subj_at_chosen >= 0) & (kind_at_chosen >= 0)
    safe_subj = subj_at_chosen.clamp(min=0, max=max(max_k - 1, 0))

    is_atk = valid_choice & (kind_at_chosen == atk_kind)
    is_blk = valid_choice & (kind_at_chosen == blk_kind)
    is_atk_or_blk = is_atk | is_blk

    # Single fused write for both ATK and BLK: pick the kind row from
    # ``kind_at_chosen`` (already validated for atk/blk above; fall back to
    # 0 for inactive rows where ``is_atk_or_blk`` is False — the OR with
    # False is a no-op for that cell).
    upd_kind = torch.where(is_atk_or_blk, kind_at_chosen, kind_at_chosen.new_zeros(()))
    cur = chosen_per_kind[b_arange, upd_kind, safe_subj]
    chosen_per_kind = chosen_per_kind.clone()
    chosen_per_kind[b_arange, upd_kind, safe_subj] = cur | is_atk_or_blk

    last_chosen_blk_subj = torch.where(is_blk, safe_subj, last_chosen_blk_subj)

    is_vocab_step = ~is_pointer_step
    is_digit = (
        is_vocab_step & active & (sampled_vocab >= digit_0_id) & (sampled_vocab <= digit_9_id)
    )
    digit_val = (sampled_vocab - digit_0_id).clamp(min=0, max=9)
    current_int = torch.where(is_digit, current_int * 10 + digit_val, current_int)
    n_digits = torch.where(is_digit, n_digits + 1, n_digits)

    is_end = is_vocab_step & active & (sampled_vocab == end_id)
    ended = ended | is_end
    step = step + 1
    return chosen_per_kind, last_chosen_blk_subj, current_int, n_digits, ended, step


# Lazily compiled (per the project's pattern: ``dynamic=True`` so a single
# trace serves all batch sizes / MAX_K values).
_compiled_grammar_update_body: object | None = None


def _get_grammar_update_fn() -> object:
    global _compiled_grammar_update_body
    if _compiled_grammar_update_body is None:
        _compiled_grammar_update_body = torch.compile(_grammar_update_body, dynamic=True)
    return _compiled_grammar_update_body


class GrammarMaskState:
    """Per-row grammar state, held as device tensors.

    Inputs match the encoder-batch fields (``decision_type``, the four
    pointer-anchor arrays, optional ``legal_edge_bitmap`` and per-row
    ``max_value``). After construction, call :meth:`next_mask` to get the
    legal mask for the current step, then :meth:`update` with the chosen
    action to advance.
    """

    def __init__(
        self,
        decision_type: Tensor,  # [B] long
        pointer_anchor_kinds: Tensor,  # [B, N] long, -1 pad
        pointer_anchor_subjects: Tensor,  # [B, N] long, -1 pad
        pointer_anchor_positions: Tensor,  # [B, N] long, -1 pad
        encoded_seq_len: int,
        legal_edge_bitmap: Tensor | None = None,  # [B, N_blk, N_atk] bool
        max_value: Tensor | None = None,  # [B] long, -1 = N/A
        compile_update: bool = False,
    ) -> None:
        device = decision_type.device
        b, n = pointer_anchor_kinds.shape
        t_enc = int(encoded_seq_len)
        self.B = int(b)
        self.T_enc = t_enc
        self.device = device
        self.decision_type = decision_type
        self.pointer_anchor_positions = pointer_anchor_positions
        self.pointer_anchor_kinds = pointer_anchor_kinds
        self.pointer_anchor_subjects = pointer_anchor_subjects
        self.legal_edge_bitmap = (
            legal_edge_bitmap.to(device=device, dtype=torch.bool)
            if legal_edge_bitmap is not None
            else None
        )
        self.max_value = (
            max_value.to(device=device, dtype=torch.long)
            if max_value is not None
            else torch.full((self.B,), -1, dtype=torch.long, device=device)
        )
        self.compile_update = bool(compile_update)

        # MAX_K bounds the per-kind subject_index. The renderer assigns
        # subject_index < (anchor count of that kind) ≤ N, so N is a safe
        # upper bound for the per-kind chosen-bitmap width.
        max_k = max(int(n), 1)
        self.MAX_K = max_k

        # n_per_kind[b, k]: count of anchors of kind k for row b.
        n_per_kind = torch.zeros((self.B, N_ANCHOR_KINDS), dtype=torch.long, device=device)
        if n > 0:
            valid_anchor = pointer_anchor_subjects >= 0
            # one-hot over kind dim, then sum along anchor dim.
            kinds_clamped = pointer_anchor_kinds.clamp(min=0, max=N_ANCHOR_KINDS - 1)
            one_hot = torch.zeros((self.B, n, N_ANCHOR_KINDS), dtype=torch.long, device=device)
            one_hot.scatter_(2, kinds_clamped.unsqueeze(-1), valid_anchor.long().unsqueeze(-1))
            # zero out invalid anchor rows (kinds<0 means no real anchor).
            valid_kind = pointer_anchor_kinds >= 0
            one_hot = one_hot * (valid_kind & valid_anchor).long().unsqueeze(-1)
            n_per_kind = one_hot.sum(dim=1)
        self.n_per_kind = n_per_kind

        # Mutable state.
        self.step = torch.zeros(self.B, dtype=torch.long, device=device)
        self.ended = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.chosen_per_kind = torch.zeros(
            (self.B, N_ANCHOR_KINDS, max_k), dtype=torch.bool, device=device
        )
        self.last_chosen_blk_subj = torch.full((self.B,), -1, dtype=torch.long, device=device)
        self.current_int = torch.zeros(self.B, dtype=torch.long, device=device)
        self.n_digits = torch.zeros(self.B, dtype=torch.long, device=device)

        # Cached ``arange(B)`` for advanced indexing in update().
        self._b_arange = torch.arange(self.B, device=device)
        # Cached digit-id arange for the integer mask.
        self._digits_arange = torch.arange(10, device=device, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Mask construction                                                  #
    # ------------------------------------------------------------------ #

    def next_mask(self) -> tuple[Tensor, Tensor]:
        """Return ``(vocab_mask [B, V], pointer_anchor_mask [B, N_anchor])`` for the
        current step, fully on-device."""

        device = self.device
        b, v, n = self.step.shape[0], GRAMMAR_VOCAB_SIZE, self.pointer_anchor_kinds.shape[1]
        dt = self.decision_type
        s = self.step

        vocab = torch.zeros((b, v), dtype=torch.bool, device=device)
        ptr = torch.zeros((b, n), dtype=torch.bool, device=device)

        # PRIORITY: OPEN, ptr LEGAL_ACTION, END.
        is_pri = dt == int(DecisionType.PRIORITY)
        vocab[:, int(GrammarVocab.PRIORITY_OPEN)] |= is_pri & (s == 0)
        self._or_pointer(ptr, is_pri & (s == 1), int(AnchorKind.LEGAL_ACTION), no_repeat=False)
        vocab[:, int(GrammarVocab.END)] |= is_pri & (s == 2)

        # CHOOSE_TARGETS: OPEN, ptr LEGAL_TARGET, END.
        is_ct = dt == int(DecisionType.CHOOSE_TARGETS)
        vocab[:, int(GrammarVocab.CHOOSE_TARGETS_OPEN)] |= is_ct & (s == 0)
        self._or_pointer(ptr, is_ct & (s == 1), int(AnchorKind.LEGAL_TARGET), no_repeat=False)
        vocab[:, int(GrammarVocab.END)] |= is_ct & (s == 2)

        # MAY: OPEN, {YES, NO}, END.
        is_may = dt == int(DecisionType.MAY)
        vocab[:, int(GrammarVocab.MAY_OPEN)] |= is_may & (s == 0)
        m1 = is_may & (s == 1)
        vocab[:, int(GrammarVocab.YES)] |= m1
        vocab[:, int(GrammarVocab.NO)] |= m1
        vocab[:, int(GrammarVocab.END)] |= is_may & (s == 2)

        # DECLARE_ATTACKERS: step 0 OPEN; body cycles
        # phase 0: ATTACK or END (END always; ATTACK only if not all chosen)
        # phase 1: ptr LEGAL_ATTACKER (no-repeat)
        # phase 2: DEFENDER
        # phase 3: ptr DEFENDER
        is_da = dt == int(DecisionType.DECLARE_ATTACKERS)
        vocab[:, int(GrammarVocab.DECLARE_ATTACKERS_OPEN)] |= is_da & (s == 0)
        body_step = (s - 1).clamp(min=0)
        phase = body_step % 4
        in_da_body = is_da & (s >= 1)
        n_atk = self.n_per_kind[:, int(AnchorKind.LEGAL_ATTACKER)]
        n_chosen_atk = self.chosen_per_kind[:, int(AnchorKind.LEGAL_ATTACKER), :].sum(dim=-1)
        all_atk_chosen = n_chosen_atk >= n_atk
        da_p0 = in_da_body & (phase == 0)
        vocab[:, int(GrammarVocab.ATTACK)] |= da_p0 & ~all_atk_chosen
        vocab[:, int(GrammarVocab.END)] |= da_p0
        self._or_pointer(
            ptr, in_da_body & (phase == 1), int(AnchorKind.LEGAL_ATTACKER), no_repeat=True
        )
        vocab[:, int(GrammarVocab.DEFENDER)] |= in_da_body & (phase == 2)
        self._or_pointer(ptr, in_da_body & (phase == 3), int(AnchorKind.DEFENDER), no_repeat=False)

        # DECLARE_BLOCKERS: same shape but BLOCK / ATTACKER, and the
        # phase-3 attacker pointer is restricted by legal_edge_bitmap.
        is_db = dt == int(DecisionType.DECLARE_BLOCKERS)
        vocab[:, int(GrammarVocab.DECLARE_BLOCKERS_OPEN)] |= is_db & (s == 0)
        in_db_body = is_db & (s >= 1)
        n_blk = self.n_per_kind[:, int(AnchorKind.LEGAL_BLOCKER)]
        n_chosen_blk = self.chosen_per_kind[:, int(AnchorKind.LEGAL_BLOCKER), :].sum(dim=-1)
        all_blk_chosen = n_chosen_blk >= n_blk
        db_p0 = in_db_body & (phase == 0)
        vocab[:, int(GrammarVocab.BLOCK)] |= db_p0 & ~all_blk_chosen
        vocab[:, int(GrammarVocab.END)] |= db_p0
        self._or_pointer(
            ptr, in_db_body & (phase == 1), int(AnchorKind.LEGAL_BLOCKER), no_repeat=True
        )
        vocab[:, int(GrammarVocab.ATTACKER)] |= in_db_body & (phase == 2)
        self._or_pointer_edge(ptr, in_db_body & (phase == 3))

        # CHOOSE_MODE / CHOOSE_X: OPEN then digits (0..9 bounded by
        # ``current * 10 + d <= max_value``); END allowed once at least one
        # digit is emitted.
        for cm_dt, cm_open in (
            (int(DecisionType.CHOOSE_MODE), int(GrammarVocab.CHOOSE_MODE_OPEN)),
            (int(DecisionType.CHOOSE_X), int(GrammarVocab.CHOOSE_X_OPEN)),
        ):
            is_cm = dt == cm_dt
            vocab[:, cm_open] |= is_cm & (s == 0)
            in_cm_body = is_cm & (s >= 1)
            base = self.current_int * 10
            base_ok = base <= self.max_value
            upper = (self.max_value - base).clamp(min=0)
            digit_allowed = (
                in_cm_body.unsqueeze(-1)
                & base_ok.unsqueeze(-1)
                & (self._digits_arange.unsqueeze(0) <= upper.unsqueeze(-1))
            )
            vocab[:, DIGIT_0_ID : DIGIT_9_ID + 1] |= digit_allowed
            vocab[:, int(GrammarVocab.END)] |= in_cm_body & (self.n_digits >= 1)

        # Ended rows: clear their masks. ``combined_sample`` falls back to
        # a uniform dummy when both heads are empty, so log_softmax stays
        # well-defined; the caller masks the resulting log-probs to zero
        # via ``output_pad_mask``.
        not_ended = ~self.ended
        vocab &= not_ended.unsqueeze(-1)
        ptr &= not_ended.unsqueeze(-1)
        return vocab, ptr

    def _or_pointer(
        self, ptr_mask: Tensor, row_active: Tensor, kind: int, *, no_repeat: bool
    ) -> None:
        """OR into ``ptr_mask`` the legal pointer positions for ``kind`` on the
        active rows. ``no_repeat`` excludes already-chosen subjects."""
        kind_match = self.pointer_anchor_kinds == kind  # [B, N_anchor]
        valid_anchor = (
            kind_match
            & (self.pointer_anchor_positions >= 0)
            & (self.pointer_anchor_positions < self.T_enc)
            & (self.pointer_anchor_subjects >= 0)
        )
        if no_repeat:
            safe_subj = self.pointer_anchor_subjects.clamp(min=0, max=max(self.MAX_K - 1, 0))
            chosen_at_pos = self.chosen_per_kind[:, kind, :].gather(1, safe_subj)
            chosen_at_pos = chosen_at_pos & (self.pointer_anchor_subjects >= 0)
            avail = valid_anchor & ~chosen_at_pos
        else:
            avail = valid_anchor
        ptr_mask |= avail & row_active.unsqueeze(-1)

    def _or_pointer_edge(self, ptr_mask: Tensor, row_active: Tensor) -> None:
        """DECLARE_BLOCKERS phase 3: pointer LEGAL_ATTACKER restricted by the
        per-blocker legal-edge bitmap."""
        kind = int(AnchorKind.LEGAL_ATTACKER)
        kind_match = self.pointer_anchor_kinds == kind
        valid_anchor = (
            kind_match
            & (self.pointer_anchor_positions >= 0)
            & (self.pointer_anchor_positions < self.T_enc)
            & (self.pointer_anchor_subjects >= 0)
        )
        # Treat an empty-shape bitmap (``[B, 0, 0]``) like ``None``: the
        # inference-server batch merger pads non-DECLARE_BLOCKERS batches
        # with an empty bitmap when *any* batch has one, so a stray empty
        # tensor would otherwise reach the gather and crash on its
        # zero-sized dim.
        edge = self.legal_edge_bitmap
        if edge is None or edge.shape[1] == 0 or edge.shape[2] == 0:
            avail = valid_anchor
        else:
            n_blk, n_atk = int(edge.shape[1]), int(edge.shape[2])
            blk_idx = self.last_chosen_blk_subj.clamp(min=0, max=max(n_blk - 1, 0))
            b = row_active.shape[0]
            edge_per_atk = edge.gather(1, blk_idx.view(b, 1, 1).expand(b, 1, n_atk)).squeeze(
                1
            )  # [B, n_atk]
            safe_subj = self.pointer_anchor_subjects.clamp(min=0, max=max(n_atk - 1, 0))
            edge_at_pos = edge_per_atk.gather(1, safe_subj)
            avail = valid_anchor & edge_at_pos
        ptr_mask |= avail & row_active.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    # State advance                                                      #
    # ------------------------------------------------------------------ #

    def update(
        self,
        sampled_vocab: Tensor,  # [B] long
        sampled_pointer_pos: Tensor,  # [B] long anchor index
        is_pointer_step: Tensor,  # [B] bool
    ) -> None:
        """Advance state given the sampled action.

        Optionally dispatches to ``_grammar_update_body`` wrapped in
        ``torch.compile`` so the ~20 per-step tensor ops fuse into a
        single graph launch. The eager default avoids a multi-minute cold
        compile on first decoder use.
        """
        fn = (
            cast(Any, _get_grammar_update_fn())
            if self.compile_update
            else cast(Any, _grammar_update_body)
        )
        (
            self.chosen_per_kind,
            self.last_chosen_blk_subj,
            self.current_int,
            self.n_digits,
            self.ended,
            self.step,
        ) = fn(
            sampled_vocab,
            sampled_pointer_pos,
            is_pointer_step,
            anchor_subjects=self.pointer_anchor_subjects,
            anchor_kinds=self.pointer_anchor_kinds,
            b_arange=self._b_arange,
            chosen_per_kind=self.chosen_per_kind,
            last_chosen_blk_subj=self.last_chosen_blk_subj,
            current_int=self.current_int,
            n_digits=self.n_digits,
            ended=self.ended,
            step=self.step,
            max_k=self.MAX_K,
            n_anchor=int(self.pointer_anchor_kinds.shape[1]),
            atk_kind=int(AnchorKind.LEGAL_ATTACKER),
            blk_kind=int(AnchorKind.LEGAL_BLOCKER),
            end_id=int(GrammarVocab.END),
            digit_0_id=DIGIT_0_ID,
            digit_9_id=DIGIT_9_ID,
        )


__all__ = ["GrammarMaskState", "N_ANCHOR_KINDS"]
