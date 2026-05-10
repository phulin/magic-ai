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
    ) -> None:
        device = decision_type.device
        b, n = pointer_anchor_kinds.shape
        t_enc = int(encoded_seq_len)
        self.B = int(b)
        self.T_enc = t_enc
        self.device = device
        self.decision_type = decision_type
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

        # MAX_K bounds the per-kind subject_index. The renderer assigns
        # subject_index < (anchor count of that kind) ≤ N, so N is a safe
        # upper bound for the per-kind chosen-bitmap width.
        max_k = max(int(n), 1)
        self.MAX_K = max_k

        # pos_to_kind / pos_to_subj: dense [B, T_enc] lookup tables built
        # once from the anchor lists. Values at non-anchor positions are -1.
        # Built via a sync-free scatter: invalid (b, j) anchor entries scatter
        # into a trailing trash column that gets sliced off at the end.
        valid = (
            (pointer_anchor_positions >= 0)
            & (pointer_anchor_positions < t_enc)
            & (pointer_anchor_kinds >= 0)
            & (pointer_anchor_subjects >= 0)
        )
        if n > 0:
            pos_to_kind_full = torch.full((self.B, t_enc + 1), -1, dtype=torch.long, device=device)
            pos_to_subj_full = torch.full((self.B, t_enc + 1), -1, dtype=torch.long, device=device)
            trash = torch.full_like(pointer_anchor_positions, t_enc)
            safe_pos = torch.where(valid, pointer_anchor_positions, trash)
            pos_to_kind_full.scatter_(1, safe_pos, pointer_anchor_kinds)
            pos_to_subj_full.scatter_(1, safe_pos, pointer_anchor_subjects)
            self.pos_to_kind = pos_to_kind_full[:, :t_enc].contiguous()
            self.pos_to_subj = pos_to_subj_full[:, :t_enc].contiguous()
        else:
            self.pos_to_kind = torch.full((self.B, t_enc), -1, dtype=torch.long, device=device)
            self.pos_to_subj = torch.full((self.B, t_enc), -1, dtype=torch.long, device=device)

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
        """Return ``(vocab_mask [B, V], pointer_pos_mask [B, T_enc])`` for the
        current step, fully on-device."""

        device = self.device
        b, v, t = self.B, GRAMMAR_VOCAB_SIZE, self.T_enc
        dt = self.decision_type
        s = self.step

        vocab = torch.zeros((b, v), dtype=torch.bool, device=device)
        ptr = torch.zeros((b, t), dtype=torch.bool, device=device)

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
        kind_match = self.pos_to_kind == kind  # [B, T_enc]
        if no_repeat:
            safe_subj = self.pos_to_subj.clamp(min=0)
            chosen_at_pos = self.chosen_per_kind[:, kind, :].gather(1, safe_subj)
            chosen_at_pos = chosen_at_pos & (self.pos_to_subj >= 0)
            avail = kind_match & ~chosen_at_pos
        else:
            avail = kind_match
        ptr_mask |= avail & row_active.unsqueeze(-1)

    def _or_pointer_edge(self, ptr_mask: Tensor, row_active: Tensor) -> None:
        """DECLARE_BLOCKERS phase 3: pointer LEGAL_ATTACKER restricted by the
        per-blocker legal-edge bitmap."""
        kind = int(AnchorKind.LEGAL_ATTACKER)
        kind_match = self.pos_to_kind == kind
        if self.legal_edge_bitmap is None:
            avail = kind_match
        else:
            edge = self.legal_edge_bitmap
            n_blk, n_atk = int(edge.shape[1]), int(edge.shape[2])
            blk_idx = self.last_chosen_blk_subj.clamp(min=0, max=max(n_blk - 1, 0))
            edge_per_atk = edge.gather(
                1, blk_idx.view(self.B, 1, 1).expand(self.B, 1, n_atk)
            ).squeeze(1)  # [B, n_atk]
            safe_subj = self.pos_to_subj.clamp(min=0, max=max(n_atk - 1, 0))
            edge_at_pos = edge_per_atk.gather(1, safe_subj)
            avail = kind_match & edge_at_pos
        ptr_mask |= avail & row_active.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    # State advance                                                      #
    # ------------------------------------------------------------------ #

    def update(
        self,
        sampled_vocab: Tensor,  # [B] long
        sampled_pointer_pos: Tensor,  # [B] long
        is_pointer_step: Tensor,  # [B] bool
    ) -> None:
        """Advance state given the sampled action."""
        active = ~self.ended

        # Look up subject_index and kind of the chosen pointer position.
        safe_pos = sampled_pointer_pos.clamp(min=0)
        subj_at_chosen = self.pos_to_subj.gather(1, safe_pos.unsqueeze(-1)).squeeze(-1)
        kind_at_chosen = self.pos_to_kind.gather(1, safe_pos.unsqueeze(-1)).squeeze(-1)
        valid_choice = is_pointer_step & active & (subj_at_chosen >= 0)
        safe_subj = subj_at_chosen.clamp(min=0, max=max(self.MAX_K - 1, 0))

        atk_kind = int(AnchorKind.LEGAL_ATTACKER)
        blk_kind = int(AnchorKind.LEGAL_BLOCKER)
        is_atk = valid_choice & (kind_at_chosen == atk_kind)
        is_blk = valid_choice & (kind_at_chosen == blk_kind)

        # Set ``chosen_per_kind[b, kind, safe_subj[b]] |= is_atk[b]`` (and
        # the same for blockers). Single advanced-indexing update.
        b_idx = self._b_arange
        cur = self.chosen_per_kind[b_idx, atk_kind, safe_subj]
        self.chosen_per_kind[b_idx, atk_kind, safe_subj] = cur | is_atk
        cur = self.chosen_per_kind[b_idx, blk_kind, safe_subj]
        self.chosen_per_kind[b_idx, blk_kind, safe_subj] = cur | is_blk
        self.last_chosen_blk_subj = torch.where(is_blk, safe_subj, self.last_chosen_blk_subj)

        # Digit emission for CHOOSE_MODE/X.
        is_vocab_step = ~is_pointer_step
        is_digit = (
            is_vocab_step & active & (sampled_vocab >= DIGIT_0_ID) & (sampled_vocab <= DIGIT_9_ID)
        )
        digit_val = (sampled_vocab - DIGIT_0_ID).clamp(min=0, max=9)
        self.current_int = torch.where(
            is_digit, self.current_int * 10 + digit_val, self.current_int
        )
        self.n_digits = torch.where(is_digit, self.n_digits + 1, self.n_digits)

        # END.
        is_end = is_vocab_step & active & (sampled_vocab == int(GrammarVocab.END))
        self.ended = self.ended | is_end

        # Step always increments. Past-ended rows draw masked-empty masks
        # next call so their tokens are no-ops.
        self.step = self.step + 1


__all__ = ["GrammarMaskState", "N_ANCHOR_KINDS"]
