"""Per-decision-type grammar state machines for the autoregressive decoder.

The decoder owns its own small vocabulary (``GrammarVocab``) — distinct
from the encoder's BPE token table. At every decode step the engine
supplies a mask over (vocab tokens) ∪ (pointer anchors); ``next_mask``
is the Python mirror of that mask used by tests and by the assembler
on the non-native path. The native callback (step 9 of the decoder
migration plan) replaces this on the live path.

Conventions:
- ``prefix`` holds previously-emitted grammar token ids (members of
  ``GrammarVocab`` or one of the digit ids in ``[DIGIT_0_ID,
  DIGIT_9_ID]``).
- ``prefix_pointers`` is a parallel list whose entries hold the
  ``subject_index`` chosen at each pointer step and ``-1`` for
  non-pointer steps. Lengths must match.
- All masks are numpy bool arrays. The pointer mask is ``None`` whenever
  the next step is a vocab-only step.
- Inconsistent prefixes (i.e. prefixes that the previous mask would
  have rejected) raise ``ValueError`` — the decoder is the only caller
  in production and never produces them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from torch import Tensor

from magic_ai.text_encoder.decision_spec import (
    AnchorKind,
    DecisionSpec,
    DecisionType,
)


class GrammarVocab(IntEnum):
    """Stable decoder-vocab token ids.

    Ids 0..15 are control / structural tokens. Ids 16..25 are reserved
    for digit emission (``'0'``..``'9'``) when decoding MODE / X
    integers; see ``bpe_digit_str_to_grammar_ids``. Together they form
    a closed vocabulary of size ``GRAMMAR_VOCAB_SIZE``.
    """

    PAD = 0
    END = 1
    DECLARE_ATTACKERS_OPEN = 2
    ATTACK = 3
    DEFENDER = 4
    DECLARE_BLOCKERS_OPEN = 5
    BLOCK = 6
    ATTACKER = 7
    PRIORITY_OPEN = 8
    CHOOSE_TARGETS_OPEN = 9
    CHOOSE_MODE_OPEN = 10
    CHOOSE_X_OPEN = 11
    MAY_OPEN = 12
    YES = 13
    NO = 14
    NONE = 15


# Digit token id block: DIGIT_0_ID + d for d in 0..9.
DIGIT_0_ID: int = 16
DIGIT_9_ID: int = 25
GRAMMAR_VOCAB_SIZE: int = 26


def bpe_digit_str_to_grammar_ids(s: str) -> list[int]:
    """Map a decimal string (digits only) to decoder grammar ids.

    Used to translate target integers (MODE / X labels) into the
    decoder's digit-token sequence at training time.
    """

    out: list[int] = []
    for ch in s:
        if not ("0" <= ch <= "9"):
            raise ValueError(f"non-digit character {ch!r} in {s!r}")
        out.append(DIGIT_0_ID + (ord(ch) - ord("0")))
    return out


@dataclass(frozen=True)
class StepMask:
    """Mask over the next decoder step.

    ``vocab_mask`` is always ``[GRAMMAR_VOCAB_SIZE]``. ``pointer_mask``
    is ``None`` for vocab-only steps; otherwise its length equals the
    number of anchors of the kind expected at this step.
    """

    vocab_mask: np.ndarray
    pointer_mask: np.ndarray | None


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _empty_vocab() -> np.ndarray:
    return np.zeros(GRAMMAR_VOCAB_SIZE, dtype=bool)


def _vocab_with(*ids: int) -> np.ndarray:
    m = _empty_vocab()
    for i in ids:
        m[i] = True
    return m


def _is_digit_id(tok: int) -> bool:
    return DIGIT_0_ID <= tok <= DIGIT_9_ID


def _digit_value(tok: int) -> int:
    return tok - DIGIT_0_ID


def _validate_lengths(prefix: Sequence[int], prefix_pointers: Sequence[int]) -> None:
    if len(prefix) != len(prefix_pointers):
        raise ValueError(
            f"prefix / prefix_pointers length mismatch: {len(prefix)} vs {len(prefix_pointers)}"
        )


# --------------------------------------------------------------------------- #
# Per-decision-type state machines                                            #
# --------------------------------------------------------------------------- #


def _mask_priority(spec: DecisionSpec, prefix: Sequence[int]) -> StepMask:
    legal_actions = spec.anchors_of_kind(AnchorKind.LEGAL_ACTION)
    n = len(legal_actions)
    if len(prefix) == 0:
        return StepMask(_vocab_with(GrammarVocab.PRIORITY_OPEN), None)
    if len(prefix) == 1:
        if prefix[0] != GrammarVocab.PRIORITY_OPEN:
            raise ValueError("PRIORITY prefix must start with PRIORITY_OPEN")
        return StepMask(_empty_vocab(), np.ones(n, dtype=bool))
    if len(prefix) == 2:
        return StepMask(_vocab_with(GrammarVocab.END), None)
    raise ValueError(f"PRIORITY prefix too long: {len(prefix)}")


def _mask_declare_attackers(
    spec: DecisionSpec,
    prefix: Sequence[int],
    prefix_pointers: Sequence[int],
) -> StepMask:
    attackers = spec.anchors_of_kind(AnchorKind.LEGAL_ATTACKER)
    defenders = spec.anchors_of_kind(AnchorKind.DEFENDER)
    n_atk, n_def = len(attackers), len(defenders)

    if len(prefix) == 0:
        return StepMask(_vocab_with(GrammarVocab.DECLARE_ATTACKERS_OPEN), None)
    if prefix[0] != GrammarVocab.DECLARE_ATTACKERS_OPEN:
        raise ValueError("DECLARE_ATTACKERS prefix must start with OPEN")

    # Walk the pairs (ATTACK ptr DEFENDER ptr)+ after the OPEN token.
    # Determine current position within the repeating cycle.
    body = prefix[1:]
    body_ptrs = prefix_pointers[1:]
    chosen: set[int] = set()
    i = 0
    while i < len(body):
        # Expect ATTACK
        if body[i] != GrammarVocab.ATTACK:
            raise ValueError(f"expected ATTACK at body index {i}, got {body[i]}")
        if i + 1 >= len(body):
            # Need attacker pointer next.
            avail = np.array([a.subject_index not in chosen for a in attackers], dtype=bool)
            return StepMask(_empty_vocab(), avail)
        atk_ptr = body_ptrs[i + 1]
        if atk_ptr in chosen:
            raise ValueError(f"attacker {atk_ptr} chosen twice")
        chosen.add(atk_ptr)
        if i + 2 >= len(body):
            return StepMask(_vocab_with(GrammarVocab.DEFENDER), None)
        if body[i + 2] != GrammarVocab.DEFENDER:
            raise ValueError("expected DEFENDER token")
        if i + 3 >= len(body):
            return StepMask(_empty_vocab(), np.ones(n_def, dtype=bool))
        i += 4

    # Completed an even number of pairs (or just OPEN): expect ATTACK or END.
    if len(chosen) >= n_atk:
        return StepMask(_vocab_with(GrammarVocab.END), None)
    return StepMask(_vocab_with(GrammarVocab.ATTACK, GrammarVocab.END), None)


def _mask_declare_blockers(
    spec: DecisionSpec,
    prefix: Sequence[int],
    prefix_pointers: Sequence[int],
) -> StepMask:
    blockers = spec.anchors_of_kind(AnchorKind.LEGAL_BLOCKER)
    attackers = spec.anchors_of_kind(AnchorKind.LEGAL_ATTACKER)
    n_blk, n_atk = len(blockers), len(attackers)
    edges = spec.legal_edge_bitmap

    if len(prefix) == 0:
        return StepMask(_vocab_with(GrammarVocab.DECLARE_BLOCKERS_OPEN), None)
    if prefix[0] != GrammarVocab.DECLARE_BLOCKERS_OPEN:
        raise ValueError("DECLARE_BLOCKERS prefix must start with OPEN")

    body = prefix[1:]
    body_ptrs = prefix_pointers[1:]
    chosen_blockers: set[int] = set()
    i = 0
    while i < len(body):
        if body[i] != GrammarVocab.BLOCK:
            raise ValueError(f"expected BLOCK at body index {i}, got {body[i]}")
        if i + 1 >= len(body):
            avail = np.array([b.subject_index not in chosen_blockers for b in blockers], dtype=bool)
            return StepMask(_empty_vocab(), avail)
        blk_ptr = body_ptrs[i + 1]
        if blk_ptr in chosen_blockers:
            raise ValueError(f"blocker {blk_ptr} chosen twice")
        chosen_blockers.add(blk_ptr)
        if i + 2 >= len(body):
            return StepMask(_vocab_with(GrammarVocab.ATTACKER), None)
        if body[i + 2] != GrammarVocab.ATTACKER:
            raise ValueError("expected ATTACKER token")
        if i + 3 >= len(body):
            if edges is None:
                avail = np.ones(n_atk, dtype=bool)
            else:
                avail = edges[blk_ptr].astype(bool, copy=True)
            return StepMask(_empty_vocab(), avail)
        i += 4

    if len(chosen_blockers) >= n_blk:
        return StepMask(_vocab_with(GrammarVocab.END), None)
    return StepMask(_vocab_with(GrammarVocab.BLOCK, GrammarVocab.END), None)


def _mask_choose_targets(spec: DecisionSpec, prefix: Sequence[int]) -> StepMask:
    targets = spec.anchors_of_kind(AnchorKind.LEGAL_TARGET)
    n = len(targets)
    if len(prefix) == 0:
        return StepMask(_vocab_with(GrammarVocab.CHOOSE_TARGETS_OPEN), None)
    if prefix[0] != GrammarVocab.CHOOSE_TARGETS_OPEN:
        raise ValueError("CHOOSE_TARGETS prefix must start with OPEN")
    if len(prefix) == 1:
        return StepMask(_empty_vocab(), np.ones(n, dtype=bool))
    if len(prefix) == 2:
        return StepMask(_vocab_with(GrammarVocab.END), None)
    raise ValueError(f"CHOOSE_TARGETS prefix too long: {len(prefix)}")


def _mask_may(prefix: Sequence[int]) -> StepMask:
    if len(prefix) == 0:
        return StepMask(_vocab_with(GrammarVocab.MAY_OPEN), None)
    if prefix[0] != GrammarVocab.MAY_OPEN:
        raise ValueError("MAY prefix must start with MAY_OPEN")
    if len(prefix) == 1:
        return StepMask(_vocab_with(GrammarVocab.YES, GrammarVocab.NO), None)
    if len(prefix) == 2:
        if prefix[1] not in (GrammarVocab.YES, GrammarVocab.NO):
            raise ValueError("MAY second token must be YES or NO")
        return StepMask(_vocab_with(GrammarVocab.END), None)
    raise ValueError(f"MAY prefix too long: {len(prefix)}")


def _digit_mask_for_int(max_value: int, current: int) -> np.ndarray:
    """Mask of digits whose append yields a prefix-int still ≤ max_value.

    A "prefix-int" is the partial decimal value if we keep accumulating
    more digits. Appending digit ``d`` to ``current`` yields
    ``current * 10 + d``; for that to be a prefix of some integer in
    ``[0, max_value]``, we need ``current * 10 + d <= max_value``.
    """

    mask = np.zeros(GRAMMAR_VOCAB_SIZE, dtype=bool)
    base = current * 10
    if base > max_value:
        return mask
    upper = min(9, max_value - base)
    for d in range(0, upper + 1):
        mask[DIGIT_0_ID + d] = True
    return mask


def _mask_choose_int(
    spec: DecisionSpec,
    prefix: Sequence[int],
    open_token: int,
) -> StepMask:
    max_value = spec.max_value
    if max_value < 0:
        raise ValueError(f"max_value must be set for {spec.decision_type.name}")
    if len(prefix) == 0:
        return StepMask(_vocab_with(open_token), None)
    if prefix[0] != open_token:
        raise ValueError(f"prefix must start with {GrammarVocab(open_token).name}")

    digits = prefix[1:]
    # Validate digits emitted so far.
    current = 0
    for tok in digits:
        if not _is_digit_id(tok):
            raise ValueError(f"expected digit token, got {tok}")
        current = current * 10 + _digit_value(tok)
        if current > max_value:
            raise ValueError(f"prefix integer {current} exceeds max {max_value}")

    digit_mask = _digit_mask_for_int(max_value, current)
    if len(digits) == 0:
        # Need at least one digit before END is allowed.
        return StepMask(digit_mask, None)
    # At least one digit present and current ≤ max_value: END is allowed.
    digit_mask[GrammarVocab.END] = True
    return StepMask(digit_mask, None)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def next_mask(
    spec: DecisionSpec,
    prefix: Sequence[int],
    prefix_pointers: Sequence[int],
) -> StepMask:
    """Compute the legal next-token mask given a decoded prefix.

    ``prefix`` is the sequence of grammar token ids emitted so far.
    ``prefix_pointers`` is the parallel sequence of subject indices for
    pointer steps (``-1`` for non-pointer steps). Returns the mask over
    the combined vocab+pointer slot set for the *next* step.
    """

    _validate_lengths(prefix, prefix_pointers)
    dt = spec.decision_type
    if dt == DecisionType.PRIORITY:
        return _mask_priority(spec, prefix)
    if dt == DecisionType.DECLARE_ATTACKERS:
        return _mask_declare_attackers(spec, prefix, prefix_pointers)
    if dt == DecisionType.DECLARE_BLOCKERS:
        return _mask_declare_blockers(spec, prefix, prefix_pointers)
    if dt == DecisionType.CHOOSE_TARGETS:
        return _mask_choose_targets(spec, prefix)
    if dt == DecisionType.MAY:
        return _mask_may(prefix)
    if dt == DecisionType.CHOOSE_MODE:
        return _mask_choose_int(spec, prefix, GrammarVocab.CHOOSE_MODE_OPEN)
    if dt == DecisionType.CHOOSE_X:
        return _mask_choose_int(spec, prefix, GrammarVocab.CHOOSE_X_OPEN)
    raise ValueError(f"unknown decision type {dt}")


def batch_next_mask(
    specs: Sequence[DecisionSpec],
    prefix_tokens: np.ndarray,
    prefix_pointers: np.ndarray,
    prefix_lens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized next-mask over a batch of (spec, prefix) rows.

    Args:
        specs: per-row ``DecisionSpec``; length B.
        prefix_tokens: ``[B, T_max]`` int array of grammar token ids.
        prefix_pointers: ``[B, T_max]`` int array of pointer subject
            indices (-1 for non-pointer steps).
        prefix_lens: ``[B]`` int array of valid prefix lengths.

    Returns:
        ``(vocab_mask [B, GRAMMAR_VOCAB_SIZE], pointer_mask [B, N_max])``
        where ``N_max`` is the largest anchor count among rows whose
        next step is a pointer step (0 if none). Pointer rows whose
        anchor count is below ``N_max`` are right-padded with ``False``;
        vocab-only rows are also all-``False`` in the pointer slice.
    """

    b = len(specs)
    decision_type = np.fromiter((int(s.decision_type) for s in specs), dtype=np.int32, count=b)
    max_value = np.fromiter((int(s.max_value) for s in specs), dtype=np.int32, count=b)
    anchor_counts = _batched_anchor_counts(specs)
    legal_edges = _batched_legal_edges(specs, anchor_counts)
    return batch_next_mask_arrays(
        decision_type,
        anchor_counts,
        max_value,
        legal_edges,
        prefix_tokens,
        prefix_pointers,
        prefix_lens,
    )


def _batched_anchor_counts(specs: Sequence[DecisionSpec]) -> np.ndarray:
    counts = np.zeros((len(specs), len(AnchorKind)), dtype=np.int32)
    for row, spec in enumerate(specs):
        if spec.anchors:
            kinds = np.fromiter((int(a.kind) for a in spec.anchors), dtype=np.int32)
            counts[row] = np.bincount(kinds, minlength=len(AnchorKind))[: len(AnchorKind)]
    return counts


def _batched_legal_edges(
    specs: Sequence[DecisionSpec],
    anchor_counts: np.ndarray,
) -> np.ndarray | None:
    n_blk_max = int(anchor_counts[:, int(AnchorKind.LEGAL_BLOCKER)].max(initial=0))
    n_atk_max = int(anchor_counts[:, int(AnchorKind.LEGAL_ATTACKER)].max(initial=0))
    if n_blk_max == 0 or n_atk_max == 0:
        return None
    edges = np.ones((len(specs), n_blk_max, n_atk_max), dtype=bool)
    for row, spec in enumerate(specs):
        if spec.legal_edge_bitmap is None:
            continue
        n_blk, n_atk = spec.legal_edge_bitmap.shape
        edges[row, :n_blk, :n_atk] = spec.legal_edge_bitmap
    return edges


def batch_next_mask_arrays(
    decision_type: np.ndarray,
    anchor_counts: np.ndarray,
    max_value: np.ndarray,
    legal_edge_bitmap: np.ndarray | None,
    prefix_tokens: np.ndarray,
    prefix_pointers: np.ndarray,
    prefix_lens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized next-mask over primitive arrays.

    This is the hot implementation used by :func:`batch_next_mask`. It avoids
    the per-row scalar grammar state machine; all prefix-state decisions are
    computed with array masks over the whole batch.
    """

    b = int(decision_type.shape[0])
    vocab = np.zeros((b, GRAMMAR_VOCAB_SIZE), dtype=bool)
    desired_ptr_width = np.zeros((b,), dtype=np.int32)
    rows = np.arange(b)
    lens = prefix_lens.astype(np.int32, copy=False)
    first_tok = np.where(lens > 0, prefix_tokens[:, 0], -1)

    _mask_batch_priority(decision_type, anchor_counts, lens, first_tok, vocab, desired_ptr_width)
    _mask_batch_choose_targets(
        decision_type, anchor_counts, lens, first_tok, vocab, desired_ptr_width
    )
    _mask_batch_may(decision_type, lens, first_tok, prefix_tokens, vocab)
    _mask_batch_choose_int(decision_type, max_value, lens, first_tok, prefix_tokens, vocab)
    _mask_batch_attackers(
        decision_type,
        anchor_counts,
        lens,
        first_tok,
        prefix_pointers,
        vocab,
        desired_ptr_width,
    )
    _mask_batch_blockers(
        decision_type,
        anchor_counts,
        legal_edge_bitmap,
        lens,
        first_tok,
        prefix_pointers,
        vocab,
        desired_ptr_width,
    )

    n_max = int(desired_ptr_width.max(initial=0))
    pointer = np.zeros((b, n_max), dtype=bool)
    if n_max == 0:
        return vocab, pointer

    subjects = np.arange(n_max, dtype=np.int32)
    ptr_kind = _next_pointer_kind(decision_type, lens, first_tok, prefix_tokens)

    _fill_count_pointer(
        pointer,
        ptr_kind == int(AnchorKind.LEGAL_ACTION),
        subjects,
        anchor_counts,
        AnchorKind.LEGAL_ACTION,
    )
    _fill_count_pointer(
        pointer,
        ptr_kind == int(AnchorKind.LEGAL_TARGET),
        subjects,
        anchor_counts,
        AnchorKind.LEGAL_TARGET,
    )
    _fill_count_pointer(
        pointer, ptr_kind == int(AnchorKind.DEFENDER), subjects, anchor_counts, AnchorKind.DEFENDER
    )

    _fill_attacker_pointer(pointer, ptr_kind, subjects, anchor_counts, prefix_pointers, lens)
    _fill_blocker_pointer(pointer, ptr_kind, subjects, anchor_counts, prefix_pointers, lens)
    _fill_block_attack_target_pointer(
        pointer,
        ptr_kind,
        subjects,
        anchor_counts,
        legal_edge_bitmap,
        prefix_pointers,
        lens,
        rows,
    )
    return vocab, pointer


def batch_next_mask_torch(
    decision_type: Tensor,
    anchor_counts: Tensor,
    max_value: Tensor,
    legal_edge_bitmap: Tensor | None,
    prefix_tokens: Tensor,
    prefix_pointers: Tensor,
    prefix_lens: Tensor,
) -> tuple[Tensor, Tensor]:
    """Torch-native batched next-mask over primitive tensors."""

    device = prefix_tokens.device
    b = int(decision_type.shape[0])
    vocab = torch.zeros((b, GRAMMAR_VOCAB_SIZE), dtype=torch.bool, device=device)
    desired_ptr_width = torch.zeros((b,), dtype=torch.long, device=device)
    rows_idx = torch.arange(b, device=device)
    lens = prefix_lens.to(device=device, dtype=torch.long)
    decision_type = decision_type.to(device=device, dtype=torch.long)
    anchor_counts = anchor_counts.to(device=device, dtype=torch.long)
    max_value = max_value.to(device=device, dtype=torch.long)
    first_tok = torch.where(lens > 0, prefix_tokens[:, 0].to(torch.long), -torch.ones_like(lens))

    _mask_batch_priority_torch(
        decision_type, anchor_counts, lens, first_tok, vocab, desired_ptr_width
    )
    _mask_batch_choose_targets_torch(
        decision_type, anchor_counts, lens, first_tok, vocab, desired_ptr_width
    )
    _mask_batch_may_torch(decision_type, lens, first_tok, prefix_tokens, vocab)
    _mask_batch_choose_int_torch(decision_type, max_value, lens, first_tok, prefix_tokens, vocab)
    _mask_batch_attackers_torch(
        decision_type,
        anchor_counts,
        lens,
        first_tok,
        prefix_pointers,
        vocab,
        desired_ptr_width,
    )
    _mask_batch_blockers_torch(
        decision_type,
        anchor_counts,
        lens,
        first_tok,
        prefix_pointers,
        vocab,
        desired_ptr_width,
    )

    n_max = int(desired_ptr_width.max().item()) if desired_ptr_width.numel() else 0
    pointer = torch.zeros((b, n_max), dtype=torch.bool, device=device)
    if n_max == 0:
        return vocab, pointer

    subjects = torch.arange(n_max, dtype=torch.long, device=device)
    ptr_kind = _next_pointer_kind_torch(decision_type, lens, first_tok)
    _fill_count_pointer_torch(
        pointer,
        ptr_kind == int(AnchorKind.LEGAL_ACTION),
        subjects,
        anchor_counts,
        AnchorKind.LEGAL_ACTION,
    )
    _fill_count_pointer_torch(
        pointer,
        ptr_kind == int(AnchorKind.LEGAL_TARGET),
        subjects,
        anchor_counts,
        AnchorKind.LEGAL_TARGET,
    )
    _fill_count_pointer_torch(
        pointer,
        ptr_kind == int(AnchorKind.DEFENDER),
        subjects,
        anchor_counts,
        AnchorKind.DEFENDER,
    )
    _fill_attacker_pointer_torch(pointer, ptr_kind, subjects, anchor_counts, prefix_pointers, lens)
    _fill_blocker_pointer_torch(pointer, ptr_kind, subjects, anchor_counts, prefix_pointers, lens)
    _fill_block_attack_target_pointer_torch(
        pointer,
        ptr_kind,
        subjects,
        anchor_counts,
        legal_edge_bitmap,
        prefix_pointers,
        lens,
        rows_idx,
    )
    return vocab, pointer


def _mask_batch_priority_torch(
    decision_type: Tensor,
    anchor_counts: Tensor,
    lens: Tensor,
    first_tok: Tensor,
    vocab: Tensor,
    desired_ptr_width: Tensor,
) -> None:
    rows = decision_type == int(DecisionType.PRIORITY)
    vocab[rows & (lens == 0), int(GrammarVocab.PRIORITY_OPEN)] = True
    ptr_rows = rows & (lens == 1) & (first_tok == int(GrammarVocab.PRIORITY_OPEN))
    desired_ptr_width[ptr_rows] = anchor_counts[ptr_rows, int(AnchorKind.LEGAL_ACTION)]
    vocab[rows & (lens == 2), int(GrammarVocab.END)] = True


def _mask_batch_choose_targets_torch(
    decision_type: Tensor,
    anchor_counts: Tensor,
    lens: Tensor,
    first_tok: Tensor,
    vocab: Tensor,
    desired_ptr_width: Tensor,
) -> None:
    rows = decision_type == int(DecisionType.CHOOSE_TARGETS)
    vocab[rows & (lens == 0), int(GrammarVocab.CHOOSE_TARGETS_OPEN)] = True
    ptr_rows = rows & (lens == 1) & (first_tok == int(GrammarVocab.CHOOSE_TARGETS_OPEN))
    desired_ptr_width[ptr_rows] = anchor_counts[ptr_rows, int(AnchorKind.LEGAL_TARGET)]
    vocab[rows & (lens == 2), int(GrammarVocab.END)] = True


def _mask_batch_may_torch(
    decision_type: Tensor,
    lens: Tensor,
    first_tok: Tensor,
    prefix_tokens: Tensor,
    vocab: Tensor,
) -> None:
    rows = decision_type == int(DecisionType.MAY)
    vocab[rows & (lens == 0), int(GrammarVocab.MAY_OPEN)] = True
    choose_rows = rows & (lens == 1) & (first_tok == int(GrammarVocab.MAY_OPEN))
    vocab[choose_rows, int(GrammarVocab.YES)] = True
    vocab[choose_rows, int(GrammarVocab.NO)] = True
    end_rows = (
        rows
        & (lens == 2)
        & (first_tok == int(GrammarVocab.MAY_OPEN))
        & (
            (prefix_tokens[:, 1] == int(GrammarVocab.YES))
            | (prefix_tokens[:, 1] == int(GrammarVocab.NO))
        )
    )
    vocab[end_rows, int(GrammarVocab.END)] = True


def _mask_batch_choose_int_torch(
    decision_type: Tensor,
    max_value: Tensor,
    lens: Tensor,
    first_tok: Tensor,
    prefix_tokens: Tensor,
    vocab: Tensor,
) -> None:
    mode_rows = decision_type == int(DecisionType.CHOOSE_MODE)
    x_rows = decision_type == int(DecisionType.CHOOSE_X)
    vocab[mode_rows & (lens == 0), int(GrammarVocab.CHOOSE_MODE_OPEN)] = True
    vocab[x_rows & (lens == 0), int(GrammarVocab.CHOOSE_X_OPEN)] = True
    rows = (
        (mode_rows & (first_tok == int(GrammarVocab.CHOOSE_MODE_OPEN)))
        | (x_rows & (first_tok == int(GrammarVocab.CHOOSE_X_OPEN)))
    ) & (lens > 0)
    if not bool(rows.any()):
        return

    digit_len = (lens - 1).clamp_min(0)
    digits = prefix_tokens[:, 1:].to(torch.long) - DIGIT_0_ID
    valid_digit = (digits >= 0) & (digits <= 9)
    cols = torch.arange(digits.shape[1], dtype=torch.long, device=prefix_tokens.device)
    live = cols.unsqueeze(0) < digit_len.unsqueeze(1)
    valid_prefix = rows & (max_value >= 0) & (valid_digit | ~live).all(dim=1)
    exponents = (digit_len.unsqueeze(1) - cols.unsqueeze(0) - 1).clamp_min(0)
    powers = torch.where(live, torch.pow(torch.full_like(exponents, 10), exponents), 0)
    current = (digits.clamp(0, 9) * powers).sum(dim=1)
    valid_prefix = valid_prefix & (current <= max_value)
    base = current * 10
    digit_ids = torch.arange(10, dtype=torch.long, device=prefix_tokens.device)
    allowed_digits = valid_prefix.unsqueeze(1) & (
        (base.unsqueeze(1) + digit_ids.unsqueeze(0)) <= max_value.unsqueeze(1)
    )
    vocab[:, DIGIT_0_ID : DIGIT_9_ID + 1] |= allowed_digits
    vocab[valid_prefix & (digit_len > 0), int(GrammarVocab.END)] = True


def _mask_batch_attackers_torch(
    decision_type: Tensor,
    anchor_counts: Tensor,
    lens: Tensor,
    first_tok: Tensor,
    prefix_pointers: Tensor,
    vocab: Tensor,
    desired_ptr_width: Tensor,
) -> None:
    rows = decision_type == int(DecisionType.DECLARE_ATTACKERS)
    vocab[rows & (lens == 0), int(GrammarVocab.DECLARE_ATTACKERS_OPEN)] = True
    valid = rows & (first_tok == int(GrammarVocab.DECLARE_ATTACKERS_OPEN)) & (lens > 0)
    rem = (lens - 1).clamp_min(0) % 4
    completed_rows = valid & (rem == 0)
    chosen = _chosen_count_torch(prefix_pointers, lens, first_pointer_col=2, stride=4)
    can_continue = completed_rows & (chosen < anchor_counts[:, int(AnchorKind.LEGAL_ATTACKER)])
    vocab[can_continue, int(GrammarVocab.ATTACK)] = True
    vocab[completed_rows, int(GrammarVocab.END)] = True
    ptr_attacker = valid & (rem == 1)
    desired_ptr_width[ptr_attacker] = anchor_counts[ptr_attacker, int(AnchorKind.LEGAL_ATTACKER)]
    vocab[valid & (rem == 2), int(GrammarVocab.DEFENDER)] = True
    ptr_defender = valid & (rem == 3)
    desired_ptr_width[ptr_defender] = anchor_counts[ptr_defender, int(AnchorKind.DEFENDER)]


def _mask_batch_blockers_torch(
    decision_type: Tensor,
    anchor_counts: Tensor,
    lens: Tensor,
    first_tok: Tensor,
    prefix_pointers: Tensor,
    vocab: Tensor,
    desired_ptr_width: Tensor,
) -> None:
    rows = decision_type == int(DecisionType.DECLARE_BLOCKERS)
    vocab[rows & (lens == 0), int(GrammarVocab.DECLARE_BLOCKERS_OPEN)] = True
    valid = rows & (first_tok == int(GrammarVocab.DECLARE_BLOCKERS_OPEN)) & (lens > 0)
    rem = (lens - 1).clamp_min(0) % 4
    completed_rows = valid & (rem == 0)
    chosen = _chosen_count_torch(prefix_pointers, lens, first_pointer_col=2, stride=4)
    can_continue = completed_rows & (chosen < anchor_counts[:, int(AnchorKind.LEGAL_BLOCKER)])
    vocab[can_continue, int(GrammarVocab.BLOCK)] = True
    vocab[completed_rows, int(GrammarVocab.END)] = True
    ptr_blocker = valid & (rem == 1)
    desired_ptr_width[ptr_blocker] = anchor_counts[ptr_blocker, int(AnchorKind.LEGAL_BLOCKER)]
    vocab[valid & (rem == 2), int(GrammarVocab.ATTACKER)] = True
    ptr_attacker = valid & (rem == 3)
    desired_ptr_width[ptr_attacker] = anchor_counts[ptr_attacker, int(AnchorKind.LEGAL_ATTACKER)]


def _chosen_count_torch(
    prefix_pointers: Tensor,
    lens: Tensor,
    *,
    first_pointer_col: int,
    stride: int,
) -> Tensor:
    cols = torch.arange(prefix_pointers.shape[1], dtype=torch.long, device=prefix_pointers.device)
    live = (cols.unsqueeze(0) < lens.unsqueeze(1)) & (cols.unsqueeze(0) >= first_pointer_col)
    chosen_cols = live & (((cols.unsqueeze(0) - first_pointer_col) % stride) == 0)
    return chosen_cols.sum(dim=1)


def _next_pointer_kind_torch(decision_type: Tensor, lens: Tensor, first_tok: Tensor) -> Tensor:
    out = torch.full_like(decision_type, -1)
    priority = (
        (decision_type == int(DecisionType.PRIORITY))
        & (lens == 1)
        & (first_tok == int(GrammarVocab.PRIORITY_OPEN))
    )
    out[priority] = int(AnchorKind.LEGAL_ACTION)
    targets = (
        (decision_type == int(DecisionType.CHOOSE_TARGETS))
        & (lens == 1)
        & (first_tok == int(GrammarVocab.CHOOSE_TARGETS_OPEN))
    )
    out[targets] = int(AnchorKind.LEGAL_TARGET)
    attackers = (
        (decision_type == int(DecisionType.DECLARE_ATTACKERS))
        & (first_tok == int(GrammarVocab.DECLARE_ATTACKERS_OPEN))
        & (lens > 0)
    )
    atk_rem = (lens - 1).clamp_min(0) % 4
    out[attackers & (atk_rem == 1)] = int(AnchorKind.LEGAL_ATTACKER)
    out[attackers & (atk_rem == 3)] = int(AnchorKind.DEFENDER)
    blockers = (
        (decision_type == int(DecisionType.DECLARE_BLOCKERS))
        & (first_tok == int(GrammarVocab.DECLARE_BLOCKERS_OPEN))
        & (lens > 0)
    )
    blk_rem = (lens - 1).clamp_min(0) % 4
    out[blockers & (blk_rem == 1)] = int(AnchorKind.LEGAL_BLOCKER)
    out[blockers & (blk_rem == 3)] = int(AnchorKind.LEGAL_ATTACKER)
    return out


def _fill_count_pointer_torch(
    pointer: Tensor,
    rows: Tensor,
    subjects: Tensor,
    anchor_counts: Tensor,
    kind: AnchorKind,
) -> None:
    if not bool(rows.any()):
        return
    pointer[rows] = subjects.unsqueeze(0) < anchor_counts[rows, int(kind)].unsqueeze(1)


def _fill_attacker_pointer_torch(
    pointer: Tensor,
    ptr_kind: Tensor,
    subjects: Tensor,
    anchor_counts: Tensor,
    prefix_pointers: Tensor,
    lens: Tensor,
) -> None:
    rows = ptr_kind == int(AnchorKind.LEGAL_ATTACKER)
    if not bool(rows.any()):
        return
    allowed = subjects.unsqueeze(0) < anchor_counts[:, int(AnchorKind.LEGAL_ATTACKER)].unsqueeze(1)
    blocker_rows = rows & _is_block_attack_target_step_torch(lens)
    attack_rows = rows & ~blocker_rows
    if bool(attack_rows.any()):
        chosen = _chosen_subject_matrix_torch(prefix_pointers, lens, subjects, first_pointer_col=2)
        pointer[attack_rows] = allowed[attack_rows] & ~chosen[attack_rows]


def _fill_blocker_pointer_torch(
    pointer: Tensor,
    ptr_kind: Tensor,
    subjects: Tensor,
    anchor_counts: Tensor,
    prefix_pointers: Tensor,
    lens: Tensor,
) -> None:
    rows = ptr_kind == int(AnchorKind.LEGAL_BLOCKER)
    if not bool(rows.any()):
        return
    allowed = subjects.unsqueeze(0) < anchor_counts[:, int(AnchorKind.LEGAL_BLOCKER)].unsqueeze(1)
    chosen = _chosen_subject_matrix_torch(prefix_pointers, lens, subjects, first_pointer_col=2)
    pointer[rows] = allowed[rows] & ~chosen[rows]


def _fill_block_attack_target_pointer_torch(
    pointer: Tensor,
    ptr_kind: Tensor,
    subjects: Tensor,
    anchor_counts: Tensor,
    legal_edge_bitmap: Tensor | None,
    prefix_pointers: Tensor,
    lens: Tensor,
    rows_idx: Tensor,
) -> None:
    rows = (ptr_kind == int(AnchorKind.LEGAL_ATTACKER)) & _is_block_attack_target_step_torch(lens)
    if not bool(rows.any()):
        return
    allowed = subjects.unsqueeze(0) < anchor_counts[:, int(AnchorKind.LEGAL_ATTACKER)].unsqueeze(1)
    if legal_edge_bitmap is None:
        pointer[rows] = allowed[rows]
        return
    edge = legal_edge_bitmap.to(device=pointer.device, dtype=torch.bool)
    blocker_col = (lens - 2).clamp_min(0)
    blocker = prefix_pointers[rows_idx, blocker_col].clamp(min=0, max=edge.shape[1] - 1)
    edge_slice = edge[rows_idx, blocker]
    edge_allowed = torch.zeros_like(pointer)
    width = min(pointer.shape[1], edge_slice.shape[1])
    edge_allowed[:, :width] = edge_slice[:, :width]
    pointer[rows] = allowed[rows] & edge_allowed[rows]


def _chosen_subject_matrix_torch(
    prefix_pointers: Tensor,
    lens: Tensor,
    subjects: Tensor,
    *,
    first_pointer_col: int,
) -> Tensor:
    cols = torch.arange(prefix_pointers.shape[1], dtype=torch.long, device=prefix_pointers.device)
    live = (cols.unsqueeze(0) < lens.unsqueeze(1)) & (cols.unsqueeze(0) >= first_pointer_col)
    chosen_cols = live & (((cols.unsqueeze(0) - first_pointer_col) % 4) == 0)
    chosen_values = torch.where(chosen_cols, prefix_pointers.to(torch.long), -2)
    return (chosen_values.unsqueeze(2) == subjects.view(1, 1, -1)).any(dim=1)


def _is_block_attack_target_step_torch(lens: Tensor) -> Tensor:
    return ((lens - 1).clamp_min(0) % 4) == 3


def _mask_batch_priority(
    decision_type: np.ndarray,
    anchor_counts: np.ndarray,
    lens: np.ndarray,
    first_tok: np.ndarray,
    vocab: np.ndarray,
    desired_ptr_width: np.ndarray,
) -> None:
    rows = decision_type == int(DecisionType.PRIORITY)
    vocab[rows & (lens == 0), int(GrammarVocab.PRIORITY_OPEN)] = True
    ptr_rows = rows & (lens == 1) & (first_tok == int(GrammarVocab.PRIORITY_OPEN))
    desired_ptr_width[ptr_rows] = anchor_counts[ptr_rows, int(AnchorKind.LEGAL_ACTION)]
    vocab[rows & (lens == 2), int(GrammarVocab.END)] = True


def _mask_batch_choose_targets(
    decision_type: np.ndarray,
    anchor_counts: np.ndarray,
    lens: np.ndarray,
    first_tok: np.ndarray,
    vocab: np.ndarray,
    desired_ptr_width: np.ndarray,
) -> None:
    rows = decision_type == int(DecisionType.CHOOSE_TARGETS)
    vocab[rows & (lens == 0), int(GrammarVocab.CHOOSE_TARGETS_OPEN)] = True
    ptr_rows = rows & (lens == 1) & (first_tok == int(GrammarVocab.CHOOSE_TARGETS_OPEN))
    desired_ptr_width[ptr_rows] = anchor_counts[ptr_rows, int(AnchorKind.LEGAL_TARGET)]
    vocab[rows & (lens == 2), int(GrammarVocab.END)] = True


def _mask_batch_may(
    decision_type: np.ndarray,
    lens: np.ndarray,
    first_tok: np.ndarray,
    prefix_tokens: np.ndarray,
    vocab: np.ndarray,
) -> None:
    rows = decision_type == int(DecisionType.MAY)
    vocab[rows & (lens == 0), int(GrammarVocab.MAY_OPEN)] = True
    choose_rows = rows & (lens == 1) & (first_tok == int(GrammarVocab.MAY_OPEN))
    vocab[choose_rows, int(GrammarVocab.YES)] = True
    vocab[choose_rows, int(GrammarVocab.NO)] = True
    end_rows = (
        rows
        & (lens == 2)
        & (first_tok == int(GrammarVocab.MAY_OPEN))
        & (
            (prefix_tokens[:, 1] == int(GrammarVocab.YES))
            | (prefix_tokens[:, 1] == int(GrammarVocab.NO))
        )
    )
    vocab[end_rows, int(GrammarVocab.END)] = True


def _mask_batch_choose_int(
    decision_type: np.ndarray,
    max_value: np.ndarray,
    lens: np.ndarray,
    first_tok: np.ndarray,
    prefix_tokens: np.ndarray,
    vocab: np.ndarray,
) -> None:
    mode_rows = decision_type == int(DecisionType.CHOOSE_MODE)
    x_rows = decision_type == int(DecisionType.CHOOSE_X)
    vocab[mode_rows & (lens == 0), int(GrammarVocab.CHOOSE_MODE_OPEN)] = True
    vocab[x_rows & (lens == 0), int(GrammarVocab.CHOOSE_X_OPEN)] = True
    rows = (
        (mode_rows & (first_tok == int(GrammarVocab.CHOOSE_MODE_OPEN)))
        | (x_rows & (first_tok == int(GrammarVocab.CHOOSE_X_OPEN)))
    ) & (lens > 0)
    if not rows.any():
        return

    digit_len = np.maximum(lens - 1, 0)
    digits = prefix_tokens[:, 1:] - DIGIT_0_ID
    valid_digit = (digits >= 0) & (digits <= 9)
    cols = np.arange(digits.shape[1], dtype=np.int32)
    live = cols[None, :] < digit_len[:, None]
    valid_prefix = rows & (max_value >= 0) & (valid_digit | ~live).all(axis=1)
    powers = np.where(
        live,
        np.power(10, np.maximum(digit_len[:, None] - cols[None, :] - 1, 0)),
        0,
    )
    current = (np.clip(digits, 0, 9) * powers).sum(axis=1).astype(np.int64)
    valid_prefix &= current <= max_value
    base = current * 10
    digit_ids = np.arange(10, dtype=np.int64)
    allowed_digits = valid_prefix[:, None] & (
        (base[:, None] + digit_ids[None, :]) <= max_value[:, None]
    )
    vocab[:, DIGIT_0_ID : DIGIT_9_ID + 1] |= allowed_digits
    vocab[valid_prefix & (digit_len > 0), int(GrammarVocab.END)] = True


def _mask_batch_attackers(
    decision_type: np.ndarray,
    anchor_counts: np.ndarray,
    lens: np.ndarray,
    first_tok: np.ndarray,
    prefix_pointers: np.ndarray,
    vocab: np.ndarray,
    desired_ptr_width: np.ndarray,
) -> None:
    rows = decision_type == int(DecisionType.DECLARE_ATTACKERS)
    vocab[rows & (lens == 0), int(GrammarVocab.DECLARE_ATTACKERS_OPEN)] = True
    valid = rows & (first_tok == int(GrammarVocab.DECLARE_ATTACKERS_OPEN)) & (lens > 0)
    body_len = np.maximum(lens - 1, 0)
    rem = body_len % 4
    completed_rows = valid & (rem == 0)
    chosen = _chosen_count(prefix_pointers, lens, first_pointer_col=2, stride=4)
    can_continue = completed_rows & (chosen < anchor_counts[:, int(AnchorKind.LEGAL_ATTACKER)])
    vocab[can_continue, int(GrammarVocab.ATTACK)] = True
    vocab[completed_rows, int(GrammarVocab.END)] = True
    ptr_attacker = valid & (rem == 1)
    desired_ptr_width[ptr_attacker] = anchor_counts[ptr_attacker, int(AnchorKind.LEGAL_ATTACKER)]
    vocab[valid & (rem == 2), int(GrammarVocab.DEFENDER)] = True
    ptr_defender = valid & (rem == 3)
    desired_ptr_width[ptr_defender] = anchor_counts[ptr_defender, int(AnchorKind.DEFENDER)]


def _mask_batch_blockers(
    decision_type: np.ndarray,
    anchor_counts: np.ndarray,
    legal_edge_bitmap: np.ndarray | None,
    lens: np.ndarray,
    first_tok: np.ndarray,
    prefix_pointers: np.ndarray,
    vocab: np.ndarray,
    desired_ptr_width: np.ndarray,
) -> None:
    del legal_edge_bitmap
    rows = decision_type == int(DecisionType.DECLARE_BLOCKERS)
    vocab[rows & (lens == 0), int(GrammarVocab.DECLARE_BLOCKERS_OPEN)] = True
    valid = rows & (first_tok == int(GrammarVocab.DECLARE_BLOCKERS_OPEN)) & (lens > 0)
    body_len = np.maximum(lens - 1, 0)
    rem = body_len % 4
    completed_rows = valid & (rem == 0)
    chosen = _chosen_count(prefix_pointers, lens, first_pointer_col=2, stride=4)
    can_continue = completed_rows & (chosen < anchor_counts[:, int(AnchorKind.LEGAL_BLOCKER)])
    vocab[can_continue, int(GrammarVocab.BLOCK)] = True
    vocab[completed_rows, int(GrammarVocab.END)] = True
    ptr_blocker = valid & (rem == 1)
    desired_ptr_width[ptr_blocker] = anchor_counts[ptr_blocker, int(AnchorKind.LEGAL_BLOCKER)]
    vocab[valid & (rem == 2), int(GrammarVocab.ATTACKER)] = True
    ptr_attacker = valid & (rem == 3)
    desired_ptr_width[ptr_attacker] = anchor_counts[ptr_attacker, int(AnchorKind.LEGAL_ATTACKER)]


def _chosen_count(
    prefix_pointers: np.ndarray,
    lens: np.ndarray,
    *,
    first_pointer_col: int,
    stride: int,
) -> np.ndarray:
    cols = np.arange(prefix_pointers.shape[1], dtype=np.int32)
    live = (cols[None, :] < lens[:, None]) & (cols[None, :] >= first_pointer_col)
    chosen_cols = live & (((cols[None, :] - first_pointer_col) % stride) == 0)
    return chosen_cols.sum(axis=1)


def _next_pointer_kind(
    decision_type: np.ndarray,
    lens: np.ndarray,
    first_tok: np.ndarray,
    prefix_tokens: np.ndarray,
) -> np.ndarray:
    del prefix_tokens
    out = np.full((decision_type.shape[0],), -1, dtype=np.int32)
    priority = (
        (decision_type == int(DecisionType.PRIORITY))
        & (lens == 1)
        & (first_tok == int(GrammarVocab.PRIORITY_OPEN))
    )
    out[priority] = int(AnchorKind.LEGAL_ACTION)
    targets = (
        (decision_type == int(DecisionType.CHOOSE_TARGETS))
        & (lens == 1)
        & (first_tok == int(GrammarVocab.CHOOSE_TARGETS_OPEN))
    )
    out[targets] = int(AnchorKind.LEGAL_TARGET)

    attackers = (
        (decision_type == int(DecisionType.DECLARE_ATTACKERS))
        & (first_tok == int(GrammarVocab.DECLARE_ATTACKERS_OPEN))
        & (lens > 0)
    )
    atk_rem = np.maximum(lens - 1, 0) % 4
    out[attackers & (atk_rem == 1)] = int(AnchorKind.LEGAL_ATTACKER)
    out[attackers & (atk_rem == 3)] = int(AnchorKind.DEFENDER)

    blockers = (
        (decision_type == int(DecisionType.DECLARE_BLOCKERS))
        & (first_tok == int(GrammarVocab.DECLARE_BLOCKERS_OPEN))
        & (lens > 0)
    )
    blk_rem = np.maximum(lens - 1, 0) % 4
    out[blockers & (blk_rem == 1)] = int(AnchorKind.LEGAL_BLOCKER)
    out[blockers & (blk_rem == 3)] = int(AnchorKind.LEGAL_ATTACKER)
    return out


def _fill_count_pointer(
    pointer: np.ndarray,
    rows: np.ndarray,
    subjects: np.ndarray,
    anchor_counts: np.ndarray,
    kind: AnchorKind,
) -> None:
    if not rows.any():
        return
    pointer[rows] = subjects[None, :] < anchor_counts[rows, int(kind), None]


def _fill_attacker_pointer(
    pointer: np.ndarray,
    ptr_kind: np.ndarray,
    subjects: np.ndarray,
    anchor_counts: np.ndarray,
    prefix_pointers: np.ndarray,
    lens: np.ndarray,
) -> None:
    rows = ptr_kind == int(AnchorKind.LEGAL_ATTACKER)
    if not rows.any():
        return
    allowed = subjects[None, :] < anchor_counts[:, int(AnchorKind.LEGAL_ATTACKER), None]
    blocker_rows = (ptr_kind == int(AnchorKind.LEGAL_ATTACKER)) & (
        _is_block_attack_target_step(lens)
    )
    attack_rows = rows & ~blocker_rows
    if attack_rows.any():
        chosen = _chosen_subject_matrix(prefix_pointers, lens, subjects, first_pointer_col=2)
        pointer[attack_rows] = allowed[attack_rows] & ~chosen[attack_rows]


def _fill_blocker_pointer(
    pointer: np.ndarray,
    ptr_kind: np.ndarray,
    subjects: np.ndarray,
    anchor_counts: np.ndarray,
    prefix_pointers: np.ndarray,
    lens: np.ndarray,
) -> None:
    rows = ptr_kind == int(AnchorKind.LEGAL_BLOCKER)
    if not rows.any():
        return
    allowed = subjects[None, :] < anchor_counts[:, int(AnchorKind.LEGAL_BLOCKER), None]
    chosen = _chosen_subject_matrix(prefix_pointers, lens, subjects, first_pointer_col=2)
    pointer[rows] = allowed[rows] & ~chosen[rows]


def _fill_block_attack_target_pointer(
    pointer: np.ndarray,
    ptr_kind: np.ndarray,
    subjects: np.ndarray,
    anchor_counts: np.ndarray,
    legal_edge_bitmap: np.ndarray | None,
    prefix_pointers: np.ndarray,
    lens: np.ndarray,
    rows_idx: np.ndarray,
) -> None:
    rows = (ptr_kind == int(AnchorKind.LEGAL_ATTACKER)) & _is_block_attack_target_step(lens)
    if not rows.any():
        return
    allowed = subjects[None, :] < anchor_counts[:, int(AnchorKind.LEGAL_ATTACKER), None]
    if legal_edge_bitmap is None:
        pointer[rows] = allowed[rows]
        return
    blocker_col = np.maximum(lens - 2, 0)
    blocker = prefix_pointers[rows_idx, blocker_col].clip(min=0)
    edge_slice = legal_edge_bitmap[rows_idx, blocker]
    edge_allowed = np.zeros_like(pointer)
    width = min(pointer.shape[1], edge_slice.shape[1])
    edge_allowed[:, :width] = edge_slice[:, :width]
    pointer[rows] = allowed[rows] & edge_allowed[rows]


def _chosen_subject_matrix(
    prefix_pointers: np.ndarray,
    lens: np.ndarray,
    subjects: np.ndarray,
    *,
    first_pointer_col: int,
) -> np.ndarray:
    cols = np.arange(prefix_pointers.shape[1], dtype=np.int32)
    live = (cols[None, :] < lens[:, None]) & (cols[None, :] >= first_pointer_col)
    chosen_cols = live & (((cols[None, :] - first_pointer_col) % 4) == 0)
    chosen_values = np.where(chosen_cols, prefix_pointers, -2)
    return (chosen_values[:, :, None] == subjects[None, None, :]).any(axis=1)


def _is_block_attack_target_step(lens: np.ndarray) -> np.ndarray:
    return (np.maximum(lens - 1, 0) % 4) == 3


__all__ = [
    "DIGIT_0_ID",
    "DIGIT_9_ID",
    "GRAMMAR_VOCAB_SIZE",
    "GrammarVocab",
    "StepMask",
    "batch_next_mask",
    "batch_next_mask_torch",
    "bpe_digit_str_to_grammar_ids",
    "next_mask",
]
