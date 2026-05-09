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

    Called once per decoder step from ``actor_critic.step``; per-call
    cost is ``O(B * prefix_len)``. Acceptable up to ``B≈256`` since the
    per-row work is tiny (a handful of dict/list ops). For higher batch
    sizes the native callback (step 9 of the decoder migration) replaces
    this entirely.

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
    vocab = np.zeros((b, GRAMMAR_VOCAB_SIZE), dtype=bool)
    per_row_ptr: list[np.ndarray | None] = []
    n_max = 0
    for i, spec in enumerate(specs):
        ln = int(prefix_lens[i])
        prefix = prefix_tokens[i, :ln].tolist()
        ptrs = prefix_pointers[i, :ln].tolist()
        m = next_mask(spec, prefix, ptrs)
        vocab[i] = m.vocab_mask
        per_row_ptr.append(m.pointer_mask)
        if m.pointer_mask is not None and m.pointer_mask.shape[0] > n_max:
            n_max = m.pointer_mask.shape[0]

    pointer = np.zeros((b, n_max), dtype=bool)
    for i, m in enumerate(per_row_ptr):
        if m is None:
            continue
        pointer[i, : m.shape[0]] = m
    return vocab, pointer


__all__ = [
    "DIGIT_0_ID",
    "DIGIT_9_ID",
    "GRAMMAR_VOCAB_SIZE",
    "GrammarVocab",
    "StepMask",
    "batch_next_mask",
    "bpe_digit_str_to_grammar_ids",
    "next_mask",
]
