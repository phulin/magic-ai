"""Tests for the per-decision-type grammar state machines."""

from __future__ import annotations

import numpy as np
import pytest
from magic_ai.text_encoder.decision_spec import (
    AnchorKind,
    DecisionSpec,
    DecisionType,
    PointerAnchor,
)
from magic_ai.text_encoder.grammar import (
    DIGIT_0_ID,
    GRAMMAR_VOCAB_SIZE,
    GrammarVocab,
    batch_next_mask,
    bpe_digit_str_to_grammar_ids,
    next_mask,
)


def _anchor(kind: AnchorKind, subject_index: int) -> PointerAnchor:
    return PointerAnchor(kind=kind, token_position=100 + subject_index, subject_index=subject_index)


def _only_true(mask: np.ndarray, ids: list[int]) -> bool:
    expected = np.zeros_like(mask)
    expected[ids] = True
    return bool(np.array_equal(mask, expected))


# --------------------------------------------------------------------------- #
# PRIORITY                                                                    #
# --------------------------------------------------------------------------- #


def test_priority_trajectory():
    spec = DecisionSpec(
        decision_type=DecisionType.PRIORITY,
        anchors=[_anchor(AnchorKind.LEGAL_ACTION, i) for i in range(4)],
    )
    # Step 0: must emit PRIORITY_OPEN.
    m = next_mask(spec, [], [])
    assert _only_true(m.vocab_mask, [GrammarVocab.PRIORITY_OPEN])
    assert m.pointer_mask is None
    # Step 1: pointer over 4 actions.
    m = next_mask(spec, [GrammarVocab.PRIORITY_OPEN], [-1])
    assert not m.vocab_mask.any()
    assert m.pointer_mask is not None and m.pointer_mask.shape == (4,)
    assert m.pointer_mask.all()
    # Step 2: END.
    m = next_mask(spec, [GrammarVocab.PRIORITY_OPEN, GrammarVocab.PAD], [-1, 2])
    assert _only_true(m.vocab_mask, [GrammarVocab.END])


# --------------------------------------------------------------------------- #
# DECLARE_ATTACKERS                                                           #
# --------------------------------------------------------------------------- #


def test_declare_attackers_trajectory_and_uniqueness():
    spec = DecisionSpec(
        decision_type=DecisionType.DECLARE_ATTACKERS,
        anchors=(
            [_anchor(AnchorKind.LEGAL_ATTACKER, i) for i in range(3)]
            + [_anchor(AnchorKind.DEFENDER, i) for i in range(2)]
        ),
    )
    # OPEN required.
    m = next_mask(spec, [], [])
    assert _only_true(m.vocab_mask, [GrammarVocab.DECLARE_ATTACKERS_OPEN])

    # After OPEN: ATTACK or END.
    pre = [GrammarVocab.DECLARE_ATTACKERS_OPEN]
    pp = [-1]
    m = next_mask(spec, pre, pp)
    assert _only_true(m.vocab_mask, [GrammarVocab.ATTACK, GrammarVocab.END])

    # ATTACK -> attacker pointer over all three.
    pre += [GrammarVocab.ATTACK]
    pp += [-1]
    m = next_mask(spec, pre, pp)
    assert m.pointer_mask is not None and m.pointer_mask.tolist() == [True, True, True]

    # Choose attacker subject 0 -> DEFENDER token.
    pre += [GrammarVocab.PAD]  # placeholder for pointer step
    pp += [0]
    m = next_mask(spec, pre, pp)
    assert _only_true(m.vocab_mask, [GrammarVocab.DEFENDER])

    # DEFENDER -> defender pointer over 2.
    pre += [GrammarVocab.DEFENDER]
    pp += [-1]
    m = next_mask(spec, pre, pp)
    assert m.pointer_mask is not None and m.pointer_mask.tolist() == [True, True]

    # Choose defender 1 -> ATTACK or END again.
    pre += [GrammarVocab.PAD]
    pp += [1]
    m = next_mask(spec, pre, pp)
    assert _only_true(m.vocab_mask, [GrammarVocab.ATTACK, GrammarVocab.END])

    # Continue with another ATTACK; attacker 0 must be excluded.
    pre += [GrammarVocab.ATTACK]
    pp += [-1]
    m = next_mask(spec, pre, pp)
    assert m.pointer_mask is not None and m.pointer_mask.tolist() == [False, True, True]


# --------------------------------------------------------------------------- #
# DECLARE_BLOCKERS                                                            #
# --------------------------------------------------------------------------- #


def test_declare_blockers_legal_edge_bitmap():
    # 2 blockers, 3 attackers; blocker 0 can block attackers {0, 2};
    # blocker 1 can block only attacker 1.
    edges = np.array(
        [
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    spec = DecisionSpec(
        decision_type=DecisionType.DECLARE_BLOCKERS,
        anchors=(
            [_anchor(AnchorKind.LEGAL_BLOCKER, i) for i in range(2)]
            + [_anchor(AnchorKind.LEGAL_ATTACKER, i) for i in range(3)]
        ),
        legal_edge_bitmap=edges,
    )

    pre = [GrammarVocab.DECLARE_BLOCKERS_OPEN]
    pp = [-1]
    # Choose BLOCK, then blocker 0.
    pre += [GrammarVocab.BLOCK]
    pp += [-1]
    m = next_mask(spec, pre, pp)
    assert m.pointer_mask is not None and m.pointer_mask.tolist() == [True, True]

    pre += [GrammarVocab.PAD]
    pp += [0]
    m = next_mask(spec, pre, pp)
    assert _only_true(m.vocab_mask, [GrammarVocab.ATTACKER])

    pre += [GrammarVocab.ATTACKER]
    pp += [-1]
    m = next_mask(spec, pre, pp)
    # Blocker 0's row of the bitmap.
    assert m.pointer_mask is not None and m.pointer_mask.tolist() == [True, False, True]

    # Pick attacker 2; then BLOCK or END; blocker 0 should be unavailable next.
    pre += [GrammarVocab.PAD]
    pp += [2]
    m = next_mask(spec, pre, pp)
    assert _only_true(m.vocab_mask, [GrammarVocab.BLOCK, GrammarVocab.END])

    pre += [GrammarVocab.BLOCK]
    pp += [-1]
    m = next_mask(spec, pre, pp)
    assert m.pointer_mask is not None and m.pointer_mask.tolist() == [False, True]

    # Blocker 1 -> ATTACKER -> only attacker 1 legal.
    pre += [GrammarVocab.PAD]
    pp += [1]
    pre += [GrammarVocab.ATTACKER]
    pp += [-1]
    m = next_mask(spec, pre, pp)
    assert m.pointer_mask is not None and m.pointer_mask.tolist() == [False, True, False]


# --------------------------------------------------------------------------- #
# CHOOSE_TARGETS                                                              #
# --------------------------------------------------------------------------- #


def test_choose_targets_trajectory():
    spec = DecisionSpec(
        decision_type=DecisionType.CHOOSE_TARGETS,
        anchors=[_anchor(AnchorKind.LEGAL_TARGET, i) for i in range(5)],
    )
    m = next_mask(spec, [], [])
    assert _only_true(m.vocab_mask, [GrammarVocab.CHOOSE_TARGETS_OPEN])

    m = next_mask(spec, [GrammarVocab.CHOOSE_TARGETS_OPEN], [-1])
    assert m.pointer_mask is not None and m.pointer_mask.shape == (5,) and m.pointer_mask.all()

    m = next_mask(spec, [GrammarVocab.CHOOSE_TARGETS_OPEN, GrammarVocab.PAD], [-1, 3])
    assert _only_true(m.vocab_mask, [GrammarVocab.END])


# --------------------------------------------------------------------------- #
# MAY                                                                         #
# --------------------------------------------------------------------------- #


def test_may_trajectory():
    spec = DecisionSpec(decision_type=DecisionType.MAY)
    m = next_mask(spec, [], [])
    assert _only_true(m.vocab_mask, [GrammarVocab.MAY_OPEN])
    m = next_mask(spec, [GrammarVocab.MAY_OPEN], [-1])
    assert _only_true(m.vocab_mask, [GrammarVocab.YES, GrammarVocab.NO])
    m = next_mask(spec, [GrammarVocab.MAY_OPEN, GrammarVocab.YES], [-1, -1])
    assert _only_true(m.vocab_mask, [GrammarVocab.END])
    m = next_mask(spec, [GrammarVocab.MAY_OPEN, GrammarVocab.NO], [-1, -1])
    assert _only_true(m.vocab_mask, [GrammarVocab.END])


# --------------------------------------------------------------------------- #
# CHOOSE_X / CHOOSE_MODE                                                      #
# --------------------------------------------------------------------------- #


def test_choose_x_digit_mask_with_max_12():
    spec = DecisionSpec(decision_type=DecisionType.CHOOSE_X, max_value=12)
    # Step 0: OPEN.
    m = next_mask(spec, [], [])
    assert _only_true(m.vocab_mask, [GrammarVocab.CHOOSE_X_OPEN])

    # Step 1: digits 0..9 all legal (each ≤ 12 as a prefix).
    m = next_mask(spec, [GrammarVocab.CHOOSE_X_OPEN], [-1])
    assert m.pointer_mask is None
    assert m.vocab_mask[GrammarVocab.END] is np.False_ or not m.vocab_mask[GrammarVocab.END]
    assert m.vocab_mask[DIGIT_0_ID : DIGIT_0_ID + 10].all()
    # No END yet.
    assert not m.vocab_mask[GrammarVocab.END]

    # After "1": next prefix-int = 10, 11, 12 -> digits {0,1,2}; plus END (current=1).
    pre = [GrammarVocab.CHOOSE_X_OPEN] + bpe_digit_str_to_grammar_ids("1")
    pp = [-1, -1]
    m = next_mask(spec, pre, pp)
    assert m.vocab_mask[GrammarVocab.END]
    legal_digits = [d for d in range(10) if m.vocab_mask[DIGIT_0_ID + d]]
    assert legal_digits == [0, 1, 2]

    # After "12": current=12 ≤ 12, no further digits allowed (12*10=120 > 12).
    pre = [GrammarVocab.CHOOSE_X_OPEN] + bpe_digit_str_to_grammar_ids("12")
    pp = [-1, -1, -1]
    m = next_mask(spec, pre, pp)
    assert m.vocab_mask[GrammarVocab.END]
    for d in range(10):
        assert not m.vocab_mask[DIGIT_0_ID + d]

    # Attempting to pass "13" raises (would never be reachable through masks).
    with pytest.raises(ValueError):
        next_mask(
            spec,
            [GrammarVocab.CHOOSE_X_OPEN] + bpe_digit_str_to_grammar_ids("13"),
            [-1, -1, -1],
        )


def test_choose_mode_uses_mode_open_token():
    spec = DecisionSpec(decision_type=DecisionType.CHOOSE_MODE, max_value=3)
    m = next_mask(spec, [], [])
    assert _only_true(m.vocab_mask, [GrammarVocab.CHOOSE_MODE_OPEN])
    m = next_mask(spec, [GrammarVocab.CHOOSE_MODE_OPEN], [-1])
    # Single-digit: 0..3 legal as start, END not yet legal.
    assert not m.vocab_mask[GrammarVocab.END]
    legal = [d for d in range(10) if m.vocab_mask[DIGIT_0_ID + d]]
    assert legal == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Helpers / batch API                                                         #
# --------------------------------------------------------------------------- #


def test_bpe_digit_str_to_grammar_ids_roundtrip():
    assert bpe_digit_str_to_grammar_ids("305") == [
        DIGIT_0_ID + 3,
        DIGIT_0_ID + 0,
        DIGIT_0_ID + 5,
    ]
    with pytest.raises(ValueError):
        bpe_digit_str_to_grammar_ids("3a")


def test_batch_next_mask_mixed_rows():
    specs = [
        DecisionSpec(decision_type=DecisionType.MAY),
        DecisionSpec(
            decision_type=DecisionType.PRIORITY,
            anchors=[_anchor(AnchorKind.LEGAL_ACTION, i) for i in range(3)],
        ),
    ]
    # Both at step 1: MAY -> {YES, NO}, PRIORITY -> pointer over 3 actions.
    prefix_tokens = np.array(
        [[GrammarVocab.MAY_OPEN, 0], [GrammarVocab.PRIORITY_OPEN, 0]], dtype=np.int32
    )
    prefix_pointers = np.array([[-1, -1], [-1, -1]], dtype=np.int32)
    prefix_lens = np.array([1, 1], dtype=np.int32)

    vocab, pointer = batch_next_mask(specs, prefix_tokens, prefix_pointers, prefix_lens)
    assert vocab.shape == (2, GRAMMAR_VOCAB_SIZE)
    assert pointer.shape == (2, 3)
    # Row 0: vocab YES/NO, no pointer.
    assert vocab[0, GrammarVocab.YES] and vocab[0, GrammarVocab.NO]
    assert not pointer[0].any()
    # Row 1: pointer all-true, vocab empty.
    assert not vocab[1].any()
    assert pointer[1].tolist() == [True, True, True]


def test_inconsistent_prefix_raises():
    spec = DecisionSpec(decision_type=DecisionType.MAY)
    with pytest.raises(ValueError):
        next_mask(spec, [GrammarVocab.YES], [-1])
    with pytest.raises(ValueError):
        next_mask(spec, [GrammarVocab.MAY_OPEN], [-1, -1])
