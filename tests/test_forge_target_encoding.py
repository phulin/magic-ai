"""Tests for the Forge observed-event → decoder-target translators.

Each test builds a synthetic ``PendingState`` + observed-event dict, runs
the matching translator, and asserts that the produced sequence is legal
under :func:`magic_ai.text_encoder.grammar.next_mask` step-by-step.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from magic_ai.game_state import PendingOptionState, PendingState
from magic_ai.text_encoder.decision_spec import (
    AnchorKind,
    DecisionSpec,
    DecisionType,
    PointerAnchor,
)
from magic_ai.text_encoder.forge_target_encoding import (
    DecoderTarget,
    translate_attackers,
    translate_blockers,
    translate_choose_mode,
    translate_choose_target,
    translate_choose_x,
    translate_may,
    translate_priority,
)
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, GrammarVocab, next_mask


def _opt(**kwargs: object) -> PendingOptionState:
    return cast(PendingOptionState, dict(kwargs))


def _pending(kind: str, options: list[PendingOptionState], **extras: object) -> PendingState:
    return cast(PendingState, {"kind": kind, "player_idx": 0, "options": options, **extras})


def _assert_grammar_legal(spec: DecisionSpec, target: DecoderTarget) -> None:
    """Walk the prefix and assert every emitted token was legal."""

    prefix_tokens: list[int] = []
    prefix_pointers: list[int] = []
    for tok, subj, is_ptr in zip(
        target.output_token_ids,
        target.output_pointer_subjects,
        target.output_is_pointer,
        strict=True,
    ):
        mask = next_mask(spec, prefix_tokens, prefix_pointers)
        if is_ptr:
            assert mask.pointer_mask is not None, (
                f"expected pointer step at index {len(prefix_tokens)}; got vocab-only mask"
            )
            assert 0 <= subj < mask.pointer_mask.shape[0], (
                f"pointer subject {subj} out of range {mask.pointer_mask.shape[0]}"
            )
            assert bool(mask.pointer_mask[subj]), (
                f"pointer subject {subj} not in legal mask {mask.pointer_mask}"
            )
        else:
            assert mask.pointer_mask is None or not mask.pointer_mask.any(), (
                f"expected vocab step at index {len(prefix_tokens)}; got pointer mask"
            )
            assert 0 <= tok < GRAMMAR_VOCAB_SIZE, f"token id {tok} out of range"
            assert bool(mask.vocab_mask[tok]), (
                f"token {tok} ({GrammarVocab(tok).name}) not in legal vocab mask "
                f"at step {len(prefix_tokens)}"
            )
        prefix_tokens.append(tok)
        prefix_pointers.append(subj)


# --------------------------------------------------------------------------- #
# PRIORITY                                                                    #
# --------------------------------------------------------------------------- #


def test_priority_translates_play_event() -> None:
    options = [
        _opt(id="play:1", kind="play", card_name="Forest"),
        _opt(id="cast:2", kind="cast", card_name="Lightning Bolt"),
        _opt(id="pass", kind="pass"),
    ]
    pending = _pending("priority", options)
    observed = {"raw": "PlayerA played Forest.", "event_type": "ZONE"}
    target = translate_priority(pending, observed)
    assert target is not None
    assert target.decision_type == int(DecisionType.PRIORITY)
    assert target.output_token_ids == [
        int(GrammarVocab.PRIORITY_OPEN),
        int(GrammarVocab.PAD),
        int(GrammarVocab.END),
    ]
    assert target.output_is_pointer == [False, True, False]
    assert target.output_pointer_subjects == [-1, 0, -1]

    spec = DecisionSpec(
        decision_type=DecisionType.PRIORITY,
        anchors=[
            PointerAnchor(kind=AnchorKind.LEGAL_ACTION, token_position=10 + i, subject_index=i)
            for i in range(len(options))
        ],
    )
    _assert_grammar_legal(spec, target)


def test_priority_translates_pass() -> None:
    options = [
        _opt(id="play:1", kind="play", card_name="Forest"),
        _opt(id="pass", kind="pass"),
    ]
    pending = _pending("priority", options)
    target = translate_priority(pending, {"raw": "PlayerA passed priority."})
    assert target is not None
    assert target.output_pointer_subjects == [-1, 1, -1]


# --------------------------------------------------------------------------- #
# DECLARE_ATTACKERS                                                           #
# --------------------------------------------------------------------------- #


def test_attackers_translates_two_attackers() -> None:
    options = [
        _opt(id="atk:Bear", kind="attacker", card_name="Grizzly Bears"),
        _opt(id="atk:Elf", kind="attacker", card_name="Llanowar Elves"),
        _opt(id="atk:Dryad", kind="attacker", card_name="Dryad"),
    ]
    pending = _pending("attackers", options)
    observed = {
        "raw": "PlayerA assigned Grizzly Bears, Llanowar Elves to attack PlayerB.",
        "actor_name": "PlayerA",
        "attackers_text": "Grizzly Bears, Llanowar Elves",
        "defender_text": "PlayerB",
    }
    target = translate_attackers(pending, observed)
    assert target is not None
    assert target.decision_type == int(DecisionType.DECLARE_ATTACKERS)
    # OPEN, ATTACK, ptr=0, DEFENDER, ptr=1, ATTACK, ptr=1, DEFENDER, ptr=1, END
    assert target.output_token_ids[0] == int(GrammarVocab.DECLARE_ATTACKERS_OPEN)
    assert target.output_token_ids[-1] == int(GrammarVocab.END)
    assert target.output_pointer_subjects.count(0) == 1  # one attacker
    # Defender subject (opponent) appears twice.
    assert target.output_pointer_subjects.count(1) == 3  # attacker idx 1 + 2 defenders

    spec = DecisionSpec(
        decision_type=DecisionType.DECLARE_ATTACKERS,
        anchors=(
            [
                PointerAnchor(
                    kind=AnchorKind.LEGAL_ATTACKER,
                    token_position=10 + i,
                    subject_index=i,
                )
                for i in range(len(options))
            ]
            + [
                PointerAnchor(
                    kind=AnchorKind.DEFENDER,
                    token_position=20 + i,
                    subject_index=i,
                )
                for i in range(2)
            ]
        ),
    )
    _assert_grammar_legal(spec, target)


# --------------------------------------------------------------------------- #
# DECLARE_BLOCKERS                                                            #
# --------------------------------------------------------------------------- #


def test_blockers_translates_one_block() -> None:
    options = [
        _opt(
            id="block:WallA",
            kind="block",
            card_name="Wall of Wood",
            valid_targets=[{"id": "atk:1", "label": "Grizzly Bears"}],
        ),
        _opt(
            id="block:WallB",
            kind="block",
            card_name="Wall of Stone",
            valid_targets=[{"id": "atk:1", "label": "Grizzly Bears"}],
        ),
    ]
    pending = _pending("blockers", options)
    observed = {
        "raw": "PlayerB blocked Grizzly Bears with Wall of Wood.",
        "actor_name": "PlayerB",
        "assignments": [
            {
                "kind": "block",
                "actor_name": "PlayerB",
                "attacker_text": "Grizzly Bears",
                "blockers_text": "Wall of Wood",
            }
        ],
    }
    target = translate_blockers(pending, observed)
    assert target is not None
    assert target.decision_type == int(DecisionType.DECLARE_BLOCKERS)
    assert target.output_token_ids[0] == int(GrammarVocab.DECLARE_BLOCKERS_OPEN)
    assert target.output_token_ids[-1] == int(GrammarVocab.END)

    bitmap = np.ones((2, 1), dtype=np.bool_)
    spec = DecisionSpec(
        decision_type=DecisionType.DECLARE_BLOCKERS,
        anchors=(
            [
                PointerAnchor(
                    kind=AnchorKind.LEGAL_BLOCKER,
                    token_position=10 + i,
                    subject_index=i,
                )
                for i in range(2)
            ]
            + [
                PointerAnchor(
                    kind=AnchorKind.LEGAL_ATTACKER,
                    token_position=30,
                    subject_index=0,
                )
            ]
        ),
        legal_edge_bitmap=bitmap,
    )
    _assert_grammar_legal(spec, target)


# --------------------------------------------------------------------------- #
# CHOOSE_TARGETS                                                              #
# --------------------------------------------------------------------------- #


def test_choose_targets_picks_named_option() -> None:
    options = [
        _opt(id="t:1", kind="permanent", card_name="Forest"),
        _opt(id="t:2", kind="permanent", card_name="Mountain"),
    ]
    pending = _pending("permanent", options)
    observed = {"raw": "PlayerA chose Mountain"}
    target = translate_choose_target(pending, observed)
    assert target is not None
    assert target.decision_type == int(DecisionType.CHOOSE_TARGETS)
    assert target.output_pointer_subjects == [-1, 1, -1]

    spec = DecisionSpec(
        decision_type=DecisionType.CHOOSE_TARGETS,
        anchors=[
            PointerAnchor(kind=AnchorKind.LEGAL_TARGET, token_position=10 + i, subject_index=i)
            for i in range(2)
        ],
    )
    _assert_grammar_legal(spec, target)


# --------------------------------------------------------------------------- #
# MAY                                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("accepted,expected", [(True, GrammarVocab.YES), (False, GrammarVocab.NO)])
def test_may_yes_no(accepted: bool, expected: GrammarVocab) -> None:
    pending = _pending("may", [])
    target = translate_may(pending, {"accepted": accepted})
    assert target is not None
    assert target.output_token_ids == [
        int(GrammarVocab.MAY_OPEN),
        int(expected),
        int(GrammarVocab.END),
    ]
    spec = DecisionSpec(decision_type=DecisionType.MAY)
    _assert_grammar_legal(spec, target)


# --------------------------------------------------------------------------- #
# CHOOSE_MODE / CHOOSE_X                                                      #
# --------------------------------------------------------------------------- #


def test_choose_mode_emits_digits() -> None:
    options = [_opt(id=f"mode:{i}", kind="mode") for i in range(5)]
    pending = _pending("mode", options)
    target = translate_choose_mode(pending, {"chosen_index": 2})
    assert target is not None
    assert target.output_token_ids[0] == int(GrammarVocab.CHOOSE_MODE_OPEN)
    assert target.output_token_ids[-1] == int(GrammarVocab.END)
    spec = DecisionSpec(decision_type=DecisionType.CHOOSE_MODE, max_value=5)
    _assert_grammar_legal(spec, target)


def test_choose_x_emits_multidigit_value() -> None:
    pending = _pending("number", [], amount=12)
    target = translate_choose_x(pending, {"chosen_value": 12})
    assert target is not None
    assert target.output_token_ids[0] == int(GrammarVocab.CHOOSE_X_OPEN)
    assert target.output_token_ids[-1] == int(GrammarVocab.END)
    spec = DecisionSpec(decision_type=DecisionType.CHOOSE_X, max_value=12)
    _assert_grammar_legal(spec, target)
