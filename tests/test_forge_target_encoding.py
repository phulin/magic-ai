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


def test_attackers_translates_new_format_by_id_prefix() -> None:
    """Real-Forge-log format: observed carries [{name, id_prefix}, ...]."""

    options = [
        _opt(
            id="3c792cf2-befb-4440-a7d7-f3a26a680b61",
            kind="attacker",
            card_name="Forest",
            permanent_id="3c792cf2-befb-4440-a7d7-f3a26a680b61",
        ),
        _opt(
            id="bc148f21-3f0b-4ea5-8624-cfba2554d608",
            kind="attacker",
            card_name="Forest",
            permanent_id="bc148f21-3f0b-4ea5-8624-cfba2554d608",
        ),
        _opt(
            id="acd1075b-c185-45dd-9f86-48b21d4e1ee8",
            kind="attacker",
            card_name="Plains",
            permanent_id="acd1075b-c185-45dd-9f86-48b21d4e1ee8",
        ),
    ]
    pending = _pending("attackers", options)
    observed = {
        "raw": "PlayerA attacks PlayerB with 2 creatures",
        "actor_name": "PlayerA",
        "defender_name": "PlayerB",
        "attackers": [
            {"name": "Forest", "id_prefix": "bc1"},
            {"name": "Forest", "id_prefix": "3c7"},
        ],
    }
    target = translate_attackers(pending, observed)
    assert target is not None
    # Tokens: OPEN, ATTACK, ptr, DEFENDER, ptr, ATTACK, ptr, DEFENDER, ptr, END
    # Subject indices reflect the order observed: option 1 then option 0.
    assert target.output_pointer_subjects[2] == 1  # first ATTACK ptr
    assert target.output_pointer_subjects[6] == 0  # second ATTACK ptr
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


def test_attackers_translates_no_attack() -> None:
    """observed.no_attack=True yields the empty OPEN-then-END target."""

    options = [
        _opt(
            id="3c792cf2-befb-4440-a7d7-f3a26a680b61",
            kind="attacker",
            card_name="Grizzly Bears",
            permanent_id="3c792cf2-befb-4440-a7d7-f3a26a680b61",
        ),
    ]
    pending = _pending("attackers", options)
    observed = {
        "raw": "PlayerA declares no attackers",
        "actor_name": "PlayerA",
        "attackers": [],
        "no_attack": True,
    }
    target = translate_attackers(pending, observed)
    assert target is not None
    assert target.output_token_ids == [
        int(GrammarVocab.DECLARE_ATTACKERS_OPEN),
        int(GrammarVocab.END),
    ]
    spec = DecisionSpec(
        decision_type=DecisionType.DECLARE_ATTACKERS,
        anchors=(
            [
                PointerAnchor(
                    kind=AnchorKind.LEGAL_ATTACKER,
                    token_position=10,
                    subject_index=0,
                )
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


def test_blockers_translates_new_format_by_id_prefix() -> None:
    """Real-Forge-log format: observed carries assignments with id prefixes."""

    attacker_id = "c828901c-42b5-4359-baa2-b6f6f7ae183f"  # Craw Wurm [c82]
    blocker_id = "df484445-d419-4485-a2f3-f8b2e2892dd5"  # Djinn Token [df4]
    other_blocker_id = "aaaa1234-0000-0000-0000-000000000000"
    options = [
        _opt(
            id=blocker_id,
            kind="block",
            card_name="Djinn Token",
            permanent_id=blocker_id,
            valid_targets=[{"id": attacker_id, "label": "Craw Wurm"}],
        ),
        _opt(
            id=other_blocker_id,
            kind="block",
            card_name="Wall of Stone",
            permanent_id=other_blocker_id,
            valid_targets=[{"id": attacker_id, "label": "Craw Wurm"}],
        ),
    ]
    pending = _pending("blockers", options)
    observed = {
        "raw": "PlayerA attacks PlayerB with 1 creature (blockers declared)",
        "actor_name": "PlayerB",
        "attacker_player": "PlayerA",
        "attackers": [{"name": "Craw Wurm", "id_prefix": "c82"}],
        "assignments": [
            {
                "attacker_name": "Craw Wurm",
                "attacker_id_prefix": "c82",
                "blocker_name": "Djinn Token",
                "blocker_id_prefix": "df4",
            }
        ],
    }
    target = translate_blockers(pending, observed)
    assert target is not None
    # OPEN, BLOCK, ptr_blk=0, ATTACKER, ptr_atk=0, END
    assert target.output_pointer_subjects[2] == 0  # blocker subject = option 0
    assert target.output_pointer_subjects[4] == 0  # attacker subject = first attacker

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


def test_blockers_translates_no_block() -> None:
    """observed.no_block=True yields the empty OPEN-then-END target."""

    attacker_id = "c828901c-42b5-4359-baa2-b6f6f7ae183f"
    blocker_id = "df484445-d419-4485-a2f3-f8b2e2892dd5"
    options = [
        _opt(
            id=blocker_id,
            kind="block",
            card_name="Wall of Wood",
            permanent_id=blocker_id,
            valid_targets=[{"id": attacker_id, "label": "Craw Wurm"}],
        ),
    ]
    pending = _pending("blockers", options)
    observed = {
        "raw": "no blocks declared",
        "actor_name": "PlayerB",
        "attackers": [{"name": "Craw Wurm", "id_prefix": "c82"}],
        "assignments": [],
        "no_block": True,
    }
    target = translate_blockers(pending, observed)
    assert target is not None
    assert target.output_token_ids == [
        int(GrammarVocab.DECLARE_BLOCKERS_OPEN),
        int(GrammarVocab.END),
    ]
    bitmap = np.ones((1, 1), dtype=np.bool_)
    spec = DecisionSpec(
        decision_type=DecisionType.DECLARE_BLOCKERS,
        anchors=(
            [
                PointerAnchor(
                    kind=AnchorKind.LEGAL_BLOCKER,
                    token_position=10,
                    subject_index=0,
                )
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
