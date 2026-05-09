"""Tests for ``magic_ai.text_encoder.render_spec.render_decision_spec``.

Step 1 of the decoder-grammar migration. Each test builds a minimal
snapshot with a specific pending kind, renders the decision spec, and
asserts that decision-type, anchors, scalar params, and side-tensors
are shaped correctly. The state-text renderer is *not* exercised here
— `card_refs` is supplied directly.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
    PlayerState,
    TargetState,
)
from magic_ai.text_encoder.decision_spec import AnchorKind, DecisionType
from magic_ai.text_encoder.render_spec import (
    DecisionSpecRenderer,
    render_decision_spec,
)
from magic_ai.text_encoder.tokenizer import load_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def renderer(tokenizer) -> DecisionSpecRenderer:
    return DecisionSpecRenderer(tokenizer)


def _card(cid: str, name: str = "Card") -> GameCardState:
    return cast(GameCardState, {"ID": cid, "Name": name})


def _player(pid: str, *, battlefield: list[GameCardState] | None = None) -> PlayerState:
    return cast(
        PlayerState,
        {
            "ID": pid,
            "Name": pid,
            "Life": 20,
            "LibraryCount": 53,
            "HandCount": 0,
            "GraveyardCount": 0,
            "Hand": [],
            "Battlefield": battlefield or [],
            "Graveyard": [],
            "ManaPool": {
                "White": 0,
                "Blue": 0,
                "Black": 0,
                "Red": 0,
                "Green": 0,
                "Colorless": 0,
            },
        },
    )


def _snapshot(pending: PendingState) -> GameStateSnapshot:
    return cast(
        GameStateSnapshot,
        {
            "turn": 1,
            "active_player": "p1",
            "step": "Precombat Main",
            "players": [_player("p1"), _player("p2")],
            "pending": pending,
        },
    )


def _spec_brackets(tokenizer, spec_tokens: list[int]) -> tuple[int, int]:
    open_id = int(tokenizer.convert_tokens_to_ids("<spec-open>"))
    close_id = int(tokenizer.convert_tokens_to_ids("<spec-close>"))
    return open_id, close_id


def _check_envelope(tokenizer, spec) -> None:
    open_id, close_id = _spec_brackets(tokenizer, spec.spec_tokens)
    assert spec.spec_tokens, "spec_tokens must be non-empty"
    assert spec.spec_tokens[0] == open_id
    assert spec.spec_tokens[-1] == close_id


def _check_subject_indices(spec, kind: AnchorKind) -> None:
    indices = [a.subject_index for a in spec.anchors if a.kind == kind]
    assert indices == list(range(len(indices)))


def test_priority(renderer, tokenizer) -> None:
    pending = cast(
        PendingState,
        {
            "kind": "priority",
            "player_idx": 0,
            "options": [
                cast(PendingOptionState, {"id": "a", "kind": "pass"}),
                cast(PendingOptionState, {"id": "b", "kind": "pass"}),
                cast(PendingOptionState, {"id": "c", "kind": "pass"}),
            ],
        },
    )
    spec = render_decision_spec(_snapshot(pending), card_refs={}, tokenizer=tokenizer)
    assert spec.decision_type is DecisionType.PRIORITY
    assert len(spec.anchors) == 3
    assert all(a.kind is AnchorKind.LEGAL_ACTION for a in spec.anchors)
    _check_envelope(tokenizer, spec)
    _check_subject_indices(spec, AnchorKind.LEGAL_ACTION)


def test_declare_attackers(renderer, tokenizer) -> None:
    pending = cast(
        PendingState,
        {
            "kind": "attackers",
            "player_idx": 0,
            "options": [
                cast(PendingOptionState, {"id": "a", "kind": "attacker", "permanent_id": "p1"}),
                cast(PendingOptionState, {"id": "b", "kind": "attacker", "permanent_id": "p2"}),
            ],
        },
    )
    spec = renderer.render(_snapshot(pending), card_refs={})
    assert spec.decision_type is DecisionType.DECLARE_ATTACKERS
    attackers = [a for a in spec.anchors if a.kind is AnchorKind.LEGAL_ATTACKER]
    defenders = [a for a in spec.anchors if a.kind is AnchorKind.DEFENDER]
    assert len(attackers) == 2
    assert len(defenders) == 2  # implicit player defenders for v1
    _check_envelope(tokenizer, spec)
    _check_subject_indices(spec, AnchorKind.LEGAL_ATTACKER)


def test_declare_blockers_bitmap(renderer, tokenizer) -> None:
    # Two blockers; blocker 0 can block attackers A, B; blocker 1 only B.
    pending = cast(
        PendingState,
        {
            "kind": "blockers",
            "player_idx": 1,
            "options": [
                cast(
                    PendingOptionState,
                    {
                        "id": "blk0",
                        "kind": "blocker",
                        "permanent_id": "b0",
                        "valid_targets": [
                            cast(TargetState, {"id": "atkA"}),
                            cast(TargetState, {"id": "atkB"}),
                        ],
                    },
                ),
                cast(
                    PendingOptionState,
                    {
                        "id": "blk1",
                        "kind": "blocker",
                        "permanent_id": "b1",
                        "valid_targets": [cast(TargetState, {"id": "atkB"})],
                    },
                ),
            ],
        },
    )
    spec = renderer.render(_snapshot(pending), card_refs={})
    assert spec.decision_type is DecisionType.DECLARE_BLOCKERS
    blockers = [a for a in spec.anchors if a.kind is AnchorKind.LEGAL_BLOCKER]
    attackers = [a for a in spec.anchors if a.kind is AnchorKind.LEGAL_ATTACKER]
    assert len(blockers) == 2
    assert len(attackers) == 2
    assert spec.legal_edge_bitmap is not None
    assert spec.legal_edge_bitmap.shape == (2, 2)
    expected = np.array([[True, True], [False, True]], dtype=np.bool_)
    assert np.array_equal(spec.legal_edge_bitmap, expected)
    _check_envelope(tokenizer, spec)
    _check_subject_indices(spec, AnchorKind.LEGAL_BLOCKER)
    _check_subject_indices(spec, AnchorKind.LEGAL_ATTACKER)


@pytest.mark.parametrize("kind", ["permanent", "cards_from_hand", "card_from_library"])
def test_choose_targets(renderer, tokenizer, kind: str) -> None:
    pending = cast(
        PendingState,
        {
            "kind": kind,
            "player_idx": 0,
            "options": [
                cast(PendingOptionState, {"id": "t0", "kind": "choice"}),
                cast(PendingOptionState, {"id": "t1", "kind": "choice"}),
            ],
        },
    )
    spec = renderer.render(_snapshot(pending), card_refs={})
    assert spec.decision_type is DecisionType.CHOOSE_TARGETS
    targets = [a for a in spec.anchors if a.kind is AnchorKind.LEGAL_TARGET]
    assert len(targets) == 2
    _check_envelope(tokenizer, spec)
    _check_subject_indices(spec, AnchorKind.LEGAL_TARGET)


def test_may(renderer, tokenizer) -> None:
    pending = cast(PendingState, {"kind": "may", "player_idx": 0, "options": []})
    spec = renderer.render(_snapshot(pending), card_refs={})
    assert spec.decision_type is DecisionType.MAY
    assert spec.anchors == []
    _check_envelope(tokenizer, spec)


def test_choose_mode_max_value_digits(renderer, tokenizer) -> None:
    pending = cast(
        PendingState,
        {
            "kind": "mode",
            "player_idx": 0,
            "options": [
                cast(PendingOptionState, {"id": f"m{i}", "kind": "choice"}) for i in range(3)
            ],
        },
    )
    spec = renderer.render(_snapshot(pending), card_refs={})
    assert spec.decision_type is DecisionType.CHOOSE_MODE
    assert spec.max_value == 3
    digit_ids = [int(t) for t in tokenizer.encode("3", add_special_tokens=False)]
    # The digit ids should appear in spec_tokens (they live between
    # <max-value> and </max-value>).
    assert all(d in spec.spec_tokens for d in digit_ids)
    _check_envelope(tokenizer, spec)


def test_choose_x_uses_amount(renderer, tokenizer) -> None:
    pending = cast(
        PendingState,
        {
            "kind": "number",
            "player_idx": 0,
            "amount": 12,
            "options": [],
        },
    )
    spec = renderer.render(_snapshot(pending), card_refs={})
    assert spec.decision_type is DecisionType.CHOOSE_X
    assert spec.max_value == 12
    digit_ids = [int(t) for t in tokenizer.encode("12", add_special_tokens=False)]
    assert all(d in spec.spec_tokens for d in digit_ids)
    _check_envelope(tokenizer, spec)


def test_choose_x_falls_back_to_options_minus_one(renderer, tokenizer) -> None:
    pending = cast(
        PendingState,
        {
            "kind": "number",
            "player_idx": 0,
            "options": [
                cast(PendingOptionState, {"id": f"n{i}", "kind": "choice"}) for i in range(5)
            ],
        },
    )
    spec = renderer.render(_snapshot(pending), card_refs={})
    assert spec.max_value == 4


def test_anchor_token_positions_index_into_spec_tokens(renderer, tokenizer) -> None:
    """token_position is an offset into spec_tokens (0-based, before assembler shift)."""

    pending = cast(
        PendingState,
        {
            "kind": "priority",
            "player_idx": 0,
            "options": [
                cast(PendingOptionState, {"id": "a", "kind": "pass"}),
                cast(PendingOptionState, {"id": "b", "kind": "pass"}),
            ],
        },
    )
    spec = renderer.render(_snapshot(pending), card_refs={})
    legal_action_id = int(tokenizer.convert_tokens_to_ids("<legal-action>"))
    for anchor in spec.anchors:
        assert spec.spec_tokens[anchor.token_position] == legal_action_id


@pytest.mark.parametrize("kind", ["mana_color", "mulligan"])
def test_deferred_kinds_raise(renderer, tokenizer, kind: str) -> None:
    pending = cast(PendingState, {"kind": kind, "player_idx": 0, "options": []})
    with pytest.raises(NotImplementedError):
        renderer.render(_snapshot(pending), card_refs={})
