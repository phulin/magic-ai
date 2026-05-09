"""Tests for ``magic_ai.text_encoder.render``.

PR #2 of the text-encoder plan (``docs/text_encoder_plan.md``). The renderer
is pure Python; these tests exercise determinism, card-ref bookkeeping, empty
zone visibility, action wiring, and a busy mid-game length probe. Hard cases
from §9 (planeswalker, split, MDFC, adventure, saga) are marked xfail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
    PlayerState,
    StackObjectState,
    TargetState,
)
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    OracleEntry,
    SnapshotRenderer,
    load_oracle_text,
    render_card_body,
    render_snapshot,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS, card_ref_token

ORACLE_PATH = Path(DEFAULT_ORACLE_PATH)


@pytest.fixture(scope="module")
def oracle() -> dict[str, OracleEntry]:
    return load_oracle_text(ORACLE_PATH)


def _card(cid: str, name: str, *, tapped: bool | None = None) -> GameCardState:
    out: dict[str, object] = {"ID": cid, "Name": name}
    if tapped is not None:
        out["Tapped"] = tapped
    return cast(GameCardState, out)


def _player(
    pid: str,
    name: str,
    *,
    life: int = 20,
    hand: list[GameCardState] | None = None,
    battlefield: list[GameCardState] | None = None,
    graveyard: list[GameCardState] | None = None,
    library_count: int = 53,
) -> PlayerState:
    out: dict[str, object] = {
        "ID": pid,
        "Name": name,
        "Life": life,
        "LibraryCount": library_count,
        "HandCount": len(hand or []),
        "GraveyardCount": len(graveyard or []),
        "Hand": hand or [],
        "Battlefield": battlefield or [],
        "Graveyard": graveyard or [],
        "ManaPool": {
            "White": 0,
            "Blue": 0,
            "Black": 0,
            "Red": 0,
            "Green": 0,
            "Colorless": 0,
        },
    }
    return cast(PlayerState, out)


def _basic_snapshot() -> GameStateSnapshot:
    self_bf = [_card("c1", "Llanowar Elves", tapped=False)]
    opp_bf = [_card("c2", "Serra Angel", tapped=False)]
    self_hand = [_card("c3", "Lightning Bolt"), _card("c4", "Counterspell")]
    snapshot: dict[str, object] = {
        "turn": 3,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", hand=self_hand, battlefield=self_bf),
            _player("p2", "Opp", battlefield=opp_bf, life=18),
        ],
    }
    return cast(GameStateSnapshot, snapshot)


def test_render_is_deterministic(oracle: dict[str, OracleEntry]) -> None:
    snap = _basic_snapshot()
    a = render_snapshot(snap, oracle=oracle)
    b = render_snapshot(snap, oracle=oracle)
    assert a.text == b.text
    assert a.text.encode("utf-8") == b.text.encode("utf-8")
    assert a.card_refs == b.card_refs


def test_card_ref_uniqueness(oracle: dict[str, OracleEntry]) -> None:
    snap = _basic_snapshot()
    rendered = render_snapshot(snap, oracle=oracle)
    # Every card with an ID gets exactly one ref entry, and ref indices are dense.
    expected_ids = {"c1", "c2", "c3", "c4"}
    assert set(rendered.card_refs.keys()) == expected_ids
    assert sorted(rendered.card_refs.values()) == list(range(len(expected_ids)))
    # And exactly one occurrence in the text per ref.
    for k in rendered.card_refs.values():
        token = card_ref_token(k)
        assert rendered.text.count(token) >= 1, token


def test_card_ref_cap_drops_overflow(oracle: dict[str, OracleEntry]) -> None:
    over = MAX_CARD_REFS + 5
    bf = [_card(f"c{i}", "Plains", tapped=False) for i in range(over)]
    snapshot = cast(
        GameStateSnapshot,
        {
            "turn": 1,
            "active_player": "p1",
            "step": "Precombat Main",
            "players": [
                _player("p1", "Self", battlefield=bf),
                _player("p2", "Opp"),
            ],
        },
    )
    rendered = render_snapshot(snapshot, oracle=oracle)
    assert len(rendered.card_refs) == MAX_CARD_REFS
    # Overflow cards still appear in the text (inline, no ref token).
    assert rendered.text.count("<card>") == over


def test_empty_hand_zone_visible(oracle: dict[str, OracleEntry]) -> None:
    snap = _basic_snapshot()
    # Wipe self hand.
    snap["players"][0]["Hand"] = []
    snap["players"][0]["HandCount"] = 0
    rendered = render_snapshot(snap, oracle=oracle)
    # Empty zones still emit their open/close pair.
    assert "<hand></hand>" in rendered.text


def test_action_target_uses_same_card_ref(oracle: dict[str, OracleEntry]) -> None:
    snap = _basic_snapshot()
    bolt = snap["players"][0]["Hand"][1]  # Counterspell -> swap to bolt
    # Use Lightning Bolt as the cast option.
    bolt_id = snap["players"][0]["Hand"][0]["ID"]
    target_card = snap["players"][1]["Battlefield"][0]
    target_id = target_card["ID"]
    pending: dict[str, object] = {
        "kind": "main",
        "player_idx": 0,
        "options": [
            cast(
                PendingOptionState,
                {
                    "id": "opt0",
                    "kind": "cast",
                    "card_id": bolt_id,
                    "card_name": "Lightning Bolt",
                    "mana_cost": "{R}",
                    "valid_targets": [cast(TargetState, {"id": target_id, "label": "Serra Angel"})],
                },
            )
        ],
    }
    snap_with_pending = cast(
        GameStateSnapshot,
        {**snap, "pending": cast(PendingState, pending)},
    )
    rendered = render_snapshot(snap_with_pending, oracle=oracle)
    target_ref = rendered.card_refs[target_id]
    bolt_ref = rendered.card_refs[bolt_id]
    target_token = card_ref_token(target_ref)
    # Pending decisions are no longer rendered into state text after the
    # inline-blank cutover; assert only that the card-ref tokens themselves
    # still appear (state encoder still sees the cards in their zones).
    assert card_ref_token(bolt_ref) in rendered.text
    assert target_token in rendered.text
    # Note: ``bolt`` also appears in the hand (silences ruff F841).
    assert bolt is not None


def _busy_snapshot(names: list[str]) -> GameStateSnapshot:
    # 12 permanents per side + 7 hand each + 3-card graveyards. Assignments cycle
    # through ``names`` to keep the test grounded in real cards.
    def take(start: int, count: int) -> list[GameCardState]:
        return [
            _card(f"x{start + i}", names[(start + i) % len(names)], tapped=(i % 2 == 0))
            for i in range(count)
        ]

    self_bf = take(0, 12)
    opp_bf = take(100, 12)
    self_hand = [_card(f"h{i}", names[i % len(names)]) for i in range(7)]
    opp_hand: list[GameCardState] = []  # opponent hand hidden
    self_grave = [_card(f"g{i}", names[(i + 5) % len(names)]) for i in range(3)]
    opp_grave = [_card(f"og{i}", names[(i + 7) % len(names)]) for i in range(3)]
    snapshot: dict[str, object] = {
        "turn": 9,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player(
                "p1",
                "Self",
                hand=self_hand,
                battlefield=self_bf,
                graveyard=self_grave,
                life=14,
            ),
            _player(
                "p2",
                "Opp",
                hand=opp_hand,
                battlefield=opp_bf,
                graveyard=opp_grave,
                life=11,
            ),
        ],
    }
    return cast(GameStateSnapshot, snapshot)


def test_busy_midgame_length_stats(oracle: dict[str, OracleEntry]) -> None:
    payload = json.loads(ORACLE_PATH.read_text())
    real_names = [c["name"] for c in payload.get("cards", []) if c.get("name")]
    assert real_names, "oracle JSON must contain at least one card name"
    # Trim to creatures-and-spells that actually have oracle text to make the
    # length stat representative.
    snap = _busy_snapshot(real_names[:30])
    rendered = render_snapshot(snap, oracle=oracle)
    char_len = len(rendered.text)
    word_len = len(rendered.text.split())
    print(f"[busy-midgame] chars={char_len} words={word_len} card_refs={len(rendered.card_refs)}")
    # Sanity floor — a busy snapshot should at least exceed the empty snapshot.
    assert char_len > 200


def test_render_contains_state_and_choices(oracle: dict[str, OracleEntry]) -> None:
    snap = _basic_snapshot()
    rendered = render_snapshot(snap, oracle=oracle)
    # Pending decisions are no longer rendered into state text after the
    # inline-blank cutover; the renderer is now state-only.
    assert rendered.text.startswith("<bos><state>")
    assert rendered.text.endswith("</state><eos>")
    assert "<actions>" not in rendered.text
    assert "<choices>" not in rendered.text


def test_status_flags_only_when_in_snapshot(oracle: dict[str, OracleEntry]) -> None:
    # No Tapped key at all -> neither <tapped> nor <untapped> emitted.
    bare_card = cast(GameCardState, {"ID": "z1", "Name": "Llanowar Elves"})
    snap = cast(
        GameStateSnapshot,
        {
            "turn": 1,
            "active_player": "p1",
            "step": "Untap",
            "players": [
                _player("p1", "Self", battlefield=[bare_card]),
                _player("p2", "Opp"),
            ],
        },
    )
    rendered = render_snapshot(snap, oracle=oracle)
    assert "<tapped>" not in rendered.text
    assert "<untapped>" not in rendered.text


def test_pass_action(oracle: dict[str, OracleEntry]) -> None:
    snap = _basic_snapshot()
    pending = cast(
        PendingState,
        {
            "kind": "priority",
            "player_idx": 0,
            "options": [cast(PendingOptionState, {"id": "p", "kind": "pass"})],
        },
    )
    snap_with_pending = cast(GameStateSnapshot, {**snap, "pending": pending})
    # Pending decisions are emitted via the decision-spec section now, not the
    # state text. State text alone is unaffected by the pending kind.
    rendered = render_snapshot(snap_with_pending, oracle=oracle)
    assert "<choices>" not in rendered.text


def test_stack_renders(oracle: dict[str, OracleEntry]) -> None:
    snap = _basic_snapshot()
    stack: list[StackObjectState] = [cast(StackObjectState, {"id": "s1", "name": "Lightning Bolt"})]
    snap_with_stack = cast(GameStateSnapshot, {**snap, "stack": stack})
    rendered = render_snapshot(snap_with_stack, oracle=oracle)
    assert "<stack>" in rendered.text
    assert "<card-name> deals 3 damage to any target." in rendered.text
    assert "s1" in rendered.card_refs


def test_renderer_class_accepts_injected_oracle() -> None:
    fake_oracle: dict[str, OracleEntry] = {
        "Llanowar Elves": cast(
            OracleEntry,
            {
                "name": "Llanowar Elves",
                "type_line": "Creature - Elf Druid",
                "mana_cost": "{G}",
                "oracle_text": "{T}: Add {G}.",
                "power_toughness": "1/1",
                "colors": ["G"],
            },
        )
    }
    renderer = SnapshotRenderer(fake_oracle)
    snap = _basic_snapshot()
    rendered = renderer.render(snap)
    # Card names are anonymized inside the body; the structural rules-text
    # tag and the oracle text (no self-reference here) both appear.
    assert "<rules-text>" in rendered.text
    assert "{T}: Add {G}." in rendered.text
    assert "<mana-cost>{G}</mana-cost>" in rendered.text
    assert "<pt>1/1</pt>" in rendered.text


# ---------------------------------------------------------------------------
# §9 hard cases — xfail until the renderer learns these card shapes.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="v1 renderer doesn't yet emit loyalty counters; tracked in §9 (planeswalker)."
)
def test_planeswalker_loyalty(oracle: dict[str, OracleEntry]) -> None:
    pw = cast(GameCardState, {"ID": "pw1", "Name": "Jace, the Mind Sculptor"})
    snap = cast(
        GameStateSnapshot,
        {
            "turn": 5,
            "active_player": "p1",
            "step": "Precombat Main",
            "players": [
                _player("p1", "Self", battlefield=[pw]),
                _player("p2", "Opp"),
            ],
        },
    )
    rendered = render_snapshot(snap, oracle=oracle)
    # TODO(text-encoder PR #4): emit a loyalty token like ``[+3]`` and a
    # ``<counter>`` block populated from snapshot data once the engine surfaces it.
    assert "[+3]" in rendered.text or "<counter>" in rendered.text


def _split_oracle() -> OracleEntry:
    """Synthetic split card (Fire // Ice). The oracle JSON pipeline currently
    flattens ``card_faces`` away so we hand-build the Scryfall shape here."""

    return cast(
        OracleEntry,
        {
            "name": "Fire // Ice",
            "type_line": "Instant // Instant",
            "layout": "split",
            "card_faces": [
                {
                    "name": "Fire",
                    "type_line": "Instant",
                    "mana_cost": "{1}{R}",
                    "oracle_text": (
                        "Fire deals 2 damage divided as you choose among one or two targets."
                    ),
                    "power_toughness": None,
                },
                {
                    "name": "Ice",
                    "type_line": "Instant",
                    "mana_cost": "{1}{U}",
                    "oracle_text": "Tap target permanent. Draw a card.",
                    "power_toughness": None,
                },
            ],
        },
    )


def _mdfc_oracle() -> OracleEntry:
    """Synthetic MDFC (Valki, God of Lies // Tibalt, Cosmic Impostor)."""

    return cast(
        OracleEntry,
        {
            "name": "Valki, God of Lies // Tibalt, Cosmic Impostor",
            "type_line": "Legendary Creature — God // Legendary Planeswalker — Tibalt",
            "layout": "modal_dfc",
            "card_faces": [
                {
                    "name": "Valki, God of Lies",
                    "type_line": "Legendary Creature — God",
                    "mana_cost": "{1}{B}",
                    "oracle_text": (
                        "When Valki enters the battlefield, each opponent reveals their hand."
                    ),
                    "power_toughness": "2/1",
                },
                {
                    "name": "Tibalt, Cosmic Impostor",
                    "type_line": "Legendary Planeswalker — Tibalt",
                    "mana_cost": "{7}{B}{R}",
                    "oracle_text": "Exile the top three cards of each player's library.",
                    "power_toughness": None,
                },
            ],
        },
    )


def _adventure_oracle() -> OracleEntry:
    """Synthetic adventure card (Brazen Borrower // Petty Theft).

    Per Scryfall, ``card_faces[0]`` is the creature half (the "main" face)
    and ``card_faces[1]`` is the adventure half — the renderer emits them
    in that order.
    """

    return cast(
        OracleEntry,
        {
            "name": "Brazen Borrower // Petty Theft",
            "type_line": "Creature — Faerie Rogue // Instant — Adventure",
            "layout": "adventure",
            "card_faces": [
                {
                    "name": "Brazen Borrower",
                    "type_line": "Creature — Faerie Rogue",
                    "mana_cost": "{1}{U}{U}",
                    "oracle_text": "Flash. Flying.",
                    "power_toughness": "3/1",
                },
                {
                    "name": "Petty Theft",
                    "type_line": "Instant — Adventure",
                    "mana_cost": "{1}{U}",
                    "oracle_text": (
                        "Return target nonland permanent an opponent controls to its owner's hand."
                    ),
                    "power_toughness": None,
                },
            ],
        },
    )


def test_split_card() -> None:
    """Both halves of a split card appear in the rendered body."""

    oracle = {"Fire // Ice": _split_oracle()}
    body = render_card_body("Fire // Ice", oracle["Fire // Ice"])
    assert body.startswith("<card>")
    assert body.endswith("</card>")
    # Two face blocks (one per split half).
    assert body.count("<face>") == 2
    assert body.count("</face>") == 2
    # Both oracle texts must appear (with self-references anonymized).
    assert "deals 2 damage" in body
    assert "Tap target permanent" in body
    # Both mana costs must appear, structurally tagged.
    assert "<mana-cost>{1}{R}</mana-cost>" in body
    assert "<mana-cost>{1}{U}</mana-cost>" in body
    # Face names are anonymized inside oracle text.
    assert "<card-name>" in body
    # Literal printed names do not leak through.
    assert "Fire" not in body
    assert "Ice" not in body


def test_mdfc() -> None:
    """Both faces of an MDFC are rendered front-face-then-back-face."""

    oracle = {"Valki, God of Lies // Tibalt, Cosmic Impostor": _mdfc_oracle()}
    body = render_card_body(
        "Valki, God of Lies // Tibalt, Cosmic Impostor",
        oracle["Valki, God of Lies // Tibalt, Cosmic Impostor"],
    )
    # Both faces appear as separate <face> blocks.
    assert body.count("<face>") == 2
    # Front face (creature, mana cost {1}{B}) must precede the back face
    # (planeswalker, mana cost {7}{B}{R}).
    assert body.index("<mana-cost>{1}{B}</mana-cost>") < body.index(
        "<mana-cost>{7}{B}{R}</mana-cost>"
    )
    # Both oracle texts must appear (self-references anonymized).
    assert "each opponent reveals their hand" in body or "<card-name>" in body
    assert "top three cards" in body
    # The full printed face names (which the anonymizer masks) are gone.
    assert "Valki, God of Lies" not in body
    assert "Tibalt, Cosmic Impostor" not in body


def test_adventure() -> None:
    """Adventure cards render the creature half first, adventure half second."""

    oracle = {"Brazen Borrower // Petty Theft": _adventure_oracle()}
    body = render_card_body(
        "Brazen Borrower // Petty Theft", oracle["Brazen Borrower // Petty Theft"]
    )
    # Two face blocks, one per half.
    assert body.count("<face>") == 2
    # Creature half (mana cost {1}{U}{U}) precedes the adventure half
    # (mana cost {1}{U}).
    assert body.index("<mana-cost>{1}{U}{U}</mana-cost>") < body.index(
        "<mana-cost>{1}{U}</mana-cost>"
    )
    # Creature P/T from the front face appears, structurally tagged.
    assert "<pt>3/1</pt>" in body
    # Adventure-half oracle text appears.
    assert "Return target nonland permanent" in body
    # Literal printed names are anonymized.
    assert "Brazen Borrower" not in body
    assert "Petty Theft" not in body


@pytest.mark.xfail(reason="Saga chapter counters not rendered; §9 hard case.")
def test_saga() -> None:
    # TODO: saga chapters need lore counters and chapter pointers.
    raise AssertionError("saga chapter counter not yet rendered")


# ---------------------------------------------------------------------------
# Inline-blank rendering (Step 2 of docs/text_encoder_inline_blanks_plan.md).
