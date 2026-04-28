"""Byte-for-byte parity test: slow-path renderer ≡ fast-path assembler.

PR 13-C from ``docs/text_encoder_plan.md`` §13. For each fixture snapshot we
run two paths:

  - Slow: ``render_snapshot`` -> ``tokenize_snapshot`` -> ``collate``.
  - Fast: ``emit_render_plan`` -> ``assemble_batch``.

The fast path must produce the same int32 ``token_ids``, anchor positions,
masks, and seq lengths as the slow path. Any divergence is a wire-format
bug — that's the contract the eventual Go-side emitter must satisfy.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
    PlayerState,
    StackObjectState,
    TargetState,
)
from magic_ai.text_encoder.assembler import assemble_batch
from magic_ai.text_encoder.batch import collate, tokenize_snapshot
from magic_ai.text_encoder.card_cache import (
    CardTokenCache,
    build_card_cache,
)
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    OracleEntry,
    load_oracle_text,
    render_snapshot,
)
from magic_ai.text_encoder.render_plan import emit_render_plan
from magic_ai.text_encoder.tokenizer import load_tokenizer
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    return load_tokenizer()


_MULTI_FACE_NAME = "Brazen Borrower // Petty Theft"


def _multi_face_oracle_entry() -> OracleEntry:
    """Synthetic adventure-card oracle entry.

    The production ``card_oracle_embeddings.json`` flattens ``card_faces``
    away (see ``scripts/build_card_embeddings.py``), so we construct the
    Scryfall-shaped entry inline. Adventure was chosen because it covers
    the renderer's "creature-half-first" ordering rule.
    """

    return cast(
        OracleEntry,
        {
            "name": _MULTI_FACE_NAME,
            "type_line": "Creature — Faerie Rogue // Instant — Adventure",
            "layout": "adventure",
            "card_faces": cast(
                list,
                [
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
                            "Return target nonland permanent an opponent controls"
                            " to its owner's hand."
                        ),
                        "power_toughness": None,
                    },
                ],
            ),
        },
    )


@pytest.fixture(scope="module")
def oracle() -> dict[str, OracleEntry]:
    base = load_oracle_text(Path(DEFAULT_ORACLE_PATH))
    # Inject one synthetic multi-face entry so the parity fixtures can use it.
    base[_MULTI_FACE_NAME] = _multi_face_oracle_entry()
    return base


@pytest.fixture(scope="module")
def cache(oracle: dict[str, OracleEntry], tokenizer: PreTrainedTokenizerFast) -> CardTokenCache:
    names = sorted(oracle.keys())
    return build_card_cache(names, oracle, tokenizer, missing_policy="raise")


@pytest.fixture(scope="module")
def card_row_lookup(cache: CardTokenCache) -> Callable[[str], int]:
    name_to_row = {name: idx for idx, name in enumerate(cache.row_to_name)}
    name_to_row[""] = 0  # unknown sentinel for missing names

    def _lookup(name: str) -> int:
        return name_to_row.get(name, 0)

    return _lookup


# ---------------------------------------------------------------------------
# Snapshot fixture builders (real card names from the oracle JSON)
# ---------------------------------------------------------------------------


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
    mana: dict[str, int] | None = None,
) -> PlayerState:
    pool = {
        "White": 0,
        "Blue": 0,
        "Black": 0,
        "Red": 0,
        "Green": 0,
        "Colorless": 0,
    }
    if mana:
        pool.update(mana)
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
        "ManaPool": pool,
    }
    return cast(PlayerState, out)


def _snap_lone_creature() -> GameStateSnapshot:
    bf = [_card("c1", "Llanowar Elves", tapped=False)]
    return cast(
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


def _snap_basic() -> GameStateSnapshot:
    self_bf = [
        _card("c1", "Llanowar Elves", tapped=False),
        _card("c2", "Forest", tapped=True),
    ]
    opp_bf = [_card("c3", "Serra Angel", tapped=False)]
    self_hand = [_card("c4", "Lightning Bolt"), _card("c5", "Counterspell")]
    return cast(
        GameStateSnapshot,
        {
            "turn": 3,
            "active_player": "p1",
            "step": "Precombat Main",
            "players": [
                _player("p1", "Self", hand=self_hand, battlefield=self_bf),
                _player("p2", "Opp", battlefield=opp_bf, life=18),
            ],
        },
    )


def _snap_mana_pool() -> GameStateSnapshot:
    snap = _snap_basic()
    snap["players"][0]["ManaPool"] = {
        "White": 0,
        "Blue": 1,
        "Black": 0,
        "Red": 1,
        "Green": 2,
        "Colorless": 0,
    }
    return snap


def _snap_with_pass() -> GameStateSnapshot:
    snap = _snap_basic()
    snap["pending"] = cast(
        PendingState,
        {
            "kind": "priority",
            "player_idx": 0,
            "options": [cast(PendingOptionState, {"id": "p", "kind": "pass"})],
        },
    )
    return snap


def _snap_cast_with_target() -> GameStateSnapshot:
    snap = _snap_basic()
    bolt_id = snap["players"][0]["Hand"][0]["ID"]
    target_id = snap["players"][1]["Battlefield"][0]["ID"]
    snap["pending"] = cast(
        PendingState,
        {
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
                        "valid_targets": [
                            cast(
                                TargetState,
                                {"id": target_id, "label": "Serra Angel"},
                            )
                        ],
                    },
                )
            ],
        },
    )
    return snap


def _snap_with_stack() -> GameStateSnapshot:
    snap = _snap_basic()
    snap["stack"] = [cast(StackObjectState, {"id": "s1", "name": "Lightning Bolt"})]
    return snap


def _snap_attack() -> GameStateSnapshot:
    snap = _snap_basic()
    bf_id = snap["players"][0]["Battlefield"][0]["ID"]
    snap["pending"] = cast(
        PendingState,
        {
            "kind": "combat",
            "player_idx": 0,
            "options": [
                cast(
                    PendingOptionState,
                    {
                        "id": "atk",
                        "kind": "attack",
                        "card_id": bf_id,
                        "card_name": "Llanowar Elves",
                    },
                )
            ],
        },
    )
    return snap


def _snap_activate() -> GameStateSnapshot:
    snap = _snap_basic()
    forest_id = snap["players"][0]["Battlefield"][1]["ID"]
    snap["pending"] = cast(
        PendingState,
        {
            "kind": "main",
            "player_idx": 0,
            "options": [
                cast(
                    PendingOptionState,
                    {
                        "id": "act",
                        "kind": "activate",
                        "card_id": forest_id,
                        "card_name": "Forest",
                        "ability_index": 0,
                        "mana_cost": "{T}",
                    },
                )
            ],
        },
    )
    return snap


def _snap_with_graveyard() -> GameStateSnapshot:
    snap = _snap_basic()
    snap["players"][0]["Graveyard"] = [
        _card("g1", "Lightning Bolt"),
        _card("g2", "Counterspell"),
    ]
    snap["players"][0]["GraveyardCount"] = 2
    return snap


def _snap_busy_midgame(oracle: dict[str, OracleEntry]) -> GameStateSnapshot:
    names = sorted(oracle.keys())[:20]

    def take(start: int, count: int) -> list[GameCardState]:
        return [
            _card(f"x{start + i}", names[(start + i) % len(names)], tapped=(i % 2 == 0))
            for i in range(count)
        ]

    self_bf = take(0, 6)
    opp_bf = take(50, 5)
    self_hand = [_card(f"h{i}", names[i % len(names)]) for i in range(4)]
    self_grave = [_card(f"g{i}", names[(i + 5) % len(names)]) for i in range(2)]
    return cast(
        GameStateSnapshot,
        {
            "turn": 7,
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
                _player("p2", "Opp", battlefield=opp_bf, life=11),
            ],
        },
    )


def _snap_multi_options() -> GameStateSnapshot:
    snap = _snap_basic()
    bolt_id = snap["players"][0]["Hand"][0]["ID"]
    target_id = snap["players"][1]["Battlefield"][0]["ID"]
    snap["pending"] = cast(
        PendingState,
        {
            "kind": "main",
            "player_idx": 0,
            "options": [
                cast(PendingOptionState, {"id": "p", "kind": "pass"}),
                cast(
                    PendingOptionState,
                    {
                        "id": "opt0",
                        "kind": "cast",
                        "card_id": bolt_id,
                        "card_name": "Lightning Bolt",
                        "mana_cost": "{R}",
                        "valid_targets": [
                            cast(TargetState, {"id": target_id, "label": "Serra Angel"})
                        ],
                    },
                ),
            ],
        },
    )
    return snap


def _snap_multi_face_card() -> GameStateSnapshot:
    """Snapshot containing a multi-face (adventure) card on the battlefield.

    Exercises the multi-face branch of ``render_card_body`` end-to-end: the
    slow path renders both faces inline; the fast path memcpys the cached
    body. Byte-equal parity here proves the cache keys multi-face cards
    correctly and that no face-specific rendering is happening during
    snapshot assembly.
    """

    bf = [_card("c1", _MULTI_FACE_NAME, tapped=False)]
    return cast(
        GameStateSnapshot,
        {
            "turn": 4,
            "active_player": "p1",
            "step": "Precombat Main",
            "players": [
                _player("p1", "Self", battlefield=bf),
                _player("p2", "Opp"),
            ],
        },
    )


FIXTURE_BUILDERS: list[tuple[str, Callable[..., GameStateSnapshot]]] = [
    ("lone_creature", _snap_lone_creature),
    ("basic", _snap_basic),
    ("mana_pool", _snap_mana_pool),
    ("pass_action", _snap_with_pass),
    ("cast_with_target", _snap_cast_with_target),
    ("attack", _snap_attack),
    ("activate", _snap_activate),
    ("with_stack", _snap_with_stack),
    ("with_graveyard", _snap_with_graveyard),
    ("multi_options", _snap_multi_options),
    ("busy_midgame", _snap_busy_midgame),
    ("multi_face_card", _snap_multi_face_card),
]


def _build_fixture(name: str, oracle: dict[str, OracleEntry]) -> GameStateSnapshot:
    builder = dict(FIXTURE_BUILDERS)[name]
    if name == "busy_midgame":
        return builder(oracle)
    return builder()


# ---------------------------------------------------------------------------
# Parity assertion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", [n for n, _ in FIXTURE_BUILDERS])
def test_render_plan_parity(
    fixture_name: str,
    oracle: dict[str, OracleEntry],
    tokenizer: PreTrainedTokenizerFast,
    cache: CardTokenCache,
    card_row_lookup: Callable[[str], int],
) -> None:
    snap = _build_fixture(fixture_name, oracle)
    actions = snap.get("pending", {}).get("options") if "pending" in snap else None

    # Slow path (the renderer is the source of truth).
    rendered = render_snapshot(snap, actions, oracle=oracle)
    example = tokenize_snapshot(rendered, tokenizer)
    pad_id = int(tokenizer.pad_token_id or 0)
    expected = collate([example], pad_id=pad_id)

    # Fast path.
    def _tokenize(s: str) -> list[int]:
        return list(tokenizer.encode(s, add_special_tokens=False))

    plan = emit_render_plan(
        snap,
        actions,
        card_row_lookup=card_row_lookup,
        tokenize=_tokenize,
        oracle=oracle,
    )
    actual = assemble_batch(
        [plan], cache, tokenizer, max_tokens=int(expected.token_ids.shape[1]) + 16
    )

    # seq lengths must match.
    expected_len = int(expected.seq_lengths[0])
    actual_len = int(actual.seq_lengths[0])
    if actual_len != expected_len:
        # Print the divergence for easier debug.
        e = expected.token_ids[0, :expected_len].tolist()
        a = actual.token_ids[0, :actual_len].tolist()
        first_diverge = next(
            (i for i in range(min(len(e), len(a))) if e[i] != a[i]), min(len(e), len(a))
        )
        ctx = 8
        lo = max(0, first_diverge - ctx)
        e_ctx = [tokenizer.convert_ids_to_tokens(int(t)) for t in e[lo : first_diverge + ctx]]
        a_ctx = [tokenizer.convert_ids_to_tokens(int(t)) for t in a[lo : first_diverge + ctx]]
        msg = (
            f"seq length mismatch: expected={expected_len} actual={actual_len}; "
            f"first divergence @ pos={first_diverge}\n"
            f"expected: {e_ctx}\nactual:   {a_ctx}"
        )
        raise AssertionError(msg)

    assert torch.equal(actual.token_ids[0, :actual_len], expected.token_ids[0, :expected_len]), (
        "token_ids divergence"
    )
    # card_ref_positions / option_positions / target_positions / masks.
    assert torch.equal(actual.card_ref_positions[0], expected.card_ref_positions[0])
    # option / target shapes may differ in width; compare valid prefix.
    n_opts = int(expected.option_mask[0].sum())
    assert int(actual.option_mask[0].sum()) == n_opts
    assert torch.equal(actual.option_positions[0, :n_opts], expected.option_positions[0, :n_opts])
    if n_opts and expected.target_positions.shape[2]:
        n_tgts = int(expected.target_mask[0].sum())
        assert int(actual.target_mask[0].sum()) == n_tgts
        for o in range(n_opts):
            cnt = int(expected.target_mask[0, o].sum())
            assert int(actual.target_mask[0, o].sum()) == cnt
            if cnt:
                assert torch.equal(
                    actual.target_positions[0, o, :cnt],
                    expected.target_positions[0, o, :cnt],
                )
