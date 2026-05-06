"""Tests for ``magic_ai.text_encoder.batch``.

PR #3/#4 of the text-encoder plan — verifies that
:func:`tokenize_snapshot` recovers correct anchor token positions and that
:func:`collate` produces correctly-shaped padded tensors with ``-1``
sentinels and boolean masks.
"""

from __future__ import annotations

import json
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
    TargetState,
)
from magic_ai.text_encoder.batch import (
    TextEncodedBatch,
    TextEncodedExample,
    collate,
    tokenize_snapshot,
)
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    OracleEntry,
    load_oracle_text,
    render_snapshot,
)
from magic_ai.text_encoder.tokenizer import (
    MAX_CARD_REFS,
    TOKENIZER_DIR,
    card_ref_token,
    load_tokenizer,
)

ORACLE_PATH = Path(DEFAULT_ORACLE_PATH)


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR.exists():
        pytest.skip(
            f"tokenizer artifact missing at {TOKENIZER_DIR}; "
            "run scripts/build_text_encoder_vocab.py first"
        )
    return load_tokenizer()


@pytest.fixture(scope="module")
def oracle() -> dict[str, OracleEntry]:
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle JSON missing at {ORACLE_PATH}")
    return load_oracle_text(ORACLE_PATH)


@pytest.fixture(scope="module")
def real_card_name() -> str:
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle JSON missing at {ORACLE_PATH}")
    payload = json.loads(ORACLE_PATH.read_text())
    for record in payload.get("cards", []):
        name = record.get("name")
        if name:
            return cast(str, name)
    pytest.skip("no card names in oracle JSON")


# ---------------------------------------------------------------------------
# Snapshot helpers (mirror tests/test_text_render.py)
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


def _snapshot_with_action(real_name: str) -> GameStateSnapshot:
    self_bf = [_card("c1", real_name, tapped=False)]
    opp_bf = [_card("c2", real_name, tapped=True)]
    self_hand = [_card("c3", real_name)]
    snap: dict[str, object] = {
        "turn": 2,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", hand=self_hand, battlefield=self_bf),
            _player("p2", "Opp", battlefield=opp_bf, life=18),
        ],
        "pending": cast(
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
                            "card_id": "c3",
                            "card_name": real_name,
                            "mana_cost": "{R}",
                            "valid_targets": [
                                cast(TargetState, {"id": "c2", "label": real_name}),
                                cast(TargetState, {"id": "c1", "label": real_name}),
                            ],
                        },
                    ),
                    cast(
                        PendingOptionState,
                        {"id": "p", "kind": "pass"},
                    ),
                ],
            },
        ),
    }
    return cast(GameStateSnapshot, snap)


# ---------------------------------------------------------------------------
# tokenize_snapshot
# ---------------------------------------------------------------------------


def test_tokenize_recovers_anchor_positions(
    tokenizer, oracle: dict[str, OracleEntry], real_card_name: str
) -> None:
    snap = _snapshot_with_action(real_card_name)
    rendered = render_snapshot(snap, oracle=oracle)
    example = tokenize_snapshot(rendered, tokenizer)

    assert len(example.token_ids) == len(example.attention_mask)
    assert all(m == 1 for m in example.attention_mask)

    # Card-ref positions decode to <card-ref:K> for the right K.
    assert example.card_ref_positions, "expected at least one card-ref position"
    for k, pos in example.card_ref_positions.items():
        tok_str = tokenizer.convert_ids_to_tokens(int(example.token_ids[pos]))
        assert tok_str == card_ref_token(k), (
            f"card-ref pos {pos} decoded to {tok_str!r}, expected {card_ref_token(k)!r}"
        )

    # Engine-id mapping survives.
    assert example.card_ref_engine_ids, "expected engine-id mapping"
    for k, engine_id in example.card_ref_engine_ids.items():
        assert isinstance(engine_id, str)
        assert 0 <= k < MAX_CARD_REFS


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------


def _basic_snapshot(real_name: str, *, hand_size: int) -> GameStateSnapshot:
    self_hand = [_card(f"h{i}", real_name) for i in range(hand_size)]
    snap: dict[str, object] = {
        "turn": 1,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", hand=self_hand),
            _player("p2", "Opp"),
        ],
    }
    return cast(GameStateSnapshot, snap)


def test_collate_shapes_and_masks(
    tokenizer, oracle: dict[str, OracleEntry], real_card_name: str
) -> None:
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None

    examples: list[TextEncodedExample] = []
    # Three differing-length examples, with different inline blank layouts.
    examples.append(
        tokenize_snapshot(
            render_snapshot(_basic_snapshot(real_card_name, hand_size=1), oracle=oracle), tokenizer
        )
    )
    examples.append(
        tokenize_snapshot(
            render_snapshot(_snapshot_with_action(real_card_name), oracle=oracle), tokenizer
        )
    )
    examples.append(
        tokenize_snapshot(
            render_snapshot(_basic_snapshot(real_card_name, hand_size=4), oracle=oracle), tokenizer
        )
    )

    seq_lens = [len(ex.token_ids) for ex in examples]
    assert len(set(seq_lens)) > 1, "expected differing-length examples"

    batch: TextEncodedBatch = collate(examples, pad_id=pad_id)
    b = len(examples)
    max_t = max(seq_lens)

    # Shapes.
    assert batch.token_ids.shape == (b, max_t)
    assert batch.attention_mask.shape == (b, max_t)
    assert batch.card_ref_positions.shape == (b, MAX_CARD_REFS)
    assert batch.seq_lengths.shape == (b,)

    # dtypes.
    assert batch.token_ids.dtype == torch.int64
    assert batch.attention_mask.dtype == torch.int64
    assert batch.card_ref_positions.dtype == torch.int64

    # Per-row content + masks.
    for i, ex in enumerate(examples):
        t_i = seq_lens[i]
        assert int(batch.seq_lengths[i]) == t_i
        # Real tokens copied; pad region matches pad_id with mask=0.
        assert torch.equal(
            batch.token_ids[i, :t_i],
            torch.as_tensor(ex.token_ids, dtype=torch.int64),
        )
        assert torch.all(batch.attention_mask[i, :t_i] == 1)
        if t_i < max_t:
            assert torch.all(batch.token_ids[i, t_i:] == pad_id)
            assert torch.all(batch.attention_mask[i, t_i:] == 0)

        # card_ref_positions: -1 for absent K, exact match for present K.
        for k in range(MAX_CARD_REFS):
            expected = ex.card_ref_positions.get(k, -1)
            assert int(batch.card_ref_positions[i, k]) == expected


def test_collate_empty_raises(tokenizer) -> None:
    with pytest.raises(ValueError):
        collate([], pad_id=int(tokenizer.pad_token_id))
