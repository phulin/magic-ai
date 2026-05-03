"""Tests for ``magic_ai.text_encoder.card_cache`` (PR 13-A).

Cover build + roundtrip, slice extraction, missing-oracle policy, hash
stability, and parity with the slow snapshot renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from magic_ai.game_state import GameCardState, GameStateSnapshot, PlayerState
from magic_ai.text_encoder.card_cache import (
    UNKNOWN_NAME,
    CardTokenCache,
    MissingOracleTextError,
    build_card_cache,
    compute_card_set_hash,
    load_card_cache,
    save_card_cache,
)
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    OracleEntry,
    load_oracle_text,
    render_snapshot,
)
from magic_ai.text_encoder.tokenizer import load_tokenizer

TEST_NAMES = [
    "Lightning Bolt",
    "Llanowar Elves",
    "Counterspell",
    "Black Lotus",
    "Forest",
]


@pytest.fixture(scope="module")
def oracle() -> dict[str, OracleEntry]:
    return load_oracle_text(Path(DEFAULT_ORACLE_PATH))


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def small_cache(
    oracle: dict[str, OracleEntry],
    tokenizer,
) -> CardTokenCache:
    return build_card_cache(TEST_NAMES, oracle, tokenizer)


# ---------------------------------------------------------------------------


def test_cache_shape_and_unknown_row(small_cache: CardTokenCache) -> None:
    assert small_cache.row_to_name[0] == UNKNOWN_NAME
    assert small_cache.row_to_name[1:] == TEST_NAMES
    # Row 0 has zero-length body (offsets[0] == offsets[1] == 0).
    assert int(small_cache.offsets[0]) == 0
    assert int(small_cache.offsets[1]) == 0
    assert small_cache.offsets.shape == (len(TEST_NAMES) + 2,)
    assert small_cache.token_buffer.dtype == torch.int32
    assert small_cache.offsets.dtype == torch.int64
    # Each real card body must be non-empty.
    for k in range(2, len(TEST_NAMES) + 2):
        assert int(small_cache.offsets[k]) > int(small_cache.offsets[k - 1])


def test_save_and_load_roundtrip(small_cache: CardTokenCache, tmp_path: Path) -> None:
    out = tmp_path / "cache.pt"
    save_card_cache(small_cache, out)
    loaded = load_card_cache(out)
    assert torch.equal(loaded.token_buffer, small_cache.token_buffer)
    assert torch.equal(loaded.offsets, small_cache.offsets)
    assert loaded.row_to_name == small_cache.row_to_name
    assert loaded.engine_card_set_hash == small_cache.engine_card_set_hash


def test_slice_decodes_to_card_name(small_cache: CardTokenCache, tokenizer) -> None:
    # Card names are anonymized inside the body — the encoder learns from
    # rules-text mechanics, not from the printed name. Each cached slice must
    # still be a well-formed ``<card>...</card>`` fragment.
    for k, _name in enumerate(TEST_NAMES, start=1):
        ids = small_cache.body_tokens(k)
        decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        assert decoded.startswith("<card>"), decoded
        assert decoded.endswith("</card>"), decoded


def test_missing_raises_lists_name(oracle: dict[str, OracleEntry], tokenizer) -> None:
    bogus = "Definitely Not A Real Magic Card 9999"
    names = [*TEST_NAMES, bogus]
    # ``oracle_db_path=None`` skips the on-disk Scryfall DB fallback; without
    # it, each call here re-parses the bulk oracle JSON to look up the bogus
    # name and adds ~1s.
    with pytest.raises(MissingOracleTextError) as info:
        build_card_cache(names, oracle, tokenizer, missing_policy="raise", oracle_db_path=None)
    assert bogus in info.value.missing
    assert bogus in str(info.value)


def test_missing_skip_emits_empty_body(oracle: dict[str, OracleEntry], tokenizer) -> None:
    bogus = "Definitely Not A Real Magic Card 9999"
    names = [*TEST_NAMES, bogus]
    cache = build_card_cache(names, oracle, tokenizer, missing_policy="skip", oracle_db_path=None)
    # Bogus is the last row -> row index = len(names)
    bogus_row = len(names)
    assert cache.row_to_name[bogus_row] == bogus
    assert int(cache.offsets[bogus_row + 1]) == int(cache.offsets[bogus_row])


def test_hash_stable_and_order_invariant() -> None:
    h1 = compute_card_set_hash(TEST_NAMES)
    h2 = compute_card_set_hash(TEST_NAMES)
    h3 = compute_card_set_hash(list(reversed(TEST_NAMES)))
    assert h1 == h2 == h3


def test_build_twice_same_hash(oracle: dict[str, OracleEntry], tokenizer) -> None:
    a = build_card_cache(TEST_NAMES, oracle, tokenizer)
    b = build_card_cache(list(reversed(TEST_NAMES)), oracle, tokenizer)
    assert a.engine_card_set_hash == b.engine_card_set_hash


# ---------------------------------------------------------------------------
# Refactor parity: cached body tokens equal the slice of
# tokenize-ish output for a snapshot containing only that card.
# ---------------------------------------------------------------------------


def _single_card_snapshot(card_name: str) -> GameStateSnapshot:
    card = cast(GameCardState, {"ID": "card-1", "Name": card_name})
    player_self: PlayerState = cast(
        PlayerState,
        {
            "ID": "p0",
            "Name": "self",
            "Life": 20,
            "LibraryCount": 53,
            "HandCount": 1,
            "GraveyardCount": 0,
            "Hand": [card],
            "Battlefield": [],
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
    player_opp: PlayerState = cast(
        PlayerState,
        {
            "ID": "p1",
            "Name": "opp",
            "Life": 20,
            "LibraryCount": 53,
            "HandCount": 0,
            "GraveyardCount": 0,
            "Hand": [],
            "Battlefield": [],
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
    return cast(
        GameStateSnapshot,
        {
            "turn": 1,
            "step": "main1",
            "active_player": "self",
            "players": [player_self, player_opp],
            "stack": [],
        },
    )


def test_cached_tokens_match_snapshot_render(
    small_cache: CardTokenCache,
    oracle: dict[str, OracleEntry],
    tokenizer,
) -> None:
    """Cache slice == BPE of the corresponding span in render_snapshot output.

    The snapshot renderer uses ``render_card_body`` for the static portion of
    each ``<card>`` block. For a card with no status flags, the rendered body
    is byte-identical to what the cache stored, so tokenizing it matches the
    cached slice exactly. Locks in the contract PR 13-C parity tests rely on.
    """

    name = "Lightning Bolt"
    rendered = render_snapshot(_single_card_snapshot(name), [], oracle=oracle)
    # Hand has no status flags for an instant -> the inline body == cache body.
    from magic_ai.text_encoder.render import render_card_body

    body_text = render_card_body(name, oracle[name])
    assert body_text in rendered.text
    cached_ids = small_cache.body_tokens(TEST_NAMES.index(name) + 1).tolist()
    direct_ids = tokenizer.encode(body_text, add_special_tokens=False)
    assert cached_ids == direct_ids


def _multi_face_oracle() -> OracleEntry:
    """Hand-built MDFC oracle entry — the production JSON flattens
    ``card_faces`` away, so multi-face fixtures are synthesized for tests."""

    return cast(
        OracleEntry,
        {
            "name": "Valki, God of Lies // Tibalt, Cosmic Impostor",
            "type_line": "Legendary Creature — God // Legendary Planeswalker — Tibalt",
            "layout": "modal_dfc",
            "card_faces": cast(
                list,
                [
                    {
                        "name": "Valki, God of Lies",
                        "type_line": "Legendary Creature — God",
                        "mana_cost": "{1}{B}",
                        "oracle_text": "When Valki enters, each opponent reveals their hand.",
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
            ),
        },
    )


def test_cached_tokens_match_snapshot_render_multi_face(
    oracle: dict[str, OracleEntry],
    tokenizer,
) -> None:
    """Cache parity must hold for multi-face cards too.

    The MDFC fixture exercises the ``card_faces`` branch of
    ``render_card_body``: the cached body must contain both face names and
    the cached token slice must match a fresh BPE of the body produced by
    ``render_card_body`` and round-trip through ``render_snapshot``.
    """

    from magic_ai.text_encoder.render import render_card_body

    name = "Valki, God of Lies // Tibalt, Cosmic Impostor"
    multi_oracle: dict[str, OracleEntry] = {**oracle, name: _multi_face_oracle()}
    cache = build_card_cache([name], multi_oracle, tokenizer)

    body_text = render_card_body(name, multi_oracle[name])
    # Card names are anonymized inside the body; both faces still surface as
    # separate ``<face>`` blocks with their own mana costs.
    assert body_text.count("<face>") == 2
    assert "<mana-cost>{1}{B}</mana-cost>" in body_text
    assert "<mana-cost>{7}{B}{R}</mana-cost>" in body_text

    rendered = render_snapshot(_single_card_snapshot(name), [], oracle=multi_oracle)
    assert body_text in rendered.text

    # Row 1 == only real card in this single-name cache.
    cached_ids = cache.body_tokens(1).tolist()
    direct_ids = tokenizer.encode(body_text, add_special_tokens=False)
    assert cached_ids == direct_ids
