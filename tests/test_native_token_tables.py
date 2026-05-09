"""Phase 3 round-trip parity test: TokenTables -> Go FFI -> echo back."""

from __future__ import annotations

from pathlib import Path

import pytest
from magic_ai.text_encoder.card_cache import build_card_cache
from magic_ai.text_encoder.native_token_tables import (
    LOOKUP_ABILITY,
    LOOKUP_ACTION_VERB,
    LOOKUP_CARD_BODY,
    LOOKUP_CARD_NAME,
    LOOKUP_CARD_REF,
    LOOKUP_COUNT,
    LOOKUP_FRAGMENT,
    LOOKUP_LIFE_OWNER,
    LOOKUP_MANA_GLYPH,
    LOOKUP_TURN_STEP,
    LOOKUP_ZONE_CLOSE,
    LOOKUP_ZONE_OPEN,
    native_lookup,
    native_summary,
    register_native_token_tables,
)
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    load_oracle_text,
)
from magic_ai.text_encoder.token_tables import (
    ABILITY_MAX,
    ABILITY_MIN,
    ACTION_VERBS_BY_ID,
    COUNT_MAX,
    COUNT_MIN,
    LIFE_MAX,
    LIFE_MIN,
    MANA_SYMBOLS,
    OWNER_NAMES,
    STEP_NAMES,
    TURN_MAX,
    TURN_MIN,
    ZONE_TAGS_BY_ID,
    Frag,
    build_token_tables,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS, load_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def oracle():
    return load_oracle_text(Path(DEFAULT_ORACLE_PATH))


@pytest.fixture(scope="module")
def cache(oracle, tokenizer):
    names = sorted(oracle.keys())
    return build_card_cache(names, oracle, tokenizer, missing_policy="raise")


@pytest.fixture(scope="module")
def registered_tables(tokenizer, cache):
    tables = build_token_tables(tokenizer, cache)
    register_native_token_tables(tables, tokenizer=tokenizer)
    return tables


def test_summary_sizes_match_python(registered_tables) -> None:
    summary = native_summary()
    assert summary["fragment_count"] == len(Frag)
    assert summary["turn_min"] == TURN_MIN
    assert summary["turn_max"] == TURN_MAX
    assert summary["step_count"] == len(STEP_NAMES)
    assert summary["life_min"] == LIFE_MIN
    assert summary["life_max"] == LIFE_MAX
    assert summary["owner_count"] == len(OWNER_NAMES)
    assert summary["ability_min"] == ABILITY_MIN
    assert summary["ability_max"] == ABILITY_MAX
    assert summary["count_min"] == COUNT_MIN
    assert summary["count_max"] == COUNT_MAX
    assert summary["zone_count"] == len(ZONE_TAGS_BY_ID)
    assert summary["action_verb_count"] == len(ACTION_VERBS_BY_ID)
    assert summary["mana_color_count"] == len(MANA_SYMBOLS)
    assert summary["card_ref_count"] == MAX_CARD_REFS
    assert summary["card_row_count"] == len(registered_tables.card_body)
    assert summary["pad_id"] == registered_tables.pad_id
    assert summary["option_id"] == registered_tables.option_id
    assert summary["card_closer"] == registered_tables.card_closer
    assert summary["status_tapped"] == registered_tables.status_tapped
    assert summary["status_untapped"] == registered_tables.status_untapped


def test_fragment_round_trip(registered_tables) -> None:
    for frag in Frag:
        assert native_lookup(LOOKUP_FRAGMENT, int(frag)) == registered_tables.structural[frag]


def test_turn_step_round_trip(registered_tables) -> None:
    for turn in (TURN_MIN, 1, 7, 42, TURN_MAX):
        for step_id in range(len(STEP_NAMES)):
            assert (
                native_lookup(LOOKUP_TURN_STEP, turn, step_id)
                == registered_tables.turn_step[(turn, step_id)]
            )


def test_life_owner_round_trip(registered_tables) -> None:
    for life in (LIFE_MIN, -1, 0, 20, 100, LIFE_MAX):
        for owner in range(len(OWNER_NAMES)):
            assert (
                native_lookup(LOOKUP_LIFE_OWNER, life, owner)
                == registered_tables.life_owner[(life, owner)]
            )


def test_ability_round_trip(registered_tables) -> None:
    for n in range(ABILITY_MIN, ABILITY_MAX + 1):
        assert native_lookup(LOOKUP_ABILITY, n) == registered_tables.ability[n]


def test_count_round_trip(registered_tables) -> None:
    for n in (COUNT_MIN, 1, 5, 53, COUNT_MAX):
        assert native_lookup(LOOKUP_COUNT, n) == registered_tables.count[n]


def test_zone_round_trip(registered_tables) -> None:
    for zone in range(len(ZONE_TAGS_BY_ID)):
        for owner in range(len(OWNER_NAMES)):
            assert (
                native_lookup(LOOKUP_ZONE_OPEN, zone, owner)
                == registered_tables.zone_open[(zone, owner)]
            )
            assert (
                native_lookup(LOOKUP_ZONE_CLOSE, zone, owner)
                == registered_tables.zone_close[(zone, owner)]
            )


def test_action_verb_round_trip(registered_tables) -> None:
    for kind_id in range(len(ACTION_VERBS_BY_ID)):
        assert native_lookup(LOOKUP_ACTION_VERB, kind_id) == registered_tables.action_verb[kind_id]


def test_mana_glyph_round_trip(registered_tables) -> None:
    for color_id in range(len(MANA_SYMBOLS)):
        assert native_lookup(LOOKUP_MANA_GLYPH, color_id) == registered_tables.mana_glyph[color_id]


def test_card_ref_round_trip(registered_tables) -> None:
    for k in (0, 1, 50, MAX_CARD_REFS - 1):
        assert native_lookup(LOOKUP_CARD_REF, k) == [registered_tables.card_ref[k]]


def test_card_body_round_trip(registered_tables) -> None:
    for row in (0, 1, 100, 1000, len(registered_tables.card_body) - 1):
        if 0 <= row < len(registered_tables.card_body):
            assert native_lookup(LOOKUP_CARD_BODY, row) == registered_tables.card_body[row]


def test_card_name_round_trip(registered_tables) -> None:
    for row in (0, 1, 100, 1000, len(registered_tables.card_name) - 1):
        if 0 <= row < len(registered_tables.card_name):
            assert native_lookup(LOOKUP_CARD_NAME, row) == registered_tables.card_name[row]


def test_out_of_bounds_returns_empty(registered_tables) -> None:
    assert native_lookup(LOOKUP_FRAGMENT, 9999) == []
    assert native_lookup(LOOKUP_TURN_STEP, TURN_MAX + 1, 0) == []
    assert native_lookup(LOOKUP_LIFE_OWNER, LIFE_MIN - 1, 0) == []
    assert native_lookup(LOOKUP_ABILITY, ABILITY_MAX + 1) == []
    assert native_lookup(LOOKUP_CARD_REF, MAX_CARD_REFS + 1) == []
