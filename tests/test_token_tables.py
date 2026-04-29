"""Parity test: every TokenTables entry equals the live tokenizer's output.

Phase 1 of the option-(c) assembler-port. This is the gate: if every
precomputed token list matches what ``tokenizer.encode(...)`` would produce
for the equivalent text, then the table is a faithful stand-in for the live
tokenizer on the assembler hot path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from magic_ai.text_encoder.card_cache import build_card_cache
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    load_oracle_text,
)
from magic_ai.text_encoder.token_tables import (
    ABILITY_MAX,
    ABILITY_MIN,
    ACTION_VERBS_BY_ID,
    CARD_CLOSER_TEXT,
    COUNT_MAX,
    COUNT_MIN,
    LIFE_MAX,
    LIFE_MIN,
    MANA_SYMBOLS,
    OWNER_NAMES,
    STATUS_TAPPED_TEXT,
    STATUS_UNTAPPED_TEXT,
    STEP_NAMES,
    TURN_MAX,
    TURN_MIN,
    ZONE_TAGS_BY_ID,
    Frag,
    build_token_tables,
    fragment_text,
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
def tables(tokenizer, cache):
    return build_token_tables(tokenizer, cache)


def _enc(tokenizer, text: str) -> list[int]:
    return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]


def test_singletons_resolve_to_single_ids(tokenizer, tables) -> None:
    assert tables.option_id == tokenizer.convert_tokens_to_ids("<option>")
    assert tables.target_open_id == tokenizer.convert_tokens_to_ids("<target>")
    assert tables.target_close_id == tokenizer.convert_tokens_to_ids("</target>")
    assert tables.tapped_id == tokenizer.convert_tokens_to_ids("<tapped>")
    assert tables.untapped_id == tokenizer.convert_tokens_to_ids("<untapped>")
    assert tables.pad_id == int(tokenizer.pad_token_id)


def test_status_and_card_closer_match(tokenizer, tables) -> None:
    assert tables.card_closer == _enc(tokenizer, CARD_CLOSER_TEXT)
    assert tables.status_tapped == _enc(tokenizer, STATUS_TAPPED_TEXT)
    assert tables.status_untapped == _enc(tokenizer, STATUS_UNTAPPED_TEXT)


def test_structural_fragments_match(tokenizer, tables) -> None:
    for frag in Frag:
        assert tables.structural[frag] == _enc(tokenizer, fragment_text(frag)), frag


def test_zone_tags_match(tokenizer, tables) -> None:
    for zone_id, tag in ZONE_TAGS_BY_ID.items():
        for owner_id, owner in enumerate(OWNER_NAMES):
            assert tables.zone_open[(zone_id, owner_id)] == _enc(tokenizer, f"<{owner}><{tag}>")
            assert tables.zone_close[(zone_id, owner_id)] == _enc(tokenizer, f"</{tag}></{owner}>")


def test_action_verbs_match(tokenizer, tables) -> None:
    for kind_id, verb in ACTION_VERBS_BY_ID.items():
        assert tables.action_verb[kind_id] == _enc(tokenizer, f" {verb}")


def test_mana_glyphs_match(tokenizer, tables) -> None:
    for color_id, sym in enumerate(MANA_SYMBOLS):
        assert tables.mana_glyph[color_id] == _enc(tokenizer, f"{{{sym}}}")


def test_turn_step_match(tokenizer, tables) -> None:
    for turn in (TURN_MIN, 1, 7, 42, TURN_MAX):
        for step_id, step in enumerate(STEP_NAMES):
            assert tables.turn_step[(turn, step_id)] == _enc(
                tokenizer, f" turn={turn} step={step} "
            )


def test_life_owner_match(tokenizer, tables) -> None:
    for life in (LIFE_MIN, -1, 0, 20, 100, LIFE_MAX):
        for owner_id, owner in enumerate(OWNER_NAMES):
            assert tables.life_owner[(life, owner_id)] == _enc(
                tokenizer, f"<{owner}> life={life} mana="
            )


def test_ability_match(tokenizer, tables) -> None:
    for n in range(ABILITY_MIN, ABILITY_MAX + 1):
        assert tables.ability[n] == _enc(tokenizer, f" ability {n}")


def test_count_match(tokenizer, tables) -> None:
    for n in (COUNT_MIN, 1, 5, 53, COUNT_MAX):
        assert tables.count[n] == _enc(tokenizer, f" count={n}")


def test_card_ref_singletons(tokenizer, tables) -> None:
    assert len(tables.card_ref) == MAX_CARD_REFS
    for k in range(MAX_CARD_REFS):
        assert tables.card_ref[k] == tokenizer.convert_tokens_to_ids(f"<card-ref:{k}>")


def test_card_body_strips_card_closer(tables) -> None:
    closer = tables.card_closer
    closer_len = len(closer)
    assert closer_len > 0
    for body in tables.card_body:
        if not body:
            continue
        # Either body is shorter than the closer or its tail differs from it.
        assert len(body) < closer_len or body[-closer_len:] != closer


def test_card_name_match(tokenizer, cache, tables) -> None:
    # Spot-check a few rows; full sweep would be ~20K encode calls.
    sample_rows = [0, 1, 100, 1000, len(cache.row_to_name) - 1]
    for row in sample_rows:
        if row < 0 or row >= len(cache.row_to_name):
            continue
        assert tables.card_name[row] == _enc(tokenizer, cache.row_to_name[row])


def test_table_sizes_match_bounds(tables) -> None:
    assert len(tables.turn_step) == (TURN_MAX - TURN_MIN + 1) * len(STEP_NAMES)
    assert len(tables.life_owner) == (LIFE_MAX - LIFE_MIN + 1) * len(OWNER_NAMES)
    assert len(tables.ability) == ABILITY_MAX - ABILITY_MIN + 1
    assert len(tables.count) == COUNT_MAX - COUNT_MIN + 1
    assert len(tables.mana_glyph) == len(MANA_SYMBOLS)
    assert len(tables.zone_open) == len(ZONE_TAGS_BY_ID) * len(OWNER_NAMES)
    assert len(tables.zone_close) == len(ZONE_TAGS_BY_ID) * len(OWNER_NAMES)
