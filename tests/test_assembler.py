"""Smaller, hand-written tests for ``assemble_batch``.

PR 13-C from ``docs/text_encoder_plan.md`` §13. These tests exercise the
opcode-decoder path with no coupling to ``emit_render_plan``: each test
hand-builds a plan via ``RenderPlanWriter`` and asserts the assembled
token stream contains the expected ids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from magic_ai.text_encoder.assembler import assemble_batch
from magic_ai.text_encoder.card_cache import build_card_cache
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    load_oracle_text,
)
from magic_ai.text_encoder.render_plan import (
    OWNER_SELF,
    STATUS_TAPPED,
    STATUS_TAPPED_KNOWN,
    ZONE_BATTLEFIELD,
    RenderPlanWriter,
)
from magic_ai.text_encoder.tokenizer import load_tokenizer


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
def name_to_row(cache):
    return {name: idx for idx, name in enumerate(cache.row_to_name)}


# ---------------------------------------------------------------------------
# Empty plan
# ---------------------------------------------------------------------------


def test_empty_plan_list_raises(cache, tokenizer) -> None:
    with pytest.raises(ValueError):
        assemble_batch([], cache, tokenizer, max_tokens=128)


# ---------------------------------------------------------------------------
# Single-card plan
# ---------------------------------------------------------------------------


def _bos_state_ids(tokenizer) -> list[int]:
    return list(tokenizer.encode("<bos><state>", add_special_tokens=False))


def _state_eos_ids(tokenizer) -> list[int]:
    return list(tokenizer.encode("</state><eos>", add_special_tokens=False))


def test_single_card_plan_decodes_card_name(cache, tokenizer, name_to_row, oracle) -> None:
    bolt_row = name_to_row["Lightning Bolt"]
    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens(_bos_state_ids(tokenizer))
    w.emit_open_zone(ZONE_BATTLEFIELD, OWNER_SELF)
    w.emit_place_card(slot_idx=0, card_row_id=bolt_row, status_bits=0, uuid_idx=0)
    w.emit_end_card()
    w.emit_close_zone()
    w.emit_literal_tokens(_state_eos_ids(tokenizer))
    w.emit_close_state()

    batch = assemble_batch([w.finalize()], cache, tokenizer, max_tokens=512)
    decoded = tokenizer.decode(batch.token_ids[0, : int(batch.seq_lengths[0])].tolist())
    assert "Lightning Bolt" in decoded

    # <card-ref:0> appears in the stream and is anchored.
    card_ref_id = tokenizer.convert_tokens_to_ids("<card-ref:0>")
    assert card_ref_id in batch.token_ids[0].tolist()
    assert int(batch.card_ref_positions[0, 0]) >= 0
    pos = int(batch.card_ref_positions[0, 0])
    assert int(batch.token_ids[0, pos]) == card_ref_id


# ---------------------------------------------------------------------------
# Status bit decode
# ---------------------------------------------------------------------------


def test_status_bits_emit_tapped(cache, tokenizer, name_to_row) -> None:
    forest_row = name_to_row["Forest"]
    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens(_bos_state_ids(tokenizer))
    w.emit_open_zone(ZONE_BATTLEFIELD, OWNER_SELF)
    w.emit_place_card(
        slot_idx=0,
        card_row_id=forest_row,
        status_bits=STATUS_TAPPED | STATUS_TAPPED_KNOWN,
        uuid_idx=0,
    )
    w.emit_end_card()
    w.emit_close_zone()
    w.emit_literal_tokens(_state_eos_ids(tokenizer))
    w.emit_close_state()

    batch = assemble_batch([w.finalize()], cache, tokenizer, max_tokens=512)
    tapped_id = tokenizer.convert_tokens_to_ids("<tapped>")
    untapped_id = tokenizer.convert_tokens_to_ids("<untapped>")
    ids = batch.token_ids[0, : int(batch.seq_lengths[0])].tolist()
    assert tapped_id in ids
    assert untapped_id not in ids


def test_status_bits_emit_untapped(cache, tokenizer, name_to_row) -> None:
    forest_row = name_to_row["Forest"]
    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens(_bos_state_ids(tokenizer))
    w.emit_open_zone(ZONE_BATTLEFIELD, OWNER_SELF)
    w.emit_place_card(
        slot_idx=0, card_row_id=forest_row, status_bits=STATUS_TAPPED_KNOWN, uuid_idx=0
    )
    w.emit_end_card()
    w.emit_close_zone()
    w.emit_literal_tokens(_state_eos_ids(tokenizer))
    w.emit_close_state()

    batch = assemble_batch([w.finalize()], cache, tokenizer, max_tokens=512)
    untapped_id = tokenizer.convert_tokens_to_ids("<untapped>")
    ids = batch.token_ids[0, : int(batch.seq_lengths[0])].tolist()
    assert untapped_id in ids


# ---------------------------------------------------------------------------
# Overflow
# ---------------------------------------------------------------------------


def test_overflow_raises(cache, tokenizer, name_to_row) -> None:
    bolt_row = name_to_row["Lightning Bolt"]
    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens(_bos_state_ids(tokenizer))
    w.emit_place_card(0, bolt_row, 0, 0)
    w.emit_end_card()
    w.emit_close_state()

    with pytest.raises(ValueError, match="exceeds max_tokens"):
        assemble_batch([w.finalize()], cache, tokenizer, max_tokens=4)


def test_overflow_truncate_path(cache, tokenizer, name_to_row) -> None:
    bolt_row = name_to_row["Lightning Bolt"]
    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens(_bos_state_ids(tokenizer))
    w.emit_place_card(0, bolt_row, 0, 0)
    w.emit_end_card()
    w.emit_close_state()

    batch = assemble_batch([w.finalize()], cache, tokenizer, max_tokens=4, on_overflow="truncate")
    assert int(batch.seq_lengths[0]) == 4
    assert batch.token_ids.shape == (1, 4)


# ---------------------------------------------------------------------------
# Sanity: unknown opcode rejected
# ---------------------------------------------------------------------------


def test_unknown_opcode_raises(cache, tokenizer) -> None:
    plan = np.asarray([999], dtype=np.int32)
    with pytest.raises(ValueError, match="unknown opcode"):
        assemble_batch([plan], cache, tokenizer, max_tokens=8)
