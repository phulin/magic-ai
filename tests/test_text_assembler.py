"""Smaller, hand-written tests for ``assemble_batch``.

PR 13-C from ``docs/text_encoder_plan.md`` §13. These tests exercise the
opcode-decoder path with no coupling to ``emit_render_plan``: each test
hand-builds a plan via ``RenderPlanWriter`` and asserts the assembled
token stream contains the expected ids.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from magic_ai.text_encoder.assembler import assemble_batch
from magic_ai.text_encoder.card_cache import build_card_cache
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    load_oracle_text,
)
from magic_ai.text_encoder.render_plan import (
    OP_CLOSE_STATE,
    OP_CLOSE_ZONE,
    OP_LIFE,
    OP_MANA,
    OP_OPEN_STATE,
    OP_OPEN_ZONE,
    OP_PLACE_CARD,
    OP_TURN,
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
    # Card names are anonymized inside the body to ``<card-name>``; the
    # mana cost / type / anonymized rules text are what surface instead.
    assert "<mana-cost>{R}</mana-cost>" in decoded
    assert "<rules-text>" in decoded

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


def test_structured_go_plan_decodes_without_literal_tokens(cache, tokenizer, name_to_row) -> None:
    bears_row = name_to_row["Grizzly Bears"]
    plan = torch.tensor(
        [
            OP_OPEN_STATE,
            OP_TURN,
            3,
            3,  # Precombat Main
            OP_LIFE,
            0,
            20,
            OP_MANA,
            0,
            4,  # G
            2,
            OP_OPEN_ZONE,
            ZONE_BATTLEFIELD,
            OWNER_SELF,
            OP_PLACE_CARD,
            0,
            bears_row,
            STATUS_TAPPED,
            0,
            OP_CLOSE_ZONE,
            OP_CLOSE_STATE,
        ],
        dtype=torch.int32,
    )

    batch = assemble_batch([plan], cache, tokenizer, max_tokens=512)
    decoded = tokenizer.decode(batch.token_ids[0, : int(batch.seq_lengths[0])].tolist())

    assert "<turn>3</turn>" in decoded
    assert "<life>20</life>" in decoded
    # Floating mana surfaces as two green glyphs in the pool block.
    assert "<mana-pool>{G}{G}</mana-pool>" in decoded
    # Card names are anonymized in the body — assert the structural surface
    # instead. Grizzly Bears is a vanilla 2/2 with no oracle text.
    assert "<pt>2/2</pt>" in decoded
    assert "<actions>" not in decoded
    assert "<cast>" not in decoded
    assert int(batch.card_ref_positions[0, 0]) >= 0


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
    plan = torch.tensor([999], dtype=torch.int32)
    with pytest.raises(ValueError, match="unknown opcode"):
        assemble_batch([plan], cache, tokenizer, max_tokens=8)


# ---------------------------------------------------------------------------
# Inline-blank opcodes (Step 3 of inline-blanks plan).
# ---------------------------------------------------------------------------


def test_emit_blank_round_trips_through_assembler(cache, tokenizer) -> None:
    """OP_EMIT_BLANK / OP_EMIT_BLANK_LEGAL populate TextEncodedBatch fields."""
    from magic_ai.text_encoder.render_plan import (
        BLANK_GROUP_CROSS_BLANK,
        OP_EMIT_BLANK,
        OP_EMIT_BLANK_LEGAL,
        OPCODE_ARITY,
    )

    choose_play = int(tokenizer.convert_tokens_to_ids("<choose-play>"))
    pass_blank = int(tokenizer.convert_tokens_to_ids("<pass>"))
    chosen = int(tokenizer.convert_tokens_to_ids("<chosen>"))

    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens(_bos_state_ids(tokenizer))
    w.emit_blank(
        kind_id=choose_play,
        group_id=0,
        group_kind="CROSS_BLANK",
        legal_token_ids=(chosen,),
    )
    w.emit_blank(
        kind_id=pass_blank,
        group_id=0,
        group_kind="CROSS_BLANK",
        legal_token_ids=(chosen,),
    )
    w.emit_literal_tokens(_state_eos_ids(tokenizer))
    w.emit_close_state()
    plan = w.finalize()

    assert OPCODE_ARITY[OP_EMIT_BLANK] == 4
    assert OPCODE_ARITY[OP_EMIT_BLANK_LEGAL] == 1

    batch = assemble_batch([plan], cache, tokenizer, max_tokens=128)

    assert batch.blank_positions.shape == (1, 2)
    assert batch.blank_kind.tolist() == [[choose_play, pass_blank]]
    assert batch.blank_group.tolist() == [[0, 0]]
    assert batch.blank_group_kind.tolist() == [[BLANK_GROUP_CROSS_BLANK, BLANK_GROUP_CROSS_BLANK]]
    assert batch.blank_legal_ids.shape == (1, 2, 1)
    assert batch.blank_legal_ids.tolist() == [[[chosen], [chosen]]]
    assert batch.blank_legal_mask.tolist() == [[[True], [True]]]
    pos0 = int(batch.blank_positions[0, 0])
    pos1 = int(batch.blank_positions[0, 1])
    assert int(batch.token_ids[0, pos0]) == choose_play
    assert int(batch.token_ids[0, pos1]) == pass_blank


def test_blank_legal_padding_across_examples(cache, tokenizer) -> None:
    """Blank legal-id width pads to the per-batch V_max with mask bits."""
    from magic_ai.text_encoder.render_plan import BLANK_GROUP_PER_BLANK

    choose_target = int(tokenizer.convert_tokens_to_ids("<choose-target>"))
    none_id = int(tokenizer.convert_tokens_to_ids("<none>"))
    cref0 = int(tokenizer.convert_tokens_to_ids("<card-ref:0>"))
    cref1 = int(tokenizer.convert_tokens_to_ids("<card-ref:1>"))

    def _plan(legal_ids: tuple[int, ...]) -> torch.Tensor:
        w = RenderPlanWriter()
        w.emit_open_state()
        w.emit_literal_tokens(_bos_state_ids(tokenizer))
        w.emit_blank(
            kind_id=choose_target,
            group_id=7,
            group_kind=BLANK_GROUP_PER_BLANK,
            legal_token_ids=legal_ids,
        )
        w.emit_literal_tokens(_state_eos_ids(tokenizer))
        w.emit_close_state()
        return w.finalize()

    plans = [_plan((cref0, none_id)), _plan((cref0, cref1, none_id))]
    batch = assemble_batch(plans, cache, tokenizer, max_tokens=128)

    assert batch.blank_legal_ids.shape == (2, 1, 3)
    assert batch.blank_legal_ids[0, 0].tolist() == [cref0, none_id, 0]
    assert batch.blank_legal_mask[0, 0].tolist() == [True, True, False]
    assert batch.blank_legal_ids[1, 0].tolist() == [cref0, cref1, none_id]
    assert batch.blank_legal_mask[1, 0].tolist() == [True, True, True]


def test_emit_blank_legal_outside_blank_raises(cache, tokenizer) -> None:
    """Stray OP_EMIT_BLANK_LEGAL with no preceding header is a wire-format error."""
    from magic_ai.text_encoder.render_plan import OP_EMIT_BLANK_LEGAL

    plan = torch.tensor([OP_EMIT_BLANK_LEGAL, 1234], dtype=torch.int32)
    with pytest.raises(ValueError, match="OP_EMIT_BLANK_LEGAL"):
        assemble_batch([plan], cache, tokenizer, max_tokens=8)


def test_blank_singletons_round_trip_via_packed(tokenizer) -> None:
    """Inline-blank singletons + num_ids surface through the held-alive _Packed.

    The currently-shipped libmage cffi ``MageTokenTables`` struct does not yet
    include the new fields (the wheel needs a rebuild with mage-go's
    ``cmd/pylib/abi.h`` and matching ``_CDEF`` updated; see Step 3 mage-go
    wiring TODO). The Python-side mirror still holds the data so this is the
    parity gate available today.
    """
    from magic_ai.text_encoder.native_token_tables import (
        active_packed,
        register_native_token_tables,
    )
    from magic_ai.text_encoder.token_tables import build_token_tables

    tables = build_token_tables(tokenizer, cache=None)
    register_native_token_tables(tables)
    packed = active_packed()
    assert packed is not None

    for attr in (
        "choose_target_id",
        "choose_block_id",
        "choose_damage_order_id",
        "choose_mode_id",
        "choose_may_id",
        "choose_x_digit_id",
        "choose_mana_source_id",
        "choose_play_id",
        "use_ability_id",
        "chosen_id",
        "yes_id",
        "no_id",
        "none_id",
        "x_end_id",
    ):
        assert getattr(packed, attr) == getattr(tables, attr), attr
    assert packed.num_ids.tolist() == list(tables.num_ids)


def test_native_assembler_blank_output_buffers_allocate() -> None:
    """Native packed outputs include the Step-3 blank tensor buffers."""
    from magic_ai.text_encoder.native_assembler import allocate_packed_outputs

    outputs = allocate_packed_outputs(
        2,
        max_tokens=16,
        max_options=3,
        max_targets=2,
        max_card_refs=8,
        max_blanks=5,
        max_legal_per_blank=7,
    )

    assert outputs.blank_positions.shape == (2, 5)
    assert outputs.blank_kind.shape == (2, 5)
    assert outputs.blank_group.shape == (2, 5)
    assert outputs.blank_group_kind.shape == (2, 5)
    assert outputs.blank_legal_ids.shape == (2, 5, 7)
    assert outputs.blank_legal_mask.shape == (2, 5, 7)
    assert outputs.blank_overflow.shape == (2,)
