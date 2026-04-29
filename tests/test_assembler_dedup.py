"""Tests for the v2 (card-body-dedup) render-plan opcode set.

Exercises ``OP_OPEN_DICT`` / ``OP_DICT_ENTRY`` / ``OP_CLOSE_DICT`` and
``OP_PLACE_CARD_REF`` against a small hand-built plan, asserting that:

* The dict block emits each unique card body exactly once.
* Each per-zone occurrence emits ``<card-ref:K> <card> <dict-entry:R> </card>``
  (~5 tokens) instead of splicing the ~13-25 token body.
* Per-row token count of the v2 plan is materially smaller than the v1 plan
  encoding the same logical state.
* Status flags still surface inside per-occurrence references.
"""

from __future__ import annotations

from pathlib import Path

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
    ZONE_GRAVEYARD,
    RenderPlanWriter,
)
from magic_ai.text_encoder.tokenizer import (
    MAX_DICT_ENTRIES,
    dict_entry_token,
    load_tokenizer,
)


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


def _bos_state_ids(tokenizer) -> list[int]:
    return list(tokenizer.encode("<bos><state>", add_special_tokens=False))


def _state_eos_ids(tokenizer) -> list[int]:
    return list(tokenizer.encode("</state><eos>", add_special_tokens=False))


def _decoded_ids(batch) -> list[int]:
    return batch.token_ids[0, : int(batch.seq_lengths[0])].tolist()


def test_dict_entry_emits_body_once(cache, tokenizer, name_to_row) -> None:
    forest_row = name_to_row["Forest"]

    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens(_bos_state_ids(tokenizer))
    w.emit_open_dict()
    w.emit_dict_entry(forest_row)
    w.emit_close_dict()
    w.emit_literal_tokens(_state_eos_ids(tokenizer))
    w.emit_close_state()

    batch = assemble_batch([w.finalize()], cache, tokenizer, max_tokens=512)
    decoded = tokenizer.decode(_decoded_ids(batch))
    # Use a body-internal fragment that doesn't appear elsewhere — the type
    # line "Basic Land" is unique to the splice. (The card name "Forest"
    # appears twice in Forest's body: once as the name and once inside the
    # type line "Basic Land — Forest", so it isn't a clean dedup signal.)
    assert decoded.count("Basic Land") == 1, decoded
    dict_entry_id = tokenizer.convert_tokens_to_ids(dict_entry_token(forest_row))
    assert dict_entry_id in _decoded_ids(batch)


def test_place_card_ref_skips_body(cache, tokenizer, name_to_row) -> None:
    forest_row = name_to_row["Forest"]

    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens(_bos_state_ids(tokenizer))
    w.emit_open_dict()
    w.emit_dict_entry(forest_row)
    w.emit_close_dict()
    w.emit_open_zone(ZONE_BATTLEFIELD, OWNER_SELF)
    # Three Forests on the battlefield, distinct uuids.
    w.emit_place_card_ref(
        slot_idx=0,
        card_row_id=forest_row,
        status_bits=STATUS_TAPPED | STATUS_TAPPED_KNOWN,
        uuid_idx=0,
    )
    w.emit_place_card_ref(
        slot_idx=1, card_row_id=forest_row, status_bits=STATUS_TAPPED_KNOWN, uuid_idx=1
    )
    w.emit_place_card_ref(
        slot_idx=2, card_row_id=forest_row, status_bits=STATUS_TAPPED_KNOWN, uuid_idx=2
    )
    w.emit_close_zone()
    w.emit_literal_tokens(_state_eos_ids(tokenizer))
    w.emit_close_state()

    batch = assemble_batch([w.finalize()], cache, tokenizer, max_tokens=512)
    ids = _decoded_ids(batch)
    decoded = tokenizer.decode(ids)

    # Body still appears exactly once (in the dict block); use a unique
    # body-internal fragment so duplicates inside the body itself don't
    # confound the count.
    assert decoded.count("Basic Land") == 1, decoded

    # Each occurrence emitted exactly one <dict-entry:R> token.
    dict_entry_id = tokenizer.convert_tokens_to_ids(dict_entry_token(forest_row))
    assert ids.count(dict_entry_id) == 4  # 1 dict + 3 refs

    # Three card-refs (uuids 0/1/2) anchored.
    assert int(batch.card_ref_positions[0, 0]) >= 0
    assert int(batch.card_ref_positions[0, 1]) >= 0
    assert int(batch.card_ref_positions[0, 2]) >= 0

    # Status flags surfaced in the per-occurrence ref form.
    tapped_id = tokenizer.convert_tokens_to_ids("<tapped>")
    untapped_id = tokenizer.convert_tokens_to_ids("<untapped>")
    assert ids.count(tapped_id) == 1  # only uuid 0 is tapped
    assert ids.count(untapped_id) == 2  # uuids 1 and 2 are untapped


def test_v2_is_smaller_than_v1_under_repetition(cache, tokenizer, name_to_row) -> None:
    forest_row = name_to_row["Forest"]
    bolt_row = name_to_row["Lightning Bolt"]

    # Construct a state that emulates a midgame zone-heavy snapshot:
    # 8 Forests on battlefield, 12 Lightning Bolts in graveyard.
    counts = [(forest_row, 8, ZONE_BATTLEFIELD), (bolt_row, 12, ZONE_GRAVEYARD)]

    # v1: full body splice per occurrence.
    w1 = RenderPlanWriter()
    w1.emit_open_state()
    w1.emit_literal_tokens(_bos_state_ids(tokenizer))
    uuid = 0
    for row, n, zone in counts:
        w1.emit_open_zone(zone, OWNER_SELF)
        for slot in range(n):
            w1.emit_place_card(slot_idx=slot, card_row_id=row, status_bits=0, uuid_idx=uuid)
            w1.emit_end_card()
            uuid += 1
        w1.emit_close_zone()
    w1.emit_literal_tokens(_state_eos_ids(tokenizer))
    w1.emit_close_state()

    # v2: dict at the top, refs in zones.
    w2 = RenderPlanWriter()
    w2.emit_open_state()
    w2.emit_literal_tokens(_bos_state_ids(tokenizer))
    w2.emit_open_dict()
    for row, _n, _zone in counts:
        w2.emit_dict_entry(row)
    w2.emit_close_dict()
    uuid = 0
    for row, n, zone in counts:
        w2.emit_open_zone(zone, OWNER_SELF)
        for slot in range(n):
            w2.emit_place_card_ref(slot_idx=slot, card_row_id=row, status_bits=0, uuid_idx=uuid)
            uuid += 1
        w2.emit_close_zone()
    w2.emit_literal_tokens(_state_eos_ids(tokenizer))
    w2.emit_close_state()

    b1 = assemble_batch([w1.finalize()], cache, tokenizer, max_tokens=4096)
    b2 = assemble_batch([w2.finalize()], cache, tokenizer, max_tokens=4096)

    len1 = int(b1.seq_lengths[0])
    len2 = int(b2.seq_lengths[0])
    # Two unique cards, 20 occurrences. v1 splices ~20 bodies; v2 splices 2
    # bodies + 20 short references. The win must be at least 3x.
    assert len2 < len1, (len1, len2)
    assert len1 / max(len2, 1) >= 3.0, f"v1={len1} v2={len2} ratio={len1 / len2:.2f}"


def test_dict_entry_token_namespace_disjoint_from_card_ref(tokenizer) -> None:
    # Tokenizer must treat dict-entry and card-ref namespaces as distinct ids.
    e0 = tokenizer.convert_tokens_to_ids(dict_entry_token(0))
    e_last = tokenizer.convert_tokens_to_ids(dict_entry_token(MAX_DICT_ENTRIES - 1))
    c0 = tokenizer.convert_tokens_to_ids("<card-ref:0>")
    assert e0 != c0
    assert e0 != e_last
    assert e_last != c0
