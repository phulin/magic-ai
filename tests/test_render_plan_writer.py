"""Round-trip test for the int32 render-plan wire format.

PR 13-C from ``docs/text_encoder_plan.md`` §13. Independent of the
assembler — exercises that ``RenderPlanWriter`` emits a self-describing
stream that can be walked using only ``OPCODE_ARITY`` per the ABI doc.
"""

from __future__ import annotations

import torch
from magic_ai.text_encoder.render_plan import (
    OP_CLOSE_ACTIONS,
    OP_CLOSE_STATE,
    OP_CLOSE_ZONE,
    OP_LITERAL_TOKENS,
    OP_OPEN_ACTIONS,
    OP_OPEN_STATE,
    OP_OPEN_ZONE,
    OP_OPTION,
    OP_PLACE_CARD,
    OP_TARGET,
    OPCODE_ARITY,
    OWNER_OPP,
    OWNER_SELF,
    STATUS_TAPPED,
    STATUS_TAPPED_KNOWN,
    ZONE_BATTLEFIELD,
    RenderPlanWriter,
)


def test_writer_round_trip() -> None:
    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_literal_tokens([10, 11, 12])
    w.emit_open_zone(ZONE_BATTLEFIELD, OWNER_SELF)
    w.emit_place_card(0, 7, STATUS_TAPPED | STATUS_TAPPED_KNOWN, 0)
    w.emit_end_card()
    w.emit_close_zone()
    w.emit_open_actions()
    w.emit_option(0, 0, 0, -1, -1)
    w.emit_target(7, 0, 1)
    w.emit_close_actions()
    w.emit_close_state()

    buf = w.finalize()
    assert buf.dtype == torch.int32

    # Walk + decode by header alone.
    seen: list[tuple[int, list[int]]] = []
    i = 0
    while i < buf.shape[0]:
        op = int(buf[i])
        arity = OPCODE_ARITY[op]
        if arity == -1:
            length = int(buf[i + 1])
            payload = [int(buf[i + 2 + k]) for k in range(length)]
            seen.append((op, [length] + payload))
            i += 2 + length
        else:
            payload = [int(buf[i + 1 + k]) for k in range(arity)]
            seen.append((op, payload))
            i += 1 + arity
    assert i == buf.shape[0]

    # Order + payload values match what we wrote.
    expected: list[tuple[int, list[int]]] = [
        (OP_OPEN_STATE, []),
        (OP_LITERAL_TOKENS, [3, 10, 11, 12]),
        (OP_OPEN_ZONE, [ZONE_BATTLEFIELD, OWNER_SELF]),
        (OP_PLACE_CARD, [0, 7, STATUS_TAPPED | STATUS_TAPPED_KNOWN, 0]),
        # OP_END_CARD has arity 0.
        (18, []),  # OP_END_CARD = 18
        (OP_CLOSE_ZONE, []),
        (OP_OPEN_ACTIONS, []),
        (OP_OPTION, [0, 0, 0, -1, -1]),
        (OP_TARGET, [7, 0, 1]),
        (OP_CLOSE_ACTIONS, []),
        (OP_CLOSE_STATE, []),
    ]
    assert seen == expected


def test_owner_opp_constant() -> None:
    # Sanity: owner ids are 0/1 per ABI §7.
    assert OWNER_SELF == 0
    assert OWNER_OPP == 1


def test_arity_table_covers_every_emit_method() -> None:
    # Every emit_* method on the writer corresponds to an opcode with a
    # registered arity. We probe by spinning the writer and walking the
    # buffer (same as round-trip).
    w = RenderPlanWriter()
    w.emit_open_state()
    w.emit_close_state()
    w.emit_open_player(0)
    w.emit_close_player()
    w.emit_open_zone(0, 0)
    w.emit_close_zone()
    w.emit_open_actions()
    w.emit_close_actions()
    w.emit_place_card(0, 0, 0, -1)
    w.emit_end_card()
    w.emit_counter(0, 0)
    w.emit_attached_to(-1)
    w.emit_option(0, 0, -1, -1, -1)
    w.emit_target(0, -1, 0)
    w.emit_open_raw_card(-1)
    w.emit_close_raw_card()
    w.emit_literal_tokens([1, 2, 3])
    buf = w.finalize()

    i = 0
    n_ops = 0
    while i < buf.shape[0]:
        op = int(buf[i])
        arity = OPCODE_ARITY[op]
        if arity == -1:
            length = int(buf[i + 1])
            i += 2 + length
        else:
            i += 1 + arity
        n_ops += 1
    assert n_ops == 17
    assert i == buf.shape[0]
