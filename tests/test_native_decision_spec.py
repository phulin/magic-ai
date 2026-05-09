"""Smoke tests for the native decision-spec FFI bindings.

These tests verify that the cgo entry points are reachable from Python
and respond predictably to bad input. End-to-end parity tests against
``render_decision_spec`` and ``grammar.next_mask`` will land in a follow-up
once the magic-ai-side wiring (a Python helper that builds the spec-token
ids from the magic-ai tokenizer and calls ``register_decision_spec_tokens``)
is in place; until then the Go-side tests in ``cmd/pylib/`` cover emit /
mask parity using a manufactured token-id table.
"""

from __future__ import annotations

import mage
import numpy as np


def test_decision_mask_next_unknown_handle_returns_error():
    mage.load()
    # Allocate dummy buffers; the call should fail fast before reading them
    # because the handle has never been registered.
    prefix_tokens = np.zeros(8, dtype=np.int32)
    prefix_pointers = np.zeros(8, dtype=np.int32)
    prefix_lens = np.zeros(1, dtype=np.int32)
    out_vocab = np.zeros(26, dtype=np.uint8)
    out_pointer = np.zeros(8, dtype=np.uint8)
    pt = mage._ffi.cast("int32_t *", prefix_tokens.ctypes.data)
    pp = mage._ffi.cast("int32_t *", prefix_pointers.ctypes.data)
    pl = mage._ffi.cast("int32_t *", prefix_lens.ctypes.data)
    ov = mage._ffi.cast("uint8_t *", out_vocab.ctypes.data)
    op = mage._ffi.cast("uint8_t *", out_pointer.ctypes.data)
    rc = mage.decision_mask_next(
        batch_handle=999_999_999,
        prefix_tokens=pt,
        prefix_pointers=pp,
        prefix_lens=pl,
        batch_size=1,
        prefix_len_max=8,
        grammar_vocab_size=26,
        n_anchors_max=8,
        out_vocab_mask=ov,
        out_pointer_mask=op,
    )
    assert rc != 0


def test_release_batch_handle_unknown_is_noop():
    mage.load()
    # No exception expected.
    mage.release_batch_handle(0)
    mage.release_batch_handle(123_456_789)


def test_register_decision_spec_tokens_roundtrip():
    mage.load()
    keepalive = mage.register_decision_spec_tokens(
        spec_open_id=1,
        spec_close_id=2,
        decision_type_id=3,
        legal_attacker_id=4,
        legal_blocker_id=5,
        legal_target_id=6,
        legal_action_id=7,
        for_action_id=8,
        max_value_open_id=9,
        max_value_close_id=10,
        player_ref0_id=11,
        player_ref1_id=12,
        dt_name_ids=list(range(100, 107)),
        stack_ref_ids=list(range(200, 216)),
        max_value_digit_max=10,
        max_value_digits=[300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310],
        max_value_digit_offsets=list(range(12)),
    )
    # Hold reference so cffi-allocated buffers stay alive.
    assert keepalive is not None
