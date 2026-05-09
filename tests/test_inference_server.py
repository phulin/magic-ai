"""Tests for magic_ai.native.inference_server.

These tests exercise the dynamic-batching inference server against a fake
``sample_native_tensor_batch`` callable; we verify that:

* requests from concurrent submitters are coalesced into one forward call
  when enough rows are queued,
* per-request host-side scalars are scattered back to the right futures,
* each reply receives its sliced ``replay_payload`` without committed replay rows,
* ``pause()`` blocks new forwards and ``resume()`` re-enables them,
* PackedTextBatch concatenation rebases token / row / anchor offsets
  correctly (vectorized — no per-row loops).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import torch
from magic_ai.native.inference_server import (
    _concat_packed_text_batches,
)
from magic_ai.text_encoder.batch import PackedTextBatch


def _make_packed_batch(
    rows: list[int], *, anchor_token_offset: int = 0, with_blank: bool = False
) -> PackedTextBatch:
    """Tiny PackedTextBatch with one option per row, anchor at row's first token."""

    del with_blank  # legacy inline-blank flag, ignored after the cutover
    rows_t = torch.tensor(rows, dtype=torch.int32)
    cu = torch.zeros(len(rows) + 1, dtype=torch.int32)
    cu[1:] = rows_t.cumsum(0)
    total = int(cu[-1].item())
    state_pos = cu[:-1].clone()
    seq_id = torch.repeat_interleave(torch.arange(len(rows), dtype=torch.int32), rows_t)
    pos_in_seq = torch.arange(total, dtype=torch.int32) - state_pos.repeat_interleave(rows_t)
    token_ids = torch.arange(total, dtype=torch.int32) + 1
    card_ref_pos = state_pos.unsqueeze(1).clone() + anchor_token_offset
    card_ref_pos = torch.where(card_ref_pos >= 0, card_ref_pos, torch.full_like(card_ref_pos, -1))
    n = len(rows)
    pointer_anchor_positions = torch.full((n, 1), -1, dtype=torch.int32)
    pointer_anchor_kinds = torch.full((n, 1), -1, dtype=torch.int32)
    pointer_anchor_subjects = torch.full((n, 1), -1, dtype=torch.int32)
    pointer_anchor_handles = torch.full((n, 1), -1, dtype=torch.int32)
    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu,
        seq_lengths=rows_t,
        state_positions=state_pos,
        card_ref_positions=card_ref_pos,
        seq_lengths_host=tuple(rows),
        max_seqlen=max(rows, default=0),
        spec_lens=torch.zeros(n, dtype=torch.int32),
        decision_type=torch.full((n,), -1, dtype=torch.int32),
        pointer_anchor_positions=pointer_anchor_positions,
        pointer_anchor_kinds=pointer_anchor_kinds,
        pointer_anchor_subjects=pointer_anchor_subjects,
        pointer_anchor_handles=pointer_anchor_handles,
        legal_edge_bitmap=None,
    )


class ConcatPackedBatchTest(unittest.TestCase):
    def test_offsets_and_anchors_rebased(self) -> None:
        a = _make_packed_batch([3, 2], anchor_token_offset=0)  # token 0,3
        b = _make_packed_batch([4], anchor_token_offset=1)  # row anchor at token 1
        merged = _concat_packed_text_batches([a, b])
        # cu_seqlens stitched: [0,3,5,9]
        torch.testing.assert_close(
            merged.cu_seqlens,
            torch.tensor([0, 3, 5, 9], dtype=torch.int32),
        )
        # state_positions of b's row should land at 5 (a had 5 tokens).
        torch.testing.assert_close(
            merged.state_positions,
            torch.tensor([0, 3, 5], dtype=torch.int32),
        )
        # b's card-ref position (row-local 1, state 5 -> global 6).
        torch.testing.assert_close(
            merged.card_ref_positions[2],
            torch.tensor([6], dtype=torch.int32),
        )
        # seq_id contiguous and shifted (a: rows 0,1; b: row 2)
        self.assertEqual(int(merged.seq_id[-1].item()), 2)
        # tokens are concatenated in order
        self.assertEqual(int(merged.token_ids.shape[0]), 9)

    def test_rejects_missing_token_metadata(self) -> None:
        batch = _make_packed_batch([3, 2])
        batch.seq_id = batch.seq_id[:0]
        batch.pos_in_seq = batch.pos_in_seq[:0]
        with self.assertRaisesRegex(ValueError, "token metadata length"):
            _concat_packed_text_batches([batch])


@dataclass
class _FakeNativeBatch:
    trace_kind_id: torch.Tensor
    decision_count: torch.Tensor
    decision_rows_written: int
    decision_start: torch.Tensor
    decision_option_idx: torch.Tensor
    decision_target_idx: torch.Tensor
    decision_mask: torch.Tensor
    uses_none_head: torch.Tensor
    # Unused-but-required-for-_concat fields, all zero-shaped.
    slot_card_rows: torch.Tensor
    slot_occupied: torch.Tensor
    slot_tapped: torch.Tensor
    game_info: torch.Tensor
    pending_kind_id: torch.Tensor
    num_present_options: torch.Tensor
    option_kind_ids: torch.Tensor
    option_scalars: torch.Tensor
    option_mask: torch.Tensor
    option_ref_slot_idx: torch.Tensor
    option_ref_card_row: torch.Tensor
    target_mask: torch.Tensor
    target_type_ids: torch.Tensor
    target_scalars: torch.Tensor
    target_overflow: torch.Tensor
    target_ref_slot_idx: torch.Tensor
    target_ref_is_player: torch.Tensor
    target_ref_is_self: torch.Tensor
    may_mask: torch.Tensor
    pendings: list[Any]
    trace_kinds: list[Any]
    render_plan: torch.Tensor | None = None
    render_plan_lengths: torch.Tensor | None = None
    render_plan_overflow: torch.Tensor | None = None


def _make_native(batch_size: int, *, trace_offset: int = 0) -> _FakeNativeBatch:
    decision_count = torch.ones(batch_size, dtype=torch.int64)
    decision_start = torch.arange(batch_size, dtype=torch.int64)
    rows = batch_size
    z = lambda *s, dtype=torch.int64: torch.zeros(s, dtype=dtype)  # noqa: E731
    return _FakeNativeBatch(
        trace_kind_id=torch.full((batch_size,), trace_offset, dtype=torch.int64),
        decision_count=decision_count,
        decision_rows_written=rows,
        decision_start=decision_start,
        decision_option_idx=z(rows, 2),
        decision_target_idx=z(rows, 2),
        decision_mask=torch.ones(rows, 2, dtype=torch.bool),
        uses_none_head=z(rows, dtype=torch.bool),
        slot_card_rows=z(batch_size, 1),
        slot_occupied=z(batch_size, 1, dtype=torch.float32),
        slot_tapped=z(batch_size, 1, dtype=torch.float32),
        game_info=z(batch_size, 1, dtype=torch.float32),
        pending_kind_id=z(batch_size),
        num_present_options=z(batch_size),
        option_kind_ids=z(batch_size, 1),
        option_scalars=z(batch_size, 1, 1, dtype=torch.float32),
        option_mask=z(batch_size, 1, dtype=torch.bool),
        option_ref_slot_idx=z(batch_size, 1),
        option_ref_card_row=z(batch_size, 1),
        target_mask=z(batch_size, 1, 1, dtype=torch.bool),
        target_type_ids=z(batch_size, 1, 1),
        target_scalars=z(batch_size, 1, 1, 1, dtype=torch.float32),
        target_overflow=z(batch_size),
        target_ref_slot_idx=z(batch_size, 1, 1),
        target_ref_is_player=z(batch_size, 1, 1, dtype=torch.bool),
        target_ref_is_self=z(batch_size, 1, 1, dtype=torch.bool),
        may_mask=z(batch_size, dtype=torch.bool),
        pendings=[None] * batch_size,
        trace_kinds=[None] * batch_size,
    )


def _release_reply(reply: Any) -> None:
    release = getattr(reply, "release_item", None)
    if release is not None:
        release()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
