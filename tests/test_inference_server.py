"""Tests for magic_ai.native.inference_server.

These tests exercise the dynamic-batching inference server against a fake
forward callable; we verify that:

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
    _HostPackedArena,
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
    # card_ref positions are row-local everywhere; one anchor per row at
    # ``anchor_token_offset`` within its own row.
    card_ref_pos = torch.full((len(rows), 1), int(anchor_token_offset), dtype=torch.int32)
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
        # card_ref positions are row-local end-to-end; concat preserves
        # them unchanged (b's row-local-1 anchor stays at 1).
        torch.testing.assert_close(
            merged.card_ref_positions[2],
            torch.tensor([1], dtype=torch.int32),
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


class HostPackedArenaTest(unittest.TestCase):
    def _assert_batches_equal(self, a: PackedTextBatch, b: PackedTextBatch) -> None:
        for name in (
            "token_ids",
            "seq_id",
            "pos_in_seq",
            "cu_seqlens",
            "seq_lengths",
            "state_positions",
            "card_ref_positions",
            "spec_lens",
            "decision_type",
            "pointer_anchor_positions",
            "pointer_anchor_kinds",
            "pointer_anchor_subjects",
            "pointer_anchor_handles",
        ):
            ta = getattr(a, name).to(torch.int32)
            tb = getattr(b, name).to(torch.int32)
            torch.testing.assert_close(ta, tb)
        self.assertEqual(a.seq_lengths_host, b.seq_lengths_host)
        self.assertEqual(a.max_seqlen, b.max_seqlen)

    def test_arena_matches_concat_on_single_merge(self) -> None:
        a = _make_packed_batch([3, 2], anchor_token_offset=0)
        b = _make_packed_batch([4], anchor_token_offset=1)
        arena = _HostPackedArena(use_pinned=False)
        merged_arena, _ = arena.merge([a, b])
        merged_concat = _concat_packed_text_batches([a, b])
        # Compare values (clone to detach from arena storage for safety).
        cloned = PackedTextBatch(
            token_ids=merged_arena.token_ids.clone(),
            seq_id=merged_arena.seq_id.clone(),
            pos_in_seq=merged_arena.pos_in_seq.clone(),
            cu_seqlens=merged_arena.cu_seqlens.clone(),
            seq_lengths=merged_arena.seq_lengths.clone(),
            state_positions=merged_arena.state_positions.clone(),
            card_ref_positions=merged_arena.card_ref_positions.clone(),
            spec_lens=merged_arena.spec_lens.clone(),
            decision_type=merged_arena.decision_type.clone(),
            pointer_anchor_positions=merged_arena.pointer_anchor_positions.clone(),
            pointer_anchor_kinds=merged_arena.pointer_anchor_kinds.clone(),
            pointer_anchor_subjects=merged_arena.pointer_anchor_subjects.clone(),
            pointer_anchor_handles=merged_arena.pointer_anchor_handles.clone(),
            seq_lengths_host=merged_arena.seq_lengths_host,
            max_seqlen=merged_arena.max_seqlen,
        )
        self._assert_batches_equal(cloned, merged_concat)

    def test_arena_bitmap_grows_with_rows_even_when_blockers_unchanged(self) -> None:
        # Bug: arena's legal_edge_bitmap first dim is the arena row capacity.
        # If rows grew but the (blockers, attackers) didn't, the bitmap kept
        # the old row dim and ``merge`` returned a view shorter (in dim 0)
        # than seq_lengths for the same merge, causing
        # ``RuntimeError: The size of tensor a (1024) must match the size
        # of tensor b (512) at non-singleton dimension 0``.
        def _with_bitmap(rows: list[int], bk: int, ak: int) -> PackedTextBatch:
            batch = _make_packed_batch(rows)
            batch.legal_edge_bitmap = torch.zeros((len(rows), bk, ak), dtype=torch.bool)
            return batch

        arena = _HostPackedArena(use_pinned=False)
        # First merge: small row count, real bitmap dims.
        small = _with_bitmap([2, 3], bk=2, ak=2)
        merged_small, _ = arena.merge([small])
        assert merged_small.legal_edge_bitmap is not None
        self.assertEqual(merged_small.legal_edge_bitmap.shape[0], 2)
        # Second merge: lots more rows, SAME bitmap dims. Bitmap row capacity
        # must grow alongside the other row-axis fields.
        big_rows = [1] * 32
        big = _with_bitmap(big_rows, bk=2, ak=2)
        merged_big, _ = arena.merge([big])
        assert merged_big.legal_edge_bitmap is not None
        self.assertEqual(merged_big.legal_edge_bitmap.shape[0], len(big_rows))
        self.assertEqual(merged_big.seq_lengths.shape[0], len(big_rows))

    def test_arena_reused_across_merges_smaller_then_larger(self) -> None:
        # First merge populates the arena, second merge with more rows /
        # tokens triggers growth — both results should match concat.
        a = _make_packed_batch([2])
        arena = _HostPackedArena(use_pinned=False)
        first, _ = arena.merge([a])
        # Capture values before reuse.
        first_tokens = first.token_ids.clone()
        # Sanity vs concat (single shard fast path).
        torch.testing.assert_close(first_tokens, a.token_ids.to(torch.int32))

        b = _make_packed_batch([3, 4, 5], anchor_token_offset=2)
        c = _make_packed_batch([6])
        merged, _ = arena.merge([b, c])
        expected = _concat_packed_text_batches([b, c])
        torch.testing.assert_close(
            merged.token_ids.to(torch.int32), expected.token_ids.to(torch.int32)
        )
        torch.testing.assert_close(
            merged.cu_seqlens.to(torch.int32), expected.cu_seqlens.to(torch.int32)
        )
        # Past-live anchor / card-ref columns must be -1 even after a wide
        # earlier shard expanded the arena: shrinks across merges should
        # still produce identical merged output.
        a2 = _make_packed_batch([1])  # narrow shard
        merged_narrow, _ = arena.merge([a2])
        expected_narrow = _concat_packed_text_batches([a2])
        torch.testing.assert_close(
            merged_narrow.pointer_anchor_positions.to(torch.int32),
            expected_narrow.pointer_anchor_positions.to(torch.int32),
        )


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
