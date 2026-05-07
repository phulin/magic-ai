"""Tests for magic_ai.native.inference_server.

These tests exercise the dynamic-batching inference server against a fake
``sample_native_tensor_batch`` callable plus a fake staging buffer; we verify
that:

* requests from concurrent submitters are coalesced into one forward call
  when they arrive within ``max_wait_ms``,
* per-request host-side scalars are scattered back to the right futures,
* the merged ``replay_payload`` is staged exactly once,
* ``pause()`` blocks new forwards and ``resume()`` re-enables them,
* PackedTextBatch concatenation rebases token / row / anchor offsets
  correctly (vectorized — no per-row loops).
"""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass
from typing import Any

import torch
from magic_ai.native.inference_server import (
    TextInferenceRequest,
    TextInferenceServer,
    _concat_packed_text_batches,
)
from magic_ai.text_encoder.batch import PackedTextBatch


def _make_packed_batch(
    rows: list[int], *, anchor_token_offset: int = 0, with_blank: bool = False
) -> PackedTextBatch:
    """Tiny PackedTextBatch with one option per row, anchor at row's first token."""

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
    kwargs: dict[str, Any] = {}
    if with_blank:
        blank_positions = state_pos.unsqueeze(1).clone()
        kwargs = {
            "blank_positions": blank_positions,
            "blank_kind": torch.full((len(rows), 1), 11, dtype=torch.int32),
            "blank_group": torch.zeros((len(rows), 1), dtype=torch.int32),
            "blank_group_kind": torch.zeros((len(rows), 1), dtype=torch.int32),
            "blank_option_index": torch.zeros((len(rows), 1), dtype=torch.int32),
            "blank_legal_ids": torch.full((len(rows), 1, 2), 13, dtype=torch.int32),
            "blank_legal_mask": torch.ones((len(rows), 1, 2), dtype=torch.bool),
        }
    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu,
        seq_lengths=rows_t,
        state_positions=state_pos,
        card_ref_positions=card_ref_pos,
        **kwargs,
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


class _FakePolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.last_env_indices: list[int] = []
        self.last_blank_shape: tuple[int, ...] | None = None

    def sample_native_tensor_batch(self, **kwargs: Any) -> Any:
        self.calls += 1
        env_indices = kwargs["env_indices"]
        self.last_env_indices = list(env_indices)
        self.last_blank_shape = tuple(kwargs["packed_batch"].blank_positions.shape)
        b = len(env_indices)
        from magic_ai.text_encoder.actor_critic import (
            NativeTextReplayPayload,
            NativeTextSampleBatch,
        )

        device = torch.device("cpu")
        payload = NativeTextReplayPayload(
            encoded=kwargs["packed_batch"],
            trace_kind_id=torch.zeros(b, dtype=torch.int64),
            decision_count=torch.ones(b, dtype=torch.int64),
            decision_count_host=tuple(1 for _ in range(b)),
            total_decision_groups=b,
            total_stored_decision_groups=b,
            decision_option_idx=torch.zeros(b, 2, dtype=torch.int64),
            decision_target_idx=torch.zeros(b, 2, dtype=torch.int64),
            decision_mask=torch.ones(b, 2, dtype=torch.bool),
            uses_none_head=torch.zeros(b, dtype=torch.bool),
            selected_indices=torch.zeros(b, dtype=torch.int64),
            behavior_action_log_prob=torch.zeros(b, dtype=torch.float32),
            may_selected=torch.zeros(b, dtype=torch.float32),
            old_log_prob=torch.full((b,), -0.5, dtype=torch.float32),
            value=torch.linspace(0.0, 1.0, b, dtype=torch.float32, device=device),
            perspective_player_idx=torch.tensor(kwargs["perspective_player_indices"]),
            lstm_h_in=torch.zeros(1, b, 2),
            lstm_c_in=torch.zeros(1, b, 2),
        )
        return NativeTextSampleBatch(
            decision_counts=[1] * b,
            selected_choice_cols=[7] * b,
            may_selected=[0] * b,
            old_log_prob=[-0.5] * b,
            value=[0.0] * b,
            replay_rows=[-1] * b,
            replay_payload=payload,
        )


class _FakeStaging:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], int]] = []

    def stage_batch(self, env_indices: list[int], payload: Any) -> None:
        # Record env_indices and payload encoded-row count (sanity).
        self.calls.append((list(env_indices), int(payload.encoded.seq_lengths.shape[0])))


class InferenceWorkRingTest(unittest.TestCase):
    def test_submit_rejects_item_larger_than_ring_capacity(self) -> None:
        policy = _FakePolicy()
        server = TextInferenceServer(
            sampling_policy=policy,
            staging_buffer=_FakeStaging(),
            max_batch=8,
            max_wait_ms=0.0,
            ring_capacity_rows=2,
        )
        server.start()
        try:
            fut = server.submit(
                TextInferenceRequest(
                    native_batch=_make_native(3),
                    packed_batch=_make_packed_batch([1, 1, 1]),
                    env_indices=[0, 1, 2],
                    perspective_player_indices=[0, 0, 0],
                )
            )
            with self.assertRaisesRegex(RuntimeError, "exceeds ring capacity"):
                fut.result(timeout=1.0)
        finally:
            server.stop()


class InferenceServerBatchingTest(unittest.TestCase):
    def test_partial_batch_waits_for_explicit_flush(self) -> None:
        policy = _FakePolicy()
        server = TextInferenceServer(
            sampling_policy=policy,
            staging_buffer=_FakeStaging(),
            max_batch=8,
            max_wait_ms=0.0,
            min_batch_rows=2,
        )
        server.start()
        try:
            fut = server.submit(
                TextInferenceRequest(
                    native_batch=_make_native(1),
                    packed_batch=_make_packed_batch([2]),
                    env_indices=[0],
                    perspective_player_indices=[0],
                )
            )
            time.sleep(0.05)
            self.assertFalse(fut.done())
            server.flush()
            reply = fut.result(timeout=5.0)
        finally:
            server.stop()

        self.assertEqual(policy.calls, 1)
        self.assertEqual(reply.decision_counts, [1])

    def test_dynamic_batches_concurrent_submits(self) -> None:
        policy = _FakePolicy()
        staging = _FakeStaging()
        server = TextInferenceServer(
            sampling_policy=policy,
            staging_buffer=staging,
            max_batch=64,
            max_wait_ms=0.0,
            min_batch_rows=8,
        )
        # Pre-build requests so the submit loop is purely queue puts; otherwise
        # tensor construction between submits can outrun ``max_wait_ms`` on a
        # contended host and split the batch across two forwards.
        reqs = [
            TextInferenceRequest(
                native_batch=_make_native(2),
                packed_batch=_make_packed_batch([2, 3]),
                env_indices=[i * 2, i * 2 + 1],
                perspective_player_indices=[0, 1],
            )
            for i in range(4)
        ]
        server.start()
        try:
            futs = [server.submit(req) for req in reqs]
            replies = [f.result(timeout=5.0) for f in futs]
        finally:
            server.stop()

        # All four requests should be coalesced into one forward call.
        self.assertEqual(policy.calls, 1)
        self.assertEqual(policy.last_env_indices, [0, 1, 2, 3, 4, 5, 6, 7])
        # Each reply should carry exactly its 2-row slice.
        for reply in replies:
            self.assertEqual(len(reply.decision_counts), 2)
            self.assertEqual(reply.selected_choice_cols, [7, 7])
            self.assertIsNotNone(reply.replay_payload)
        # Actors own staging now; the server only returns per-request payloads.
        self.assertEqual(staging.calls, [])

    def test_arena_keeps_later_blank_metadata_after_no_blank_first_request(self) -> None:
        policy = _FakePolicy()
        server = TextInferenceServer(
            sampling_policy=policy,
            staging_buffer=_FakeStaging(),
            max_batch=8,
            max_wait_ms=0.0,
            min_batch_rows=2,
        )
        reqs = [
            TextInferenceRequest(
                native_batch=_make_native(1),
                packed_batch=_make_packed_batch([2], with_blank=False),
                env_indices=[0],
                perspective_player_indices=[0],
            ),
            TextInferenceRequest(
                native_batch=_make_native(1),
                packed_batch=_make_packed_batch([2], with_blank=True),
                env_indices=[1],
                perspective_player_indices=[1],
            ),
        ]
        server.start()
        try:
            futs = [server.submit(req) for req in reqs]
            replies = [fut.result(timeout=5.0) for fut in futs]
        finally:
            server.stop()

        self.assertEqual([len(reply.decision_counts) for reply in replies], [1, 1])
        self.assertEqual(policy.last_blank_shape, (2, 1))

    def test_pause_blocks_until_resume(self) -> None:
        policy = _FakePolicy()
        staging = _FakeStaging()
        server = TextInferenceServer(
            sampling_policy=policy,
            staging_buffer=staging,
            max_batch=8,
            max_wait_ms=0.0,
        )
        server.start()
        try:
            # Drain start-up so pause doesn't race with the very first poll.
            time.sleep(0.05)
            server.pause()
            req = TextInferenceRequest(
                native_batch=_make_native(1),
                packed_batch=_make_packed_batch([1]),
                env_indices=[0],
                perspective_player_indices=[0],
            )
            fut = server.submit(req)
            time.sleep(0.1)
            self.assertFalse(fut.done(), "future should not resolve while paused")
            server.resume()
            reply = fut.result(timeout=5.0)
            self.assertEqual(len(reply.decision_counts), 1)
        finally:
            server.stop()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
