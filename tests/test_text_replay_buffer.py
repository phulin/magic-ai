"""Replay buffer tests for the V2 (decoder-target) layout."""

import unittest

import torch
from magic_ai.text_encoder.grammar import GrammarVocab
from magic_ai.text_encoder.replay_buffer import (
    _REPLAY_FORMAT_VERSION,
    DecoderDecisionPayload,
    TextReplayBuffer,
)


def _buffer(
    *,
    capacity: int = 4,
    max_decoder_len: int = 8,
    max_anchors: int = 6,
    max_blockers: int = 4,
    max_attackers: int = 4,
) -> TextReplayBuffer:
    return TextReplayBuffer(
        capacity=capacity,
        max_tokens=16,
        max_options=4,
        max_targets_per_option=2,
        max_decision_groups=2,
        max_cached_choices=4,
        max_decoder_len=max_decoder_len,
        max_anchors=max_anchors,
        max_blockers=max_blockers,
        max_attackers=max_attackers,
    )


def _make_payload(
    *,
    batch: int,
    decoder_len: int,
    n_anchors: int,
    n_blockers: int,
    n_attackers: int,
) -> DecoderDecisionPayload:
    rng = torch.Generator().manual_seed(0xC0DE)
    output_token_ids = torch.randint(
        0, int(GrammarVocab.PAD) + 5, (batch, decoder_len), dtype=torch.int32, generator=rng
    )
    output_pointer_pos = torch.full((batch, decoder_len), -1, dtype=torch.int32)
    output_pointer_pos[:, 0] = torch.arange(batch, dtype=torch.int32)
    output_is_pointer = torch.zeros(batch, decoder_len, dtype=torch.bool)
    output_is_pointer[:, 0] = True
    output_pad_mask = torch.zeros(batch, decoder_len, dtype=torch.bool)
    output_pad_mask[:, : decoder_len - 1] = True
    decision_type = torch.tensor([b % 5 for b in range(batch)], dtype=torch.int32)
    pointer_anchor_positions = torch.full((batch, n_anchors), -1, dtype=torch.int32)
    pointer_anchor_kinds = torch.full((batch, n_anchors), -1, dtype=torch.int32)
    pointer_anchor_subjects = torch.full((batch, n_anchors), -1, dtype=torch.int32)
    pointer_anchor_handles = torch.full((batch, n_anchors), -1, dtype=torch.int32)
    if n_anchors > 0:
        pointer_anchor_positions[:, 0] = torch.arange(batch, dtype=torch.int32) + 10
        pointer_anchor_kinds[:, 0] = 1
        pointer_anchor_subjects[:, 0] = torch.arange(batch, dtype=torch.int32)
        pointer_anchor_handles[:, 0] = torch.arange(batch, dtype=torch.int32) * 100
    pointer_anchor_count = torch.full((batch,), 1, dtype=torch.int32)
    legal_edge_bitmap = torch.zeros(batch, n_blockers, n_attackers, dtype=torch.bool)
    if n_blockers > 0 and n_attackers > 0:
        legal_edge_bitmap[:, 0, 0] = True
    legal_edge_n_blockers = torch.full((batch,), n_blockers, dtype=torch.int32)
    legal_edge_n_attackers = torch.full((batch,), n_attackers, dtype=torch.int32)
    output_log_prob = torch.zeros(batch, decoder_len, dtype=torch.float32)
    # Synthetic per-step grammar masks — small fixed widths sufficient for
    # the buffer round-trip tests (the buffer pads to its configured caps).
    vocab_mask = torch.zeros(batch, decoder_len, 4, dtype=torch.bool)
    pointer_mask = torch.zeros(batch, decoder_len, 4, dtype=torch.bool)
    return DecoderDecisionPayload(
        output_token_ids=output_token_ids,
        output_pointer_pos=output_pointer_pos,
        output_is_pointer=output_is_pointer,
        output_pad_mask=output_pad_mask,
        output_log_prob=output_log_prob,
        decision_type=decision_type,
        pointer_anchor_positions=pointer_anchor_positions,
        pointer_anchor_kinds=pointer_anchor_kinds,
        pointer_anchor_subjects=pointer_anchor_subjects,
        pointer_anchor_handles=pointer_anchor_handles,
        pointer_anchor_count=pointer_anchor_count,
        legal_edge_bitmap=legal_edge_bitmap,
        legal_edge_n_blockers=legal_edge_n_blockers,
        legal_edge_n_attackers=legal_edge_n_attackers,
        vocab_mask=vocab_mask,
        pointer_mask=pointer_mask,
    )


class FormatVersionTests(unittest.TestCase):
    def test_version_constant(self) -> None:
        self.assertEqual(_REPLAY_FORMAT_VERSION, 2)


class ConstructionTests(unittest.TestCase):
    def test_decoder_tensors_have_expected_shapes_and_dtypes(self) -> None:
        buffer = _buffer(
            capacity=3, max_decoder_len=12, max_anchors=7, max_blockers=5, max_attackers=6
        )
        d = buffer.decoder
        self.assertEqual(tuple(d.output_token_ids.shape), (3, 12))
        self.assertEqual(d.output_token_ids.dtype, torch.int32)
        self.assertTrue(bool((d.output_token_ids == int(GrammarVocab.PAD)).all().item()))

        self.assertEqual(tuple(d.output_pointer_pos.shape), (3, 12))
        self.assertEqual(d.output_pointer_pos.dtype, torch.int32)
        self.assertTrue(bool((d.output_pointer_pos == -1).all().item()))

        self.assertEqual(tuple(d.output_is_pointer.shape), (3, 12))
        self.assertEqual(d.output_is_pointer.dtype, torch.bool)
        self.assertFalse(bool(d.output_is_pointer.any().item()))

        self.assertEqual(tuple(d.output_pad_mask.shape), (3, 12))
        self.assertEqual(d.output_pad_mask.dtype, torch.bool)
        self.assertFalse(bool(d.output_pad_mask.any().item()))

        self.assertEqual(tuple(d.decision_type.shape), (3,))
        self.assertEqual(d.decision_type.dtype, torch.int32)
        self.assertTrue(bool((d.decision_type == -1).all().item()))

        for name in (
            "pointer_anchor_positions",
            "pointer_anchor_kinds",
            "pointer_anchor_subjects",
            "pointer_anchor_handles",
        ):
            tensor = getattr(d, name)
            self.assertEqual(tuple(tensor.shape), (3, 7), name)
            self.assertEqual(tensor.dtype, torch.int32, name)
            self.assertTrue(bool((tensor == -1).all().item()), name)

        self.assertEqual(tuple(d.pointer_anchor_count.shape), (3,))
        self.assertEqual(d.pointer_anchor_count.dtype, torch.int32)
        self.assertTrue(bool((d.pointer_anchor_count == 0).all().item()))

        self.assertEqual(tuple(d.legal_edge_bitmap.shape), (3, 5, 6))
        self.assertEqual(d.legal_edge_bitmap.dtype, torch.bool)
        self.assertFalse(bool(d.legal_edge_bitmap.any().item()))

        self.assertEqual(tuple(d.legal_edge_n_blockers.shape), (3,))
        self.assertEqual(d.legal_edge_n_blockers.dtype, torch.int32)
        self.assertEqual(tuple(d.legal_edge_n_attackers.shape), (3,))
        self.assertEqual(d.legal_edge_n_attackers.dtype, torch.int32)


class DecoderRoundTripTests(unittest.TestCase):
    def test_commit_decoder_decision_round_trip(self) -> None:
        buffer = _buffer()
        reservation = buffer.reserve_append(row_count=2, token_count=0, decision_count=0)
        payload = _make_payload(batch=2, decoder_len=8, n_anchors=6, n_blockers=4, n_attackers=4)
        buffer.commit_decoder_decision(reservation, payload)

        rows = torch.tensor([reservation.row_start, reservation.row_start + 1], dtype=torch.long)
        gathered = buffer.gather_decoder(rows)

        torch.testing.assert_close(gathered.output_token_ids, payload.output_token_ids)
        torch.testing.assert_close(gathered.output_pointer_pos, payload.output_pointer_pos)
        torch.testing.assert_close(gathered.output_is_pointer, payload.output_is_pointer)
        torch.testing.assert_close(gathered.output_pad_mask, payload.output_pad_mask)
        torch.testing.assert_close(gathered.decision_type, payload.decision_type)
        torch.testing.assert_close(
            gathered.pointer_anchor_positions, payload.pointer_anchor_positions
        )
        torch.testing.assert_close(gathered.pointer_anchor_kinds, payload.pointer_anchor_kinds)
        torch.testing.assert_close(
            gathered.pointer_anchor_subjects, payload.pointer_anchor_subjects
        )
        torch.testing.assert_close(gathered.pointer_anchor_handles, payload.pointer_anchor_handles)
        torch.testing.assert_close(gathered.pointer_anchor_count, payload.pointer_anchor_count)
        torch.testing.assert_close(gathered.legal_edge_bitmap, payload.legal_edge_bitmap)
        torch.testing.assert_close(gathered.legal_edge_n_blockers, payload.legal_edge_n_blockers)
        torch.testing.assert_close(gathered.legal_edge_n_attackers, payload.legal_edge_n_attackers)

    def test_commit_writes_at_reservation_offset(self) -> None:
        buffer = _buffer(capacity=4)
        # Burn the first reservation slot so the next commit lands at row 1.
        first = buffer.reserve_append(row_count=1, token_count=0, decision_count=0)
        second = buffer.reserve_append(row_count=1, token_count=0, decision_count=0)

        payload = _make_payload(batch=1, decoder_len=8, n_anchors=6, n_blockers=4, n_attackers=4)
        buffer.commit_decoder_decision(second, payload)

        # Row 0 (the first reservation) is still untouched / fill values.
        self.assertEqual(int(buffer.decoder.decision_type[0].item()), -1)
        self.assertEqual(int(buffer.decoder.pointer_anchor_count[0].item()), 0)

        # Row 1 carries the payload we wrote.
        self.assertEqual(int(buffer.decoder.decision_type[1].item()), int(payload.decision_type[0]))
        self.assertEqual(int(buffer.decoder.pointer_anchor_count[1].item()), 1)
        # Cleanup the unused reservations to keep state tidy.
        buffer.commit(first)
        buffer.commit(second)


class PaddingTests(unittest.TestCase):
    def test_smaller_payload_is_padded_with_fill_values(self) -> None:
        buffer = _buffer(
            capacity=2, max_decoder_len=8, max_anchors=6, max_blockers=4, max_attackers=4
        )
        reservation = buffer.reserve_append(row_count=1, token_count=0, decision_count=0)
        payload = _make_payload(batch=1, decoder_len=3, n_anchors=2, n_blockers=2, n_attackers=2)
        buffer.commit_decoder_decision(reservation, payload)

        rows = torch.tensor([reservation.row_start], dtype=torch.long)
        gathered = buffer.gather_decoder(rows)

        # First three steps come from the payload.
        torch.testing.assert_close(gathered.output_token_ids[:, :3], payload.output_token_ids)
        # Remaining steps are PAD.
        self.assertTrue(
            bool((gathered.output_token_ids[:, 3:] == int(GrammarVocab.PAD)).all().item())
        )
        # Pointer pos -1 fill in tail.
        self.assertTrue(bool((gathered.output_pointer_pos[:, 3:] == -1).all().item()))
        # is_pointer / pad_mask false in tail.
        self.assertFalse(bool(gathered.output_is_pointer[:, 3:].any().item()))
        self.assertFalse(bool(gathered.output_pad_mask[:, 3:].any().item()))

        # Anchor tail is padded with -1.
        self.assertTrue(bool((gathered.pointer_anchor_positions[:, 2:] == -1).all().item()))
        self.assertTrue(bool((gathered.pointer_anchor_kinds[:, 2:] == -1).all().item()))
        self.assertTrue(bool((gathered.pointer_anchor_subjects[:, 2:] == -1).all().item()))
        self.assertTrue(bool((gathered.pointer_anchor_handles[:, 2:] == -1).all().item()))

        # Legal-edge bitmap padded with False outside the smaller block.
        self.assertFalse(bool(gathered.legal_edge_bitmap[:, 2:, :].any().item()))
        self.assertFalse(bool(gathered.legal_edge_bitmap[:, :, 2:].any().item()))

    def test_payload_oversize_raises(self) -> None:
        buffer = _buffer(max_decoder_len=4)
        reservation = buffer.reserve_append(row_count=1, token_count=0, decision_count=0)
        payload = _make_payload(batch=1, decoder_len=5, n_anchors=1, n_blockers=1, n_attackers=1)
        with self.assertRaisesRegex(ValueError, "max_decoder_len"):
            buffer.commit_decoder_decision(reservation, payload)


class ReservationLifecycleTests(unittest.TestCase):
    def test_commit_order_train_window(self) -> None:
        buffer = _buffer(capacity=4)
        first = buffer.reserve_append(row_count=2, token_count=0, decision_count=0)
        second = buffer.reserve_append(row_count=1, token_count=0, decision_count=0)

        # Out-of-order seal stages but doesn't publish.
        buffer.commit(second)
        self.assertEqual(buffer.committed_size, 0)
        self.assertIsNone(buffer.claim_train_window(min_rows=1, max_rows=4))

        buffer.commit(first)
        # Sealed but not yet metadata-marked complete.
        self.assertEqual(buffer.committed_size, 0)

        buffer.write_episode_metadata(
            torch.tensor([0, 1]),
            episode_id=42,
            terminal_reward_p0=-1.0,
            zero_sum=True,
        )
        self.assertEqual(buffer.committed_size, 2)
        window = buffer.claim_train_window(min_rows=2, max_rows=2)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual((window.row_start, window.row_end), (0, 2))
        torch.testing.assert_close(window.rows, torch.tensor([0, 1], dtype=torch.long))
        buffer.release_train_window(window)

    def test_terminal_boundary_train_window_does_not_split_episodes(self) -> None:
        buffer = _buffer(capacity=8)
        reservation = buffer.reserve_append(row_count=5, token_count=0, decision_count=0)
        buffer.commit(reservation)
        buffer.write_episode_metadata(
            torch.tensor([0, 1, 2]),
            episode_id=10,
            terminal_reward_p0=1.0,
            zero_sum=True,
        )
        buffer.write_episode_metadata(
            torch.tensor([3, 4]),
            episode_id=11,
            terminal_reward_p0=-1.0,
            zero_sum=True,
        )

        first = buffer.claim_train_window(
            min_rows=1,
            max_rows=4,
            target_rows=1,
            require_terminal_boundary=True,
        )
        assert first is not None
        self.assertEqual((first.row_start, first.row_end), (0, 3))
        buffer.release_train_window(first)

        # The next complete episode is longer than max_rows=1 from the new
        # head; R-NaD should extend to the terminal row instead of splitting.
        second = buffer.claim_train_window(
            min_rows=1,
            max_rows=1,
            target_rows=1,
            require_terminal_boundary=True,
        )
        assert second is not None
        self.assertEqual((second.row_start, second.row_end), (3, 5))
        buffer.release_train_window(second)

    def test_wraparound_reservation_survives_claim_release_cycle(self) -> None:
        """Repro: claim spans a contiguous chain, then a wrap-around reservation
        sits behind it; the next claim/release of the wrap-around window must not
        crash with "claimed replay window must begin at the row ring head".

        Specifically exercises _consume_committed_prefix_locked: it must only
        pop windows whose row_start chains forward from the claim_start, never
        a wrap-around window that happens to satisfy row_end <= claim_end.
        """
        # capacity=6. Sequence: reserve (0,2), reserve (2,6); commit + meta
        # both, claim+release (0,2) so ring start advances to 2 while ring is
        # still partially used; then reserve a wrap-around (0,2) before
        # claiming the (2,6) chain. The next claim must take the chain,
        # leaving the wrap-around window for the following claim — and the
        # release of the chain claim must not pop the wrap-around window
        # by accident.
        buffer = _buffer(capacity=6)

        r0 = buffer.reserve_append(row_count=2, token_count=0, decision_count=0)
        r1 = buffer.reserve_append(row_count=4, token_count=0, decision_count=0)
        buffer.commit(r0)
        buffer.commit(r1)
        buffer.write_episode_metadata(
            torch.tensor([0, 1]), episode_id=0, terminal_reward_p0=0.0, zero_sum=True
        )
        buffer.write_episode_metadata(
            torch.tensor([2, 3, 4, 5]), episode_id=1, terminal_reward_p0=0.0, zero_sum=True
        )

        # Claim+release the first window so the ring start advances to 2
        # while r1 (rows 2..5) is still held.
        w0 = buffer.claim_train_window(min_rows=2, max_rows=2)
        assert w0 is not None
        self.assertEqual((w0.row_start, w0.row_end), (0, 2))
        buffer.release_train_window(w0)

        # Now r2 wraps: cursor=0, count=2, count <= start=2.
        r2 = buffer.reserve_append(row_count=2, token_count=0, decision_count=0)
        buffer.commit(r2)
        buffer.write_episode_metadata(
            torch.tensor([0, 1]), episode_id=2, terminal_reward_p0=0.0, zero_sum=True
        )

        # committed_windows now == [(2,6), (0,2)]; the chain prefix is
        # just (2,6) because (0,2).row_start (0) != (2,6).row_end (6).
        w1 = buffer.claim_train_window(min_rows=4, max_rows=4)
        assert w1 is not None
        self.assertEqual((w1.row_start, w1.row_end), (2, 6))
        # Releasing must NOT pop the wrap-around window (0,2) by accident.
        buffer.release_train_window(w1)

        # The wrap-around window (rows 0..1) must still be claimable.
        w2 = buffer.claim_train_window(min_rows=2, max_rows=2)
        assert w2 is not None
        self.assertEqual((w2.row_start, w2.row_end), (0, 2))
        buffer.release_train_window(w2)

    def test_wrapped_ring_does_not_reserve_across_live_start(self) -> None:
        """When cursor < start, only cursor..start is contiguous free space."""

        buffer = _buffer(capacity=32)

        initial = buffer.reserve_append(row_count=28, token_count=0, decision_count=0)
        self.assertEqual((initial.row_start, initial.row_end), (0, 28))
        buffer.commit(initial)
        buffer.write_episode_metadata(
            torch.arange(0, 28, dtype=torch.long),
            episode_id=0,
            terminal_reward_p0=0.0,
            zero_sum=True,
        )

        head = buffer.claim_train_window(min_rows=16, max_rows=16)
        assert head is not None
        self.assertEqual((head.row_start, head.row_end), (0, 16))
        buffer.release_train_window(head)

        wrapped = buffer.reserve_append(row_count=11, token_count=0, decision_count=0)
        self.assertEqual((wrapped.row_start, wrapped.row_end), (0, 11))
        buffer.commit(wrapped)
        buffer.write_episode_metadata(
            torch.arange(0, 11, dtype=torch.long),
            episode_id=1,
            terminal_reward_p0=0.0,
            zero_sum=True,
        )

        # Total free rows is 9, but only rows 11..15 are contiguous at the
        # cursor. Reserving 8 rows would overlap live rows 16..18.
        self.assertFalse(buffer.can_reserve(row_count=8, token_count=0, decision_count=0))
        self.assertTrue(buffer.can_reserve(row_count=5, token_count=0, decision_count=0))


if __name__ == "__main__":
    unittest.main()
