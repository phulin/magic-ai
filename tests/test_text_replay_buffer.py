import unittest

import torch
from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch, pack_batch
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


def _encoded_batch() -> TextEncodedBatch:
    token_ids = torch.tensor([[11, 12, 13, 0, 0], [21, 22, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    card_ref_positions = torch.full((2, MAX_CARD_REFS), -1, dtype=torch.long)
    card_ref_positions[0, 0] = 1
    card_ref_positions[1, 2] = 0
    option_positions = torch.tensor([[1, 2, -1], [0, -1, -1]])
    option_mask = option_positions >= 0
    target_positions = torch.full((2, 3, 2), -1, dtype=torch.long)
    target_positions[0, 0, 0] = 2
    target_mask = target_positions >= 0
    seq_lengths = torch.tensor([3, 2])
    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        option_positions=option_positions,
        option_mask=option_mask,
        target_positions=target_positions,
        target_mask=target_mask,
        seq_lengths=seq_lengths,
    )


def _buffer() -> TextReplayBuffer:
    return TextReplayBuffer(
        capacity=2,
        max_tokens=5,
        max_options=3,
        max_targets_per_option=2,
        max_decision_groups=3,
        max_cached_choices=4,
        recurrent_layers=1,
        recurrent_hidden_dim=6,
    )


def _unpack(batch: PackedTextBatch, *, max_tokens: int, pad_id: int = 0) -> TextEncodedBatch:
    b = int(batch.seq_lengths.shape[0])
    token_ids = torch.full((b, max_tokens), pad_id, dtype=batch.token_ids.dtype)
    attention_mask = torch.zeros((b, max_tokens), dtype=torch.bool)
    token_ids[batch.seq_id.long(), batch.pos_in_seq.long()] = batch.token_ids
    attention_mask[batch.seq_id.long(), batch.pos_in_seq.long()] = True
    base = batch.state_positions

    def rebase(pos: torch.Tensor, view_shape: tuple[int, ...]) -> torch.Tensor:
        valid = pos >= 0
        shifted = pos - base.view(view_shape)
        return torch.where(valid, shifted, pos)

    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=rebase(batch.card_ref_positions, (b, 1)),
        option_positions=rebase(batch.option_positions, (b, 1)),
        option_mask=batch.option_mask,
        target_positions=rebase(batch.target_positions, (b, 1, 1)),
        target_mask=batch.target_mask,
        seq_lengths=batch.seq_lengths,
    )


class TextReplayBufferTests(unittest.TestCase):
    def test_append_and_gather_round_trip(self) -> None:
        buffer = _buffer()
        encoded = _encoded_batch()
        decision_option_idx = torch.tensor([[0, 1, -1, -1], [-1, 2, -1, -1]])
        decision_target_idx = torch.tensor([[-1, 0, -1, -1], [-1, -1, -1, -1]])
        decision_mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
        uses_none_head = torch.tensor([False, True])
        selected_indices = torch.tensor([1, 0])
        h_in = torch.full((1, 6), 0.25)
        c_in = torch.full((1, 6), -0.5)

        row = buffer.append(
            encoded=encoded,
            batch_index=0,
            trace_kind_id=3,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            selected_indices=selected_indices,
            may_selected=0.0,
            old_log_prob=-1.25,
            value=0.75,
            perspective_player_idx=1,
            lstm_h_in=h_in,
            lstm_c_in=c_in,
        )
        gathered = buffer.gather([row])
        gathered_encoded = _unpack(gathered.encoded, max_tokens=buffer.max_tokens)

        # The buffer stores narrower dtypes than the encoder produces (e.g.
        # int32 token_ids, bool attention_mask, int16 decision indices); the
        # round-trip preserves values, not dtype.
        torch.testing.assert_close(
            gathered_encoded.token_ids[0], encoded.token_ids[0], check_dtype=False
        )
        torch.testing.assert_close(
            gathered_encoded.attention_mask[0], encoded.attention_mask[0], check_dtype=False
        )
        torch.testing.assert_close(
            gathered_encoded.card_ref_positions[0],
            encoded.card_ref_positions[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered.decision_option_idx[0, :2], decision_option_idx, check_dtype=False
        )
        torch.testing.assert_close(
            gathered.decision_target_idx[0, :2], decision_target_idx, check_dtype=False
        )
        torch.testing.assert_close(gathered.decision_mask[0, :2], decision_mask)
        torch.testing.assert_close(gathered.uses_none_head[0, :2], uses_none_head)
        torch.testing.assert_close(
            gathered.selected_indices[0, :2], selected_indices, check_dtype=False
        )
        self.assertEqual(int(gathered.trace_kind_id[0]), 3)
        self.assertEqual(int(gathered.decision_count[0]), 2)
        self.assertAlmostEqual(float(gathered.old_log_prob[0]), -1.25)
        self.assertAlmostEqual(float(gathered.value[0]), 0.75)
        self.assertEqual(int(gathered.perspective_player_idx[0]), 1)
        self.assertIsNotNone(gathered.lstm_h_in)
        self.assertIsNotNone(gathered.lstm_c_in)
        assert gathered.lstm_h_in is not None
        assert gathered.lstm_c_in is not None
        torch.testing.assert_close(gathered.lstm_h_in[0], h_in)
        torch.testing.assert_close(gathered.lstm_c_in[0], c_in)

    def test_append_packed_and_gather_round_trip(self) -> None:
        buffer = _buffer()
        encoded = _encoded_batch()
        packed = pack_batch(encoded)
        decision_option_idx = torch.tensor([[0, 1, -1, -1]])
        decision_target_idx = torch.tensor([[-1, 0, -1, -1]])
        decision_mask = torch.tensor([[True, True, False, False]])
        uses_none_head = torch.tensor([False])
        selected_indices = torch.tensor([1])
        h_in = torch.full((1, 6), 0.25)
        c_in = torch.full((1, 6), -0.5)

        row = buffer.append_packed(
            encoded=packed,
            batch_index=0,
            trace_kind_id=3,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            selected_indices=selected_indices,
            may_selected=0.0,
            old_log_prob=-1.25,
            value=0.75,
            perspective_player_idx=1,
            lstm_h_in=h_in,
            lstm_c_in=c_in,
        )
        gathered = buffer.gather([row])
        gathered_encoded = _unpack(gathered.encoded, max_tokens=buffer.max_tokens)

        torch.testing.assert_close(
            gathered_encoded.token_ids[0], encoded.token_ids[0], check_dtype=False
        )
        torch.testing.assert_close(
            gathered_encoded.attention_mask[0], encoded.attention_mask[0], check_dtype=False
        )
        torch.testing.assert_close(
            gathered_encoded.card_ref_positions[0],
            encoded.card_ref_positions[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.option_positions[0],
            encoded.option_positions[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.target_positions[0],
            encoded.target_positions[0],
            check_dtype=False,
        )
        torch.testing.assert_close(gathered.decision_mask[0, :1], decision_mask)
        self.assertEqual(int(gathered.decision_count[0]), 1)
        self.assertAlmostEqual(float(gathered.old_log_prob[0]), -1.25)

    def test_append_multiple_packed_batches_and_gather_minibatch(self) -> None:
        buffer = TextReplayBuffer(
            capacity=4,
            max_tokens=6,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=2,
            max_cached_choices=4,
            recurrent_layers=1,
            recurrent_hidden_dim=6,
        )
        encoded_a = TextEncodedBatch(
            token_ids=torch.tensor([[101, 102, 103, 104, 0, 0], [201, 202, 0, 0, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]]),
            card_ref_positions=torch.full((2, MAX_CARD_REFS), -1, dtype=torch.long),
            option_positions=torch.tensor([[1, 2, -1], [0, -1, -1]]),
            option_mask=torch.tensor([[True, True, False], [True, False, False]]),
            target_positions=torch.tensor(
                [[[3, -1], [-1, -1], [-1, -1]], [[1, -1], [-1, -1], [-1, -1]]]
            ),
            target_mask=torch.tensor(
                [
                    [[True, False], [False, False], [False, False]],
                    [[True, False], [False, False], [False, False]],
                ]
            ),
            seq_lengths=torch.tensor([4, 2]),
        )
        encoded_a.card_ref_positions[0, 4] = 2
        encoded_a.card_ref_positions[1, 5] = 1
        encoded_b = TextEncodedBatch(
            token_ids=torch.tensor([[301, 302, 303, 0, 0, 0], [401, 402, 403, 404, 405, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]]),
            card_ref_positions=torch.full((2, MAX_CARD_REFS), -1, dtype=torch.long),
            option_positions=torch.tensor([[1, -1, -1], [2, 3, -1]]),
            option_mask=torch.tensor([[True, False, False], [True, True, False]]),
            target_positions=torch.tensor(
                [[[2, -1], [-1, -1], [-1, -1]], [[4, -1], [1, -1], [-1, -1]]]
            ),
            target_mask=torch.tensor(
                [
                    [[True, False], [False, False], [False, False]],
                    [[True, False], [True, False], [False, False]],
                ]
            ),
            seq_lengths=torch.tensor([3, 5]),
        )
        encoded_b.card_ref_positions[0, 6] = 0
        encoded_b.card_ref_positions[1, 7] = 4

        rows_a = buffer.append_batch(
            encoded=pack_batch(encoded_a),
            trace_kind_id=torch.tensor([1, 2]),
            decision_count=torch.tensor([1, 2]),
            decision_option_idx=torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1]]),
            decision_target_idx=torch.tensor([[-1, -1, -1, -1], [0, -1, -1, -1], [1, -1, -1, -1]]),
            decision_mask=torch.tensor(
                [
                    [True, False, False, False],
                    [True, False, False, False],
                    [True, False, False, False],
                ]
            ),
            uses_none_head=torch.tensor([False, True, False]),
            selected_indices=torch.tensor([0, 0, 0]),
            may_selected=torch.tensor([0.0, 1.0]),
            old_log_prob=torch.tensor([-0.1, -0.2]),
            value=torch.tensor([0.3, 0.4]),
            perspective_player_idx=torch.tensor([0, 1]),
            lstm_h_in=torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),
            lstm_c_in=torch.arange(100, 112, dtype=torch.float32).reshape(1, 2, 6),
        )
        rows_b = buffer.append_batch(
            encoded=pack_batch(encoded_b),
            trace_kind_id=torch.tensor([3, 4]),
            decision_count=torch.tensor([1, 1]),
            decision_option_idx=torch.tensor([[0, 1, -1, -1], [2, 3, -1, -1]]),
            decision_target_idx=torch.tensor([[-1, 0, -1, -1], [1, -1, -1, -1]]),
            decision_mask=torch.tensor([[True, True, False, False], [True, True, False, False]]),
            uses_none_head=torch.tensor([True, False]),
            selected_indices=torch.tensor([1, 0]),
            may_selected=torch.tensor([1.0, 0.0]),
            old_log_prob=torch.tensor([-0.3, -0.4]),
            value=torch.tensor([0.5, 0.6]),
            perspective_player_idx=torch.tensor([1, 0]),
            lstm_h_in=torch.arange(200, 212, dtype=torch.float32).reshape(1, 2, 6),
            lstm_c_in=torch.arange(300, 312, dtype=torch.float32).reshape(1, 2, 6),
        )

        gathered = buffer.gather([int(rows_b[1]), int(rows_a[0]), int(rows_b[0])])
        gathered_encoded = _unpack(gathered.encoded, max_tokens=buffer.max_tokens)

        torch.testing.assert_close(
            gathered_encoded.token_ids[0], encoded_b.token_ids[1], check_dtype=False
        )
        torch.testing.assert_close(
            gathered_encoded.token_ids[1], encoded_a.token_ids[0], check_dtype=False
        )
        torch.testing.assert_close(
            gathered_encoded.token_ids[2], encoded_b.token_ids[0], check_dtype=False
        )
        torch.testing.assert_close(
            gathered_encoded.card_ref_positions[0],
            encoded_b.card_ref_positions[1],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.option_positions[1], encoded_a.option_positions[0], check_dtype=False
        )
        torch.testing.assert_close(
            gathered_encoded.target_positions[2], encoded_b.target_positions[0], check_dtype=False
        )
        torch.testing.assert_close(
            gathered.trace_kind_id, torch.tensor([4, 1, 3]), check_dtype=False
        )
        torch.testing.assert_close(
            gathered.decision_count, torch.tensor([1, 1, 1]), check_dtype=False
        )
        torch.testing.assert_close(
            gathered.decision_option_idx[:, 0],
            torch.tensor([[2, 3, -1, -1], [0, -1, -1, -1], [0, 1, -1, -1]]),
            check_dtype=False,
        )
        torch.testing.assert_close(gathered.old_log_prob, torch.tensor([-0.4, -0.1, -0.3]))
        torch.testing.assert_close(gathered.value, torch.tensor([0.6, 0.3, 0.5]))
        assert gathered.lstm_h_in is not None
        assert gathered.lstm_c_in is not None
        torch.testing.assert_close(
            gathered.lstm_h_in[0], torch.arange(206, 212, dtype=torch.float32).reshape(1, 6)
        )
        torch.testing.assert_close(
            gathered.lstm_c_in[1], torch.arange(100, 106, dtype=torch.float32).reshape(1, 6)
        )

    def test_release_reuses_rows_and_reset_clears_occupancy(self) -> None:
        buffer = _buffer()
        encoded = _encoded_batch()
        empty_groups = torch.empty(0, 4, dtype=torch.long)
        empty_mask = torch.empty(0, 4, dtype=torch.bool)
        empty_bool = torch.empty(0, dtype=torch.bool)
        empty_selected = torch.empty(0, dtype=torch.long)
        h_in = torch.zeros(1, 6)
        c_in = torch.zeros(1, 6)

        row0 = buffer.append(
            encoded=encoded,
            batch_index=0,
            trace_kind_id=0,
            decision_option_idx=empty_groups,
            decision_target_idx=empty_groups,
            decision_mask=empty_mask,
            uses_none_head=empty_bool,
            selected_indices=empty_selected,
            may_selected=1.0,
            old_log_prob=0.0,
            value=0.0,
            perspective_player_idx=0,
            lstm_h_in=h_in,
            lstm_c_in=c_in,
        )
        row1 = buffer.append(
            encoded=encoded,
            batch_index=1,
            trace_kind_id=6,
            decision_option_idx=empty_groups,
            decision_target_idx=empty_groups,
            decision_mask=empty_mask,
            uses_none_head=empty_bool,
            selected_indices=empty_selected,
            may_selected=0.0,
            old_log_prob=0.0,
            value=0.0,
            perspective_player_idx=1,
            lstm_h_in=h_in,
            lstm_c_in=c_in,
        )
        self.assertEqual(buffer.size, 2)
        with self.assertRaisesRegex(RuntimeError, "full"):
            buffer.append(
                encoded=encoded,
                batch_index=0,
                trace_kind_id=0,
                decision_option_idx=empty_groups,
                decision_target_idx=empty_groups,
                decision_mask=empty_mask,
                uses_none_head=empty_bool,
                selected_indices=empty_selected,
                may_selected=0.0,
                old_log_prob=0.0,
                value=0.0,
                perspective_player_idx=0,
                lstm_h_in=h_in,
                lstm_c_in=c_in,
            )

        buffer.release_replay_rows([row0])
        self.assertEqual(buffer.size, 1)
        reused = buffer.append(
            encoded=encoded,
            batch_index=1,
            trace_kind_id=1,
            decision_option_idx=empty_groups,
            decision_target_idx=empty_groups,
            decision_mask=empty_mask,
            uses_none_head=empty_bool,
            selected_indices=empty_selected,
            may_selected=0.0,
            old_log_prob=0.0,
            value=0.0,
            perspective_player_idx=1,
            lstm_h_in=h_in,
            lstm_c_in=c_in,
        )
        self.assertEqual(reused, row0)
        buffer.reset()
        self.assertEqual(buffer.size, 0)
        with self.assertRaisesRegex(ValueError, f"replay row {row1} is not occupied"):
            buffer.gather([row1])

    def test_rejects_oversized_encoded_rows(self) -> None:
        buffer = TextReplayBuffer(
            capacity=1,
            max_tokens=4,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
        )
        encoded = _encoded_batch()

        with self.assertRaisesRegex(ValueError, "token width exceeds"):
            buffer.append(
                encoded=encoded,
                batch_index=0,
                trace_kind_id=0,
                decision_option_idx=torch.empty(0, 4, dtype=torch.long),
                decision_target_idx=torch.empty(0, 4, dtype=torch.long),
                decision_mask=torch.empty(0, 4, dtype=torch.bool),
                uses_none_head=torch.empty(0, dtype=torch.bool),
                selected_indices=torch.empty(0, dtype=torch.long),
                may_selected=0.0,
                old_log_prob=0.0,
                value=0.0,
                perspective_player_idx=0,
            )


if __name__ == "__main__":
    unittest.main()
