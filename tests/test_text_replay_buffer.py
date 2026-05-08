import unittest
from typing import Any

import torch
from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch, pack_batch
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer
from magic_ai.text_encoder.replay_triton import TRITON_AVAILABLE
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


def _encoded_batch() -> TextEncodedBatch:
    token_ids = torch.tensor([[11, 12, 13, 0, 0], [21, 22, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    card_ref_positions = torch.full((2, MAX_CARD_REFS), -1, dtype=torch.long)
    card_ref_positions[0, 0] = 1
    card_ref_positions[1, 2] = 0
    seq_lengths = torch.tensor([3, 2])
    blank_positions = torch.tensor([[1, 2, -1], [0, -1, -1]], dtype=torch.int32)
    blank_kind = torch.tensor([[901, 902, 0], [901, 0, 0]], dtype=torch.int32)
    blank_group = torch.tensor([[0, 0, -1], [1, -1, -1]], dtype=torch.int32)
    blank_group_kind = torch.tensor([[3, 3, 0], [2, 0, 0]], dtype=torch.int32)
    blank_option_index = torch.tensor([[1, 0, -1], [2, -1, -1]], dtype=torch.int32)
    blank_legal_ids = torch.tensor(
        [
            [[100, 101, 0], [100, 102, 103], [0, 0, 0]],
            [[100, 201, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.int32,
    )
    blank_legal_mask = blank_legal_ids > 0
    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lengths,
        seq_lengths_host=tuple(int(x) for x in seq_lengths.tolist()),
        blank_positions=blank_positions,
        blank_kind=blank_kind,
        blank_group=blank_group,
        blank_group_kind=blank_group_kind,
        blank_option_index=blank_option_index,
        blank_legal_ids=blank_legal_ids,
        blank_legal_mask=blank_legal_mask,
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
        seq_lengths=batch.seq_lengths,
        seq_lengths_host=batch.seq_lengths_host,
        blank_positions=rebase(batch.blank_positions, (b, 1)),
        blank_kind=batch.blank_kind,
        blank_group=batch.blank_group,
        blank_group_kind=batch.blank_group_kind,
        blank_option_index=batch.blank_option_index,
        blank_legal_ids=batch.blank_legal_ids,
        blank_legal_mask=batch.blank_legal_mask,
    )


def _packed_to_device(batch: PackedTextBatch, device: torch.device) -> PackedTextBatch:
    return PackedTextBatch(
        token_ids=batch.token_ids.to(device),
        seq_id=batch.seq_id.to(device),
        pos_in_seq=batch.pos_in_seq.to(device),
        cu_seqlens=batch.cu_seqlens.to(device),
        seq_lengths=batch.seq_lengths.to(device),
        state_positions=batch.state_positions.to(device),
        card_ref_positions=batch.card_ref_positions.to(device),
        seq_lengths_host=batch.seq_lengths_host,
        max_seqlen=batch.max_seqlen,
        blank_positions=batch.blank_positions.to(device),
        blank_kind=batch.blank_kind.to(device),
        blank_group=batch.blank_group.to(device),
        blank_group_kind=batch.blank_group_kind.to(device),
        blank_option_index=batch.blank_option_index.to(device),
        blank_legal_ids=batch.blank_legal_ids.to(device),
        blank_legal_mask=batch.blank_legal_mask.to(device),
    )


def _host_rows(rows: torch.Tensor) -> list[int]:
    return [int(row) for row in rows.detach().cpu().tolist()]


def _staged_encoded_kwargs(
    encoded: TextEncodedBatch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    envs = 3
    steps = 2
    flat_env = torch.tensor([2, 1], dtype=torch.long, device=device)
    flat_step = torch.tensor([0, 1], dtype=torch.long, device=device)
    token_ids = torch.zeros(envs, steps, 5, dtype=torch.int32, device=device)
    seq_lengths = torch.zeros(envs, steps, dtype=torch.int32, device=device)
    card_ref_positions = torch.full(
        (envs, steps, MAX_CARD_REFS), -1, dtype=torch.int32, device=device
    )
    blank_positions = torch.full((envs, steps, 3), -1, dtype=torch.int32, device=device)
    blank_kind = torch.zeros(envs, steps, 3, dtype=torch.int32, device=device)
    blank_group = torch.full((envs, steps, 3), -1, dtype=torch.int32, device=device)
    blank_group_kind = torch.zeros(envs, steps, 3, dtype=torch.int32, device=device)
    blank_option_index = torch.full((envs, steps, 3), -1, dtype=torch.int32, device=device)
    blank_legal_ids = torch.zeros(envs, steps, 3, 3, dtype=torch.int32, device=device)
    blank_legal_mask = torch.zeros(envs, steps, 3, 3, dtype=torch.bool, device=device)
    for row, (env_idx, step_idx) in enumerate(zip([2, 1], [0, 1], strict=True)):
        token_ids[env_idx, step_idx] = encoded.token_ids[row].to(device=device, dtype=torch.int32)
        seq_lengths[env_idx, step_idx] = encoded.seq_lengths[row].to(
            device=device, dtype=torch.int32
        )
        card_ref_positions[env_idx, step_idx] = encoded.card_ref_positions[row].to(
            device=device, dtype=torch.int32
        )
        blank_positions[env_idx, step_idx] = encoded.blank_positions[row].to(
            device=device, dtype=torch.int32
        )
        blank_kind[env_idx, step_idx] = encoded.blank_kind[row].to(device=device, dtype=torch.int32)
        blank_group[env_idx, step_idx] = encoded.blank_group[row].to(
            device=device, dtype=torch.int32
        )
        blank_group_kind[env_idx, step_idx] = encoded.blank_group_kind[row].to(
            device=device, dtype=torch.int32
        )
        blank_option_index[env_idx, step_idx] = encoded.blank_option_index[row].to(
            device=device, dtype=torch.int32
        )
        blank_legal_ids[env_idx, step_idx] = encoded.blank_legal_ids[row].to(
            device=device, dtype=torch.int32
        )
        blank_legal_mask[env_idx, step_idx] = encoded.blank_legal_mask[row].to(device=device)
    return {
        "flat_env": flat_env,
        "flat_step": flat_step,
        "token_ids": token_ids,
        "seq_lengths": seq_lengths[flat_env, flat_step].to(dtype=torch.long),
        "seq_lengths_host": tuple(int(x) for x in encoded.seq_lengths.tolist()),
        "card_ref_positions": card_ref_positions,
        "blank_positions": blank_positions,
        "blank_kind": blank_kind,
        "blank_group": blank_group,
        "blank_group_kind": blank_group_kind,
        "blank_option_index": blank_option_index,
        "blank_legal_ids": blank_legal_ids,
        "blank_legal_mask": blank_legal_mask,
    }


def _assert_replay_batch_close(
    test: unittest.TestCase,
    actual,
    expected,
) -> None:
    torch.testing.assert_close(actual.encoded.token_ids, expected.encoded.token_ids)
    torch.testing.assert_close(actual.encoded.pos_in_seq, expected.encoded.pos_in_seq)
    torch.testing.assert_close(actual.encoded.cu_seqlens, expected.encoded.cu_seqlens)
    torch.testing.assert_close(actual.encoded.seq_lengths, expected.encoded.seq_lengths)
    torch.testing.assert_close(actual.encoded.state_positions, expected.encoded.state_positions)
    torch.testing.assert_close(
        actual.encoded.card_ref_positions, expected.encoded.card_ref_positions
    )
    torch.testing.assert_close(actual.encoded.blank_positions, expected.encoded.blank_positions)
    torch.testing.assert_close(actual.encoded.blank_kind, expected.encoded.blank_kind)
    torch.testing.assert_close(actual.encoded.blank_group, expected.encoded.blank_group)
    torch.testing.assert_close(actual.encoded.blank_group_kind, expected.encoded.blank_group_kind)
    torch.testing.assert_close(
        actual.encoded.blank_option_index, expected.encoded.blank_option_index
    )
    torch.testing.assert_close(actual.encoded.blank_legal_ids, expected.encoded.blank_legal_ids)
    torch.testing.assert_close(actual.encoded.blank_legal_mask, expected.encoded.blank_legal_mask)
    torch.testing.assert_close(actual.trace_kind_id, expected.trace_kind_id)
    torch.testing.assert_close(actual.decision_start, expected.decision_start)
    torch.testing.assert_close(actual.decision_count, expected.decision_count)
    torch.testing.assert_close(actual.decision_option_idx, expected.decision_option_idx)
    torch.testing.assert_close(actual.decision_target_idx, expected.decision_target_idx)
    torch.testing.assert_close(actual.decision_mask, expected.decision_mask)
    torch.testing.assert_close(actual.uses_none_head, expected.uses_none_head)
    torch.testing.assert_close(actual.selected_indices, expected.selected_indices)
    torch.testing.assert_close(actual.step_for_decision_group, expected.step_for_decision_group)
    torch.testing.assert_close(actual.may_selected, expected.may_selected)
    torch.testing.assert_close(actual.old_log_prob, expected.old_log_prob)
    torch.testing.assert_close(actual.value, expected.value)
    torch.testing.assert_close(actual.perspective_player_idx, expected.perspective_player_idx)
    test.assertEqual(actual.lstm_h_in is None, expected.lstm_h_in is None)
    test.assertEqual(actual.lstm_c_in is None, expected.lstm_c_in is None)
    if actual.lstm_h_in is not None and expected.lstm_h_in is not None:
        torch.testing.assert_close(actual.lstm_h_in, expected.lstm_h_in)
    if actual.lstm_c_in is not None and expected.lstm_c_in is not None:
        torch.testing.assert_close(actual.lstm_c_in, expected.lstm_c_in)


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
            gathered_encoded.blank_positions[0],
            encoded.blank_positions[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.blank_kind[0],
            encoded.blank_kind[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.blank_group[0],
            encoded.blank_group[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.blank_group_kind[0],
            encoded.blank_group_kind[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.blank_option_index[0],
            encoded.blank_option_index[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.blank_legal_ids[0],
            encoded.blank_legal_ids[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.blank_legal_mask[0], encoded.blank_legal_mask[0]
        )
        torch.testing.assert_close(
            gathered.decision_option_idx, decision_option_idx, check_dtype=False
        )
        torch.testing.assert_close(
            gathered.decision_target_idx, decision_target_idx, check_dtype=False
        )
        torch.testing.assert_close(gathered.decision_mask, decision_mask)
        torch.testing.assert_close(gathered.uses_none_head, uses_none_head)
        torch.testing.assert_close(gathered.selected_indices, selected_indices, check_dtype=False)
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
            gathered_encoded.blank_positions[0],
            encoded.blank_positions[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.blank_legal_ids[0],
            encoded.blank_legal_ids[0],
            check_dtype=False,
        )
        torch.testing.assert_close(
            gathered_encoded.blank_legal_mask[0], encoded.blank_legal_mask[0]
        )
        torch.testing.assert_close(gathered.decision_mask, decision_mask)
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
            seq_lengths=torch.tensor([4, 2]),
            seq_lengths_host=(4, 2),
        )
        encoded_a.card_ref_positions[0, 4] = 2
        encoded_a.card_ref_positions[1, 5] = 1
        encoded_b = TextEncodedBatch(
            token_ids=torch.tensor([[301, 302, 303, 0, 0, 0], [401, 402, 403, 404, 405, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]]),
            card_ref_positions=torch.full((2, MAX_CARD_REFS), -1, dtype=torch.long),
            seq_lengths=torch.tensor([3, 5]),
            seq_lengths_host=(3, 5),
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

        self.assertEqual(gathered.encoded.max_seqlen, 5)
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
            gathered.trace_kind_id, torch.tensor([4, 1, 3]), check_dtype=False
        )
        torch.testing.assert_close(
            gathered.decision_count, torch.tensor([1, 1, 1]), check_dtype=False
        )
        torch.testing.assert_close(
            gathered.decision_option_idx,
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

    def test_commit_order_train_window_and_episode_metadata(self) -> None:
        buffer = TextReplayBuffer(
            capacity=4,
            max_tokens=6,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=2,
            max_cached_choices=4,
        )
        first = buffer.reserve_append(row_count=2, token_count=2, decision_count=0)
        second = buffer.reserve_append(row_count=1, token_count=1, decision_count=0)

        buffer.commit(second)
        self.assertEqual(buffer.committed_size, 0)
        self.assertIsNone(buffer.claim_train_window(min_rows=1, max_rows=4))

        buffer.commit(first)
        self.assertEqual(buffer.committed_size, 0)
        self.assertIsNone(buffer.claim_train_window(min_rows=1, max_rows=4))
        buffer.write_episode_metadata(
            torch.tensor([0, 1]),
            episode_id=42,
            terminal_reward_p0=-1.0,
            zero_sum=True,
            actor_id=7,
            behavior_policy_version=3,
            inference_policy_version=4,
            target_policy_version=5,
        )
        self.assertEqual(buffer.committed_size, 2)
        window = buffer.claim_train_window(min_rows=2, max_rows=2)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual((window.row_start, window.row_end), (0, 2))
        torch.testing.assert_close(window.rows.cpu(), torch.tensor([0, 1]))
        meta = buffer.episode_meta
        torch.testing.assert_close(meta.episode_id[:2].cpu(), torch.tensor([42, 42]))
        torch.testing.assert_close(meta.step_idx[:2].cpu(), torch.tensor([0, 1]))
        torch.testing.assert_close(meta.is_terminal[:2].cpu(), torch.tensor([False, True]))
        torch.testing.assert_close(meta.actor_id[:2].cpu(), torch.tensor([7, 7]))
        torch.testing.assert_close(
            meta.behavior_policy_version[:2].cpu(),
            torch.tensor([3, 3]),
        )
        torch.testing.assert_close(
            meta.inference_policy_version[:2].cpu(),
            torch.tensor([4, 4]),
        )
        torch.testing.assert_close(
            meta.target_policy_version[:2].cpu(),
            torch.tensor([5, 5]),
        )
        buffer.release_train_window(window)

    def test_ring_release_reuses_row_slots_without_reset(self) -> None:
        buffer = TextReplayBuffer(
            capacity=3,
            max_tokens=2,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            use_triton_gather=False,
        )

        def append_one(token: int) -> torch.Tensor:
            encoded = TextEncodedBatch(
                token_ids=torch.tensor([[token, 0]]),
                attention_mask=torch.tensor([[1, 0]]),
                card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.long),
                seq_lengths=torch.tensor([1]),
                seq_lengths_host=(1,),
            )
            return buffer.append_batch(
                encoded=pack_batch(encoded),
                trace_kind_id=torch.tensor([1]),
                decision_count=torch.tensor([1]),
                decision_option_idx=torch.tensor([[0, -1]]),
                decision_target_idx=torch.tensor([[-1, -1]]),
                decision_mask=torch.tensor([[True, False]]),
                uses_none_head=torch.tensor([False]),
                selected_indices=torch.tensor([0]),
                may_selected=torch.tensor([0.0]),
                old_log_prob=torch.tensor([-0.1]),
                value=torch.tensor([0.0]),
                perspective_player_idx=torch.tensor([0]),
            )

        row0 = append_one(101)
        row1 = append_one(102)
        buffer.release_rows(row0)
        row2 = append_one(103)
        buffer.release_rows(row1)
        row0_reused = append_one(104)

        torch.testing.assert_close(row2.cpu(), torch.tensor([2]))
        torch.testing.assert_close(row0_reused.cpu(), torch.tensor([0]))
        gathered = buffer.gather(torch.cat((row2, row0_reused)))
        torch.testing.assert_close(
            gathered.encoded.token_ids.cpu(),
            torch.tensor([103, 104], dtype=torch.int32),
        )

    def test_append_only_rows_and_reset_clears_occupancy(self) -> None:
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

        self.assertEqual(row0, 0)
        self.assertEqual(row1, 1)
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

    @unittest.skipUnless(
        torch.cuda.is_available() and TRITON_AVAILABLE,
        "requires CUDA and Triton",
    )
    def test_append_batch_triton_matches_torch_path(self) -> None:
        device = torch.device("cuda")
        encoded = _packed_to_device(pack_batch(_encoded_batch()), device)

        kwargs: dict[str, Any] = dict(
            encoded=encoded,
            trace_kind_id=torch.tensor([1, 2], device=device),
            decision_count=torch.tensor([2, 4], device=device),
            decision_option_idx=torch.tensor(
                [
                    [0, 1, -1, -1],
                    [2, -1, -1, -1],
                    [1, -1, -1, -1],
                    [0, 2, -1, -1],
                    [3, -1, -1, -1],
                    [2, 1, -1, -1],
                ],
                device=device,
            ),
            decision_target_idx=torch.tensor(
                [
                    [-1, 0, -1, -1],
                    [1, -1, -1, -1],
                    [0, -1, -1, -1],
                    [-1, 1, -1, -1],
                    [2, -1, -1, -1],
                    [0, 2, -1, -1],
                ],
                device=device,
            ),
            decision_mask=torch.tensor(
                [
                    [True, True, False, False],
                    [True, False, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                ],
                device=device,
            ),
            uses_none_head=torch.tensor([False, True, True, False, True, False], device=device),
            selected_indices=torch.tensor([1, 0, 0, 1, 0, 1], device=device),
            may_selected=torch.tensor([0.0, 1.0], device=device),
            old_log_prob=torch.tensor([-0.25, -0.5], device=device),
            value=torch.tensor([0.75, -0.125], device=device),
            perspective_player_idx=torch.tensor([0, 1], device=device),
            lstm_h_in=torch.arange(12, dtype=torch.float32, device=device).reshape(1, 2, 6),
            lstm_c_in=torch.arange(100, 112, dtype=torch.float32, device=device).reshape(1, 2, 6),
        )
        triton_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            recurrent_layers=1,
            recurrent_hidden_dim=6,
            device=device,
            use_triton_append=True,
        )
        torch_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            recurrent_layers=1,
            recurrent_hidden_dim=6,
            device=device,
            use_triton_append=False,
            use_triton_gather=False,
        )

        rows_triton = triton_buffer.append_batch(**kwargs)
        rows_torch = torch_buffer.append_batch(**kwargs)
        torch.testing.assert_close(rows_triton, rows_torch)
        _assert_replay_batch_close(
            self,
            triton_buffer.gather(_host_rows(rows_triton)),
            torch_buffer.gather(_host_rows(rows_torch)),
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and TRITON_AVAILABLE,
        "requires CUDA and Triton",
    )
    def test_append_batch_triton_matches_zero_and_truncated_decisions(self) -> None:
        device = torch.device("cuda")
        encoded = _packed_to_device(pack_batch(_encoded_batch()), device)
        kwargs: dict[str, Any] = dict(
            encoded=encoded,
            trace_kind_id=torch.tensor([2, 3], device=device),
            decision_count=torch.tensor([0, 4], device=device),
            decision_option_idx=torch.tensor(
                [
                    [0, -1, -1, -1],
                    [1, 2, -1, -1],
                    [3, -1, -1, -1],
                    [2, 0, -1, -1],
                ],
                device=device,
            ),
            decision_target_idx=torch.tensor(
                [
                    [-1, -1, -1, -1],
                    [0, 1, -1, -1],
                    [2, -1, -1, -1],
                    [1, 0, -1, -1],
                ],
                device=device,
            ),
            decision_mask=torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                ],
                device=device,
            ),
            uses_none_head=torch.tensor([False, True, False, True], device=device),
            selected_indices=torch.tensor([0, 1, 0, 1], device=device),
            may_selected=torch.tensor([1.0, 0.0], device=device),
            old_log_prob=torch.tensor([-0.75, -1.25], device=device),
            value=torch.tensor([0.125, 0.875], device=device),
            perspective_player_idx=torch.tensor([1, 0], device=device),
        )
        triton_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            device=device,
            use_triton_append=True,
        )
        torch_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            device=device,
            use_triton_append=False,
            use_triton_gather=False,
        )

        rows_triton = triton_buffer.append_batch(**kwargs)
        rows_torch = torch_buffer.append_batch(**kwargs)
        _assert_replay_batch_close(
            self,
            triton_buffer.gather(_host_rows(rows_triton)),
            torch_buffer.gather(_host_rows(rows_torch)),
        )

    def test_append_staged_batch_matches_packed_append_torch_path(self) -> None:
        device = torch.device("cpu")
        encoded_dense = _encoded_batch()
        encoded = pack_batch(encoded_dense)
        staged_kwargs = _staged_encoded_kwargs(encoded_dense, device=device)
        meta: dict[str, Any] = dict(
            trace_kind_id=torch.tensor([1, 2], device=device),
            decision_count=torch.tensor([2, 4], device=device),
            decision_count_host=(2, 4),
            total_decision_groups=6,
            total_stored_decision_groups=5,
            decision_option_idx=torch.tensor(
                [
                    [0, 1, -1, -1],
                    [2, -1, -1, -1],
                    [1, -1, -1, -1],
                    [0, 2, -1, -1],
                    [3, -1, -1, -1],
                    [2, 1, -1, -1],
                ],
                device=device,
            ),
            decision_target_idx=torch.tensor(
                [
                    [-1, 0, -1, -1],
                    [1, -1, -1, -1],
                    [0, -1, -1, -1],
                    [-1, 1, -1, -1],
                    [2, -1, -1, -1],
                    [0, 2, -1, -1],
                ],
                device=device,
            ),
            decision_mask=torch.tensor(
                [
                    [True, True, False, False],
                    [True, False, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                ],
                device=device,
            ),
            uses_none_head=torch.tensor([False, True, True, False, True, False], device=device),
            selected_indices=torch.tensor([1, 0, 0, 1, 0, 1], device=device),
            may_selected=torch.tensor([0.0, 1.0], device=device),
            old_log_prob=torch.tensor([-0.25, -0.5], device=device),
            value=torch.tensor([0.75, -0.125], device=device),
            perspective_player_idx=torch.tensor([0, 1], device=device),
            lstm_h_in=torch.arange(12, dtype=torch.float32, device=device).reshape(1, 2, 6),
            lstm_c_in=torch.arange(100, 112, dtype=torch.float32, device=device).reshape(1, 2, 6),
        )
        staged_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            recurrent_layers=1,
            recurrent_hidden_dim=6,
            device=device,
            use_triton_append=False,
            use_triton_gather=False,
        )
        packed_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            recurrent_layers=1,
            recurrent_hidden_dim=6,
            device=device,
            use_triton_append=False,
            use_triton_gather=False,
        )

        rows_staged = staged_buffer.append_staged_batch(**staged_kwargs, **meta)
        rows_packed = packed_buffer.append_batch(encoded=encoded, **meta)
        _assert_replay_batch_close(
            self,
            staged_buffer.gather(rows_staged),
            packed_buffer.gather(rows_packed),
        )

    def test_unsealed_staged_batch_is_invisible_until_metadata_ready(self) -> None:
        device = torch.device("cpu")
        encoded_dense = _encoded_batch()
        staged_kwargs = _staged_encoded_kwargs(encoded_dense, device=device)
        meta: dict[str, Any] = dict(
            trace_kind_id=torch.tensor([1, 2], device=device),
            decision_count=torch.tensor([1, 1], device=device),
            decision_count_host=(1, 1),
            total_decision_groups=2,
            total_stored_decision_groups=2,
            decision_option_idx=torch.tensor(
                [[0, 1, -1, -1], [2, -1, -1, -1]],
                device=device,
            ),
            decision_target_idx=torch.tensor(
                [[-1, 0, -1, -1], [1, -1, -1, -1]],
                device=device,
            ),
            decision_mask=torch.tensor(
                [[True, True, False, False], [True, False, False, False]],
                device=device,
            ),
            uses_none_head=torch.tensor([False, True], device=device),
            selected_indices=torch.tensor([1, 0], device=device),
            may_selected=torch.tensor([0.0, 1.0], device=device),
            old_log_prob=torch.tensor([-0.25, -0.5], device=device),
            value=torch.tensor([0.75, -0.125], device=device),
            perspective_player_idx=torch.tensor([0, 1], device=device),
        )
        buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            device=device,
            use_triton_append=False,
            use_triton_gather=False,
        )

        rows = buffer.append_staged_batch(**staged_kwargs, **meta, seal=False)
        self.assertIsNone(buffer.claim_train_window(min_rows=1, max_rows=2, allow_partial=True))
        buffer.write_episode_metadata(
            rows,
            episode_id=10,
            terminal_reward_p0=1.0,
            zero_sum=True,
            actor_id=3,
            behavior_policy_version=4,
            inference_policy_version=5,
            target_policy_version=6,
        )
        self.assertIsNone(buffer.claim_train_window(min_rows=1, max_rows=2, allow_partial=True))

        buffer.seal_staged_rows(rows)
        window = buffer.claim_train_window(min_rows=2, max_rows=2)
        self.assertIsNotNone(window)
        assert window is not None
        torch.testing.assert_close(window.rows.cpu(), rows.cpu())
        buffer.release_train_window(window)

    def test_pending_ready_event_blocks_train_window_claim(self) -> None:
        class FakeReadyEvent:
            def __init__(self) -> None:
                self.ready = False

            def query(self) -> bool:
                return self.ready

        buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=2,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            use_triton_gather=False,
        )
        rows = buffer.append_batch(
            encoded=pack_batch(
                TextEncodedBatch(
                    token_ids=torch.tensor([[101, 0]]),
                    attention_mask=torch.tensor([[1, 0]]),
                    card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.long),
                    seq_lengths=torch.tensor([1]),
                    seq_lengths_host=(1,),
                )
            ),
            trace_kind_id=torch.tensor([1]),
            decision_count=torch.tensor([1]),
            decision_option_idx=torch.tensor([[0, -1]]),
            decision_target_idx=torch.tensor([[-1, -1]]),
            decision_mask=torch.tensor([[True, False]]),
            uses_none_head=torch.tensor([False]),
            selected_indices=torch.tensor([0]),
            may_selected=torch.tensor([0.0]),
            old_log_prob=torch.tensor([-0.1]),
            value=torch.tensor([0.0]),
            perspective_player_idx=torch.tensor([0]),
        )
        buffer.write_episode_metadata(
            rows,
            episode_id=11,
            terminal_reward_p0=0.0,
            zero_sum=True,
        )
        event = FakeReadyEvent()
        buffer._row_ready_events[int(rows[0])] = event

        self.assertIsNone(buffer.claim_train_window(min_rows=1, max_rows=1, allow_partial=True))
        event.ready = True
        window = buffer.claim_train_window(min_rows=1, max_rows=1, allow_partial=True)
        self.assertIsNotNone(window)
        assert window is not None
        torch.testing.assert_close(window.rows.cpu(), rows.cpu())
        buffer.release_train_window(window)

    @unittest.skipUnless(
        torch.cuda.is_available() and TRITON_AVAILABLE,
        "requires CUDA and Triton",
    )
    def test_append_staged_batch_triton_matches_torch_path(self) -> None:
        device = torch.device("cuda")
        encoded_dense = _encoded_batch()
        staged_kwargs = _staged_encoded_kwargs(encoded_dense, device=device)
        meta: dict[str, Any] = dict(
            trace_kind_id=torch.tensor([1, 2], device=device),
            decision_count=torch.tensor([1, 1], device=device),
            decision_count_host=(1, 1),
            total_decision_groups=2,
            total_stored_decision_groups=2,
            decision_option_idx=torch.tensor(
                [[0, 1, -1, -1], [2, -1, -1, -1]],
                device=device,
            ),
            decision_target_idx=torch.tensor(
                [[-1, 0, -1, -1], [1, -1, -1, -1]],
                device=device,
            ),
            decision_mask=torch.tensor(
                [[True, True, False, False], [True, False, False, False]],
                device=device,
            ),
            uses_none_head=torch.tensor([False, True], device=device),
            selected_indices=torch.tensor([1, 0], device=device),
            may_selected=torch.tensor([0.0, 1.0], device=device),
            old_log_prob=torch.tensor([-0.25, -0.5], device=device),
            value=torch.tensor([0.75, -0.125], device=device),
            perspective_player_idx=torch.tensor([0, 1], device=device),
        )
        triton_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            device=device,
            use_triton_append=True,
        )
        torch_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            device=device,
            use_triton_append=False,
            use_triton_gather=False,
        )

        rows_triton = triton_buffer.append_staged_batch(**staged_kwargs, **meta)
        rows_torch = torch_buffer.append_staged_batch(**staged_kwargs, **meta)
        _assert_replay_batch_close(
            self,
            triton_buffer.gather(_host_rows(rows_triton)),
            torch_buffer.gather(_host_rows(rows_torch)),
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and TRITON_AVAILABLE,
        "requires CUDA and Triton",
    )
    def test_gather_triton_matches_torch_path(self) -> None:
        device = torch.device("cuda")
        encoded = _packed_to_device(pack_batch(_encoded_batch()), device)
        kwargs: dict[str, Any] = dict(
            encoded=encoded,
            trace_kind_id=torch.tensor([1, 2], device=device),
            decision_count=torch.tensor([1, 1], device=device),
            decision_option_idx=torch.tensor(
                [[0, 1, -1, -1], [2, -1, -1, -1]],
                device=device,
            ),
            decision_target_idx=torch.tensor(
                [[-1, 0, -1, -1], [1, -1, -1, -1]],
                device=device,
            ),
            decision_mask=torch.tensor(
                [[True, True, False, False], [True, False, False, False]],
                device=device,
            ),
            uses_none_head=torch.tensor([False, True], device=device),
            selected_indices=torch.tensor([1, 0], device=device),
            may_selected=torch.tensor([0.0, 1.0], device=device),
            old_log_prob=torch.tensor([-0.25, -0.5], device=device),
            value=torch.tensor([0.75, -0.125], device=device),
            perspective_player_idx=torch.tensor([0, 1], device=device),
        )
        triton_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            device=device,
            use_triton_append=False,
            use_triton_gather=True,
        )
        torch_buffer = TextReplayBuffer(
            capacity=2,
            max_tokens=5,
            max_options=3,
            max_targets_per_option=2,
            max_decision_groups=3,
            max_cached_choices=4,
            device=device,
            use_triton_append=False,
            use_triton_gather=False,
        )

        rows_triton = triton_buffer.append_batch(**kwargs)
        rows_torch = torch_buffer.append_batch(**kwargs)
        _assert_replay_batch_close(
            self,
            triton_buffer.gather(_host_rows(rows_triton.flip(0))),
            torch_buffer.gather(_host_rows(rows_torch.flip(0))),
        )


if __name__ == "__main__":
    unittest.main()
