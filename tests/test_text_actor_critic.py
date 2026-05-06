import unittest
from types import SimpleNamespace
from typing import cast

import torch
from magic_ai.actions import TRACE_KIND_TO_ID
from magic_ai.game_state import PendingState
from magic_ai.text_encoder.actor_critic import (
    TextActorCritic,
    TextDecisionLayout,
    _evaluate_inline_blocker_replay_groups,
    _evaluate_inline_choice_index_replay_groups,
    _evaluate_inline_priority_replay_groups,
    _sample_inline_blockers_for_step,
    _sample_inline_choice_index_for_step,
    _sample_inline_may_for_step,
    _sample_inline_priority_for_step,
    _score_inline_may_decisions,
    build_text_decision_layout,
    infer_text_trace_kind,
)
from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.recurrent import RecurrentTextPolicyConfig, RecurrentTextPolicyOutput
from magic_ai.text_encoder.render_plan import BLANK_GROUP_CROSS_BLANK, BLANK_GROUP_PER_BLANK
from magic_ai.text_encoder.replay_buffer import TextReplayBatch, TextReplayBuffer
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


def _batch(batch_size: int = 2) -> TextEncodedBatch:
    token_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])[:batch_size]
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])[:batch_size]
    card_ref_positions = torch.full((batch_size, MAX_CARD_REFS), -1, dtype=torch.long)
    blank_positions = torch.full((batch_size, 2), -1, dtype=torch.long)
    blank_positions[:, 0] = 1
    if batch_size > 1:
        blank_positions[0, 1] = 2
    else:
        blank_positions[0, 1] = 2
    blank_kind = torch.full((batch_size, 2), 7, dtype=torch.long)
    blank_group = torch.full((batch_size, 2), -1, dtype=torch.long)
    blank_group[:, 0] = 0
    blank_group[0, 1] = 0
    blank_group_kind = torch.full((batch_size, 2), BLANK_GROUP_CROSS_BLANK, dtype=torch.long)
    blank_option_index = torch.full((batch_size, 2), -1, dtype=torch.long)
    blank_option_index[:, 0] = 0
    blank_option_index[0, 1] = 1
    blank_legal_ids = torch.full((batch_size, 2, 1), 3, dtype=torch.long)
    blank_legal_mask = blank_positions.unsqueeze(-1) >= 0
    seq_lengths = torch.tensor([4, 3])[:batch_size]
    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        blank_positions=blank_positions,
        blank_kind=blank_kind,
        blank_group=blank_group,
        blank_group_kind=blank_group_kind,
        blank_option_index=blank_option_index,
        blank_legal_ids=blank_legal_ids,
        blank_legal_mask=blank_legal_mask,
        seq_lengths=seq_lengths,
    )


def _model() -> TextActorCritic:
    cfg = RecurrentTextPolicyConfig(
        encoder=TextEncoderConfig(
            vocab_size=16,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_seq_len=8,
            pad_id=0,
        ),
        lstm_hidden=8,
        lstm_layers=1,
    )
    return TextActorCritic(cfg)


class TextActorCriticTests(unittest.TestCase):
    def test_live_state_gather_scatter_is_per_env_and_player(self) -> None:
        model = _model()
        model.init_lstm_env_states(2)
        h, c = model.lstm_env_state_inputs([0, 1], [1, 0])
        self.assertEqual(tuple(h.shape), (1, 2, 8))
        self.assertEqual(tuple(c.shape), (1, 2, 8))

        h_out = torch.ones(1, 2, 8)
        c_out = torch.full((1, 2, 8), 2.0)
        model.scatter_lstm_env_states([0, 1], [1, 0], h_out, c_out)

        self.assertTrue(torch.equal(model.live_lstm_h[:, 1], h_out[:, 0]))
        self.assertTrue(torch.equal(model.live_lstm_h[:, 2], h_out[:, 1]))
        self.assertTrue(torch.equal(model.live_lstm_c[:, 1], c_out[:, 0]))
        self.assertTrue(torch.equal(model.live_lstm_c[:, 2], c_out[:, 1]))

        model.reset_lstm_env_states([0])
        self.assertTrue(torch.equal(model.live_lstm_h[:, 0], torch.zeros(1, 8)))
        self.assertTrue(torch.equal(model.live_lstm_h[:, 1], torch.zeros(1, 8)))
        self.assertTrue(torch.equal(model.live_lstm_h[:, 2], h_out[:, 1]))

    def test_forward_live_updates_selected_player_states(self) -> None:
        torch.manual_seed(0)
        model = _model()
        model.init_lstm_env_states(2)

        step = model.forward_live(
            _batch(),
            env_indices=[0, 1],
            perspective_player_indices=[0, 1],
        )

        self.assertEqual(tuple(step.output.values.shape), (2,))
        self.assertFalse(torch.equal(model.live_lstm_h[:, 0], torch.zeros(1, 8)))
        self.assertFalse(torch.equal(model.live_lstm_h[:, 3], torch.zeros(1, 8)))
        self.assertTrue(torch.equal(model.live_lstm_h[:, 1], torch.zeros(1, 8)))
        self.assertTrue(torch.equal(model.live_lstm_h[:, 2], torch.zeros(1, 8)))

    def test_evaluate_replay_batch_produces_finite_ppo_tensors(self) -> None:
        torch.manual_seed(0)
        model = _model()
        replay = TextReplayBuffer(
            capacity=4,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        row = replay.append(
            encoded=_batch(batch_size=1),
            batch_index=0,
            trace_kind_id=0,
            decision_option_idx=torch.tensor([[0, 1]]),
            decision_target_idx=torch.tensor([[-1, -1]]),
            decision_mask=torch.tensor([[True, True]]),
            uses_none_head=torch.tensor([False]),
            selected_indices=torch.tensor([0]),
            may_selected=0.0,
            old_log_prob=-0.5,
            value=0.1,
            perspective_player_idx=0,
            lstm_h_in=torch.zeros(1, 8),
            lstm_c_in=torch.zeros(1, 8),
        )

        log_probs, entropies, values, extras = model.evaluate_replay_batch([row])
        loss = -(log_probs + 0.01 * entropies).mean() + values.square().mean()
        loss.backward()

        self.assertIsNone(extras)
        self.assertEqual(tuple(log_probs.shape), (1,))
        self.assertEqual(tuple(entropies.shape), (1,))
        self.assertEqual(tuple(values.shape), (1,))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(entropies).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_sample_text_batch_appends_replay_row(self) -> None:
        torch.manual_seed(0)
        model = _model()
        replay = TextReplayBuffer(
            capacity=4,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        model.init_lstm_env_states(1)
        pending = cast(
            PendingState,
            {
                "kind": "priority",
                "player_idx": 0,
                "options": [
                    {"kind": "pass"},
                    {"kind": "play_land", "card_id": "c1"},
                ],
            },
        )
        layout = build_text_decision_layout(
            infer_text_trace_kind(pending),
            pending,
            max_options=2,
            max_targets_per_option=1,
            max_cached_choices=2,
        )

        steps = model.sample_text_batch(
            _batch(batch_size=1),
            env_indices=[0],
            perspective_player_indices=[0],
            layouts=[layout],
            deterministic=True,
        )

        self.assertEqual(len(steps), 1)
        replay_idx = steps[0].replay_idx
        self.assertIsNotNone(replay_idx)
        assert replay_idx is not None
        self.assertEqual(replay.size, 1)
        self.assertEqual(len(steps[0].selected_choice_cols), 1)
        log_probs, entropies, values, extras = model.evaluate_replay_batch([replay_idx])
        self.assertIsNone(extras)
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(entropies).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_inline_blocker_sampler_uses_blank_option_indices(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[0.0, 5.0], [4.0, 0.0]]]),
        )
        batch = TextEncodedBatch(
            token_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.long),
            seq_lengths=torch.tensor([3]),
            blank_group_kind=torch.full((1, 2), BLANK_GROUP_PER_BLANK, dtype=torch.int32),
            blank_option_index=torch.tensor([[1, 0]], dtype=torch.int32),
            blank_legal_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        )
        layout = TextDecisionLayout(
            trace_kind="blockers",
            decision_option_idx=torch.tensor([[-1, 0], [-1, 1]]),
            decision_target_idx=torch.tensor([[-1, 0], [-1, 0]]),
            decision_mask=torch.ones(2, 2, dtype=torch.bool),
            uses_none_head=torch.ones(2, dtype=torch.bool),
            pending=cast(PendingState, {"kind": "blockers", "player_idx": 0, "options": []}),
        )

        sampled = _sample_inline_blockers_for_step(
            output,
            batch,
            layout,
            step_idx=0,
            deterministic=True,
        )

        assert sampled is not None
        selected, log_prob, entropy = sampled
        self.assertEqual([int(t.item()) for t in selected], [0, 1])
        self.assertTrue(torch.isfinite(log_prob))
        self.assertTrue(torch.isfinite(entropy))

    def test_inline_blocker_sampler_handles_none_only_row(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[3.0, 0.0]]]),
        )
        batch = TextEncodedBatch(
            token_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.long),
            seq_lengths=torch.tensor([3]),
            blank_group_kind=torch.full((1, 1), BLANK_GROUP_PER_BLANK, dtype=torch.int32),
            blank_option_index=torch.tensor([[0]], dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, False]]]),
        )
        layout = TextDecisionLayout(
            trace_kind="blockers",
            decision_option_idx=torch.tensor([[-1, -1]]),
            decision_target_idx=torch.tensor([[-1, -1]]),
            decision_mask=torch.tensor([[True, False]]),
            uses_none_head=torch.ones(1, dtype=torch.bool),
            pending=cast(PendingState, {"kind": "blockers", "player_idx": 0, "options": []}),
        )

        sampled = _sample_inline_blockers_for_step(
            output,
            batch,
            layout,
            step_idx=0,
            deterministic=True,
        )

        assert sampled is not None
        selected, log_prob, entropy = sampled
        self.assertEqual([int(t.item()) for t in selected], [0])
        self.assertTrue(torch.isfinite(log_prob))
        self.assertTrue(torch.isfinite(entropy))

    def test_inline_priority_sampler_maps_target_blank_to_candidate_column(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[5.0, 0.0], [0.0, 4.0], [0.0, 1.0]]]),
        )
        batch = TextEncodedBatch(
            token_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.long),
            seq_lengths=torch.tensor([3]),
            blank_group_kind=torch.tensor(
                [[BLANK_GROUP_CROSS_BLANK, BLANK_GROUP_PER_BLANK, BLANK_GROUP_CROSS_BLANK]],
                dtype=torch.int32,
            ),
            blank_option_index=torch.tensor([[0, 0, 1]], dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, False], [True, True], [True, False]]]),
        )
        layout = TextDecisionLayout(
            trace_kind="priority",
            decision_option_idx=torch.tensor([[0, 0, 1]]),
            decision_target_idx=torch.tensor([[0, 1, -1]]),
            decision_mask=torch.ones(1, 3, dtype=torch.bool),
            uses_none_head=torch.zeros(1, dtype=torch.bool),
            pending=cast(PendingState, {"kind": "priority", "player_idx": 0, "options": []}),
        )

        sampled = _sample_inline_priority_for_step(
            output,
            batch,
            layout,
            step_idx=0,
            deterministic=True,
        )

        assert sampled is not None
        selected, log_prob, entropy = sampled
        self.assertEqual([int(t.item()) for t in selected], [1])
        self.assertTrue(torch.isfinite(log_prob))
        self.assertTrue(torch.isfinite(entropy))

    def test_inline_may_sampler_uses_yes_no_blank(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[0.0, 3.0]]]),
        )
        batch = TextEncodedBatch(
            token_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.long),
            seq_lengths=torch.tensor([1]),
            blank_group_kind=torch.tensor([[BLANK_GROUP_PER_BLANK]], dtype=torch.int32),
            blank_option_index=torch.tensor([[-1]], dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, True]]]),
        )

        sampled = _sample_inline_may_for_step(
            output,
            batch,
            step_idx=0,
            deterministic=True,
        )

        assert sampled is not None
        may_selected, log_prob, entropy = sampled
        self.assertEqual(int(may_selected.item()), 1)
        self.assertTrue(torch.isfinite(log_prob))
        self.assertTrue(torch.isfinite(entropy))

    def test_inline_choice_index_sampler_uses_mode_blank(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[0.0, 4.0, 1.0]]]),
        )
        batch = TextEncodedBatch(
            token_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.long),
            seq_lengths=torch.tensor([1]),
            blank_group_kind=torch.tensor([[BLANK_GROUP_PER_BLANK]], dtype=torch.int32),
            blank_option_index=torch.tensor([[-1]], dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, True, True]]]),
        )

        sampled = _sample_inline_choice_index_for_step(
            output,
            batch,
            step_idx=0,
            deterministic=True,
        )

        assert sampled is not None
        selected, log_prob, entropy = sampled
        self.assertEqual([int(t.item()) for t in selected], [1])
        self.assertTrue(torch.isfinite(log_prob))
        self.assertTrue(torch.isfinite(entropy))

    def test_inline_blocker_replay_scoring_uses_blank_logits(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[1.0, 0.0, 4.0], [0.0, 3.0, -2.0]]]),
        )
        encoded = PackedTextBatch(
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            seq_id=torch.tensor([0, 0, 0], dtype=torch.int32),
            pos_in_seq=torch.tensor([0, 1, 2], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            seq_lengths=torch.tensor([3], dtype=torch.int32),
            state_positions=torch.tensor([0], dtype=torch.int32),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.int32),
            blank_positions=torch.tensor([[1, 2]], dtype=torch.int32),
            blank_kind=torch.ones(1, 2, dtype=torch.int32),
            blank_group=torch.zeros(1, 2, dtype=torch.int32),
            blank_group_kind=torch.full((1, 2), BLANK_GROUP_PER_BLANK, dtype=torch.int32),
            blank_option_index=torch.tensor([[1, 0]], dtype=torch.int32),
            blank_legal_ids=torch.ones(1, 2, 3, dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, True, True], [True, True, False]]]),
        )
        batch = TextReplayBatch(
            encoded=encoded,
            trace_kind_id=torch.tensor([TRACE_KIND_TO_ID["blockers"]]),
            decision_start=torch.tensor([0]),
            decision_count=torch.tensor([2]),
            decision_option_idx=torch.tensor([[-1, 0], [-1, 1]]),
            decision_target_idx=torch.tensor([[-1, 0], [-1, 0]]),
            decision_mask=torch.ones(2, 2, dtype=torch.bool),
            uses_none_head=torch.ones(2, dtype=torch.bool),
            selected_indices=torch.tensor([1, 2]),
            step_for_decision_group=torch.tensor([0, 0]),
            may_selected=torch.zeros(1),
            old_log_prob=torch.zeros(1),
            value=torch.zeros(1),
            perspective_player_idx=torch.zeros(1, dtype=torch.long),
            lstm_h_in=None,
            lstm_c_in=None,
        )

        log_probs, entropies, group_mask, per_choice = _evaluate_inline_blocker_replay_groups(
            output,
            batch,
            return_per_choice=True,
        )

        blank_logits = output.blank_logits
        assert blank_logits is not None
        expected = torch.log_softmax(blank_logits[0, 1, :2], dim=0)[1]
        expected = expected + torch.log_softmax(blank_logits[0, 0], dim=0)[2]
        torch.testing.assert_close(log_probs[0], expected)
        self.assertTrue(torch.isfinite(entropies).all())
        torch.testing.assert_close(group_mask, torch.tensor([True, True]))
        assert per_choice is not None
        self.assertEqual(tuple(per_choice.flat_logits.shape), (5,))
        self.assertEqual(int(per_choice.is_sampled_flat.sum()), 2)

    def test_inline_blocker_replay_scoring_handles_none_only_row(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[2.0, -1.0]]]),
        )
        encoded = PackedTextBatch(
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            seq_id=torch.tensor([0, 0, 0], dtype=torch.int32),
            pos_in_seq=torch.tensor([0, 1, 2], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            seq_lengths=torch.tensor([3], dtype=torch.int32),
            state_positions=torch.tensor([0], dtype=torch.int32),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.int32),
            blank_positions=torch.tensor([[1]], dtype=torch.int32),
            blank_kind=torch.ones(1, 1, dtype=torch.int32),
            blank_group=torch.zeros(1, 1, dtype=torch.int32),
            blank_group_kind=torch.full((1, 1), BLANK_GROUP_PER_BLANK, dtype=torch.int32),
            blank_option_index=torch.tensor([[0]], dtype=torch.int32),
            blank_legal_ids=torch.ones(1, 1, 2, dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, False]]]),
        )
        batch = TextReplayBatch(
            encoded=encoded,
            trace_kind_id=torch.tensor([TRACE_KIND_TO_ID["blockers"]]),
            decision_start=torch.tensor([0]),
            decision_count=torch.tensor([1]),
            decision_option_idx=torch.tensor([[-1, -1]]),
            decision_target_idx=torch.tensor([[-1, -1]]),
            decision_mask=torch.tensor([[True, False]]),
            uses_none_head=torch.ones(1, dtype=torch.bool),
            selected_indices=torch.tensor([0]),
            step_for_decision_group=torch.tensor([0]),
            may_selected=torch.zeros(1),
            old_log_prob=torch.zeros(1),
            value=torch.zeros(1),
            perspective_player_idx=torch.zeros(1, dtype=torch.long),
            lstm_h_in=None,
            lstm_c_in=None,
        )

        log_probs, entropies, group_mask, per_choice = _evaluate_inline_blocker_replay_groups(
            output,
            batch,
            return_per_choice=True,
        )

        blank_logits = output.blank_logits
        assert blank_logits is not None
        expected = torch.log_softmax(blank_logits[0, 0, :1], dim=0)[0]
        torch.testing.assert_close(log_probs[0], expected)
        self.assertTrue(torch.isfinite(entropies).all())
        torch.testing.assert_close(group_mask, torch.tensor([True]))
        assert per_choice is not None
        self.assertEqual(tuple(per_choice.flat_logits.shape), (1,))
        self.assertEqual(int(per_choice.is_sampled_flat.sum()), 1)

    def test_inline_priority_replay_scoring_uses_target_blank_logits(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[5.0, 0.0], [0.0, 4.0], [0.0, 1.0]]]),
        )
        encoded = PackedTextBatch(
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            seq_id=torch.tensor([0, 0, 0], dtype=torch.int32),
            pos_in_seq=torch.tensor([0, 1, 2], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            seq_lengths=torch.tensor([3], dtype=torch.int32),
            state_positions=torch.tensor([0], dtype=torch.int32),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.int32),
            blank_positions=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            blank_kind=torch.ones(1, 3, dtype=torch.int32),
            blank_group=torch.zeros(1, 3, dtype=torch.int32),
            blank_group_kind=torch.tensor(
                [[BLANK_GROUP_CROSS_BLANK, BLANK_GROUP_PER_BLANK, BLANK_GROUP_CROSS_BLANK]],
                dtype=torch.int32,
            ),
            blank_option_index=torch.tensor([[0, 0, 1]], dtype=torch.int32),
            blank_legal_ids=torch.ones(1, 3, 2, dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, False], [True, True], [True, False]]]),
        )
        batch = TextReplayBatch(
            encoded=encoded,
            trace_kind_id=torch.tensor([TRACE_KIND_TO_ID["priority"]]),
            decision_start=torch.tensor([0]),
            decision_count=torch.tensor([1]),
            decision_option_idx=torch.tensor([[0, 0, 1]]),
            decision_target_idx=torch.tensor([[0, 1, -1]]),
            decision_mask=torch.ones(1, 3, dtype=torch.bool),
            uses_none_head=torch.zeros(1, dtype=torch.bool),
            selected_indices=torch.tensor([1]),
            step_for_decision_group=torch.tensor([0]),
            may_selected=torch.zeros(1),
            old_log_prob=torch.zeros(1),
            value=torch.zeros(1),
            perspective_player_idx=torch.zeros(1, dtype=torch.long),
            lstm_h_in=None,
            lstm_c_in=None,
        )

        log_probs, entropies, group_mask, per_choice = _evaluate_inline_priority_replay_groups(
            output,
            batch,
            return_per_choice=True,
        )

        blank_logits = output.blank_logits
        assert blank_logits is not None
        expected = torch.log_softmax(blank_logits[0, [0, 2], 0], dim=0)[0]
        expected = expected + torch.log_softmax(blank_logits[0, 1], dim=0)[1]
        torch.testing.assert_close(log_probs[0], expected)
        self.assertTrue(torch.isfinite(entropies).all())
        torch.testing.assert_close(group_mask, torch.tensor([True]))
        assert per_choice is not None
        self.assertEqual(tuple(per_choice.flat_logits.shape), (3,))
        self.assertEqual(int(per_choice.is_sampled_flat.sum()), 1)

    def test_inline_may_replay_scoring_uses_yes_no_blank(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[0.0, 3.0]]]),
        )
        encoded = PackedTextBatch(
            token_ids=torch.tensor([1], dtype=torch.int32),
            seq_id=torch.tensor([0], dtype=torch.int32),
            pos_in_seq=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
            seq_lengths=torch.tensor([1], dtype=torch.int32),
            state_positions=torch.tensor([0], dtype=torch.int32),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.int32),
            blank_positions=torch.tensor([[0]], dtype=torch.int32),
            blank_kind=torch.ones(1, 1, dtype=torch.int32),
            blank_group=torch.zeros(1, 1, dtype=torch.int32),
            blank_group_kind=torch.tensor([[BLANK_GROUP_PER_BLANK]], dtype=torch.int32),
            blank_option_index=torch.tensor([[-1]], dtype=torch.int32),
            blank_legal_ids=torch.ones(1, 1, 2, dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, True]]]),
        )
        batch = TextReplayBatch(
            encoded=encoded,
            trace_kind_id=torch.tensor([TRACE_KIND_TO_ID["may"]]),
            decision_start=torch.tensor([0]),
            decision_count=torch.tensor([0]),
            decision_option_idx=torch.empty(0, 2, dtype=torch.long),
            decision_target_idx=torch.empty(0, 2, dtype=torch.long),
            decision_mask=torch.empty(0, 2, dtype=torch.bool),
            uses_none_head=torch.empty(0, dtype=torch.bool),
            selected_indices=torch.empty(0, dtype=torch.long),
            step_for_decision_group=torch.empty(0, dtype=torch.long),
            may_selected=torch.ones(1),
            old_log_prob=torch.zeros(1),
            value=torch.zeros(1),
            perspective_player_idx=torch.zeros(1, dtype=torch.long),
            lstm_h_in=None,
            lstm_c_in=None,
        )

        log_probs, entropies, active, logits_per_step, selected_per_step = (
            _score_inline_may_decisions(output, batch)
        )

        blank_logits = output.blank_logits
        assert blank_logits is not None
        expected = torch.log_softmax(blank_logits[0, 0], dim=0)[1]
        torch.testing.assert_close(log_probs[0], expected)
        self.assertTrue(torch.isfinite(entropies).all())
        torch.testing.assert_close(active, torch.tensor([True]))
        torch.testing.assert_close(logits_per_step, torch.tensor([3.0]))
        torch.testing.assert_close(selected_per_step, torch.tensor([1.0]))

    def test_inline_choice_index_replay_scoring_uses_mode_blank(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[0.0, 4.0, 1.0]]]),
        )
        encoded = PackedTextBatch(
            token_ids=torch.tensor([1], dtype=torch.int32),
            seq_id=torch.tensor([0], dtype=torch.int32),
            pos_in_seq=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
            seq_lengths=torch.tensor([1], dtype=torch.int32),
            state_positions=torch.tensor([0], dtype=torch.int32),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.int32),
            blank_positions=torch.tensor([[0]], dtype=torch.int32),
            blank_kind=torch.ones(1, 1, dtype=torch.int32),
            blank_group=torch.zeros(1, 1, dtype=torch.int32),
            blank_group_kind=torch.tensor([[BLANK_GROUP_PER_BLANK]], dtype=torch.int32),
            blank_option_index=torch.tensor([[-1]], dtype=torch.int32),
            blank_legal_ids=torch.ones(1, 1, 3, dtype=torch.int32),
            blank_legal_mask=torch.tensor([[[True, True, True]]]),
        )
        batch = TextReplayBatch(
            encoded=encoded,
            trace_kind_id=torch.tensor([TRACE_KIND_TO_ID["choice_index"]]),
            decision_start=torch.tensor([0]),
            decision_count=torch.tensor([1]),
            decision_option_idx=torch.tensor([[0, 1, 2, -1]]),
            decision_target_idx=torch.tensor([[-1, -1, -1, -1]]),
            decision_mask=torch.tensor([[True, True, True, False]]),
            uses_none_head=torch.zeros(1, dtype=torch.bool),
            selected_indices=torch.tensor([1]),
            step_for_decision_group=torch.tensor([0]),
            may_selected=torch.zeros(1),
            old_log_prob=torch.zeros(1),
            value=torch.zeros(1),
            perspective_player_idx=torch.zeros(1, dtype=torch.long),
            lstm_h_in=None,
            lstm_c_in=None,
        )

        log_probs, entropies, group_mask, per_choice = _evaluate_inline_choice_index_replay_groups(
            output,
            batch,
            return_per_choice=True,
            trace_kind_id=TRACE_KIND_TO_ID["choice_index"],
        )

        blank_logits = output.blank_logits
        assert blank_logits is not None
        expected = torch.log_softmax(blank_logits[0, 0], dim=0)[1]
        torch.testing.assert_close(log_probs[0], expected)
        self.assertTrue(torch.isfinite(entropies).all())
        torch.testing.assert_close(group_mask, torch.tensor([True]))
        assert per_choice is not None
        self.assertEqual(tuple(per_choice.flat_logits.shape), (3,))
        self.assertEqual(int(per_choice.is_sampled_flat.sum()), 1)

    def test_inline_choice_color_replay_scoring_uses_mana_blank(self) -> None:
        output = RecurrentTextPolicyOutput(
            values=torch.zeros(1),
            state_hidden=torch.zeros(1, 8),
            card_vectors=torch.empty(1, MAX_CARD_REFS, 8),
            card_mask=torch.zeros(1, MAX_CARD_REFS, dtype=torch.bool),
            blank_logits=torch.tensor([[[0.0, 1.0, 2.0, 5.0, 3.0, 4.0]]]),
        )
        encoded = PackedTextBatch(
            token_ids=torch.tensor([1], dtype=torch.int32),
            seq_id=torch.tensor([0], dtype=torch.int32),
            pos_in_seq=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
            seq_lengths=torch.tensor([1], dtype=torch.int32),
            state_positions=torch.tensor([0], dtype=torch.int32),
            card_ref_positions=torch.full((1, MAX_CARD_REFS), -1, dtype=torch.int32),
            blank_positions=torch.tensor([[0]], dtype=torch.int32),
            blank_kind=torch.ones(1, 1, dtype=torch.int32),
            blank_group=torch.zeros(1, 1, dtype=torch.int32),
            blank_group_kind=torch.tensor([[BLANK_GROUP_PER_BLANK]], dtype=torch.int32),
            blank_option_index=torch.tensor([[-1]], dtype=torch.int32),
            blank_legal_ids=torch.ones(1, 1, 6, dtype=torch.int32),
            blank_legal_mask=torch.ones(1, 1, 6, dtype=torch.bool),
        )
        batch = TextReplayBatch(
            encoded=encoded,
            trace_kind_id=torch.tensor([TRACE_KIND_TO_ID["choice_color"]]),
            decision_start=torch.tensor([0]),
            decision_count=torch.tensor([1]),
            decision_option_idx=torch.tensor([[0, 1, 2, 3, 4, 5]]),
            decision_target_idx=torch.full((1, 6), -1),
            decision_mask=torch.ones(1, 6, dtype=torch.bool),
            uses_none_head=torch.zeros(1, dtype=torch.bool),
            selected_indices=torch.tensor([3]),
            step_for_decision_group=torch.tensor([0]),
            may_selected=torch.zeros(1),
            old_log_prob=torch.zeros(1),
            value=torch.zeros(1),
            perspective_player_idx=torch.zeros(1, dtype=torch.long),
            lstm_h_in=None,
            lstm_c_in=None,
        )

        log_probs, entropies, group_mask, per_choice = _evaluate_inline_choice_index_replay_groups(
            output,
            batch,
            return_per_choice=True,
            trace_kind_id=TRACE_KIND_TO_ID["choice_color"],
        )

        blank_logits = output.blank_logits
        assert blank_logits is not None
        expected = torch.log_softmax(blank_logits[0, 0], dim=0)[3]
        torch.testing.assert_close(log_probs[0], expected)
        self.assertTrue(torch.isfinite(entropies).all())
        torch.testing.assert_close(group_mask, torch.tensor([True]))
        assert per_choice is not None
        self.assertEqual(tuple(per_choice.flat_logits.shape), (6,))
        self.assertEqual(int(per_choice.is_sampled_flat.sum()), 1)

    def test_sample_native_tensor_batch_appends_replay_rows(self) -> None:
        torch.manual_seed(0)
        model = _model()
        replay = TextReplayBuffer(
            capacity=4,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        model.init_lstm_env_states(2)
        native_batch = SimpleNamespace(
            trace_kind_id=torch.tensor([0, 0], dtype=torch.long),
            decision_count=torch.tensor([1, 1], dtype=torch.long),
            decision_rows_written=2,
            decision_option_idx=torch.tensor([[0, 1], [0, -1]], dtype=torch.long),
            decision_target_idx=torch.tensor([[-1, -1], [-1, -1]], dtype=torch.long),
            decision_mask=torch.tensor([[True, True], [True, False]], dtype=torch.bool),
            uses_none_head=torch.tensor([False, False], dtype=torch.bool),
        )

        out = model.sample_native_tensor_batch(
            native_batch=native_batch,
            env_indices=[0, 1],
            perspective_player_indices=[0, 0],
            text_batch=_batch(batch_size=2),
            deterministic=True,
        )

        self.assertEqual(out.decision_counts, [1, 1])
        self.assertEqual(len(out.selected_choice_cols), 2)
        self.assertEqual(len(out.replay_rows), 2)
        self.assertEqual(replay.size, 2)
        gathered = replay.gather(out.replay_rows)
        self.assertEqual(tuple(gathered.encoded.token_ids.shape), (7,))
        self.assertEqual(tuple(gathered.encoded.seq_lengths.tolist()), (4, 3))
        self.assertTrue(torch.isfinite(gathered.old_log_prob).all())

    def test_sample_native_tensor_batch_without_replay_buffer_for_eval(self) -> None:
        torch.manual_seed(0)
        model = _model()
        model.rollout_buffer = None
        model.init_lstm_env_states(2)
        native_batch = SimpleNamespace(
            trace_kind_id=torch.tensor([0, 0], dtype=torch.long),
            decision_count=torch.tensor([1, 1], dtype=torch.long),
            decision_rows_written=2,
            decision_option_idx=torch.tensor([[0, 1], [0, -1]], dtype=torch.long),
            decision_target_idx=torch.tensor([[-1, -1], [-1, -1]], dtype=torch.long),
            decision_mask=torch.tensor([[True, True], [True, False]], dtype=torch.bool),
            uses_none_head=torch.tensor([False, False], dtype=torch.bool),
        )

        out = model.sample_native_tensor_batch(
            native_batch=native_batch,
            env_indices=[0, 1],
            perspective_player_indices=[0, 0],
            text_batch=_batch(batch_size=2),
            deterministic=True,
        )

        self.assertEqual(out.decision_counts, [1, 1])
        self.assertEqual(len(out.selected_choice_cols), 2)
        self.assertEqual(out.replay_rows, [-1, -1])
        self.assertFalse(torch.equal(model.live_lstm_h[:, 0], torch.zeros(1, 8)))
        self.assertFalse(torch.equal(model.live_lstm_h[:, 2], torch.zeros(1, 8)))

    def test_sample_text_batch_handles_may_head(self) -> None:
        torch.manual_seed(0)
        model = _model()
        replay = TextReplayBuffer(
            capacity=4,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        model.init_lstm_env_states(1)
        pending = cast(PendingState, {"kind": "may", "player_idx": 0, "options": []})
        layout = build_text_decision_layout(
            "may",
            pending,
            max_options=2,
            max_targets_per_option=1,
            max_cached_choices=2,
        )

        steps = model.sample_text_batch(
            _batch(batch_size=1),
            env_indices=[0],
            perspective_player_indices=[0],
            layouts=[layout],
            deterministic=True,
        )

        self.assertEqual(steps[0].trace.kind, "may")
        replay_idx = steps[0].replay_idx
        self.assertIsNotNone(replay_idx)
        assert replay_idx is not None
        batch = replay.gather([replay_idx])
        self.assertEqual(int(batch.trace_kind_id[0]), 6)
        self.assertEqual(int(batch.decision_count[0]), 0)
        self.assertIn(steps[0].may_selected, (0, 1))

    def test_evaluate_replay_batch_per_choice_returns_rnad_shape(self) -> None:
        torch.manual_seed(0)
        model = _model()
        replay = TextReplayBuffer(
            capacity=4,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        row = replay.append(
            encoded=_batch(batch_size=1),
            batch_index=0,
            trace_kind_id=0,
            decision_option_idx=torch.tensor([[0, 1]]),
            decision_target_idx=torch.tensor([[-1, -1]]),
            decision_mask=torch.tensor([[True, True]]),
            uses_none_head=torch.tensor([False]),
            selected_indices=torch.tensor([1]),
            may_selected=0.0,
            old_log_prob=-0.5,
            value=0.1,
            perspective_player_idx=0,
            lstm_h_in=torch.zeros(1, 8),
            lstm_c_in=torch.zeros(1, 8),
        )

        log_probs, entropies, values, per_choice = model.evaluate_replay_batch_per_choice([row])

        self.assertEqual(tuple(log_probs.shape), (1,))
        self.assertEqual(tuple(entropies.shape), (1,))
        self.assertEqual(tuple(values.shape), (1,))
        self.assertEqual(tuple(per_choice.flat_logits.shape), (2,))
        self.assertEqual(tuple(per_choice.flat_log_probs.shape), (2,))
        torch.testing.assert_close(per_choice.group_idx, torch.tensor([0, 0]))
        torch.testing.assert_close(per_choice.choice_cols, torch.tensor([0, 1]))
        torch.testing.assert_close(per_choice.is_sampled_flat, torch.tensor([False, True]))
        torch.testing.assert_close(per_choice.decision_group_id_flat, torch.tensor([0, 0]))
        torch.testing.assert_close(per_choice.step_for_decision_group, torch.tensor([0]))
        self.assertFalse(bool(per_choice.may_is_active[0]))

    def test_recompute_lstm_states_and_outputs_for_rnad(self) -> None:
        torch.manual_seed(0)
        model = _model()
        replay = TextReplayBuffer(
            capacity=4,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        rows = []
        for batch_index in (0, 1):
            rows.append(
                replay.append(
                    encoded=_batch(batch_size=2),
                    batch_index=batch_index,
                    trace_kind_id=0,
                    decision_option_idx=torch.tensor([[0, 1]]),
                    decision_target_idx=torch.tensor([[-1, -1]]),
                    decision_mask=torch.tensor([[True, False]]),
                    uses_none_head=torch.tensor([False]),
                    selected_indices=torch.tensor([0]),
                    may_selected=0.0,
                    old_log_prob=-0.5,
                    value=0.1,
                    perspective_player_idx=0,
                    lstm_h_in=torch.zeros(1, 8),
                    lstm_c_in=torch.zeros(1, 8),
                )
            )

        states = model.recompute_lstm_states_for_episodes([rows])
        outputs = model.recompute_lstm_outputs_for_episodes([rows])
        log_probs, entropies, values, _per_choice = model.evaluate_replay_batch_per_choice(
            rows,
            lstm_state_override=states[0],
        )

        self.assertEqual(len(states), 1)
        self.assertEqual(tuple(states[0][0].shape), (1, 2, 8))
        self.assertEqual(tuple(states[0][1].shape), (1, 2, 8))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(tuple(outputs[0].shape), (2, 8))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(entropies).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_rejects_mismatched_env_and_player_lists(self) -> None:
        model = _model()
        model.init_lstm_env_states(1)

        with self.assertRaisesRegex(ValueError, "equal length"):
            model.lstm_env_state_inputs([0], [0, 1])

    def test_clone_for_rnad_shares_replay_buffer(self) -> None:
        model = _model()
        replay = TextReplayBuffer(
            capacity=4,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        model.init_lstm_env_states(2)

        clone = model.clone_for_rnad()
        self.assertIs(clone.rollout_buffer, replay)
        self.assertEqual(tuple(clone.live_lstm_h.shape), (1, 0, 8))
        self.assertEqual(tuple(clone.live_lstm_c.shape), (1, 0, 8))
        # Parameter independence.
        with torch.no_grad():
            clone.none_head.weight.fill_(0.5)
        self.assertFalse(torch.equal(model.none_head.weight, clone.none_head.weight))

    def test_run_rnad_update_on_text_policy(self) -> None:
        import tempfile
        from pathlib import Path

        import torch.optim as optim
        from magic_ai.ppo import RolloutStep
        from magic_ai.rnad import RNaDConfig
        from magic_ai.rnad_trainer import EpisodeBatch, build_trainer_state, run_rnad_update

        torch.manual_seed(0)
        model = _model()
        replay = TextReplayBuffer(
            capacity=8,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        rows: list[int] = []
        # All rows use batch_index=0 so both decision options are unmasked
        # (batch row 1 only has one valid option in the test fixture).
        for perspective in (0, 1, 0, 1):
            rows.append(
                replay.append(
                    encoded=_batch(batch_size=2),
                    batch_index=0,
                    trace_kind_id=0,
                    decision_option_idx=torch.tensor([[0, 1]]),
                    decision_target_idx=torch.tensor([[-1, -1]]),
                    decision_mask=torch.tensor([[True, True]]),
                    uses_none_head=torch.tensor([False]),
                    selected_indices=torch.tensor([1]),
                    may_selected=0.0,
                    old_log_prob=-0.5,
                    value=0.0,
                    perspective_player_idx=perspective,
                    lstm_h_in=torch.zeros(1, 8),
                    lstm_c_in=torch.zeros(1, 8),
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            state = build_trainer_state(
                model,
                config=RNaDConfig(delta_m=2, num_outer_iterations=1),
                reg_snapshot_dir=Path(tmp),
                device=model.device,
            )
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            episodes = [
                EpisodeBatch(
                    steps=[
                        RolloutStep(
                            perspective_player_idx=0,
                            old_log_prob=-0.5,
                            value=0.0,
                            replay_idx=rows[0],
                        ),
                        RolloutStep(
                            perspective_player_idx=1,
                            old_log_prob=-0.5,
                            value=0.0,
                            replay_idx=rows[1],
                        ),
                    ],
                    terminal_reward_p0=1.0,
                    zero_sum=True,
                ),
                EpisodeBatch(
                    steps=[
                        RolloutStep(
                            perspective_player_idx=0,
                            old_log_prob=-0.5,
                            value=0.0,
                            replay_idx=rows[2],
                        ),
                        RolloutStep(
                            perspective_player_idx=1,
                            old_log_prob=-0.5,
                            value=0.0,
                            replay_idx=rows[3],
                        ),
                    ],
                    terminal_reward_p0=-1.0,
                    zero_sum=True,
                ),
            ]
            stats = run_rnad_update(model, optimizer, state, episodes)

        import math

        self.assertTrue(math.isfinite(stats.loss))
        self.assertTrue(math.isfinite(stats.policy_loss))
        self.assertTrue(math.isfinite(stats.value_loss))

    def test_count_active_replay_steps(self) -> None:
        model = _model()
        replay = TextReplayBuffer(
            capacity=4,
            max_tokens=4,
            max_options=2,
            max_targets_per_option=1,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=8,
        )
        model.rollout_buffer = replay
        row_priority = replay.append(
            encoded=_batch(batch_size=1),
            batch_index=0,
            trace_kind_id=0,  # priority
            decision_option_idx=torch.tensor([[0, 1]]),
            decision_target_idx=torch.tensor([[-1, -1]]),
            decision_mask=torch.tensor([[True, True]]),
            uses_none_head=torch.tensor([False]),
            selected_indices=torch.tensor([0]),
            may_selected=0.0,
            old_log_prob=-0.5,
            value=0.1,
            perspective_player_idx=0,
            lstm_h_in=torch.zeros(1, 8),
            lstm_c_in=torch.zeros(1, 8),
        )
        from magic_ai.actions import TRACE_KIND_TO_ID

        row_may = replay.append(
            encoded=_batch(batch_size=1),
            batch_index=0,
            trace_kind_id=TRACE_KIND_TO_ID["may"],
            decision_option_idx=torch.tensor([[-1, -1]]),
            decision_target_idx=torch.tensor([[-1, -1]]),
            decision_mask=torch.tensor([[False, False]]),
            uses_none_head=torch.tensor([False]),
            selected_indices=torch.tensor([-1]),
            may_selected=1.0,
            old_log_prob=-0.5,
            value=0.0,
            perspective_player_idx=1,
            lstm_h_in=torch.zeros(1, 8),
            lstm_c_in=torch.zeros(1, 8),
        )

        cl, pl = model.count_active_replay_steps([[row_priority, row_may]])
        self.assertEqual(cl, 2)
        # priority row contributes 2 valid choice cells; may row contributes
        # 1 may-active step; total pl = 3.
        self.assertEqual(pl, 3)


if __name__ == "__main__":
    unittest.main()
