import unittest

import torch
from magic_ai.text_encoder.actor_critic import TextActorCritic
from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.recurrent import RecurrentTextPolicyConfig
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


def _batch(batch_size: int = 2) -> TextEncodedBatch:
    token_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])[:batch_size]
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])[:batch_size]
    card_ref_positions = torch.full((batch_size, MAX_CARD_REFS), -1, dtype=torch.long)
    option_positions = torch.tensor([[1, 2], [1, -1]])[:batch_size]
    option_mask = option_positions >= 0
    target_positions = torch.full((batch_size, 2, 1), -1, dtype=torch.long)
    target_mask = target_positions >= 0
    seq_lengths = torch.tensor([4, 3])[:batch_size]
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

        self.assertEqual(tuple(step.output.policy_logits.shape), (2, 2))
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


if __name__ == "__main__":
    unittest.main()
