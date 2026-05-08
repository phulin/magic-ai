from __future__ import annotations

import unittest
from typing import cast

import torch
from magic_ai.actions import ParsedBatch
from magic_ai.game_state import GameStateSnapshot, PendingState
from magic_ai.slot_encoder.buffer import NativeTrajectoryBuffer
from magic_ai.slot_encoder.game_state import GameStateEncoder
from magic_ai.slot_encoder.model import PPOPolicy
from magic_ai.slot_encoder.native_encoder import (
    NativeBatchEncoder,
    NativeEncodedBatch,
    NativeEncodingError,
    _validate_decision_layout,
)


def _sample_state() -> dict:
    pending = {
        "kind": "priority",
        "player_idx": 0,
        "options": [
            {"kind": "pass", "label": "Pass"},
            {
                "kind": "play_land",
                "label": "Play Mountain",
                "card_id": "card-1",
                "card_name": "Mountain",
            },
        ],
    }
    return {
        "turn": 1,
        "active_player": "A",
        "step": "Precombat Main",
        "players": [
            {
                "ID": "p1",
                "Name": "A",
                "Life": 20,
                "HandCount": 1,
                "GraveyardCount": 0,
                "LibraryCount": 53,
                "Hand": [{"ID": "card-1", "Name": "Mountain"}],
                "Graveyard": [],
                "Battlefield": [],
                "ManaPool": {},
            },
            {
                "ID": "p2",
                "Name": "B",
                "Life": 20,
                "HandCount": 0,
                "GraveyardCount": 0,
                "LibraryCount": 53,
                "Hand": [],
                "Graveyard": [],
                "Battlefield": [],
                "ManaPool": {},
            },
        ],
        "pending": pending,
        "stack": [],
    }


class _LibWithoutEncode:
    MageFreeString = None


def _native_batch_from_parsed(parsed: ParsedBatch, pending: PendingState) -> NativeEncodedBatch:
    return NativeEncodedBatch(
        trace_kind_id=parsed.trace_kind_ids,
        slot_card_rows=parsed.parsed_state.slot_card_rows,
        slot_occupied=parsed.parsed_state.slot_occupied,
        slot_tapped=parsed.parsed_state.slot_tapped,
        game_info=parsed.parsed_state.game_info,
        pending_kind_id=parsed.parsed_action.pending_kind_id,
        num_present_options=parsed.parsed_action.num_present_options,
        option_kind_ids=parsed.parsed_action.option_kind_ids,
        option_scalars=parsed.parsed_action.option_scalars,
        option_mask=parsed.parsed_action.option_mask,
        option_ref_slot_idx=parsed.parsed_action.option_ref_slot_idx,
        option_ref_card_row=parsed.parsed_action.option_ref_card_row,
        target_mask=parsed.parsed_action.target_mask,
        target_type_ids=parsed.parsed_action.target_type_ids,
        target_scalars=parsed.parsed_action.target_scalars,
        target_overflow=parsed.parsed_action.target_overflow,
        target_ref_slot_idx=parsed.parsed_action.target_ref_slot_idx,
        target_ref_is_player=parsed.parsed_action.target_ref_is_player,
        target_ref_is_self=parsed.parsed_action.target_ref_is_self,
        may_mask=parsed.trace_kind_ids == 6,
        decision_start=torch.tensor(parsed.decision_starts, dtype=torch.int64),
        decision_count=torch.tensor(parsed.decision_counts, dtype=torch.int64),
        decision_option_idx=parsed.decision_option_idx,
        decision_target_idx=parsed.decision_target_idx,
        decision_mask=parsed.decision_mask,
        uses_none_head=parsed.uses_none_head,
        decision_rows_written=parsed.decision_option_idx.shape[0],
        pendings=[pending for _ in parsed.trace_kinds],
        trace_kinds=list(parsed.trace_kinds),
    )


class NativeEncoderTests(unittest.TestCase):
    def test_native_encoder_reports_unavailable_without_symbol(self) -> None:
        encoder = NativeBatchEncoder(
            max_options=4,
            max_targets_per_option=2,
            max_cached_choices=8,
            lib=_LibWithoutEncode(),
        )
        self.assertFalse(encoder.is_available)

    def test_policy_accepts_native_encoded_batch(self) -> None:
        state = cast(GameStateSnapshot, _sample_state())
        pending: PendingState = state["pending"]
        encoder = GameStateEncoder({"Mountain": [0.1, 0.2, 0.3]}, d_model=8)
        policy = PPOPolicy(
            encoder,
            hidden_dim=16,
            hidden_layers=1,
            max_options=4,
            max_targets_per_option=2,
            rollout_capacity=16,
        )

        parsed = policy.parse_inputs_batch(
            [state],
            [pending],
            perspective_player_indices=[0],
        )
        native_batch = _native_batch_from_parsed(parsed, pending)

        with torch.no_grad():
            parsed_step = policy.act_parsed_batch(parsed, deterministic=True)[0]
            native_step = policy.act_parsed_batch(native_batch, deterministic=True)[0]

        self.assertEqual(parsed_step.action, native_step.action)
        self.assertEqual(parsed_step.trace, native_step.trace)
        self.assertAlmostEqual(float(parsed_step.log_prob), float(native_step.log_prob), places=6)
        self.assertAlmostEqual(float(parsed_step.value), float(native_step.value), places=6)

    def test_lstm_native_rollout_stores_replay_state_inputs(self) -> None:
        state = cast(GameStateSnapshot, _sample_state())
        pending: PendingState = state["pending"]
        encoder = GameStateEncoder({"Mountain": [0.1, 0.2, 0.3]}, d_model=8)
        policy = PPOPolicy(
            encoder,
            hidden_dim=16,
            hidden_layers=1,
            max_options=4,
            max_targets_per_option=2,
            rollout_capacity=16,
            use_lstm=True,
        )
        policy.init_lstm_env_states(2)
        parsed = policy.parse_inputs_batch(
            [state, state],
            [pending, pending],
            perspective_player_indices=[0, 0],
        )
        native_batch = _native_batch_from_parsed(parsed, pending)
        env_indices = [0, 1]
        state_inputs = policy.lstm_env_state_inputs(env_indices)
        assert state_inputs is not None

        with torch.no_grad():
            policy_steps = policy.sample_native_batch(
                native_batch,
                env_indices=env_indices,
                deterministic=True,
            )

        self.assertFalse(torch.equal(policy.live_lstm_h, torch.zeros_like(policy.live_lstm_h)))
        selected_cols = [
            col for policy_step in policy_steps for col in policy_step.selected_choice_cols
        ]
        staging = NativeTrajectoryBuffer(
            num_envs=2,
            max_steps_per_trajectory=4,
            decision_capacity_per_env=8,
            max_options=4,
            max_targets_per_option=2,
            max_cached_choices=policy.max_cached_choices,
            zone_slot_count=policy.rollout_buffer.slot_card_rows.shape[1],
            game_info_dim=policy.rollout_buffer.game_info.shape[1],
            option_scalar_dim=policy.rollout_buffer.option_scalars.shape[2],
            target_scalar_dim=policy.rollout_buffer.target_scalars.shape[3],
            recurrent_layers=policy.hidden_layers,
            recurrent_hidden_dim=policy.hidden_dim,
        )
        staging.stage_batch(
            env_indices,
            native_batch,
            selected_choice_cols_flat=torch.tensor(selected_cols, dtype=torch.long),
            behavior_action_log_probs_flat=torch.tensor(
                [lp for step in policy_steps for lp in step.selected_action_log_probs],
                dtype=torch.float32,
            ),
            may_selected=[policy_step.may_selected for policy_step in policy_steps],
            old_log_probs=torch.stack([policy_step.log_prob for policy_step in policy_steps]),
            values=torch.stack([policy_step.value for policy_step in policy_steps]),
            perspective_player_indices=[0, 0],
            decision_counts=[len(step.selected_choice_cols) for step in policy_steps],
            lstm_h_in=state_inputs[0],
            lstm_c_in=state_inputs[1],
        )
        self.assertTrue(torch.equal(staging.lstm_h_in[0, 0], state_inputs[0][0]))

        write = policy.rollout_buffer.ingest_staged_episodes(staging, env_indices)
        assert policy.rollout_buffer.lstm_h_in is not None
        self.assertTrue(
            torch.equal(policy.rollout_buffer.lstm_h_in[write.step_indices[0]], state_inputs[0][0])
        )
        log_probs, entropies, values, _extras = policy.evaluate_replay_batch(
            [int(idx) for idx in write.step_indices.detach().cpu().tolist()]
        )
        self.assertEqual(tuple(log_probs.shape), (2,))
        self.assertEqual(tuple(entropies.shape), (2,))
        self.assertEqual(tuple(values.shape), (2,))

    def test_spr_auxiliary_loss_runs_and_updates_target(self) -> None:
        state = cast(GameStateSnapshot, _sample_state())
        pending: PendingState = state["pending"]
        encoder = GameStateEncoder({"Mountain": [0.1, 0.2, 0.3]}, d_model=8)
        policy = PPOPolicy(
            encoder,
            hidden_dim=16,
            hidden_layers=1,
            max_options=4,
            max_targets_per_option=2,
            rollout_capacity=16,
            use_lstm=True,
            spr_enabled=True,
            spr_action_dim=8,
            spr_ema_decay=0.5,
        )
        policy.init_lstm_env_states(1)
        env_indices = [0]
        staging = NativeTrajectoryBuffer(
            num_envs=1,
            max_steps_per_trajectory=4,
            decision_capacity_per_env=8,
            max_options=4,
            max_targets_per_option=2,
            max_cached_choices=policy.max_cached_choices,
            zone_slot_count=policy.rollout_buffer.slot_card_rows.shape[1],
            game_info_dim=policy.rollout_buffer.game_info.shape[1],
            option_scalar_dim=policy.rollout_buffer.option_scalars.shape[2],
            target_scalar_dim=policy.rollout_buffer.target_scalars.shape[3],
            recurrent_layers=policy.hidden_layers,
            recurrent_hidden_dim=policy.hidden_dim,
        )
        for _ in range(3):
            parsed = policy.parse_inputs_batch(
                [state],
                [pending],
                perspective_player_indices=[0],
            )
            native_batch = _native_batch_from_parsed(parsed, pending)
            state_inputs = policy.lstm_env_state_inputs(env_indices)
            assert state_inputs is not None
            with torch.no_grad():
                policy_steps = policy.sample_native_batch(
                    native_batch,
                    env_indices=env_indices,
                    deterministic=True,
                )
            selected_cols = [col for step in policy_steps for col in step.selected_choice_cols]
            staging.stage_batch(
                env_indices,
                native_batch,
                selected_choice_cols_flat=torch.tensor(selected_cols, dtype=torch.long),
                behavior_action_log_probs_flat=torch.tensor(
                    [lp for step in policy_steps for lp in step.selected_action_log_probs],
                    dtype=torch.float32,
                ),
                may_selected=[step.may_selected for step in policy_steps],
                old_log_probs=torch.stack([step.log_prob for step in policy_steps]),
                values=torch.stack([step.value for step in policy_steps]),
                perspective_player_indices=[0],
                decision_counts=[len(step.selected_choice_cols) for step in policy_steps],
                lstm_h_in=state_inputs[0],
                lstm_c_in=state_inputs[1],
            )

        write = policy.rollout_buffer.ingest_staged_episodes(staging, env_indices)
        rows = write.step_indices
        self.assertEqual(int(policy.rollout_buffer.has_next[rows[0]].item()), 1)
        self.assertEqual(int(policy.rollout_buffer.has_next[rows[1]].item()), 1)
        self.assertEqual(int(policy.rollout_buffer.has_next[rows[2]].item()), 0)
        self.assertEqual(int(policy.rollout_buffer.next_step_idx[rows[0]].item()), int(rows[1]))
        self.assertEqual(int(policy.rollout_buffer.next_step_idx[rows[1]].item()), int(rows[2]))

        before = {
            name: param.detach().clone() for name, param in policy.target_lstm.named_parameters()
        }
        loss = policy.compute_spr_loss(rows)
        self.assertEqual(loss.dim(), 0)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy.update_spr_target()
        changed = any(
            not torch.equal(before[name], param)
            for name, param in policy.target_lstm.named_parameters()
        )
        self.assertTrue(changed)

    def test_native_decision_validation_rejects_out_of_range_option(self) -> None:
        with self.assertRaisesRegex(NativeEncodingError, "outside \\[0, 4\\)"):
            _validate_decision_layout(
                decision_option_idx=torch.tensor([[4]], dtype=torch.int64),
                decision_target_idx=torch.tensor([[-1]], dtype=torch.int64),
                decision_mask=torch.tensor([[True]], dtype=torch.bool),
                uses_none_head=torch.tensor([False], dtype=torch.bool),
                max_options=4,
                max_targets_per_option=2,
            )


if __name__ == "__main__":
    unittest.main()
