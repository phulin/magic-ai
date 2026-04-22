from __future__ import annotations

import unittest
from typing import cast

import torch
from magic_ai.game_state import GameStateEncoder, GameStateSnapshot, PendingState
from magic_ai.model import PPOPolicy
from magic_ai.native_encoder import NativeBatchEncoder, NativeEncodedBatch


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
        native_batch = NativeEncodedBatch(
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
            may_mask=torch.zeros((1,), dtype=torch.bool),
            decision_start=torch.tensor(parsed.decision_starts, dtype=torch.int64),
            decision_count=torch.tensor(parsed.decision_counts, dtype=torch.int64),
            decision_option_idx=parsed.decision_option_idx,
            decision_target_idx=parsed.decision_target_idx,
            decision_mask=parsed.decision_mask,
            uses_none_head=parsed.uses_none_head,
            decision_rows_written=parsed.decision_option_idx.shape[0],
            pendings=[pending],
            trace_kinds=list(parsed.trace_kinds),
        )

        with torch.no_grad():
            parsed_step = policy.act_parsed_batch(parsed, deterministic=True)[0]
            native_step = policy.act_parsed_batch(native_batch, deterministic=True)[0]

        self.assertEqual(parsed_step.action, native_step.action)
        self.assertEqual(parsed_step.trace, native_step.trace)
        self.assertAlmostEqual(float(parsed_step.log_prob), float(native_step.log_prob), places=6)
        self.assertAlmostEqual(float(parsed_step.value), float(native_step.value), places=6)


if __name__ == "__main__":
    unittest.main()
