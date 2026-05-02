import unittest

import torch
from magic_ai.replay_buffer import ReplayCore


class ReplayCoreTests(unittest.TestCase):
    def test_append_rows_reset_and_capacity(self) -> None:
        core = ReplayCore(
            capacity=3,
            decision_capacity=8,
            max_decision_groups=2,
            max_cached_choices=3,
            device="cpu",
        )

        rows = core.append_batch(
            trace_kind_id=torch.tensor([1, 2]),
            decision_count=torch.tensor([0, 0]),
            decision_option_idx=torch.empty(0, 3, dtype=torch.long),
            decision_target_idx=torch.empty(0, 3, dtype=torch.long),
            decision_mask=torch.empty(0, 3, dtype=torch.bool),
            uses_none_head=torch.empty(0, dtype=torch.bool),
            selected_indices=torch.empty(0, dtype=torch.long),
        )
        torch.testing.assert_close(rows, torch.tensor([0, 1]))
        self.assertEqual(core.size, 2)

        core.reset()
        self.assertEqual(core.size, 0)
        rows = core.append_batch(
            trace_kind_id=torch.tensor([1, 2, 3]),
            decision_count=torch.tensor([0, 0, 0]),
            decision_option_idx=torch.empty(0, 3, dtype=torch.long),
            decision_target_idx=torch.empty(0, 3, dtype=torch.long),
            decision_mask=torch.empty(0, 3, dtype=torch.bool),
            uses_none_head=torch.empty(0, dtype=torch.bool),
            selected_indices=torch.empty(0, dtype=torch.long),
        )
        torch.testing.assert_close(rows, torch.tensor([0, 1, 2]))
        with self.assertRaisesRegex(RuntimeError, "full"):
            core.append_row(
                trace_kind_id=4,
                decision_option_idx=torch.empty(0, 3, dtype=torch.long),
                decision_target_idx=torch.empty(0, 3, dtype=torch.long),
                decision_mask=torch.empty(0, 3, dtype=torch.bool),
                uses_none_head=torch.empty(0, dtype=torch.bool),
                selected_indices=torch.empty(0, dtype=torch.long),
                may_selected=0.0,
                old_log_prob=0.0,
                value=0.0,
                perspective_player_idx=0,
            )

    def test_decision_append_gather_and_valid_choice_count(self) -> None:
        core = ReplayCore(
            capacity=2,
            decision_capacity=4,
            max_decision_groups=2,
            max_cached_choices=3,
            device="cpu",
            index_dtype=torch.int16,
        )
        row = core.append_row(
            trace_kind_id=0,
            decision_option_idx=torch.tensor([[0, 1, -1], [2, -1, -1]]),
            decision_target_idx=torch.tensor([[-1, 0, -1], [1, -1, -1]]),
            decision_mask=torch.tensor([[True, True, False], [True, False, False]]),
            uses_none_head=torch.tensor([False, True]),
            selected_indices=torch.tensor([1, 0]),
            may_selected=0.0,
            old_log_prob=0.0,
            value=0.0,
            perspective_player_idx=0,
        )

        self.assertEqual(int(core.decision_count[row]), 2)
        gathered = core.gather_decisions(torch.tensor([row]))
        torch.testing.assert_close(gathered.decision_start, torch.tensor([0]))
        torch.testing.assert_close(gathered.decision_count, torch.tensor([2]))
        torch.testing.assert_close(gathered.step_for_group, torch.tensor([0, 0]))
        torch.testing.assert_close(
            gathered.decision_option_idx,
            torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            gathered.decision_target_idx,
            torch.tensor([[-1, 0, -1], [1, -1, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            gathered.decision_mask,
            torch.tensor([[True, True, False], [True, False, False]]),
        )
        torch.testing.assert_close(gathered.uses_none_head, torch.tensor([False, True]))
        torch.testing.assert_close(
            gathered.selected_indices, torch.tensor([1, 0], dtype=torch.int32)
        )
        self.assertEqual(int(core.valid_choice_count(torch.tensor([row])).item()), 3)

    def test_decision_batch_truncates_to_max_groups(self) -> None:
        core = ReplayCore(
            capacity=1,
            decision_capacity=2,
            max_decision_groups=2,
            max_cached_choices=2,
            device="cpu",
        )
        rows = core.append_batch(
            trace_kind_id=torch.tensor([0]),
            decision_count=torch.tensor([3]),
            decision_option_idx=torch.tensor([[0, -1], [1, -1], [2, -1]]),
            decision_target_idx=torch.tensor([[-1, -1], [0, -1], [1, -1]]),
            decision_mask=torch.tensor([[True, False], [True, False], [True, False]]),
            uses_none_head=torch.tensor([False, True, False]),
            selected_indices=torch.tensor([0, 0, 0]),
        )

        torch.testing.assert_close(core.decision_count[rows], torch.tensor([2]))
        gathered = core.gather_decisions(rows)
        self.assertEqual(int(gathered.decision_count[0]), 2)
        torch.testing.assert_close(gathered.step_for_group, torch.tensor([0, 0]))
        torch.testing.assert_close(
            gathered.decision_option_idx,
            torch.tensor([[0, -1], [1, -1]]),
        )

    def test_ppo_targets_and_common_gather(self) -> None:
        core = ReplayCore(
            capacity=2,
            decision_capacity=2,
            max_decision_groups=1,
            max_cached_choices=2,
            recurrent_layers=1,
            recurrent_hidden_dim=3,
            device="cpu",
        )
        h_in = torch.full((1, 3), 0.25)
        c_in = torch.full((1, 3), -0.5)
        row = core.append_row(
            trace_kind_id=5,
            decision_option_idx=torch.empty(0, 2, dtype=torch.long),
            decision_target_idx=torch.empty(0, 2, dtype=torch.long),
            decision_mask=torch.empty(0, 2, dtype=torch.bool),
            uses_none_head=torch.empty(0, dtype=torch.bool),
            selected_indices=torch.empty(0, dtype=torch.long),
            may_selected=1.0,
            old_log_prob=-0.75,
            value=0.5,
            perspective_player_idx=1,
            lstm_h_in=h_in,
            lstm_c_in=c_in,
        )
        core.write_ppo_targets(
            torch.tensor([row]),
            torch.tensor([-0.25]),
            torch.tensor([2.0]),
            torch.tensor([1.5]),
        )

        old_log_prob, returns, advantages = core.gather_ppo_targets(torch.tensor([row]))
        torch.testing.assert_close(old_log_prob, torch.tensor([-0.25]))
        torch.testing.assert_close(returns, torch.tensor([2.0]))
        torch.testing.assert_close(advantages, torch.tensor([1.5]))

        common = core.gather_common(torch.tensor([row]))
        self.assertEqual(int(common.trace_kind_id[0]), 5)
        self.assertEqual(int(common.perspective_player_idx[0]), 1)
        assert common.lstm_h_in is not None
        assert common.lstm_c_in is not None
        torch.testing.assert_close(common.lstm_h_in[0], h_in)
        torch.testing.assert_close(common.lstm_c_in[0], c_in)


if __name__ == "__main__":
    unittest.main()
