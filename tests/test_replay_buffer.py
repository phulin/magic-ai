import unittest

import torch
from magic_ai.replay_buffer import ReplayCore


class ReplayCoreTests(unittest.TestCase):
    def test_row_allocation_reset_and_release(self) -> None:
        core = ReplayCore(
            capacity=3,
            decision_capacity=8,
            max_decision_groups=2,
            max_cached_choices=3,
            device="cpu",
        )

        rows = core.allocate_rows(2)
        self.assertEqual(rows, [0, 1])
        self.assertEqual(core.size, 2)

        core.release_rows([rows[0]])
        self.assertEqual(core.size, 1)
        self.assertEqual(core.allocate_row(), rows[0])

        core.reset()
        self.assertEqual(core.size, 0)
        self.assertEqual(core.allocate_rows(3), [0, 1, 2])
        with self.assertRaisesRegex(RuntimeError, "full"):
            core.allocate_row()

    def test_decision_append_gather_and_valid_choice_count(self) -> None:
        core = ReplayCore(
            capacity=2,
            decision_capacity=4,
            max_decision_groups=2,
            max_cached_choices=3,
            device="cpu",
            index_dtype=torch.int16,
        )
        row = core.allocate_row()

        stored = core.write_decision_row(
            row,
            decision_option_idx=torch.tensor([[0, 1, -1], [2, -1, -1]]),
            decision_target_idx=torch.tensor([[-1, 0, -1], [1, -1, -1]]),
            decision_mask=torch.tensor([[True, True, False], [True, False, False]]),
            uses_none_head=torch.tensor([False, True]),
            selected_indices=torch.tensor([1, 0]),
        )

        self.assertEqual(stored, 2)
        gathered = core.gather_dense_decisions(torch.tensor([row]))
        torch.testing.assert_close(
            gathered.decision_option_idx[0],
            torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int16),
        )
        torch.testing.assert_close(
            gathered.decision_target_idx[0],
            torch.tensor([[-1, 0, -1], [1, -1, -1]], dtype=torch.int16),
        )
        torch.testing.assert_close(
            gathered.decision_mask[0],
            torch.tensor([[True, True, False], [True, False, False]]),
        )
        torch.testing.assert_close(gathered.uses_none_head[0], torch.tensor([False, True]))
        torch.testing.assert_close(
            gathered.selected_indices[0], torch.tensor([1, 0], dtype=torch.int16)
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
        rows = torch.tensor(core.allocate_rows(1), dtype=torch.long)

        stored = core.write_decision_batch(
            rows,
            decision_count=torch.tensor([3]),
            decision_option_idx=torch.tensor([[0, -1], [1, -1], [2, -1]]),
            decision_target_idx=torch.tensor([[-1, -1], [0, -1], [1, -1]]),
            decision_mask=torch.tensor([[True, False], [True, False], [True, False]]),
            uses_none_head=torch.tensor([False, True, False]),
            selected_indices=torch.tensor([0, 0, 0]),
        )

        torch.testing.assert_close(stored, torch.tensor([2]))
        gathered = core.gather_dense_decisions(rows)
        self.assertEqual(int(gathered.decision_count[0]), 2)
        torch.testing.assert_close(
            gathered.decision_option_idx[0],
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
        row = core.allocate_row()
        h_in = torch.full((1, 3), 0.25)
        c_in = torch.full((1, 3), -0.5)
        core.write_common_row(
            row,
            trace_kind_id=5,
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
