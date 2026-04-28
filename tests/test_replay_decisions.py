import unittest

import torch
from magic_ai.model import PPOPolicy
from magic_ai.replay_decisions import (
    decision_logits_reference,
    flat_decision_distribution,
    validate_flat_scored_indices,
)


class ReplayDecisionHelpersTests(unittest.TestCase):
    def test_flat_distribution_matches_manual_logits(self) -> None:
        option_vectors = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 2.0], [0.5, 0.5]],
                [[2.0, 1.0], [1.0, -1.0], [0.0, 1.0]],
            ]
        )
        target_vectors = torch.tensor(
            [
                [
                    [[0.1, 0.2], [0.2, 0.3]],
                    [[0.4, -0.5], [0.0, 0.1]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                [
                    [[-0.5, 0.25], [0.3, 0.7]],
                    [[0.2, 0.2], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
            ]
        )
        query = torch.tensor([[0.5, -1.0], [1.5, 0.25]])
        none_logits = torch.tensor([-0.75, 0.5])
        step_positions = torch.tensor([0, 0, 1])
        option_idx = torch.tensor(
            [
                [0, 1, -1],
                [-1, 0, -1],
                [-1, 2, -1],
            ]
        )
        target_idx = torch.tensor(
            [
                [-1, 0, -1],
                [-1, 1, -1],
                [-1, -1, -1],
            ]
        )
        masks = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [True, True, False],
            ]
        )
        uses_none = torch.tensor([False, True, True])

        logits = decision_logits_reference(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
            query=query,
            none_logits=none_logits,
            validate=True,
        )

        expected = torch.tensor(
            [
                [0.5, -1.3, -torch.inf],
                [-0.75, 0.3, -torch.inf],
                [0.5, 0.25, -torch.inf],
            ]
        )
        torch.testing.assert_close(logits, expected)

        group_idx, choice_cols, flat_logits, flat_log_probs, entropies = flat_decision_distribution(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
            query=query,
            none_logits=none_logits,
            validate=True,
        )

        expected_flat = expected[masks]
        expected_log_probs = torch.cat(
            [
                torch.log_softmax(expected[0, :2], dim=0),
                torch.log_softmax(expected[1, :2], dim=0),
                torch.log_softmax(expected[2, :2], dim=0),
            ]
        )
        expected_probs = torch.softmax(expected[:, :2], dim=1)
        expected_group_log_probs = torch.log_softmax(expected[:, :2], dim=1)
        expected_entropies = torch.stack(
            [
                -(expected_probs[0] * expected_group_log_probs[0]).sum(),
                -(expected_probs[1] * expected_group_log_probs[1]).sum(),
                -(expected_probs[2] * expected_group_log_probs[2]).sum(),
            ]
        )
        torch.testing.assert_close(group_idx, torch.tensor([0, 0, 1, 1, 2, 2]))
        torch.testing.assert_close(choice_cols, torch.tensor([0, 1, 0, 1, 0, 1]))
        torch.testing.assert_close(flat_logits, expected_flat)
        torch.testing.assert_close(flat_log_probs, expected_log_probs)
        torch.testing.assert_close(entropies, expected_entropies)

    def test_policy_adapter_uses_backend_neutral_flat_distribution(self) -> None:
        policy = PPOPolicy.__new__(PPOPolicy)
        policy.validate = True
        policy.compile_forward = False
        policy._compiled_compute_forward_impl = None
        policy._compiled_compute_hidden_target_impl = None
        policy._compiled_flat_decision_distribution_impl = None

        option_vectors = torch.randn(2, 3, 4)
        target_vectors = torch.randn(2, 3, 2, 4)
        query = torch.randn(2, 4)
        none_logits = torch.randn(2)
        step_positions = torch.tensor([0, 1])
        option_idx = torch.tensor([[0, 1], [-1, 2]])
        target_idx = torch.tensor([[-1, 0], [-1, -1]])
        masks = torch.tensor([[True, True], [True, True]])
        uses_none = torch.tensor([False, True])

        direct = flat_decision_distribution(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
            query=query,
            none_logits=none_logits,
            validate=True,
        )
        adapted = policy._flat_decision_distribution(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
            query=query,
            none_logits=none_logits,
        )
        for direct_tensor, adapted_tensor in zip(direct, adapted, strict=True):
            torch.testing.assert_close(adapted_tensor, direct_tensor)

    def test_validate_flat_scored_indices_reports_context(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "group=2 col=1 step=3 option=4 target=5 bounds=\\(steps=3, options=4, targets=5\\)",
        ):
            validate_flat_scored_indices(
                scored_groups=torch.tensor([2]),
                scored_cols=torch.tensor([1]),
                scored_steps=torch.tensor([3]),
                scored_option_idx=torch.tensor([4]),
                scored_target_idx=torch.tensor([5]),
                max_steps=3,
                max_options=4,
                max_targets=5,
            )


if __name__ == "__main__":
    unittest.main()
