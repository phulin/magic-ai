import unittest

import torch
from magic_ai.replay_decisions import (
    ReplayScoringForward,
    decision_logits_reference,
    direct_decision_logits_from_forward,
    flat_decision_distribution,
    flat_decision_distribution_from_forward,
    score_may_decisions,
    score_may_decisions_from_forward,
    validate_flat_scored_indices,
)
from magic_ai.slot_encoder.model import PPOPolicy


class ReplayDecisionHelpersTests(unittest.TestCase):
    def test_score_may_decisions_matches_bernoulli_and_preserves_gradient(self) -> None:
        may_logits = torch.tensor([0.25, -0.75, 1.5], requires_grad=True)
        may_selected = torch.tensor([1.0, 0.0, 1.0])
        may_mask = torch.tensor([True, False, True])

        log_probs, entropies, logits_per_step, selected_per_step = score_may_decisions(
            may_logits=may_logits,
            may_selected=may_selected,
            may_mask=may_mask,
        )

        expected_dist = torch.distributions.Bernoulli(logits=may_logits[may_mask])
        expected_log_probs = torch.zeros(3)
        expected_entropies = torch.zeros(3)
        expected_log_probs[may_mask] = expected_dist.log_prob(may_selected[may_mask])
        expected_entropies[may_mask] = expected_dist.entropy()
        torch.testing.assert_close(log_probs, expected_log_probs)
        torch.testing.assert_close(entropies, expected_entropies)
        torch.testing.assert_close(logits_per_step, torch.tensor([0.25, 0.0, 1.5]))
        torch.testing.assert_close(selected_per_step, torch.tensor([1.0, 0.0, 1.0]))

        logits_per_step.sum().backward()
        torch.testing.assert_close(may_logits.grad, torch.tensor([1.0, 0.0, 1.0]))

    def test_replay_scoring_forward_drives_shared_helpers(self) -> None:
        query = torch.tensor([[2.0, -1.0]])
        forward = ReplayScoringForward(
            values=torch.tensor([0.1]),
            option_vectors=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            target_vectors=torch.zeros(1, 2, 1, 2),
            none_logits=torch.tensor([-0.5]),
            may_logits=torch.tensor([0.25]),
            hidden=torch.tensor([[0.5, 0.5]]),
            query=query,
        )
        step_positions = torch.tensor([0])
        option_idx = torch.tensor([[0, 1]])
        target_idx = torch.tensor([[-1, -1]])
        masks = torch.tensor([[True, True]])
        uses_none = torch.tensor([False])

        direct = flat_decision_distribution(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            option_vectors=forward.option_vectors,
            target_vectors=forward.target_vectors,
            query=query,
            none_logits=forward.none_logits,
        )
        via_forward = flat_decision_distribution_from_forward(
            forward,
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
        )
        for direct_tensor, forward_tensor in zip(direct, via_forward, strict=True):
            torch.testing.assert_close(forward_tensor, direct_tensor)

        may_direct = score_may_decisions(
            may_logits=forward.may_logits,
            may_selected=torch.tensor([1.0]),
            may_mask=torch.tensor([True]),
        )
        may_via_forward = score_may_decisions_from_forward(
            forward,
            may_selected=torch.tensor([1.0]),
            may_mask=torch.tensor([True]),
        )
        for direct_tensor, forward_tensor in zip(may_direct, may_via_forward, strict=True):
            torch.testing.assert_close(forward_tensor, direct_tensor)

    def test_direct_decision_logits_from_forward_scores_text_heads(self) -> None:
        forward = ReplayScoringForward(
            values=torch.zeros(2),
            option_vectors=torch.zeros(2, 3, 2),
            target_vectors=torch.zeros(2, 3, 2, 2),
            none_logits=torch.tensor([-1.0, 0.5]),
            may_logits=torch.zeros(2),
            hidden=torch.zeros(2, 4),
            option_logits=torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [1.0, 1.5, 2.0],
                ]
            ),
            target_logits=torch.tensor(
                [
                    [[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
                    [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]],
                ]
            ),
        )

        logits = direct_decision_logits_from_forward(
            forward,
            step_positions=torch.tensor([0, 1]),
            option_idx=torch.tensor([[0, 1, -1], [-1, 2, 1]]),
            target_idx=torch.tensor([[-1, 1, -1], [-1, 0, -1]]),
            masks=torch.tensor([[True, True, False], [True, True, True]]),
            uses_none=torch.tensor([False, True]),
            validate=True,
        )

        expected = torch.tensor(
            [
                [0.1, 0.7, -torch.inf],
                [0.5, 2.5, 1.5],
            ]
        )
        torch.testing.assert_close(logits, expected)

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
