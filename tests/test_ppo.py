from __future__ import annotations

import unittest
from collections.abc import Iterator
from unittest.mock import patch

import torch
from magic_ai.ppo import RolloutStep, ppo_update
from torch import Tensor, nn


class _ReplayPolicy:
    spr_enabled = False

    def __init__(self) -> None:
        self.param = nn.Parameter(torch.tensor(0.0))
        self.replay_batches: list[list[int]] = []
        self.old_log_probs = torch.empty(0)
        self.returns = torch.empty(0)
        self.advantages = torch.empty(0)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        del recurse
        yield self.param

    def evaluate_replay_batch(
        self,
        replay_rows: list[int] | Tensor,
        *,
        return_extras: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        del return_extras
        rows = (
            [int(row) for row in replay_rows.detach().cpu().tolist()]
            if isinstance(replay_rows, Tensor)
            else list(replay_rows)
        )
        self.replay_batches.append(rows)
        n = len(rows)
        log_probs = self.param.expand(n)
        values = self.param.expand(n)
        entropies = self.param.new_zeros(n)
        return log_probs, entropies, values, None

    def write_ppo_targets(
        self,
        replay_rows: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> None:
        size = int(replay_rows.max().item()) + 1 if int(replay_rows.numel()) else 0
        self.old_log_probs = torch.zeros(size, dtype=torch.float32, device=replay_rows.device)
        self.returns = torch.zeros_like(self.old_log_probs)
        self.advantages = torch.zeros_like(self.old_log_probs)
        self.old_log_probs[replay_rows] = old_log_probs
        self.returns[replay_rows] = returns
        self.advantages[replay_rows] = advantages

    def gather_ppo_targets(self, replay_rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.old_log_probs[replay_rows],
            self.returns[replay_rows],
            self.advantages[replay_rows],
        )

    def compute_spr_loss(
        self,
        step_indices: Tensor,
        *,
        extras: object | None = None,
    ) -> Tensor:
        del step_indices, extras
        return self.param.new_zeros(())

    def update_spr_target(self, decay: float | None = None) -> None:
        del decay


class PPOUpdateTests(unittest.TestCase):
    def test_minibatch_token_limit_caps_tensor_slices_by_max_row_width(self) -> None:
        policy = _ReplayPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
        steps = [
            RolloutStep(perspective_player_idx=0, old_log_prob=0.0, value=0.0, replay_idx=row)
            for row in range(5)
        ]

        with patch(
            "magic_ai.ppo.torch.randperm",
            lambda n, device=None: torch.arange(n, device=device),
        ):
            ppo_update(
                policy,
                optimizer,
                steps,
                torch.zeros(len(steps)),
                epochs=1,
                minibatch_size=10,
                minibatch_token_limit=120,
                minibatch_max_tokens_per_row=50,
            )

        self.assertEqual(policy.replay_batches, [[0, 1], [2, 3], [4]])
        self.assertTrue(all(len(batch) <= 2 for batch in policy.replay_batches))

    def test_minibatch_token_limit_still_respects_row_limit(self) -> None:
        policy = _ReplayPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
        steps = [
            RolloutStep(perspective_player_idx=0, old_log_prob=0.0, value=0.0, replay_idx=row)
            for row in range(5)
        ]

        with patch(
            "magic_ai.ppo.torch.randperm",
            lambda n, device=None: torch.arange(n, device=device),
        ):
            ppo_update(
                policy,
                optimizer,
                steps,
                torch.zeros(len(steps)),
                epochs=1,
                minibatch_size=2,
                minibatch_token_limit=100,
                minibatch_max_tokens_per_row=1,
            )

        self.assertEqual(policy.replay_batches, [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()
