from __future__ import annotations

import unittest
from collections.abc import Iterator
from unittest.mock import patch

import torch
from magic_ai.ppo import RolloutStep, gae_returns, gae_returns_batched, ppo_update
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

    def gather_replay_old_log_prob_value(
        self,
        replay_rows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n = int(replay_rows.numel())
        zeros = self.param.expand(n).detach().contiguous()
        return zeros.clone(), zeros.clone()

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
        replay_rows = torch.arange(5, dtype=torch.long)

        with patch(
            "magic_ai.ppo.torch.randperm",
            lambda n, device=None: torch.arange(n, device=device),
        ):
            ppo_update(
                policy,
                optimizer,
                replay_rows,
                torch.zeros(replay_rows.numel()),
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
        replay_rows = torch.arange(5, dtype=torch.long)

        with patch(
            "magic_ai.ppo.torch.randperm",
            lambda n, device=None: torch.arange(n, device=device),
        ):
            ppo_update(
                policy,
                optimizer,
                replay_rows,
                torch.zeros(replay_rows.numel()),
                epochs=1,
                minibatch_size=2,
                minibatch_token_limit=100,
                minibatch_max_tokens_per_row=1,
            )

        self.assertEqual(policy.replay_batches, [[0, 1], [2, 3], [4]])


class GaeReturnsDrawTest(unittest.TestCase):
    """A drawn game must hand BOTH players a negative terminal return.

    Regression: an earlier version assigned ``-draw_penalty`` to the last
    step alone and let the perspective-aware GAE propagation flip its sign
    across player switches, effectively rewarding one side for stalling
    on the opponent's turn.
    """

    @staticmethod
    def _step(player: int, value: float = 0.0) -> RolloutStep:
        return RolloutStep(
            perspective_player_idx=player,
            old_log_prob=0.0,
            value=value,
            replay_idx=None,
        )

    def test_both_players_get_draw_penalty(self) -> None:
        steps = [self._step(p) for p in (0, 1, 0, 1, 0, 1)]
        gamma, lam, dp = 1.0, 0.95, 1.0
        returns = gae_returns(steps, winner_idx=-1, gamma=gamma, gae_lambda=lam, draw_penalty=dp)
        # Every step's return is negative regardless of perspective —
        # the cross-player sign for draws is +1, so the discounted
        # -draw_penalty reaches both players' steps without flipping sign.
        self.assertTrue(bool((returns < 0).all().item()), f"returns={returns!r}")
        # Magnitude should grow monotonically toward the terminal step.
        diffs = returns[1:] - returns[:-1]
        self.assertTrue(bool((diffs <= 0).all().item()), f"diffs={diffs!r}")
        # Terminal reward is exactly -draw_penalty.
        self.assertAlmostEqual(float(returns[-1]), -dp, places=6)
        # Closed form for V=0 with all-alternating players:
        # advantage_t = -(gamma*lam)^(T-1-t), and returns = advantages.
        expected = -torch.tensor(
            [(gamma * lam) ** (len(steps) - 1 - t) for t in range(len(steps))],
            dtype=torch.float32,
        )
        torch.testing.assert_close(returns, expected)

    def test_win_loss_unchanged(self) -> None:
        # Sanity check that the non-draw path is intact: terminal player
        # gets +1, the other side's perspective steps see propagated -1.
        steps = [self._step(0), self._step(1), self._step(0)]
        returns = gae_returns(steps, winner_idx=0, draw_penalty=1.0, gamma=1.0, gae_lambda=0.95)
        # Last step (player 0, winner) → +1; player 1's prior step → -1.
        self.assertGreater(float(returns[-1]), 0.0)
        self.assertLess(float(returns[1]), 0.0)


class GaeReturnsBatchedParityTest(unittest.TestCase):
    """The batched GAE must match stacking the per-episode reference."""

    @staticmethod
    def _episode(
        length: int,
        players: list[int],
        values: list[float],
    ) -> list[RolloutStep]:
        assert len(players) == length and len(values) == length
        return [
            RolloutStep(
                perspective_player_idx=p,
                old_log_prob=0.0,
                value=v,
                replay_idx=None,
            )
            for p, v in zip(players, values, strict=True)
        ]

    def _run_parity(
        self,
        cases: list[tuple[int, int, list[int], list[float]]],
        *,
        gamma: float,
        gae_lambda: float,
        draw_penalty: float,
    ) -> None:
        max_steps = max(length for length, _, _, _ in cases)
        batch = len(cases)
        values = torch.zeros(batch, max_steps, dtype=torch.float32)
        players = torch.zeros(batch, max_steps, dtype=torch.long)
        step_count = torch.zeros(batch, dtype=torch.long)
        winner = torch.zeros(batch, dtype=torch.long)
        ref_padded = torch.zeros(batch, max_steps, dtype=torch.float32)

        for b, (length, w, ps, vs) in enumerate(cases):
            steps = self._episode(length, ps, vs)
            for t, step in enumerate(steps):
                values[b, t] = step.value
                players[b, t] = step.perspective_player_idx
                # Fill padding with a *garbage* sentinel to confirm batched
                # impl masks it out: garbage values, garbage player flips.
            for t in range(length, max_steps):
                values[b, t] = 999.0
                players[b, t] = (b + t) % 2
            step_count[b] = length
            winner[b] = w
            ref = gae_returns(
                steps,
                winner_idx=w,
                gamma=gamma,
                gae_lambda=gae_lambda,
                draw_penalty=draw_penalty,
            )
            ref_padded[b, :length] = ref

        got = gae_returns_batched(
            values,
            players,
            step_count,
            winner,
            gamma=gamma,
            gae_lambda=gae_lambda,
            draw_penalty=draw_penalty,
        )
        torch.testing.assert_close(got, ref_padded, atol=1e-5, rtol=1e-5)

    def test_parity_mixed_episodes_default_hparams(self) -> None:
        cases = [
            (5, 0, [0, 1, 0, 1, 0], [0.1, -0.2, 0.3, 0.4, -0.5]),
            (3, 1, [1, 1, 0], [0.0, 0.5, -0.1]),
            (1, 0, [0], [0.7]),
            (4, -1, [0, 1, 0, 1], [0.2, 0.2, 0.2, 0.2]),  # draw
            (7, 1, [0, 1, 0, 1, 0, 1, 0], [-0.4, 0.0, 0.3, -0.1, 0.6, 0.2, -0.3]),
        ]
        self._run_parity(cases, gamma=1.0, gae_lambda=0.95, draw_penalty=1.0)

    def test_parity_gamma_lambda_sweep(self) -> None:
        cases = [
            (6, 0, [0, 1, 1, 0, 1, 0], [0.1, 0.2, -0.3, 0.4, 0.0, -0.5]),
            (2, 1, [0, 1], [0.0, 0.0]),
            (8, -1, [0, 1, 0, 1, 0, 1, 0, 1], [0.05] * 8),
        ]
        for gamma in (1.0, 0.99, 0.0):
            for lam in (0.95, 0.5, 0.0):
                with self.subTest(gamma=gamma, gae_lambda=lam):
                    self._run_parity(cases, gamma=gamma, gae_lambda=lam, draw_penalty=0.5)

    def test_parity_random_episodes(self) -> None:
        torch.manual_seed(0)
        cases: list[tuple[int, int, list[int], list[float]]] = []
        for _ in range(16):
            length = int(torch.randint(1, 20, (1,)).item())
            ps = torch.randint(0, 2, (length,)).tolist()
            vs = torch.randn(length).tolist()
            w = int(torch.randint(-1, 2, (1,)).item())
            cases.append((length, w, ps, vs))
        self._run_parity(cases, gamma=1.0, gae_lambda=0.95, draw_penalty=1.0)

    def test_empty_step_count_row_is_zero(self) -> None:
        # Empty rows aren't supported by the per-episode reference, but the
        # batched API tolerates them and returns zeros so callers can pass
        # ragged staging tensors directly.
        values = torch.zeros(2, 4, dtype=torch.float32)
        players = torch.zeros(2, 4, dtype=torch.long)
        step_count = torch.tensor([0, 3], dtype=torch.long)
        winner = torch.tensor([0, 1], dtype=torch.long)
        values[1, :3] = torch.tensor([0.1, 0.2, 0.3])
        players[1, :3] = torch.tensor([1, 0, 1])
        got = gae_returns_batched(values, players, step_count, winner)
        torch.testing.assert_close(got[0], torch.zeros(4))
        ref = gae_returns(
            GaeReturnsBatchedParityTest._episode(3, [1, 0, 1], [0.1, 0.2, 0.3]),
            winner_idx=1,
        )
        torch.testing.assert_close(got[1, :3], ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got[1, 3], torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
