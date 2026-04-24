"""Unit tests for the R-NaD primitives."""

from __future__ import annotations

import math
import unittest

import torch
from magic_ai.rnad import (
    RNaDConfig,
    critic_loss,
    episodes_from_rollout_steps,
    neurd_loss,
    threshold_discretize,
    transform_rewards,
    two_player_vtrace,
)


class RNaDConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = RNaDConfig()
        self.assertAlmostEqual(cfg.eta, 0.2)
        self.assertEqual(cfg.delta_m, 25_000)
        self.assertEqual(cfg.num_outer_iterations, 20)
        self.assertAlmostEqual(cfg.neurd_beta, 2.0)
        self.assertGreater(cfg.target_ema_gamma, 0.0)
        self.assertLess(cfg.target_ema_gamma, 1.0)


class TransformRewardsTests(unittest.TestCase):
    def test_zero_when_policies_agree(self) -> None:
        t = 4
        logp = torch.full((t,), -1.0)
        rewards = torch.arange(t, dtype=torch.float32)
        out = transform_rewards(
            rewards,
            logp_theta=logp,
            logp_reg_cur=logp,
            logp_reg_prev=logp,
            alpha=0.5,
            eta=0.7,
            perspective_is_player_i=torch.tensor([True, False, True, False]),
        )
        self.assertTrue(torch.allclose(out, rewards))

    def test_sign_flip_between_players(self) -> None:
        rewards = torch.zeros(4)
        logp_theta = torch.zeros(4)
        logp_reg = torch.full((4,), -1.0)  # log_ratio = +1
        out = transform_rewards(
            rewards,
            logp_theta=logp_theta,
            logp_reg_cur=logp_reg,
            logp_reg_prev=logp_reg,
            alpha=1.0,
            eta=0.2,
            perspective_is_player_i=torch.tensor([True, False, True, False]),
        )
        expected = torch.tensor([-0.2, 0.2, -0.2, 0.2])
        self.assertTrue(torch.allclose(out, expected))

    def test_interpolates_between_reg_snapshots(self) -> None:
        rewards = torch.zeros(2)
        logp_theta = torch.zeros(2)
        logp_reg_cur = torch.full((2,), -2.0)
        logp_reg_prev = torch.full((2,), 0.0)
        perspective = torch.tensor([False, False])
        # alpha=0.25 -> blended_reg = -0.5, ratio = 0.5
        # opp-turn sign = +1, eta=1.0 -> +0.5
        out = transform_rewards(
            rewards,
            logp_theta=logp_theta,
            logp_reg_cur=logp_reg_cur,
            logp_reg_prev=logp_reg_prev,
            alpha=0.25,
            eta=1.0,
            perspective_is_player_i=perspective,
        )
        self.assertTrue(torch.allclose(out, torch.tensor([0.5, 0.5])))

    def test_rejects_bad_alpha(self) -> None:
        x = torch.zeros(1)
        with self.assertRaises(ValueError):
            transform_rewards(
                x,
                logp_theta=x,
                logp_reg_cur=x,
                logp_reg_prev=x,
                alpha=1.5,
                eta=0.1,
                perspective_is_player_i=torch.tensor([True]),
            )


def _reference_vtrace(
    rewards: torch.Tensor,
    values: torch.Tensor,
    logp_theta: torch.Tensor,
    logp_mu: torch.Tensor,
    is_own: torch.Tensor,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> torch.Tensor:
    """Naive reference implementation used to pin the fast path."""
    t_len = rewards.numel()
    ratio = (logp_theta - logp_mu).exp()
    v_hat = torch.zeros(t_len, dtype=rewards.dtype)
    r_acc = 0.0
    v_next = 0.0
    v_next_own = 0.0
    xi_next = 1.0
    for t in range(t_len - 1, -1, -1):
        rho_t = min(rho_bar, float(ratio[t]) * xi_next)
        c_t = min(c_bar, float(ratio[t]) * xi_next)
        if bool(is_own[t]):
            delta = rho_t * (float(rewards[t]) + r_acc + v_next_own - float(values[t]))
            v_hat_t = float(values[t]) + delta + c_t * (v_next - v_next_own)
            v_hat[t] = v_hat_t
            r_acc = 0.0
            v_next = v_hat_t
            v_next_own = float(values[t])
            xi_next = 1.0
        else:
            r_acc = float(rewards[t]) + float(ratio[t]) * r_acc
            xi_next = float(ratio[t]) * xi_next
            v_hat[t] = v_next
    return v_hat


class TwoPlayerVTraceTests(unittest.TestCase):
    def test_matches_reference_on_alternating_trajectory(self) -> None:
        torch.manual_seed(0)
        t_len = 9
        rewards = torch.zeros(t_len)
        rewards[-1] = 1.0
        values = torch.linspace(0.1, 0.5, t_len)
        logp_theta = -torch.rand(t_len)
        logp_mu = -torch.rand(t_len)
        is_own = torch.tensor([t % 2 == 0 for t in range(t_len)])
        out = two_player_vtrace(
            rewards=rewards,
            values=values,
            logp_theta=logp_theta,
            logp_mu=logp_mu,
            perspective_is_player_i=is_own,
        )
        expected = _reference_vtrace(rewards, values, logp_theta, logp_mu, is_own)
        self.assertTrue(torch.allclose(out.v_hat, expected, atol=1e-6))

    def test_only_own_steps_on_policy(self) -> None:
        t_len = 5
        rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        values = torch.zeros(t_len)
        logp_theta = torch.zeros(t_len)
        logp_mu = torch.zeros(t_len)
        is_own = torch.ones(t_len, dtype=torch.bool)
        out = two_player_vtrace(
            rewards=rewards,
            values=values,
            logp_theta=logp_theta,
            logp_mu=logp_mu,
            perspective_is_player_i=is_own,
        )
        # On-policy, all own turns, zero values, terminal reward 1 propagates.
        for t in range(t_len):
            self.assertAlmostEqual(float(out.v_hat[t]), 1.0, places=5)

    def test_rejects_shape_mismatch(self) -> None:
        ok = torch.zeros(3)
        bad = torch.zeros(4)
        with self.assertRaises(ValueError):
            two_player_vtrace(
                rewards=ok,
                values=bad,
                logp_theta=ok,
                logp_mu=ok,
                perspective_is_player_i=torch.zeros(3, dtype=torch.bool),
            )

    def test_rejects_empty(self) -> None:
        empty = torch.zeros(0)
        with self.assertRaises(ValueError):
            two_player_vtrace(
                rewards=empty,
                values=empty,
                logp_theta=empty,
                logp_mu=empty,
                perspective_is_player_i=torch.zeros(0, dtype=torch.bool),
            )


class NeuRDLossTests(unittest.TestCase):
    def test_gradient_matches_expected_in_range(self) -> None:
        torch.manual_seed(0)
        t_len, a_len = 3, 4
        logits = torch.zeros(t_len, a_len, requires_grad=True)
        q = torch.randn(t_len, a_len)
        mask = torch.ones(t_len, a_len, dtype=torch.bool)
        loss = neurd_loss(logits=logits, q_hat=q, legal_mask=mask, beta=10.0, clip=1e6)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        expected = -q / t_len
        assert logits.grad is not None
        self.assertTrue(torch.allclose(logits.grad, expected, atol=1e-6))

    def test_gradient_zero_outside_beta(self) -> None:
        logits = torch.full((2, 3), 10.0, requires_grad=True)
        q = torch.ones(2, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)
        loss = neurd_loss(logits=logits, q_hat=q, legal_mask=mask, beta=2.0, clip=1e6)
        loss.backward()
        assert logits.grad is not None
        self.assertTrue(torch.allclose(logits.grad, torch.zeros_like(logits)))

    def test_respects_legal_mask(self) -> None:
        logits = torch.zeros(1, 3, requires_grad=True)
        q = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True]])
        loss = neurd_loss(logits=logits, q_hat=q, legal_mask=mask, beta=10.0, clip=1e6)
        loss.backward()
        assert logits.grad is not None
        self.assertAlmostEqual(float(logits.grad[0, 1]), 0.0)
        self.assertAlmostEqual(float(logits.grad[0, 0]), -1.0)
        self.assertAlmostEqual(float(logits.grad[0, 2]), -3.0)

    def test_clips_q(self) -> None:
        logits = torch.zeros(1, 1, requires_grad=True)
        q = torch.tensor([[1000.0]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        loss = neurd_loss(logits=logits, q_hat=q, legal_mask=mask, beta=10.0, clip=5.0)
        loss.backward()
        assert logits.grad is not None
        self.assertAlmostEqual(float(logits.grad.item()), -5.0)


class CriticLossTests(unittest.TestCase):
    def test_only_own_turns(self) -> None:
        v_theta = torch.tensor([1.0, 10.0, 2.0, 10.0])
        v_hat = torch.tensor([0.0, 0.0, 0.0, 0.0])
        mask = torch.tensor([True, False, True, False])
        loss = critic_loss(v_theta=v_theta, v_hat=v_hat, perspective_is_player_i=mask)
        self.assertAlmostEqual(float(loss), 1.5)

    def test_stops_gradient_on_target(self) -> None:
        v_theta = torch.tensor([0.5], requires_grad=True)
        v_hat = torch.tensor([1.0], requires_grad=True)
        mask = torch.tensor([True])
        loss = critic_loss(v_theta=v_theta, v_hat=v_hat, perspective_is_player_i=mask)
        loss.backward()
        self.assertIsNotNone(v_theta.grad)
        # v_hat gradient should be None (unused in graph) or zero.
        grad = v_hat.grad
        self.assertTrue(grad is None or float(grad.abs().sum()) == 0.0)

    def test_no_own_turns_returns_zero(self) -> None:
        v_theta = torch.ones(3, requires_grad=True)
        v_hat = torch.zeros(3)
        mask = torch.zeros(3, dtype=torch.bool)
        loss = critic_loss(v_theta=v_theta, v_hat=v_hat, perspective_is_player_i=mask)
        self.assertEqual(float(loss), 0.0)


class ThresholdDiscretizeTests(unittest.TestCase):
    def test_drops_low_prob(self) -> None:
        probs = torch.tensor([[0.4, 0.01, 0.59]])
        out = threshold_discretize(probs, eps=0.03, n_disc=16)
        self.assertAlmostEqual(float(out[0, 1]), 0.0, places=5)
        self.assertTrue(math.isclose(float(out.sum()), 1.0, abs_tol=1e-5))

    def test_leaves_unchanged_when_no_survivors(self) -> None:
        tiny = torch.tensor([[0.1, 0.1, 0.1]])
        out = threshold_discretize(tiny, eps=0.5, n_disc=16)
        self.assertTrue(torch.allclose(out, tiny))

    def test_sums_to_one(self) -> None:
        torch.manual_seed(0)
        raw = torch.rand(5, 8)
        probs = raw / raw.sum(dim=-1, keepdim=True)
        out = threshold_discretize(probs, eps=0.03, n_disc=8)
        sums = out.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_rejects_bad_args(self) -> None:
        probs = torch.tensor([[0.5, 0.5]])
        with self.assertRaises(ValueError):
            threshold_discretize(probs, eps=-0.1, n_disc=8)
        with self.assertRaises(ValueError):
            threshold_discretize(probs, eps=0.1, n_disc=0)


class EpisodesFromRolloutStepsTests(unittest.TestCase):
    def test_alternation(self) -> None:
        self.assertEqual(
            episodes_from_rollout_steps([0, 1, 0, 1, 0, 1]),
            [(0, 6)],
        )

    def test_boundary_on_repeat(self) -> None:
        self.assertEqual(
            episodes_from_rollout_steps([0, 1, 0, 0, 1, 0]),
            [(0, 3), (3, 6)],
        )

    def test_empty(self) -> None:
        self.assertEqual(episodes_from_rollout_steps([]), [])


if __name__ == "__main__":
    unittest.main()
