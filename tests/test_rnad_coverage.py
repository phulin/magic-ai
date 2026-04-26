"""Coverage tests beyond the primitives for the four 'not yet proven' R-NaD claims.

Each test targets one of the open items from the NFG-test summary:

1. Two-player v-trace end-to-end on a multi-step alternating-turn toy
   game (beyond the static algebraic pin in test_rnad.py).
2. Target-network Polyak stabilization: compare convergence with vs
   without Polyak averaging on a noisy estimator.
3. beta-gate logit-explosion guard: a NeuRD loop with constant positive Q
   explodes without the gate and stays bounded with it.
4. Full per-action NeuRD (not the sampled-action estimator): using
   :func:`neurd_loss_per_choice`, verify it matches :func:`neurd_loss`
   in the dense case and works with ragged group data.
"""

from __future__ import annotations

import unittest

import torch
from magic_ai.rnad import (
    neurd_loss,
    neurd_loss_per_choice,
    polyak_update_,
    two_player_vtrace,
)

# ---------------------------------------------------------------------------
# 1. Two-player v-trace end-to-end on a multi-step alternating-turn toy.
# ---------------------------------------------------------------------------


class TwoPlayerVTraceIntegrationTests(unittest.TestCase):
    def test_alternating_turns_propagate_terminal_reward(self) -> None:
        """Alternating-turn trajectory with own-side terminal +1 should
        produce v_hat ~ +1 for all own-turn steps and the carryover value
        on opponent-turn steps."""
        # Trajectory: P0, P1, P0, P1, P0 (own=P0, terminal win for P0).
        t_len = 5
        rewards = torch.zeros(t_len)
        rewards[-1] = 1.0
        values = torch.full((t_len,), 0.25)  # deliberately wrong to see v_hat override
        logp_theta = torch.zeros(t_len)
        logp_mu = torch.zeros(t_len)  # on-policy
        is_own = torch.tensor([True, False, True, False, True])
        out = two_player_vtrace(
            rewards=rewards,
            values=values,
            logp_theta=logp_theta,
            logp_mu=logp_mu,
            perspective_is_player_i=is_own,
        )
        # Own-turn steps should all see v_hat >= 0.25 (terminal reward flowing
        # back through the alternating chain).
        for t in range(t_len):
            if bool(is_own[t]):
                self.assertGreaterEqual(float(out.v_hat[t]), 0.25)
        # Terminal own step picks up the reward directly.
        self.assertGreater(float(out.v_hat[-1]), 0.9)

    def test_opponent_only_trajectory_carries_reward_through(self) -> None:
        """A degenerate trajectory where the own player acts only at
        terminal: intermediate opponent turns must not clobber the signal."""
        t_len = 4
        rewards = torch.zeros(t_len)
        rewards[-1] = 1.0
        values = torch.zeros(t_len)
        logp_theta = torch.zeros(t_len)
        logp_mu = torch.zeros(t_len)
        is_own = torch.tensor([False, False, False, True])
        out = two_player_vtrace(
            rewards=rewards,
            values=values,
            logp_theta=logp_theta,
            logp_mu=logp_mu,
            perspective_is_player_i=is_own,
        )
        # Terminal own-turn step gets the reward.
        self.assertAlmostEqual(float(out.v_hat[-1]), 1.0, places=5)

    def test_importance_clip_bounds_off_policy_estimate(self) -> None:
        """rho_bar/c_bar clip ensures off-policy ratios > 1 cannot
        arbitrarily inflate v_hat."""
        rewards = torch.tensor([0.0, 0.0, 2.0])
        values = torch.zeros(3)
        logp_theta = torch.tensor([0.0, 0.0, 0.0])
        is_own = torch.tensor([True, False, False])
        out_on = two_player_vtrace(
            rewards=rewards,
            values=values,
            logp_theta=logp_theta,
            logp_mu=torch.tensor([0.0, 0.0, 0.0]),
            perspective_is_player_i=is_own,
        )
        # Off-policy: mu on terminal opp step = 0.5, so ratio = 2.
        logp_mu_off = torch.tensor([0.0, 0.0, float(torch.tensor(0.5).log())])
        out_off = two_player_vtrace(
            rewards=rewards,
            values=values,
            logp_theta=logp_theta,
            logp_mu=logp_mu_off,
            perspective_is_player_i=is_own,
        )
        # Both on-policy and clipped off-policy converge to the same
        # bounded estimate at the own-turn step; without rho_bar/c_bar
        # clipping the off-policy version would be strictly larger.
        self.assertAlmostEqual(float(out_on.v_hat[0]), 2.0, places=5)
        self.assertAlmostEqual(float(out_off.v_hat[0]), 2.0, places=5)


# ---------------------------------------------------------------------------
# 2. Target-network Polyak stabilization.
# ---------------------------------------------------------------------------


class PolyakStabilizationTests(unittest.TestCase):
    def test_polyak_reduces_variance_of_tracked_signal(self) -> None:
        """Given a noisy online parameter, the Polyak target should track
        the mean with lower variance than the instantaneous signal."""
        from torch import nn

        online = nn.Linear(1, 1, bias=False)
        target = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            online.weight.fill_(0.0)
            target.weight.fill_(0.0)
        gamma = 0.05
        torch.manual_seed(0)
        online_samples: list[float] = []
        target_samples: list[float] = []
        for _ in range(2000):
            with torch.no_grad():
                # Simulate noisy gradient updates: online drifts around 1.0.
                noise = float(torch.randn(1)) * 0.5
                online.weight.fill_(1.0 + noise)
            polyak_update_(target, online, gamma=gamma)
            online_samples.append(float(online.weight.squeeze()))
            target_samples.append(float(target.weight.squeeze()))
        online_t = torch.tensor(online_samples)
        target_t = torch.tensor(target_samples)
        # After burn-in, target mean matches online mean but variance is smaller.
        burn = 500
        self.assertAlmostEqual(
            float(online_t[burn:].mean()), float(target_t[burn:].mean()), delta=0.1
        )
        self.assertLess(float(target_t[burn:].var()), float(online_t[burn:].var()))


# ---------------------------------------------------------------------------
# 3. beta-gate logit-explosion guard.
# ---------------------------------------------------------------------------


class BetaGateStabilizationTests(unittest.TestCase):
    def test_gate_bounds_logits_under_constant_positive_q(self) -> None:
        """Pushing NeuRD with a constant positive Q signal: without the
        beta-gate, Adam drives logits to arbitrarily large values; with
        the gate (beta=2), the update stops once the logit crosses the
        threshold, so the magnitude stays close to beta."""
        torch.manual_seed(0)
        logits_gated = torch.zeros(1, 2, requires_grad=True)
        opt_gated = torch.optim.SGD([logits_gated], lr=0.1)
        q = torch.tensor([[1.0, 0.0]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        for _ in range(200):
            loss = neurd_loss(logits=logits_gated, q_hat=q, legal_mask=mask, beta=2.0, clip=1e6)
            opt_gated.zero_grad(set_to_none=True)
            loss.backward()
            opt_gated.step()
        # With gate: logit saturates around beta (+ one-step overshoot).
        max_gated = float(logits_gated.detach().abs().max())
        self.assertLess(max_gated, 3.0)

        # Without gate: logits grow linearly per step, well past beta.
        torch.manual_seed(0)
        logits_open = torch.zeros(1, 2, requires_grad=True)
        opt_open = torch.optim.SGD([logits_open], lr=0.1)
        for _ in range(200):
            loss = neurd_loss(logits=logits_open, q_hat=q, legal_mask=mask, beta=1e6, clip=1e6)
            opt_open.zero_grad(set_to_none=True)
            loss.backward()
            opt_open.step()
        max_open = float(logits_open.detach().abs().max())
        self.assertGreater(max_open, 15.0)


# ---------------------------------------------------------------------------
# 4. Full per-action NeuRD (ragged-group form).
# ---------------------------------------------------------------------------


class NeuRDPerChoiceTests(unittest.TestCase):
    def test_matches_dense_form_when_all_groups_same_size(self) -> None:
        """Ragged per-choice NeuRD with uniform group size == dense NeuRD."""
        torch.manual_seed(0)
        t_len, a_len = 3, 4
        logits_dense = torch.randn(t_len, a_len, requires_grad=True)
        q_dense = torch.randn(t_len, a_len)
        mask = torch.ones(t_len, a_len, dtype=torch.bool)
        loss_dense = neurd_loss(
            logits=logits_dense, q_hat=q_dense, legal_mask=mask, beta=100.0, clip=1e6
        )
        loss_dense.backward()
        grad_dense = logits_dense.grad
        assert grad_dense is not None

        # Same data in flat per-choice form. The ragged form returns
        # ``(sum_loss, n_active, n_clipped)`` (issue 7); divide by ``t_len``
        # to match the dense form's per-step normalization.
        logits_flat = logits_dense.detach().clone().reshape(-1).requires_grad_(True)
        q_flat = q_dense.reshape(-1)
        loss_flat_sum, _n_active, _n_clipped = neurd_loss_per_choice(
            flat_logits=logits_flat,
            flat_q=q_flat,
            beta=100.0,
            clip=1e6,
        )
        (loss_flat_sum / t_len).backward()
        grad_flat = logits_flat.grad
        assert grad_flat is not None
        self.assertTrue(torch.allclose(grad_flat.reshape(t_len, a_len), grad_dense, atol=1e-6))

    def test_ragged_groups_emit_per_action_gradient(self) -> None:
        # 2 decision steps, first has 2 legal choices, second has 5.
        logits = torch.zeros(7, requires_grad=True)
        q = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        loss_sum, n_active, _n_clipped = neurd_loss_per_choice(
            flat_logits=logits,
            flat_q=q,
            beta=100.0,
            clip=1e6,
        )
        self.assertEqual(n_active, 7)
        loss_sum.backward()
        assert logits.grad is not None
        # Sum form gradient is just -q (no per-group division).
        self.assertTrue(torch.allclose(logits.grad, -q, atol=1e-6))

    def test_per_choice_respects_beta_gate(self) -> None:
        logits = torch.full((4,), 100.0, requires_grad=True)
        q = torch.ones(4)
        loss_sum, _n_active, _n_clipped = neurd_loss_per_choice(
            flat_logits=logits,
            flat_q=q,
            beta=2.0,
            clip=1e6,
        )
        loss_sum.backward()
        assert logits.grad is not None
        self.assertTrue(torch.allclose(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
