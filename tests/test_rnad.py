"""Unit tests for the R-NaD primitives."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import torch
from magic_ai.rnad import (
    RNaDConfig,
    RNaDStats,
    critic_loss,
    episodes_from_rollout_steps,
    load_reg_snapshot_into,
    neurd_loss,
    polyak_update_,
    rnad_update_trajectory,
    rnad_update_trajectory_full_neurd,
    sampled_neurd_loss,
    save_reg_snapshot,
    threshold_discretize,
    transform_rewards,
    two_player_vtrace,
)
from torch import nn


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


class SampledNeuRDLossTests(unittest.TestCase):
    def test_gradient_is_negative_q_on_own_turns(self) -> None:
        log_prob = torch.zeros(4, requires_grad=True)
        q_hat = torch.tensor([1.0, -2.0, 3.0, -4.0])
        own = torch.tensor([True, True, True, True])
        loss = sampled_neurd_loss(log_prob=log_prob, q_hat=q_hat, own_turn_mask=own, clip=1e6)
        loss.backward()
        assert log_prob.grad is not None
        # loss = -mean(log_prob * q), d/dlogp = -q / N
        self.assertTrue(torch.allclose(log_prob.grad, -q_hat / 4, atol=1e-6))

    def test_ignores_opponent_turns(self) -> None:
        log_prob = torch.zeros(4, requires_grad=True)
        q_hat = torch.tensor([1.0, -2.0, 3.0, -4.0])
        own = torch.tensor([True, False, True, False])
        loss = sampled_neurd_loss(log_prob=log_prob, q_hat=q_hat, own_turn_mask=own, clip=1e6)
        loss.backward()
        assert log_prob.grad is not None
        # Only indices 0 and 2 contribute: grad[0] = -1/2, grad[2] = -3/2.
        expected = torch.tensor([-0.5, 0.0, -1.5, 0.0])
        self.assertTrue(torch.allclose(log_prob.grad, expected, atol=1e-6))

    def test_clips_q(self) -> None:
        log_prob = torch.zeros(1, requires_grad=True)
        q_hat = torch.tensor([1000.0])
        own = torch.tensor([True])
        loss = sampled_neurd_loss(log_prob=log_prob, q_hat=q_hat, own_turn_mask=own, clip=5.0)
        loss.backward()
        assert log_prob.grad is not None
        self.assertAlmostEqual(float(log_prob.grad.item()), -5.0)

    def test_no_own_turns_returns_zero(self) -> None:
        log_prob = torch.zeros(3, requires_grad=True)
        q_hat = torch.ones(3)
        own = torch.zeros(3, dtype=torch.bool)
        loss = sampled_neurd_loss(log_prob=log_prob, q_hat=q_hat, own_turn_mask=own, clip=1.0)
        self.assertEqual(float(loss), 0.0)


class PolyakUpdateTests(unittest.TestCase):
    def _make_module(self, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))

    def test_zero_gamma_leaves_target_unchanged(self) -> None:
        online = self._make_module(0)
        target = self._make_module(1)
        target_snapshot = {k: v.clone() for k, v in target.state_dict().items()}
        polyak_update_(target, online, gamma=0.0)
        for k, v in target.state_dict().items():
            self.assertTrue(torch.allclose(v, target_snapshot[k]))

    def test_full_gamma_copies_online_to_target(self) -> None:
        online = self._make_module(0)
        target = self._make_module(1)
        polyak_update_(target, online, gamma=1.0)
        for k, v in target.state_dict().items():
            self.assertTrue(torch.allclose(v, online.state_dict()[k]))

    def test_partial_gamma_blends(self) -> None:
        online = self._make_module(0)
        target = self._make_module(1)
        online_snap = {k: v.clone() for k, v in online.state_dict().items()}
        target_snap = {k: v.clone() for k, v in target.state_dict().items()}
        gamma = 0.25
        polyak_update_(target, online, gamma=gamma)
        for k, blended in target.state_dict().items():
            expected = gamma * online_snap[k] + (1.0 - gamma) * target_snap[k]
            self.assertTrue(torch.allclose(blended, expected, atol=1e-6))

    def test_rejects_bad_gamma(self) -> None:
        online = self._make_module(0)
        target = self._make_module(1)
        with self.assertRaises(ValueError):
            polyak_update_(target, online, gamma=1.5)
        with self.assertRaises(ValueError):
            polyak_update_(target, online, gamma=-0.1)


class RegSnapshotTests(unittest.TestCase):
    def test_roundtrip_preserves_parameters(self) -> None:
        torch.manual_seed(0)
        src = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
        dst = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reg" / "snap.pt"
            save_reg_snapshot(src, path)
            self.assertTrue(path.exists())
            load_reg_snapshot_into(dst, path)
        for a, b in zip(src.state_dict().values(), dst.state_dict().values(), strict=True):
            self.assertTrue(torch.allclose(a, b))

    def test_load_freezes_gradients(self) -> None:
        src = nn.Linear(3, 2)
        dst = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snap.pt"
            save_reg_snapshot(src, path)
            load_reg_snapshot_into(dst, path)
        for p in dst.parameters():
            self.assertFalse(p.requires_grad)

    def test_save_creates_parent_dirs(self) -> None:
        src = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c" / "snap.pt"
            save_reg_snapshot(src, nested)
            self.assertTrue(nested.exists())


class _StubPolicy(nn.Module):
    """Minimal `evaluate_replay_batch`-compatible policy for unit tests.

    Produces log-probs and values as linear functions of learnable per-row
    parameters keyed by ``replay_idx``, so rnad_update_trajectory can run a
    real Adam step and we can observe its effects.
    """

    def __init__(
        self,
        t_len: int,
        *,
        logp_init: float,
        value_init: float,
        num_choices: int = 3,
    ) -> None:
        super().__init__()
        self.logp = nn.Parameter(torch.full((t_len,), float(logp_init)))
        self.values = nn.Parameter(torch.full((t_len,), float(value_init)))
        self.num_choices = num_choices
        # Per-step per-choice logits; sampled column is choice 0 by construction.
        self.per_choice_logits = nn.Parameter(torch.zeros(t_len, num_choices, dtype=torch.float32))

    def evaluate_replay_batch(
        self, replay_rows: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        idx = torch.tensor(replay_rows, dtype=torch.long)
        logp = self.logp[idx]
        values = self.values[idx]
        entropies = torch.zeros_like(logp)
        return logp, entropies, values, None

    def evaluate_replay_batch_per_choice(self, replay_rows: list[int]):  # type: ignore[no-untyped-def]
        from magic_ai.model import ReplayPerChoice

        idx = torch.tensor(replay_rows, dtype=torch.long)
        n = int(idx.numel())
        values = self.values[idx]
        # Flatten per-step choices; group_idx is the step-in-batch index.
        group_idx = torch.arange(n, dtype=torch.long).repeat_interleave(self.num_choices)
        choice_cols = torch.arange(self.num_choices).repeat(n)
        flat_logits = self.per_choice_logits[idx].reshape(-1)
        flat_log_probs = torch.log_softmax(self.per_choice_logits[idx], dim=-1).reshape(-1)
        logp = flat_log_probs.reshape(n, self.num_choices)[:, 0]  # sampled col 0
        entropies = torch.zeros_like(logp)
        sampled_col_per_step = torch.zeros(n, dtype=torch.long)
        return (
            logp,
            entropies,
            values,
            ReplayPerChoice(
                flat_logits=flat_logits,
                flat_log_probs=flat_log_probs,
                group_idx=group_idx,
                choice_cols=choice_cols,
                sampled_col_per_step=sampled_col_per_step,
                may_is_active=torch.zeros(n, dtype=torch.bool),
            ),
        )


class RNaDUpdateTrajectoryTests(unittest.TestCase):
    def _stub(self, t_len: int, *, logp: float = -1.0, value: float = 0.0) -> _StubPolicy:
        return _StubPolicy(t_len, logp_init=logp, value_init=value)

    def test_runs_and_returns_finite_stats(self) -> None:
        t_len = 6
        online = self._stub(t_len, logp=-1.0, value=0.0)
        target = self._stub(t_len, logp=-1.0, value=0.0)
        reg_cur = self._stub(t_len, logp=-1.0, value=0.0)
        reg_prev = self._stub(t_len, logp=-1.0, value=0.0)
        opt = torch.optim.Adam(online.parameters(), lr=1e-2)
        stats = rnad_update_trajectory(
            online=online,
            target=target,
            reg_cur=reg_cur,
            reg_prev=reg_prev,
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0, 1, 0, 1],
            winner_idx=0,
            logp_mu=torch.full((t_len,), -1.0),
            config=RNaDConfig(delta_m=1, num_outer_iterations=1),
            alpha=1.0,
        )
        self.assertIsInstance(stats, RNaDStats)
        self.assertTrue(math.isfinite(stats.loss))
        self.assertTrue(math.isfinite(stats.critic_loss))
        self.assertTrue(math.isfinite(stats.policy_loss))
        self.assertTrue(math.isfinite(stats.grad_norm))

    def test_critic_moves_toward_terminal_reward_in_on_policy(self) -> None:
        # With on-policy data, zero reg entropy bonus (theta == reg), and a
        # terminal +1 reward on the own player's last step, the v-hat target
        # for every own-turn step is +1; after one Adam step the online value
        # parameters should move up from 0.
        t_len = 5
        online = self._stub(t_len, logp=-1.0, value=0.0)
        target = self._stub(t_len, logp=-1.0, value=0.0)
        reg_cur = self._stub(t_len, logp=-1.0, value=0.0)
        reg_prev = self._stub(t_len, logp=-1.0, value=0.0)
        opt = torch.optim.Adam(online.parameters(), lr=1e-1)
        before = online.values.detach().clone()
        rnad_update_trajectory(
            online=online,
            target=target,
            reg_cur=reg_cur,
            reg_prev=reg_prev,
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 0, 0, 0, 0],  # all own turns
            winner_idx=0,
            logp_mu=torch.full((t_len,), -1.0),
            config=RNaDConfig(eta=0.0),  # disable entropy regularization
            alpha=1.0,
        )
        after = online.values.detach()
        self.assertTrue(torch.all(after > before))

    def test_polyak_updates_target(self) -> None:
        t_len = 3
        online = self._stub(t_len, logp=-1.0, value=0.5)
        # Target starts at a different value so Polyak movement is observable.
        target = self._stub(t_len, logp=-1.0, value=0.0)
        reg_cur = self._stub(t_len, logp=-1.0, value=0.0)
        reg_prev = self._stub(t_len, logp=-1.0, value=0.0)
        opt = torch.optim.Adam(online.parameters(), lr=1e-3)
        target_before = target.values.detach().clone()
        rnad_update_trajectory(
            online=online,
            target=target,
            reg_cur=reg_cur,
            reg_prev=reg_prev,
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0],
            winner_idx=0,
            logp_mu=torch.full((t_len,), -1.0),
            config=RNaDConfig(target_ema_gamma=0.5),
            alpha=1.0,
        )
        # target should have blended ~halfway toward online.
        self.assertTrue(torch.all(target.values.detach() > target_before))

    def test_rejects_mismatched_shapes(self) -> None:
        online = self._stub(3)
        target = self._stub(3)
        reg = self._stub(3)
        opt = torch.optim.Adam(online.parameters(), lr=1e-3)
        with self.assertRaises(ValueError):
            rnad_update_trajectory(
                online=online,
                target=target,
                reg_cur=reg,
                reg_prev=reg,
                optimizer=opt,
                replay_rows=[0, 1, 2],
                perspective_player_idx=[0, 1],  # wrong length
                winner_idx=0,
                logp_mu=torch.zeros(3),
                config=RNaDConfig(),
                alpha=1.0,
            )

    def test_full_neurd_trains_may_head_on_may_steps(self) -> None:
        """Full-NeuRD must not drop the may head from policy training on
        steps whose action came from the Bernoulli may branch."""
        from magic_ai.model import ReplayPerChoice

        class MayStub(nn.Module):
            def __init__(self, t_len: int) -> None:
                super().__init__()
                self.logp = nn.Parameter(torch.zeros(t_len))
                self.values = nn.Parameter(torch.zeros(t_len))

            def evaluate_replay_batch(self, rows: list[int]):  # type: ignore[no-untyped-def]
                idx = torch.tensor(rows, dtype=torch.long)
                return (
                    self.logp[idx],
                    torch.zeros_like(self.logp[idx]),
                    self.values[idx],
                    None,
                )

            def evaluate_replay_batch_per_choice(self, rows: list[int]):  # type: ignore[no-untyped-def]
                idx = torch.tensor(rows, dtype=torch.long)
                n = int(idx.numel())
                # All steps are may-kind: no decision-group entries at all.
                return (
                    self.logp[idx],
                    torch.zeros_like(self.logp[idx]),
                    self.values[idx],
                    ReplayPerChoice(
                        flat_logits=torch.zeros(0),
                        flat_log_probs=torch.zeros(0),
                        group_idx=torch.zeros(0, dtype=torch.long),
                        choice_cols=torch.zeros(0, dtype=torch.long),
                        sampled_col_per_step=torch.full((n,), -1, dtype=torch.long),
                        may_is_active=torch.ones(n, dtype=torch.bool),
                    ),
                )

        t_len = 4
        online = MayStub(t_len)
        target = MayStub(t_len)
        reg_cur = MayStub(t_len)
        reg_prev = MayStub(t_len)
        opt = torch.optim.SGD(online.parameters(), lr=1.0)
        logp_before = online.logp.detach().clone()
        rnad_update_trajectory_full_neurd(
            online=cast(Any, online),
            target=cast(Any, target),
            reg_cur=cast(Any, reg_cur),
            reg_prev=cast(Any, reg_prev),
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0, 1],
            winner_idx=0,
            logp_mu=torch.zeros(t_len),
            config=RNaDConfig(eta=0.0),
            alpha=1.0,
        )
        # Without the may-head fix, logp would stay at zeros; with the fix
        # the sampled-action NeuRD term moves it at own-turn may steps.
        self.assertFalse(torch.allclose(online.logp.detach(), logp_before))

    def test_full_neurd_variant_runs_and_moves_per_choice_logits(self) -> None:
        """Full per-action NeuRD should deposit gradient on per-choice
        logits at own-turn sampled entries."""
        t_len = 4
        online = self._stub(t_len, logp=-1.0, value=0.0)
        target = self._stub(t_len, logp=-1.0, value=0.0)
        reg_cur = self._stub(t_len, logp=-1.0, value=0.0)
        reg_prev = self._stub(t_len, logp=-1.0, value=0.0)
        opt = torch.optim.SGD(online.parameters(), lr=1.0)
        before = online.per_choice_logits.detach().clone()
        rnad_update_trajectory_full_neurd(
            online=online,
            target=target,
            reg_cur=reg_cur,
            reg_prev=reg_prev,
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0, 1],
            winner_idx=0,
            logp_mu=torch.full((t_len,), -1.0),
            config=RNaDConfig(eta=0.0, neurd_beta=100.0),
            alpha=1.0,
        )
        after = online.per_choice_logits.detach()
        # Steps carrying nonzero Q (closest to the terminal +1 / -1 reward)
        # must have moved on their sampled column (col 0).
        self.assertFalse(torch.equal(after[2, 0], before[2, 0]))
        self.assertFalse(torch.equal(after[3, 0], before[3, 0]))
        # Non-sampled columns get zero per-entry Q -> no movement.
        self.assertTrue(torch.allclose(after[:, 1], before[:, 1]))
        self.assertTrue(torch.allclose(after[:, 2], before[:, 2]))

    def test_both_players_receive_critic_signal(self) -> None:
        """The self-play update must train values on both perspective
        players, not only the first-step player's side."""
        t_len = 4
        online = self._stub(t_len, logp=-1.0, value=0.0)
        target = self._stub(t_len, logp=-1.0, value=0.0)
        reg_cur = self._stub(t_len, logp=-1.0, value=0.0)
        reg_prev = self._stub(t_len, logp=-1.0, value=0.0)
        opt = torch.optim.SGD(online.parameters(), lr=1.0)
        before = online.values.detach().clone()
        rnad_update_trajectory(
            online=online,
            target=target,
            reg_cur=reg_cur,
            reg_prev=reg_prev,
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0, 1],
            winner_idx=0,
            logp_mu=torch.full((t_len,), -1.0),
            config=RNaDConfig(eta=0.0),
            alpha=1.0,
        )
        after = online.values.detach()
        # Every timestep belongs to exactly one player; both-player update
        # should nudge every entry (none should be untouched).
        self.assertTrue(torch.all(after != before))


if __name__ == "__main__":
    unittest.main()
