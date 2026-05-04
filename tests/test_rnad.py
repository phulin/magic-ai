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
        self.assertEqual(cfg.delta_m, 1_000)
        self.assertEqual(cfg.num_outer_iterations, 50)
        self.assertAlmostEqual(cfg.neurd_beta, 2.0)
        self.assertGreater(cfg.target_ema_gamma, 0.0)
        self.assertLess(cfg.target_ema_gamma, 1.0)
        # Polyak target-tracking sanity (paper §191): ``delta_m * target_ema``
        # must be large enough that target meaningfully tracks online inside
        # one outer iter. See the warning in scripts/train.py.
        self.assertGreaterEqual(cfg.delta_m * cfg.target_ema_gamma, 0.5)


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
            # Paper §177: the (π/μ)_t factor on r̂_{t+1} is the unclipped
            # own-turn importance ratio.
            pi_mu_t = float(ratio[t])
            delta = rho_t * (float(rewards[t]) + pi_mu_t * r_acc + v_next_own - float(values[t]))
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

    def test_batched_matches_per_episode(self) -> None:
        from magic_ai.rnad import _two_player_vtrace_batched

        torch.manual_seed(7)
        eps = []
        for t_len in (5, 9, 1, 12):
            rewards = torch.zeros(t_len)
            rewards[-1] = 1.0 if torch.rand(()) > 0.5 else -1.0
            values = torch.linspace(0.1, 0.5, t_len)
            logp_theta = -torch.rand(t_len)
            logp_mu = -torch.rand(t_len)
            is_own = torch.tensor([t % 2 == 0 for t in range(t_len)])
            eps.append((rewards, values, logp_theta, logp_mu, is_own))

        per_ep = [
            two_player_vtrace(
                rewards=r,
                values=v,
                logp_theta=lt,
                logp_mu=lm,
                perspective_is_player_i=io,
            )
            for r, v, lt, lm, io in eps
        ]

        flat_rewards = torch.cat([e[0] for e in eps])
        flat_values = torch.cat([e[1] for e in eps])
        flat_lt = torch.cat([e[2] for e in eps])
        flat_lm = torch.cat([e[3] for e in eps])
        flat_io = torch.cat([e[4] for e in eps])
        offsets = torch.tensor(
            [0, *torch.cumsum(torch.tensor([e[0].numel() for e in eps]), 0).tolist()],
            dtype=torch.long,
        )
        out = _two_player_vtrace_batched(
            rewards=flat_rewards,
            values=flat_values,
            logp_theta=flat_lt,
            logp_mu=flat_lm,
            perspective_is_player_i=flat_io,
            ep_offsets=offsets,
        )
        expected_v_hat = torch.cat([p.v_hat for p in per_ep])
        expected_q_hat = torch.cat([p.q_hat for p in per_ep])
        expected_r_next = torch.cat([p.r_hat_next for p in per_ep])
        expected_v_next = torch.cat([p.v_hat_next for p in per_ep])
        self.assertTrue(torch.allclose(out.v_hat, expected_v_hat, atol=1e-6))
        self.assertTrue(torch.allclose(out.q_hat, expected_q_hat, atol=1e-6))
        self.assertTrue(torch.allclose(out.r_hat_next, expected_r_next, atol=1e-6))
        self.assertTrue(torch.allclose(out.v_hat_next, expected_v_next, atol=1e-6))

    def test_batched_handles_ragged_own_turn_counts(self) -> None:
        """Multi-episode batches with very different own-turn counts and a
        single-own-turn episode — exercises the padded reverse-scan path."""
        from magic_ai.rnad import _two_player_vtrace_batched

        torch.manual_seed(13)
        eps = []
        # Mixed: long with even own-turns, short with one own-turn,
        # one episode with no own-turns at all (all opponent steps),
        # one with a streak of own-turns at the end.
        configs = [
            ([t % 2 == 0 for t in range(11)], 11),
            ([True], 1),
            ([False, False, False], 3),
            ([False, False, True, True, True, True], 6),
            ([t % 3 == 0 for t in range(8)], 8),
        ]
        for is_own_pattern, t_len in configs:
            rewards = torch.zeros(t_len)
            rewards[-1] = 1.0 if torch.rand(()) > 0.5 else -1.0
            values = torch.randn(t_len) * 0.3
            logp_theta = -torch.rand(t_len) * 0.5
            logp_mu = -torch.rand(t_len) * 0.5
            is_own = torch.tensor(is_own_pattern)
            eps.append((rewards, values, logp_theta, logp_mu, is_own))

        per_ep = [
            two_player_vtrace(
                rewards=r,
                values=v,
                logp_theta=lt,
                logp_mu=lm,
                perspective_is_player_i=io,
            )
            for r, v, lt, lm, io in eps
        ]

        flat_rewards = torch.cat([e[0] for e in eps])
        flat_values = torch.cat([e[1] for e in eps])
        flat_lt = torch.cat([e[2] for e in eps])
        flat_lm = torch.cat([e[3] for e in eps])
        flat_io = torch.cat([e[4] for e in eps])
        offsets = torch.tensor(
            [0, *torch.cumsum(torch.tensor([e[0].numel() for e in eps]), 0).tolist()],
            dtype=torch.long,
        )
        out = _two_player_vtrace_batched(
            rewards=flat_rewards,
            values=flat_values,
            logp_theta=flat_lt,
            logp_mu=flat_lm,
            perspective_is_player_i=flat_io,
            ep_offsets=offsets,
        )
        expected_v_hat = torch.cat([p.v_hat for p in per_ep])
        expected_q_hat = torch.cat([p.q_hat for p in per_ep])
        expected_r_next = torch.cat([p.r_hat_next for p in per_ep])
        expected_v_next = torch.cat([p.v_hat_next for p in per_ep])
        self.assertTrue(torch.allclose(out.v_hat, expected_v_hat, atol=1e-5))
        self.assertTrue(torch.allclose(out.q_hat, expected_q_hat, atol=1e-5))
        self.assertTrue(torch.allclose(out.r_hat_next, expected_r_next, atol=1e-5))
        self.assertTrue(torch.allclose(out.v_hat_next, expected_v_next, atol=1e-5))


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
        loss_sum, count = critic_loss(v_theta=v_theta, v_hat=v_hat, perspective_is_player_i=mask)
        self.assertEqual(int(count), 2)
        # sum |1-0| + |2-0| = 3; mean = 1.5.
        self.assertAlmostEqual(float(loss_sum) / int(count), 1.5)

    def test_stops_gradient_on_target(self) -> None:
        v_theta = torch.tensor([0.5], requires_grad=True)
        v_hat = torch.tensor([1.0], requires_grad=True)
        mask = torch.tensor([True])
        loss_sum, _count = critic_loss(v_theta=v_theta, v_hat=v_hat, perspective_is_player_i=mask)
        loss_sum.backward()
        self.assertIsNotNone(v_theta.grad)
        grad = v_hat.grad
        self.assertTrue(grad is None or float(grad.abs().sum()) == 0.0)

    def test_no_own_turns_returns_zero(self) -> None:
        v_theta = torch.ones(3, requires_grad=True)
        v_hat = torch.zeros(3)
        mask = torch.zeros(3, dtype=torch.bool)
        loss_sum, count = critic_loss(v_theta=v_theta, v_hat=v_hat, perspective_is_player_i=mask)
        self.assertEqual(int(count), 0)
        self.assertEqual(float(loss_sum), 0.0)


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

    def test_does_not_touch_actor_runtime_buffers(self) -> None:
        """Issue 1/8: live LSTM cache and rollout buffer entries are actor
        runtime state, not model state. Polyak averaging must skip them so
        target/reg never inherit (or accidentally mutate) the online actor's
        per-env LSTM cache."""

        class FakeWithRuntimeBuffers(nn.Module):
            def __init__(self, h_value: float, c_value: float) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.register_buffer(
                    "live_lstm_h", torch.full((2, 4, 8), h_value), persistent=False
                )
                self.register_buffer(
                    "live_lstm_c", torch.full((2, 4, 8), c_value), persistent=False
                )

        online = FakeWithRuntimeBuffers(1.0, 2.0)
        target = FakeWithRuntimeBuffers(99.0, -99.0)
        polyak_update_(target, online, gamma=1.0)
        # Trainable params copy fully.
        for p_o, p_t in zip(online.linear.parameters(), target.linear.parameters(), strict=True):
            self.assertTrue(torch.allclose(p_o, p_t))
        # live_lstm_* buffers must remain at target's pre-call values.
        target_h = target.get_buffer("live_lstm_h")
        target_c = target.get_buffer("live_lstm_c")
        self.assertTrue(torch.allclose(target_h, torch.full_like(target_h, 99.0)))
        self.assertTrue(torch.allclose(target_c, torch.full_like(target_c, -99.0)))


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

    def evaluate_replay_batch_per_choice(
        self,
        replay_rows: list[int],
        *,
        lstm_state_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        hidden_override: torch.Tensor | None = None,
    ):  # type: ignore[no-untyped-def]
        del lstm_state_override
        del hidden_override
        from magic_ai.replay_decisions import ReplayPerChoice

        idx = torch.tensor(replay_rows, dtype=torch.long)
        n = int(idx.numel())
        values = self.values[idx]
        # One decision group per step. ``group_idx`` (per-flat -> step) and
        # ``decision_group_id_flat`` (per-flat -> decision-group) coincide
        # here because each step has exactly one group.
        group_idx = torch.arange(n, dtype=torch.long).repeat_interleave(self.num_choices)
        choice_cols = torch.arange(self.num_choices).repeat(n)
        flat_logits = self.per_choice_logits[idx].reshape(-1)
        flat_log_probs = torch.log_softmax(self.per_choice_logits[idx], dim=-1).reshape(-1)
        logp = flat_log_probs.reshape(n, self.num_choices)[:, 0]  # sampled col 0
        entropies = torch.zeros_like(logp)
        is_sampled_flat = choice_cols == 0
        decision_group_id_flat = group_idx.clone()
        step_for_decision_group = torch.arange(n, dtype=torch.long)
        return (
            logp,
            entropies,
            values,
            ReplayPerChoice(
                flat_logits=flat_logits,
                flat_log_probs=flat_log_probs,
                group_idx=group_idx,
                choice_cols=choice_cols,
                is_sampled_flat=is_sampled_flat,
                decision_group_id_flat=decision_group_id_flat,
                step_for_decision_group=step_for_decision_group,
                may_is_active=torch.zeros(n, dtype=torch.bool),
                may_logits_per_step=torch.zeros(n),
                may_selected_per_step=torch.zeros(n),
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
            terminal_reward_p0=1.0,
            zero_sum=True,
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
            terminal_reward_p0=1.0,
            zero_sum=True,
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
            terminal_reward_p0=1.0,
            zero_sum=True,
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
                terminal_reward_p0=1.0,
                zero_sum=True,
                logp_mu=torch.zeros(3),
                config=RNaDConfig(),
                alpha=1.0,
            )

    def test_full_neurd_per_action_regularization_touches_unsampled_choices(
        self,
    ) -> None:
        """Paper §179-180: Q(a) includes a -eta*log_ratio(a) term for every
        legal a, so unsampled choices must receive nonzero NeuRD gradient
        as soon as pi_theta and pi_reg diverge."""
        t_len = 4
        online = self._stub(t_len, logp=-1.0, value=0.0)
        # Give online different per-choice logits from the regs so the
        # log_ratio per action is nonzero.
        with torch.no_grad():
            online.per_choice_logits.copy_(
                torch.tensor(
                    [[2.0, 0.0, -2.0], [-2.0, 0.0, 2.0], [0.0, 1.0, -1.0], [1.0, -1.0, 0.0]]
                )
            )
        target = self._stub(t_len, logp=-1.0, value=0.0)
        reg_cur = self._stub(t_len, logp=-1.0, value=0.0)
        reg_prev = self._stub(t_len, logp=-1.0, value=0.0)
        opt = torch.optim.SGD(online.parameters(), lr=1.0)
        before = online.per_choice_logits.detach().clone()
        rnad_update_trajectory(
            online=cast(Any, online),
            target=cast(Any, target),
            reg_cur=cast(Any, reg_cur),
            reg_prev=cast(Any, reg_prev),
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0, 1],
            terminal_reward_p0=1.0,
            zero_sum=True,
            logp_mu=torch.full((t_len,), -1.0),
            config=RNaDConfig(eta=0.5, neurd_beta=100.0),
            alpha=1.0,
        )
        after = online.per_choice_logits.detach()
        # Sampled col 0 must have moved (has both base Q and sampled correction).
        self.assertFalse(torch.allclose(after[0, 0], before[0, 0]))
        # Unsampled cols at own-turn steps must also have moved — that's the
        # paper's -eta*log_ratio(a) term, which is the whole point of the
        # full per-action Q form.
        own_step_unsampled_moved = (
            (after[0, 1] != before[0, 1]).item()
            or (after[0, 2] != before[0, 2]).item()
            or (after[2, 1] != before[2, 1]).item()
            or (after[2, 2] != before[2, 2]).item()
        )
        self.assertTrue(own_step_unsampled_moved)

    def test_full_neurd_trains_may_head_on_may_steps(self) -> None:
        """Full-NeuRD must not drop the may head from policy training on
        steps whose action came from the Bernoulli may branch."""
        from magic_ai.replay_decisions import ReplayPerChoice

        class MayStub(nn.Module):
            def __init__(self, t_len: int) -> None:
                super().__init__()
                self.logp = nn.Parameter(torch.zeros(t_len))
                self.values = nn.Parameter(torch.zeros(t_len))
                self.may_logits = nn.Parameter(torch.zeros(t_len))

            def evaluate_replay_batch(self, rows: list[int]):  # type: ignore[no-untyped-def]
                idx = torch.tensor(rows, dtype=torch.long)
                return (
                    self.logp[idx],
                    torch.zeros_like(self.logp[idx]),
                    self.values[idx],
                    None,
                )

            def evaluate_replay_batch_per_choice(
                self,
                rows: list[int],
                *,
                lstm_state_override: tuple[torch.Tensor, torch.Tensor] | None = None,
            ):  # type: ignore[no-untyped-def]
                del lstm_state_override
                idx = torch.tensor(rows, dtype=torch.long)
                n = int(idx.numel())
                return (
                    self.logp[idx],
                    torch.zeros_like(self.logp[idx]),
                    self.values[idx],
                    ReplayPerChoice(
                        flat_logits=torch.zeros(0),
                        flat_log_probs=torch.zeros(0),
                        group_idx=torch.zeros(0, dtype=torch.long),
                        choice_cols=torch.zeros(0, dtype=torch.long),
                        is_sampled_flat=torch.zeros(0, dtype=torch.bool),
                        decision_group_id_flat=torch.zeros(0, dtype=torch.long),
                        step_for_decision_group=torch.zeros(0, dtype=torch.long),
                        may_is_active=torch.ones(n, dtype=torch.bool),
                        may_logits_per_step=self.may_logits[idx],
                        may_selected_per_step=torch.ones(n),
                    ),
                )

        t_len = 4
        online = MayStub(t_len)
        target = MayStub(t_len)
        reg_cur = MayStub(t_len)
        reg_prev = MayStub(t_len)
        opt = torch.optim.SGD(online.parameters(), lr=1.0)
        may_before = online.may_logits.detach().clone()
        rnad_update_trajectory(
            online=cast(Any, online),
            target=cast(Any, target),
            reg_cur=cast(Any, reg_cur),
            reg_prev=cast(Any, reg_prev),
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0, 1],
            terminal_reward_p0=1.0,
            zero_sum=True,
            logp_mu=torch.zeros(t_len),
            config=RNaDConfig(eta=0.0),
            alpha=1.0,
        )
        # Without the may-head fix, may_logits stays at zeros; with the
        # beta-gated may-head NeuRD term, it moves at own-turn may steps.
        self.assertFalse(torch.allclose(online.may_logits.detach(), may_before))

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
        rnad_update_trajectory(
            online=online,
            target=target,
            reg_cur=reg_cur,
            reg_prev=reg_prev,
            optimizer=opt,
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0, 1],
            terminal_reward_p0=1.0,
            zero_sum=True,
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
            terminal_reward_p0=1.0,
            zero_sum=True,
            logp_mu=torch.full((t_len,), -1.0),
            config=RNaDConfig(eta=0.0),
            alpha=1.0,
        )
        after = online.values.detach()
        # Every timestep belongs to exactly one player; both-player update
        # should nudge every entry (none should be untouched).
        self.assertTrue(torch.all(after != before))


class MayTwoActionNeuRDTests(unittest.TestCase):
    """Issue 5/8: two-action may NeuRD must move BOTH branches' regularization."""

    def test_unsampled_may_branch_receives_reg_gradient(self) -> None:
        from magic_ai.replay_decisions import ReplayPerChoice
        from magic_ai.rnad import rnad_trajectory_loss

        class MayStub(nn.Module):
            def __init__(self, t_len: int, may_logit: float) -> None:
                super().__init__()
                self.logp = nn.Parameter(torch.zeros(t_len))
                self.values = nn.Parameter(torch.zeros(t_len))
                self.may_logits = nn.Parameter(torch.full((t_len,), float(may_logit)))

            def evaluate_replay_batch(self, rows: list[int]):  # type: ignore[no-untyped-def]
                idx = torch.tensor(rows, dtype=torch.long)
                return (
                    self.logp[idx],
                    torch.zeros_like(self.logp[idx]),
                    self.values[idx],
                    None,
                )

            def evaluate_replay_batch_per_choice(
                self,
                rows: list[int],
                *,
                lstm_state_override: tuple[torch.Tensor, torch.Tensor] | None = None,
            ):  # type: ignore[no-untyped-def]
                del lstm_state_override
                idx = torch.tensor(rows, dtype=torch.long)
                n = int(idx.numel())
                return (
                    self.logp[idx],
                    torch.zeros_like(self.logp[idx]),
                    self.values[idx],
                    ReplayPerChoice(
                        flat_logits=torch.zeros(0),
                        flat_log_probs=torch.zeros(0),
                        group_idx=torch.zeros(0, dtype=torch.long),
                        choice_cols=torch.zeros(0, dtype=torch.long),
                        is_sampled_flat=torch.zeros(0, dtype=torch.bool),
                        decision_group_id_flat=torch.zeros(0, dtype=torch.long),
                        step_for_decision_group=torch.zeros(0, dtype=torch.long),
                        may_is_active=torch.ones(n, dtype=torch.bool),
                        may_logits_per_step=self.may_logits[idx],
                        may_selected_per_step=torch.ones(n),  # always sample "accept"
                    ),
                )

        # Online and reg differ on the may head, but the trajectory always
        # samples "accept". With the OLD 1-logit may NeuRD, the unsampled
        # decline branch's eta * log(pi/pi_reg) term would NOT influence the
        # gradient. With the new two-action form it must.
        t_len = 4
        online = MayStub(t_len, may_logit=1.0)
        target = MayStub(t_len, may_logit=1.0)
        reg_cur = MayStub(t_len, may_logit=-1.0)  # diverges -> nonzero log_ratio
        reg_prev = MayStub(t_len, may_logit=-1.0)
        # Reward 0 everywhere to isolate the sampled-correction-free regime.
        # Use eta > 0 so the regularization term carries gradient.
        pieces = rnad_trajectory_loss(
            online=cast(Any, online),
            target=cast(Any, target),
            reg_cur=cast(Any, reg_cur),
            reg_prev=cast(Any, reg_prev),
            replay_rows=list(range(t_len)),
            perspective_player_idx=[0, 1, 0, 1],
            terminal_reward_p0=0.0,  # no terminal reward
            zero_sum=False,
            logp_mu=torch.zeros(t_len),
            config=RNaDConfig(eta=0.5, neurd_beta=100.0),
            alpha=1.0,
        )
        loss = pieces.pl_sum / max(pieces.pl_count, 1)
        loss.backward()
        # NeuRD on a Bernoulli with logit l: gradient =
        #   -[(1-p) Q_a - p Q_d]
        # If only the sampled branch's reg term were applied (old form), an
        # entropy-only run with sampled=accept would yield gradient
        # = (1-p) * eta * log(pi/pi_reg)_accept — never including the decline
        # branch's reg contribution. The new form blends both → different sign
        # and magnitude even when sampled is always "accept". Concretely:
        # since pi_accept(online) > pi_accept(reg) but pi_decline(online) <
        # pi_decline(reg), the two reg terms point in opposite directions
        # and partially cancel — the grad is strictly smaller than the old
        # one-branch form. Easier-to-pin invariant: gradient is finite,
        # nonzero, and shares the *opposite* sign from the one-branch form
        # because the decline branch dominates here.
        assert online.may_logits.grad is not None
        # Gradient must be nonzero on own-turn may steps.
        own_step_idx = [0, 2]
        for i in own_step_idx:
            self.assertNotAlmostEqual(float(online.may_logits.grad[i]), 0.0, places=6)


class FactoredAutoregressiveTests(unittest.TestCase):
    """Issue 4/8: per-group sampled correction stays bounded as the number
    of decision groups in a step grows. Joint 1/mu_t blew up multiplicatively;
    the per-group form does not."""

    def test_per_group_correction_does_not_compound_with_group_count(self) -> None:
        from magic_ai.replay_decisions import ReplayPerChoice

        class MultiGroupStub(nn.Module):
            def __init__(self, t_len: int, num_groups_per_step: int) -> None:
                super().__init__()
                self.t_len = t_len
                self.num_groups_per_step = num_groups_per_step
                self.values = nn.Parameter(torch.zeros(t_len))
                # 2 choices per group, K groups per step.
                self.per_choice = nn.Parameter(torch.zeros(t_len, num_groups_per_step, 2))
                self.logp = nn.Parameter(torch.zeros(t_len))

            def evaluate_replay_batch(self, rows: list[int]):  # type: ignore[no-untyped-def]
                idx = torch.tensor(rows, dtype=torch.long)
                return (
                    self.logp[idx],
                    torch.zeros_like(self.logp[idx]),
                    self.values[idx],
                    None,
                )

            def evaluate_replay_batch_per_choice(
                self,
                rows: list[int],
                *,
                lstm_state_override: tuple[torch.Tensor, torch.Tensor] | None = None,
            ):  # type: ignore[no-untyped-def]
                del lstm_state_override
                idx = torch.tensor(rows, dtype=torch.long)
                n = int(idx.numel())
                k = self.num_groups_per_step
                # Decision group ids: K groups per step, total n*K groups.
                step_for_decision_group = torch.arange(n).repeat_interleave(k)
                # Per-flat: each group has 2 choices.
                decision_group_id_flat = torch.arange(n * k).repeat_interleave(2)
                group_idx_step_for_flat = step_for_decision_group[decision_group_id_flat]
                choice_cols = torch.tensor([0, 1] * (n * k), dtype=torch.long)
                logits_flat = self.per_choice[idx].reshape(-1)
                # log-softmax within each group of 2.
                logits_3d = self.per_choice[idx].reshape(n * k, 2)
                log_probs_3d = torch.log_softmax(logits_3d, dim=-1)
                flat_log_probs = log_probs_3d.reshape(-1)
                # Sampled = col 0 always.
                is_sampled_flat = choice_cols == 0
                # Per-step joint log_prob = sum over groups of log_prob_at_col_0.
                logp_per_step = log_probs_3d[:, 0].reshape(n, k).sum(dim=-1)
                return (
                    logp_per_step,
                    torch.zeros_like(logp_per_step),
                    self.values[idx],
                    ReplayPerChoice(
                        flat_logits=logits_flat,
                        flat_log_probs=flat_log_probs,
                        group_idx=group_idx_step_for_flat,
                        choice_cols=choice_cols,
                        is_sampled_flat=is_sampled_flat,
                        decision_group_id_flat=decision_group_id_flat,
                        step_for_decision_group=step_for_decision_group,
                        may_is_active=torch.zeros(n, dtype=torch.bool),
                        may_logits_per_step=torch.zeros(n),
                        may_selected_per_step=torch.zeros(n),
                    ),
                )

        from magic_ai.rnad import rnad_trajectory_loss

        # logp_mu = K * log(0.5) (joint behavior under uniform per-group). With
        # the OLD joint 1/mu_t = 2^K, doubling K from 2 to 6 would multiply the
        # sampled-correction magnitude by 16x. Per-group 1/mu_k stays at 2.
        results: dict[int, float] = {}
        for k in (2, 6):
            t_len = 2
            online = MultiGroupStub(t_len, k)
            target = MultiGroupStub(t_len, k)
            reg_cur = MultiGroupStub(t_len, k)
            reg_prev = MultiGroupStub(t_len, k)
            joint_logp = float(k) * math.log(0.5)
            pieces = rnad_trajectory_loss(
                online=cast(Any, online),
                target=cast(Any, target),
                reg_cur=cast(Any, reg_cur),
                reg_prev=cast(Any, reg_prev),
                replay_rows=list(range(t_len)),
                perspective_player_idx=[0] * t_len,
                terminal_reward_p0=1.0,  # +1 for player 0 at last step
                zero_sum=True,
                logp_mu=torch.full((t_len,), joint_logp),
                config=RNaDConfig(eta=0.0, neurd_beta=100.0, q_corr_rho_bar=1e9),
                alpha=1.0,
            )
            results[k] = float((pieces.pl_sum / max(pieces.pl_count, 1)).detach().abs())
        # The per-group decomposition keeps the magnitude roughly constant
        # in K; it certainly does NOT scale as 2^K. Allow some growth from
        # extra groups participating but bound it well below the joint
        # blow-up (16x).
        self.assertLess(results[6] / max(results[2], 1e-9), 4.0)


class PerPolicyLSTMRecomputeTests(unittest.TestCase):
    """Issue 2/8: recompute_lstm_states_for_episode rejects empty input
    (would otherwise silently produce zero-length tensors that the
    consumers cannot interpret as 'one episode')."""

    def test_rejects_empty_replay_rows(self) -> None:
        # Use a minimal policy stub that only declares ``use_lstm=True`` and
        # the method under test. Full PPOPolicy construction is too heavy
        # for a unit test — the empty-rows guard is part of the contract
        # exercised even on the stub.
        from magic_ai.slot_encoder.model import PPOPolicy

        # Pull the unbound method off the class and call its empty-input
        # validation; the path raises before touching any of the heavy
        # forward machinery.
        with self.assertRaises(ValueError):
            method: Any = getattr(  # noqa: B009
                PPOPolicy.recompute_lstm_states_for_episode,
                "__wrapped__",
                PPOPolicy.recompute_lstm_states_for_episode,
            )
            method(cast(PPOPolicy, _LstmStubPolicy()), [])


class _LstmStubPolicy:
    """Just enough surface area for ``recompute_lstm_states_for_episode``'s
    ``not replay_rows`` guard to be reachable (needed because the real
    PPOPolicy construction has a long dependency chain)."""

    use_lstm = True


if __name__ == "__main__":
    unittest.main()
