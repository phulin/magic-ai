"""Integration tests for the R-NaD trainer-state glue."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from magic_ai.game_state import GameStateEncoder
from magic_ai.model import PPOPolicy
from magic_ai.rnad import RNaDConfig
from magic_ai.rnad_trainer import (
    _advance_outer_iteration,
    _alpha_for_step,
    _clone_policy_sharing_buffer,
    build_trainer_state,
    resume_from_snapshot_dir,
)


def _make_policy() -> PPOPolicy:
    encoder = GameStateEncoder({"Mountain": [0.1, 0.2, 0.3]}, d_model=8)
    return PPOPolicy(
        encoder,
        hidden_dim=16,
        hidden_layers=1,
        max_options=4,
        max_targets_per_option=2,
        rollout_capacity=16,
    )


class AlphaScheduleTests(unittest.TestCase):
    def test_linear_ramp(self) -> None:
        self.assertAlmostEqual(_alpha_for_step(0, 10), 0.0)
        self.assertAlmostEqual(_alpha_for_step(5, 10), 1.0)
        self.assertAlmostEqual(_alpha_for_step(20, 10), 1.0)

    def test_zero_delta_returns_one(self) -> None:
        self.assertAlmostEqual(_alpha_for_step(0, 0), 1.0)


class ClonePolicyTests(unittest.TestCase):
    def test_clone_shares_rollout_buffer(self) -> None:
        policy = _make_policy()
        clone = _clone_policy_sharing_buffer(policy)
        self.assertIs(clone.rollout_buffer, policy.rollout_buffer)

    def test_clone_parameters_are_independent(self) -> None:
        policy = _make_policy()
        clone = _clone_policy_sharing_buffer(policy)
        clone.value_head.weight.data.fill_(42.0)
        self.assertFalse(torch.equal(policy.value_head.weight.data, clone.value_head.weight.data))


class TrainerStateTests(unittest.TestCase):
    def test_build_persists_initial_snapshot(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmp:
            state = build_trainer_state(
                policy,
                config=RNaDConfig(delta_m=5),
                reg_snapshot_dir=Path(tmp),
                device=policy.device,
            )
            self.assertTrue((Path(tmp) / "reg_m000.pt").exists())
            # target/regs share the buffer with online.
            self.assertIs(state.target.rollout_buffer, policy.rollout_buffer)
            self.assertIs(state.reg_cur.rollout_buffer, policy.rollout_buffer)
            self.assertIs(state.reg_prev.rollout_buffer, policy.rollout_buffer)
            for p in state.target.parameters():
                self.assertFalse(p.requires_grad)
            for p in state.reg_cur.parameters():
                self.assertFalse(p.requires_grad)

    def test_advance_outer_iteration_shifts_regs(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmp:
            state = build_trainer_state(
                policy,
                config=RNaDConfig(delta_m=5),
                reg_snapshot_dir=Path(tmp),
                device=policy.device,
            )
            # Perturb target so the advance produces observable changes.
            with torch.no_grad():
                state.target.value_head.weight.fill_(0.75)
            reg_cur_before = state.reg_cur.value_head.weight.clone()
            _advance_outer_iteration(state)
            self.assertEqual(state.outer_iteration, 1)
            self.assertEqual(state.gradient_step, 0)
            # reg_prev should now equal the previous reg_cur value.
            self.assertTrue(torch.allclose(state.reg_prev.value_head.weight, reg_cur_before))
            # reg_cur should now equal target.
            self.assertTrue(torch.allclose(state.reg_cur.value_head.weight, torch.tensor(0.75)))
            self.assertTrue((Path(tmp) / "reg_m001.pt").exists())

    def test_resume_loads_snapshots(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmp:
            state = build_trainer_state(
                policy,
                config=RNaDConfig(delta_m=5),
                reg_snapshot_dir=Path(tmp),
                device=policy.device,
            )
            with torch.no_grad():
                state.target.value_head.weight.fill_(0.5)
            _advance_outer_iteration(state)
            # Clobber the in-memory regs so resume has real work to do.
            with torch.no_grad():
                state.reg_cur.value_head.weight.fill_(-99.0)
                state.reg_prev.value_head.weight.fill_(-99.0)
            resume_from_snapshot_dir(state, outer_iteration=1, gradient_step=3)
        self.assertEqual(state.outer_iteration, 1)
        self.assertEqual(state.gradient_step, 3)
        self.assertTrue(
            torch.allclose(state.reg_cur.value_head.weight, torch.tensor(0.5)),
        )


if __name__ == "__main__":
    unittest.main()
