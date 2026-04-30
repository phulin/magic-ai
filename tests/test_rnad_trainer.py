"""Integration tests for the R-NaD trainer-state glue."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch
from magic_ai.rnad import RNaDConfig
from magic_ai.rnad_trainer import (
    _advance_outer_iteration,
    _alpha_for_step,
    _delta_m_for_outer_iteration,
    build_trainer_state,
    resume_from_snapshot_dir,
)
from magic_ai.slot_encoder.game_state import GameStateEncoder
from magic_ai.slot_encoder.model import PPOPolicy
from torch import nn


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

    def test_first_two_outer_intervals_are_half_delta_m(self) -> None:
        intervals = [_delta_m_for_outer_iteration(outer_iteration=m, delta_m=500) for m in range(5)]
        self.assertEqual(intervals, [250, 250, 500, 500, 500])


class ClonePolicyTests(unittest.TestCase):
    def test_clone_shares_rollout_buffer(self) -> None:
        policy = _make_policy()
        clone = policy.clone_for_rnad()
        self.assertIs(clone.rollout_buffer, policy.rollout_buffer)
        # nn.Module tracks submodules in ``_modules``; if the buffer is only
        # spliced into ``__dict__`` then ``.to(device)`` and ``named_buffers``
        # still walk a hidden deep-copy, defeating the memory sharing.
        self.assertIs(clone._modules["rollout_buffer"], policy.rollout_buffer)
        self.assertIs(policy._modules["rollout_buffer"], policy.rollout_buffer)

    def test_clone_parameters_are_independent(self) -> None:
        policy = _make_policy()
        clone = policy.clone_for_rnad()
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
            self.assertIs(cast(PPOPolicy, state.target).rollout_buffer, policy.rollout_buffer)
            self.assertIs(cast(PPOPolicy, state.reg_cur).rollout_buffer, policy.rollout_buffer)
            self.assertIs(cast(PPOPolicy, state.reg_prev).rollout_buffer, policy.rollout_buffer)
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
            target = cast(PPOPolicy, state.target)
            reg_cur = cast(PPOPolicy, state.reg_cur)
            reg_prev = cast(PPOPolicy, state.reg_prev)
            # Perturb target so the advance produces observable changes.
            with torch.no_grad():
                target.value_head.weight.fill_(0.75)
            reg_cur_before = reg_cur.value_head.weight.clone()
            _advance_outer_iteration(state)
            self.assertEqual(state.outer_iteration, 1)
            self.assertEqual(state.gradient_step, 0)
            # reg_prev should now equal the previous reg_cur value.
            self.assertTrue(torch.allclose(reg_prev.value_head.weight, reg_cur_before))
            # reg_cur should now equal target.
            self.assertTrue(torch.allclose(reg_cur.value_head.weight, torch.tensor(0.75)))
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
            target = cast(PPOPolicy, state.target)
            reg_cur = cast(PPOPolicy, state.reg_cur)
            reg_prev = cast(PPOPolicy, state.reg_prev)
            with torch.no_grad():
                target.value_head.weight.fill_(0.5)
            _advance_outer_iteration(state)
            # Clobber the in-memory regs so resume has real work to do.
            with torch.no_grad():
                reg_cur.value_head.weight.fill_(-99.0)
                reg_prev.value_head.weight.fill_(-99.0)
            resume_from_snapshot_dir(state, outer_iteration=1, gradient_step=3)
        self.assertEqual(state.outer_iteration, 1)
        self.assertEqual(state.gradient_step, 3)
        self.assertTrue(
            torch.allclose(reg_cur.value_head.weight, torch.tensor(0.5)),
        )


class CheckpointRoundTripTests(unittest.TestCase):
    def test_save_and_restore_rnad_state(self) -> None:
        from argparse import Namespace

        from magic_ai.opponent_pool import OpponentPool
        from scripts.train import (
            _restore_rnad_state,
            load_training_checkpoint,
            save_checkpoint,
        )

        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reg_dir = tmp / "rnad"
            state = build_trainer_state(
                policy,
                config=RNaDConfig(delta_m=5),
                reg_snapshot_dir=reg_dir,
                device=policy.device,
            )
            target = cast(PPOPolicy, state.target)
            # Mutate target so the roundtrip is observable.
            with torch.no_grad():
                target.value_head.weight.fill_(0.321)
            state.outer_iteration = 2
            state.gradient_step = 7
            # Ensure reg_m001.pt and reg_m002.pt exist so resume can load both.
            from magic_ai.rnad import save_reg_snapshot

            save_reg_snapshot(cast(nn.Module, state.reg_cur), reg_dir / "reg_m001.pt")
            save_reg_snapshot(cast(nn.Module, state.target), reg_dir / "reg_m002.pt")

            ckpt_path = tmp / "ckpt.pt"
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            args = Namespace(
                output=ckpt_path,
                opponent_pool_dir=tmp / "opponent_pool",
            )
            save_checkpoint(
                ckpt_path,
                policy,
                optimizer,
                args,
                opponent_pool=OpponentPool(),
                rnad_state=state,
            )

            checkpoint = load_training_checkpoint(ckpt_path)
            self.assertIsNotNone(checkpoint)

            # Rebuild a fresh trainer state and restore into it.
            fresh_policy = _make_policy()
            fresh = build_trainer_state(
                fresh_policy,
                config=RNaDConfig(delta_m=5),
                reg_snapshot_dir=reg_dir,
                device=fresh_policy.device,
            )
            _restore_rnad_state(fresh, checkpoint)

        self.assertEqual(fresh.outer_iteration, 2)
        self.assertEqual(fresh.gradient_step, 7)
        self.assertTrue(
            torch.allclose(cast(PPOPolicy, fresh.target).value_head.weight, torch.tensor(0.321)),
        )


class FinetuneGateTests(unittest.TestCase):
    def test_cap_triggers_finetuning_mode(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmp:
            state = build_trainer_state(
                policy,
                config=RNaDConfig(delta_m=1, num_outer_iterations=2),
                reg_snapshot_dir=Path(tmp),
                device=policy.device,
            )
            # Advance through the configured iterations.
            _advance_outer_iteration(state)
            self.assertFalse(state.is_finetuning)
            _advance_outer_iteration(state)
            self.assertTrue(state.is_finetuning)
            # Further advances stay in finetuning mode and do not create
            # new snapshot files past the configured cap.
            _advance_outer_iteration(state)
            self.assertEqual(state.outer_iteration, 2)
            self.assertTrue(state.is_finetuning)
            self.assertFalse((Path(tmp) / "reg_m003.pt").exists())


class FinetuneSamplingTests(unittest.TestCase):
    def test_finetune_eps_drops_low_probability_choices(self) -> None:
        """PPOPolicy._sample_flat_decisions with finetune_eps > 0 must
        zero out choices below the threshold and never sample them."""
        policy = _make_policy()
        # Three choices in one group; probs after softmax are ~uniform for
        # equal logits, so bias heavily toward choice 0.
        flat_logits = torch.tensor([3.0, -3.0, -3.0])
        flat_log_probs = torch.log_softmax(flat_logits, dim=-1)
        group_idx = torch.zeros(3, dtype=torch.long)
        choice_cols = torch.arange(3, dtype=torch.long)
        torch.manual_seed(0)
        for _ in range(50):
            selected_cols, selected_lp = policy._sample_flat_decisions(
                group_idx=group_idx,
                choice_cols=choice_cols,
                flat_logits=flat_logits,
                flat_log_probs=flat_log_probs,
                deterministic=False,
                finetune_eps=0.25,  # survivors: choice 0 (prob > 0.25)
            )
            # Only choice 0 has prob above the threshold, so it's the
            # only survivor; should be the sole sampled option.
            self.assertEqual(int(selected_cols[0]), 0)

    def test_finetune_n_disc_quantizes_survivor_probabilities(self) -> None:
        """n_disc quantization must snap surviving probabilities to
        multiples of 1/n_disc and truncate past cumulative 1.0."""
        policy = _make_policy()
        # Four choices; uniform raw prob is 0.25 each so all survive any
        # threshold <= 0.25. With n_disc=2 (quantum=0.5), rounded probs
        # per entry ceil(0.25/0.5)*0.5 = 0.5, so cumulative hits 1.0 at
        # index 2 and the last two entries truncate to zero.
        flat_logits = torch.zeros(4)
        flat_log_probs = torch.log_softmax(flat_logits, dim=-1)
        group_idx = torch.zeros(4, dtype=torch.long)
        choice_cols = torch.arange(4, dtype=torch.long)
        torch.manual_seed(0)
        survivor_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for _ in range(200):
            selected, _ = policy._sample_flat_decisions(
                group_idx=group_idx,
                choice_cols=choice_cols,
                flat_logits=flat_logits,
                flat_log_probs=flat_log_probs,
                deterministic=False,
                finetune_eps=0.0,
                finetune_n_disc=2,
            )
            survivor_counts[int(selected[0])] += 1
        # With n_disc=2 and uniform input, only the top-2 tied entries can
        # be sampled. Which two get picked depends on tie-breaking in the
        # sort; the point is two entries should account for ~all samples.
        nonzero_buckets = sum(1 for v in survivor_counts.values() if v > 0)
        self.assertLessEqual(nonzero_buckets, 2)

    def test_finetune_eps_zero_is_identity(self) -> None:
        policy = _make_policy()
        flat_logits = torch.tensor([1.0, 0.0, -1.0])
        flat_log_probs = torch.log_softmax(flat_logits, dim=-1)
        group_idx = torch.zeros(3, dtype=torch.long)
        choice_cols = torch.arange(3, dtype=torch.long)
        torch.manual_seed(0)
        cols_no_eps, _ = policy._sample_flat_decisions(
            group_idx=group_idx,
            choice_cols=choice_cols,
            flat_logits=flat_logits,
            flat_log_probs=flat_log_probs,
            deterministic=False,
            finetune_eps=0.0,
        )
        torch.manual_seed(0)
        cols_explicit, _ = policy._sample_flat_decisions(
            group_idx=group_idx,
            choice_cols=choice_cols,
            flat_logits=flat_logits,
            flat_log_probs=flat_log_probs,
            deterministic=False,
        )
        self.assertTrue(torch.equal(cols_no_eps, cols_explicit))


class RegPathRelocationTests(unittest.TestCase):
    def test_restore_uses_saved_reg_dir_when_configured_path_is_empty(self) -> None:
        """Checkpoint moved to a new machine: serialized reg_snapshot_dir
        should take precedence if the configured path has no snapshots."""
        from argparse import Namespace

        from magic_ai.opponent_pool import OpponentPool
        from scripts.train import (
            _restore_rnad_state,
            load_training_checkpoint,
            save_checkpoint,
        )

        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            saved_dir = tmp / "original" / "rnad"
            state = build_trainer_state(
                policy,
                config=RNaDConfig(delta_m=5),
                reg_snapshot_dir=saved_dir,
                device=policy.device,
            )
            with torch.no_grad():
                cast(PPOPolicy, state.target).value_head.weight.fill_(0.9)
            _advance_outer_iteration(state)
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            ckpt_path = tmp / "ckpt.pt"
            args = Namespace(output=ckpt_path, opponent_pool_dir=tmp / "op")
            save_checkpoint(
                ckpt_path,
                policy,
                optimizer,
                args,
                opponent_pool=OpponentPool(),
                rnad_state=state,
            )
            checkpoint = load_training_checkpoint(ckpt_path)

            # Rebuild with a DIFFERENT (empty) reg dir and verify restore
            # repoints to the saved dir.
            fresh_policy = _make_policy()
            empty_dir = tmp / "moved" / "rnad"
            fresh = build_trainer_state(
                fresh_policy,
                config=RNaDConfig(delta_m=5),
                reg_snapshot_dir=empty_dir,
                device=fresh_policy.device,
            )
            _restore_rnad_state(fresh, checkpoint)
            # reg_snapshot_dir should have been repointed to the saved dir.
            self.assertEqual(fresh.reg_snapshot_dir, saved_dir)
            self.assertEqual(fresh.outer_iteration, 1)


if __name__ == "__main__":
    unittest.main()
