"""R-NaD trainer state + batch-level update dispatch.

Glue between :mod:`magic_ai.rnad` (algorithmic primitives) and the
``scripts/train_ppo.py`` outer rollout/update loop. Owns:

- the target network (Polyak-averaged with the online policy),
- the two regularization snapshots (``reg_cur`` / ``reg_prev``) required
  for the smooth reward-transform interpolation (paper §164),
- a gradient-step counter that drives the outer fixed-point iteration
  (advance ``reg_prev <- reg_cur``, ``reg_cur <- target`` every
  ``config.delta_m`` steps).

First-cut simplifications (see ``docs/rnad_design.md`` for rationale):

- Rollouts sample from the **online** policy, not the target. The paper
  samples from target; we keep the existing training-loop action-sampling
  code unchanged for this pass. Moving to target-sampled rollouts is a
  later phase.
- NeuRD is the sampled-action variant
  (:func:`magic_ai.rnad.sampled_neurd_loss`). The full per-action form
  (§188 with the ``[-beta, beta]`` logit gate) lands when
  :meth:`PPOPolicy.evaluate_replay_batch` grows a per-choice-logit return
  mode.
- The opponent pool is not used for R-NaD self-play; opponents are the
  current target (equivalent here to the online policy under the first
  simplification above). Snapshots continue to land in the pool for
  TrueSkill evaluation against PPO baselines, unchanged from PPO training.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from magic_ai.model import PPOPolicy
from magic_ai.ppo import PPOStats, RolloutStep
from magic_ai.rnad import (
    RNaDConfig,
    RNaDStats,
    load_reg_snapshot_into,
    rnad_update_trajectory,
    rnad_update_trajectory_full_neurd,
    save_reg_snapshot,
)


@dataclass
class EpisodeBatch:
    """Full-episode view of the RolloutSteps extending a single game."""

    steps: list[RolloutStep]
    winner_idx: int


@dataclass
class RNaDTrainerState:
    """Live state for a single R-NaD training run."""

    config: RNaDConfig
    target: PPOPolicy
    reg_cur: PPOPolicy
    reg_prev: PPOPolicy
    reg_snapshot_dir: Path
    gradient_step: int = 0
    outer_iteration: int = 0
    is_finetuning: bool = False
    last_stats: list[RNaDStats] = field(default_factory=list)


def _clone_policy_sharing_buffer(src: PPOPolicy) -> PPOPolicy:
    """Deep-copy parameter state, share the rollout buffer.

    The target/reg policies only need to run forward over the rollout buffer
    populated by the online policy; by sharing the buffer instance we avoid
    duplicating gigabytes of ingested trajectory tensors.
    """

    # Deepcopy everything except the rollout buffer, which we splice back in.
    original_buffer = src.rollout_buffer
    object.__setattr__(src, "rollout_buffer", None)
    try:
        clone = copy.deepcopy(src)
    finally:
        object.__setattr__(src, "rollout_buffer", original_buffer)
    object.__setattr__(clone, "rollout_buffer", original_buffer)
    return clone


def build_trainer_state(
    policy: PPOPolicy,
    *,
    config: RNaDConfig,
    reg_snapshot_dir: Path,
    device: torch.device,
) -> RNaDTrainerState:
    """Construct target + reg_cur + reg_prev from the online policy.

    All three siblings start as exact clones of ``policy`` and share its
    rollout buffer (see :func:`_clone_policy_sharing_buffer`). They are put
    in ``eval()`` mode and have ``requires_grad`` disabled; only the online
    ``policy`` receives gradients.
    """

    reg_snapshot_dir.mkdir(parents=True, exist_ok=True)

    target = _clone_policy_sharing_buffer(policy).to(device)
    reg_cur = _clone_policy_sharing_buffer(policy).to(device)
    reg_prev = _clone_policy_sharing_buffer(policy).to(device)
    for aux in (target, reg_cur, reg_prev):
        aux.eval()
        for p in aux.parameters():
            p.requires_grad_(False)

    # Persist the initial reg snapshot so resume has something to load.
    save_reg_snapshot(reg_cur, reg_snapshot_dir / "reg_m000.pt")

    return RNaDTrainerState(
        config=config,
        target=target,
        reg_cur=reg_cur,
        reg_prev=reg_prev,
        reg_snapshot_dir=reg_snapshot_dir,
    )


def _alpha_for_step(step_in_m: int, delta_m: int) -> float:
    """Smooth interpolation schedule (paper §164): ``min(1, 2n/Δ_m)``."""
    if delta_m <= 0:
        return 1.0
    return min(1.0, 2.0 * step_in_m / float(delta_m))


def _advance_outer_iteration(state: RNaDTrainerState) -> None:
    """Perform one fixed-point step: ``reg_prev <- reg_cur``, ``reg_cur <- target``.

    Once the configured ``num_outer_iterations`` cap is reached, further
    calls are no-ops: ``gradient_step`` resets to zero (so ``alpha``
    restarts the smooth ramp on each subsequent rollout batch) and
    ``is_finetuning`` is set so the trainer can switch to test-time
    sampling heuristics.
    """

    if state.outer_iteration >= state.config.num_outer_iterations:
        state.is_finetuning = True
        state.gradient_step = 0
        return

    # Snapshot target before rewiring the regs (target state is what becomes
    # the new reg_cur).
    next_iter = state.outer_iteration + 1
    target_path = state.reg_snapshot_dir / f"reg_m{next_iter:03d}.pt"
    save_reg_snapshot(state.target, target_path)

    # Sink reg_prev parameters with reg_cur's; sink reg_cur parameters with
    # target's. We reuse the nn.Modules rather than reallocating to avoid
    # churning CUDA memory.
    with torch.no_grad():
        reg_prev_src = dict(state.reg_cur.named_parameters())
        for name, p in state.reg_prev.named_parameters():
            p.data.copy_(reg_prev_src[name].data)
        reg_cur_src = dict(state.target.named_parameters())
        for name, p in state.reg_cur.named_parameters():
            p.data.copy_(reg_cur_src[name].data)

    state.outer_iteration = next_iter
    state.gradient_step = 0
    if state.outer_iteration >= state.config.num_outer_iterations:
        state.is_finetuning = True


def run_rnad_update(
    policy: PPOPolicy,
    optimizer: torch.optim.Optimizer,
    state: RNaDTrainerState,
    episodes: Sequence[EpisodeBatch],
    *,
    entropy_coef: float = 0.0,
    full_neurd: bool = False,
) -> PPOStats:
    """Run R-NaD on a batch of freshly-finished episodes.

    Each episode becomes one :func:`rnad_update_trajectory` call; per-call
    stats are averaged into a :class:`PPOStats` so the existing logger
    continues to work unchanged. ``entropy_coef`` is unused for R-NaD (the
    reward transform already injects an entropy bonus) but kept in the
    signature for parity with :func:`magic_ai.ppo.ppo_update`.
    """

    del entropy_coef  # parity-only arg

    if not episodes:
        raise ValueError("run_rnad_update requires at least one episode")

    per_episode: list[RNaDStats] = []
    for episode in episodes:
        if not episode.steps:
            continue
        replay_rows: list[int] = []
        perspective: list[int] = []
        logp_mu: list[float] = []
        for step in episode.steps:
            if step.replay_idx is None:
                raise ValueError("R-NaD requires replay_idx on every rollout step")
            replay_rows.append(int(step.replay_idx))
            perspective.append(int(step.perspective_player_idx))
            logp_mu.append(float(step.old_log_prob))
        alpha = _alpha_for_step(state.gradient_step, state.config.delta_m)

        update_fn = rnad_update_trajectory_full_neurd if full_neurd else rnad_update_trajectory
        stats = update_fn(
            online=policy,
            target=state.target,
            reg_cur=state.reg_cur,
            reg_prev=state.reg_prev,
            optimizer=optimizer,
            replay_rows=replay_rows,
            perspective_player_idx=perspective,
            winner_idx=int(episode.winner_idx),
            logp_mu=torch.tensor(logp_mu, dtype=torch.float32),
            config=state.config,
            alpha=alpha,
        )
        per_episode.append(stats)
        state.gradient_step += 1
        if state.gradient_step >= state.config.delta_m:
            _advance_outer_iteration(state)

    state.last_stats = per_episode

    # Aggregate for PPO-shaped logging. Leave PPO-specific fields at zero;
    # the three R-NaD-specific numbers populate the closest analogues.
    n = max(1, len(per_episode))
    mean_loss = sum(s.loss for s in per_episode) / n
    mean_critic = sum(s.critic_loss for s in per_episode) / n
    mean_policy = sum(s.policy_loss for s in per_episode) / n
    return PPOStats(
        loss=mean_loss,
        policy_loss=mean_policy,
        value_loss=mean_critic,
        entropy=0.0,
        approx_kl=0.0,
        clip_fraction=0.0,
        spr_loss=0.0,
    )


def resume_from_snapshot_dir(
    state: RNaDTrainerState,
    *,
    outer_iteration: int,
    gradient_step: int = 0,
) -> None:
    """Restore reg snapshots from ``state.reg_snapshot_dir``.

    Loads ``reg_m{outer_iteration}.pt`` into ``reg_cur`` and
    ``reg_m{outer_iteration-1}.pt`` into ``reg_prev`` (or both into the
    same if ``outer_iteration == 0``). The online/target policies are
    expected to be restored by the caller via the main PPO checkpoint.
    """

    if outer_iteration < 0:
        raise ValueError("outer_iteration must be non-negative")
    cur_path = state.reg_snapshot_dir / f"reg_m{outer_iteration:03d}.pt"
    prev_idx = max(0, outer_iteration - 1)
    prev_path = state.reg_snapshot_dir / f"reg_m{prev_idx:03d}.pt"
    load_reg_snapshot_into(state.reg_cur, cur_path)
    load_reg_snapshot_into(state.reg_prev, prev_path)
    state.outer_iteration = outer_iteration
    state.gradient_step = gradient_step
