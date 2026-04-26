"""R-NaD trainer state + batch-level update dispatch.

Glue between :mod:`magic_ai.rnad` (algorithmic primitives) and the
``scripts/train_ppo.py`` outer rollout/update loop. Owns:

- the target network (Polyak-averaged with the online policy),
- the two regularization snapshots (``reg_cur`` / ``reg_prev``) required
  for the smooth reward-transform interpolation (paper §164),
- a gradient-step counter that drives the outer fixed-point iteration
  (advance ``reg_prev <- reg_cur``, ``reg_cur <- target`` every
  ``config.delta_m`` steps).

First-cut simplifications (see ``docs/rnad_deviations.md`` for rationale):

- Rollouts sample from the **online** policy, not the target. The paper
  samples from target; with Polyak tracking inside one outer iteration the
  two are close, and the trainer treats target as the behavior policy when
  computing per-group ``1/mu_k`` corrections (issue 4 — autoregressive
  decomposition).
- NeuRD is always the full per-action form (paper §188 with the
  ``[-beta, beta]`` logit gate); there is no sampled-action fallback in
  this trainer.
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
from torch import nn

from magic_ai.model import PPOPolicy
from magic_ai.ppo import PPOStats, RolloutStep
from magic_ai.rnad import (
    RNaDConfig,
    RNaDStats,
    load_reg_snapshot_into,
    polyak_update_,
    rnad_trajectory_loss,
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

    The live LSTM env-state cache (``live_lstm_h``/``live_lstm_c``) is *not*
    cloned: it is the online actor's per-env runtime sampling cache, not
    trainable model state. The clone gets empty cache buffers; target/reg
    policies do not sample so they never need to read it, and recurrent-state
    recomputation during R-NaD updates explicitly re-runs the LSTM from a
    zero initial hidden state per policy.
    """

    # Deepcopy everything except the rollout buffer, which we splice back in.
    # Use normal attribute assignment so nn.Module updates ``_modules`` in
    # lockstep with ``__dict__``: ``object.__setattr__`` would leave the old
    # buffer registered in ``_modules`` and silently deep-copy it anyway via
    # ``nn.Module.__deepcopy__``'s traversal of submodules, defeating the
    # whole point of sharing the buffer.
    original_buffer = src.rollout_buffer
    src.rollout_buffer = None  # ty: ignore[invalid-assignment]
    try:
        clone = copy.deepcopy(src)
    finally:
        src.rollout_buffer = original_buffer
    clone.rollout_buffer = original_buffer
    if getattr(clone, "use_lstm", False):
        # Replace the deep-copied live LSTM cache (which mirrors src's per-env
        # state at clone time) with empty buffers. Target/reg never sample
        # from a live env, so this is the right starting state.
        empty_h = torch.zeros(clone.hidden_layers, 0, clone.hidden_dim, dtype=torch.float32)
        empty_c = empty_h.clone()
        clone.live_lstm_h = empty_h.to(clone.live_lstm_h.device)
        clone.live_lstm_c = empty_c.to(clone.live_lstm_c.device)
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
) -> PPOStats:
    """Run R-NaD on a batch of freshly-finished episodes.

    Each episode becomes one :func:`rnad_trajectory_loss` call; the resulting
    sums/counts are aggregated and a single backward + Adam + Polyak step is
    taken (issue 7 — paper-faithful 1/t_effective normalization, where
    ``t_effective`` is the total own-turn step / per-choice-action count
    across the batch, not the episode count).

    ``entropy_coef`` is unused for R-NaD (the reward transform already
    injects an entropy bonus) but kept in the signature for parity with
    :func:`magic_ai.ppo.ppo_update`.
    """

    del entropy_coef  # parity-only arg

    if not episodes:
        raise ValueError("run_rnad_update requires at least one episode")

    cl_sum: torch.Tensor | None = None
    pl_sum: torch.Tensor | None = None
    cl_count_total = 0
    pl_count_total = 0
    n_q_clipped_total = 0
    flat_active_total = 0
    v_hat_mean_acc = 0.0
    transformed_mean_acc = 0.0
    n_pieces = 0
    alpha = _alpha_for_step(state.gradient_step, state.config.delta_m)

    # Per-policy LSTM recompute, batched once across the whole rollout batch
    # so cuDNN sees one fused (N_episodes, T_max, hidden) call per policy
    # instead of one (1, T_episode, hidden) call per (policy, episode).
    per_episode_replay_rows: list[list[int]] = []
    per_episode_perspective: list[list[int]] = []
    per_episode_logp_mu: list[list[float]] = []
    per_episode_winner_idx: list[int] = []
    for episode in episodes:
        if not episode.steps:
            continue
        rows: list[int] = []
        persp: list[int] = []
        lp: list[float] = []
        for step in episode.steps:
            if step.replay_idx is None:
                raise ValueError("R-NaD requires replay_idx on every rollout step")
            rows.append(int(step.replay_idx))
            persp.append(int(step.perspective_player_idx))
            lp.append(float(step.old_log_prob))
        per_episode_replay_rows.append(rows)
        per_episode_perspective.append(persp)
        per_episode_logp_mu.append(lp)
        per_episode_winner_idx.append(int(episode.winner_idx))

    online_h_out_per_ep: list[torch.Tensor] | None = None
    target_h_out_per_ep: list[torch.Tensor] | None = None
    reg_cur_h_out_per_ep: list[torch.Tensor] | None = None
    reg_prev_h_out_per_ep: list[torch.Tensor] | None = None
    chunk_size = state.config.bptt_chunk_size
    if per_episode_replay_rows:
        recompute_fn = getattr(policy, "recompute_lstm_outputs_for_episodes", None)
        if recompute_fn is not None:
            online_h_out_per_ep = recompute_fn(per_episode_replay_rows, chunk_size=chunk_size)
            with torch.no_grad():
                tgt_fn = state.target.recompute_lstm_outputs_for_episodes
                rcr_fn = state.reg_cur.recompute_lstm_outputs_for_episodes
                rpr_fn = state.reg_prev.recompute_lstm_outputs_for_episodes
                target_h_out_per_ep = tgt_fn(per_episode_replay_rows, chunk_size=chunk_size)
                reg_cur_h_out_per_ep = rcr_fn(per_episode_replay_rows, chunk_size=chunk_size)
                reg_prev_h_out_per_ep = rpr_fn(per_episode_replay_rows, chunk_size=chunk_size)

    for ep_idx, replay_rows in enumerate(per_episode_replay_rows):
        perspective = per_episode_perspective[ep_idx]
        logp_mu = per_episode_logp_mu[ep_idx]
        online_h_out = online_h_out_per_ep[ep_idx] if online_h_out_per_ep is not None else None
        target_h_out = target_h_out_per_ep[ep_idx] if target_h_out_per_ep is not None else None
        reg_cur_h_out = reg_cur_h_out_per_ep[ep_idx] if reg_cur_h_out_per_ep is not None else None
        reg_prev_h_out = (
            reg_prev_h_out_per_ep[ep_idx] if reg_prev_h_out_per_ep is not None else None
        )

        pieces = rnad_trajectory_loss(
            online=policy,
            target=state.target,
            reg_cur=state.reg_cur,
            reg_prev=state.reg_prev,
            replay_rows=replay_rows,
            perspective_player_idx=perspective,
            winner_idx=per_episode_winner_idx[ep_idx],
            logp_mu=torch.tensor(logp_mu, dtype=torch.float32),
            config=state.config,
            alpha=alpha,
            online_h_out=online_h_out,
            target_h_out=target_h_out,
            reg_cur_h_out=reg_cur_h_out,
            reg_prev_h_out=reg_prev_h_out,
        )
        cl_sum = pieces.cl_sum if cl_sum is None else cl_sum + pieces.cl_sum
        pl_sum = pieces.pl_sum if pl_sum is None else pl_sum + pieces.pl_sum
        cl_count_total += pieces.cl_count
        pl_count_total += pieces.pl_count
        n_q_clipped_total += pieces.n_q_clipped
        flat_active_total += pieces.pl_count
        v_hat_mean_acc += pieces.v_hat_mean
        transformed_mean_acc += pieces.transformed_mean
        n_pieces += 1

    if cl_sum is None or pl_sum is None or n_pieces == 0:
        raise ValueError("no non-empty episodes to update on")

    # Issue 7: normalize by total counts across the rollout batch, not by
    # episode count. Critic divides by total own-turn steps; policy divides
    # by total active per-choice actions across all decision groups + may
    # branches. Short games no longer get over-weighted.
    cl = cl_sum / max(cl_count_total, 1)
    pl = pl_sum / max(pl_count_total, 1)
    total_loss = cl + pl

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    grad_norm = float(nn.utils.clip_grad_norm_(trainable, max_norm=state.config.grad_clip))
    optimizer.step()
    polyak_update_(state.target, policy, gamma=state.config.target_ema_gamma)

    state.gradient_step += 1
    if state.gradient_step >= state.config.delta_m:
        _advance_outer_iteration(state)

    q_clip_fraction = (
        float(n_q_clipped_total) / float(flat_active_total) if flat_active_total > 0 else 0.0
    )
    aggregate = RNaDStats(
        loss=float(total_loss.detach()),
        critic_loss=float(cl.detach()),
        policy_loss=float(pl.detach()),
        v_hat_mean=v_hat_mean_acc / n_pieces,
        grad_norm=grad_norm,
        transformed_reward_mean=transformed_mean_acc / n_pieces,
        q_clip_fraction=q_clip_fraction,
    )
    state.last_stats = [aggregate]

    return PPOStats(
        loss=aggregate.loss,
        policy_loss=aggregate.policy_loss,
        value_loss=aggregate.critic_loss,
        entropy=0.0,
        approx_kl=0.0,
        clip_fraction=q_clip_fraction,
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
