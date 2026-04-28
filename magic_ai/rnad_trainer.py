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
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
from torch import nn

from magic_ai.buffer import RolloutBuffer
from magic_ai.model import TRACE_KIND_TO_ID, PPOPolicy
from magic_ai.ppo import PPOStats, RolloutStep
from magic_ai.rnad import (
    RNaDConfig,
    RNaDStats,
    load_reg_snapshot_into,
    polyak_update_,
    rnad_batched_trajectory_loss,
    save_reg_snapshot,
)
from magic_ai.training_interfaces import RNaDTrainablePolicy


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


def _delta_m_for_outer_iteration(outer_iteration: int, delta_m: int) -> int:
    """Gradient-step interval before the next outer update.

    The initial ``reg_m000`` snapshot is persisted at construction time. For
    early training stability, take the next two fixed-point snapshots halfway
    through the normal interval, then return to the configured cadence:
    ``0, Δ/2, Δ, 2Δ, ...``.
    """

    if delta_m <= 1:
        return 1
    if outer_iteration < 2:
        return max(1, delta_m // 2)
    return delta_m


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


def _count_active_steps(
    rb: RolloutBuffer,
    per_episode_replay_rows: Sequence[Sequence[int]],
) -> tuple[int, int]:
    """Compute the issue-7 ``cl_count`` / ``pl_count`` normalizer totals from
    rollout-buffer fields alone -- no policy forward required.

    Mirrors the counts that :func:`rnad_trajectory_loss` would accumulate by
    summing each per-episode loss helper's ``count`` return value across
    both ``own_idx`` perspectives:

    - ``cl_count`` = ``critic_loss`` count summed over ``own_idx`` =
      total number of (step, perspective) pairs where ``perspective ==
      own_idx``. Since every step has exactly one perspective in
      ``{0, 1}``, this collapses to the total replay-step count.
    - ``pl_count`` = ``may_neurd_loss`` count + ``neurd_loss_per_choice``
      count, both summed over ``own_idx``. The may count collapses to
      the total number of may-active steps in the batch; the per-choice
      count collapses to the total number of valid (decision-group,
      choice-column) cells across all active decision-group entries
      (``rb.decision_mask[idx_t].sum()``).
    """

    if not per_episode_replay_rows:
        return 0, 0

    flat_rows: list[int] = [r for ep in per_episode_replay_rows for r in ep]
    device = rb.trace_kind_id.device
    step_indices = torch.tensor(flat_rows, dtype=torch.long, device=device)
    cl_count_total = int(step_indices.numel())
    trace_kind = rb.trace_kind_id[step_indices]
    may_count_total = int((trace_kind == TRACE_KIND_TO_ID["may"]).sum().item())

    decision_starts = rb.decision_start[step_indices]
    decision_counts = rb.decision_count[step_indices]
    active_mask = decision_counts > 0
    flat_count_total = 0
    if active_mask.any():
        pos_t = active_mask.nonzero(as_tuple=False).squeeze(-1)
        active_counts = decision_counts[pos_t]
        max_count = int(active_counts.max().item())
        offsets = torch.arange(max_count, dtype=torch.long, device=device).unsqueeze(0)
        expanded_offsets = offsets.expand(pos_t.shape[0], -1)
        valid_offsets = expanded_offsets < active_counts.unsqueeze(1)
        idx_t = (decision_starts[pos_t].unsqueeze(1) + offsets)[valid_offsets]
        flat_count_total = int(rb.decision_mask[idx_t].sum().item())

    return cl_count_total, may_count_total + flat_count_total


def run_rnad_update(
    policy: RNaDTrainablePolicy,
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

    current_delta_m = _delta_m_for_outer_iteration(
        state.outer_iteration,
        state.config.delta_m,
    )
    alpha = _alpha_for_step(state.gradient_step, current_delta_m)

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

    if not per_episode_replay_rows:
        raise ValueError("no non-empty episodes to update on")

    cl_count_total, pl_count_total = _count_active_steps(
        policy.rollout_buffer,
        per_episode_replay_rows,
    )

    episodes_logp_mu = [torch.tensor(lp, dtype=torch.float32) for lp in per_episode_logp_mu]

    diag_every = state.config.diagnostic_v_target_reg_share_every
    compute_v_target_reg_share = diag_every > 0 and (state.gradient_step % diag_every == 0)

    n_episodes = len(per_episode_replay_rows)
    step_budget = state.config.step_minibatch_size
    if step_budget <= 0:
        chunks = [(0, n_episodes)]
    else:
        # Greedy step-budgeted packing: extend the chunk while it fits,
        # but never split an episode (chunk granularity is whole episodes).
        chunks = []
        lo = 0
        cum = 0
        for i, ep in enumerate(per_episode_replay_rows):
            ep_len = len(ep)
            if cum > 0 and cum + ep_len > step_budget:
                chunks.append((lo, i))
                lo = i
                cum = 0
            cum += ep_len
        chunks.append((lo, n_episodes))

    def _chunk_pieces(lo: int, hi: int) -> list:
        return rnad_batched_trajectory_loss(
            online=policy,
            target=state.target,
            reg_cur=state.reg_cur,
            reg_prev=state.reg_prev,
            episodes_replay_rows=per_episode_replay_rows[lo:hi],
            episodes_perspective=per_episode_perspective[lo:hi],
            episodes_winner_idx=per_episode_winner_idx[lo:hi],
            episodes_logp_mu=episodes_logp_mu[lo:hi],
            config=state.config,
            alpha=alpha,
            compute_v_target_reg_share=compute_v_target_reg_share,
        )

    if os.environ.get("RNAD_VERIFY_COUNTS"):
        # Cross-check: rerun under no_grad and confirm the buffer-derived
        # counts match the per-policy forward's accumulated cl/pl totals.
        cl_check = 0
        pl_check = 0
        with torch.no_grad():
            for lo, hi in chunks:
                for pieces in _chunk_pieces(lo, hi):
                    cl_check += pieces.cl_count
                    pl_check += pieces.pl_count
        assert cl_check == cl_count_total, (cl_check, cl_count_total)
        assert pl_check == pl_count_total, (pl_check, pl_count_total)

    # Mini-batched forward + backward, with batch-global normalization.
    # ``rnad_batched_trajectory_loss`` runs each policy's LSTM scan and
    # per-choice forward once per chunk (4 forwards per chunk instead of
    # 4 × n_episodes_per_chunk). Each chunk's loss is normalized by the
    # *global* cl/pl counts from ``_count_active_steps`` and backwarded
    # independently so activations free between chunks; gradients
    # accumulate across chunks and a single ``optimizer.step()`` runs
    # at the end. With ``episode_minibatch_size <= 0`` (default) the
    # whole batch is one chunk -- maximum throughput, peak activation
    # memory ∝ n_episodes. Lower it to cap activation memory.
    cl_norm = max(cl_count_total, 1)
    pl_norm = max(pl_count_total, 1)

    optimizer.zero_grad(set_to_none=True)
    cl_loss_total = 0.0
    pl_loss_total = 0.0
    n_q_clipped_total = 0
    flat_active_total = 0
    v_hat_mean_acc = 0.0
    transformed_mean_acc = 0.0
    n_pieces = 0
    diag_lr_sum = 0.0
    diag_lr_count = 0
    diag_lr_absmax = 0.0
    diag_isup_sum = 0.0
    diag_isup_count = 0
    diag_isdn_sum = 0.0
    diag_isdn_count = 0
    diag_vshare_sum = 0.0
    diag_vshare_count = 0
    for lo, hi in chunks:
        pieces_list = _chunk_pieces(lo, hi)
        cl_chunk = pieces_list[0].cl_sum.new_zeros(())
        pl_chunk = pieces_list[0].pl_sum.new_zeros(())
        for pieces in pieces_list:
            cl_chunk = cl_chunk + pieces.cl_sum
            pl_chunk = pl_chunk + pieces.pl_sum
            n_q_clipped_total += pieces.n_q_clipped
            flat_active_total += pieces.pl_count
            v_hat_mean_acc += pieces.v_hat_mean
            transformed_mean_acc += pieces.transformed_mean
            n_pieces += 1
            diag_lr_sum += pieces.sampled_log_ratio_sum
            diag_lr_count += pieces.sampled_log_ratio_count
            if pieces.sampled_log_ratio_absmax > diag_lr_absmax:
                diag_lr_absmax = pieces.sampled_log_ratio_absmax
            diag_isup_sum += pieces.is_bias_up_sum
            diag_isup_count += pieces.is_bias_up_count
            diag_isdn_sum += pieces.is_bias_down_sum
            diag_isdn_count += pieces.is_bias_down_count
            diag_vshare_sum += pieces.v_target_reg_share_sum
            diag_vshare_count += pieces.v_target_reg_share_count
        cl_part = cl_chunk / cl_norm
        pl_part = pl_chunk / pl_norm
        (cl_part + pl_part).backward()
        cl_loss_total += float(cl_part.detach())
        pl_loss_total += float(pl_part.detach())

    trainable = [p for p in policy.parameters() if p.requires_grad]
    grad_norm = float(nn.utils.clip_grad_norm_(trainable, max_norm=state.config.grad_clip))
    optimizer.step()
    polyak_update_(state.target, cast(nn.Module, policy), gamma=state.config.target_ema_gamma)

    state.gradient_step += 1
    if state.gradient_step >= current_delta_m:
        _advance_outer_iteration(state)

    q_clip_fraction = (
        float(n_q_clipped_total) / float(flat_active_total) if flat_active_total > 0 else 0.0
    )
    aggregate = RNaDStats(
        loss=cl_loss_total + pl_loss_total,
        critic_loss=cl_loss_total,
        policy_loss=pl_loss_total,
        v_hat_mean=v_hat_mean_acc / n_pieces,
        grad_norm=grad_norm,
        transformed_reward_mean=transformed_mean_acc / n_pieces,
        q_clip_fraction=q_clip_fraction,
        sampled_log_ratio_mean=(diag_lr_sum / diag_lr_count) if diag_lr_count > 0 else 0.0,
        sampled_log_ratio_absmax=diag_lr_absmax,
        is_bias_up_mean=(diag_isup_sum / diag_isup_count) if diag_isup_count > 0 else 0.0,
        is_bias_down_mean=(diag_isdn_sum / diag_isdn_count) if diag_isdn_count > 0 else 0.0,
        v_target_reg_share=(
            (diag_vshare_sum / diag_vshare_count) if diag_vshare_count > 0 else 0.0
        ),
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
