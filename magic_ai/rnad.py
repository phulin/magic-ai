"""Regularized Nash Dynamics (R-NaD / DeepNash) training primitives.

This module implements the algorithmic core of the R-NaD training option:

- :func:`transform_rewards` — entropy-regularized reward transformation
  (DeepNash paper eq. at §161, with the smooth-interpolation variant of §164).
- :func:`two_player_vtrace` — backward v-trace estimator adapted to the
  two-player alternating-turn setting (paper §170–182).
- :func:`neurd_loss` — Neural Replicator Dynamics policy loss with
  gradient gating on logits outside ``[-beta, beta]`` (paper §188).
- :func:`critic_loss` — L1 regression of the value head on the v-trace target
  (paper §186).
- :func:`threshold_discretize` — post-softmax fine-tuning / test-time
  projection that drops low-probability actions and quantizes the survivors
  (paper §197, §279–282).

The higher-level training orchestration (multi-policy replay forward,
Polyak-averaged target network, outer-loop fixed-point iteration) is built on
top of these primitives in the trainer module. See ``docs/rnad_design.md`` and
``docs/rnad_implementation_plan.md`` for the full design and phasing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RNaDConfig:
    """Hyperparameters for the R-NaD trainer.

    Defaults are scaled ~100x down from the DeepNash paper (which targeted
    768 TPU nodes over 7.21M steps) to be a sensible single-GPU starting
    point for magic-ai. All values are exposed via CLI flags.
    """

    eta: float = 0.2
    """Regularization strength for the reward transform (paper: 0.2)."""

    delta_m: int = 1_000
    """Gradient steps per outer iteration. Paper §199 uses 10k-100k on 768
    TPU learners; this default is scaled for a single-GPU rollout-batch
    cadence so the outer fixed-point iteration actually advances within a
    typical run. Increase proportionally with rollout-batch size."""

    num_outer_iterations: int = 50
    """Number of fixed-point outer iterations ``m`` (paper: ~200)."""

    vtrace_rho_bar: float = 1.0
    """Importance-weight clip for the advantage term in v-trace."""

    vtrace_c_bar: float = 1.0
    """Importance-weight clip for the trace term in v-trace."""

    neurd_beta: float = 2.0
    """Logit magnitude threshold; NeuRD gradient is zeroed outside ``[-beta, beta]``."""

    neurd_clip: float = 10_000.0
    """Clip applied to the Q estimates fed into the NeuRD loss."""

    grad_clip: float = 10_000.0
    """Global grad-norm clip applied before the optimizer step."""

    target_ema_gamma: float = 0.005
    """Polyak averaging rate for the target network. Paper §199 uses 1e-3
    paired with delta_m=10k-100k (i.e. ``delta_m · γ ≈ 10-100`` so the
    target tracks online many times over inside one outer iter); this
    default is scaled to match the smaller :attr:`delta_m` above so the
    target meaningfully tracks online within one outer iteration on
    single-GPU compute."""

    finetune_eps: float = 0.03
    """Probability threshold for fine-tune / test-time discretization."""

    finetune_n_disc: int = 16
    """Number of probability quanta for fine-tune / test-time discretization."""

    q_corr_rho_bar: float = 100.0
    """Clip on the per-group inverse-sampling weight ``1/mu_k`` used in the
    per-action Q estimator (paper §179-180, applied per autoregressive
    decision group — see :func:`rnad_trajectory_loss`).

    Issue 6: with the per-group decomposition introduced in issue 4 the
    weight is bounded by ``1/min_a pi_target_k(a)`` for a single group rather
    than the multiplicative ``1/∏_k mu_k`` of the joint formulation, so the
    clip can be loosened substantially without risking blow-up. The default
    100 lets policies near-deterministic on a single legal choice still
    update without aggressive bias, while bounding the worst-case sampled-
    correction magnitude. ``rnad_trainer`` logs how often the clip actually
    fires (``RNaDStats.q_clip_fraction``); if that climbs above a few
    percent, the per-action Q estimator has drifted and the clip is doing
    real work — investigate before raising it further."""

    learning_rate: float = 5e-5
    """Optimizer learning rate (paper §199: 5e-5). Used by the NeuRD
    gradient gate (paper §189) to evaluate the post-update logit predicate
    ``|logit + lr · Clip(Q, c)| <= beta`` rather than the looser
    current-logit predicate ``|logit| <= beta``."""

    diagnostic_v_target_reg_share_every: int = 0
    """Compute the ``v_target_reg_share`` diagnostic every K gradient steps
    (0 disables it entirely; default off). The diagnostic requires a *second*
    full ``two_player_vtrace`` pass per perspective per episode (fed only
    the terminal ±1 reward), which doubles v-trace work in the R-NaD loss.
    Trainers that want this signal should enable it at a low cadence (e.g.
    50) so the cost is amortized."""

    step_minibatch_size: int = 0
    """Approximate per-chunk replay-step budget for the batched R-NaD
    update (0 disables mini-batching: all episodes go through one fused
    forward + one backward). With mini-batching, episodes are packed
    greedily into chunks until cumulative replay-step count would exceed
    this budget; each chunk runs its own batched forward and
    ``.backward()``, gradients accumulate across chunks, and a single
    ``optimizer.step()`` runs at the end of the update. Mathematically
    identical to the all-at-once path (per-episode sums are normalized
    by the *global* cl/pl counts in both cases) but caps peak activation
    memory at one chunk's worth of episodes. Wired to ``--minibatch-size``
    on the CLI to share the PPO step budget."""

    bptt_chunk_size: int = 200
    """Chunk length for the chunked-BPTT recompute (DeepNash R-NaD paper
    arxiv 2206.15378 §"Full games learning"): trajectories are split along
    the time axis into chunks of this many steps, each chunk processed in
    one fused cuDNN ``nn.LSTM`` call, with state detached at chunk
    boundaries so gradients don't flow across them. Default 200 matches
    the production ``--max-steps-per-game=200`` cap, so the full trace is
    one chunk by default (full BPTT through the fused call). Lower this to
    cap activation memory at the cost of truncating gradient flow at
    chunk boundaries."""


# ---------------------------------------------------------------------------
# Reward transformation
# ---------------------------------------------------------------------------


def transform_rewards(
    rewards: Tensor,
    logp_theta: Tensor,
    logp_reg_cur: Tensor,
    logp_reg_prev: Tensor,
    *,
    alpha: float,
    eta: float,
    perspective_is_player_i: Tensor,
) -> Tensor:
    """Apply the R-NaD entropy-regularization reward transformation.

    The transformed reward at step ``t`` for player ``i`` is

        r'_t = r_t + (1 - 2 * 1[i == psi_t]) * eta * (log pi_theta(a_t|o_t)
                                                       - log pi_reg_blend(a_t|o_t))

    with ``log pi_reg_blend = alpha * log pi_reg_cur + (1 - alpha) * log pi_reg_prev``
    per the smooth transition of paper §164. ``alpha = min(1, 2n/Δ_m)`` is supplied
    by the caller.

    All tensors are 1-D of length ``T`` (one entry per step in the trajectory).
    ``perspective_is_player_i`` is ``True`` when the current-turn player is the
    player whose reward we are transforming (the sign inside the parenthesis
    becomes ``-1`` on own turns — cost — and ``+1`` on opponent turns — gain).
    """

    if rewards.shape != logp_theta.shape:
        raise ValueError("rewards and logp_theta must share shape")
    if rewards.shape != logp_reg_cur.shape:
        raise ValueError("rewards and logp_reg_cur must share shape")
    if rewards.shape != logp_reg_prev.shape:
        raise ValueError("rewards and logp_reg_prev must share shape")
    if rewards.shape != perspective_is_player_i.shape:
        raise ValueError("rewards and perspective_is_player_i must share shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    blended_logp_reg = alpha * logp_reg_cur + (1.0 - alpha) * logp_reg_prev
    log_ratio = logp_theta - blended_logp_reg
    # sign = -1 on own-turn steps, +1 on opponent-turn steps
    sign = 1.0 - 2.0 * perspective_is_player_i.to(dtype=rewards.dtype)
    return rewards + sign * eta * log_ratio


# ---------------------------------------------------------------------------
# Two-player v-trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VTraceOutput:
    """Output of :func:`two_player_vtrace`."""

    v_hat: Tensor
    """Value target :math:`\\hat v_t` per step, shape ``(T,)``."""

    q_hat: Tensor
    """Q estimate on the sampled action only, shape ``(T,)``.

    R-NaD actually requires Q over all legal actions for NeuRD. For the
    per-head NeuRD formulation (see ``docs/rnad_design.md``), the full Q
    tensor is assembled by the trainer from per-head logits plus this scalar
    signal. The unit tests in ``tests/test_rnad.py`` pin the scalar form
    against a Python-reference implementation.
    """

    r_hat_next: Tensor
    """Propagated opp-turn reward tail :math:`\\hat r^i_{t+1}` per own-turn
    step, broadcast to shape ``(T,)`` (zero on opp-turn steps).

    Required by the paper's per-action Q estimator (§179-180): the sampled
    correction includes :math:`(\\pi/\\mu)_t \\cdot (\\hat r^i_{t+1} +
    \\hat v^i_{t+1})`, where :math:`\\hat r^i_{t+1}` is the v-trace
    accumulator over the opp-turn rewards starting at the next step.
    """

    v_hat_next: Tensor
    """V-trace target at the next own-turn step :math:`\\hat v^i_{t+1}` per
    own-turn step, broadcast to shape ``(T,)`` (zero on opp-turn steps and
    on the final own-turn step). Required by paper §179-180."""


def two_player_vtrace(
    *,
    rewards: Tensor,
    values: Tensor,
    logp_theta: Tensor,
    logp_mu: Tensor,
    perspective_is_player_i: Tensor,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> VTraceOutput:
    """Two-player v-trace estimator (paper §170–182).

    All tensors are 1-D of length ``T`` and correspond to a single full
    episode traversed in forward time. The trajectory alternates player
    turns; ``perspective_is_player_i[t]`` is ``True`` when step ``t`` is
    played by the player ``i`` whose value we are estimating.

    The recursion runs backward from the final step. On own-turn steps
    (``i == psi_t``) we accumulate a clipped-importance bootstrap onto the
    current value estimate; on opponent-turn steps we carry the reward
    forward multiplicatively by the importance ratio (paper §172).

    No bootstrap past the end of the episode: the paper's formulation runs
    the recursion to ``t_effective`` and the trajectory-tail sentinel is
    zero (equivalent to assuming the terminal reward is already included in
    ``rewards[-1]``).
    """

    if rewards.ndim != 1:
        raise ValueError("two_player_vtrace expects 1-D trajectory tensors")
    T = rewards.shape[0]
    if T == 0:
        raise ValueError("cannot run v-trace on an empty trajectory")
    for name, t in (
        ("values", values),
        ("logp_theta", logp_theta),
        ("logp_mu", logp_mu),
        ("perspective_is_player_i", perspective_is_player_i),
    ):
        if t.shape != rewards.shape:
            raise ValueError(f"{name} must match rewards shape")

    dtype = rewards.dtype
    device = rewards.device
    is_own = perspective_is_player_i.to(dtype=torch.bool)
    ratio = (logp_theta - logp_mu).exp()

    v_hat = torch.zeros(T, dtype=dtype, device=device)
    q_hat = torch.zeros(T, dtype=dtype, device=device)
    r_hat_next_step = torch.zeros(T, dtype=dtype, device=device)
    v_hat_next_step = torch.zeros(T, dtype=dtype, device=device)

    # Vectorized backward pass (paper §170-182, algebraically rewritten).
    #
    # The per-step recursion resets at every own-turn step, so we split
    # the trajectory into segments delimited by own-turn indices. Between
    # own-turn k and own-turn k+1 there are zero or more opponent-turn
    # steps; all of their reward / importance-ratio accumulation can be
    # computed with cumsum/cumprod in O(T) tensor ops. Only the final
    # own-turn-indexed ``v_hat`` linear recurrence remains, and that runs
    # on K scalars on-device (no .item() sync per step, unlike the
    # previous per-T Python loop).
    own_idx = is_own.nonzero(as_tuple=False).squeeze(-1)  # (K,) sorted ascending
    K = int(own_idx.numel())
    if K == 0:
        # No own-turn steps in this trajectory: both outputs stay zero
        # (the original recursion's sentinels never bootstrap).
        return VTraceOutput(
            v_hat=v_hat,
            q_hat=q_hat,
            r_hat_next=r_hat_next_step,
            v_hat_next=v_hat_next_step,
        )

    # Per-step segmentation: ``seg[t]`` = index of the most recent own-turn
    # at or before t (= k when t == own_idx[k]; = k when t is opp-turn in
    # the segment immediately after own_idx[k]; = -1 for opp-turn steps
    # before the first own-turn).
    cum_own = is_own.to(dtype=torch.long).cumsum(0)
    seg = cum_own - 1  # (T,); -1 before any own-turn
    safe_seg = seg.clamp_min(0)

    # ``r_acc[k]`` = sum over opp-turn steps s in (own_idx[k], own_idx[k+1])
    # of rewards[s] * prod_{s' in that range, s' < s} ratio[s']. The
    # "per-step multiplier" is exp(cumA[s] - opp_log_ratio[s] - cumA[own_idx[k]])
    # where opp_log_ratio is log(ratio) on opp-turn steps only.
    log_ratio = ratio.log()
    opp_log_ratio = torch.where(is_own, torch.zeros_like(log_ratio), log_ratio)
    cumA = opp_log_ratio.cumsum(0)
    cumA_at_own = cumA[own_idx]  # (K,)
    base_cumA = cumA_at_own[safe_seg]  # (T,); garbage for seg == -1 but zeroed below
    multiplier = torch.exp(cumA - opp_log_ratio - base_cumA)

    opp_contrib = torch.where(
        is_own | (seg < 0),
        torch.zeros_like(rewards),
        rewards * multiplier.to(dtype=dtype),
    )
    r_acc = torch.zeros(K, dtype=dtype, device=device)
    r_acc.scatter_add_(0, safe_seg, opp_contrib)

    # ``xi_prod[k]`` = prod of ratio over opp-turn steps in the segment after
    # own_idx[k]. Last segment extends to the trajectory end.
    if K > 1:
        right_boundary = torch.cat(
            [own_idx[1:] - 1, torch.tensor([T - 1], dtype=torch.long, device=device)]
        )
    else:
        right_boundary = torch.tensor([T - 1], dtype=torch.long, device=device)
    xi_prod = torch.exp(cumA[right_boundary] - cumA_at_own).to(dtype=dtype)

    # Per-own-turn rho / c clips.
    ratio_own = ratio[own_idx].to(dtype=dtype)
    rho_arg = ratio_own * xi_prod
    rho = torch.clamp(rho_arg, max=rho_bar)
    c = torch.clamp(rho_arg, max=c_bar)

    # Per-own-turn pre-c quantity: this is also ``q_hat_sampled`` for the
    # paper's per-action Q form used by the full-NeuRD trainer.
    #
    # Paper §177:
    #   δV = ρ_t · (r^i_t + (π/μ)_t · r̂^i_{t+1} + V_next - v(o_t))
    # The (π/μ)_t prefactor on the opp-tail accumulator r̂_{t+1} is the
    # *unclipped* own-turn importance ratio. ``ratio_own`` already holds
    # this scalar (clipped variants are computed below as ``rho``/``c``);
    # multiply r_acc by it before the ρ-clip on the bracket as a whole.
    values_own = values[own_idx]
    rewards_own = rewards[own_idx]
    v_next_own_vec = torch.cat([values_own[1:], values_own.new_zeros(1)])
    pre_c = values_own + rho * (rewards_own + ratio_own * r_acc + v_next_own_vec - values_own)

    # Own-turn linear recurrence:
    #   v_hat_own[k] = A[k] + B[k] * v_hat_own[k+1],  v_hat_own[K] = 0
    A = pre_c - c * v_next_own_vec
    B = c
    v_hat_own = torch.zeros(K, dtype=dtype, device=device)
    next_val = values_own.new_zeros(())
    for k in range(K - 1, -1, -1):
        next_val = A[k] + B[k] * next_val
        v_hat_own[k] = next_val

    # Scatter own-turn results back; opp-turn steps inherit v_hat from the
    # next own-turn (or zero if none follows).
    v_hat.index_copy_(0, own_idx, v_hat_own)
    q_hat.index_copy_(0, own_idx, pre_c)
    # Per-own-turn r̂_{t+1} (= r_acc[k]) and v̂_{t+1} (= v_hat_own[k+1] | 0),
    # required by the paper's per-action Q estimator (§179-180).
    v_hat_next_own = torch.cat([v_hat_own[1:], v_hat_own.new_zeros(1)])
    r_hat_next_step.index_copy_(0, own_idx, r_acc)
    v_hat_next_step.index_copy_(0, own_idx, v_hat_next_own)

    next_k = seg + 1
    valid_next = next_k < K
    safe_next_k = next_k.clamp(min=0, max=max(K - 1, 0))
    opp_v = v_hat_own[safe_next_k]
    opp_mask = ~is_own & valid_next
    v_hat = torch.where(opp_mask, opp_v, v_hat)
    q_hat = torch.where(opp_mask, opp_v, q_hat)

    return VTraceOutput(
        v_hat=v_hat,
        q_hat=q_hat,
        r_hat_next=r_hat_next_step,
        v_hat_next=v_hat_next_step,
    )


# ---------------------------------------------------------------------------
# NeuRD policy loss
# ---------------------------------------------------------------------------


def neurd_loss(
    *,
    logits: Tensor,
    q_hat: Tensor,
    legal_mask: Tensor,
    beta: float,
    clip: float,
    lr: float = 0.0,
) -> Tensor:
    """NeuRD policy loss with the paper's logit-magnitude gate (§188).

    Shapes:
      - ``logits``: ``(T, A)`` per-step per-legal-action logits.
      - ``q_hat``: ``(T, A)`` per-step per-action Q estimates.
      - ``legal_mask``: ``(T, A)`` bool; ``True`` for legal actions.

    The loss is

        L = - sum_t sum_a [ logits_detached_to_scalar(t, a) is unused;
                           use logits(t, a) * clip(Q(t, a), [-c, c]) ]
            where the gradient contribution is zeroed on any (t, a) whose
            logit falls outside ``[-beta, beta]``.

    The sign convention is that minimizing this loss increases the log-prob
    of high-Q actions, consistent with replicator dynamics.

    The $\\hat\\nabla$ gate of the paper is approximated here by applying a
    differentiable mask to the logits: ``m_t,a = 1[-beta <= logit <= beta]``
    as a stop-grad factor. This produces the same gradient for the online
    update as the paper's formulation.
    """

    if logits.ndim != 2 or q_hat.ndim != 2 or legal_mask.ndim != 2:
        raise ValueError("neurd_loss expects 2-D (T, A) tensors")
    if logits.shape != q_hat.shape or logits.shape != legal_mask.shape:
        raise ValueError("logits, q_hat, and legal_mask must share shape")

    q_clipped = q_hat.clamp(-clip, clip).detach()
    with torch.no_grad():
        # Paper §189: gate on the post-update logit value. The NeuRD gradient
        # on logit_a is Clip(Q, c), so the post-update logit is approximately
        # ``logit + lr · Clip(Q, c)``. Gating on this anticipates the step
        # and zeroes the gradient before the logit exits ``[-beta, beta]``.
        # With lr=0.0 (default) this reduces to the current-logit predicate.
        post_update = logits + lr * q_clipped
        in_range = (post_update >= -beta) & (post_update <= beta)
        active = legal_mask & in_range
    # Loss = -sum_{t,a in active} logits * Q. Negative because Adam minimizes
    # and we want d logits proportional to +Q * grad_flag_in_range.
    per_entry = -logits * q_clipped * active.to(dtype=logits.dtype)
    T = logits.shape[0]
    return per_entry.sum() / max(T, 1)


# ---------------------------------------------------------------------------
# Ragged per-choice NeuRD
# ---------------------------------------------------------------------------


def neurd_loss_per_choice(
    *,
    flat_logits: Tensor,
    flat_q: Tensor,
    flat_active_mask: Tensor | None = None,
    beta: float,
    clip: float,
    lr: float = 0.0,
) -> tuple[Tensor, int, int]:
    """NeuRD loss over a flattened, ragged set of per-choice logits.

    Returns ``(sum_loss, num_active_entries, num_clipped_entries)``. The
    caller is responsible for global normalization by total trajectory step
    count (issue 7). ``num_clipped_entries`` reports how many ``flat_q``
    values were clamped by ``[-clip, clip]`` — the trainer logs this so an
    operator can detect when the per-action Q estimator is producing
    abnormally large values (issue 6).

    ``flat_active_mask`` (default: all-ones) selects which flat entries
    actually contribute. Use it to mask out non-own-turn steps or
    decision groups that should not receive policy gradient.
    """

    if flat_logits.shape != flat_q.shape:
        raise ValueError("flat_logits and flat_q must share shape")
    if flat_active_mask is not None and flat_active_mask.shape != flat_logits.shape:
        raise ValueError("flat_active_mask must match flat_logits shape")
    q_pre = flat_q.detach()
    q_clipped = q_pre.clamp(-clip, clip)
    n_clipped = int((q_pre.abs() > clip).sum().item()) if flat_logits.numel() > 0 else 0
    with torch.no_grad():
        # Paper §189 post-update logit predicate; see :func:`neurd_loss`.
        post_update = flat_logits + lr * q_clipped
        in_range = (post_update >= -beta) & (post_update <= beta)
    active_f = (
        flat_active_mask.to(dtype=flat_logits.dtype)
        if flat_active_mask is not None
        else torch.ones_like(flat_logits)
    )
    per_entry = -flat_logits * q_clipped * in_range.to(dtype=flat_logits.dtype) * active_f
    sum_loss = per_entry.sum()
    n_active = int(active_f.sum().item()) if flat_logits.numel() > 0 else 0
    return sum_loss, n_active, n_clipped


# ---------------------------------------------------------------------------
# Critic loss
# ---------------------------------------------------------------------------


def critic_loss(
    *,
    v_theta: Tensor,
    v_hat: Tensor,
    perspective_is_player_i: Tensor,
) -> tuple[Tensor, int]:
    """L1 regression of the online value head against the v-trace target.

    Only own-turn steps contribute (paper §186). Returns ``(sum_loss, count)``
    so the caller can normalize globally over a multi-trajectory batch by
    total own-turn step count (issue 7).
    """

    if v_theta.shape != v_hat.shape:
        raise ValueError("v_theta and v_hat must share shape")
    if v_theta.shape != perspective_is_player_i.shape:
        raise ValueError("v_theta and perspective mask must share shape")

    mask = perspective_is_player_i.to(dtype=torch.bool)
    if not mask.any():
        return v_theta.new_zeros(()), 0
    diff = (v_theta - v_hat.detach()).abs()
    return diff[mask].sum(), int(mask.sum().item())


# ---------------------------------------------------------------------------
# Threshold / discretize (fine-tune + test-time projection)
# ---------------------------------------------------------------------------


def threshold_discretize(
    probs: Tensor,
    *,
    eps: float,
    n_disc: int,
) -> Tensor:
    """Fine-tune / test-time policy projection (paper §197, §279–282).

    For each row of ``probs`` (assumed summing to 1 over the last axis):

    1. Drop every entry below ``eps`` and renormalize. If no survivors,
       leave the row unchanged.
    2. Sort survivors high-to-low and quantize probabilities to multiples
       of ``1 / n_disc``, rounding up; once cumulative weight reaches 1,
       truncate remaining entries to zero.

    Operates on the last axis. Arbitrary leading shape supported.
    """

    if eps < 0.0 or eps >= 1.0:
        raise ValueError("eps must be in [0, 1)")
    if n_disc < 1:
        raise ValueError("n_disc must be >= 1")

    original_shape = probs.shape
    flat = probs.reshape(-1, original_shape[-1])
    out = flat.clone()
    for i in range(flat.shape[0]):
        row = flat[i]
        kept = row.clone()
        kept[kept < eps] = 0.0
        total = kept.sum()
        if float(total.item()) <= 0.0:
            out[i] = row
            continue
        kept = kept / total
        # Sort and quantize
        sorted_vals, sort_idx = torch.sort(kept, descending=True)
        quantum = 1.0 / float(n_disc)
        rounded = torch.ceil(sorted_vals / quantum) * quantum
        cum = torch.cumsum(rounded, dim=0)
        # Clip the one that crosses 1.0 and zero out the tail.
        over_idx = (cum >= 1.0).nonzero(as_tuple=False)
        if over_idx.numel() > 0:
            first_over = int(over_idx[0].item())
            prev_cum = float(cum[first_over - 1].item()) if first_over > 0 else 0.0
            rounded[first_over] = max(0.0, 1.0 - prev_cum)
            if first_over + 1 < rounded.numel():
                rounded[first_over + 1 :] = 0.0
        # Un-sort back to original order.
        restored = torch.zeros_like(row)
        restored[sort_idx] = rounded
        # Normalize to absorb any floating-point drift.
        s = restored.sum()
        if float(s.item()) > 0.0:
            restored = restored / s
        out[i] = restored
    return out.reshape(original_shape)


# ---------------------------------------------------------------------------
# Episode-assembly helper
# ---------------------------------------------------------------------------


def may_neurd_loss(
    *,
    may_logits: Tensor,
    q_accept: Tensor,
    q_decline: Tensor,
    own_turn_may_mask: Tensor,
    beta: float,
    clip: float,
    lr: float = 0.0,
) -> tuple[Tensor, int]:
    """True two-action NeuRD loss for the Bernoulli ``may`` head.

    Models accept/decline as a 2-action softmax with logits ``(l, 0)`` (the
    canonical Bernoulli-as-softmax encoding; the all-zero second logit is a
    constant and so receives no gradient). The full NeuRD gradient over both
    actions is

        d(loss)/dl = -[(1 - p) · Clip(Q_accept, c) - p · Clip(Q_decline, c)]
                       · 1[|l_post| <= beta]

    where ``p = sigmoid(l)``. Both branches' Q values appear, so both branches
    contribute the per-action ``-eta · log(pi/pi_reg)`` regularization term —
    not just the sampled one. ``Q_accept`` and ``Q_decline`` must already have
    the regularization built in (the trainer assembles them per branch).

    Returns ``(sum_loss, count)`` rather than a mean so the caller can
    normalize globally over the trajectory batch's effective step count
    (issue 7 — paper-faithful 1/t_effective weighting).
    """

    if may_logits.shape != q_accept.shape:
        raise ValueError("may_logits and q_accept must share shape")
    if may_logits.shape != q_decline.shape:
        raise ValueError("may_logits and q_decline must share shape")
    mask = own_turn_may_mask.to(dtype=torch.bool)
    if not mask.any():
        return may_logits.new_zeros(()), 0
    p_accept = torch.sigmoid(may_logits)
    q_accept_c = q_accept.clamp(-clip, clip).detach()
    q_decline_c = q_decline.clamp(-clip, clip).detach()
    # NeuRD gradient on logit l = (1-p) Q_a - p Q_d; surrogate that produces
    # this gradient is p_accept_grad * Q_a + p_decline_grad * Q_d where the
    # softmax-style sum is realised by the two-action log-probs.
    log_p_accept = torch.nn.functional.logsigmoid(may_logits)
    log_p_decline = torch.nn.functional.logsigmoid(-may_logits)
    surrogate = log_p_accept * q_accept_c + log_p_decline * q_decline_c
    with torch.no_grad():
        # Gate using the worst-case post-update direction on l (paper §189).
        grad_l = (1.0 - p_accept) * q_accept_c - p_accept * q_decline_c
        post_update = may_logits + lr * grad_l
        in_range = post_update.abs() <= beta
    per_step = -surrogate * in_range.to(dtype=may_logits.dtype)
    sum_loss = per_step[mask].sum()
    count = int(mask.sum().item())
    return sum_loss, count


# ---------------------------------------------------------------------------
# Polyak target network + reg snapshot persistence
# ---------------------------------------------------------------------------


# Buffers that hold per-actor runtime state (env-indexed LSTM cache, rollout
# storage). They are not part of the trainable model and must NOT be Polyak-
# averaged or copied between online and target/reg policies; doing so would
# replace one policy's actor cache with another's (or with a stale snapshot)
# and silently corrupt sampling for whichever module shares the buffer.
_ACTOR_RUNTIME_BUFFER_PREFIXES: tuple[str, ...] = (
    "live_lstm_h",
    "live_lstm_c",
    "rollout_buffer.",
)


def _is_actor_runtime_buffer(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in _ACTOR_RUNTIME_BUFFER_PREFIXES)


@torch.no_grad()
def polyak_update_(target: nn.Module, online: nn.Module, gamma: float) -> None:
    """In-place Polyak averaging: ``target <- gamma * online + (1 - gamma) * target``.

    ``gamma`` is the "target learning rate" in the DeepNash paper (§191),
    typically very small (1e-3). Only floating-point parameters and buffers
    are averaged; integer buffers (e.g. LSTM indices) are copied verbatim.

    Per-actor runtime buffers (``live_lstm_h``/``live_lstm_c`` and the rollout
    buffer) are skipped entirely: those are not model state, they are the
    online actor's per-env LSTM cache and shared rollout storage, and copying
    them across policies corrupts sampling.
    """

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    target_params = dict(target.named_parameters())
    for name, p_online in online.named_parameters():
        if name not in target_params:
            raise ValueError(f"target missing parameter {name!r}")
        p_target = target_params[name]
        p_target.mul_(1.0 - gamma).add_(p_online.data, alpha=gamma)
    target_buffers = dict(target.named_buffers())
    for name, b_online in online.named_buffers():
        if name not in target_buffers:
            continue
        if _is_actor_runtime_buffer(name):
            continue
        b_target = target_buffers[name]
        if b_target.dtype.is_floating_point and b_online.dtype.is_floating_point:
            b_target.mul_(1.0 - gamma).add_(b_online.data, alpha=gamma)
        else:
            b_target.copy_(b_online.data)


def save_reg_snapshot(policy: nn.Module, path: str | Path) -> None:
    """Persist a regularization-policy snapshot to disk.

    Only the ``state_dict`` is stored — the caller is responsible for
    constructing a fresh ``PPOPolicy`` with matching hyperparameters at load
    time. This mirrors how the PPO opponent pool handles snapshots.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict()}, path)


def load_reg_snapshot_into(policy: nn.Module, path: str | Path) -> None:
    """Load a reg-policy snapshot into ``policy`` in-place, freezing grads."""

    payload: dict[str, Any] = torch.load(
        str(path),
        map_location="cpu",
        weights_only=False,
    )
    policy.load_state_dict(payload["state_dict"])
    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()


# ---------------------------------------------------------------------------
# Trajectory-level update orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RNaDStats:
    """Per-update statistics logged by :func:`rnad_update_trajectory`."""

    loss: float
    critic_loss: float
    policy_loss: float
    v_hat_mean: float
    grad_norm: float
    transformed_reward_mean: float
    q_clip_fraction: float = 0.0
    """Fraction of per-action Q values clipped by ``[-neurd_clip, +neurd_clip]``
    in the per-choice NeuRD loss this update. Tracked because — under issue 6 —
    the looser ``q_corr_rho_bar`` default lets the per-group sampled correction
    grow naturally, and a high clip fraction is the signal that the estimator
    has drifted."""

    sampled_log_ratio_mean: float = 0.0
    """Mean over own-turn sampled actions of ``log(π_θ(a*|o) / π_reg_blend(a*|o))``.
    This is the per-step regularization signal magnitude. If it grows over an
    inner loop, the online policy is drifting from the (frozen) reg snapshots
    faster than NeuRD's gradient is pulling it back — the regularization term
    is dominating the value target."""

    sampled_log_ratio_absmax: float = 0.0
    """Max over own-turn sampled actions of ``|log(π_θ(a*) / π_reg_blend(a*))|``.
    Outliers here directly produce huge transformed-reward injections per step."""

    is_bias_up_mean: float = 0.0
    """Mean of ``lp_target − logp_mu`` on steps where the online log-prob
    increased relative to the rollout-time log-prob (``lp_online > logp_mu``).
    A consistently *negative* value here means the target lags online's recent
    learning, so v-trace's IS ratio under-weights newly-discovered good actions
    in the value target — newly-learned wins get systematically discounted."""

    is_bias_down_mean: float = 0.0
    """Mean of ``lp_target − logp_mu`` on steps where ``lp_online < logp_mu``.
    Symmetric to :attr:`is_bias_up_mean`."""

    v_target_reg_share: float = 0.0
    """Mean over own-turn steps of ``|v_hat_reg − v_hat_terminal| /
    (|v_hat_terminal| + |v_hat_reg − v_hat_terminal| + 1e-8)``, where
    ``v_hat_terminal`` is v-trace fed only the terminal ±1 reward and
    ``v_hat_reg`` (= ``v_hat`` in the actual loss) is v-trace fed the full
    transformed reward. >> 0.5 means the value head is fitting the
    regularization landscape rather than the win/loss landscape."""


class _ReplayEvaluator(Protocol):
    """Structural type for the policy callable into :func:`rnad_update_trajectory`.

    See :meth:`magic_ai.model.PPOPolicy.evaluate_replay_batch`.
    """

    def evaluate_replay_batch(
        self, replay_rows: list[int]
    ) -> tuple[Tensor, Tensor, Tensor, Any]: ...

    def parameters(self) -> Any: ...  # for optimizer introspection + device


@dataclass(frozen=True)
class _TrajLossPieces:
    """Raw loss tensors + counts + scalar stats for one trajectory.

    Issue 7 (paper-faithful 1/t_effective normalization): each loss is the
    *sum* over its applicable steps/actions, paired with the count that
    sum was taken over. Multi-trajectory batchers can then normalize by the
    aggregate count rather than averaging per-episode means (which biases
    short games and players with fewer own-turn decisions).
    """

    cl_sum: Tensor
    cl_count: int
    pl_sum: Tensor
    pl_count: int
    n_q_clipped: int
    v_hat_mean: float
    transformed_mean: float
    # Diagnostics (issue: within-inner-loop policy degradation). All are sums
    # / maxes paired with their counts so the trainer can aggregate across
    # episodes without weighting bias.
    sampled_log_ratio_sum: float = 0.0
    sampled_log_ratio_absmax: float = 0.0
    sampled_log_ratio_count: int = 0
    is_bias_up_sum: float = 0.0
    is_bias_up_count: int = 0
    is_bias_down_sum: float = 0.0
    is_bias_down_count: int = 0
    v_target_reg_share_sum: float = 0.0
    v_target_reg_share_count: int = 0


def _maybe_recompute_lstm(policy: Any, replay_rows: list[int]) -> tuple[Tensor, Tensor] | None:
    """Re-run the policy's LSTM scan over a single episode (issue 2).

    Returns ``None`` when the policy has no LSTM or no
    :meth:`PPOPolicy.recompute_lstm_states_for_episode` method (e.g. a
    minimal stub used in unit tests).
    """

    fn = getattr(policy, "recompute_lstm_states_for_episode", None)
    if fn is None:
        return None
    return fn(list(replay_rows))


def _maybe_recompute_lstm_h_out(policy: Any, replay_rows: list[int]) -> Tensor | None:
    """Fused-recompute variant: returns top-layer ``h_out`` per replay step.

    Returns ``None`` when the policy has no LSTM or no
    :meth:`PPOPolicy.recompute_lstm_outputs_for_episodes` method. Caller
    is responsible for wrapping in ``torch.no_grad()`` for non-online
    policies (target / reg_cur / reg_prev) -- the helper itself runs
    under whatever grad mode the caller is in.
    """

    fn = getattr(policy, "recompute_lstm_outputs_for_episodes", None)
    if fn is None:
        return None
    result = fn([list(replay_rows)])
    if result is None:
        return None
    return result[0]


def _sampled_may_log_prob(pc: Any) -> Tensor:
    """Per-step Bernoulli log-prob of the sampled may decision under ``pc``.

    ``pc.may_logits_per_step`` holds zeros on non-may steps, so the
    returned tensor is also zero there — convenient for the combiner in
    :func:`_gather_sampled_scalar`.
    """

    prob_accept = torch.sigmoid(pc.may_logits_per_step)
    selected = pc.may_selected_per_step
    p_sel = torch.where(selected > 0.5, prob_accept, 1.0 - prob_accept)
    safe = p_sel.clamp_min(1e-30)
    return safe.log() * pc.may_is_active.to(dtype=safe.dtype)


def _gather_sampled_scalar(pc: Any, may_logp: Tensor) -> Tensor:
    """Assemble a per-step scalar log-prob from a per-choice forward.

    Decision-group steps pick the flat entry whose ``choice_col`` equals
    the step's sampled column; may steps use the precomputed
    ``may_logp``; steps with no action (shouldn't occur in practice) stay
    at zero.
    """

    device = may_logp.device
    n = may_logp.shape[0]
    out = may_logp.clone()
    if pc.flat_logits.numel() > 0:
        contributions = torch.where(
            pc.is_sampled_flat,
            pc.flat_log_probs,
            torch.zeros_like(pc.flat_log_probs),
        )
        per_step = torch.zeros(n, dtype=may_logp.dtype, device=device)
        per_step.scatter_add_(0, pc.group_idx, contributions.to(dtype=may_logp.dtype))
        out = out + per_step
    return out


def rnad_trajectory_loss(
    *,
    online: _ReplayEvaluator,
    target: _ReplayEvaluator,
    reg_cur: _ReplayEvaluator,
    reg_prev: _ReplayEvaluator,
    replay_rows: list[int],
    perspective_player_idx: Sequence[int],
    winner_idx: int,
    logp_mu: Tensor,
    config: RNaDConfig,
    alpha: float,
    online_h_out: Tensor | None = None,
    target_h_out: Tensor | None = None,
    reg_cur_h_out: Tensor | None = None,
    reg_prev_h_out: Tensor | None = None,
    compute_v_target_reg_share: bool = False,
) -> _TrajLossPieces:
    """Full per-action R-NaD trajectory loss (paper §170-189) with MTG factoring.

    Production R-NaD path: full per-action NeuRD with the ``[-beta, beta]``
    logit-magnitude gate, two-action Bernoulli NeuRD on the ``may`` head, and
    paper §179-180 per-action Q estimation. There is no sampled-action
    fallback — for a clean policy-gradient baseline use PPO instead.

    **Factored Magic actions (issue 4 — autoregressive decomposition.)**
    The paper assumes a single-categorical action per step. Magic steps are
    factored into one or more decision groups (each its own conditional
    softmax) plus an optional ``may`` Bernoulli. Each group is treated as
    its *own* paper-faithful step for purposes of NeuRD: per-group sampled
    correction with per-group ``1/mu_k``, per-group β-gated logit update,
    per-group ``-eta · log(pi/pi_reg)`` regularization on every legal
    choice. The trajectory-level pieces (v-trace, reward transform,
    critic) stay at step granularity since the reward is delivered at the
    joint step, not per group.

    **Behavior policy approximation.** Per-group rollout-time log-prob is
    not stored in the buffer; we substitute the *target* policy's per-group
    log-prob at the sampled action as ``mu_k``. This is consistent with the
    rnad_trainer simplification "rollouts effectively follow target via
    Polyak tracking" and means ``mu_k = pi_target(a_k* | o, a_<k)``. The
    joint v-trace IS ratio still uses the stored joint ``logp_mu`` from
    rollout time. See ``docs/rnad_design.md`` deviations section.

    **May head two-action NeuRD (issue 5).** The Bernoulli ``may`` is
    expanded to a 2-action softmax with logits ``(l, 0)``; both branches
    contribute to the gradient and both branches receive the per-action
    ``-eta · log(pi/pi_reg)`` regularization term (whereas the previous
    1-logit form regularized only the sampled branch).

    Returns sums + counts (issue 7) so the caller can normalize globally
    over the rollout batch's effective step / action counts rather than
    averaging per-episode means.
    """

    if not replay_rows:
        raise ValueError("replay_rows must be non-empty")
    t_len = len(replay_rows)
    if len(perspective_player_idx) != t_len:
        raise ValueError("perspective_player_idx must match replay_rows length")
    if logp_mu.shape != (t_len,):
        raise ValueError(f"logp_mu must have shape ({t_len},), got {tuple(logp_mu.shape)}")

    device = next(iter(online.parameters())).device
    perspective = torch.tensor(perspective_player_idx, dtype=torch.long, device=device)

    online_per_choice = getattr(online, "evaluate_replay_batch_per_choice")  # noqa: B009
    target_per_choice = getattr(target, "evaluate_replay_batch_per_choice")  # noqa: B009
    reg_cur_per_choice = getattr(reg_cur, "evaluate_replay_batch_per_choice")  # noqa: B009
    reg_prev_per_choice = getattr(reg_prev, "evaluate_replay_batch_per_choice")  # noqa: B009

    # Issue 2: per-policy LSTM recompute via the fused single-call path.
    # Each policy re-runs the LSTM scan from h=0 over the episode under its
    # own parameters; ``recompute_lstm_outputs_for_episodes`` does this in
    # one fused ``nn.LSTM`` call and returns the top-layer ``h_out`` per
    # step. The per-choice forward then consumes ``h_out`` directly via
    # ``hidden_override``, skipping the per-step LSTM cell. Online runs
    # under grad (full BPTT through the fused cuDNN backward); target /
    # reg_cur / reg_prev are no-grad. Callers (e.g. ``run_rnad_update``)
    # may precompute ``*_h_out`` once across the whole rollout batch -- if
    # provided, we skip recompute here and use them directly.
    if online_h_out is None:
        online_h_out = _maybe_recompute_lstm_h_out(online, replay_rows)
    if online_h_out is not None:
        lp_online, _, v_online, pc_online = online_per_choice(
            list(replay_rows), hidden_override=online_h_out
        )
    else:
        online_lstm = _maybe_recompute_lstm(online, replay_rows)
        lp_online, _, v_online, pc_online = online_per_choice(
            list(replay_rows), lstm_state_override=online_lstm
        )

    with torch.no_grad():
        if target_h_out is None:
            target_h_out = _maybe_recompute_lstm_h_out(target, replay_rows)
        if reg_cur_h_out is None:
            reg_cur_h_out = _maybe_recompute_lstm_h_out(reg_cur, replay_rows)
        if reg_prev_h_out is None:
            reg_prev_h_out = _maybe_recompute_lstm_h_out(reg_prev, replay_rows)

        if target_h_out is not None:
            _lp_tgt_scalar, _, v_tgt, pc_tgt = target_per_choice(
                list(replay_rows), hidden_override=target_h_out
            )
        else:
            target_lstm = _maybe_recompute_lstm(target, replay_rows)
            _lp_tgt_scalar, _, v_tgt, pc_tgt = target_per_choice(
                list(replay_rows), lstm_state_override=target_lstm
            )

        if reg_cur_h_out is not None:
            _, _, _, pc_reg_cur = reg_cur_per_choice(
                list(replay_rows), hidden_override=reg_cur_h_out
            )
        else:
            reg_cur_lstm = _maybe_recompute_lstm(reg_cur, replay_rows)
            _, _, _, pc_reg_cur = reg_cur_per_choice(
                list(replay_rows), lstm_state_override=reg_cur_lstm
            )

        if reg_prev_h_out is not None:
            _, _, _, pc_reg_prev = reg_prev_per_choice(
                list(replay_rows), hidden_override=reg_prev_h_out
            )
        else:
            reg_prev_lstm = _maybe_recompute_lstm(reg_prev, replay_rows)
            _, _, _, pc_reg_prev = reg_prev_per_choice(
                list(replay_rows), lstm_state_override=reg_prev_lstm
            )

    return _trajectory_loss_from_forwards(
        lp_online=lp_online,
        v_online=v_online,
        pc_online=pc_online,
        lp_tgt_scalar=_lp_tgt_scalar,
        v_tgt=v_tgt,
        pc_tgt=pc_tgt,
        pc_reg_cur=pc_reg_cur,
        pc_reg_prev=pc_reg_prev,
        perspective=perspective,
        winner_idx=winner_idx,
        logp_mu=logp_mu,
        config=config,
        alpha=alpha,
        compute_v_target_reg_share=compute_v_target_reg_share,
    )


def _trajectory_loss_from_forwards(
    *,
    lp_online: Tensor,
    v_online: Tensor,
    pc_online: Any,
    lp_tgt_scalar: Tensor,
    v_tgt: Tensor,
    pc_tgt: Any,
    pc_reg_cur: Any,
    pc_reg_prev: Any,
    perspective: Tensor,
    winner_idx: int,
    logp_mu: Tensor,
    config: RNaDConfig,
    alpha: float,
    compute_v_target_reg_share: bool = False,
) -> _TrajLossPieces:
    """Episode-local R-NaD loss from pre-computed per-policy forward outputs.

    Split out of :func:`rnad_trajectory_loss` so that
    :func:`rnad_batched_trajectory_loss` can run the four per-policy forwards
    once across an entire rollout batch and then assemble per-episode losses
    by slicing the batched outputs (issue: per-episode forwards dominated R-NaD
    update wall time).
    """

    t_len = int(lp_online.shape[0])
    device = v_online.device
    dtype = v_online.dtype
    logp_mu_dev = logp_mu.to(device=device, dtype=dtype)
    lp_online_dt = lp_online.detach().to(dtype=dtype)
    lp_tgt_scalar = lp_tgt_scalar.detach().to(dtype=dtype)
    lp_reg_cur_scalar_flat = pc_reg_cur.flat_log_probs.to(dtype=dtype)
    lp_reg_prev_scalar_flat = pc_reg_prev.flat_log_probs.to(dtype=dtype)
    v_tgt_dt = v_tgt.detach().to(dtype=dtype)

    lp_reg_cur_scalar = _gather_sampled_scalar(
        pc_reg_cur, may_logp=_sampled_may_log_prob(pc_reg_cur)
    ).to(dtype=dtype)
    lp_reg_prev_scalar = _gather_sampled_scalar(
        pc_reg_prev, may_logp=_sampled_may_log_prob(pc_reg_prev)
    ).to(dtype=dtype)

    # Blended reg log-probs per flat choice (paper §164 alpha interpolation).
    blended_reg_flat = alpha * lp_reg_cur_scalar_flat + (1.0 - alpha) * lp_reg_prev_scalar_flat
    log_ratio_flat = pc_online.flat_log_probs.to(dtype=dtype) - blended_reg_flat

    has_groups = pc_online.flat_logits.numel() > 0
    if has_groups:
        is_sampled_flat = pc_online.is_sampled_flat
        # Per-decision-group sampled log-ratio (sums to per-group scalar). Used
        # by the per-group sampled-correction term in flat_q below.
        n_groups = int(pc_online.step_for_decision_group.numel())
        per_group_log_ratio_sampled = torch.zeros(n_groups, dtype=dtype, device=device)
        per_group_log_ratio_sampled.scatter_add_(
            0,
            pc_online.decision_group_id_flat,
            torch.where(is_sampled_flat, log_ratio_flat, torch.zeros_like(log_ratio_flat)),
        )
        # Target per-group sampled log-prob (= log mu_k under the
        # rollouts-follow-target approximation). At training time, target's
        # per-flat log-prob is pc_tgt.flat_log_probs; the same is_sampled_flat
        # (selected from buffer.selected_indices, policy-independent) picks
        # out the sampled cell per group.
        lp_tgt_flat = pc_tgt.flat_log_probs.to(dtype=dtype)
        per_group_lp_mu = torch.zeros(n_groups, dtype=dtype, device=device)
        per_group_lp_mu.scatter_add_(
            0,
            pc_online.decision_group_id_flat,
            torch.where(is_sampled_flat, lp_tgt_flat, torch.zeros_like(lp_tgt_flat)),
        )
        # 1/mu_k clipped to ``q_corr_rho_bar`` (issue 6: per-group decomposition
        # makes blow-up far less likely than the joint 1/mu_t ever was, so the
        # default clip is loosened in :class:`RNaDConfig`). Track how often
        # the clip actually fires so the trainer can log it.
        inv_mu_per_group_pre = (-per_group_lp_mu).exp()
        inv_mu_per_group = inv_mu_per_group_pre.clamp(max=config.q_corr_rho_bar)
    else:
        is_sampled_flat = torch.zeros(0, dtype=torch.bool, device=device)
        per_group_log_ratio_sampled = torch.zeros(0, dtype=dtype, device=device)
        inv_mu_per_group = torch.zeros(0, dtype=dtype, device=device)
        inv_mu_per_group_pre = inv_mu_per_group

    cl_sum = v_online.new_zeros(())
    pl_sum = v_online.new_zeros(())
    cl_count_total = 0
    pl_count_total = 0
    n_q_clipped_total = 0
    v_hat_means: list[float] = []
    transformed_means: list[float] = []

    # ---- Diagnostics: per-step sampled log-ratio + IS bias ----
    # Both are per-step (one row per step), perspective-independent.
    with torch.no_grad():
        blended_reg_scalar = alpha * lp_reg_cur_scalar + (1.0 - alpha) * lp_reg_prev_scalar
        sampled_log_ratio_step = lp_online_dt - blended_reg_scalar
        sampled_log_ratio_sum_t = float(sampled_log_ratio_step.sum())
        sampled_log_ratio_absmax_t = float(sampled_log_ratio_step.abs().max()) if t_len > 0 else 0.0
        sampled_log_ratio_count_t = int(t_len)

        is_bias_step = lp_tgt_scalar - logp_mu_dev
        online_moved_up = lp_online_dt > logp_mu_dev
        up_count = int(online_moved_up.sum().item())
        down_count = t_len - up_count
        if up_count > 0:
            is_bias_up_sum_t = float(is_bias_step[online_moved_up].sum())
        else:
            is_bias_up_sum_t = 0.0
        if down_count > 0:
            is_bias_down_sum_t = float(is_bias_step[~online_moved_up].sum())
        else:
            is_bias_down_sum_t = 0.0

    v_target_reg_share_sum_t = 0.0
    v_target_reg_share_count_t = 0

    # ---- Perspective-independent setup (hoisted out of the (0, 1) loop) ----
    # Everything below is a function only of the per-policy forwards and
    # rollout-time data — none depends on which player's value we're
    # estimating. Computing once and reusing across both perspectives
    # halves the loop's work for these tensors.
    ratio_own_per_step = (lp_tgt_scalar - logp_mu_dev).exp()

    if has_groups:
        per_step_log_ratio_sampled = torch.zeros(t_len, dtype=dtype, device=device)
        per_step_log_ratio_sampled.scatter_add_(
            0, pc_online.step_for_decision_group, per_group_log_ratio_sampled
        )
        step_for_flat = pc_online.group_idx
        v_per_choice = v_tgt_dt[step_for_flat]
        base_q_flat = -config.eta * log_ratio_flat
    else:
        per_step_log_ratio_sampled = torch.zeros(0, dtype=dtype, device=device)
        step_for_flat = torch.zeros(0, dtype=torch.long, device=device)
        v_per_choice = torch.zeros(0, dtype=dtype, device=device)
        base_q_flat = torch.zeros(0, dtype=dtype, device=device)

    may_active_any = bool(pc_online.may_is_active.any().item())
    if may_active_any:
        may_logits_step = pc_online.may_logits_per_step.to(dtype=dtype)
        log_pi_accept = torch.nn.functional.logsigmoid(may_logits_step)
        log_pi_decline = torch.nn.functional.logsigmoid(-may_logits_step)
        with torch.no_grad():
            may_reg_cur_logits = pc_reg_cur.may_logits_per_step.to(dtype=dtype)
            may_reg_prev_logits = pc_reg_prev.may_logits_per_step.to(dtype=dtype)
            may_lp_reg_blend_accept = alpha * torch.nn.functional.logsigmoid(may_reg_cur_logits) + (
                1.0 - alpha
            ) * torch.nn.functional.logsigmoid(may_reg_prev_logits)
            may_lp_reg_blend_decline = alpha * torch.nn.functional.logsigmoid(
                -may_reg_cur_logits
            ) + (1.0 - alpha) * torch.nn.functional.logsigmoid(-may_reg_prev_logits)
        log_ratio_accept = log_pi_accept.detach() - may_lp_reg_blend_accept
        log_ratio_decline = log_pi_decline.detach() - may_lp_reg_blend_decline
        sampled_is_accept = pc_online.may_selected_per_step > 0.5
        with torch.no_grad():
            tgt_p_accept = torch.sigmoid(pc_tgt.may_logits_per_step.to(dtype=dtype))
            may_mu_sampled = torch.where(sampled_is_accept, tgt_p_accept, 1.0 - tgt_p_accept)
            may_inv_mu = (1.0 / may_mu_sampled.clamp_min(1e-30)).clamp(max=config.q_corr_rho_bar)
        may_sampled_log_ratio = torch.where(sampled_is_accept, log_ratio_accept, log_ratio_decline)

    for own_idx in (0, 1):
        is_own = perspective == own_idx
        rewards = torch.zeros(t_len, dtype=dtype, device=device)
        if winner_idx >= 0:
            rewards[-1] = 1.0 if winner_idx == own_idx else -1.0

        transformed = transform_rewards(
            rewards,
            logp_theta=lp_online_dt,
            logp_reg_cur=lp_reg_cur_scalar,
            logp_reg_prev=lp_reg_prev_scalar,
            alpha=alpha,
            eta=config.eta,
            perspective_is_player_i=is_own,
        )
        v_out = two_player_vtrace(
            rewards=transformed.detach(),
            values=v_tgt_dt,
            logp_theta=lp_tgt_scalar,
            logp_mu=logp_mu_dev,
            perspective_is_player_i=is_own,
            rho_bar=config.vtrace_rho_bar,
            c_bar=config.vtrace_c_bar,
        )

        # Diagnostic: v-trace fed only the terminal ±1 reward (no per-step
        # regularization injection). Comparing |v_hat_reg - v_hat_terminal|
        # against |v_hat_terminal| tells us how much of the value target is
        # the regularization landscape vs the win/loss landscape. This is a
        # full extra v-trace pass per perspective per episode -- gated to
        # opt-in via ``compute_v_target_reg_share``; the trainer typically
        # enables it once per N gradient steps so the cost is amortized.
        if compute_v_target_reg_share:
            with torch.no_grad():
                v_out_term = two_player_vtrace(
                    rewards=rewards.detach(),
                    values=v_tgt_dt,
                    logp_theta=lp_tgt_scalar,
                    logp_mu=logp_mu_dev,
                    perspective_is_player_i=is_own,
                    rho_bar=config.vtrace_rho_bar,
                    c_bar=config.vtrace_c_bar,
                )
                if is_own.any():
                    v_reg_part = (v_out.v_hat - v_out_term.v_hat)[is_own].abs()
                    v_term_part = v_out_term.v_hat[is_own].abs()
                    share = v_reg_part / (v_term_part + v_reg_part + 1e-8)
                    v_target_reg_share_sum_t += float(share.sum())
                    v_target_reg_share_count_t += int(is_own.sum().item())

        cl_part_sum, cl_part_count = critic_loss(
            v_theta=v_online,
            v_hat=v_out.v_hat,
            perspective_is_player_i=is_own,
        )
        cl_sum = cl_sum + cl_part_sum
        cl_count_total += cl_part_count

        # ---- Per-group sampled inner bracket (paper §177 adapted) ----
        # inner_t = r_t + eta * log_ratio_at_sampled_step + (pi/mu)_t * (r̂+v̂) - v(o_t)
        # ``per_step_log_ratio_sampled``, ``ratio_own_per_step``, and
        # ``v_per_choice`` are perspective-independent and hoisted above; only
        # ``rewards`` and ``v_out`` change between perspectives.
        if has_groups:
            sampled_inner_per_step = (
                rewards
                + config.eta * per_step_log_ratio_sampled
                + ratio_own_per_step * (v_out.r_hat_next + v_out.v_hat_next)
                - v_tgt_dt
            )
        else:
            sampled_inner_per_step = torch.zeros(t_len, dtype=dtype, device=device)

        # ---- May head: true two-action NeuRD (issue 5) ----
        # Reg blends, log-ratios, and 1/mu are perspective-independent and
        # hoisted above; only rewards / v_out / is_own change here.
        if may_active_any:
            may_and_own = pc_online.may_is_active & is_own
            may_sampled_inner = (
                rewards
                + config.eta * may_sampled_log_ratio
                + ratio_own_per_step * (v_out.r_hat_next + v_out.v_hat_next)
                - v_tgt_dt
            )
            may_q_sampled_correction = may_inv_mu * may_sampled_inner
            q_accept = (
                -config.eta * log_ratio_accept
                + torch.where(
                    sampled_is_accept, may_q_sampled_correction, torch.zeros_like(rewards)
                )
                + v_tgt_dt
            )
            q_decline = (
                -config.eta * log_ratio_decline
                + torch.where(
                    ~sampled_is_accept, may_q_sampled_correction, torch.zeros_like(rewards)
                )
                + v_tgt_dt
            )
            may_pl_sum, may_pl_count = may_neurd_loss(
                may_logits=may_logits_step,
                q_accept=q_accept,
                q_decline=q_decline,
                own_turn_may_mask=may_and_own,
                beta=config.neurd_beta,
                clip=config.neurd_clip,
                lr=config.learning_rate,
            )
            pl_sum = pl_sum + may_pl_sum
            pl_count_total += may_pl_count

        # ---- Decision-group per-action NeuRD with per-group correction ----
        if has_groups:
            step_is_own_flat = is_own[step_for_flat]
            sampled_correction_per_group = (
                inv_mu_per_group * sampled_inner_per_step[pc_online.step_for_decision_group]
            )
            sampled_correction_flat = sampled_correction_per_group[pc_online.decision_group_id_flat]
            flat_q = (
                base_q_flat
                + torch.where(
                    is_sampled_flat,
                    sampled_correction_flat,
                    torch.zeros_like(sampled_correction_flat),
                )
                + v_per_choice
            )
            flat_q = torch.where(step_is_own_flat, flat_q, torch.zeros_like(flat_q))
            pl_part_sum, pl_part_count, n_q_clipped = neurd_loss_per_choice(
                flat_logits=pc_online.flat_logits,
                flat_q=flat_q,
                flat_active_mask=step_is_own_flat,
                beta=config.neurd_beta,
                clip=config.neurd_clip,
                lr=config.learning_rate,
            )
            pl_sum = pl_sum + pl_part_sum
            pl_count_total += pl_part_count
            n_q_clipped_total += n_q_clipped

        v_hat_means.append(float(v_out.v_hat.detach().mean()))
        transformed_means.append(float(transformed.detach().mean()))

    return _TrajLossPieces(
        cl_sum=cl_sum,
        cl_count=cl_count_total,
        pl_sum=pl_sum,
        pl_count=pl_count_total,
        n_q_clipped=n_q_clipped_total,
        v_hat_mean=sum(v_hat_means) / len(v_hat_means),
        transformed_mean=sum(transformed_means) / len(transformed_means),
        sampled_log_ratio_sum=sampled_log_ratio_sum_t,
        sampled_log_ratio_absmax=sampled_log_ratio_absmax_t,
        sampled_log_ratio_count=sampled_log_ratio_count_t,
        is_bias_up_sum=is_bias_up_sum_t,
        is_bias_up_count=up_count,
        is_bias_down_sum=is_bias_down_sum_t,
        is_bias_down_count=down_count,
        v_target_reg_share_sum=v_target_reg_share_sum_t,
        v_target_reg_share_count=v_target_reg_share_count_t,
    )


def _slice_per_choice(
    pc: Any,
    *,
    step_lo: int,
    step_hi: int,
    flat_lo: int,
    flat_hi: int,
    dg_lo: int,
    dg_hi: int,
) -> Any:
    """Slice a batched ``ReplayPerChoice`` to a single episode.

    Re-bases ``group_idx`` (per-flat-entry → step), ``decision_group_id_flat``
    (per-flat-entry → unique decision-group id), and ``step_for_decision_group``
    (per-decision-group → step) into the episode-local index space [0, T_ep)
    and [0, n_dg_ep). Caller is responsible for supplying the correct
    boundary indices (``rnad_batched_trajectory_loss`` derives them via
    one ``torch.searchsorted`` per pc).
    """

    flat_logits = pc.flat_logits[flat_lo:flat_hi]
    flat_log_probs = pc.flat_log_probs[flat_lo:flat_hi]
    group_idx = pc.group_idx[flat_lo:flat_hi]
    if group_idx.numel() > 0:
        group_idx = group_idx - step_lo
    decision_group_id_flat = pc.decision_group_id_flat[flat_lo:flat_hi]
    if decision_group_id_flat.numel() > 0:
        decision_group_id_flat = decision_group_id_flat - dg_lo
    step_for_decision_group = pc.step_for_decision_group[dg_lo:dg_hi]
    if step_for_decision_group.numel() > 0:
        step_for_decision_group = step_for_decision_group - step_lo

    return dataclasses.replace(
        pc,
        flat_logits=flat_logits,
        flat_log_probs=flat_log_probs,
        group_idx=group_idx,
        choice_cols=pc.choice_cols[flat_lo:flat_hi],
        is_sampled_flat=pc.is_sampled_flat[flat_lo:flat_hi],
        decision_group_id_flat=decision_group_id_flat,
        step_for_decision_group=step_for_decision_group,
        may_is_active=pc.may_is_active[step_lo:step_hi],
        may_logits_per_step=pc.may_logits_per_step[step_lo:step_hi],
        may_selected_per_step=pc.may_selected_per_step[step_lo:step_hi],
    )


def _maybe_recompute_lstm_h_out_episodes(
    policy: Any, episodes: Sequence[Sequence[int]]
) -> list[Tensor] | None:
    """Batched LSTM-output recompute over a list of episodes.

    Calls :meth:`PPOPolicy.recompute_lstm_outputs_for_episodes` once with the
    full episode list -- the model-side helper performs a single fused
    ``nn.LSTM`` call for the whole batch. Returns ``None`` for non-recurrent
    policies or stubs lacking the helper.
    """

    fn = getattr(policy, "recompute_lstm_outputs_for_episodes", None)
    if fn is None:
        return None
    return fn([list(ep) for ep in episodes])


def rnad_batched_trajectory_loss(
    *,
    online: _ReplayEvaluator,
    target: _ReplayEvaluator,
    reg_cur: _ReplayEvaluator,
    reg_prev: _ReplayEvaluator,
    episodes_replay_rows: Sequence[Sequence[int]],
    episodes_perspective: Sequence[Sequence[int]],
    episodes_winner_idx: Sequence[int],
    episodes_logp_mu: Sequence[Tensor],
    config: RNaDConfig,
    alpha: float,
    compute_v_target_reg_share: bool = False,
) -> list[_TrajLossPieces]:
    """Batched per-policy forwards across all episodes; per-episode loss assembly.

    Replaces the per-episode forward pattern of :func:`rnad_trajectory_loss`
    when called once per episode (4 small per-choice forwards × N episodes ×
    4 LSTM scans). This function:

    1. Concatenates all episode replay rows into one flat batch.
    2. Runs one fused LSTM scan per policy (online with grad, target/reg
       no-grad) over the concatenated batch.
    3. Runs one ``evaluate_replay_batch_per_choice`` per policy on the
       full flat batch.
    4. Slices the batched outputs per episode and reuses the existing
       :func:`_trajectory_loss_from_forwards` to assemble per-episode
       loss pieces.

    Memory: holds activations for *all* episodes simultaneously. The caller
    is expected to size the rollout batch accordingly; if memory becomes a
    constraint, mini-batch the episode list and call this function multiple
    times per optimizer step.
    """

    if not episodes_replay_rows:
        raise ValueError("episodes_replay_rows must be non-empty")
    n_eps = len(episodes_replay_rows)
    if len(episodes_perspective) != n_eps:
        raise ValueError("episodes_perspective length mismatch")
    if len(episodes_winner_idx) != n_eps:
        raise ValueError("episodes_winner_idx length mismatch")
    if len(episodes_logp_mu) != n_eps:
        raise ValueError("episodes_logp_mu length mismatch")

    flat_rows: list[int] = []
    ep_offsets: list[int] = [0]
    for ep in episodes_replay_rows:
        if not ep:
            raise ValueError("each episode must contain at least one row")
        flat_rows.extend(int(r) for r in ep)
        ep_offsets.append(ep_offsets[-1] + len(ep))

    device = next(iter(online.parameters())).device

    # 1) Single fused LSTM recompute per policy.
    online_h_list = _maybe_recompute_lstm_h_out_episodes(online, episodes_replay_rows)
    with torch.no_grad():
        target_h_list = _maybe_recompute_lstm_h_out_episodes(target, episodes_replay_rows)
        reg_cur_h_list = _maybe_recompute_lstm_h_out_episodes(reg_cur, episodes_replay_rows)
        reg_prev_h_list = _maybe_recompute_lstm_h_out_episodes(reg_prev, episodes_replay_rows)

    online_h_concat = torch.cat(online_h_list, dim=0) if online_h_list is not None else None
    target_h_concat = torch.cat(target_h_list, dim=0) if target_h_list is not None else None
    reg_cur_h_concat = torch.cat(reg_cur_h_list, dim=0) if reg_cur_h_list is not None else None
    reg_prev_h_concat = torch.cat(reg_prev_h_list, dim=0) if reg_prev_h_list is not None else None

    online_per_choice = getattr(online, "evaluate_replay_batch_per_choice")  # noqa: B009
    target_per_choice = getattr(target, "evaluate_replay_batch_per_choice")  # noqa: B009
    reg_cur_per_choice = getattr(reg_cur, "evaluate_replay_batch_per_choice")  # noqa: B009
    reg_prev_per_choice = getattr(reg_prev, "evaluate_replay_batch_per_choice")  # noqa: B009

    def _forward(per_choice: Any, h_concat: Tensor | None) -> Any:
        if h_concat is not None:
            return per_choice(list(flat_rows), hidden_override=h_concat)
        return per_choice(list(flat_rows))

    # 2) Single per-choice forward per policy across the full batch.
    lp_online_b, _, v_online_b, pc_online_b = _forward(online_per_choice, online_h_concat)
    with torch.no_grad():
        lp_tgt_b, _, v_tgt_b, pc_tgt_b = _forward(target_per_choice, target_h_concat)
        _, _, _, pc_reg_cur_b = _forward(reg_cur_per_choice, reg_cur_h_concat)
        _, _, _, pc_reg_prev_b = _forward(reg_prev_per_choice, reg_prev_h_concat)

    # 3) Episode-boundary slice indices (one searchsorted + one .tolist()
    # sync per policy, instead of 4 .item() syncs per episode per policy).
    boundaries_t = torch.tensor(ep_offsets, dtype=torch.long, device=device)

    def _slice_bounds(pc: Any) -> tuple[list[int], list[int]]:
        if pc.flat_logits.numel() == 0:
            zeros = [0] * (n_eps + 1)
            return zeros, zeros
        flat_bounds = torch.searchsorted(pc.group_idx, boundaries_t).tolist()
        dg_bounds = torch.searchsorted(pc.step_for_decision_group, boundaries_t).tolist()
        return flat_bounds, dg_bounds

    fb_on, dg_on = _slice_bounds(pc_online_b)
    fb_tg, dg_tg = _slice_bounds(pc_tgt_b)
    fb_rc, dg_rc = _slice_bounds(pc_reg_cur_b)
    fb_rp, dg_rp = _slice_bounds(pc_reg_prev_b)

    # 4) Per-episode loss assembly via slicing.
    pieces_list: list[_TrajLossPieces] = []
    for ep_idx in range(n_eps):
        step_lo = ep_offsets[ep_idx]
        step_hi = ep_offsets[ep_idx + 1]
        ep_perspective = torch.tensor(
            list(episodes_perspective[ep_idx]), dtype=torch.long, device=device
        )
        pc_online_ep = _slice_per_choice(
            pc_online_b,
            step_lo=step_lo,
            step_hi=step_hi,
            flat_lo=fb_on[ep_idx],
            flat_hi=fb_on[ep_idx + 1],
            dg_lo=dg_on[ep_idx],
            dg_hi=dg_on[ep_idx + 1],
        )
        pc_tgt_ep = _slice_per_choice(
            pc_tgt_b,
            step_lo=step_lo,
            step_hi=step_hi,
            flat_lo=fb_tg[ep_idx],
            flat_hi=fb_tg[ep_idx + 1],
            dg_lo=dg_tg[ep_idx],
            dg_hi=dg_tg[ep_idx + 1],
        )
        pc_reg_cur_ep = _slice_per_choice(
            pc_reg_cur_b,
            step_lo=step_lo,
            step_hi=step_hi,
            flat_lo=fb_rc[ep_idx],
            flat_hi=fb_rc[ep_idx + 1],
            dg_lo=dg_rc[ep_idx],
            dg_hi=dg_rc[ep_idx + 1],
        )
        pc_reg_prev_ep = _slice_per_choice(
            pc_reg_prev_b,
            step_lo=step_lo,
            step_hi=step_hi,
            flat_lo=fb_rp[ep_idx],
            flat_hi=fb_rp[ep_idx + 1],
            dg_lo=dg_rp[ep_idx],
            dg_hi=dg_rp[ep_idx + 1],
        )
        pieces = _trajectory_loss_from_forwards(
            lp_online=lp_online_b[step_lo:step_hi],
            v_online=v_online_b[step_lo:step_hi],
            pc_online=pc_online_ep,
            lp_tgt_scalar=lp_tgt_b[step_lo:step_hi],
            v_tgt=v_tgt_b[step_lo:step_hi],
            pc_tgt=pc_tgt_ep,
            pc_reg_cur=pc_reg_cur_ep,
            pc_reg_prev=pc_reg_prev_ep,
            perspective=ep_perspective,
            winner_idx=int(episodes_winner_idx[ep_idx]),
            logp_mu=episodes_logp_mu[ep_idx],
            config=config,
            alpha=alpha,
            compute_v_target_reg_share=compute_v_target_reg_share,
        )
        pieces_list.append(pieces)

    return pieces_list


def rnad_update_trajectory(
    *,
    online: _ReplayEvaluator,
    target: _ReplayEvaluator,
    reg_cur: _ReplayEvaluator,
    reg_prev: _ReplayEvaluator,
    optimizer: torch.optim.Optimizer,
    replay_rows: list[int],
    perspective_player_idx: Sequence[int],
    winner_idx: int,
    logp_mu: Tensor,
    config: RNaDConfig,
    alpha: float,
) -> RNaDStats:
    """Single-trajectory wrapper: loss + backward + optimizer + Polyak."""

    pieces = rnad_trajectory_loss(
        online=online,
        target=target,
        reg_cur=reg_cur,
        reg_prev=reg_prev,
        replay_rows=replay_rows,
        perspective_player_idx=perspective_player_idx,
        winner_idx=winner_idx,
        logp_mu=logp_mu,
        config=config,
        alpha=alpha,
    )
    cl = pieces.cl_sum / max(pieces.cl_count, 1)
    pl = pieces.pl_sum / max(pieces.pl_count, 1)
    loss = cl + pl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [p for p in online.parameters() if p.requires_grad]
    grad_norm = float(
        nn.utils.clip_grad_norm_(
            trainable,
            max_norm=config.grad_clip,
        )
    )
    optimizer.step()

    assert isinstance(target, nn.Module)
    assert isinstance(online, nn.Module)
    polyak_update_(target, online, gamma=config.target_ema_gamma)

    return RNaDStats(
        loss=float(loss.detach()),
        critic_loss=float(cl.detach()),
        policy_loss=float(pl.detach()),
        v_hat_mean=pieces.v_hat_mean,
        grad_norm=grad_norm,
        transformed_reward_mean=pieces.transformed_mean,
    )


def episodes_from_rollout_steps(
    perspective_player_idx: Sequence[int],
) -> list[tuple[int, int]]:
    """Segment a flat rollout into (start, end_exclusive) episode bounds.

    An episode is a maximal run of steps in which the alternation pattern
    of ``perspective_player_idx`` is preserved. We detect boundaries by
    comparing the current step's perspective with the *expected* alternation
    from the previous step. This is a lightweight heuristic used only when
    the caller does not already have episode-id metadata; the real trainer
    passes explicit episode boundaries from the rollout buffer.
    """

    if not perspective_player_idx:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    prev = perspective_player_idx[0]
    for i in range(1, len(perspective_player_idx)):
        cur = perspective_player_idx[i]
        if cur == prev:
            # Two consecutive same-perspective rows => episode boundary.
            bounds.append((start, i))
            start = i
        prev = cur
    bounds.append((start, len(perspective_player_idx)))
    return bounds
