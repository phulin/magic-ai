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

    learning_rate: float = 5e-5
    """Optimizer learning rate (paper §199: 5e-5). Used by the NeuRD
    gradient gate (paper §189) to evaluate the post-update logit predicate
    ``|logit + lr · Clip(Q, c)| <= beta`` rather than the looser
    current-logit predicate ``|logit| <= beta``."""


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
    group_idx: Tensor,
    num_groups: int,
    beta: float,
    clip: float,
    lr: float = 0.0,
) -> Tensor:
    """NeuRD loss over a flattened, ragged set of per-choice logits.

    The decision-group structure of :meth:`PPOPolicy.evaluate_replay_batch`
    produces a flat tensor of per-legal-choice logits together with a
    ``group_idx`` mapping each entry to its source step. This form
    supports variable-size legal-action sets without padding.

    ``num_groups`` is the number of decision steps in the batch (used
    only as the averaging denominator, to keep the loss scale
    comparable to the dense :func:`neurd_loss`).
    """

    if flat_logits.shape != flat_q.shape:
        raise ValueError("flat_logits and flat_q must share shape")
    if flat_logits.shape != group_idx.shape:
        raise ValueError("flat_logits and group_idx must share shape")
    if num_groups < 1:
        raise ValueError("num_groups must be >= 1")
    q_clipped = flat_q.clamp(-clip, clip).detach()
    with torch.no_grad():
        # Paper §189 post-update logit predicate; see :func:`neurd_loss`.
        post_update = flat_logits + lr * q_clipped
        in_range = (post_update >= -beta) & (post_update <= beta)
    per_entry = -flat_logits * q_clipped * in_range.to(dtype=flat_logits.dtype)
    return per_entry.sum() / max(num_groups, 1)


# ---------------------------------------------------------------------------
# Critic loss
# ---------------------------------------------------------------------------


def critic_loss(
    *,
    v_theta: Tensor,
    v_hat: Tensor,
    perspective_is_player_i: Tensor,
) -> Tensor:
    """L1 regression of the online value head against the v-trace target.

    Only own-turn steps contribute (paper §186). ``v_theta`` and ``v_hat``
    are both shape ``(T,)``; the boolean mask selects own-turn indices.
    """

    if v_theta.shape != v_hat.shape:
        raise ValueError("v_theta and v_hat must share shape")
    if v_theta.shape != perspective_is_player_i.shape:
        raise ValueError("v_theta and perspective mask must share shape")

    mask = perspective_is_player_i.to(dtype=torch.bool)
    if not mask.any():
        return v_theta.new_zeros(())
    diff = (v_theta - v_hat.detach()).abs()
    return diff[mask].mean()


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


# ---------------------------------------------------------------------------
# Sampled-action NeuRD (simplified policy loss usable from the existing
# scalar-log-prob replay pipeline — see :func:`neurd_loss` for the full
# per-action form used with ``evaluate_replay_batch_rnad``)
# ---------------------------------------------------------------------------


def sampled_neurd_loss(
    *,
    log_prob: Tensor,
    q_hat: Tensor,
    own_turn_mask: Tensor,
    clip: float,
) -> Tensor:
    """Sampled-action NeuRD loss.

    The full NeuRD gradient (paper §188) is

        -sum_a grad(logit_a) * clip(Q(a), c) * 1[logit_a in [-beta, beta]]

    summed over all legal actions. This variant uses only the sampled action's
    log-prob as a stochastic estimator:

        L = - mean_{t: own-turn} log_prob(t) * clip(Q(t), [-clip, clip])

    This is the policy-gradient-theorem form of NeuRD and is consistent with
    how the PPO replay pipeline currently exposes per-step log-probs. The β
    gate on individual logit magnitudes is not applied here; the full form
    with per-action logits lands in a later phase once ``evaluate_replay_batch``
    grows a per-choice-logit return mode.
    """

    if log_prob.shape != q_hat.shape:
        raise ValueError("log_prob and q_hat must share shape")
    if log_prob.shape != own_turn_mask.shape:
        raise ValueError("log_prob and own_turn_mask must share shape")
    mask = own_turn_mask.to(dtype=torch.bool)
    if not mask.any():
        return log_prob.new_zeros(())
    q_clipped = q_hat.clamp(-clip, clip).detach()
    per_step = -log_prob * q_clipped
    return per_step[mask].mean()


def may_neurd_loss(
    *,
    may_logits: Tensor,
    may_selected: Tensor,
    q_hat: Tensor,
    own_turn_may_mask: Tensor,
    beta: float,
    clip: float,
    lr: float = 0.0,
) -> Tensor:
    """β-gated NeuRD loss for the Bernoulli ``may`` head.

    Treats the may head as a 1-logit Bernoulli (accept vs decline). The
    full NeuRD gradient on a Bernoulli with logit ``l`` is

        d(loss)/dl = -(1[a=accept] - sigmoid(l)) * clip(Q, c) * gate(|l| <= beta)

    which matches the sampled-action policy-gradient-theorem form with a
    per-step β-magnitude gate — the Bernoulli analogue of
    :func:`neurd_loss_per_choice`.
    """

    if may_logits.shape != may_selected.shape:
        raise ValueError("may_logits and may_selected must share shape")
    if may_logits.shape != q_hat.shape:
        raise ValueError("may_logits and q_hat must share shape")
    mask = own_turn_may_mask.to(dtype=torch.bool)
    if not mask.any():
        return may_logits.new_zeros(())
    prob_accept = torch.sigmoid(may_logits)
    advantage = may_selected.to(dtype=may_logits.dtype) - prob_accept
    q_clipped = q_hat.clamp(-clip, clip).detach()
    with torch.no_grad():
        # Paper §189 post-update logit predicate. The Bernoulli NeuRD
        # gradient on may_logits is ``advantage · Clip(Q, c)``; gate on the
        # post-update logit ``logit + lr · advantage · Clip(Q, c)``.
        post_update = may_logits + lr * advantage * q_clipped
        in_range = post_update.abs() <= beta
    per_step = -advantage * q_clipped * in_range.to(dtype=may_logits.dtype)
    return per_step[mask].mean()


# ---------------------------------------------------------------------------
# Polyak target network + reg snapshot persistence
# ---------------------------------------------------------------------------


@torch.no_grad()
def polyak_update_(target: nn.Module, online: nn.Module, gamma: float) -> None:
    """In-place Polyak averaging: ``target <- gamma * online + (1 - gamma) * target``.

    ``gamma`` is the "target learning rate" in the DeepNash paper (§191),
    typically very small (1e-3). Only floating-point parameters and buffers
    are averaged; integer buffers (e.g. LSTM indices) are copied verbatim.
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
    """Raw loss tensors + scalar stats for one trajectory."""

    cl: Tensor
    pl: Tensor
    v_hat_mean: float
    transformed_mean: float


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
) -> _TrajLossPieces:
    """Compute the R-NaD critic + sampled-action NeuRD losses for one trajectory.

    Stateless: returns the loss tensors (with grad), leaving the backward +
    optimizer + Polyak step to the caller. :func:`run_rnad_update` calls
    this per episode in a rollout batch and accumulates losses for a single
    backward/step/Polyak cycle; that amortizes optimizer overhead over many
    short episodes and keeps GPU saturation high.
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

    # --- forwards (one set of passes, reused across both player branches) --
    logp_theta, _entropies_online, values_online, _ = online.evaluate_replay_batch(
        list(replay_rows)
    )
    with torch.no_grad():
        logp_tgt, _e_tgt, values_tgt, _ = target.evaluate_replay_batch(list(replay_rows))
        logp_reg_cur, _e_rc, _v_rc, _ = reg_cur.evaluate_replay_batch(list(replay_rows))
        logp_reg_prev, _e_rp, _v_rp, _ = reg_prev.evaluate_replay_batch(list(replay_rows))

    dtype = values_online.dtype
    logp_mu_dev = logp_mu.to(device=device, dtype=dtype)
    logp_theta_dt = logp_theta.detach().to(dtype=dtype)
    logp_tgt_dt = logp_tgt.detach().to(dtype=dtype)
    logp_reg_cur_dt = logp_reg_cur.to(dtype=dtype)
    logp_reg_prev_dt = logp_reg_prev.to(dtype=dtype)
    values_tgt_dt = values_tgt.detach().to(dtype=dtype)

    total_cl = values_online.new_zeros(())
    total_pl = values_online.new_zeros(())
    v_hat_means: list[float] = []
    transformed_means: list[float] = []

    for own_idx in (0, 1):
        is_own = perspective == own_idx
        rewards = torch.zeros(t_len, dtype=dtype, device=device)
        if winner_idx >= 0:
            rewards[-1] = 1.0 if winner_idx == own_idx else -1.0

        transformed = transform_rewards(
            rewards,
            logp_theta=logp_theta_dt,
            logp_reg_cur=logp_reg_cur_dt,
            logp_reg_prev=logp_reg_prev_dt,
            alpha=alpha,
            eta=config.eta,
            perspective_is_player_i=is_own,
        )

        v_out = two_player_vtrace(
            rewards=transformed.detach(),
            values=values_tgt_dt,
            logp_theta=logp_tgt_dt,
            logp_mu=logp_mu_dev,
            perspective_is_player_i=is_own,
            rho_bar=config.vtrace_rho_bar,
            c_bar=config.vtrace_c_bar,
        )

        total_cl = total_cl + critic_loss(
            v_theta=values_online,
            v_hat=v_out.v_hat,
            perspective_is_player_i=is_own,
        )
        total_pl = total_pl + sampled_neurd_loss(
            log_prob=logp_theta,
            q_hat=v_out.q_hat,
            own_turn_mask=is_own,
            clip=config.neurd_clip,
        )
        v_hat_means.append(float(v_out.v_hat.detach().mean()))
        transformed_means.append(float(transformed.detach().mean()))

    return _TrajLossPieces(
        cl=total_cl,
        pl=total_pl,
        v_hat_mean=sum(v_hat_means) / len(v_hat_means),
        transformed_mean=sum(transformed_means) / len(transformed_means),
    )


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
    """Single-trajectory convenience wrapper (loss + backward + step + Polyak).

    Thin wrapper around :func:`rnad_trajectory_loss` kept for tests and
    single-episode driver scripts. The production trainer
    (:func:`magic_ai.rnad_trainer.run_rnad_update`) batches trajectories and
    calls the loss-only helper directly to amortize optimizer + Polyak
    overhead across a rollout batch.
    """

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
    loss = pieces.cl + pieces.pl

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
        critic_loss=float(pieces.cl.detach()),
        policy_loss=float(pieces.pl.detach()),
        v_hat_mean=pieces.v_hat_mean,
        grad_norm=grad_norm,
        transformed_reward_mean=pieces.transformed_mean,
    )


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
        sampled_col_flat = pc.sampled_col_per_step[pc.group_idx]
        is_sampled = pc.choice_cols == sampled_col_flat
        contributions = torch.where(
            is_sampled,
            pc.flat_log_probs,
            torch.zeros_like(pc.flat_log_probs),
        )
        per_step = torch.zeros(n, dtype=may_logp.dtype, device=device)
        per_step.scatter_add_(0, pc.group_idx, contributions.to(dtype=may_logp.dtype))
        out = out + per_step
    return out


def rnad_trajectory_loss_full_neurd(
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
) -> _TrajLossPieces:
    """Loss-only variant of :func:`rnad_update_trajectory_full_neurd`.

    Uses :func:`neurd_loss_per_choice` over all legal per-choice logits
    with the paper's ``[-beta, beta]`` logit-magnitude gate, and
    :func:`may_neurd_loss` (beta-gated) on the Bernoulli ``may`` head.

    Per-choice Q tracks the paper's estimator (§179-180) rather than the
    sampled-action stand-in:

        Q(a) = v_tgt(o_t) - eta * log_ratio(a)
             + 1{a = a_sampled} * (q_hat_sampled(t) - v_tgt(o_t)
                                    + eta * log_ratio(a_sampled))

    where ``log_ratio(a) = log pi_theta(a) - log pi_reg_blend(a)`` is
    computed per legal choice from the same per-choice forwards used for
    the online logits, and ``q_hat_sampled`` is the v-trace scalar at the
    sampled action. The ``v(o_t)`` additive constant preserves the
    paper's form for the β-gated NeuRD (it is not shift-invariant under
    the gate).

    Requires online/target/reg policies to expose
    ``evaluate_replay_batch_per_choice``.
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

    # Per-choice forward on online; per-choice forwards on reg policies are
    # also required so the full-NeuRD Q estimate includes log(pi/pi_reg)[a]
    # for every legal a, not just the sampled one.
    online_per_choice = getattr(online, "evaluate_replay_batch_per_choice")  # noqa: B009
    target_per_choice = getattr(target, "evaluate_replay_batch_per_choice")  # noqa: B009
    reg_cur_per_choice = getattr(reg_cur, "evaluate_replay_batch_per_choice")  # noqa: B009
    reg_prev_per_choice = getattr(reg_prev, "evaluate_replay_batch_per_choice")  # noqa: B009
    lp_online, _, v_online, pc_online = online_per_choice(list(replay_rows))
    with torch.no_grad():
        _lp_tgt_scalar, _, v_tgt, pc_tgt = target_per_choice(list(replay_rows))
        _, _, _, pc_reg_cur = reg_cur_per_choice(list(replay_rows))
        _, _, _, pc_reg_prev = reg_prev_per_choice(list(replay_rows))

    dtype = v_online.dtype
    logp_mu_dev = logp_mu.to(device=device, dtype=dtype)
    lp_online_dt = lp_online.detach().to(dtype=dtype)
    # Target per-step log-prob at sampled action is needed by v-trace for
    # the importance ratio. We reconstruct it from pc_tgt.flat_log_probs by
    # picking the entry whose choice_col matches each step's sampled column.
    lp_tgt_scalar = _lp_tgt_scalar.detach().to(dtype=dtype)
    lp_reg_cur_scalar_flat = pc_reg_cur.flat_log_probs.to(dtype=dtype)
    lp_reg_prev_scalar_flat = pc_reg_prev.flat_log_probs.to(dtype=dtype)
    v_tgt_dt = v_tgt.detach().to(dtype=dtype)

    # Scalar reg log-probs (for reward transform): pick sampled-choice entry
    # per step; fall back to may-head log_prob for may steps.
    lp_reg_cur_scalar = _gather_sampled_scalar(
        pc_reg_cur, may_logp=_sampled_may_log_prob(pc_reg_cur)
    ).to(dtype=dtype)
    lp_reg_prev_scalar = _gather_sampled_scalar(
        pc_reg_prev, may_logp=_sampled_may_log_prob(pc_reg_prev)
    ).to(dtype=dtype)

    # Blended reg log-probs per flat choice (paper §164 alpha interpolation).
    blended_reg_flat = alpha * lp_reg_cur_scalar_flat + (1.0 - alpha) * lp_reg_prev_scalar_flat
    log_ratio_flat = pc_online.flat_log_probs.to(dtype=dtype) - blended_reg_flat

    # Also build the per-step scalar log_ratio at the sampled action for the
    # "sampled correction" term of the paper's Q estimator.
    if pc_online.flat_logits.numel() > 0:
        sampled_col_flat = pc_online.sampled_col_per_step[pc_online.group_idx]
        is_sampled_flat = pc_online.choice_cols == sampled_col_flat
        per_step_log_ratio_sampled = torch.zeros(t_len, dtype=dtype, device=device)
        per_step_log_ratio_sampled.scatter_add_(
            0,
            pc_online.group_idx,
            torch.where(is_sampled_flat, log_ratio_flat, torch.zeros_like(log_ratio_flat)),
        )
    else:
        is_sampled_flat = torch.zeros(0, dtype=torch.bool, device=device)
        per_step_log_ratio_sampled = torch.zeros(t_len, dtype=dtype, device=device)

    total_cl = v_online.new_zeros(())
    total_pl = v_online.new_zeros(())
    v_hat_means: list[float] = []
    transformed_means: list[float] = []

    for own_idx in (0, 1):
        is_own = perspective == own_idx
        rewards = torch.zeros(t_len, dtype=dtype, device=device)
        if winner_idx >= 0:
            # Terminal reward is game-global from ``own_idx``'s POV, not
            # perspective-sign-flipped: the zero-sum accumulator in
            # two_player_vtrace consumes rewards in the own-player frame.
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

        total_cl = total_cl + critic_loss(
            v_theta=v_online,
            v_hat=v_out.v_hat,
            perspective_is_player_i=is_own,
        )

        # beta-gated NeuRD on the Bernoulli may head (paper §188 applied to
        # the 1-logit Bernoulli form; see :func:`may_neurd_loss`).
        if pc_online.may_is_active.any():
            may_and_own = pc_online.may_is_active & is_own
            total_pl = total_pl + may_neurd_loss(
                may_logits=pc_online.may_logits_per_step.to(dtype=dtype),
                may_selected=pc_online.may_selected_per_step.to(dtype=dtype),
                q_hat=v_out.q_hat,
                own_turn_may_mask=may_and_own,
                beta=config.neurd_beta,
                clip=config.neurd_clip,
                lr=config.learning_rate,
            )

        if pc_online.flat_logits.numel() > 0:
            # Per-action Q from paper §179-180 (own-turn steps only):
            #   Q(a) = -eta * log_ratio[a]
            #        + 1[a=a_t] * (1/mu_t) * (
            #              r_t + eta * log_ratio[a_t]
            #            + (pi/mu)_t * (r̂_{t+1} + v̂_{t+1})
            #            - v(o_t)
            #          )
            #        + v(o_t)
            # The bracketed quantity uses raw (un-transformed) reward r_t,
            # the unclipped own-turn ratio (pi/mu)_t = exp(lp_tgt - lp_mu),
            # and the v-trace bootstrap r̂_{t+1} / v̂_{t+1} returned per
            # step by two_player_vtrace. The leading 1/mu_t is the importance
            # correction that converts the sampled-action contribution from
            # mu's expectation to pi's expectation.
            step_is_own = is_own[pc_online.group_idx]
            v_per_choice = v_tgt_dt[pc_online.group_idx]
            base_q_flat = -config.eta * log_ratio_flat
            inv_mu = (-logp_mu_dev).exp()
            ratio_own_per_step = (lp_tgt_scalar - logp_mu_dev).exp()
            sampled_inner = (
                rewards
                + config.eta * per_step_log_ratio_sampled
                + ratio_own_per_step * (v_out.r_hat_next + v_out.v_hat_next)
                - v_tgt_dt
            )
            sampled_correction_per_step = inv_mu * sampled_inner
            sampled_correction_flat = sampled_correction_per_step[pc_online.group_idx]
            flat_q = (
                base_q_flat
                + torch.where(
                    is_sampled_flat,
                    sampled_correction_flat,
                    torch.zeros_like(sampled_correction_flat),
                )
                + v_per_choice
            )
            flat_q = torch.where(step_is_own, flat_q, torch.zeros_like(flat_q))
            total_pl = total_pl + neurd_loss_per_choice(
                flat_logits=pc_online.flat_logits,
                flat_q=flat_q,
                group_idx=pc_online.group_idx,
                num_groups=t_len,
                beta=config.neurd_beta,
                clip=config.neurd_clip,
                lr=config.learning_rate,
            )

        v_hat_means.append(float(v_out.v_hat.detach().mean()))
        transformed_means.append(float(transformed.detach().mean()))

    return _TrajLossPieces(
        cl=total_cl,
        pl=total_pl,
        v_hat_mean=sum(v_hat_means) / len(v_hat_means),
        transformed_mean=sum(transformed_means) / len(transformed_means),
    )


def rnad_update_trajectory_full_neurd(
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
    """Single-trajectory convenience wrapper for the full-NeuRD variant."""

    pieces = rnad_trajectory_loss_full_neurd(
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
    loss = pieces.cl + pieces.pl
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
        critic_loss=float(pieces.cl.detach()),
        policy_loss=float(pieces.pl.detach()),
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
