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

import torch
from torch import Tensor


@dataclass(frozen=True)
class RNaDConfig:
    """Hyperparameters for the R-NaD trainer.

    Defaults are scaled ~100x down from the DeepNash paper (which targeted
    768 TPU nodes over 7.21M steps) to be a sensible single-GPU starting
    point for magic-ai. All values are exposed via CLI flags.
    """

    eta: float = 0.2
    """Regularization strength for the reward transform (paper: 0.2)."""

    delta_m: int = 25_000
    """Gradient steps per outer iteration (paper: 10k-100k)."""

    num_outer_iterations: int = 20
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

    target_ema_gamma: float = 0.001
    """Polyak averaging rate for the target network (paper: 1e-3)."""

    finetune_eps: float = 0.03
    """Probability threshold for fine-tune / test-time discretization."""

    finetune_n_disc: int = 16
    """Number of probability quanta for fine-tune / test-time discretization."""


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

    # Sentinels for the "past the end" state: zero future reward, and
    # v_next/q_next carry zero (no bootstrap past terminal).
    r_acc = torch.zeros((), dtype=dtype, device=device)
    v_next = torch.zeros((), dtype=dtype, device=device)
    xi_next = torch.ones((), dtype=dtype, device=device)
    v_next_own = torch.zeros((), dtype=dtype, device=device)

    for t in range(T - 1, -1, -1):
        rho_t = min(rho_bar, float(ratio[t].item()) * float(xi_next.item()))
        c_t = min(c_bar, float(ratio[t].item()) * float(xi_next.item()))
        if bool(is_own[t].item()):
            # Own-turn step: v-trace advantage bootstraps onto values[t].
            delta = rho_t * (rewards[t] + r_acc + v_next_own - values[t])
            v_hat_t = values[t] + delta + c_t * (v_next - v_next_own)
            v_hat[t] = v_hat_t
            # Q for the sampled action: same form as v_hat with the
            # policy-prior absorbed (see paper §179-180). For the scalar
            # variant we use the same estimate; the per-head NeuRD loss
            # combines this with policy logits to produce per-action Q.
            q_hat[t] = values[t] + rho_t * (rewards[t] + r_acc + v_next_own - values[t])
            # reset accumulators: the next (earlier) step's "future reward"
            # starts empty, and the bootstrap target becomes values[t].
            r_acc = torch.zeros((), dtype=dtype, device=device)
            v_next = v_hat_t
            v_next_own = values[t]
            xi_next = torch.ones((), dtype=dtype, device=device)
        else:
            # Opponent-turn step: accumulate reward with importance weight.
            r_acc = rewards[t] + ratio[t] * r_acc
            xi_next = ratio[t] * xi_next
            # v_next and v_next_own pass through unchanged.
            v_hat[t] = v_next
            q_hat[t] = v_next

    return VTraceOutput(v_hat=v_hat, q_hat=q_hat)


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
        in_range = (logits >= -beta) & (logits <= beta)
        active = legal_mask & in_range
    # Loss = -sum_{t,a in active} logits * Q. Negative because Adam minimizes
    # and we want d logits proportional to +Q * grad_flag_in_range.
    per_entry = -logits * q_clipped * active.to(dtype=logits.dtype)
    T = logits.shape[0]
    return per_entry.sum() / max(T, 1)


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
