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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from magic_ai.lstm_recompute import PerPlayerLstmLayout, build_per_player_lstm_layout
from magic_ai.model_state import is_actor_runtime_state_key


@dataclass(frozen=True)
class RNaDConfig:
    """Hyperparameters for the R-NaD trainer.

    Defaults are scaled ~100x down from the DeepNash paper (which targeted
    768 TPU nodes over 7.21M steps) to be a sensible single-GPU starting
    point for magic-ai. All values are exposed via CLI flags.
    """

    eta: float = 0.2
    """Regularization strength for the reward transform (paper: 0.2)."""

    reg_log_ratio_clip: float = float("inf")
    """Optional symmetric clip on ``log π_θ - log π_reg`` inside
    :func:`transform_rewards`. **Off by default and not in the DeepNash
    paper.**

    The reward shaping ``r'_t = r_t + sign · η · log_ratio`` lives on the
    paper's assumption that the reg policy tracks online tightly (so
    ``log_ratio`` is naturally small). With our default
    ``target_ema_gamma · delta_m = 5`` the reg lags much further behind
    online during the early-training transient, the per-step shaping can
    reach ``η · 50``, and ``v_hat`` accumulates over the trajectory until
    the L1 critic loss looks alarming (the gradient is still bounded —
    L1 — so training is stable; the loss report just becomes a
    misleading scalar). Set to a finite value (e.g. 5–10) to bound the
    transient, but the principled fix is to bump
    ``target_ema_gamma · delta_m`` so the reg tracks online tightly
    enough to keep the log-ratio small without an external clip."""

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

    diagnostic_v_target_reg_share_every: int = 10
    """Compute the ``v_target_reg_share`` diagnostic every K gradient steps
    (0 disables it entirely; default every 10 steps). The diagnostic requires a *second*
    full ``two_player_vtrace`` pass per perspective per episode (fed only
    the terminal ±1 reward), which doubles v-trace work in the R-NaD loss.
    Trainers that want this signal should enable it at a low cadence (e.g.
    50) so the cost is amortized."""

    diagnostic_policy_drift_every: int = 10
    """Compute sampled-log-ratio and target-lag diagnostics every K gradient
    steps (0 disables them; default every 10 steps). These diagnostics are useful while
    tuning R-NaD stability, but they require reductions that eventually sync
    to the host, so production throughput runs keep them out of the hot path."""

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
    log_ratio_clip: float = float("inf"),
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
    if math.isfinite(log_ratio_clip):
        log_ratio = log_ratio.clamp(min=-log_ratio_clip, max=log_ratio_clip)
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
    # Stay in log-space; clip-then-exp at every use (IMPALA-style). The
    # only quantity we ever exponentiate is the IS log-ratio, capped so
    # the resulting weight is bounded by rho_bar / c_bar. Forming
    # ``ratio = exp(log_ratio)`` and then ``log_ratio = ratio.log()``
    # was poisoning the recursion: when the gap underflowed ``ratio`` to
    # 0, ``log(0) = -inf`` propagated through the opp-segment cumsum.
    log_rho = logp_theta - logp_mu
    log_rho_bar = math.log(rho_bar) if rho_bar > 0 else float("-inf")
    log_c_bar = math.log(c_bar) if c_bar > 0 else float("-inf")

    v_hat = torch.zeros(T, dtype=dtype, device=device)
    q_hat = torch.zeros(T, dtype=dtype, device=device)
    r_hat_next_step = torch.zeros(T, dtype=dtype, device=device)
    v_hat_next_step = torch.zeros(T, dtype=dtype, device=device)

    own_idx = is_own.nonzero(as_tuple=False).squeeze(-1)  # (K,) sorted ascending
    K = int(own_idx.numel())
    if K == 0:
        return VTraceOutput(
            v_hat=v_hat,
            q_hat=q_hat,
            r_hat_next=r_hat_next_step,
            v_hat_next=v_hat_next_step,
        )

    # Per-step segmentation.
    cum_own = is_own.to(dtype=torch.long).cumsum(0)
    seg = cum_own - 1
    safe_seg = seg.clamp_min(0)

    opp_log_rho = torch.where(is_own, torch.zeros_like(log_rho), log_rho)
    cum_log_rho = opp_log_rho.cumsum(0)
    cum_log_rho_at_own = cum_log_rho[own_idx]
    base_cum_log_rho = cum_log_rho_at_own[safe_seg]
    # Cumulative IS over the opp-segment, capped at ``log(rho_bar)`` in
    # log space (mirrors IMPALA's clipped rho) so the resulting weight
    # is finite and bounded by rho_bar.
    seg_log_rho = cum_log_rho - opp_log_rho - base_cum_log_rho
    multiplier = torch.exp(seg_log_rho.clamp(max=log_rho_bar)).to(dtype=dtype)
    opp_contrib = torch.where(
        is_own | (seg < 0),
        torch.zeros_like(rewards),
        rewards * multiplier,
    )
    r_acc = torch.zeros(K, dtype=dtype, device=device)
    r_acc.scatter_add_(0, safe_seg, opp_contrib)

    if K > 1:
        right_boundary = torch.cat(
            [own_idx[1:] - 1, torch.tensor([T - 1], dtype=torch.long, device=device)]
        )
    else:
        right_boundary = torch.tensor([T - 1], dtype=torch.long, device=device)
    xi_log = cum_log_rho[right_boundary] - cum_log_rho_at_own

    log_rho_own = log_rho[own_idx]
    log_rho_arg = log_rho_own + xi_log
    rho = torch.exp(log_rho_arg.clamp(max=log_rho_bar)).to(dtype=dtype)
    c = torch.exp(log_rho_arg.clamp(max=log_c_bar)).to(dtype=dtype)
    ratio_own = torch.exp(log_rho_own.clamp(max=log_rho_bar)).to(dtype=dtype)

    values_own = values[own_idx]
    rewards_own = rewards[own_idx]
    v_next_own_vec = torch.cat([values_own[1:], values_own.new_zeros(1)])
    pre_c = values_own + rho * (rewards_own + ratio_own * r_acc + v_next_own_vec - values_own)

    # Iterative reverse scan ``v[k] = A[k] + B[k] * v[k+1]`` — numerically
    # stable for any c in (0, c_bar]. The previous closed-form
    # ``scan_coeffs = cumprod(c)`` collapsed to 0 within tens of steps
    # under small c, then ``A / scan_coeffs = inf`` and ``inf * 0 = NaN``
    # contaminated every output. K is per-trajectory and bounded, so the
    # Python loop here is cheap.
    A = pre_c - c * v_next_own_vec
    B = c
    if K == 1:
        v_hat_own = A.detach().clone()
    else:
        A_d = A.detach()
        B_d = B.detach()
        v_hat_own = torch.zeros_like(A_d)
        v_next = A_d.new_zeros(())
        for k in range(K - 1, -1, -1):
            v_next = A_d[k] + B_d[k] * v_next
            v_hat_own[k] = v_next

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


def _two_player_vtrace_batched(
    *,
    rewards: Tensor,
    values: Tensor,
    logp_theta: Tensor,
    logp_mu: Tensor,
    perspective_is_player_i: Tensor,
    ep_offsets: Tensor,
    own_idx: Tensor | None = None,
    max_ep_len: int | None = None,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> VTraceOutput:
    """Batched :func:`two_player_vtrace` over a flat concatenation of episodes.

    All 1-D tensors have length ``T_total`` and are the concatenation in
    episode order of the per-episode tensors that :func:`two_player_vtrace`
    would have received. ``ep_offsets`` is a length-``n_eps + 1`` long tensor
    on the same device, with ``ep_offsets[0] == 0`` and
    ``ep_offsets[-1] == T_total``.

    The math is identical to running ``two_player_vtrace`` once per episode
    and concatenating outputs; the cumsum / scatter / index_copy operations
    are episode-aware (segments don't bleed across boundaries) but happen
    once over the whole batch instead of once per episode. The host-side
    K-step recurrence likewise runs once over the concatenated own-turn
    list, resetting at episode boundaries.
    """

    if rewards.ndim != 1:
        raise ValueError("expects 1-D flat tensors")
    T = rewards.shape[0]
    if T == 0:
        raise ValueError("cannot run v-trace on an empty batch")
    if ep_offsets.ndim != 1 or ep_offsets.numel() < 2:
        raise ValueError("ep_offsets must have at least two entries")
    n_eps = int(ep_offsets.numel()) - 1

    dtype = rewards.dtype
    device = rewards.device
    is_own = perspective_is_player_i.to(dtype=torch.bool)
    # Stay in log-space and clip *before* exponentiating, IMPALA-style.
    # The per-step IS log-ratio is the only quantity ever exponentiated,
    # and we always cap the exponent so the resulting weight is bounded
    # by rho_bar / c_bar. Forming ``ratio = exp(log_ratio)`` and then
    # ``log_ratio = ratio.log()`` was poisoning everything: when the gap
    # underflows ``ratio`` to 0, ``log(0) = -inf`` propagates into the
    # opp-segment cumsum and the recurrence collapses to NaN.
    log_rho = logp_theta - logp_mu
    log_rho_bar = math.log(rho_bar) if rho_bar > 0 else float("-inf")
    log_c_bar = math.log(c_bar) if c_bar > 0 else float("-inf")

    v_hat = torch.zeros(T, dtype=dtype, device=device)
    q_hat = torch.zeros(T, dtype=dtype, device=device)
    r_hat_next_step = torch.zeros(T, dtype=dtype, device=device)
    v_hat_next_step = torch.zeros(T, dtype=dtype, device=device)

    if own_idx is None:
        own_idx = is_own.nonzero(as_tuple=False).squeeze(-1)
    else:
        own_idx = own_idx.to(device=device, dtype=torch.long)
    K = int(own_idx.numel())
    if K == 0:
        return VTraceOutput(
            v_hat=v_hat,
            q_hat=q_hat,
            r_hat_next=r_hat_next_step,
            v_hat_next=v_hat_next_step,
        )

    ep_lens = ep_offsets[1:] - ep_offsets[:-1]
    ep_idx_step = torch.repeat_interleave(
        torch.arange(n_eps, dtype=torch.long, device=device),
        ep_lens,
    )
    own_ep_idx = ep_idx_step[own_idx]

    # Per-step ``seg`` is the global own-turn index of the most recent own-turn
    # step at or before t. ``valid_seg`` requires that own-turn lie in the same
    # episode as t; otherwise t is before its episode's first own-turn and
    # contributes 0 to r_acc (the original per-ep form had ``seg < 0`` here).
    cum_own = is_own.to(dtype=torch.long).cumsum(0)
    seg = cum_own - 1
    safe_seg = seg.clamp_min(0)
    own_ep_at_safe = own_ep_idx[safe_seg]
    valid_seg = (seg >= 0) & (own_ep_at_safe == ep_idx_step)

    opp_log_rho = torch.where(is_own, torch.zeros_like(log_rho), log_rho)
    cum_log_rho = opp_log_rho.cumsum(0)
    cum_log_rho_at_own = cum_log_rho[own_idx]
    base_cum_log_rho = cum_log_rho_at_own[safe_seg]
    # The opp-segment multiplier is the cumulative IS ratio over the
    # opp-turn steps that have occurred since the most recent own-turn.
    # Cap *in log space* at ``log(rho_bar)`` (mirrors the IMPALA clipped
    # ``rho``) so the resulting weight is always in (0, rho_bar] —
    # finite, bounded, and ``rewards * multiplier`` cannot blow up.
    seg_log_rho = cum_log_rho - opp_log_rho - base_cum_log_rho
    multiplier = torch.exp(seg_log_rho.clamp(max=log_rho_bar)).to(dtype=dtype)
    opp_contrib = torch.where(
        is_own | (~valid_seg),
        torch.zeros_like(rewards),
        rewards * multiplier,
    )
    r_acc = torch.zeros(K, dtype=dtype, device=device)
    r_acc.scatter_add_(0, safe_seg, opp_contrib)

    # Per-own-turn ``xi_prod``: right boundary is the next own-turn in the same
    # episode minus one (or the last step of the episode if no further own-turn
    # in the episode).
    if K > 1:
        same_ep_next = own_ep_idx[1:] == own_ep_idx[:-1]
        rb_same = own_idx[1:] - 1
        rb_ep = ep_offsets[own_ep_idx[:-1] + 1] - 1
        rb_pre = torch.where(same_ep_next, rb_same, rb_ep)
        rb_last = (ep_offsets[own_ep_idx[-1] + 1] - 1).reshape(1)
        right_boundary = torch.cat([rb_pre, rb_last])
        same_ep_next_full = torch.cat(
            [same_ep_next, torch.zeros(1, dtype=torch.bool, device=device)]
        )
    else:
        right_boundary = (ep_offsets[own_ep_idx[0] + 1] - 1).reshape(1)
        same_ep_next_full = torch.zeros(1, dtype=torch.bool, device=device)
    xi_log = cum_log_rho[right_boundary] - cum_log_rho_at_own

    log_rho_own = log_rho[own_idx]
    # ``rho_arg = ratio_own * xi_prod`` becomes ``log_rho_own + xi_log``.
    # ``rho = min(rho_arg, rho_bar)`` and ``c = min(rho_arg, c_bar)`` are
    # then computed entirely in log space, exp'd at the cap, so neither
    # can overflow regardless of how mismatched the policies are.
    log_rho_arg = log_rho_own + xi_log
    rho = torch.exp(log_rho_arg.clamp(max=log_rho_bar)).to(dtype=dtype)
    c = torch.exp(log_rho_arg.clamp(max=log_c_bar)).to(dtype=dtype)
    # ``ratio_own`` only feeds ``ratio_own * r_acc`` below; cap it at
    # rho_bar too so an off-the-charts target/behavior gap can't reach
    # the recurrence as a multiplicative inf.
    ratio_own = torch.exp(log_rho_own.clamp(max=log_rho_bar)).to(dtype=dtype)

    values_own = values[own_idx]
    rewards_own = rewards[own_idx]
    next_vals = torch.cat([values_own[1:], values_own.new_zeros(1)])
    v_next_own_vec = torch.where(same_ep_next_full, next_vals, values_own.new_zeros(K))

    pre_c = values_own + rho * (rewards_own + ratio_own * r_acc + v_next_own_vec - values_own)
    A = pre_c - c * v_next_own_vec
    B = c

    # Reverse scan ``v[k] = A[k] + B[k] * v[k+1]`` per episode. The
    # previous closed form (``scan_coeffs * cumsum(A / scan_coeffs)``)
    # underflowed: ``scan_coeffs = cumprod(c)`` collapses to 0 within
    # 30-40 steps when c is small, then ``A / scan_coeffs = inf`` and
    # ``inf * 0 = NaN`` poisons every output. Iterate instead — with
    # max_own_per_ep ≈ 30-100 it's just that many fused-elementwise
    # tensor ops over (n_eps,), no per-step host syncs.
    K_per_ep = torch.zeros(n_eps, dtype=torch.long, device=device)
    K_per_ep.scatter_add_(0, own_ep_idx, torch.ones_like(own_ep_idx))
    if max_ep_len is None:
        max_own_per_ep = int(K_per_ep.max().item())
    else:
        max_own_per_ep = int(max_ep_len)
    cumlen_before_ep = torch.cat([K_per_ep.new_zeros(1), K_per_ep.cumsum(0)[:-1]])
    own_pos_in_ep = torch.arange(K, device=device) - cumlen_before_ep[own_ep_idx]

    A_pad = torch.zeros(n_eps, max_own_per_ep, dtype=dtype, device=device)
    B_pad = torch.zeros(n_eps, max_own_per_ep, dtype=dtype, device=device)
    A_pad[own_ep_idx, own_pos_in_ep] = A
    B_pad[own_ep_idx, own_pos_in_ep] = B

    v_pad = torch.zeros(n_eps, max_own_per_ep, dtype=dtype, device=device)
    v_next = torch.zeros(n_eps, dtype=dtype, device=device)
    for k in range(max_own_per_ep - 1, -1, -1):
        v_next = A_pad[:, k] + B_pad[:, k] * v_next
        v_pad[:, k] = v_next

    v_hat_own = v_pad[own_ep_idx, own_pos_in_ep]

    v_hat.index_copy_(0, own_idx, v_hat_own)
    q_hat.index_copy_(0, own_idx, pre_c)

    v_hat_own_next_full = torch.cat([v_hat_own[1:], v_hat_own.new_zeros(1)])
    v_hat_next_own = torch.where(same_ep_next_full, v_hat_own_next_full, v_hat_own.new_zeros(K))
    r_hat_next_step.index_copy_(0, own_idx, r_acc)
    v_hat_next_step.index_copy_(0, own_idx, v_hat_next_own)

    next_k = seg + 1
    safe_next_k = next_k.clamp(min=0, max=max(K - 1, 0))
    next_ep = own_ep_idx[safe_next_k]
    valid_next = (next_k >= 0) & (next_k < K) & (next_ep == ep_idx_step)
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
) -> tuple[Tensor, Tensor, Tensor]:
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
    # ``n_clipped`` / ``n_active`` were previously ``.item()``-collapsed here,
    # which forced a GPU sync per call (×4 policies / ×3 chunks per update on
    # the rnad hot path). Keeping them as 0-d Long tensors lets the trainer
    # accumulate across chunks and sync once at the end of ``run_rnad_update``.
    if flat_logits.numel() > 0:
        n_clipped = (q_pre.abs() > clip).sum().to(dtype=torch.long)
    else:
        n_clipped = flat_logits.new_zeros((), dtype=torch.long)
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
    if flat_logits.numel() > 0:
        n_active = active_f.sum().to(dtype=torch.long)
    else:
        n_active = flat_logits.new_zeros((), dtype=torch.long)
    return sum_loss, n_active, n_clipped


# ---------------------------------------------------------------------------
# Critic loss
# ---------------------------------------------------------------------------


def critic_loss(
    *,
    v_theta: Tensor,
    v_hat: Tensor,
    perspective_is_player_i: Tensor,
) -> tuple[Tensor, Tensor]:
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
    count = mask.sum().to(dtype=torch.long)
    # Always run the masked sum: ``mask.any()`` would force a sync, and
    # ``diff[all-False].sum()`` already returns a 0-tensor of the right dtype.
    diff = (v_theta - v_hat.detach()).abs()
    return (diff * mask.to(dtype=diff.dtype)).sum(), count


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
) -> tuple[Tensor, Tensor]:
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
    count = mask.sum().to(dtype=torch.long)
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
    # Always run the masked sum; the empty-mask case yields a 0-tensor without
    # forcing a sync via ``mask.any()``.
    sum_loss = per_step[mask].sum()
    return sum_loss, count


# ---------------------------------------------------------------------------
# Polyak target network + reg snapshot persistence
# ---------------------------------------------------------------------------


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
        if is_actor_runtime_state_key(name):
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
    torch.save(
        {
            "state_dict": {
                name: tensor
                for name, tensor in policy.state_dict().items()
                if not is_actor_runtime_state_key(name)
            }
        },
        path,
    )


def load_reg_snapshot_into(policy: nn.Module, path: str | Path) -> None:
    """Load a reg-policy snapshot into ``policy`` in-place, freezing grads."""

    payload: dict[str, Any] = torch.load(
        str(path),
        map_location="cpu",
        weights_only=False,
    )
    policy.load_state_dict(
        {
            name: tensor
            for name, tensor in payload["state_dict"].items()
            if not is_actor_runtime_state_key(name)
        },
        strict=False,
    )
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
    entropy: float = 0.0
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

    policy_drift_diagnostics_computed: bool = False
    """Whether sampled-log-ratio / target-lag diagnostics were computed for
    this update rather than skipped by their cadence gate."""

    v_target_reg_share_computed: bool = False
    """Whether ``v_target_reg_share`` was computed for this update rather than
    skipped by its cadence gate."""


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
    # Counters were ints (forcing per-call ``.item()`` syncs in the loss
    # helpers); they're now 0-d ``torch.long`` tensors so the trainer can
    # accumulate across chunks and force a single sync at the end of the
    # update. ``int + tensor`` and ``tensor == int`` both work transparently
    # so verify-mode and ``q_clip_fraction`` callers need no changes beyond
    # the final ``float(...)``-coerce in ``run_rnad_update``.
    cl_count: Tensor
    pl_sum: Tensor
    pl_count: Tensor
    n_q_clipped: Tensor
    v_hat_sum: Tensor
    transformed_sum: Tensor
    transformed_count: int
    entropy_sum: Tensor
    entropy_count: Tensor
    # Diagnostics (issue: within-inner-loop policy degradation). All are sums
    # / maxes paired with their counts so the trainer can aggregate across
    # episodes without weighting bias.
    sampled_log_ratio_sum: Tensor | None = None
    sampled_log_ratio_absmax: Tensor | None = None
    sampled_log_ratio_count: int = 0
    is_bias_up_sum: Tensor | None = None
    is_bias_up_count: Tensor | None = None
    is_bias_down_sum: Tensor | None = None
    is_bias_down_count: Tensor | None = None
    v_target_reg_share_sum: Tensor | None = None
    v_target_reg_share_count: Tensor | None = None


@dataclass(frozen=True)
class _RNaDBatchLayout:
    flat_rows: tuple[int, ...]
    ep_lens: tuple[int, ...]
    ep_offsets: Tensor
    perspective: Tensor
    terminal_reward_p0: Tensor
    zero_sum: Tensor
    logp_mu: Tensor
    own_idx_p0: Tensor
    own_idx_p1: Tensor
    lstm_layout: PerPlayerLstmLayout


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


def _per_group_behavior_log_prob(
    pc_online: Any,
    *,
    n_groups: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Rollout-time sampled log-prob per decision group."""

    stored = pc_online.behavior_action_log_prob_per_decision_group.to(device=device, dtype=dtype)
    if tuple(stored.shape) != (n_groups,):
        raise ValueError(
            "behavior_action_log_prob_per_decision_group must have shape "
            f"({n_groups},), got {tuple(stored.shape)}"
        )
    return stored


def rnad_trajectory_loss(
    *,
    online: Any,
    target: Any,
    reg_cur: Any,
    reg_prev: Any,
    replay_rows: list[int],
    perspective_player_idx: Sequence[int],
    terminal_reward_p0: float,
    zero_sum: bool,
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

    **Behavior policy storage.** Replay rows store the rollout-time sampled
    per-group log-prob, and the per-action Q estimator uses that stored value
    as ``mu_k``. The joint v-trace IS ratio still uses the stored joint
    ``logp_mu`` from rollout time. See ``docs/rnad_deviations.md``.

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

    if (
        online_h_out is not None
        or target_h_out is not None
        or reg_cur_h_out is not None
        or reg_prev_h_out is not None
    ):
        raise ValueError(
            "precomputed single-episode h_out overrides are no longer supported; "
            "use rnad_batched_trajectory_loss for fused recurrent recompute"
        )

    return rnad_batched_trajectory_loss(
        online=online,
        target=target,
        reg_cur=reg_cur,
        reg_prev=reg_prev,
        episodes_replay_rows=[replay_rows],
        episodes_perspective=[perspective_player_idx],
        episodes_terminal_reward_p0=[terminal_reward_p0],
        episodes_zero_sum=[zero_sum],
        episodes_logp_mu=[logp_mu],
        config=config,
        alpha=alpha,
        compute_v_target_reg_share=compute_v_target_reg_share,
    )[0]


def _trajectory_loss_from_forwards(
    *,
    lp_online: Tensor,
    entropy_online: Tensor,
    v_online: Tensor,
    pc_online: Any,
    lp_tgt_scalar: Tensor,
    v_tgt: Tensor,
    pc_tgt: Any,
    pc_reg_cur: Any,
    pc_reg_prev: Any,
    perspective: Tensor,
    terminal_reward_p0: float,
    zero_sum: bool,
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
        per_group_lp_mu = _per_group_behavior_log_prob(
            pc_online,
            n_groups=n_groups,
            dtype=dtype,
            device=device,
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
    cl_count_total = torch.zeros((), dtype=torch.long, device=device)
    pl_count_total = torch.zeros((), dtype=torch.long, device=device)
    n_q_clipped_total = torch.zeros((), dtype=torch.long, device=device)
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
        # Sync-free up/down sums: mirror the pattern used by
        # :func:`_batched_trajectory_loss_from_forwards` so the diagnostics
        # block doesn't host-sync on ``.item()`` per training step.
        up_count_t = online_moved_up.to(torch.long).sum()
        down_count_t = (~online_moved_up).to(torch.long).sum()
        zeros_b = torch.zeros_like(is_bias_step)
        is_bias_up_sum_t = torch.where(online_moved_up, is_bias_step, zeros_b).sum()
        is_bias_down_sum_t = torch.where(~online_moved_up, is_bias_step, zeros_b).sum()

    v_target_reg_share_sum_t = torch.zeros((), dtype=dtype, device=device)
    v_target_reg_share_count_t = torch.zeros((), dtype=torch.long, device=device)

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

    # May-head setup runs unconditionally — ``may_neurd_loss`` masks padded
    # positions out by ``may_and_own``, so when no may steps are active the
    # contribution is exactly zero. Always running avoids a per-call
    # ``.any().item()`` sync on ``may_is_active``. Mirrors the same
    # pattern in :func:`_batched_trajectory_loss_from_forwards`.
    may_logits_step = pc_online.may_logits_per_step.to(dtype=dtype)
    log_pi_accept = torch.nn.functional.logsigmoid(may_logits_step)
    log_pi_decline = torch.nn.functional.logsigmoid(-may_logits_step)
    with torch.no_grad():
        may_reg_cur_logits = pc_reg_cur.may_logits_per_step.to(dtype=dtype)
        may_reg_prev_logits = pc_reg_prev.may_logits_per_step.to(dtype=dtype)
        may_lp_reg_blend_accept = alpha * torch.nn.functional.logsigmoid(may_reg_cur_logits) + (
            1.0 - alpha
        ) * torch.nn.functional.logsigmoid(may_reg_prev_logits)
        may_lp_reg_blend_decline = alpha * torch.nn.functional.logsigmoid(-may_reg_cur_logits) + (
            1.0 - alpha
        ) * torch.nn.functional.logsigmoid(-may_reg_prev_logits)
    log_ratio_accept = log_pi_accept.detach() - may_lp_reg_blend_accept
    log_ratio_decline = log_pi_decline.detach() - may_lp_reg_blend_decline
    sampled_is_accept = pc_online.may_selected_per_step > 0.5
    with torch.no_grad():
        tgt_p_accept = torch.sigmoid(pc_tgt.may_logits_per_step.to(dtype=dtype))
        may_mu_sampled = torch.where(sampled_is_accept, tgt_p_accept, 1.0 - tgt_p_accept)
        may_inv_mu = (1.0 / may_mu_sampled.clamp_min(1e-30)).clamp(max=config.q_corr_rho_bar)
    may_sampled_log_ratio = torch.where(sampled_is_accept, log_ratio_accept, log_ratio_decline)

    # Per-perspective terminal rewards. Zero-sum (engine win/loss,
    # life-tiebreak timeout) flips p1's sign; non-zero-sum (engine draw)
    # gives both players the same value. Mirrors PPO's gae_returns.
    r_p0 = float(terminal_reward_p0)
    r_p1 = -r_p0 if zero_sum else r_p0
    terminal_per_own = (r_p0, r_p1)

    for own_idx in (0, 1):
        is_own = perspective == own_idx
        rewards = torch.zeros(t_len, dtype=dtype, device=device)
        rewards[-1] = terminal_per_own[own_idx]

        transformed = transform_rewards(
            rewards,
            logp_theta=lp_online_dt,
            logp_reg_cur=lp_reg_cur_scalar,
            logp_reg_prev=lp_reg_prev_scalar,
            alpha=alpha,
            eta=config.eta,
            perspective_is_player_i=is_own,
            log_ratio_clip=config.reg_log_ratio_clip,
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
                    v_target_reg_share_count_t += is_own.sum().to(dtype=torch.long)

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
        # hoisted above; only rewards / v_out / is_own change here. Runs
        # unconditionally — the ``may_and_own`` mask zeros out padded
        # positions when no may steps are active.
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
            + torch.where(sampled_is_accept, may_q_sampled_correction, torch.zeros_like(rewards))
            + v_tgt_dt
        )
        q_decline = (
            -config.eta * log_ratio_decline
            + torch.where(~sampled_is_accept, may_q_sampled_correction, torch.zeros_like(rewards))
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
        v_hat_sum=v_online.new_tensor(sum(v_hat_means)),
        transformed_sum=v_online.new_tensor(sum(transformed_means)),
        transformed_count=max(len(v_hat_means), 1),
        entropy_sum=entropy_online.detach().to(dtype=dtype).sum(),
        entropy_count=torch.tensor(t_len, dtype=torch.long, device=device),
        sampled_log_ratio_sum=v_online.new_tensor(sampled_log_ratio_sum_t),
        sampled_log_ratio_absmax=v_online.new_tensor(sampled_log_ratio_absmax_t),
        sampled_log_ratio_count=sampled_log_ratio_count_t,
        is_bias_up_sum=is_bias_up_sum_t.to(dtype=dtype),
        is_bias_up_count=up_count_t,
        is_bias_down_sum=is_bias_down_sum_t.to(dtype=dtype),
        is_bias_down_count=down_count_t,
        v_target_reg_share_sum=v_target_reg_share_sum_t,
        v_target_reg_share_count=v_target_reg_share_count_t,
    )


def _batched_trajectory_loss_from_forwards(
    *,
    lp_online: Tensor,
    entropy_online: Tensor,
    v_online: Tensor,
    pc_online: Any,
    lp_tgt_scalar: Tensor,
    v_tgt: Tensor,
    pc_tgt: Any,
    pc_reg_cur: Any,
    pc_reg_prev: Any,
    perspective: Tensor,
    terminal_reward_p0: Tensor,
    zero_sum: Tensor,
    logp_mu: Tensor,
    ep_offsets: Tensor,
    own_idx_p0: Tensor | None = None,
    own_idx_p1: Tensor | None = None,
    max_ep_len: int | None = None,
    config: RNaDConfig,
    alpha: float,
    compute_v_target_reg_share: bool = False,
    compute_policy_drift_diagnostics: bool = False,
) -> _TrajLossPieces:
    """Vectorized R-NaD loss assembly across a flat concatenation of episodes.

    Equivalent (up to floating-point reduction order and the
    ``v_hat_mean`` / ``transformed_mean`` aggregation flavor — see notes
    below) to summing per-episode :func:`_trajectory_loss_from_forwards`
    over the same episodes. The per-episode Python loop is replaced with
    flat-batch tensor ops; the per-perspective vtrace pass uses
    :func:`_two_player_vtrace_batched`.

    ``perspective`` is a ``(T_total,)`` long tensor with values in ``{0, 1}``
    (the per-step current-turn player). ``terminal_reward_p0`` is
    ``(n_eps,)`` float (terminal reward in p0's perspective per episode);
    ``zero_sum`` is ``(n_eps,)`` bool — ``True`` means p1's terminal is
    ``-terminal_reward_p0`` (engine win/loss, life-tiebreak timeout),
    ``False`` means p1 sees the same value (engine-draw symmetric absorbing
    state). Mirrors ``magic_ai.returns.gae_returns_batched`` so trainers handle
    win/loss/draw/timeout identically. ``logp_mu`` is ``(T_total,)``
    (concatenation of episodes' rollout-time log-probs). ``ep_offsets`` is
    ``(n_eps + 1,)`` long with ``ep_offsets[0] == 0``,
    ``ep_offsets[-1] == T_total``.

    Diagnostic mean fields (``v_hat_mean``, ``transformed_mean``) are
    step-weighted across the flat batch (rather than the original mean of
    per-episode means). All other fields aggregate identically.
    """
    T_total = int(lp_online.shape[0])
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

    blended_reg_flat = alpha * lp_reg_cur_scalar_flat + (1.0 - alpha) * lp_reg_prev_scalar_flat
    log_ratio_flat = pc_online.flat_log_probs.to(dtype=dtype) - blended_reg_flat

    has_groups = pc_online.flat_logits.numel() > 0
    if has_groups:
        is_sampled_flat = pc_online.is_sampled_flat
        n_groups = int(pc_online.step_for_decision_group.numel())
        per_group_log_ratio_sampled = torch.zeros(n_groups, dtype=dtype, device=device)
        per_group_log_ratio_sampled.scatter_add_(
            0,
            pc_online.decision_group_id_flat,
            torch.where(is_sampled_flat, log_ratio_flat, torch.zeros_like(log_ratio_flat)),
        )
        per_group_lp_mu = _per_group_behavior_log_prob(
            pc_online,
            n_groups=n_groups,
            dtype=dtype,
            device=device,
        )
        inv_mu_per_group_pre = (-per_group_lp_mu).exp()
        inv_mu_per_group = inv_mu_per_group_pre.clamp(max=config.q_corr_rho_bar)
    else:
        is_sampled_flat = torch.zeros(0, dtype=torch.bool, device=device)
        per_group_log_ratio_sampled = torch.zeros(0, dtype=dtype, device=device)
        inv_mu_per_group = torch.zeros(0, dtype=dtype, device=device)

    sampled_log_ratio_sum_t: Tensor | None = None
    sampled_log_ratio_absmax_t: Tensor | None = None
    up_count_t: Tensor | None = None
    down_count_t: Tensor | None = None
    is_bias_up_sum_t: Tensor | None = None
    is_bias_down_sum_t: Tensor | None = None
    if compute_policy_drift_diagnostics:
        # Keep diagnostics as device tensors; the trainer batches their host
        # conversion with the other once-per-update scalar stats.
        blended_reg_scalar = alpha * lp_reg_cur_scalar + (1.0 - alpha) * lp_reg_prev_scalar
        sampled_log_ratio_step = (lp_online_dt - blended_reg_scalar).detach()
        sampled_log_ratio_sum_t = sampled_log_ratio_step.sum()
        sampled_log_ratio_absmax_t = (
            sampled_log_ratio_step.abs().max()
            if T_total > 0
            else torch.zeros((), dtype=dtype, device=device)
        )
        is_bias_step = (lp_tgt_scalar - logp_mu_dev).detach()
        online_moved_up = (lp_online_dt > logp_mu_dev).detach()
        up_count_t = online_moved_up.to(torch.long).sum()
        down_count_t = (~online_moved_up).to(torch.long).sum()
        is_bias_up_sum_t = torch.where(
            online_moved_up, is_bias_step, torch.zeros_like(is_bias_step)
        ).sum()
        is_bias_down_sum_t = torch.where(
            ~online_moved_up, is_bias_step, torch.zeros_like(is_bias_step)
        ).sum()

    # Terminal rewards: scatter the resolved p0-perspective terminal at each
    # episode's last step. Zero-sum episodes (engine win/loss, life-tiebreak
    # timeout) flip the sign for p1; non-zero-sum (engine draw) keeps it the
    # same — the symmetric absorbing-state branch lets the discounted draw
    # penalty reach every step under both perspectives. See
    # ``magic_ai.returns.gae_returns_batched`` for the matching PPO logic.
    last_idx = ep_offsets[1:] - 1
    tr_p0 = terminal_reward_p0.to(device=device, dtype=dtype)
    zs_dev = zero_sum.to(device=device, dtype=torch.bool)
    tr_p1 = torch.where(zs_dev, -tr_p0, tr_p0)
    rewards_p0 = torch.zeros(T_total, dtype=dtype, device=device)
    rewards_p1 = torch.zeros(T_total, dtype=dtype, device=device)
    rewards_p0.scatter_(0, last_idx, tr_p0)
    rewards_p1.scatter_(0, last_idx, tr_p1)

    is_own_p0 = perspective == 0
    is_own_p1 = perspective == 1

    ratio_own_per_step = (lp_tgt_scalar - logp_mu_dev).exp()

    if has_groups:
        per_step_log_ratio_sampled = torch.zeros(T_total, dtype=dtype, device=device)
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

    # May-head setup runs unconditionally — ``may_neurd_loss`` masks padded
    # positions out by ``own_turn_may_mask``, so when no may steps are active
    # the contribution is exactly zero. Always running avoids a per-call
    # ``.any().item()`` sync on ``may_is_active``.
    may_logits_step = pc_online.may_logits_per_step.to(dtype=dtype)
    log_pi_accept = torch.nn.functional.logsigmoid(may_logits_step)
    log_pi_decline = torch.nn.functional.logsigmoid(-may_logits_step)
    with torch.no_grad():
        may_reg_cur_logits = pc_reg_cur.may_logits_per_step.to(dtype=dtype)
        may_reg_prev_logits = pc_reg_prev.may_logits_per_step.to(dtype=dtype)
        may_lp_reg_blend_accept = alpha * torch.nn.functional.logsigmoid(may_reg_cur_logits) + (
            1.0 - alpha
        ) * torch.nn.functional.logsigmoid(may_reg_prev_logits)
        may_lp_reg_blend_decline = alpha * torch.nn.functional.logsigmoid(-may_reg_cur_logits) + (
            1.0 - alpha
        ) * torch.nn.functional.logsigmoid(-may_reg_prev_logits)
    log_ratio_accept = log_pi_accept.detach() - may_lp_reg_blend_accept
    log_ratio_decline = log_pi_decline.detach() - may_lp_reg_blend_decline
    sampled_is_accept = pc_online.may_selected_per_step > 0.5
    with torch.no_grad():
        tgt_p_accept = torch.sigmoid(pc_tgt.may_logits_per_step.to(dtype=dtype))
        may_mu_sampled = torch.where(sampled_is_accept, tgt_p_accept, 1.0 - tgt_p_accept)
        may_inv_mu = (1.0 / may_mu_sampled.clamp_min(1e-30)).clamp(max=config.q_corr_rho_bar)
    may_sampled_log_ratio = torch.where(sampled_is_accept, log_ratio_accept, log_ratio_decline)

    cl_sum = v_online.new_zeros(())
    pl_sum = v_online.new_zeros(())
    # Counters are 0-d ``torch.long`` tensors: ``critic_loss`` /
    # ``may_neurd_loss`` / ``neurd_loss_per_choice`` now return per-call
    # counts as tensors, and accumulating into a tensor zero avoids the
    # implicit ``.item()`` sync that the previous int initialization
    # triggered on the first ``+=``.
    cl_count_total = torch.zeros((), dtype=torch.long, device=device)
    pl_count_total = torch.zeros((), dtype=torch.long, device=device)
    n_q_clipped_total = torch.zeros((), dtype=torch.long, device=device)
    v_hat_sum = torch.zeros((), dtype=dtype, device=device)
    transformed_sum = torch.zeros((), dtype=dtype, device=device)
    v_target_reg_share_sum_t = torch.zeros((), dtype=dtype, device=device)
    v_target_reg_share_count_t = torch.zeros((), dtype=torch.long, device=device)

    for rewards_p, is_own_p, own_idx_p in (
        (rewards_p0, is_own_p0, own_idx_p0),
        (rewards_p1, is_own_p1, own_idx_p1),
    ):
        transformed = transform_rewards(
            rewards_p,
            logp_theta=lp_online_dt,
            logp_reg_cur=lp_reg_cur_scalar,
            logp_reg_prev=lp_reg_prev_scalar,
            alpha=alpha,
            eta=config.eta,
            perspective_is_player_i=is_own_p,
            log_ratio_clip=config.reg_log_ratio_clip,
        )
        v_out = _two_player_vtrace_batched(
            rewards=transformed.detach(),
            values=v_tgt_dt,
            logp_theta=lp_tgt_scalar,
            logp_mu=logp_mu_dev,
            perspective_is_player_i=is_own_p,
            ep_offsets=ep_offsets,
            own_idx=own_idx_p,
            max_ep_len=max_ep_len,
            rho_bar=config.vtrace_rho_bar,
            c_bar=config.vtrace_c_bar,
        )

        if compute_v_target_reg_share:
            with torch.no_grad():
                v_out_term = _two_player_vtrace_batched(
                    rewards=rewards_p.detach(),
                    values=v_tgt_dt,
                    logp_theta=lp_tgt_scalar,
                    logp_mu=logp_mu_dev,
                    perspective_is_player_i=is_own_p,
                    ep_offsets=ep_offsets,
                    own_idx=own_idx_p,
                    max_ep_len=max_ep_len,
                    rho_bar=config.vtrace_rho_bar,
                    c_bar=config.vtrace_c_bar,
                )
                own_f = is_own_p.to(dtype=dtype)
                v_reg_part = (v_out.v_hat - v_out_term.v_hat).abs() * own_f
                v_term_part = v_out_term.v_hat.abs() * own_f
                share = v_reg_part / (v_term_part + v_reg_part + 1e-8)
                v_target_reg_share_sum_t = v_target_reg_share_sum_t + share.sum()
                v_target_reg_share_count_t += is_own_p.sum().to(dtype=torch.long)

        cl_part_sum, cl_part_count = critic_loss(
            v_theta=v_online,
            v_hat=v_out.v_hat,
            perspective_is_player_i=is_own_p,
        )
        cl_sum = cl_sum + cl_part_sum
        cl_count_total += cl_part_count

        if has_groups:
            sampled_inner_per_step = (
                rewards_p
                + config.eta * per_step_log_ratio_sampled
                + ratio_own_per_step * (v_out.r_hat_next + v_out.v_hat_next)
                - v_tgt_dt
            )
        else:
            sampled_inner_per_step = torch.zeros(T_total, dtype=dtype, device=device)

        # May-head loss runs unconditionally; ``own_turn_may_mask`` masks
        # padded/inactive positions out so empty batches contribute 0.
        may_and_own = pc_online.may_is_active & is_own_p
        may_sampled_inner = (
            rewards_p
            + config.eta * may_sampled_log_ratio
            + ratio_own_per_step * (v_out.r_hat_next + v_out.v_hat_next)
            - v_tgt_dt
        )
        may_q_sampled_correction = may_inv_mu * may_sampled_inner
        q_accept = (
            -config.eta * log_ratio_accept
            + torch.where(
                sampled_is_accept,
                may_q_sampled_correction,
                torch.zeros_like(rewards_p),
            )
            + v_tgt_dt
        )
        q_decline = (
            -config.eta * log_ratio_decline
            + torch.where(
                ~sampled_is_accept,
                may_q_sampled_correction,
                torch.zeros_like(rewards_p),
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

        if has_groups:
            step_is_own_flat = is_own_p[step_for_flat]
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

        v_hat_sum = v_hat_sum + v_out.v_hat.detach().sum()
        transformed_sum = transformed_sum + transformed.detach().sum()

    return _TrajLossPieces(
        cl_sum=cl_sum,
        cl_count=cl_count_total,
        pl_sum=pl_sum,
        pl_count=pl_count_total,
        n_q_clipped=n_q_clipped_total,
        v_hat_sum=v_hat_sum,
        transformed_sum=transformed_sum,
        transformed_count=max(2 * T_total, 1),
        entropy_sum=entropy_online.detach().to(dtype=dtype).sum(),
        entropy_count=torch.tensor(T_total, dtype=torch.long, device=device),
        sampled_log_ratio_sum=sampled_log_ratio_sum_t,
        sampled_log_ratio_absmax=sampled_log_ratio_absmax_t,
        sampled_log_ratio_count=T_total if compute_policy_drift_diagnostics else 0,
        is_bias_up_sum=is_bias_up_sum_t,
        is_bias_up_count=up_count_t,
        is_bias_down_sum=is_bias_down_sum_t,
        is_bias_down_count=down_count_t,
        v_target_reg_share_sum=v_target_reg_share_sum_t if compute_v_target_reg_share else None,
        v_target_reg_share_count=v_target_reg_share_count_t,
    )


def _maybe_precompute_replay_forward(
    policy: Any,
    episodes: Sequence[Sequence[int]],
    *,
    prepared_batch: Any | None = None,
    flat_rows: Sequence[int] | None = None,
    ep_lens: Sequence[int] | None = None,
    lstm_layout: PerPlayerLstmLayout | None = None,
    lstm_chunk_size: int | None = None,
) -> Any | None:
    """Return ``policy.precompute_replay_forward(...)`` when available.

    The text-encoder policy fuses the encoder forward with the per-episode
    LSTM scan so the cached encoder outputs can be reused by the per-choice
    scoring forward (avoiding a second encoder pass over the same batch).
    Policies without the helper (legacy slot encoder, test stubs) return
    ``None``; the caller falls back to the legacy h_out + hidden_override path.
    """

    fn = getattr(policy, "precompute_replay_forward", None)
    if fn is None:
        return None
    kwargs: dict[str, Any] = {}
    if prepared_batch is not None:
        kwargs["prepared_batch"] = prepared_batch
    if flat_rows is not None:
        kwargs["flat_rows_override"] = flat_rows
    if ep_lens is not None:
        kwargs["ep_lens_override"] = ep_lens
    if lstm_layout is not None:
        kwargs["lstm_layout_override"] = lstm_layout
    if lstm_chunk_size is not None:
        kwargs["lstm_chunk_size"] = int(lstm_chunk_size)
    return fn([list(ep) for ep in episodes], **kwargs)


def _maybe_prepare_rnad_replay_batch(policy: Any, flat_rows: Sequence[int]) -> Any | None:
    """Return a backend-prepared R-NaD replay batch when available."""

    fn = getattr(policy, "prepare_rnad_replay_batch", None)
    if fn is None:
        return None
    return fn(flat_rows)


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
    online: Any,
    target: Any,
    reg_cur: Any,
    reg_prev: Any,
    episodes_replay_rows: Sequence[Sequence[int]],
    episodes_perspective: Sequence[Sequence[int]],
    episodes_terminal_reward_p0: Sequence[float],
    episodes_zero_sum: Sequence[bool],
    episodes_logp_mu: Sequence[Tensor],
    config: RNaDConfig,
    alpha: float,
    compute_v_target_reg_share: bool = False,
    compute_policy_drift_diagnostics: bool = False,
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
    if len(episodes_terminal_reward_p0) != n_eps:
        raise ValueError("episodes_terminal_reward_p0 length mismatch")
    if len(episodes_zero_sum) != n_eps:
        raise ValueError("episodes_zero_sum length mismatch")
    if len(episodes_logp_mu) != n_eps:
        raise ValueError("episodes_logp_mu length mismatch")

    flat_rows: list[int] = []
    ep_lens: list[int] = []
    ep_offsets: list[int] = [0]
    perspective_host: list[int] = []
    max_ep_len = 0
    for ep, perspectives in zip(episodes_replay_rows, episodes_perspective, strict=True):
        if not ep:
            raise ValueError("each episode must contain at least one row")
        if len(perspectives) != len(ep):
            raise ValueError("episodes_perspective entries must match episode lengths")
        flat_rows.extend(int(r) for r in ep)
        for player in perspectives:
            player_i = int(player)
            perspective_host.append(player_i)
            if player_i not in (0, 1):
                raise ValueError("episodes_perspective values must be 0 or 1")
        ep_offsets.append(ep_offsets[-1] + len(ep))
        ep_lens.append(len(ep))
        if len(ep) > max_ep_len:
            max_ep_len = len(ep)

    device = next(iter(online.parameters())).device

    # 1) Single fused encoder + LSTM recompute per policy. When the policy
    #    exposes ``precompute_replay_forward`` (text encoder), the encoder
    #    runs once on the flat batch and is reused by the per-choice forward
    #    below via ``cached=...``; otherwise we fall back to the legacy
    #    ``recompute_lstm_outputs_for_episodes`` + ``hidden_override`` path.
    with torch.profiler.record_function("rnad_batched_trajectory_loss/prepare_replay_batch"):
        shared_replay_batch = _maybe_prepare_rnad_replay_batch(online, flat_rows)

    with torch.profiler.record_function("rnad_batched_trajectory_loss/device_input_tensors"):
        batch_perspective = getattr(shared_replay_batch, "perspective_player_idx", None)
        if isinstance(batch_perspective, Tensor) and int(batch_perspective.shape[0]) == len(
            flat_rows
        ):
            perspective_flat = batch_perspective.to(device=device)
        else:
            perspective_flat = torch.tensor(perspective_host, dtype=torch.long, device=device)
        terminal_reward_p0_t = torch.tensor(
            [float(r) for r in episodes_terminal_reward_p0], dtype=torch.float32, device=device
        )
        zero_sum_t = torch.tensor(
            [bool(z) for z in episodes_zero_sum], dtype=torch.bool, device=device
        )
        ep_offsets_t = torch.tensor(ep_offsets, dtype=torch.long, device=device)
        own_idx_p0_t = (perspective_flat == 0).nonzero(as_tuple=False).squeeze(-1)
        own_idx_p1_t = (perspective_flat == 1).nonzero(as_tuple=False).squeeze(-1)
        lstm_layout = build_per_player_lstm_layout(
            perspective_flat,
            episodes_perspective,
            ep_offsets=ep_offsets_t,
            chunk_size=config.bptt_chunk_size,
        )
        batch_logp_mu = getattr(shared_replay_batch, "old_log_prob", None)
        logp_mu = (
            batch_logp_mu.to(device=device)
            if isinstance(batch_logp_mu, Tensor) and int(batch_logp_mu.shape[0]) == len(flat_rows)
            else torch.empty(0, dtype=torch.float32, device=device)
        )
    layout = _RNaDBatchLayout(
        flat_rows=tuple(flat_rows),
        ep_lens=tuple(ep_lens),
        ep_offsets=ep_offsets_t,
        perspective=perspective_flat,
        terminal_reward_p0=terminal_reward_p0_t,
        zero_sum=zero_sum_t,
        logp_mu=logp_mu,
        own_idx_p0=own_idx_p0_t,
        own_idx_p1=own_idx_p1_t,
        lstm_layout=lstm_layout,
    )
    with torch.profiler.record_function("rnad_batched_trajectory_loss/online_precompute"):
        online_cache = _maybe_precompute_replay_forward(
            online,
            episodes_replay_rows,
            prepared_batch=shared_replay_batch,
            flat_rows=flat_rows if shared_replay_batch is not None else None,
            ep_lens=ep_lens if shared_replay_batch is not None else None,
            lstm_layout=layout.lstm_layout,
            lstm_chunk_size=config.bptt_chunk_size,
        )
    if online_cache is not None:
        online_h_concat: Tensor | None = online_cache.h_concat
        if shared_replay_batch is None:
            shared_replay_batch = online_cache.batch
    else:
        shared_replay_batch = None
        with torch.profiler.record_function("rnad_batched_trajectory_loss/online_lstm_recompute"):
            online_h_list = _maybe_recompute_lstm_h_out_episodes(online, episodes_replay_rows)
        online_h_concat = torch.cat(online_h_list, dim=0) if online_h_list is not None else None

    online_per_choice = getattr(online, "evaluate_replay_batch_per_choice")  # noqa: B009
    target_per_choice = getattr(target, "evaluate_replay_batch_per_choice")  # noqa: B009
    reg_cur_per_choice = getattr(reg_cur, "evaluate_replay_batch_per_choice")  # noqa: B009
    reg_prev_per_choice = getattr(reg_prev, "evaluate_replay_batch_per_choice")  # noqa: B009

    def _forward(per_choice: Any, h_concat: Tensor | None, cache: Any) -> Any:
        if cache is not None:
            return per_choice(list(flat_rows), cached=cache)
        if h_concat is not None:
            return per_choice(list(flat_rows), hidden_override=h_concat)
        return per_choice(list(flat_rows))

    # 2) Single per-choice forward per policy across the full batch.
    with torch.profiler.record_function("rnad_batched_trajectory_loss/online_per_choice"):
        lp_online_b, entropy_online_b, v_online_b, pc_online_b = _forward(
            online_per_choice, online_h_concat, online_cache
        )
    del online_cache, online_h_concat

    def _aux_forward(policy: Any, per_choice: Any) -> Any:
        with torch.profiler.record_function("rnad_batched_trajectory_loss/aux_precompute"):
            cache = _maybe_precompute_replay_forward(
                policy,
                episodes_replay_rows,
                prepared_batch=shared_replay_batch,
                flat_rows=flat_rows if shared_replay_batch is not None else None,
                ep_lens=ep_lens if shared_replay_batch is not None else None,
                lstm_layout=layout.lstm_layout,
                lstm_chunk_size=config.bptt_chunk_size,
            )
        if cache is None:
            with torch.profiler.record_function("rnad_batched_trajectory_loss/aux_lstm_recompute"):
                h_list = _maybe_recompute_lstm_h_out_episodes(policy, episodes_replay_rows)
            h_concat = torch.cat(h_list, dim=0) if h_list is not None else None
        else:
            h_concat = cache.h_concat
        with torch.profiler.record_function("rnad_batched_trajectory_loss/aux_per_choice"):
            out = _forward(per_choice, h_concat, cache)
        del cache, h_concat
        return out

    with torch.no_grad():
        with torch.profiler.record_function("rnad_batched_trajectory_loss/target_forward"):
            lp_tgt_b, _, v_tgt_b, pc_tgt_b = _aux_forward(target, target_per_choice)
        with torch.profiler.record_function("rnad_batched_trajectory_loss/reg_cur_forward"):
            _, _, _, pc_reg_cur_b = _aux_forward(reg_cur, reg_cur_per_choice)
        with torch.profiler.record_function("rnad_batched_trajectory_loss/reg_prev_forward"):
            _, _, _, pc_reg_prev_b = _aux_forward(reg_prev, reg_prev_per_choice)

    # 3) Single batched loss assembly across the whole flat batch.
    with torch.profiler.record_function("rnad_batched_trajectory_loss/logp_mu_to_device"):
        target_dtype = v_online_b.dtype
        if int(layout.logp_mu.numel()) == len(flat_rows):
            logp_mu_flat = layout.logp_mu.to(device=device, dtype=target_dtype)
        else:
            logp_mu_flat = torch.cat(
                [t.to(device=device, dtype=target_dtype) for t in episodes_logp_mu]
            )

    with torch.profiler.record_function("rnad_batched_trajectory_loss/batched_loss_assembly"):
        pieces = _batched_trajectory_loss_from_forwards(
            lp_online=lp_online_b,
            entropy_online=entropy_online_b,
            v_online=v_online_b,
            pc_online=pc_online_b,
            lp_tgt_scalar=lp_tgt_b,
            v_tgt=v_tgt_b,
            pc_tgt=pc_tgt_b,
            pc_reg_cur=pc_reg_cur_b,
            pc_reg_prev=pc_reg_prev_b,
            perspective=layout.perspective,
            terminal_reward_p0=layout.terminal_reward_p0,
            zero_sum=layout.zero_sum,
            logp_mu=logp_mu_flat,
            ep_offsets=layout.ep_offsets,
            own_idx_p0=layout.own_idx_p0,
            own_idx_p1=layout.own_idx_p1,
            max_ep_len=max_ep_len,
            config=config,
            alpha=alpha,
            compute_v_target_reg_share=compute_v_target_reg_share,
            compute_policy_drift_diagnostics=compute_policy_drift_diagnostics,
        )
    return [pieces]


def rnad_update_trajectory(
    *,
    online: Any,
    target: Any,
    reg_cur: Any,
    reg_prev: Any,
    optimizer: torch.optim.Optimizer,
    replay_rows: list[int],
    perspective_player_idx: Sequence[int],
    terminal_reward_p0: float,
    zero_sum: bool,
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
        terminal_reward_p0=terminal_reward_p0,
        zero_sum=zero_sum,
        logp_mu=logp_mu,
        config=config,
        alpha=alpha,
    )
    cl = pieces.cl_sum / pieces.cl_count.clamp_min(1).to(dtype=pieces.cl_sum.dtype)
    pl = pieces.pl_sum / pieces.pl_count.clamp_min(1).to(dtype=pieces.pl_sum.dtype)
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
    # Skip optimizer step on non-finite grads/loss to avoid poisoning the
    # weights — otherwise one bad batch produces NaN logits forever and the
    # next Categorical sampler call hits a CUDA multinomial assert.
    if not math.isfinite(grad_norm) or not math.isfinite(float(loss.detach())):
        optimizer.zero_grad(set_to_none=True)
        print(
            f"[rnad] non-finite update skipped: loss={float(loss.detach()):.4g} "
            f"grad_norm={grad_norm:.4g} cl={float(cl.detach()):.4g} pl={float(pl.detach()):.4g}",
            flush=True,
        )
    else:
        optimizer.step()

    assert isinstance(target, nn.Module)
    assert isinstance(online, nn.Module)
    polyak_update_(target, online, gamma=config.target_ema_gamma)

    stats_h = (
        torch.stack(
            [
                pieces.v_hat_sum.detach().to(dtype=torch.float32),
                pieces.transformed_sum.detach().to(dtype=torch.float32),
                (
                    pieces.entropy_sum.detach().to(dtype=torch.float32)
                    / pieces.entropy_count.clamp_min(1).to(dtype=torch.float32)
                ),
            ]
        )
        .cpu()
        .tolist()
    )
    stats_denom = max(pieces.transformed_count, 1)
    return RNaDStats(
        loss=float(loss.detach()),
        critic_loss=float(cl.detach()),
        policy_loss=float(pl.detach()),
        v_hat_mean=float(stats_h[0]) / stats_denom,
        grad_norm=grad_norm,
        transformed_reward_mean=float(stats_h[1]) / stats_denom,
        entropy=float(stats_h[2]),
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
