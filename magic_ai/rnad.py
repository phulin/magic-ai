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
        in_range = (flat_logits >= -beta) & (flat_logits <= beta)
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
    """One R-NaD gradient step on a single trajectory's replay rows.

    Trains both players simultaneously (self-play symmetry): the loss is
    summed over both ``own_player_idx in {0, 1}`` branches so each
    trajectory contributes to both halves of the policy, matching the
    paper's self-play formulation.

    Pipeline (DeepNash paper §157-191, simplified to sampled-action NeuRD):

    1. Forward the trajectory through online, target (no_grad),
       reg_cur (no_grad), reg_prev (no_grad); pull scalar joint log-probs
       and own-side values.
    2. For each player i in {0, 1}:
         a. Construct i's terminal reward vector (±1 on winner, 0 on draw,
            sign-flipped if the final step's perspective-player != i).
         b. Apply :func:`transform_rewards` using the **online** log-probs
            (detached) so the entropy-bonus r' - eta * log(pi_theta/pi_reg)
            matches paper §161, where pi_theta is the learning policy.
         c. Run :func:`two_player_vtrace` using target log-probs and target
            values as the bootstrap estimator inputs.
         d. Accumulate :func:`critic_loss` on i's own-turn steps and
            :func:`sampled_neurd_loss` using the online log-probs.
    3. Single backward + Adam step with ``config.grad_clip``; Polyak update
       target.

    The per-player branches are disjoint in ``is_own`` masks (each step
    belongs to exactly one player), so summing the two losses covers every
    timestep exactly once.
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
        # Terminal reward from player ``own_idx``'s POV. The network value
        # at the last step is from that step's perspective-player POV, so
        # we sign-flip the reward when the final step isn't ``own_idx``.
        rewards = torch.zeros(t_len, dtype=dtype, device=device)
        if winner_idx >= 0:
            # Terminal reward is game-global from ``own_idx``'s POV, not
            # perspective-sign-flipped: the zero-sum accumulator in
            # two_player_vtrace consumes rewards in the own-player frame.
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

    loss = total_cl + total_pl

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
        critic_loss=float(total_cl.detach()),
        policy_loss=float(total_pl.detach()),
        v_hat_mean=sum(v_hat_means) / len(v_hat_means),
        grad_norm=grad_norm,
        transformed_reward_mean=sum(transformed_means) / len(transformed_means),
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
    """Full per-action NeuRD variant (paper §188, with beta gate).

    Differs from :func:`rnad_update_trajectory` only in the policy loss:
    uses :func:`neurd_loss_per_choice` over all legal per-choice logits
    with the paper's ``[-beta, beta]`` logit-magnitude gate, rather than
    the sampled-action policy-gradient estimator. Q per choice is
    simplified to ``v_hat * 1[choice == sampled]`` — a single-sample
    estimator that avoids needing counterfactual per-action value
    estimates.

    Requires the policy objects to expose ``evaluate_replay_batch_per_choice``.
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

    # Per-choice forward on online; scalar-log-prob forwards on others.
    online_per_choice = getattr(online, "evaluate_replay_batch_per_choice")  # noqa: B009
    lp_online, _, v_online, pc_online = online_per_choice(list(replay_rows))
    with torch.no_grad():
        lp_tgt, _, v_tgt, _ = target.evaluate_replay_batch(list(replay_rows))
        lp_reg_cur, _, _, _ = reg_cur.evaluate_replay_batch(list(replay_rows))
        lp_reg_prev, _, _, _ = reg_prev.evaluate_replay_batch(list(replay_rows))

    dtype = v_online.dtype
    logp_mu_dev = logp_mu.to(device=device, dtype=dtype)
    lp_online_dt = lp_online.detach().to(dtype=dtype)
    lp_tgt_dt = lp_tgt.detach().to(dtype=dtype)
    lp_reg_cur_dt = lp_reg_cur.to(dtype=dtype)
    lp_reg_prev_dt = lp_reg_prev.to(dtype=dtype)
    v_tgt_dt = v_tgt.detach().to(dtype=dtype)

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
            logp_reg_cur=lp_reg_cur_dt,
            logp_reg_prev=lp_reg_prev_dt,
            alpha=alpha,
            eta=config.eta,
            perspective_is_player_i=is_own,
        )
        v_out = two_player_vtrace(
            rewards=transformed.detach(),
            values=v_tgt_dt,
            logp_theta=lp_tgt_dt,
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

        # Sampled-action NeuRD on the Bernoulli may head. The may head is a
        # single logit per step with two implicit "choices" (accept / decline);
        # surfacing it through neurd_loss_per_choice would double the complexity
        # of ReplayPerChoice for no gain, so we use the policy-gradient-theorem
        # form here — same as sampled_neurd_loss — gated to may steps only.
        if pc_online.may_is_active.any():
            may_and_own = pc_online.may_is_active & is_own
            total_pl = total_pl + sampled_neurd_loss(
                log_prob=lp_online,
                q_hat=v_out.q_hat,
                own_turn_mask=may_and_own,
                clip=config.neurd_clip,
            )

        if pc_online.flat_logits.numel() > 0:
            # Q per choice: v_hat if (step is own-turn AND choice is sampled),
            # zero otherwise. Non-own-turn choices contribute zero gradient.
            step_is_own = is_own[pc_online.group_idx]
            sampled_for_step = pc_online.sampled_col_per_step[pc_online.group_idx]
            is_sampled = pc_online.choice_cols == sampled_for_step
            flat_q = torch.where(
                step_is_own & is_sampled,
                v_out.q_hat[pc_online.group_idx],
                torch.zeros_like(pc_online.flat_logits),
            )
            total_pl = total_pl + neurd_loss_per_choice(
                flat_logits=pc_online.flat_logits,
                flat_q=flat_q,
                group_idx=pc_online.group_idx,
                num_groups=t_len,
                beta=config.neurd_beta,
                clip=config.neurd_clip,
            )

        v_hat_means.append(float(v_out.v_hat.detach().mean()))
        transformed_means.append(float(transformed.detach().mean()))

    loss = total_cl + total_pl
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
        critic_loss=float(total_cl.detach()),
        policy_loss=float(total_pl.detach()),
        v_hat_mean=sum(v_hat_means) / len(v_hat_means),
        grad_norm=grad_norm,
        transformed_reward_mean=sum(transformed_means) / len(transformed_means),
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
