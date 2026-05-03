"""PPO update helpers for mage-go self-play."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from magic_ai.training_interfaces import PPOReplayPolicy


@dataclass(frozen=True)
class RolloutStep:
    perspective_player_idx: int
    old_log_prob: float
    value: float
    reward: float = 0.0
    replay_idx: int | None = None


@dataclass(frozen=True)
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    spr_loss: float = 0.0


def _iter_minibatch_slices(
    n: int,
    *,
    minibatch_size: int,
) -> Iterator[slice]:
    for start in range(0, n, minibatch_size):
        yield slice(start, min(start + minibatch_size, n))


def _effective_minibatch_size(
    minibatch_size: int,
    *,
    minibatch_token_limit: int | None,
    minibatch_max_tokens_per_row: int | None,
) -> int:
    if minibatch_token_limit is None or minibatch_max_tokens_per_row is None:
        return minibatch_size
    token_capped_rows = max(1, minibatch_token_limit // minibatch_max_tokens_per_row)
    return min(minibatch_size, token_capped_rows)


def ppo_update(
    policy: PPOReplayPolicy,
    optimizer: torch.optim.Optimizer,
    replay_rows: Tensor,
    returns: Tensor,
    *,
    epochs: int = 4,
    minibatch_size: int = 256,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    spr_coef: float = 0.0,
    between_epoch_fn: Callable[[], None] | None = None,
    minibatch_token_limit: int | None = None,
    minibatch_max_tokens_per_row: int | None = None,
) -> PPOStats:
    """Run PPO over cached policy inputs.

    ``replay_rows`` is a 1-D long tensor of buffer rows produced at sample
    time; ``returns`` is the matching GAE returns tensor. Both are device
    tensors — the encoder stack is re-run on every minibatch so all encoder,
    trunk, and head parameters receive gradients, and ``old_log_prob`` /
    ``value`` are gathered directly from the replay buffer rather than from
    Python-level rollout-step lists.
    """

    if replay_rows.numel() == 0:
        raise ValueError("cannot update PPO with an empty rollout")
    if returns.numel() != replay_rows.numel():
        raise ValueError("returns length must match replay_rows length")
    if minibatch_token_limit is not None and minibatch_token_limit < 1:
        raise ValueError("minibatch_token_limit must be positive when set")
    if minibatch_max_tokens_per_row is not None and minibatch_max_tokens_per_row < 1:
        raise ValueError("minibatch_max_tokens_per_row must be positive when set")

    device = next(policy.parameters()).device
    replay_rows = replay_rows.to(device=device, dtype=torch.long)
    returns = returns.to(device=device, dtype=torch.float32)
    old_log_probs, old_values = policy.gather_replay_old_log_prob_value(replay_rows)
    old_log_probs = old_log_probs.to(device=device, dtype=torch.float32)
    old_values = old_values.to(device=device, dtype=torch.float32)
    advantages = returns - old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False,
        ).clamp_min(1e-8)

    stat_names = (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "spr_loss",
    )
    sums_t = torch.zeros(len(stat_names), dtype=torch.float32, device=device)
    num_minibatches = 0
    n_steps = int(replay_rows.shape[0])
    policy.write_ppo_targets(
        replay_rows,
        old_log_probs,
        returns,
        advantages,
    )
    effective_minibatch_size = _effective_minibatch_size(
        minibatch_size,
        minibatch_token_limit=minibatch_token_limit,
        minibatch_max_tokens_per_row=minibatch_max_tokens_per_row,
    )
    for epoch_idx in range(epochs):
        permutation = torch.randperm(n_steps, device=device)
        shuffled_replay_rows = replay_rows[permutation]
        for batch_slice in _iter_minibatch_slices(
            n_steps,
            minibatch_size=effective_minibatch_size,
        ):
            mb_replay_rows = shuffled_replay_rows[batch_slice]
            spr_active = spr_coef > 0.0 and policy.spr_enabled
            batch_log_probs, batch_entropies, batch_values, extras = policy.evaluate_replay_batch(
                mb_replay_rows, return_extras=spr_active
            )
            batch_old_log_probs, batch_returns, batch_advantages = policy.gather_ppo_targets(
                mb_replay_rows
            )

            ratio = (batch_log_probs - batch_old_log_probs).exp()
            unclipped = ratio * batch_advantages
            clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(batch_values, batch_returns)
            entropy = batch_entropies.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            spr_loss_stat = loss.new_zeros(())
            if spr_active:
                assert extras is not None
                spr_loss = policy.compute_spr_loss(extras.step_indices, extras=extras)
                loss = loss + spr_coef * spr_loss
                spr_loss_stat = spr_loss.detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            # Skip the optimizer step on non-finite grads. Without this, a
            # single bad batch (e.g. an all-masked decision row producing NaN
            # log-probs) propagates NaN into the weights and every subsequent
            # forward returns NaN logits, which crashes the next sampler call
            # with the CUDA ``0 <= p <= 1`` multinomial assert. Combine both
            # checks into one host sync per minibatch.
            if not bool(torch.isfinite(grad_norm) & torch.isfinite(loss)):
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[ppo] non-finite update skipped: "
                    f"loss={float(loss.detach()):.4g} "
                    f"grad_norm={float(grad_norm):.4g} "
                    f"policy_loss={float(policy_loss.detach()):.4g} "
                    f"value_loss={float(value_loss.detach()):.4g}",
                    flush=True,
                )
            else:
                optimizer.step()

            with torch.no_grad():
                approx_kl = (batch_old_log_probs - batch_log_probs).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
            sums_t += torch.stack(
                (
                    loss.detach(),
                    policy_loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    approx_kl.detach(),
                    clip_fraction.detach(),
                    spr_loss_stat,
                )
            )
            num_minibatches += 1
        if between_epoch_fn is not None and epoch_idx < epochs - 1:
            between_epoch_fn()

    if num_minibatches == 0:
        raise RuntimeError("PPO update did not run any minibatches")
    if spr_coef > 0.0 and policy.spr_enabled:
        policy.update_spr_target()
    sums = dict(zip(stat_names, (sums_t / num_minibatches).cpu().tolist(), strict=True))
    return PPOStats(
        loss=sums["loss"],
        policy_loss=sums["policy_loss"],
        value_loss=sums["value_loss"],
        entropy=sums["entropy"],
        approx_kl=sums["approx_kl"],
        clip_fraction=sums["clip_fraction"],
        spr_loss=sums["spr_loss"],
    )


def life_tiebreak_terminal_reward(life_p0: int, life_p1: int) -> float:
    """Per-player tiebreak score for step-cap timeouts, p0's perspective.

    Both life totals are clamped at 0 first. Returns 0.0 on a tie or when
    both players are at 0; otherwise ``(l0 - l1) / (l0 + l1)`` ∈ (-1, 1).
    """

    l0 = max(0, int(life_p0))
    l1 = max(0, int(life_p1))
    if l0 == l1:
        return 0.0
    return (l0 - l1) / float(l0 + l1)


def terminal_reward_for_finish(
    *,
    winner_idx: int,
    is_timeout: bool,
    life_p0: int,
    life_p1: int,
    draw_penalty: float,
) -> tuple[float, bool]:
    """Resolve a finished episode into ``(terminal_reward_p0, zero_sum)``.

    * Engine-declared win/loss → ±1, zero-sum.
    * Engine-declared draw (``winner_idx < 0`` and not a timeout) →
      ``-draw_penalty`` for both players; symmetric absorbing state.
    * Step-cap timeout → life-total tiebreak from p0's perspective; zero-sum.
    """

    if is_timeout:
        return life_tiebreak_terminal_reward(life_p0, life_p1), True
    if winner_idx == 0:
        return 1.0, True
    if winner_idx == 1:
        return -1.0, True
    return -float(draw_penalty), False


def gae_returns(
    steps: Sequence[RolloutStep],
    *,
    terminal_reward_p0: float,
    zero_sum: bool,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tensor:
    """Compute perspective-aware GAE returns for a two-player game.

    ``terminal_reward_p0`` is the terminal reward in p0's perspective
    (+1 for a p0 win, -1 for a p1 win, life-tiebreak ∈ (-1, 1) for a
    timeout, ``-draw_penalty`` for an engine-declared draw). ``zero_sum``
    selects the cross-player sign: ``True`` flips the sign each time the
    acting player switches (zero-sum convention), ``False`` keeps it +1
    throughout (symmetric absorbing state — the engine-draw branch).
    """

    if not steps:
        raise ValueError("cannot compute GAE returns for an empty rollout")

    num_steps = len(steps)
    values_t = torch.tensor([step.value for step in steps], dtype=torch.float32)
    players_t = torch.tensor(
        [step.perspective_player_idx for step in steps],
        dtype=torch.int8,
    )
    rewards_t = torch.zeros(num_steps, dtype=torch.float32)

    last_step = steps[-1]
    last_player = int(last_step.perspective_player_idx)
    if zero_sum and last_player == 1:
        rewards_t[-1] = -float(terminal_reward_p0)
    else:
        rewards_t[-1] = float(terminal_reward_p0)

    if num_steps == 1:
        return rewards_t

    cross_player_sign = -1.0 if zero_sum else 1.0
    signs_t = torch.where(
        players_t[1:] == players_t[:-1],
        torch.ones(num_steps - 1, dtype=torch.float32),
        torch.full((num_steps - 1,), cross_player_sign, dtype=torch.float32),
    )
    deltas_t = rewards_t - values_t
    deltas_t[:-1] += gamma * signs_t * values_t[1:]

    if gamma == 0.0 or gae_lambda == 0.0:
        advantages_t = deltas_t
    else:
        coeffs_rev = (gamma * gae_lambda * signs_t).flip(0)
        deltas_rev = deltas_t.flip(0)
        scan_coeffs = torch.cat(
            (
                torch.ones(1, dtype=torch.float32),
                coeffs_rev.cumprod(0),
            )
        )
        advantages_t = (scan_coeffs * torch.cumsum(deltas_rev / scan_coeffs, dim=0)).flip(0)

    return advantages_t + values_t


@torch.compile(dynamic=True, fullgraph=False)
def _gae_returns_batched_compiled(
    values_f: Tensor,
    players_l: Tensor,
    step_count_l: Tensor,
    terminal_reward_p0_f: Tensor,
    zero_sum_b: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tensor:
    dtype = values_f.dtype
    device = values_f.device
    batch_size, max_steps = values_f.shape

    arange_t = torch.arange(max_steps, device=device)
    valid = arange_t.unsqueeze(0) < step_count_l.unsqueeze(1)
    valid_f = valid.to(dtype)

    nonempty = step_count_l > 0
    safe_terminal = (step_count_l - 1).clamp_min(0)
    last_player = players_l.gather(1, safe_terminal.unsqueeze(1)).squeeze(1)
    # Terminal reward is given in p0's perspective; flip it on rows whose
    # last actor is p1 in the zero-sum branch. Symmetric (non-zero-sum)
    # rows leave it unflipped — both players see the same absorbing reward.
    flip = zero_sum_b & (last_player == 1)
    terminal_reward = torch.where(
        flip,
        -terminal_reward_p0_f.to(dtype),
        terminal_reward_p0_f.to(dtype),
    )
    terminal_reward = terminal_reward * nonempty.to(dtype)

    rewards = torch.zeros_like(values_f)
    rewards.scatter_(1, safe_terminal.unsqueeze(1), terminal_reward.unsqueeze(1))
    rewards = rewards * valid_f

    base_deltas = (rewards - values_f) * valid_f
    if max_steps > 1:
        same_player = players_l[:, 1:] == players_l[:, :-1]
        cross_sign = torch.where(
            zero_sum_b,
            values_f.new_full((batch_size,), -1.0),
            values_f.new_ones(batch_size),
        )
        signs = torch.where(
            same_player,
            values_f.new_ones(batch_size, max_steps - 1),
            cross_sign.unsqueeze(1).expand(batch_size, max_steps - 1),
        )
        next_valid = (arange_t[: max_steps - 1].unsqueeze(0) < (step_count_l - 1).unsqueeze(1)).to(
            dtype
        )
        bootstrap = gamma * signs * values_f[:, 1:] * next_valid
        bootstrap_padded = F.pad(bootstrap, (0, 1))
        deltas = base_deltas + bootstrap_padded
    else:
        deltas = base_deltas
        signs = values_f.new_zeros(batch_size, 0)

    if max_steps == 1 or gamma == 0.0 or gae_lambda == 0.0:
        advantages = deltas
    else:
        coeffs_rev = (gamma * gae_lambda * signs).flip(1)
        deltas_rev = deltas.flip(1)
        ones_col = values_f.new_ones(batch_size, 1)
        scan_coeffs = torch.cat((ones_col, coeffs_rev.cumprod(dim=1)), dim=1)
        advantages = (scan_coeffs * torch.cumsum(deltas_rev / scan_coeffs, dim=1)).flip(1)

    returns = (advantages + values_f) * valid_f
    return returns


def gae_returns_batched(
    values: Tensor,
    perspective_player_idx: Tensor,
    step_count: Tensor,
    *,
    terminal_reward_p0: Tensor,
    zero_sum: Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tensor:
    """Batched perspective-aware GAE returns for a padded rollout.

    Parameters
    ----------
    values:
        ``(B, T)`` float tensor of value estimates per step.
    perspective_player_idx:
        ``(B, T)`` integer tensor of the acting player at each step.
    step_count:
        ``(B,)`` integer tensor; row ``b`` uses positions ``[0, step_count[b])``.
    terminal_reward_p0:
        ``(B,)`` float tensor — terminal reward in p0's perspective. The
        compiled kernel flips it on rows whose terminal-step actor is p1
        in the zero-sum branch, and leaves it unflipped (symmetric) when
        ``zero_sum[b]`` is False.
    zero_sum:
        ``(B,)`` bool tensor. ``True`` ⇒ standard zero-sum two-player GAE
        (cross-player sign-flip is -1). ``False`` ⇒ symmetric absorbing
        state (no sign-flip across player switches); used for engine-
        declared draws so both players see the same absorbing reward.
    """

    if values.dim() != 2:
        raise ValueError("values must have shape (B, T)")
    if perspective_player_idx.shape != values.shape:
        raise ValueError("perspective_player_idx must match values shape")
    if step_count.shape != (values.shape[0],):
        raise ValueError("step_count must have shape (B,)")
    if terminal_reward_p0.shape != (values.shape[0],):
        raise ValueError("terminal_reward_p0 must have shape (B,)")
    if zero_sum.shape != (values.shape[0],):
        raise ValueError("zero_sum must have shape (B,)")

    device = values.device
    values_f = values.to(torch.float32)
    step_count_l = step_count.to(device=device, dtype=torch.long)
    players_l = perspective_player_idx.to(device=device, dtype=torch.long)
    terminal_f = terminal_reward_p0.to(device=device, dtype=torch.float32)
    zero_sum_b = zero_sum.to(device=device, dtype=torch.bool)
    return _gae_returns_batched_compiled(
        values_f,
        players_l,
        step_count_l,
        terminal_f,
        zero_sum_b,
        float(gamma),
        float(gae_lambda),
    )
