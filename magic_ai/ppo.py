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
            replay_rows = shuffled_replay_rows[batch_slice]
            spr_active = spr_coef > 0.0 and policy.spr_enabled
            batch_log_probs, batch_entropies, batch_values, extras = policy.evaluate_replay_batch(
                replay_rows, return_extras=spr_active
            )
            batch_old_log_probs, batch_returns, batch_advantages = policy.gather_ppo_targets(
                replay_rows
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
            # with the CUDA ``0 <= p <= 1`` multinomial assert.
            if not torch.isfinite(grad_norm) or not torch.isfinite(loss).item():
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


def gae_returns(
    steps: Sequence[RolloutStep],
    *,
    winner_idx: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    draw_penalty: float = 1.0,
) -> Tensor:
    """Compute perspective-aware GAE returns for a zero-sum two-player game.

    ``draw_penalty`` is applied to both players' terminal reward when the game
    ends in a draw (``winner_idx < 0``); the terminal reward is
    ``-draw_penalty``.
    """

    if not steps:
        raise ValueError("cannot compute GAE returns for an empty rollout")

    num_steps = len(steps)

    # Draws are a flat outcome both players experience identically. The
    # perspective-aware GAE below uses zero-sum sign-flipping to propagate the
    # last step's reward across player switches; that's correct for win/loss
    # (one side's +1 = other side's -1) but WRONG for draws, where both
    # sides should see -draw_penalty. Without this short-circuit the
    # propagation would hand one player +draw_penalty as their terminal
    # return, incentivizing them to stall on the opponent's turn.
    if winner_idx < 0:
        return torch.full((num_steps,), -draw_penalty, dtype=torch.float32)

    values_t = torch.tensor([step.value for step in steps], dtype=torch.float32)
    players_t = torch.tensor(
        [step.perspective_player_idx for step in steps],
        dtype=torch.int8,
    )
    rewards_t = torch.zeros(num_steps, dtype=torch.float32)

    last_step = steps[-1]
    if winner_idx == last_step.perspective_player_idx:
        rewards_t[-1] = 1.0
    else:
        rewards_t[-1] = -1.0

    if num_steps == 1:
        return rewards_t

    signs_t = torch.where(
        players_t[1:] == players_t[:-1],
        torch.ones(num_steps - 1, dtype=torch.float32),
        torch.full((num_steps - 1,), -1.0, dtype=torch.float32),
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


def gae_returns_batched(
    values: Tensor,
    perspective_player_idx: Tensor,
    step_count: Tensor,
    winner_idx: Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    draw_penalty: float = 1.0,
) -> Tensor:
    """Batched perspective-aware GAE returns for a padded rollout.

    Parameters
    ----------
    values:
        ``(B, T)`` float tensor of value estimates per step. Positions past
        each row's terminal index are ignored.
    perspective_player_idx:
        ``(B, T)`` integer tensor of the acting player at each step.
    step_count:
        ``(B,)`` integer tensor; row ``b`` uses positions ``[0, step_count[b])``.
        Rows with ``step_count == 0`` produce an all-zero output row.
    winner_idx:
        ``(B,)`` integer tensor. Use ``-1`` for a draw; the row is then filled
        with ``-draw_penalty`` over its valid positions, mirroring the
        per-episode reference implementation.

    The output ``(B, T)`` matches stacking ``gae_returns`` over each row, with
    zero-padding past ``step_count``.
    """

    if values.dim() != 2:
        raise ValueError("values must have shape (B, T)")
    if perspective_player_idx.shape != values.shape:
        raise ValueError("perspective_player_idx must match values shape")
    if step_count.shape != (values.shape[0],):
        raise ValueError("step_count must have shape (B,)")
    if winner_idx.shape != (values.shape[0],):
        raise ValueError("winner_idx must have shape (B,)")

    device = values.device
    dtype = torch.float32
    values_f = values.to(dtype)
    batch_size, max_steps = values_f.shape
    step_count_l = step_count.to(device=device, dtype=torch.long)
    winner_l = winner_idx.to(device=device, dtype=torch.long)
    players_l = perspective_player_idx.to(device=device, dtype=torch.long)

    arange_t = torch.arange(max_steps, device=device)
    valid = arange_t.unsqueeze(0) < step_count_l.unsqueeze(1)  # (B, T)
    valid_f = valid.to(dtype)

    nonempty = step_count_l > 0
    is_draw = winner_l < 0
    safe_terminal = (step_count_l - 1).clamp_min(0)
    last_player = players_l.gather(1, safe_terminal.unsqueeze(1)).squeeze(1)
    terminal_reward = torch.where(
        winner_l == last_player,
        values_f.new_ones(batch_size),
        values_f.new_full((batch_size,), -1.0),
    )
    terminal_reward = terminal_reward * nonempty.to(dtype)

    rewards = torch.zeros_like(values_f)
    rewards.scatter_(1, safe_terminal.unsqueeze(1), terminal_reward.unsqueeze(1))
    rewards = rewards * valid_f  # zero out scatter into row 0 of empty rows

    base_deltas = (rewards - values_f) * valid_f
    if max_steps > 1:
        same_player = players_l[:, 1:] == players_l[:, :-1]
        signs = torch.where(
            same_player,
            values_f.new_ones(batch_size, max_steps - 1),
            values_f.new_full((batch_size, max_steps - 1), -1.0),
        )
        # Bootstrap term only when the *next* step is also valid (t+1 < step_count).
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

    draw_row = values_f.new_full((), -float(draw_penalty)) * valid_f
    returns = torch.where(is_draw.unsqueeze(1), draw_row, returns)

    return returns
