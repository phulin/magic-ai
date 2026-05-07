"""PPO update helpers for mage-go self-play."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from magic_ai.returns import gae_returns, gae_returns_batched
from magic_ai.rollout import (
    PPOStats,
    RolloutStep,
    life_tiebreak_terminal_reward,
    terminal_reward_for_finish,
)
from magic_ai.training_interfaces import PPOReplayPolicy

__all__ = [
    "PPOStats",
    "RolloutStep",
    "gae_returns",
    "gae_returns_batched",
    "life_tiebreak_terminal_reward",
    "ppo_update",
    "terminal_reward_for_finish",
]


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
    # Keep row indices on CPU. Text replay uses a host row-length mirror to
    # size packed minibatches without syncing on ``seq_lengths.sum()``; policies
    # move indices to their storage device at gather sites.
    replay_rows = replay_rows.to(device=torch.device("cpu"), dtype=torch.long)
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
    nonfinite_count_t = torch.zeros((), dtype=torch.int64, device=device)
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
        permutation = torch.randperm(n_steps)
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
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, foreach=True)
            # Neutralize non-finite grads in-place so the optimizer step is a
            # no-op for the bad batch without forcing a host sync. Without this,
            # a single bad batch (e.g. an all-masked decision row producing NaN
            # log-probs) propagates NaN into the weights and every subsequent
            # forward returns NaN logits, which crashes the next sampler call
            # with the CUDA ``0 <= p <= 1`` multinomial assert. Track the count
            # on-device and sync once at the end of the update for diagnostics.
            mb_nonfinite = (~torch.isfinite(grad_norm)) | (~torch.isfinite(loss.detach()))
            nonfinite_count_t += mb_nonfinite.to(torch.int64)
            for p in policy.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
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
    nonfinite_count = int(nonfinite_count_t.item())
    if nonfinite_count > 0:
        print(
            f"[ppo] non-finite grads neutralized in "
            f"{nonfinite_count}/{num_minibatches} minibatches",
            flush=True,
        )
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
