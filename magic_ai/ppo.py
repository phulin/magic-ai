"""PPO update helpers for mage-go self-play."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

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


def ppo_update(
    policy: PPOReplayPolicy,
    optimizer: torch.optim.Optimizer,
    steps: Sequence[RolloutStep],
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
) -> PPOStats:
    """Run PPO over cached policy inputs.

    Each rollout step must carry a replay row index produced at sample time.
    The encoder stack is re-run on every minibatch so all encoder, trunk, and
    head parameters receive gradients.
    """

    if not steps:
        raise ValueError("cannot update PPO with an empty rollout")
    if returns.numel() != len(steps):
        raise ValueError("returns length must match rollout length")
    if any(step.replay_idx is None for step in steps):
        raise ValueError("all rollout steps must include replay rows")

    device = next(policy.parameters()).device
    returns = returns.to(device=device, dtype=torch.float32)
    old_log_probs = torch.tensor(
        [step.old_log_prob for step in steps],
        dtype=torch.float32,
        device=device,
    )
    old_values = torch.tensor([step.value for step in steps], dtype=torch.float32, device=device)
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
    indices = list(range(len(steps)))
    for epoch_idx in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            replay_rows = [cast(int, steps[idx].replay_idx) for idx in batch_indices]
            spr_active = spr_coef > 0.0 and policy.spr_enabled
            batch_log_probs, batch_entropies, batch_values, extras = policy.evaluate_replay_batch(
                replay_rows, return_extras=spr_active
            )
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]

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
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
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
    values_t = torch.tensor([step.value for step in steps], dtype=torch.float32)
    players_t = torch.tensor(
        [step.perspective_player_idx for step in steps],
        dtype=torch.int8,
    )
    rewards_t = torch.zeros(num_steps, dtype=torch.float32)

    last_step = steps[-1]
    if winner_idx < 0:
        rewards_t[-1] = -draw_penalty
    elif winner_idx == last_step.perspective_player_idx:
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
