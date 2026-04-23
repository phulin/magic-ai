"""PPO update helpers for mage-go self-play."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from magic_ai.model import PPOPolicy


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
    policy: PPOPolicy,
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

    sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "spr_loss": 0.0,
    }
    num_minibatches = 0
    indices = list(range(len(steps)))
    for _ in range(epochs):
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

            spr_loss_val = 0.0
            if spr_active:
                assert extras is not None
                spr_loss = policy.compute_spr_loss(extras.step_indices, extras=extras)
                loss = loss + spr_coef * spr_loss
                spr_loss_val = float(spr_loss.detach().cpu())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (batch_old_log_probs - batch_log_probs).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
            sums["loss"] += float(loss.detach().cpu())
            sums["policy_loss"] += float(policy_loss.detach().cpu())
            sums["value_loss"] += float(value_loss.detach().cpu())
            sums["entropy"] += float(entropy.detach().cpu())
            sums["approx_kl"] += float(approx_kl.detach().cpu())
            sums["clip_fraction"] += float(clip_fraction.detach().cpu())
            sums["spr_loss"] += spr_loss_val
            num_minibatches += 1

    if num_minibatches == 0:
        raise RuntimeError("PPO update did not run any minibatches")
    if spr_coef > 0.0 and policy.spr_enabled:
        policy.update_spr_target()
    return PPOStats(
        loss=sums["loss"] / num_minibatches,
        policy_loss=sums["policy_loss"] / num_minibatches,
        value_loss=sums["value_loss"] / num_minibatches,
        entropy=sums["entropy"] / num_minibatches,
        approx_kl=sums["approx_kl"] / num_minibatches,
        clip_fraction=sums["clip_fraction"] / num_minibatches,
        spr_loss=sums["spr_loss"] / num_minibatches,
    )


def gae_returns(
    steps: Sequence[RolloutStep],
    *,
    winner_idx: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tensor:
    """Compute perspective-aware GAE returns for a zero-sum two-player game."""

    if not steps:
        raise ValueError("cannot compute GAE returns for an empty rollout")

    num_steps = len(steps)
    values_t = torch.tensor([step.value for step in steps], dtype=torch.float32)
    rewards_t = torch.zeros(num_steps, dtype=torch.float32)
    advantages_t = torch.zeros(num_steps, dtype=torch.float32)

    last_step = steps[-1]
    if winner_idx < 0:
        rewards_t[-1] = 0.0
    elif winner_idx == last_step.perspective_player_idx:
        rewards_t[-1] = 1.0
    else:
        rewards_t[-1] = -1.0

    next_advantage = 0.0
    for idx in range(num_steps - 1, -1, -1):
        if idx == num_steps - 1:
            delta = rewards_t[idx] - values_t[idx]
            next_advantage = float(delta.item())
        else:
            next_same_player = (
                steps[idx + 1].perspective_player_idx == steps[idx].perspective_player_idx
            )
            sign = 1.0 if next_same_player else -1.0
            delta = rewards_t[idx] + gamma * sign * values_t[idx + 1] - values_t[idx]
            next_advantage = float(delta.item()) + gamma * gae_lambda * sign * next_advantage
        advantages_t[idx] = next_advantage

    return advantages_t + values_t
