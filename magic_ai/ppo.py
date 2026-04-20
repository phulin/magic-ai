"""PPO update and rollout helpers for mage-go self-play."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from magic_ai.actions import ActionRequest, PendingState
from magic_ai.game_state import GameStateSnapshot
from magic_ai.model import ActionTrace, CachedPolicyInput, PPOPolicy


@dataclass(frozen=True)
class RolloutStep:
    state: GameStateSnapshot
    pending: PendingState
    perspective_player_idx: int
    player_id: str
    player_name: str
    trace: ActionTrace
    old_log_prob: float
    value: float
    reward: float = 0.0
    cache: CachedPolicyInput | None = None


@dataclass(frozen=True)
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


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
) -> PPOStats:
    """Run PPO over cached policy inputs.

    Each rollout step must carry a ``CachedPolicyInput`` produced at sample
    time. The cache stores only parsed integer/scalar tensors and decision
    layouts; the encoder stack is re-run on every minibatch so all encoder,
    trunk, and head parameters receive gradients.
    """

    if not steps:
        raise ValueError("cannot update PPO with an empty rollout")
    if returns.numel() != len(steps):
        raise ValueError("returns length must match rollout length")
    if any(step.cache is None for step in steps):
        raise ValueError("all rollout steps must include cached policy inputs")

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

    last_stats: PPOStats | None = None
    indices = list(range(len(steps)))
    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            cached_steps = [cast(CachedPolicyInput, steps[idx].cache) for idx in batch_indices]
            batch_log_probs, batch_entropies, batch_values = policy.evaluate_cached_batch(
                cached_steps
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (batch_old_log_probs - batch_log_probs).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
            last_stats = PPOStats(
                loss=float(loss.detach().cpu()),
                policy_loss=float(policy_loss.detach().cpu()),
                value_loss=float(value_loss.detach().cpu()),
                entropy=float(entropy.detach().cpu()),
                approx_kl=float(approx_kl.detach().cpu()),
                clip_fraction=float(clip_fraction.detach().cpu()),
            )

    if last_stats is None:
        raise RuntimeError("PPO update did not run any minibatches")
    return last_stats


def terminal_returns(
    steps: Sequence[RolloutStep],
    *,
    winner: str,
    gamma: float = 1.0,
) -> Tensor:
    """Assign final game outcome from each acting player's perspective."""

    values: list[float] = []
    total = len(steps)
    for idx, step in enumerate(steps):
        if not winner:
            outcome = 0.0
        elif winner in {step.player_id, step.player_name}:
            outcome = 1.0
        else:
            outcome = -1.0
        values.append(outcome * (gamma ** max(0, total - idx - 1)))
    return torch.tensor(values, dtype=torch.float32)


def merge_pending_into_state(state: dict, pending: dict | None) -> GameStateSnapshot:
    snapshot = dict(state)
    if pending is not None:
        snapshot["pending"] = pending
    return cast(GameStateSnapshot, snapshot)


def rollout_step_from_policy(
    policy: PPOPolicy,
    state: GameStateSnapshot,
    pending: PendingState,
    *,
    deterministic: bool = False,
) -> tuple[ActionRequest, RolloutStep]:
    player_idx = int(pending.get("player_idx", 0))
    player = state["players"][player_idx]
    with torch.no_grad():
        policy_step = policy.act(
            state,
            pending,
            perspective_player_idx=player_idx,
            deterministic=deterministic,
        )
    rollout_step = RolloutStep(
        state=state,
        pending=pending,
        perspective_player_idx=player_idx,
        player_id=player.get("ID", ""),
        player_name=player.get("Name", ""),
        trace=policy_step.trace,
        old_log_prob=float(policy_step.log_prob.detach().cpu()),
        value=float(policy_step.value.detach().cpu()),
        cache=policy_step.cache,
    )
    return policy_step.action, rollout_step
