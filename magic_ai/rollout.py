"""Shared rollout records, trainer stats, and terminal reward helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutStep:
    perspective_player_idx: int
    old_log_prob: float
    value: float
    reward: float = 0.0
    replay_idx: int | None = None


@dataclass(frozen=True)
class TrainerStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    spr_loss: float = 0.0


PPOStats = TrainerStats


def life_tiebreak_terminal_reward(life_p0: int, life_p1: int) -> float:
    """Per-player tiebreak score for step-cap timeouts, p0's perspective.

    Both life totals are clamped at 0 first. Returns 0.0 on a tie or when
    both players are at 0; otherwise ``(l0 - l1) / (l0 + l1)`` in (-1, 1).
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

    * Engine-declared win/loss -> +/-1, zero-sum.
    * Engine-declared draw (``winner_idx < 0`` and not a timeout) ->
      ``-draw_penalty`` for both players; symmetric absorbing state.
    * Step-cap timeout -> life-total tiebreak from p0's perspective; zero-sum.
    """

    if is_timeout:
        return life_tiebreak_terminal_reward(life_p0, life_p1), True
    if winner_idx == 0:
        return 1.0, True
    if winner_idx == 1:
        return -1.0, True
    return -float(draw_penalty), False
