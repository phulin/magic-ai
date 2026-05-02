"""Shared model-state key classification for snapshots and runtime buffers."""

from __future__ import annotations

ACTOR_RUNTIME_STATE_PREFIXES: tuple[str, ...] = (
    "live_lstm_h",
    "live_lstm_c",
    "rollout_buffer.",
)

OPPONENT_EXCLUDED_STATE_PREFIXES: tuple[str, ...] = (
    "target_",
    "spr_",
    *ACTOR_RUNTIME_STATE_PREFIXES,
)


def is_actor_runtime_state_key(name: str) -> bool:
    """Return True for per-actor runtime buffers that are not model state."""

    return name.startswith(ACTOR_RUNTIME_STATE_PREFIXES)


def is_opponent_policy_state_key(name: str) -> bool:
    """Return True when a state-dict key belongs in frozen opponent snapshots."""

    return not name.startswith(OPPONENT_EXCLUDED_STATE_PREFIXES)


__all__ = [
    "ACTOR_RUNTIME_STATE_PREFIXES",
    "OPPONENT_EXCLUDED_STATE_PREFIXES",
    "is_actor_runtime_state_key",
    "is_opponent_policy_state_key",
]
