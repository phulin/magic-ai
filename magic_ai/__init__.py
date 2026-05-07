"""Utilities for Magic AI experiments."""

from __future__ import annotations

import os as _os
from importlib import import_module as _import_module
from pathlib import Path as _Path

_os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(_Path.home() / ".cache" / "magic-ai" / "inductor"),
)
# Also direct dynamo's eval-frame cache (smaller, but worth keeping warm).
_os.environ.setdefault(
    "TORCH_COMPILE_DEBUG_DIR",
    str(_Path.home() / ".cache" / "magic-ai" / "compile-debug"),
)

__all__ = [
    "ActionOptionsEncoder",
    "ActionRequest",
    "ActionTrace",
    "EncodedSelectedAction",
    "GameCardState",
    "GameStateEncoder",
    "GameStateSnapshot",
    "LegalActionCandidate",
    "ManaPoolState",
    "NativeBatchEncoder",
    "NativeEncodedBatch",
    "NativeEncodingError",
    "NativeTrajectoryBuffer",
    "PPOPolicy",
    "PPOStats",
    "ParsedActionInputs",
    "ParsedGameState",
    "ParsedStep",
    "RolloutBuffer",
    "PendingOptionState",
    "PendingState",
    "PlayerState",
    "PolicyStep",
    "RolloutStep",
    "SelectedActionEncoder",
    "action_from_attackers",
    "action_from_blockers",
    "action_from_choice_accepted",
    "action_from_choice_color",
    "action_from_choice_ids",
    "action_from_choice_index",
    "action_from_priority_candidate",
    "build_priority_candidates",
    "gae_returns",
    "gae_returns_batched",
    "ppo_update",
    "selected_priority_candidate_index",
]

_LAZY_ATTR_TO_MODULE = {
    "ActionOptionsEncoder": "magic_ai.actions",
    "ActionRequest": "magic_ai.actions",
    "ActionTrace": "magic_ai.actions",
    "EncodedSelectedAction": "magic_ai.actions",
    "GameCardState": "magic_ai.game_state",
    "GameStateEncoder": "magic_ai.slot_encoder.game_state",
    "GameStateSnapshot": "magic_ai.game_state",
    "LegalActionCandidate": "magic_ai.actions",
    "ManaPoolState": "magic_ai.game_state",
    "NativeBatchEncoder": "magic_ai.slot_encoder.native_encoder",
    "NativeEncodedBatch": "magic_ai.slot_encoder.native_encoder",
    "NativeEncodingError": "magic_ai.slot_encoder.native_encoder",
    "NativeTrajectoryBuffer": "magic_ai.slot_encoder.buffer",
    "PPOPolicy": "magic_ai.slot_encoder.model",
    "PPOStats": "magic_ai.ppo",
    "ParsedActionInputs": "magic_ai.actions",
    "ParsedGameState": "magic_ai.game_state",
    "ParsedStep": "magic_ai.actions",
    "PendingOptionState": "magic_ai.game_state",
    "PendingState": "magic_ai.game_state",
    "PlayerState": "magic_ai.game_state",
    "PolicyStep": "magic_ai.actions",
    "RolloutBuffer": "magic_ai.slot_encoder.buffer",
    "RolloutStep": "magic_ai.ppo",
    "SelectedActionEncoder": "magic_ai.actions",
    "action_from_attackers": "magic_ai.actions",
    "action_from_blockers": "magic_ai.actions",
    "action_from_choice_accepted": "magic_ai.actions",
    "action_from_choice_color": "magic_ai.actions",
    "action_from_choice_ids": "magic_ai.actions",
    "action_from_choice_index": "magic_ai.actions",
    "action_from_priority_candidate": "magic_ai.actions",
    "build_priority_candidates": "magic_ai.actions",
    "gae_returns": "magic_ai.ppo",
    "gae_returns_batched": "magic_ai.ppo",
    "ppo_update": "magic_ai.ppo",
    "selected_priority_candidate_index": "magic_ai.actions",
}


def __getattr__(name: str):
    module_name = _LAZY_ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_name), name)
    globals()[name] = value
    return value
