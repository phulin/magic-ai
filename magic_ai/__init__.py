"""Utilities for Magic AI experiments."""

from magic_ai.actions import (
    ActionOptionsEncoder,
    ActionRequest,
    ActionTrace,
    EncodedSelectedAction,
    LegalActionCandidate,
    ParsedActionInputs,
    ParsedStep,
    PolicyStep,
    SelectedActionEncoder,
    action_from_attackers,
    action_from_blockers,
    action_from_choice_accepted,
    action_from_choice_color,
    action_from_choice_ids,
    action_from_choice_index,
    action_from_priority_candidate,
    build_priority_candidates,
    selected_priority_candidate_index,
)
from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    ManaPoolState,
    ParsedGameState,
    PendingOptionState,
    PendingState,
    PlayerState,
)
from magic_ai.ppo import (
    PPOStats,
    RolloutStep,
    gae_returns,
    ppo_update,
)
from magic_ai.slot_encoder.buffer import NativeTrajectoryBuffer, RolloutBuffer
from magic_ai.slot_encoder.game_state import GameStateEncoder
from magic_ai.slot_encoder.model import (
    PPOPolicy,
)
from magic_ai.slot_encoder.native_encoder import (
    NativeBatchEncoder,
    NativeEncodedBatch,
    NativeEncodingError,
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
    "ppo_update",
    "selected_priority_candidate_index",
]
