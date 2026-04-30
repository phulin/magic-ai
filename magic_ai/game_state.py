"""Typed game-state format and shared constants for the Magic AI package.

Shared TypedDicts, constants, and ``ParsedGameState``/``ParsedGameStateBatch``
dataclasses used by both the slot and text encoder backends.
``GameStateEncoder`` and its private helpers live in
``magic_ai.slot_encoder.game_state``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from torch import Tensor

MAX_CARDS_PER_ZONE = 10
type ZoneName = Literal["hand", "graveyard", "battlefield"]
type ZoneOwner = Literal["self", "opponent"]
type ZoneSpec = tuple[str, ZoneName, ZoneOwner]

ZONE_SPECS: tuple[ZoneSpec, ...] = (
    ("self_hand", "hand", "self"),
    ("self_graveyard", "graveyard", "self"),
    ("opponent_graveyard", "graveyard", "opponent"),
    ("self_battlefield", "battlefield", "self"),
    ("opponent_battlefield", "battlefield", "opponent"),
)
ZONE_COUNT = len(ZONE_SPECS)
MANA_COLORS = ("White", "Blue", "Black", "Red", "Green", "Colorless")
MAX_LIFE = 40.0
MAX_TURN = 20.0
MAX_MANA = 10.0
MAX_LIBRARY = 60.0
MAX_PENDING_OPTIONS = 20.0

STEP_NAMES = (
    "Untap",
    "Upkeep",
    "Draw",
    "Precombat Main",
    "Begin Combat",
    "Declare Attackers",
    "Declare Blockers",
    "Combat Damage",
    "End Combat",
    "Postcombat Main",
    "End",
    "Cleanup",
    "Unknown",
)

ZONE_SLOT_COUNT = ZONE_COUNT * MAX_CARDS_PER_ZONE
GAME_INFO_DIM = (
    1  # turn
    + 1  # active player is perspective player
    + 1  # pending player is perspective player
    + 2  # life totals
    + 8  # hand/graveyard/battlefield/library counts for self and opponent
    + 12  # mana pools for self and opponent
    + 1  # pending option count
    + 1  # stack count
    + len(STEP_NAMES)
    + ZONE_SLOT_COUNT  # occupied masks
)


class ManaPoolState(TypedDict):
    White: int | float
    Blue: int | float
    Black: int | float
    Red: int | float
    Green: int | float
    Colorless: int | float


class GameCardState(TypedDict, total=False):
    ID: str
    Name: str
    Tapped: NotRequired[bool]


class PlayerState(TypedDict):
    ID: str
    Name: str
    Life: int | float
    HandCount: NotRequired[int]
    GraveyardCount: NotRequired[int]
    LibraryCount: NotRequired[int]
    Hand: NotRequired[list[GameCardState]]
    Graveyard: NotRequired[list[GameCardState]]
    Battlefield: NotRequired[list[GameCardState]]
    ManaPool: NotRequired[ManaPoolState]


class TargetState(TypedDict):
    id: str
    label: NotRequired[str]


class PendingOptionState(TypedDict, total=False):
    id: str
    kind: str
    label: str
    card_id: str
    card_name: str
    permanent_id: str
    ability_index: int
    mana_cost: str
    color: str
    valid_targets: list[TargetState]


class PendingState(TypedDict):
    kind: str
    player_idx: int
    options: list[PendingOptionState]
    amount: NotRequired[int]


class StackObjectState(TypedDict, total=False):
    id: str
    name: str


class GameStateSnapshot(TypedDict):
    turn: int
    active_player: str
    step: str
    players: list[PlayerState]
    pending: NotRequired[PendingState]
    stack: NotRequired[list[StackObjectState]]


type CardEmbeddingInput = Mapping[str, Sequence[float] | Tensor]


@dataclass(frozen=True)
class ParsedGameState:
    """Non-differentiable parsed representation of a game state.

    Fields are Python lists of integer/float indices — no tensors are allocated
    here. ``RolloutBuffer.ingest_batch`` does the bulk CPU→GPU copy into
    preallocated buffers; ``slot_encoder.GameStateEncoder.embed_slot_vectors`` then consumes
    the gathered tensors on-device.
    """

    slot_card_rows: list[int]  # length ZONE_SLOT_COUNT — row in card_embedding_table
    slot_occupied: list[float]  # length ZONE_SLOT_COUNT — 1.0 if slot has a card
    slot_tapped: list[float]  # length ZONE_SLOT_COUNT — 1.0 if occupied+battlefield+tapped
    game_info: list[float]  # length GAME_INFO_DIM
    card_id_to_slot: dict[str, int]  # used by ActionOptionsEncoder parsing


@dataclass(frozen=True)
class ParsedGameStateBatch:
    """Batched parsed game state fields for one actor forward."""

    slot_card_rows: Tensor  # [N, ZONE_SLOT_COUNT]
    slot_occupied: Tensor  # [N, ZONE_SLOT_COUNT]
    slot_tapped: Tensor  # [N, ZONE_SLOT_COUNT]
    game_info: Tensor  # [N, GAME_INFO_DIM]
    card_id_to_slots: list[dict[str, int]]


__all__ = [
    "GAME_INFO_DIM",
    "MANA_COLORS",
    "MAX_CARDS_PER_ZONE",
    "MAX_LIBRARY",
    "MAX_LIFE",
    "MAX_MANA",
    "MAX_PENDING_OPTIONS",
    "MAX_TURN",
    "STEP_NAMES",
    "ZONE_COUNT",
    "ZONE_SLOT_COUNT",
    "ZONE_SPECS",
    "CardEmbeddingInput",
    "GameCardState",
    "GameStateSnapshot",
    "ManaPoolState",
    "ParsedGameState",
    "ParsedGameStateBatch",
    "PendingOptionState",
    "PendingState",
    "PlayerState",
    "StackObjectState",
    "TargetState",
    "ZoneName",
    "ZoneOwner",
    "ZoneSpec",
]
