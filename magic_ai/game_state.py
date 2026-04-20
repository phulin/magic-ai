"""Typed game-state format and vector encoder for frozen card embeddings."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import torch
from torch import Tensor, nn

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
    ManaCost: str
    IsLand: bool
    Types: str
    SubTypes: str
    Power: int | float | str | None
    Toughness: int | float | str | None
    RulesText: str
    Tapped: bool
    SummonSick: bool
    IsCreature: bool
    IsArtifact: bool
    Attacking: bool
    Blocking: str
    Counters: Any
    Keywords: Any
    AttachedTo: str


class PlayerState(TypedDict):
    ID: str
    Name: str
    Life: int | float
    HandCount: int
    Hand: list[GameCardState] | None
    Battlefield: list[GameCardState] | None
    Graveyard: list[GameCardState] | None
    GraveyardCount: int
    ManaPool: ManaPoolState
    LibraryCount: int


class TargetState(TypedDict):
    id: str
    label: str


class PendingOptionState(TypedDict, total=False):
    kind: str
    label: str
    card_id: str
    card_name: str
    permanent_id: str
    ability_index: int
    mana_cost: str
    valid_targets: list[TargetState]
    id: str
    color: str


class PendingState(TypedDict):
    kind: str
    player_idx: int
    options: list[PendingOptionState]
    reason: NotRequired[str]
    amount: NotRequired[int]


class StackObjectState(TypedDict, total=False):
    ID: str
    Name: str
    controller: str
    targets: list[TargetState]


class GameStateSnapshot(TypedDict):
    turn: int
    step: str
    active_player: str
    players: list[PlayerState]
    stack: list[StackObjectState] | None
    pending: NotRequired[PendingState | None]


CardEmbeddingInput = Mapping[str, Sequence[float] | Tensor]


class GameStateEncoder(nn.Module):
    """Encode one engine game-state snapshot into a fixed MLP-ready vector.

    Card embeddings are treated as frozen inputs. The trainable parts are the
    projection into `d_model`, positional/zone vectors, the empty slot vector,
    and the tapped vector. Hidden information is perspective-based: only the
    perspective player's hand is encoded, while both players' graveyards and
    battlefields are encoded.
    """

    game_info_dim = GAME_INFO_DIM

    def __init__(
        self,
        card_embeddings: CardEmbeddingInput,
        *,
        raw_embedding_dim: int | None = None,
        d_model: int = 128,
        max_cards_per_zone: int = MAX_CARDS_PER_ZONE,
    ) -> None:
        super().__init__()
        if max_cards_per_zone != MAX_CARDS_PER_ZONE:
            raise ValueError(f"only max_cards_per_zone={MAX_CARDS_PER_ZONE} is currently supported")

        self.d_model = d_model
        self.max_cards_per_zone = max_cards_per_zone
        self.raw_embedding_dim = raw_embedding_dim or _infer_embedding_dim(card_embeddings)

        self.card_projection = nn.Linear(self.raw_embedding_dim, d_model)
        self.zone_embedding = nn.Embedding(ZONE_COUNT, d_model)
        self.slot_embedding = nn.Embedding(max_cards_per_zone, d_model)
        self.empty_slot_vector = nn.Parameter(torch.zeros(d_model))
        self.tapped_vector = nn.Parameter(torch.randn(d_model) * 0.02)
        self.register_buffer("unknown_card_vector", torch.zeros(self.raw_embedding_dim))

        self._card_embeddings = _normalize_card_embedding_map(
            card_embeddings,
            raw_embedding_dim=self.raw_embedding_dim,
        )

    @property
    def output_dim(self) -> int:
        return ZONE_SLOT_COUNT * self.d_model + GAME_INFO_DIM

    @classmethod
    def from_embedding_json(
        cls,
        path: str | Path,
        *,
        d_model: int = 128,
    ) -> GameStateEncoder:
        """Load embeddings produced by scripts/build_card_embeddings.py."""

        payload = json.loads(Path(path).read_text())
        embeddings: dict[str, Sequence[float]] = {}
        for record in payload.get("cards", []):
            name = record.get("name")
            embedding = record.get("embedding")
            if name and embedding is not None:
                embeddings[name] = embedding
        return cls(embeddings, d_model=d_model)

    def forward(
        self,
        states: GameStateSnapshot | Sequence[GameStateSnapshot],
        *,
        perspective_player_idx: int | Sequence[int] | None = None,
    ) -> Tensor:
        """Encode a single state or a batch of states.

        If `perspective_player_idx` is omitted, the encoder uses
        `state["pending"]["player_idx"]` when available, otherwise the active
        player, otherwise player 0.
        """

        if isinstance(states, Mapping):
            idx = cast(int | None, perspective_player_idx)
            return self.encode_state(cast(GameStateSnapshot, states), idx)

        batch_states = list(states)
        if perspective_player_idx is None:
            indices: list[int | None] = [None] * len(batch_states)
        elif isinstance(perspective_player_idx, int):
            indices = [perspective_player_idx] * len(batch_states)
        else:
            indices = list(perspective_player_idx)
            if len(indices) != len(batch_states):
                raise ValueError("perspective_player_idx length must match states length")

        encoded = [
            self.encode_state(state, player_idx)
            for state, player_idx in zip(batch_states, indices, strict=True)
        ]
        return torch.stack(encoded, dim=0)

    def encode_state(
        self,
        state: GameStateSnapshot,
        perspective_player_idx: int | None = None,
    ) -> Tensor:
        device = self.empty_slot_vector.device
        player_idx = self._resolve_perspective_player_idx(state, perspective_player_idx)
        players = state["players"]
        if not 0 <= player_idx < len(players):
            raise IndexError(f"player index {player_idx} outside players list")

        slot_vectors: list[Tensor] = []
        occupied: list[float] = []
        player = players[player_idx]

        opponent = players[1 - player_idx] if len(players) == 2 else None

        for zone_idx, (_zone_name, zone, owner) in enumerate(ZONE_SPECS):
            zone_player = player if owner == "self" else opponent
            cards = _zone_cards(zone_player, zone)
            for slot_idx in range(self.max_cards_per_zone):
                card = cards[slot_idx] if slot_idx < len(cards) else None
                is_occupied = card is not None
                occupied.append(float(is_occupied))
                slot_vectors.append(
                    self._encode_slot(
                        card,
                        zone_idx=zone_idx,
                        slot_idx=slot_idx,
                        is_occupied=is_occupied,
                    )
                )

        card_features = torch.stack(slot_vectors, dim=0).flatten()
        game_info = self._build_game_info(
            state,
            perspective_player_idx=player_idx,
            occupied=occupied,
            device=device,
        )
        return torch.cat([card_features, game_info], dim=0)

    def _encode_slot(
        self,
        card: GameCardState | None,
        *,
        zone_idx: int,
        slot_idx: int,
        is_occupied: bool,
    ) -> Tensor:
        device = self.empty_slot_vector.device
        if is_occupied and card is not None:
            raw_embedding = self._lookup_card_embedding(card)
            slot_vector = self.card_projection(raw_embedding)
        else:
            slot_vector = self.empty_slot_vector

        zone_id = torch.tensor(zone_idx, device=device)
        slot_id = torch.tensor(slot_idx, device=device)
        slot_vector = slot_vector + self.zone_embedding(zone_id) + self.slot_embedding(slot_id)

        if is_occupied and ZONE_SPECS[zone_idx][1] == "battlefield" and card is not None:
            tapped = 1.0 if card.get("Tapped", False) else 0.0
            slot_vector = slot_vector + self.tapped_vector * tapped

        return slot_vector

    def _lookup_card_embedding(self, card: GameCardState) -> Tensor:
        name = card.get("Name", "")
        embedding = self._card_embeddings.get(_card_key(name))
        if embedding is None:
            return cast(Tensor, self.unknown_card_vector)
        return embedding.to(self.empty_slot_vector.device)

    def _resolve_perspective_player_idx(
        self,
        state: GameStateSnapshot,
        perspective_player_idx: int | None,
    ) -> int:
        if perspective_player_idx is not None:
            return perspective_player_idx

        pending = state.get("pending")
        if pending is not None:
            return int(pending.get("player_idx", 0))

        active_player = state.get("active_player")
        for idx, player in enumerate(state["players"]):
            if player.get("Name") == active_player or player.get("ID") == active_player:
                return idx
        return 0

    def _build_game_info(
        self,
        state: GameStateSnapshot,
        *,
        perspective_player_idx: int,
        occupied: Sequence[float],
        device: torch.device,
    ) -> Tensor:
        players = state["players"]
        self_player = players[perspective_player_idx]
        opponent = players[1 - perspective_player_idx] if len(players) == 2 else None
        pending = state.get("pending")

        values: list[float] = [
            _clip_norm(state.get("turn", 0), MAX_TURN),
            float(_is_active_player(state, self_player)),
            float(pending is not None and pending.get("player_idx") == perspective_player_idx),
            _clip_norm(self_player.get("Life", 0), MAX_LIFE),
            _clip_norm(opponent.get("Life", 0), MAX_LIFE) if opponent else 0.0,
        ]

        for player in (self_player, opponent):
            values.extend(_player_count_features(player))

        for player in (self_player, opponent):
            values.extend(_mana_pool_features(player))

        option_count = len(pending.get("options", [])) if pending is not None else 0
        stack = state.get("stack") or []
        stack_count = len(stack)
        values.extend(
            [
                _clip_norm(option_count, MAX_PENDING_OPTIONS),
                _clip_norm(stack_count, MAX_PENDING_OPTIONS),
            ]
        )
        values.extend(_one_hot_step(state.get("step", "")))
        values.extend(occupied)

        if len(values) != GAME_INFO_DIM:
            raise RuntimeError(f"game_info dim {len(values)} != {GAME_INFO_DIM}")
        return torch.tensor(values, dtype=torch.float32, device=device)


def _normalize_card_embedding_map(
    card_embeddings: CardEmbeddingInput,
    *,
    raw_embedding_dim: int,
) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    for name, embedding in card_embeddings.items():
        tensor = torch.as_tensor(embedding, dtype=torch.float32).detach()
        if tensor.ndim != 1:
            raise ValueError(f"embedding for {name!r} must be 1-dimensional")
        if tensor.shape[0] != raw_embedding_dim:
            raise ValueError(
                f"embedding for {name!r} has dim {tensor.shape[0]}, expected {raw_embedding_dim}"
            )
        normalized[_card_key(name)] = tensor
    return normalized


def _infer_embedding_dim(card_embeddings: CardEmbeddingInput) -> int:
    for embedding in card_embeddings.values():
        return int(torch.as_tensor(embedding).numel())
    raise ValueError("card_embeddings must contain at least one card")


def _card_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def _zone_cards(
    player: PlayerState | None,
    zone: ZoneName,
) -> list[GameCardState]:
    if player is None:
        return []
    if zone == "hand":
        cards = player.get("Hand")
    elif zone == "graveyard":
        cards = player.get("Graveyard")
    else:
        cards = player.get("Battlefield")
    return list(cards or [])[:MAX_CARDS_PER_ZONE]


def _clip_norm(value: int | float | None, maximum: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), maximum)) / maximum


def _player_count_features(player: PlayerState | None) -> list[float]:
    if player is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        _clip_norm(player.get("HandCount", 0), MAX_CARDS_PER_ZONE),
        _clip_norm(player.get("GraveyardCount", 0), MAX_CARDS_PER_ZONE),
        _clip_norm(len(player.get("Battlefield") or []), MAX_CARDS_PER_ZONE),
        _clip_norm(player.get("LibraryCount", 0), MAX_LIBRARY),
    ]


def _mana_pool_features(player: PlayerState | None) -> list[float]:
    if player is None:
        return [0.0] * len(MANA_COLORS)
    mana_pool = player.get("ManaPool", cast(ManaPoolState, {}))
    return [_clip_norm(mana_pool.get(color, 0), MAX_MANA) for color in MANA_COLORS]


def _is_active_player(state: GameStateSnapshot, player: PlayerState) -> bool:
    active_player = state.get("active_player")
    return active_player == player.get("Name") or active_player == player.get("ID")


def _one_hot_step(step: str) -> list[float]:
    values = [0.0] * len(STEP_NAMES)
    normalized = " ".join(step.split()).casefold()
    for idx, known_step in enumerate(STEP_NAMES[:-1]):
        if normalized == known_step.casefold():
            values[idx] = 1.0
            return values
    values[-1] = 1.0
    return values
