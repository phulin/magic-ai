"""Typed game-state format and vector encoder for frozen card embeddings."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

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
    preallocated buffers; ``GameStateEncoder.embed_slot_vectors`` then consumes
    the gathered tensors on-device.
    """

    slot_card_rows: list[int]  # length ZONE_SLOT_COUNT — row in card_embedding_table
    slot_occupied: list[float]  # length ZONE_SLOT_COUNT — 1.0 if slot has a card
    slot_tapped: list[float]  # length ZONE_SLOT_COUNT — 1.0 if occupied+battlefield+tapped
    game_info: list[float]  # length GAME_INFO_DIM
    card_id_to_slot: dict[str, int]  # used by ActionOptionsEncoder parsing


class GameStateEncoder(nn.Module):
    """Encode one engine game-state snapshot into a fixed MLP-ready vector.

    Raw card embeddings live on the module's device as a single
    ``card_embedding_table`` buffer (row 0 is the unknown/zero vector). All
    per-step parsing produces integer indices into that table; the actual
    vector lookup and projection happens on-device inside
    ``embed_slot_vectors`` so trainable parameters (``card_projection``,
    zone/slot embeddings, ``empty_slot_vector``, ``tapped_vector``) see
    gradients during PPO updates.
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

        table, name_to_row = _build_card_embedding_table(
            card_embeddings,
            raw_embedding_dim=self.raw_embedding_dim,
        )
        self.card_embedding_table: Tensor
        self.register_buffer("card_embedding_table", table)
        self._card_name_to_row = name_to_row

        # Pre-computed index tensors for batched slot encoding.
        zone_ids: list[int] = []
        slot_ids: list[int] = []
        is_bf: list[float] = []
        for _zone_idx in range(ZONE_COUNT):
            for _slot_idx in range(max_cards_per_zone):
                zone_ids.append(_zone_idx)
                slot_ids.append(_slot_idx)
                is_bf.append(float(ZONE_SPECS[_zone_idx][1] == "battlefield"))
        self._zone_ids: Tensor
        self.register_buffer(
            "_zone_ids", torch.tensor(zone_ids, dtype=torch.long), persistent=False
        )
        self._slot_ids: Tensor
        self.register_buffer(
            "_slot_ids", torch.tensor(slot_ids, dtype=torch.long), persistent=False
        )
        self._is_battlefield: Tensor
        self.register_buffer(
            "_is_battlefield", torch.tensor(is_bf, dtype=torch.float32), persistent=False
        )

    @property
    def output_dim(self) -> int:
        return ZONE_SLOT_COUNT * self.d_model + GAME_INFO_DIM

    @property
    def device(self) -> torch.device:
        return self.empty_slot_vector.device

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

    def parse_state(
        self,
        state: GameStateSnapshot,
        perspective_player_idx: int | None = None,
    ) -> ParsedGameState:
        """Parse a state dict into Python-native index lists.

        Pure Python + dict traversal. No tensors are allocated — downstream
        bulk ingest into ``RolloutBuffer`` handles the CPU→GPU copy.
        """

        player_idx = self._resolve_perspective_player_idx(state, perspective_player_idx)
        players = state["players"]
        if not 0 <= player_idx < len(players):
            raise IndexError(f"player index {player_idx} outside players list")

        all_cards = self._collect_slot_cards(state, player_idx)
        card_rows: list[int] = []
        occupied: list[float] = []
        tapped: list[float] = []
        card_id_to_slot: dict[str, int] = {}

        for slot_idx, card in enumerate(all_cards):
            if card is None:
                card_rows.append(0)
                occupied.append(0.0)
                tapped.append(0.0)
                continue
            name = card.get("Name", "")
            card_rows.append(self._card_name_to_row.get(_card_key(name), 0))
            occupied.append(1.0)
            is_bf = float(ZONE_SPECS[slot_idx // self.max_cards_per_zone][1] == "battlefield")
            tapped.append(1.0 if (card.get("Tapped", False) and is_bf) else 0.0)
            card_id = card.get("ID")
            if card_id:
                card_id_to_slot[card_id] = slot_idx

        game_info = self._build_game_info(
            state,
            perspective_player_idx=player_idx,
            occupied=occupied,
        )
        return ParsedGameState(
            slot_card_rows=card_rows,
            slot_occupied=occupied,
            slot_tapped=tapped,
            game_info=game_info,
            card_id_to_slot=card_id_to_slot,
        )

    def embed_slot_vectors(
        self,
        slot_card_rows: Tensor,
        slot_occupied: Tensor,
        slot_tapped: Tensor,
    ) -> Tensor:
        """Compute slot vectors from parsed indices.

        Accepts ``[..., ZONE_SLOT_COUNT]`` index tensors and returns
        ``[..., ZONE_SLOT_COUNT, d_model]``. All trainable params on this path
        (``card_projection``, zone/slot embeddings, ``empty_slot_vector``,
        ``tapped_vector``) receive gradients.
        """

        raw = self.card_embedding_table[slot_card_rows]
        projected = self.card_projection(raw)
        occupied_mask = slot_occupied.unsqueeze(-1) > 0
        empty = self.empty_slot_vector.expand_as(projected)
        slot_vectors = torch.where(occupied_mask, projected, empty)
        slot_vectors = slot_vectors + slot_tapped.unsqueeze(-1) * self.tapped_vector
        position = self.zone_embedding(self._zone_ids) + self.slot_embedding(self._slot_ids)
        return slot_vectors + position

    def state_vector_from_slots(self, slot_vectors: Tensor, game_info: Tensor) -> Tensor:
        """Flatten slot vectors and concatenate with the parsed scalar game info."""

        card_features = slot_vectors.flatten(-2, -1)
        return torch.cat([card_features, game_info], dim=-1)

    def lookup_card_row(self, name: str) -> int:
        """Return the card_embedding_table row for a card name (0 if unknown)."""

        return self._card_name_to_row.get(_card_key(name), 0)

    def _collect_slot_cards(
        self,
        state: GameStateSnapshot,
        perspective_player_idx: int,
    ) -> list[GameCardState | None]:
        """Gather the card (or None) for every zone-slot in spec order."""

        players = state["players"]
        player = players[perspective_player_idx]
        opponent = players[1 - perspective_player_idx] if len(players) == 2 else None
        all_cards: list[GameCardState | None] = []
        for _zone_name, zone, owner in ZONE_SPECS:
            zone_player = player if owner == "self" else opponent
            cards = _zone_cards(zone_player, zone)
            for slot_idx in range(self.max_cards_per_zone):
                all_cards.append(cards[slot_idx] if slot_idx < len(cards) else None)
        return all_cards

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
    ) -> list[float]:
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
        return values


def _build_card_embedding_table(
    card_embeddings: CardEmbeddingInput,
    *,
    raw_embedding_dim: int,
) -> tuple[Tensor, dict[str, int]]:
    """Pack all frozen card embeddings into one dense table.

    Row 0 is the zero / unknown-card vector. Cards get rows 1..N in the order
    they appear in ``card_embeddings``.
    """

    rows: list[Tensor] = [torch.zeros(raw_embedding_dim, dtype=torch.float32)]
    name_to_row: dict[str, int] = {}
    for name, embedding in card_embeddings.items():
        tensor = torch.as_tensor(embedding, dtype=torch.float32).detach()
        if tensor.ndim != 1:
            raise ValueError(f"embedding for {name!r} must be 1-dimensional")
        if tensor.shape[0] != raw_embedding_dim:
            raise ValueError(
                f"embedding for {name!r} has dim {tensor.shape[0]}, expected {raw_embedding_dim}"
            )
        name_to_row[_card_key(name)] = len(rows)
        rows.append(tensor)
    table = torch.stack(rows, dim=0)
    return table, name_to_row


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
