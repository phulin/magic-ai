"""Slot-encoder ``GameStateEncoder`` and its private helpers.

The shared TypedDicts, constants, and ``ParsedGameState``/``ParsedGameStateBatch``
dataclasses live in ``magic_ai.game_state``; only the encoder implementation
and its pure-function helpers live here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from magic_ai.game_state import (
    GAME_INFO_DIM,
    MANA_COLORS,
    MAX_CARDS_PER_ZONE,
    MAX_LIBRARY,
    MAX_LIFE,
    MAX_MANA,
    MAX_PENDING_OPTIONS,
    MAX_TURN,
    STEP_NAMES,
    ZONE_COUNT,
    ZONE_SLOT_COUNT,
    ZONE_SPECS,
    CardEmbeddingInput,
    GameCardState,
    GameStateSnapshot,
    ManaPoolState,
    ParsedGameState,
    ParsedGameStateBatch,
    PlayerState,
    ZoneName,
)


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

    def parse_state_batch(
        self,
        states: Sequence[GameStateSnapshot],
        perspective_player_indices: Sequence[int | None],
    ) -> ParsedGameStateBatch:
        n = len(states)
        slot_card_rows = torch.zeros((n, ZONE_SLOT_COUNT), dtype=torch.long)
        slot_occupied = torch.zeros((n, ZONE_SLOT_COUNT), dtype=torch.float32)
        slot_tapped = torch.zeros((n, ZONE_SLOT_COUNT), dtype=torch.float32)
        game_info = torch.zeros((n, GAME_INFO_DIM), dtype=torch.float32)
        card_id_to_slots: list[dict[str, int]] = []

        for batch_idx, (state, perspective_player_idx) in enumerate(
            zip(states, perspective_player_indices, strict=True)
        ):
            player_idx = self._resolve_perspective_player_idx(state, perspective_player_idx)
            players = state["players"]
            if not 0 <= player_idx < len(players):
                raise IndexError(f"player index {player_idx} outside players list")

            all_cards = self._collect_slot_cards(state, player_idx)
            occupied: list[float] = [0.0] * ZONE_SLOT_COUNT
            card_id_to_slot: dict[str, int] = {}

            for slot_idx, card in enumerate(all_cards):
                if card is None:
                    continue
                name = card.get("Name", "")
                slot_card_rows[batch_idx, slot_idx] = self._card_name_to_row.get(_card_key(name), 0)
                slot_occupied[batch_idx, slot_idx] = 1.0
                occupied[slot_idx] = 1.0
                is_bf = float(ZONE_SPECS[slot_idx // self.max_cards_per_zone][1] == "battlefield")
                if card.get("Tapped", False) and is_bf:
                    slot_tapped[batch_idx, slot_idx] = 1.0
                card_id = card.get("ID")
                if card_id:
                    card_id_to_slot[card_id] = slot_idx

            _fill_game_info(
                game_info[batch_idx],
                state,
                perspective_player_idx=player_idx,
                occupied=occupied,
            )
            card_id_to_slots.append(card_id_to_slot)

        return ParsedGameStateBatch(
            slot_card_rows=slot_card_rows,
            slot_occupied=slot_occupied,
            slot_tapped=slot_tapped,
            game_info=game_info,
            card_id_to_slots=card_id_to_slots,
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


def _fill_game_info(
    out: Tensor,
    state: GameStateSnapshot,
    *,
    perspective_player_idx: int,
    occupied: Sequence[float],
) -> None:
    players = state["players"]
    self_player = players[perspective_player_idx]
    opponent = players[1 - perspective_player_idx] if len(players) == 2 else None
    pending = state.get("pending")

    cursor = 0
    out[cursor] = _clip_norm(state.get("turn", 0), MAX_TURN)
    cursor += 1
    out[cursor] = float(_is_active_player(state, self_player))
    cursor += 1
    out[cursor] = float(pending is not None and pending.get("player_idx") == perspective_player_idx)
    cursor += 1
    out[cursor] = _clip_norm(self_player.get("Life", 0), MAX_LIFE)
    cursor += 1
    out[cursor] = _clip_norm(opponent.get("Life", 0), MAX_LIFE) if opponent else 0.0
    cursor += 1

    for player in (self_player, opponent):
        if player is None:
            out[cursor : cursor + 4] = 0.0
        else:
            out[cursor] = _clip_norm(player.get("HandCount", 0), MAX_CARDS_PER_ZONE)
            out[cursor + 1] = _clip_norm(player.get("GraveyardCount", 0), MAX_CARDS_PER_ZONE)
            out[cursor + 2] = _clip_norm(len(player.get("Battlefield") or []), MAX_CARDS_PER_ZONE)
            out[cursor + 3] = _clip_norm(player.get("LibraryCount", 0), MAX_LIBRARY)
        cursor += 4

    for player in (self_player, opponent):
        if player is None:
            out[cursor : cursor + len(MANA_COLORS)] = 0.0
        else:
            mana_pool = player.get("ManaPool", cast(ManaPoolState, {}))
            for offset, color in enumerate(MANA_COLORS):
                out[cursor + offset] = _clip_norm(mana_pool.get(color, 0), MAX_MANA)
        cursor += len(MANA_COLORS)

    option_count = len(pending.get("options", [])) if pending is not None else 0
    stack_count = len(state.get("stack") or [])
    out[cursor] = _clip_norm(option_count, MAX_PENDING_OPTIONS)
    out[cursor + 1] = _clip_norm(stack_count, MAX_PENDING_OPTIONS)
    cursor += 2

    normalized_step = " ".join(state.get("step", "").split()).casefold()
    step_idx = len(STEP_NAMES) - 1
    for idx, known_step in enumerate(STEP_NAMES[:-1]):
        if normalized_step == known_step.casefold():
            step_idx = idx
            break
    out[cursor : cursor + len(STEP_NAMES)] = 0.0
    out[cursor + step_idx] = 1.0
    cursor += len(STEP_NAMES)

    for offset, value in enumerate(occupied):
        out[cursor + offset] = value
    cursor += len(occupied)

    if cursor != GAME_INFO_DIM:
        raise RuntimeError(f"game_info dim {cursor} != {GAME_INFO_DIM}")


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
