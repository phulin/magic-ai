"""Action-option encoding and selected-action decoding for mage-go pylib."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import torch
from torch import Tensor, nn

from magic_ai.game_state import (
    GameCardState,
    GameStateEncoder,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
    PlayerState,
)

DEFAULT_MAX_OPTIONS = 64
DEFAULT_MAX_TARGETS_PER_OPTION = 4
MAX_ABILITY_INDEX = 8.0
MAX_AMOUNT = 20.0
MAX_TARGET_OVERFLOW = 32.0
MANA_SYMBOLS = ("W", "U", "B", "R", "G", "C")
ACTION_KINDS = (
    "pass",
    "play_land",
    "cast_spell",
    "activate_ability",
    "attacker",
    "blocker",
    "choice",
    "unknown",
)
PENDING_KINDS = (
    "priority",
    "attackers",
    "blockers",
    "permanent",
    "cards_from_hand",
    "mana_color",
    "card_from_library",
    "may",
    "mode",
    "number",
    "unknown",
)
TARGET_TYPES = ("player", "permanent", "card", "unknown", "empty")
COLORS = ("white", "blue", "black", "red", "green", "colorless")
OPTION_SCALAR_DIM = 14
TARGET_SCALAR_DIM = 2


class BlockerAssignState(TypedDict):
    blocker: str
    attacker: str


class ActionRequest(TypedDict, total=False):
    kind: str
    card_id: str
    permanent_id: str
    ability_index: int
    targets: list[str]
    x: int
    attackers: list[str]
    blockers: list[BlockerAssignState]
    selected_ids: list[str]
    selected_index: int
    selected_color: str
    accepted: bool


class OptionTargetState(TypedDict):
    id: str
    label: str


class EncodedActionOptions(TypedDict):
    pending_vector: Tensor
    option_vectors: Tensor
    option_mask: Tensor
    target_vectors: Tensor
    target_mask: Tensor
    target_overflow: Tensor
    priority_candidates: list[LegalActionCandidate]


class EncodedSelectedAction(TypedDict):
    priority_candidate_index: Tensor
    attacker_labels: Tensor
    blocker_labels: Tensor
    choice_selected_index: Tensor
    choice_selected_ids: Tensor
    choice_selected_color: Tensor
    choice_accepted: Tensor
    x_value: Tensor


@dataclass(frozen=True)
class LegalActionCandidate:
    """A flat legal priority action candidate.

    `payload` is already in the JSON shape accepted by mage-go's MageStep.
    """

    option_index: int
    kind: str
    payload: ActionRequest
    target_index: int | None = None

    def to_action_request(self) -> ActionRequest:
        return _copy_action_request(self.payload)


class SelectedActionEncoder:
    """Encode a selected MageStep action into supervised labels.

    The labels line up with `ActionOptionsEncoder`'s option/target dimensions.
    Irrelevant scalar labels use -100 so they can be passed to PyTorch losses
    with `ignore_index=-100`.
    """

    def __init__(
        self,
        *,
        max_options: int = DEFAULT_MAX_OPTIONS,
        max_targets_per_option: int = DEFAULT_MAX_TARGETS_PER_OPTION,
    ) -> None:
        self.max_options = max_options
        self.max_targets_per_option = max_targets_per_option

    def encode(
        self,
        pending: PendingState,
        action: ActionRequest,
    ) -> EncodedSelectedAction:
        pending_kind = pending.get("kind", "")
        priority_idx = -100
        attacker_labels = torch.zeros(self.max_options, dtype=torch.float32)
        blocker_labels = torch.full((self.max_options,), -100, dtype=torch.long)
        choice_selected_ids = torch.zeros(self.max_options, dtype=torch.float32)
        choice_selected_index = -100
        choice_selected_color = -100
        choice_accepted = -100.0

        if pending_kind == "priority":
            priority_idx = selected_priority_candidate_index(
                pending,
                action,
                max_targets_per_option=self.max_targets_per_option,
            )
        elif pending_kind == "attackers":
            selected_attackers = set(action.get("attackers", []))
            for idx, option in enumerate(pending.get("options", [])[: self.max_options]):
                attacker_labels[idx] = float(option.get("permanent_id") in selected_attackers)
        elif pending_kind == "blockers":
            blocker_labels.fill_(0)
            assignments = {
                item.get("blocker", ""): item.get("attacker", "")
                for item in action.get("blockers", [])
            }
            for idx, option in enumerate(pending.get("options", [])[: self.max_options]):
                blocker_id = option.get("permanent_id", "")
                attacker_id = assignments.get(blocker_id)
                if not attacker_id:
                    continue
                targets = option.get("valid_targets", [])[: self.max_targets_per_option]
                for target_idx, target in enumerate(targets):
                    if target.get("id") == attacker_id:
                        blocker_labels[idx] = target_idx + 1
                        break
        else:
            if "selected_index" in action:
                choice_selected_index = int(action["selected_index"])
            if "selected_color" in action:
                choice_selected_color = _color_index(action["selected_color"])
            if "accepted" in action:
                choice_accepted = 1.0 if action["accepted"] else 0.0
            selected_ids = set(action.get("selected_ids", []))
            for idx, option in enumerate(pending.get("options", [])[: self.max_options]):
                choice_selected_ids[idx] = float(option.get("id") in selected_ids)

        return {
            "priority_candidate_index": torch.tensor(priority_idx, dtype=torch.long),
            "attacker_labels": attacker_labels,
            "blocker_labels": blocker_labels,
            "choice_selected_index": torch.tensor(choice_selected_index, dtype=torch.long),
            "choice_selected_ids": choice_selected_ids,
            "choice_selected_color": torch.tensor(choice_selected_color, dtype=torch.long),
            "choice_accepted": torch.tensor(choice_accepted, dtype=torch.float32),
            "x_value": torch.tensor(float(action.get("x", 0)), dtype=torch.float32),
        }


class ActionOptionsEncoder(nn.Module):
    """Encode mage-go `pending` action options into fixed tensors.

    This encoder does not generate arbitrary action JSON. It encodes legal
    options and target choices, and helper decoders turn selected indices back
    into the exact action-request format expected by MageStep.
    """

    def __init__(
        self,
        game_state_encoder: GameStateEncoder,
        *,
        max_options: int = DEFAULT_MAX_OPTIONS,
        max_targets_per_option: int = DEFAULT_MAX_TARGETS_PER_OPTION,
    ) -> None:
        super().__init__()
        self.game_state_encoder = game_state_encoder
        self.d_action = game_state_encoder.d_model
        self.max_options = max_options
        self.max_targets_per_option = max_targets_per_option

        self.pending_kind_embedding = nn.Embedding(len(PENDING_KINDS), self.d_action)
        self.option_kind_embedding = nn.Embedding(len(ACTION_KINDS), self.d_action)
        self.target_type_embedding = nn.Embedding(len(TARGET_TYPES), self.d_action)
        self.option_scalar_projection = nn.Linear(OPTION_SCALAR_DIM, self.d_action)
        self.target_scalar_projection = nn.Linear(TARGET_SCALAR_DIM, self.d_action)
        self.empty_option_vector = nn.Parameter(torch.zeros(self.d_action))
        self.empty_target_vector = nn.Parameter(torch.zeros(self.d_action))

    def forward(
        self,
        state: GameStateSnapshot,
        pending: PendingState | None = None,
        *,
        perspective_player_idx: int | None = None,
        precomputed_object_vectors: dict[str, Tensor] | None = None,
    ) -> EncodedActionOptions:
        return self.encode_pending(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
            precomputed_object_vectors=precomputed_object_vectors,
        )

    def encode_pending(
        self,
        state: GameStateSnapshot,
        pending: PendingState | None = None,
        *,
        perspective_player_idx: int | None = None,
        precomputed_object_vectors: dict[str, Tensor] | None = None,
    ) -> EncodedActionOptions:
        pending = pending if pending is not None else state.get("pending")
        if pending is None:
            raise ValueError("state has no pending action request")

        device = self.empty_option_vector.device
        player_idx = self.game_state_encoder._resolve_perspective_player_idx(
            state,
            perspective_player_idx,
        )

        if precomputed_object_vectors is not None:
            object_vectors = precomputed_object_vectors
            player_vectors = self._build_player_vectors(state, player_idx)
        else:
            object_vectors, player_vectors = self._build_reference_vectors(state, player_idx)

        pending_kind_id = torch.tensor(
            _index_or_unknown(PENDING_KINDS, pending.get("kind", "")),
            device=device,
        )
        pending_vector = self.pending_kind_embedding(pending_kind_id)

        options = pending.get("options", [])
        num_present = min(len(options), self.max_options)

        # --- Batch encode present options ---
        option_vectors_t = (
            self.empty_option_vector.unsqueeze(0).expand(self.max_options, -1).clone()
        )

        if num_present > 0:
            kind_ids = torch.tensor(
                [
                    _index_or_unknown(ACTION_KINDS, options[i].get("kind", "unknown"))
                    for i in range(num_present)
                ],
                dtype=torch.long,
                device=device,
            )
            all_scalars = torch.tensor(
                [
                    _option_scalars(
                        options[i],
                        pending=pending,
                        option_idx=i,
                        max_options=self.max_options,
                        max_targets_per_option=self.max_targets_per_option,
                    )
                    for i in range(num_present)
                ],
                dtype=torch.float32,
                device=device,
            )
            ref_vecs = torch.stack(
                [
                    self._option_reference_vector(options[i], object_vectors)
                    for i in range(num_present)
                ]
            )
            option_vectors_t[:num_present] = (
                pending_vector
                + self.option_kind_embedding(kind_ids)
                + ref_vecs
                + self.option_scalar_projection(all_scalars)
            )

        option_mask = torch.zeros(self.max_options, dtype=torch.float32, device=device)
        option_mask[:num_present] = 1.0

        # --- Batch encode targets ---
        tgt_opt_indices: list[int] = []
        tgt_slot_indices: list[int] = []
        tgt_type_ids: list[int] = []
        tgt_scalars_list: list[list[float]] = []
        tgt_refs: list[Tensor] = []
        target_overflow = torch.zeros(self.max_options, dtype=torch.float32, device=device)
        target_mask = torch.zeros(
            self.max_options, self.max_targets_per_option, dtype=torch.float32, device=device
        )
        max_tgt_scalar = max(1.0, float(self.max_targets_per_option - 1))

        for opt_i in range(num_present):
            targets = options[opt_i].get("valid_targets", [])
            overflow_count = max(0, len(targets) - self.max_targets_per_option)
            target_overflow[opt_i] = _clip_norm(overflow_count, MAX_TARGET_OVERFLOW)

            for tgt_j in range(min(len(targets), self.max_targets_per_option)):
                target = targets[tgt_j]
                target_mask[opt_i, tgt_j] = 1.0
                tgt_opt_indices.append(opt_i)
                tgt_slot_indices.append(tgt_j)

                target_id = target.get("id", "")
                if target_id in player_vectors:
                    tgt_type_ids.append(_index_or_unknown(TARGET_TYPES, "player"))
                    tgt_refs.append(player_vectors[target_id])
                elif target_id in object_vectors:
                    tgt_type_ids.append(_index_or_unknown(TARGET_TYPES, "permanent"))
                    tgt_refs.append(object_vectors[target_id])
                else:
                    tgt_type_ids.append(_index_or_unknown(TARGET_TYPES, "unknown"))
                    tgt_refs.append(torch.zeros(self.d_action, device=device))

                tgt_scalars_list.append([_clip_norm(tgt_j, max_tgt_scalar), 1.0])

        target_vectors_t = (
            self.empty_target_vector.unsqueeze(0)
            .unsqueeze(0)
            .expand(self.max_options, self.max_targets_per_option, -1)
            .clone()
        )

        if tgt_opt_indices:
            tgt_type_ids_t = torch.tensor(tgt_type_ids, dtype=torch.long, device=device)
            tgt_scalars_t = torch.tensor(tgt_scalars_list, dtype=torch.float32, device=device)
            tgt_refs_t = torch.stack(tgt_refs)

            encoded_targets = (
                self.target_type_embedding(tgt_type_ids_t)
                + tgt_refs_t
                + self.target_scalar_projection(tgt_scalars_t)
            )

            opt_idx_t = torch.tensor(tgt_opt_indices, dtype=torch.long, device=device)
            slot_idx_t = torch.tensor(tgt_slot_indices, dtype=torch.long, device=device)
            target_vectors_t[opt_idx_t, slot_idx_t] = encoded_targets

        return {
            "pending_vector": pending_vector,
            "option_vectors": option_vectors_t,
            "option_mask": option_mask,
            "target_vectors": target_vectors_t,
            "target_mask": target_mask,
            "target_overflow": target_overflow,
            "priority_candidates": build_priority_candidates(
                pending,
                max_targets_per_option=self.max_targets_per_option,
            ),
        }

    def _option_reference_vector(
        self,
        option: PendingOptionState,
        object_vectors: dict[str, Tensor],
    ) -> Tensor:
        device = self.empty_option_vector.device
        for key in ("card_id", "permanent_id", "id"):
            value = option.get(key)
            if value and value in object_vectors:
                return object_vectors[value]

        card_name = option.get("card_name")
        if card_name:
            raw = self.game_state_encoder._lookup_card_embedding(
                cast(GameCardState, {"Name": card_name})
            )
            return self.game_state_encoder.card_projection(raw.to(device))

        return torch.zeros(self.d_action, device=device)

    def _build_reference_vectors(
        self,
        state: GameStateSnapshot,
        perspective_player_idx: int,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        all_cards = self.game_state_encoder._collect_slot_cards(state, perspective_player_idx)
        slot_vectors = self.game_state_encoder._encode_slots_batched(all_cards)

        object_vectors: dict[str, Tensor] = {}
        for i, card in enumerate(all_cards):
            if card is not None:
                card_id = card.get("ID")
                if card_id:
                    object_vectors[card_id] = slot_vectors[i]

        player_vectors = self._build_player_vectors(state, perspective_player_idx)
        return object_vectors, player_vectors

    def _build_player_vectors(
        self,
        state: GameStateSnapshot,
        perspective_player_idx: int,
    ) -> dict[str, Tensor]:
        player_vectors: dict[str, Tensor] = {}
        for idx, state_player in enumerate(state["players"]):
            player_id = state_player.get("ID")
            if not player_id:
                continue
            player_vectors[player_id] = self._player_target_vector(
                state_player,
                is_self=idx == perspective_player_idx,
            )
        return player_vectors

    def _player_target_vector(self, player: PlayerState, *, is_self: bool) -> Tensor:
        device = self.empty_option_vector.device
        type_id = torch.tensor(_index_or_unknown(TARGET_TYPES, "player"), device=device)
        scalars = torch.tensor(
            [1.0 if is_self else 0.0, 1.0],
            dtype=torch.float32,
            device=device,
        )
        return self.target_type_embedding(type_id) + self.target_scalar_projection(scalars)


def build_priority_candidates(
    pending: PendingState,
    *,
    max_targets_per_option: int = DEFAULT_MAX_TARGETS_PER_OPTION,
) -> list[LegalActionCandidate]:
    if pending.get("kind") != "priority":
        return []

    candidates: list[LegalActionCandidate] = []
    for option_idx, option in enumerate(pending.get("options", [])):
        kind = option.get("kind", "")
        targets = option.get("valid_targets", [])[:max_targets_per_option]
        if kind == "pass":
            candidates.append(
                LegalActionCandidate(option_idx, kind, cast(ActionRequest, {"kind": "pass"}))
            )
        elif kind == "play_land":
            card_id = option.get("card_id", "")
            candidates.append(
                LegalActionCandidate(
                    option_idx,
                    kind,
                    cast(ActionRequest, {"kind": "play_land", "card_id": card_id}),
                )
            )
        elif kind in {"cast_spell", "activate_ability"}:
            if targets:
                for target_idx, target in enumerate(targets):
                    candidates.append(
                        LegalActionCandidate(
                            option_idx,
                            kind,
                            _priority_payload(option, target_id=target.get("id", "")),
                            target_index=target_idx,
                        )
                    )
            else:
                candidates.append(
                    LegalActionCandidate(
                        option_idx,
                        kind,
                        _priority_payload(option, target_id=None),
                    )
                )
    return candidates


def action_from_priority_candidate(candidate: LegalActionCandidate) -> ActionRequest:
    return candidate.to_action_request()


def selected_priority_candidate_index(
    pending: PendingState,
    action: ActionRequest,
    *,
    max_targets_per_option: int = DEFAULT_MAX_TARGETS_PER_OPTION,
) -> int:
    normalized_action = _normalize_priority_payload(action)
    for idx, candidate in enumerate(
        build_priority_candidates(
            pending,
            max_targets_per_option=max_targets_per_option,
        )
    ):
        if _normalize_priority_payload(candidate.payload) == normalized_action:
            return idx
    return -100


def action_from_attackers(
    pending: PendingState,
    selected: Tensor | list[bool] | list[int],
) -> ActionRequest:
    ids: list[str] = []
    for option, is_selected in zip(pending.get("options", []), selected, strict=False):
        selected_bool = (
            bool(is_selected.item()) if isinstance(is_selected, Tensor) else bool(is_selected)
        )
        if selected_bool and option.get("permanent_id"):
            ids.append(option["permanent_id"])
    return cast(ActionRequest, {"attackers": ids})


def action_from_blockers(
    pending: PendingState,
    selected_target_indices: Tensor | list[int],
) -> ActionRequest:
    assignments: list[BlockerAssignState] = []
    for option, raw_target_idx in zip(
        pending.get("options", []),
        selected_target_indices,
        strict=False,
    ):
        target_idx = (
            int(raw_target_idx.item())
            if isinstance(raw_target_idx, Tensor)
            else int(raw_target_idx)
        )
        if target_idx < 0:
            continue
        targets = option.get("valid_targets", [])
        if target_idx >= len(targets):
            continue
        blocker_id = option.get("permanent_id")
        attacker_id = targets[target_idx].get("id")
        if blocker_id and attacker_id:
            assignments.append({"blocker": blocker_id, "attacker": attacker_id})
    return cast(ActionRequest, {"blockers": assignments})


def action_from_choice_index(index: int) -> ActionRequest:
    return cast(ActionRequest, {"selected_index": int(index)})


def action_from_choice_ids(ids: list[str]) -> ActionRequest:
    return cast(ActionRequest, {"selected_ids": ids})


def action_from_choice_color(color: str) -> ActionRequest:
    return cast(ActionRequest, {"selected_color": color})


def action_from_choice_accepted(accepted: bool) -> ActionRequest:
    return cast(ActionRequest, {"accepted": bool(accepted)})


def _priority_payload(
    option: PendingOptionState,
    *,
    target_id: str | None,
) -> ActionRequest:
    kind = option.get("kind", "")
    if kind == "cast_spell":
        payload: ActionRequest = {"kind": kind, "card_id": option.get("card_id", "")}
    else:
        payload: ActionRequest = {
            "kind": kind,
            "permanent_id": option.get("permanent_id", ""),
            "ability_index": int(option.get("ability_index", 0)),
        }
    if target_id:
        payload["targets"] = [target_id]
    return payload


def _normalize_priority_payload(action: ActionRequest) -> dict[str, Any]:
    kind = action.get("kind", "")
    normalized: dict[str, Any] = {"kind": kind}
    if kind in {"play_land", "cast_spell"}:
        normalized["card_id"] = action.get("card_id", "")
    if kind == "activate_ability":
        normalized["permanent_id"] = action.get("permanent_id", "")
        normalized["ability_index"] = int(action.get("ability_index", 0))
    targets = action.get("targets", [])
    if targets:
        normalized["targets"] = list(targets)
    return normalized


def _option_scalars(
    option: PendingOptionState,
    *,
    pending: PendingState,
    option_idx: int,
    max_options: int,
    max_targets_per_option: int,
) -> list[float]:
    targets = option.get("valid_targets", [])
    overflow_count = max(0, len(targets) - max_targets_per_option)
    mana_features = _mana_cost_features(option.get("mana_cost", ""))
    return [
        _clip_norm(option_idx, max(1.0, float(max_options - 1))),
        _clip_norm(option.get("ability_index", 0), MAX_ABILITY_INDEX),
        _clip_norm(len(targets), float(max_targets_per_option)),
        _clip_norm(overflow_count, MAX_TARGET_OVERFLOW),
        float(bool(option.get("card_id"))),
        float(bool(option.get("permanent_id"))),
        float(bool(option.get("id"))),
        *mana_features,
        _clip_norm(pending.get("amount", 0), MAX_AMOUNT),
    ]


def _mana_cost_features(mana_cost: str) -> list[float]:
    counts = {symbol: 0.0 for symbol in MANA_SYMBOLS}
    generic = 0.0
    for raw_symbol in re.findall(r"\{([^}]+)\}", mana_cost):
        symbol = raw_symbol.upper()
        if symbol in counts:
            counts[symbol] += 1.0
        elif symbol.isdigit():
            generic += float(symbol)
        elif "/" in symbol:
            for part in symbol.split("/"):
                if part in counts:
                    counts[part] += 0.5
    # Fold generic mana into colorless pressure for a compact V1 feature.
    counts["C"] += generic
    return [_clip_norm(counts[symbol], 10.0) for symbol in MANA_SYMBOLS]


def _index_or_unknown(values: tuple[str, ...], value: str) -> int:
    normalized = value.casefold()
    for idx, candidate in enumerate(values):
        if candidate.casefold() == normalized:
            return idx
    return len(values) - 1


def _color_index(value: str) -> int:
    aliases = {
        "w": "white",
        "u": "blue",
        "b": "black",
        "r": "red",
        "g": "green",
        "c": "colorless",
    }
    normalized = value.casefold()
    return _index_or_unknown(COLORS, aliases.get(normalized, normalized))


def _clip_norm(value: int | float | None, maximum: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), maximum)) / maximum


def _copy_action_request(action: ActionRequest) -> ActionRequest:
    copied: dict[str, Any] = {}
    for key, value in action.items():
        if isinstance(value, list):
            copied[key] = [item.copy() if isinstance(item, dict) else item for item in value]
        else:
            copied[key] = value
    return cast(ActionRequest, copied)
