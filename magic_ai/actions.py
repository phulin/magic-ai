"""Action-option encoding and selected-action decoding for mage-go pylib."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypedDict, cast

import torch
from torch import Tensor, nn

from magic_ai.game_state import (
    GameStateSnapshot,
    ParsedGameState,
    ParsedGameStateBatch,
    PendingOptionState,
    PendingState,
)
from magic_ai.slot_encoder.game_state import GameStateEncoder

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

PLAYER_TARGET_TYPE_ID = TARGET_TYPES.index("player")
PERMANENT_TARGET_TYPE_ID = TARGET_TYPES.index("permanent")
UNKNOWN_TARGET_TYPE_ID = TARGET_TYPES.index("unknown")


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


@dataclass(frozen=True)
class ParsedActionInputs:
    """Non-differentiable parsed representation of a pending action request.

    Fields are Python lists padded to ``max_options`` / ``max_targets_per_option``.
    Invalid entries are masked out by ``option_mask`` / ``target_mask``.
    Reference fields use ``-1`` to mean "no reference of this kind". No tensors
    are allocated here — ``RolloutBuffer.ingest_batch`` does the bulk CPU→GPU
    copy into preallocated buffers.
    """

    pending_kind_id: int
    num_present_options: int
    option_kind_ids: list[int]  # length max_options
    option_scalars: list[list[float]]  # [max_options, OPTION_SCALAR_DIM]
    option_mask: list[float]  # length max_options
    option_ref_slot_idx: list[int]  # length max_options (-1 if none)
    option_ref_card_row: list[int]  # length max_options (-1 if none)
    target_mask: list[list[float]]  # [max_options, max_targets]
    target_type_ids: list[list[int]]  # [max_options, max_targets]
    target_scalars: list[list[list[float]]]  # [max_options, max_targets, TARGET_SCALAR_DIM]
    target_overflow: list[float]  # length max_options
    target_ref_slot_idx: list[list[int]]  # [max_options, max_targets] (-1 if not permanent)
    target_ref_is_player: list[list[bool]]  # [max_options, max_targets]
    target_ref_is_self: list[list[bool]]  # [max_options, max_targets]
    priority_candidates: list[LegalActionCandidate]


@dataclass(frozen=True)
class ParsedActionBatch:
    """Batched parsed pending-request fields for one actor forward."""

    pending_kind_id: Tensor  # [N]
    num_present_options: Tensor  # [N]
    option_kind_ids: Tensor  # [N, max_options]
    option_scalars: Tensor  # [N, max_options, OPTION_SCALAR_DIM]
    option_mask: Tensor  # [N, max_options]
    option_ref_slot_idx: Tensor  # [N, max_options]
    option_ref_card_row: Tensor  # [N, max_options]
    target_mask: Tensor  # [N, max_options, max_targets]
    target_type_ids: Tensor  # [N, max_options, max_targets]
    target_scalars: Tensor  # [N, max_options, max_targets, TARGET_SCALAR_DIM]
    target_overflow: Tensor  # [N, max_options]
    target_ref_slot_idx: Tensor  # [N, max_options, max_targets]
    target_ref_is_player: Tensor  # [N, max_options, max_targets]
    target_ref_is_self: Tensor  # [N, max_options, max_targets]
    priority_candidates: list[list[LegalActionCandidate]]


class SelectedActionEncoder:
    """Encode a selected MageStep action into supervised labels.

    Label dimensions line up with ``ActionOptionsEncoder``'s option/target
    dimensions. Irrelevant scalar labels use -100 so they can be passed to
    PyTorch losses with ``ignore_index=-100``.
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
    """Parse + embed legal action options for mage-go ``pending`` requests.

    Parsing (``parse_pending``) builds integer/scalar index tensors describing
    the pending request; it does not apply any trainable operations. Embedding
    (``embed_from_parsed``) consumes those index tensors together with
    precomputed slot vectors from ``GameStateEncoder.embed_slot_vectors`` and
    produces the differentiable option/target/pending tensors used by the
    policy trunk and action heads.
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

    def parse_pending(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int,
        card_id_to_slot: dict[str, int],
    ) -> ParsedActionInputs:
        options = pending.get("options", [])
        max_opt = self.max_options
        max_tgt = self.max_targets_per_option
        num_present = min(len(options), max_opt)

        option_kind_ids: list[int] = [0] * max_opt
        option_scalars: list[list[float]] = [[0.0] * OPTION_SCALAR_DIM for _ in range(max_opt)]
        option_mask: list[float] = [0.0] * max_opt
        option_ref_slot_idx: list[int] = [-1] * max_opt
        option_ref_card_row: list[int] = [-1] * max_opt

        target_mask: list[list[float]] = [[0.0] * max_tgt for _ in range(max_opt)]
        target_type_ids: list[list[int]] = [
            [UNKNOWN_TARGET_TYPE_ID] * max_tgt for _ in range(max_opt)
        ]
        target_scalars: list[list[list[float]]] = [
            [[0.0] * TARGET_SCALAR_DIM for _ in range(max_tgt)] for _ in range(max_opt)
        ]
        target_overflow: list[float] = [0.0] * max_opt
        target_ref_slot_idx: list[list[int]] = [[-1] * max_tgt for _ in range(max_opt)]
        target_ref_is_player: list[list[bool]] = [[False] * max_tgt for _ in range(max_opt)]
        target_ref_is_self: list[list[bool]] = [[False] * max_tgt for _ in range(max_opt)]

        player_self_id, player_opp_id = _player_ids(state, perspective_player_idx)
        max_tgt_scalar = max(1.0, float(max_tgt - 1))
        name_to_row = self.game_state_encoder._card_name_to_row

        for opt_i in range(num_present):
            option = options[opt_i]
            option_kind_ids[opt_i] = _index_or_unknown(ACTION_KINDS, option.get("kind", "unknown"))
            option_scalars[opt_i] = _option_scalars(
                option,
                pending=pending,
                option_idx=opt_i,
                max_options=max_opt,
                max_targets_per_option=max_tgt,
            )
            option_mask[opt_i] = 1.0

            slot_idx, card_row = _resolve_option_reference(
                option,
                card_id_to_slot=card_id_to_slot,
                name_to_row=name_to_row,
            )
            if slot_idx is not None:
                option_ref_slot_idx[opt_i] = slot_idx
            elif card_row is not None:
                option_ref_card_row[opt_i] = card_row

            targets = option.get("valid_targets", [])
            overflow_count = max(0, len(targets) - max_tgt)
            target_overflow[opt_i] = _clip_norm(overflow_count, MAX_TARGET_OVERFLOW)

            row_mask = target_mask[opt_i]
            row_types = target_type_ids[opt_i]
            row_scalars = target_scalars[opt_i]
            row_ref_slot = target_ref_slot_idx[opt_i]
            row_ref_player = target_ref_is_player[opt_i]
            row_ref_self = target_ref_is_self[opt_i]
            for tgt_j in range(min(len(targets), max_tgt)):
                target = targets[tgt_j]
                row_mask[tgt_j] = 1.0
                row_scalars[tgt_j] = [_clip_norm(tgt_j, max_tgt_scalar), 1.0]

                target_id = target.get("id", "")
                if target_id and target_id in (player_self_id, player_opp_id):
                    row_types[tgt_j] = PLAYER_TARGET_TYPE_ID
                    row_ref_player[tgt_j] = True
                    row_ref_self[tgt_j] = target_id == player_self_id
                elif target_id in card_id_to_slot:
                    row_types[tgt_j] = PERMANENT_TARGET_TYPE_ID
                    row_ref_slot[tgt_j] = card_id_to_slot[target_id]

        priority_candidates = build_priority_candidates(
            pending,
            max_targets_per_option=max_tgt,
        )

        return ParsedActionInputs(
            pending_kind_id=_index_or_unknown(PENDING_KINDS, pending.get("kind", "")),
            num_present_options=num_present,
            option_kind_ids=option_kind_ids,
            option_scalars=option_scalars,
            option_mask=option_mask,
            option_ref_slot_idx=option_ref_slot_idx,
            option_ref_card_row=option_ref_card_row,
            target_mask=target_mask,
            target_type_ids=target_type_ids,
            target_scalars=target_scalars,
            target_overflow=target_overflow,
            target_ref_slot_idx=target_ref_slot_idx,
            target_ref_is_player=target_ref_is_player,
            target_ref_is_self=target_ref_is_self,
            priority_candidates=priority_candidates,
        )

    def parse_pending_batch(
        self,
        states: list[GameStateSnapshot],
        pendings: list[PendingState],
        *,
        perspective_player_indices: list[int],
        card_id_to_slots: list[dict[str, int]],
    ) -> ParsedActionBatch:
        n = len(states)
        max_opt = self.max_options
        max_tgt = self.max_targets_per_option
        pending_kind_id = torch.zeros((n,), dtype=torch.long)
        num_present_options = torch.zeros((n,), dtype=torch.long)
        option_kind_ids = torch.zeros((n, max_opt), dtype=torch.long)
        option_scalars = torch.zeros((n, max_opt, OPTION_SCALAR_DIM), dtype=torch.float32)
        option_mask = torch.zeros((n, max_opt), dtype=torch.float32)
        option_ref_slot_idx = torch.full((n, max_opt), -1, dtype=torch.long)
        option_ref_card_row = torch.full((n, max_opt), -1, dtype=torch.long)
        target_mask = torch.zeros((n, max_opt, max_tgt), dtype=torch.float32)
        target_type_ids = torch.full(
            (n, max_opt, max_tgt),
            UNKNOWN_TARGET_TYPE_ID,
            dtype=torch.long,
        )
        target_scalars = torch.zeros(
            (n, max_opt, max_tgt, TARGET_SCALAR_DIM),
            dtype=torch.float32,
        )
        target_overflow = torch.zeros((n, max_opt), dtype=torch.float32)
        target_ref_slot_idx = torch.full((n, max_opt, max_tgt), -1, dtype=torch.long)
        target_ref_is_player = torch.zeros((n, max_opt, max_tgt), dtype=torch.bool)
        target_ref_is_self = torch.zeros((n, max_opt, max_tgt), dtype=torch.bool)
        priority_candidates: list[list[LegalActionCandidate]] = []
        max_tgt_scalar = max(1.0, float(max_tgt - 1))
        name_to_row = self.game_state_encoder._card_name_to_row

        for batch_idx, (state, pending, perspective_player_idx, card_id_to_slot) in enumerate(
            zip(
                states,
                pendings,
                perspective_player_indices,
                card_id_to_slots,
                strict=True,
            )
        ):
            options = pending.get("options", [])
            num_present = min(len(options), max_opt)
            pending_kind_id[batch_idx] = _index_or_unknown(PENDING_KINDS, pending.get("kind", ""))
            num_present_options[batch_idx] = num_present
            player_self_id, player_opp_id = _player_ids(state, perspective_player_idx)

            for opt_i in range(num_present):
                option = options[opt_i]
                option_kind_ids[batch_idx, opt_i] = _index_or_unknown(
                    ACTION_KINDS, option.get("kind", "unknown")
                )
                _fill_option_scalars(
                    option_scalars[batch_idx, opt_i],
                    option,
                    pending=pending,
                    option_idx=opt_i,
                    max_options=max_opt,
                    max_targets_per_option=max_tgt,
                )
                option_mask[batch_idx, opt_i] = 1.0

                slot_idx, card_row = _resolve_option_reference(
                    option,
                    card_id_to_slot=card_id_to_slot,
                    name_to_row=name_to_row,
                )
                if slot_idx is not None:
                    option_ref_slot_idx[batch_idx, opt_i] = slot_idx
                elif card_row is not None:
                    option_ref_card_row[batch_idx, opt_i] = card_row

                targets = option.get("valid_targets", [])
                overflow_count = max(0, len(targets) - max_tgt)
                target_overflow[batch_idx, opt_i] = _clip_norm(overflow_count, MAX_TARGET_OVERFLOW)

                for tgt_j in range(min(len(targets), max_tgt)):
                    target = targets[tgt_j]
                    target_mask[batch_idx, opt_i, tgt_j] = 1.0
                    target_scalars[batch_idx, opt_i, tgt_j, 0] = _clip_norm(tgt_j, max_tgt_scalar)
                    target_scalars[batch_idx, opt_i, tgt_j, 1] = 1.0

                    target_id = target.get("id", "")
                    if target_id and target_id in (player_self_id, player_opp_id):
                        target_type_ids[batch_idx, opt_i, tgt_j] = PLAYER_TARGET_TYPE_ID
                        target_ref_is_player[batch_idx, opt_i, tgt_j] = True
                        target_ref_is_self[batch_idx, opt_i, tgt_j] = target_id == player_self_id
                    elif target_id in card_id_to_slot:
                        target_type_ids[batch_idx, opt_i, tgt_j] = PERMANENT_TARGET_TYPE_ID
                        target_ref_slot_idx[batch_idx, opt_i, tgt_j] = card_id_to_slot[target_id]

            priority_candidates.append(
                build_priority_candidates(
                    pending,
                    max_targets_per_option=max_tgt,
                )
            )

        return ParsedActionBatch(
            pending_kind_id=pending_kind_id,
            num_present_options=num_present_options,
            option_kind_ids=option_kind_ids,
            option_scalars=option_scalars,
            option_mask=option_mask,
            option_ref_slot_idx=option_ref_slot_idx,
            option_ref_card_row=option_ref_card_row,
            target_mask=target_mask,
            target_type_ids=target_type_ids,
            target_scalars=target_scalars,
            target_overflow=target_overflow,
            target_ref_slot_idx=target_ref_slot_idx,
            target_ref_is_player=target_ref_is_player,
            target_ref_is_self=target_ref_is_self,
            priority_candidates=priority_candidates,
        )

    def embed_from_parsed(
        self,
        *,
        slot_vectors: Tensor,
        pending_kind_id: Tensor,
        option_kind_ids: Tensor,
        option_scalars: Tensor,
        option_mask: Tensor,
        option_ref_slot_idx: Tensor,
        option_ref_card_row: Tensor,
        target_mask: Tensor,
        target_type_ids: Tensor,
        target_scalars: Tensor,
        target_ref_slot_idx: Tensor,
        target_ref_is_player: Tensor,
        target_ref_is_self: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(pending_vector, option_vectors, target_vectors)``.

        Accepts batched inputs with a leading batch dimension ``[N, ...]``.
        ``slot_vectors`` is ``[N, ZONE_SLOT_COUNT, d_model]``; outputs are
        ``[N, d_model]`` for pending, ``[N, max_options, d_model]`` for
        options, and ``[N, max_options, max_targets, d_model]`` for targets.
        """

        d_model = self.d_action
        n = slot_vectors.shape[0]
        max_options = option_kind_ids.shape[-1]
        max_targets = target_mask.shape[-1]

        pending_vector = self.pending_kind_embedding(pending_kind_id)  # [N, d]

        option_slot_idx = option_ref_slot_idx.clamp_min(0)
        option_slot_ref = torch.gather(
            slot_vectors,
            dim=1,
            index=option_slot_idx.unsqueeze(-1).expand(-1, -1, d_model),
        )  # [N, max_options, d]
        option_card_row = option_ref_card_row.clamp_min(0)
        option_card_raw = self.game_state_encoder.card_embedding_table[option_card_row]
        option_card_ref = self.game_state_encoder.card_projection(option_card_raw)
        option_slot_mask = (option_ref_slot_idx >= 0).unsqueeze(-1)
        option_card_mask = (option_ref_card_row >= 0).unsqueeze(-1)
        zero_ref = torch.zeros_like(option_slot_ref)
        option_ref = torch.where(
            option_slot_mask,
            option_slot_ref,
            torch.where(option_card_mask, option_card_ref, zero_ref),
        )
        option_kind_emb = self.option_kind_embedding(option_kind_ids)
        option_scalar_proj = self.option_scalar_projection(option_scalars)
        option_present = (
            pending_vector.unsqueeze(1) + option_kind_emb + option_ref + option_scalar_proj
        )
        empty_option = self.empty_option_vector.view(1, 1, -1).expand(n, max_options, d_model)
        option_vectors = torch.where(option_mask.unsqueeze(-1) > 0, option_present, empty_option)

        target_slot_idx = target_ref_slot_idx.clamp_min(0)
        flat_slot_idx = target_slot_idx.view(n, max_options * max_targets)
        flat_slot_ref = torch.gather(
            slot_vectors,
            dim=1,
            index=flat_slot_idx.unsqueeze(-1).expand(-1, -1, d_model),
        )
        target_slot_ref = flat_slot_ref.view(n, max_options, max_targets, d_model)

        player_type_id = torch.tensor(
            PLAYER_TARGET_TYPE_ID, dtype=torch.long, device=slot_vectors.device
        )
        player_type_emb = self.target_type_embedding(player_type_id)
        is_self_f = target_ref_is_self.to(torch.float32)
        player_scalar_input = torch.stack([is_self_f, torch.ones_like(is_self_f)], dim=-1)
        player_scalar_proj = self.target_scalar_projection(player_scalar_input)
        player_ref = player_type_emb + player_scalar_proj

        target_slot_mask = (target_ref_slot_idx >= 0).unsqueeze(-1)
        target_player_mask = target_ref_is_player.bool().unsqueeze(-1)
        zero_target = torch.zeros_like(target_slot_ref)
        target_ref = torch.where(
            target_player_mask,
            player_ref,
            torch.where(target_slot_mask, target_slot_ref, zero_target),
        )

        target_type_emb = self.target_type_embedding(target_type_ids)
        target_scalar_proj = self.target_scalar_projection(target_scalars)
        target_present = target_type_emb + target_ref + target_scalar_proj
        empty_target = self.empty_target_vector.view(1, 1, 1, -1).expand(
            n, max_options, max_targets, d_model
        )
        target_vectors = torch.where(target_mask.unsqueeze(-1) > 0, target_present, empty_target)

        return pending_vector, option_vectors, target_vectors


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


def selected_option_id(pending: PendingState, selected_idx: int) -> str:
    options = pending.get("options", [])
    if not 0 <= selected_idx < len(options):
        return ""
    option = options[selected_idx]
    return option.get("id", "") or option.get("card_id", "") or option.get("permanent_id", "")


def _resolve_option_reference(
    option: PendingOptionState,
    *,
    card_id_to_slot: dict[str, int],
    name_to_row: dict[str, int],
) -> tuple[int | None, int | None]:
    for key in ("card_id", "permanent_id", "id"):
        value = option.get(key)
        if value and value in card_id_to_slot:
            return card_id_to_slot[value], None

    card_name = option.get("card_name")
    if card_name:
        from magic_ai.slot_encoder.game_state import _card_key as _ck

        row = name_to_row.get(_ck(card_name), 0)
        return None, row

    return None, None


def _player_ids(state: GameStateSnapshot, perspective_player_idx: int) -> tuple[str, str]:
    players = state["players"]
    self_id = players[perspective_player_idx].get("ID", "") if players else ""
    opp_idx = 1 - perspective_player_idx
    opp_id = players[opp_idx].get("ID", "") if len(players) == 2 else ""
    return self_id, opp_id


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


def _fill_option_scalars(
    out: Tensor,
    option: PendingOptionState,
    *,
    pending: PendingState,
    option_idx: int,
    max_options: int,
    max_targets_per_option: int,
) -> None:
    targets = option.get("valid_targets", [])
    overflow_count = max(0, len(targets) - max_targets_per_option)
    out[0] = _clip_norm(option_idx, max(1.0, float(max_options - 1)))
    out[1] = _clip_norm(option.get("ability_index", 0), MAX_ABILITY_INDEX)
    out[2] = _clip_norm(len(targets), float(max_targets_per_option))
    out[3] = _clip_norm(overflow_count, MAX_TARGET_OVERFLOW)
    out[4] = float(bool(option.get("card_id")))
    out[5] = float(bool(option.get("permanent_id")))
    out[6] = float(bool(option.get("id")))
    _fill_mana_cost_features(out[7:13], option.get("mana_cost", ""))
    out[13] = _clip_norm(pending.get("amount", 0), MAX_AMOUNT)


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


def _fill_mana_cost_features(out: Tensor, mana_cost: str) -> None:
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
    counts["C"] += generic
    for idx, symbol in enumerate(MANA_SYMBOLS):
        out[idx] = _clip_norm(counts[symbol], 10.0)


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


def build_decision_layout_rows(
    trace_kind: TraceKind,
    *,
    max_cached_choices: int,
    option_count: int,
    priority_candidates: list[LegalActionCandidate],
    target_counts_per_option: list[int],
) -> tuple[list[list[int]], list[list[int]], list[list[bool]], list[bool]]:
    """Return (option_rows, target_rows, mask_rows, uses_none) for a decision group.

    ``target_counts_per_option`` is only used for the ``blockers`` trace kind.
    """

    choices = max_cached_choices

    if trace_kind == "may":
        return [], [], [], []

    if trace_kind == "priority":
        candidates = priority_candidates[:choices]
        if not candidates:
            return [], [], [], []
        option_row = [-1] * choices
        target_row = [-1] * choices
        mask_row = [False] * choices
        for col, cand in enumerate(candidates):
            option_row[col] = cand.option_index
            if cand.target_index is not None:
                target_row[col] = cand.target_index
            mask_row[col] = True
        return [option_row], [target_row], [mask_row], [False]

    if trace_kind == "attackers":
        if option_count == 0:
            return [], [], [], []
        option_idx_l: list[list[int]] = []
        target_idx_l: list[list[int]] = []
        mask_l: list[list[bool]] = []
        for i in range(option_count):
            option_row = [-1] * choices
            target_row = [-1] * choices
            mask_row = [False] * choices
            if choices > 0:
                mask_row[0] = True
            if choices > 1:
                option_row[1] = i
                mask_row[1] = True
            option_idx_l.append(option_row)
            target_idx_l.append(target_row)
            mask_l.append(mask_row)
        return option_idx_l, target_idx_l, mask_l, [True] * option_count

    if trace_kind == "blockers":
        if option_count == 0:
            return [], [], [], []
        option_idx_l = []
        target_idx_l = []
        mask_l = []
        for i, target_count in enumerate(target_counts_per_option):
            option_row = [-1] * choices
            target_row = [-1] * choices
            mask_row = [False] * choices
            if choices > 0:
                mask_row[0] = True
            for t in range(int(target_count)):
                col = t + 1
                if col < choices:
                    option_row[col] = i
                    target_row[col] = t
                    mask_row[col] = True
            option_idx_l.append(option_row)
            target_idx_l.append(target_row)
            mask_l.append(mask_row)
        return option_idx_l, target_idx_l, mask_l, [True] * option_count

    # choice_index / choice_ids / choice_color
    if option_count == 0:
        return [], [], [], []
    option_row = [-1] * choices
    target_row = [-1] * choices
    mask_row = [False] * choices
    for i in range(option_count):
        option_row[i] = i
        mask_row[i] = True
    return [option_row], [target_row], [mask_row], [False]


def _copy_action_request(action: ActionRequest) -> ActionRequest:
    copied: dict[str, Any] = {}
    for key, value in action.items():
        if isinstance(value, list):
            copied[key] = [item.copy() if isinstance(item, dict) else item for item in value]
        else:
            copied[key] = value
    return cast(ActionRequest, copied)


# ---------------------------------------------------------------------------
# Shared game-general types (used by both slot and text encoders)
# ---------------------------------------------------------------------------

TraceKind = Literal[
    "priority",
    "attackers",
    "blockers",
    "choice_index",
    "choice_ids",
    "choice_color",
    "may",
]

TRACE_KIND_VALUES: tuple[TraceKind, ...] = (
    "priority",
    "attackers",
    "blockers",
    "choice_index",
    "choice_ids",
    "choice_color",
    "may",
)
TRACE_KIND_TO_ID: dict[TraceKind, int] = {
    trace_kind: idx for idx, trace_kind in enumerate(TRACE_KIND_VALUES)
}


@dataclass(frozen=True)
class ActionTrace:
    """Enough information to recompute a sampled action's log-probability."""

    kind: TraceKind
    indices: tuple[int, ...] = ()
    binary: tuple[float, ...] = ()


class PolicyStep(NamedTuple):
    # NamedTuple instead of @dataclass: ~3-5x faster to construct, which
    # matters because the rollout sampling path builds ~80 of these per
    # poll (~5k per profiled iter).
    action: ActionRequest
    trace: ActionTrace
    log_prob: Tensor
    value: Tensor
    entropy: Tensor
    replay_idx: int | None = None
    selected_choice_cols: tuple[int, ...] = ()
    may_selected: int = 0


@dataclass(frozen=True)
class ParsedStep:
    """Pure-Python parsed policy inputs for one step.

    Holds the parsed game-state / action-options plus the decision-row layout.
    No tensors; ``RolloutBuffer.ingest_batch`` does the bulk CPU→GPU copy.
    """

    parsed_state: ParsedGameState
    parsed_action: ParsedActionInputs
    trace_kind: TraceKind
    trace_kind_id: int
    decision_option_idx: list[list[int]]  # [G, C]
    decision_target_idx: list[list[int]]  # [G, C]
    decision_mask: list[list[bool]]  # [G, C]
    uses_none_head: list[bool]  # [G]
    pending: PendingState


@dataclass(frozen=True)
class ParsedBatch:
    """Batched parsed policy inputs for one actor forward."""

    parsed_state: ParsedGameStateBatch
    parsed_action: ParsedActionBatch
    trace_kinds: list[TraceKind]
    trace_kind_ids: Tensor  # [N]
    pendings: list[PendingState]
    decision_option_idx: Tensor  # [total_groups, max_cached_choices]
    decision_target_idx: Tensor  # [total_groups, max_cached_choices]
    decision_mask: Tensor  # [total_groups, max_cached_choices]
    uses_none_head: Tensor  # [total_groups]
    decision_starts: list[int]
    decision_counts: list[int]
