"""Parity helpers for validating batch encoding against Python reference paths."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from magic_ai.actions import (
    ActionOptionsEncoder,
    LegalActionCandidate,
    ParsedActionBatch,
    ParsedActionInputs,
    PendingState,
)
from magic_ai.game_state import (
    GAME_INFO_DIM,
    ZONE_SLOT_COUNT,
    GameCardState,
    GameStateEncoder,
    GameStateSnapshot,
    ManaPoolState,
    ParsedGameState,
    ParsedGameStateBatch,
    PlayerState,
)


@dataclass(frozen=True)
class EncoderParityCase:
    """One regression case for batch encoder parity."""

    name: str
    state: GameStateSnapshot
    pending: PendingState
    perspective_player_idx: int | None = None


@dataclass(frozen=True)
class EncoderBatchOutputs:
    """Parsed outputs for the batched game-state and pending-action encoders."""

    parsed_state: ParsedGameStateBatch
    parsed_action: ParsedActionBatch


@dataclass(frozen=True)
class EncoderParityResult:
    """Outcome of one parity comparison."""

    batch_case_names: tuple[str, ...]
    mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches


class BatchEncoder(Protocol):
    """Callable contract used by the regression harness."""

    def __call__(
        self,
        game_state_encoder: GameStateEncoder,
        action_encoder: ActionOptionsEncoder,
        states: Sequence[GameStateSnapshot],
        pendings: Sequence[PendingState],
        perspective_player_indices: Sequence[int | None],
    ) -> EncoderBatchOutputs: ...


def encode_python_reference(
    game_state_encoder: GameStateEncoder,
    action_encoder: ActionOptionsEncoder,
    states: Sequence[GameStateSnapshot],
    pendings: Sequence[PendingState],
    perspective_player_indices: Sequence[int | None],
) -> EncoderBatchOutputs:
    """Reference implementation using the existing single-item Python parsers."""

    parsed_states: list[ParsedGameState] = []
    parsed_actions: list[ParsedActionInputs] = []

    for state, pending, perspective_player_idx in zip(
        states, pendings, perspective_player_indices, strict=True
    ):
        parsed_state = game_state_encoder.parse_state(state, perspective_player_idx)
        resolved_player_idx = game_state_encoder._resolve_perspective_player_idx(
            state, perspective_player_idx
        )
        parsed_action = action_encoder.parse_pending(
            state,
            pending,
            perspective_player_idx=resolved_player_idx,
            card_id_to_slot=parsed_state.card_id_to_slot,
        )
        parsed_states.append(parsed_state)
        parsed_actions.append(parsed_action)

    return EncoderBatchOutputs(
        parsed_state=_stack_parsed_game_states(parsed_states),
        parsed_action=_stack_parsed_actions(parsed_actions),
    )


def encode_python_batch(
    game_state_encoder: GameStateEncoder,
    action_encoder: ActionOptionsEncoder,
    states: Sequence[GameStateSnapshot],
    pendings: Sequence[PendingState],
    perspective_player_indices: Sequence[int | None],
) -> EncoderBatchOutputs:
    """Candidate implementation using the current Python batch parsers."""

    parsed_state = game_state_encoder.parse_state_batch(states, perspective_player_indices)
    resolved_player_indices = [
        game_state_encoder._resolve_perspective_player_idx(state, perspective_player_idx)
        for state, perspective_player_idx in zip(states, perspective_player_indices, strict=True)
    ]
    parsed_action = action_encoder.parse_pending_batch(
        list(states),
        list(pendings),
        perspective_player_indices=resolved_player_indices,
        card_id_to_slots=parsed_state.card_id_to_slots,
    )
    return EncoderBatchOutputs(
        parsed_state=parsed_state,
        parsed_action=parsed_action,
    )


def compare_batch_outputs(
    expected: EncoderBatchOutputs,
    actual: EncoderBatchOutputs,
) -> list[str]:
    """Return human-readable parity mismatches."""

    mismatches: list[str] = []
    _compare_tensor(
        mismatches,
        "parsed_state.slot_card_rows",
        expected.parsed_state.slot_card_rows,
        actual.parsed_state.slot_card_rows,
    )
    _compare_tensor(
        mismatches,
        "parsed_state.slot_occupied",
        expected.parsed_state.slot_occupied,
        actual.parsed_state.slot_occupied,
    )
    _compare_tensor(
        mismatches,
        "parsed_state.slot_tapped",
        expected.parsed_state.slot_tapped,
        actual.parsed_state.slot_tapped,
    )
    _compare_tensor(
        mismatches,
        "parsed_state.game_info",
        expected.parsed_state.game_info,
        actual.parsed_state.game_info,
    )
    if expected.parsed_state.card_id_to_slots != actual.parsed_state.card_id_to_slots:
        mismatches.append("parsed_state.card_id_to_slots mismatch")

    _compare_tensor(
        mismatches,
        "parsed_action.pending_kind_id",
        expected.parsed_action.pending_kind_id,
        actual.parsed_action.pending_kind_id,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.num_present_options",
        expected.parsed_action.num_present_options,
        actual.parsed_action.num_present_options,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.option_kind_ids",
        expected.parsed_action.option_kind_ids,
        actual.parsed_action.option_kind_ids,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.option_scalars",
        expected.parsed_action.option_scalars,
        actual.parsed_action.option_scalars,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.option_mask",
        expected.parsed_action.option_mask,
        actual.parsed_action.option_mask,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.option_ref_slot_idx",
        expected.parsed_action.option_ref_slot_idx,
        actual.parsed_action.option_ref_slot_idx,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.option_ref_card_row",
        expected.parsed_action.option_ref_card_row,
        actual.parsed_action.option_ref_card_row,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.target_mask",
        expected.parsed_action.target_mask,
        actual.parsed_action.target_mask,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.target_type_ids",
        expected.parsed_action.target_type_ids,
        actual.parsed_action.target_type_ids,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.target_scalars",
        expected.parsed_action.target_scalars,
        actual.parsed_action.target_scalars,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.target_overflow",
        expected.parsed_action.target_overflow,
        actual.parsed_action.target_overflow,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.target_ref_slot_idx",
        expected.parsed_action.target_ref_slot_idx,
        actual.parsed_action.target_ref_slot_idx,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.target_ref_is_player",
        expected.parsed_action.target_ref_is_player,
        actual.parsed_action.target_ref_is_player,
    )
    _compare_tensor(
        mismatches,
        "parsed_action.target_ref_is_self",
        expected.parsed_action.target_ref_is_self,
        actual.parsed_action.target_ref_is_self,
    )
    if expected.parsed_action.priority_candidates != actual.parsed_action.priority_candidates:
        mismatches.append("parsed_action.priority_candidates mismatch")

    return mismatches


def assert_batch_outputs_match(
    expected: EncoderBatchOutputs,
    actual: EncoderBatchOutputs,
) -> None:
    """Raise with a compact summary if parity fails."""

    mismatches = compare_batch_outputs(expected, actual)
    if mismatches:
        raise AssertionError("\n".join(mismatches))


def run_parity_suite(
    *,
    game_state_encoder: GameStateEncoder,
    action_encoder: ActionOptionsEncoder,
    cases: Sequence[EncoderParityCase],
    candidate_encoder: BatchEncoder = encode_python_batch,
    reference_encoder: BatchEncoder = encode_python_reference,
    batch_sizes: Sequence[int] = (1, 2, 4, 8),
) -> list[EncoderParityResult]:
    """Run parity checks across several batch partitions.

    The suite skips batch sizes larger than the corpus length and uses a final
    partial chunk when the corpus length is not divisible by the batch size.
    """

    if not cases:
        return []

    results: list[EncoderParityResult] = []
    total_cases = len(cases)
    for batch_size in batch_sizes:
        if batch_size <= 0 or batch_size > total_cases:
            continue
        for start in range(0, total_cases, batch_size):
            batch = list(cases[start : start + batch_size])
            states = [case.state for case in batch]
            pendings = [case.pending for case in batch]
            perspective_player_indices = [case.perspective_player_idx for case in batch]
            expected = reference_encoder(
                game_state_encoder,
                action_encoder,
                states,
                pendings,
                perspective_player_indices,
            )
            actual = candidate_encoder(
                game_state_encoder,
                action_encoder,
                states,
                pendings,
                perspective_player_indices,
            )
            results.append(
                EncoderParityResult(
                    batch_case_names=tuple(case.name for case in batch),
                    mismatches=tuple(compare_batch_outputs(expected, actual)),
                )
            )
    return results


def load_batch_encoder(spec: str) -> BatchEncoder:
    """Load a `module:callable` batch encoder implementation."""

    module_name, separator, attr_name = spec.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"invalid batch encoder spec {spec!r}; expected module:callable")
    module = importlib.import_module(module_name)
    candidate = getattr(module, attr_name)
    if not callable(candidate):
        raise TypeError(f"{spec!r} did not resolve to a callable")
    return candidate


def _stack_parsed_game_states(items: Sequence[ParsedGameState]) -> ParsedGameStateBatch:
    slot_card_rows = torch.tensor(
        [item.slot_card_rows for item in items],
        dtype=torch.long,
    )
    slot_occupied = torch.tensor(
        [item.slot_occupied for item in items],
        dtype=torch.float32,
    )
    slot_tapped = torch.tensor(
        [item.slot_tapped for item in items],
        dtype=torch.float32,
    )
    game_info = torch.tensor(
        [item.game_info for item in items],
        dtype=torch.float32,
    )
    return ParsedGameStateBatch(
        slot_card_rows=slot_card_rows,
        slot_occupied=slot_occupied,
        slot_tapped=slot_tapped,
        game_info=game_info,
        card_id_to_slots=[dict(item.card_id_to_slot) for item in items],
    )


def _stack_parsed_actions(items: Sequence[ParsedActionInputs]) -> ParsedActionBatch:
    pending_kind_id = torch.tensor(
        [item.pending_kind_id for item in items],
        dtype=torch.long,
    )
    num_present_options = torch.tensor(
        [item.num_present_options for item in items],
        dtype=torch.long,
    )
    option_kind_ids = torch.tensor(
        [item.option_kind_ids for item in items],
        dtype=torch.long,
    )
    option_scalars = torch.tensor(
        [item.option_scalars for item in items],
        dtype=torch.float32,
    )
    option_mask = torch.tensor(
        [item.option_mask for item in items],
        dtype=torch.float32,
    )
    option_ref_slot_idx = torch.tensor(
        [item.option_ref_slot_idx for item in items],
        dtype=torch.long,
    )
    option_ref_card_row = torch.tensor(
        [item.option_ref_card_row for item in items],
        dtype=torch.long,
    )
    target_mask = torch.tensor(
        [item.target_mask for item in items],
        dtype=torch.float32,
    )
    target_type_ids = torch.tensor(
        [item.target_type_ids for item in items],
        dtype=torch.long,
    )
    target_scalars = torch.tensor(
        [item.target_scalars for item in items],
        dtype=torch.float32,
    )
    target_overflow = torch.tensor(
        [item.target_overflow for item in items],
        dtype=torch.float32,
    )
    target_ref_slot_idx = torch.tensor(
        [item.target_ref_slot_idx for item in items],
        dtype=torch.long,
    )
    target_ref_is_player = torch.tensor(
        [item.target_ref_is_player for item in items],
        dtype=torch.bool,
    )
    target_ref_is_self = torch.tensor(
        [item.target_ref_is_self for item in items],
        dtype=torch.bool,
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
        priority_candidates=[list(item.priority_candidates) for item in items],
    )


def _compare_tensor(
    mismatches: list[str],
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    if expected.shape != actual.shape:
        mismatches.append(
            f"{name} shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
        )
        return
    if expected.dtype != actual.dtype:
        mismatches.append(f"{name} dtype mismatch: expected {expected.dtype}, got {actual.dtype}")
        return
    if not torch.equal(expected, actual):
        diff_count = int((expected != actual).sum().item())
        mismatches.append(f"{name} values mismatch: {diff_count} entries differ")


def _sample_player(
    *,
    player_id: str,
    name: str,
    life: int,
    hand: Sequence[tuple[str, str]],
    graveyard: Sequence[tuple[str, str]],
    battlefield: Sequence[tuple[str, str, bool]],
    library_count: int,
    mana_pool: dict[str, int] | None = None,
) -> PlayerState:
    hand_cards = cast(
        list[GameCardState],
        [{"ID": card_id, "Name": card_name} for card_id, card_name in hand],
    )
    graveyard_cards = cast(
        list[GameCardState],
        [{"ID": card_id, "Name": card_name} for card_id, card_name in graveyard],
    )
    battlefield_cards = cast(
        list[GameCardState],
        [
            {"ID": card_id, "Name": card_name, "Tapped": tapped}
            for card_id, card_name, tapped in battlefield
        ],
    )
    return {
        "ID": player_id,
        "Name": name,
        "Life": life,
        "HandCount": len(hand),
        "GraveyardCount": len(graveyard),
        "LibraryCount": library_count,
        "Hand": hand_cards,
        "Graveyard": graveyard_cards,
        "Battlefield": battlefield_cards,
        "ManaPool": cast(ManaPoolState, mana_pool or {}),
    }


def build_sample_parity_cases() -> list[EncoderParityCase]:
    """Representative cases for batch encoder regression coverage."""

    priority_pending: PendingState = {
        "kind": "priority",
        "player_idx": 0,
        "options": [
            {"id": "opt-pass", "kind": "pass", "label": "Pass"},
            {
                "id": "opt-land",
                "kind": "play_land",
                "label": "Play Forest",
                "card_id": "card-forest",
                "card_name": "Forest",
            },
            {
                "id": "opt-bolt",
                "kind": "cast_spell",
                "label": "Cast Lightning Bolt",
                "card_id": "card-bolt",
                "card_name": "Lightning Bolt",
                "valid_targets": [
                    {"id": "p2", "label": "Bob"},
                    {"id": "perm-angel", "label": "Serra Angel"},
                ],
            },
            {
                "id": "opt-elf-ability",
                "kind": "activate_ability",
                "label": "Tap Llanowar Elves",
                "permanent_id": "perm-elf",
                "ability_index": 1,
                "valid_targets": [{"id": "p1", "label": "Alice"}],
            },
        ],
    }
    priority_state: GameStateSnapshot = {
        "turn": 3,
        "active_player": "Alice",
        "step": "Precombat Main",
        "players": [
            _sample_player(
                player_id="p1",
                name="Alice",
                life=20,
                hand=[("card-forest", "Forest"), ("card-bolt", "Lightning Bolt")],
                graveyard=[("card-ponder", "Ponder")],
                battlefield=[
                    ("perm-elf", "Llanowar Elves", True),
                    ("perm-guide", "Goblin Guide", False),
                ],
                library_count=48,
                mana_pool={"Green": 1, "Red": 1},
            ),
            _sample_player(
                player_id="p2",
                name="Bob",
                life=18,
                hand=[("card-counter", "Counterspell")],
                graveyard=[],
                battlefield=[("perm-angel", "Serra Angel", False)],
                library_count=49,
                mana_pool={"White": 1},
            ),
        ],
        "pending": priority_pending,
        "stack": [{"id": "stack-bolt", "name": "Shock"}],
    }

    attackers_pending: PendingState = {
        "kind": "attackers",
        "player_idx": 0,
        "options": [
            {
                "id": "atk-elf",
                "kind": "attacker",
                "label": "Attack with Llanowar Elves",
                "permanent_id": "perm-elf",
            },
            {
                "id": "atk-guide",
                "kind": "attacker",
                "label": "Attack with Goblin Guide",
                "permanent_id": "perm-guide",
            },
        ],
    }
    attackers_state: GameStateSnapshot = {
        "turn": priority_state["turn"],
        "active_player": priority_state["active_player"],
        "pending": attackers_pending,
        "players": priority_state["players"],
        "step": "Declare Attackers",
        "stack": priority_state["stack"],
    }

    blockers_pending: PendingState = {
        "kind": "blockers",
        "player_idx": 1,
        "options": [
            {
                "id": "blk-wall",
                "kind": "blocker",
                "label": "Block with Wall of Omens",
                "permanent_id": "perm-wall",
                "valid_targets": [{"id": "perm-guide", "label": "Goblin Guide"}],
            }
        ],
    }
    blockers_state: GameStateSnapshot = {
        "turn": 5,
        "active_player": "Alice",
        "step": "Declare Blockers",
        "players": [
            _sample_player(
                player_id="p1",
                name="Alice",
                life=16,
                hand=[],
                graveyard=[("card-bolt", "Lightning Bolt")],
                battlefield=[
                    ("perm-guide", "Goblin Guide", False),
                    ("perm-elf", "Llanowar Elves", False),
                ],
                library_count=44,
                mana_pool={"Red": 2},
            ),
            _sample_player(
                player_id="p2",
                name="Bob",
                life=12,
                hand=[("card-visions", "Ancestral Vision")],
                graveyard=[],
                battlefield=[
                    ("perm-wall", "Wall of Omens", False),
                    ("perm-angel", "Serra Angel", True),
                ],
                library_count=41,
                mana_pool={"White": 2, "Blue": 1},
            ),
        ],
        "pending": blockers_pending,
    }

    library_choice_pending: PendingState = {
        "kind": "card_from_library",
        "player_idx": 0,
        "amount": 1,
        "options": [
            {
                "id": "lib-opt-1",
                "kind": "choice",
                "label": "Take Counterspell",
                "card_name": "Counterspell",
            },
            {
                "id": "lib-opt-2",
                "kind": "choice",
                "label": "Take Unknown",
                "card_name": "Made Up Card",
            },
        ],
    }
    library_choice_state: GameStateSnapshot = {
        "turn": 7,
        "active_player": "p1",
        "step": "Draw",
        "players": [
            _sample_player(
                player_id="p1",
                name="Alice",
                life=9,
                hand=[("card-ponder", "Ponder")],
                graveyard=[("card-forest", "Forest")],
                battlefield=[],
                library_count=30,
                mana_pool={},
            ),
            _sample_player(
                player_id="p2",
                name="Bob",
                life=11,
                hand=[],
                graveyard=[],
                battlefield=[("perm-angel", "Serra Angel", False)],
                library_count=27,
                mana_pool={"Blue": 1},
            ),
        ],
        "pending": library_choice_pending,
    }

    return [
        EncoderParityCase(
            name="priority_targets_and_references",
            state=priority_state,
            pending=priority_pending,
            perspective_player_idx=0,
        ),
        EncoderParityCase(
            name="attackers_pending",
            state=attackers_state,
            pending=attackers_pending,
            perspective_player_idx=0,
        ),
        EncoderParityCase(
            name="blockers_resolved_from_pending_player",
            state=blockers_state,
            pending=blockers_pending,
            perspective_player_idx=None,
        ),
        EncoderParityCase(
            name="library_choice_card_name_fallback",
            state=library_choice_state,
            pending=library_choice_pending,
            perspective_player_idx=None,
        ),
    ]


def build_sample_encoders() -> tuple[GameStateEncoder, ActionOptionsEncoder]:
    """Create deterministic encoders for parity tests and local regression runs."""

    embeddings = {
        "Forest": [1.0, 0.0, 0.0, 0.0],
        "Lightning Bolt": [0.0, 1.0, 0.0, 0.0],
        "Ponder": [0.0, 0.0, 1.0, 0.0],
        "Llanowar Elves": [0.0, 0.0, 0.0, 1.0],
        "Goblin Guide": [1.0, 1.0, 0.0, 0.0],
        "Serra Angel": [0.0, 1.0, 1.0, 0.0],
        "Counterspell": [0.0, 0.0, 1.0, 1.0],
        "Wall of Omens": [1.0, 0.0, 1.0, 0.0],
        "Ancestral Vision": [1.0, 0.0, 0.0, 1.0],
    }
    game_state_encoder = GameStateEncoder(embeddings, d_model=8)
    action_encoder = ActionOptionsEncoder(
        game_state_encoder,
        max_options=6,
        max_targets_per_option=2,
    )
    return game_state_encoder, action_encoder


def expected_state_shapes(batch_size: int) -> dict[str, tuple[int, ...]]:
    """Convenience helper for tests and future native wrappers."""

    return {
        "slot_card_rows": (batch_size, ZONE_SLOT_COUNT),
        "slot_occupied": (batch_size, ZONE_SLOT_COUNT),
        "slot_tapped": (batch_size, ZONE_SLOT_COUNT),
        "game_info": (batch_size, GAME_INFO_DIM),
    }


def priority_candidates_equal(
    left: Sequence[Sequence[LegalActionCandidate]],
    right: Sequence[Sequence[LegalActionCandidate]],
) -> bool:
    return list(left) == list(right)


__all__ = [
    "BatchEncoder",
    "EncoderBatchOutputs",
    "EncoderParityCase",
    "EncoderParityResult",
    "assert_batch_outputs_match",
    "build_sample_encoders",
    "build_sample_parity_cases",
    "compare_batch_outputs",
    "encode_python_batch",
    "encode_python_reference",
    "expected_state_shapes",
    "load_batch_encoder",
    "priority_candidates_equal",
    "run_parity_suite",
]
