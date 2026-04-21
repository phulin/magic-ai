"""Preallocated GPU buffers for PPO rollout data.

Replaces per-step ``torch.tensor`` allocations in the rollout hot path with
bulk writes into a persistent set of buffers. Each parsed step is copied
into the buffer with one ``torch.tensor(list)`` call per field, then later
gathered by index from both the actor path (sampling) and the learner path
(PPO evaluation). Buffers are registered non-persistent so they do not land
in checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from magic_ai.model import ParsedStep


@dataclass(frozen=True)
class BufferWrite:
    """Indices and decision offsets returned from one ingest pass."""

    step_indices: Tensor  # Long [n] — buffer rows written this call
    decision_starts: list[int]  # per-step start row in decision buffer
    decision_counts: list[int]  # per-step number of decision rows


class RolloutBuffer(nn.Module):
    """Preallocated GPU storage for a rollout's parsed policy inputs."""

    def __init__(
        self,
        *,
        capacity: int,
        decision_capacity: int,
        max_options: int,
        max_targets_per_option: int,
        max_cached_choices: int,
        zone_slot_count: int,
        game_info_dim: int,
        option_scalar_dim: int,
        target_scalar_dim: int,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.decision_capacity = decision_capacity

        def _reg(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
            self.register_buffer(name, torch.zeros(shape, dtype=dtype), persistent=False)

        _reg("slot_card_rows", (capacity, zone_slot_count), torch.long)
        _reg("slot_occupied", (capacity, zone_slot_count), torch.float32)
        _reg("slot_tapped", (capacity, zone_slot_count), torch.float32)
        _reg("game_info", (capacity, game_info_dim), torch.float32)
        _reg("pending_kind_id", (capacity,), torch.long)
        _reg("option_kind_ids", (capacity, max_options), torch.long)
        _reg("option_scalars", (capacity, max_options, option_scalar_dim), torch.float32)
        _reg("option_mask", (capacity, max_options), torch.float32)
        _reg("option_ref_slot_idx", (capacity, max_options), torch.long)
        _reg("option_ref_card_row", (capacity, max_options), torch.long)
        _reg(
            "target_mask",
            (capacity, max_options, max_targets_per_option),
            torch.float32,
        )
        _reg(
            "target_type_ids",
            (capacity, max_options, max_targets_per_option),
            torch.long,
        )
        _reg(
            "target_scalars",
            (capacity, max_options, max_targets_per_option, target_scalar_dim),
            torch.float32,
        )
        _reg("target_overflow", (capacity, max_options), torch.float32)
        _reg(
            "target_ref_slot_idx",
            (capacity, max_options, max_targets_per_option),
            torch.long,
        )
        _reg(
            "target_ref_is_player",
            (capacity, max_options, max_targets_per_option),
            torch.bool,
        )
        _reg(
            "target_ref_is_self",
            (capacity, max_options, max_targets_per_option),
            torch.bool,
        )
        _reg("may_selected", (capacity,), torch.float32)

        _reg(
            "decision_option_idx",
            (decision_capacity, max_cached_choices),
            torch.long,
        )
        _reg(
            "decision_target_idx",
            (decision_capacity, max_cached_choices),
            torch.long,
        )
        _reg(
            "decision_mask",
            (decision_capacity, max_cached_choices),
            torch.bool,
        )
        _reg("uses_none_head", (decision_capacity,), torch.bool)
        _reg("selected_indices", (decision_capacity,), torch.long)

        self._step_cursor = 0
        self._decision_cursor = 0

    # Declared so type checkers see buffer Tensors as Tensors.
    slot_card_rows: Tensor
    slot_occupied: Tensor
    slot_tapped: Tensor
    game_info: Tensor
    pending_kind_id: Tensor
    option_kind_ids: Tensor
    option_scalars: Tensor
    option_mask: Tensor
    option_ref_slot_idx: Tensor
    option_ref_card_row: Tensor
    target_mask: Tensor
    target_type_ids: Tensor
    target_scalars: Tensor
    target_overflow: Tensor
    target_ref_slot_idx: Tensor
    target_ref_is_player: Tensor
    target_ref_is_self: Tensor
    may_selected: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    selected_indices: Tensor

    @property
    def device(self) -> torch.device:
        return self.slot_card_rows.device

    def reset(self) -> None:
        self._step_cursor = 0
        self._decision_cursor = 0

    def ingest_batch(self, parsed_steps: list[ParsedStep]) -> BufferWrite:
        n = len(parsed_steps)
        device = self.device
        if n == 0:
            return BufferWrite(
                step_indices=torch.zeros(0, dtype=torch.long, device=device),
                decision_starts=[],
                decision_counts=[],
            )

        start = self._step_cursor
        if start + n > self.capacity:
            raise RuntimeError(
                f"rollout buffer capacity {self.capacity} exceeded (cursor={start}, add={n})"
            )
        end = start + n

        self.slot_card_rows[start:end] = torch.tensor(
            [p.parsed_state.slot_card_rows for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.slot_occupied[start:end] = torch.tensor(
            [p.parsed_state.slot_occupied for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.slot_tapped[start:end] = torch.tensor(
            [p.parsed_state.slot_tapped for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.game_info[start:end] = torch.tensor(
            [p.parsed_state.game_info for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.pending_kind_id[start:end] = torch.tensor(
            [p.parsed_action.pending_kind_id for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.option_kind_ids[start:end] = torch.tensor(
            [p.parsed_action.option_kind_ids for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.option_scalars[start:end] = torch.tensor(
            [p.parsed_action.option_scalars for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.option_mask[start:end] = torch.tensor(
            [p.parsed_action.option_mask for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.option_ref_slot_idx[start:end] = torch.tensor(
            [p.parsed_action.option_ref_slot_idx for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.option_ref_card_row[start:end] = torch.tensor(
            [p.parsed_action.option_ref_card_row for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.target_mask[start:end] = torch.tensor(
            [p.parsed_action.target_mask for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.target_type_ids[start:end] = torch.tensor(
            [p.parsed_action.target_type_ids for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.target_scalars[start:end] = torch.tensor(
            [p.parsed_action.target_scalars for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.target_overflow[start:end] = torch.tensor(
            [p.parsed_action.target_overflow for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.target_ref_slot_idx[start:end] = torch.tensor(
            [p.parsed_action.target_ref_slot_idx for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.target_ref_is_player[start:end] = torch.tensor(
            [p.parsed_action.target_ref_is_player for p in parsed_steps],
            dtype=torch.bool,
            device=device,
        )
        self.target_ref_is_self[start:end] = torch.tensor(
            [p.parsed_action.target_ref_is_self for p in parsed_steps],
            dtype=torch.bool,
            device=device,
        )

        decision_counts = [len(p.decision_option_idx) for p in parsed_steps]
        total = sum(decision_counts)
        dstart = self._decision_cursor
        if dstart + total > self.decision_capacity:
            raise RuntimeError(
                f"decision buffer capacity {self.decision_capacity} exceeded "
                f"(cursor={dstart}, add={total})"
            )
        decision_starts: list[int] = []
        cursor = dstart
        for count in decision_counts:
            decision_starts.append(cursor)
            cursor += count

        if total > 0:
            flat_option_idx: list[list[int]] = []
            flat_target_idx: list[list[int]] = []
            flat_mask: list[list[bool]] = []
            flat_uses_none: list[bool] = []
            for p in parsed_steps:
                flat_option_idx.extend(p.decision_option_idx)
                flat_target_idx.extend(p.decision_target_idx)
                flat_mask.extend(p.decision_mask)
                flat_uses_none.extend(p.uses_none_head)
            dend = dstart + total
            self.decision_option_idx[dstart:dend] = torch.tensor(
                flat_option_idx, dtype=torch.long, device=device
            )
            self.decision_target_idx[dstart:dend] = torch.tensor(
                flat_target_idx, dtype=torch.long, device=device
            )
            self.decision_mask[dstart:dend] = torch.tensor(
                flat_mask, dtype=torch.bool, device=device
            )
            self.uses_none_head[dstart:dend] = torch.tensor(
                flat_uses_none, dtype=torch.bool, device=device
            )

        self._step_cursor = end
        self._decision_cursor += total

        step_indices = torch.arange(start, end, dtype=torch.long, device=device)
        return BufferWrite(
            step_indices=step_indices,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )
