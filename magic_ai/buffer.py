"""Preallocated GPU buffers for PPO rollout data.

Replaces per-step ``torch.tensor`` allocations in the rollout hot path with
bulk writes into a persistent set of buffers. Each parsed step is copied
into the buffer with one ``torch.tensor(list)`` call per field, then later
gathered by index from both the actor path (sampling) and the learner path
(PPO evaluation). Buffers are registered non-persistent so they do not land
in checkpoints.
"""

from __future__ import annotations

from bisect import bisect_left
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
        _reg("trace_kind_id", (capacity,), torch.long)
        _reg("decision_start", (capacity,), torch.long)
        _reg("decision_count", (capacity,), torch.long)
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

        self._free_step_rows = list(range(capacity - 1, -1, -1))
        self._free_decision_ranges: list[tuple[int, int]] = [(0, decision_capacity)]

    # Declared so type checkers see buffer Tensors as Tensors.
    slot_card_rows: Tensor
    slot_occupied: Tensor
    slot_tapped: Tensor
    game_info: Tensor
    trace_kind_id: Tensor
    decision_start: Tensor
    decision_count: Tensor
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
        self._free_step_rows = list(range(self.capacity - 1, -1, -1))
        self._free_decision_ranges = [(0, self.decision_capacity)]

    @property
    def active_step_count(self) -> int:
        return self.capacity - len(self._free_step_rows)

    def _allocate_step_rows(self, count: int) -> list[int]:
        if count > len(self._free_step_rows):
            raise RuntimeError(
                f"rollout buffer capacity {self.capacity} exceeded "
                f"(active={self.active_step_count}, add={count})"
            )
        return [self._free_step_rows.pop() for _ in range(count)]

    def _allocate_decision_range(self, count: int) -> int:
        if count == 0:
            return 0
        for idx, (start, length) in enumerate(self._free_decision_ranges):
            if length < count:
                continue
            alloc_start = start
            remaining = length - count
            if remaining == 0:
                del self._free_decision_ranges[idx]
            else:
                self._free_decision_ranges[idx] = (start + count, remaining)
            return alloc_start
        active = self.decision_capacity - sum(length for _, length in self._free_decision_ranges)
        raise RuntimeError(
            f"decision buffer capacity {self.decision_capacity} exceeded "
            f"(active={active}, add={count})"
        )

    def _free_decision_range(self, start: int, count: int) -> None:
        if count == 0:
            return
        starts = [range_start for range_start, _ in self._free_decision_ranges]
        insert_at = bisect_left(starts, start)
        self._free_decision_ranges.insert(insert_at, (start, count))

        merged: list[tuple[int, int]] = []
        for range_start, range_len in self._free_decision_ranges:
            if not merged:
                merged.append((range_start, range_len))
                continue
            prev_start, prev_len = merged[-1]
            prev_end = prev_start + prev_len
            if range_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, range_start + range_len) - prev_start)
            else:
                merged.append((range_start, range_len))
        self._free_decision_ranges = merged

    def release_steps(self, step_indices: list[int]) -> None:
        if not step_indices:
            return
        idx_t = torch.tensor(step_indices, dtype=torch.long, device=self.device)
        decision_starts = self.decision_start[idx_t].detach().cpu().tolist()
        decision_counts = self.decision_count[idx_t].detach().cpu().tolist()
        for step_idx in step_indices:
            self._free_step_rows.append(step_idx)
        for start, count in zip(decision_starts, decision_counts, strict=True):
            self._free_decision_range(int(start), int(count))

    def ingest_batch(self, parsed_steps: list[ParsedStep]) -> BufferWrite:
        n = len(parsed_steps)
        device = self.device
        if n == 0:
            return BufferWrite(
                step_indices=torch.zeros(0, dtype=torch.long, device=device),
                decision_starts=[],
                decision_counts=[],
            )

        step_rows = self._allocate_step_rows(n)
        step_indices = torch.tensor(step_rows, dtype=torch.long, device=device)

        self.slot_card_rows[step_indices] = torch.tensor(
            [p.parsed_state.slot_card_rows for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.slot_occupied[step_indices] = torch.tensor(
            [p.parsed_state.slot_occupied for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.slot_tapped[step_indices] = torch.tensor(
            [p.parsed_state.slot_tapped for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.game_info[step_indices] = torch.tensor(
            [p.parsed_state.game_info for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.trace_kind_id[step_indices] = torch.tensor(
            [p.trace_kind_id for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.pending_kind_id[step_indices] = torch.tensor(
            [p.parsed_action.pending_kind_id for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.option_kind_ids[step_indices] = torch.tensor(
            [p.parsed_action.option_kind_ids for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.option_scalars[step_indices] = torch.tensor(
            [p.parsed_action.option_scalars for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.option_mask[step_indices] = torch.tensor(
            [p.parsed_action.option_mask for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.option_ref_slot_idx[step_indices] = torch.tensor(
            [p.parsed_action.option_ref_slot_idx for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.option_ref_card_row[step_indices] = torch.tensor(
            [p.parsed_action.option_ref_card_row for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.target_mask[step_indices] = torch.tensor(
            [p.parsed_action.target_mask for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.target_type_ids[step_indices] = torch.tensor(
            [p.parsed_action.target_type_ids for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.target_scalars[step_indices] = torch.tensor(
            [p.parsed_action.target_scalars for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.target_overflow[step_indices] = torch.tensor(
            [p.parsed_action.target_overflow for p in parsed_steps],
            dtype=torch.float32,
            device=device,
        )
        self.target_ref_slot_idx[step_indices] = torch.tensor(
            [p.parsed_action.target_ref_slot_idx for p in parsed_steps],
            dtype=torch.long,
            device=device,
        )
        self.target_ref_is_player[step_indices] = torch.tensor(
            [p.parsed_action.target_ref_is_player for p in parsed_steps],
            dtype=torch.bool,
            device=device,
        )
        self.target_ref_is_self[step_indices] = torch.tensor(
            [p.parsed_action.target_ref_is_self for p in parsed_steps],
            dtype=torch.bool,
            device=device,
        )

        decision_counts = [len(p.decision_option_idx) for p in parsed_steps]
        decision_starts: list[int] = []
        for parsed, count in zip(parsed_steps, decision_counts, strict=True):
            start = self._allocate_decision_range(count)
            decision_starts.append(start)
            if count == 0:
                continue
            end = start + count
            self.decision_option_idx[start:end] = torch.tensor(
                parsed.decision_option_idx, dtype=torch.long, device=device
            )
            self.decision_target_idx[start:end] = torch.tensor(
                parsed.decision_target_idx, dtype=torch.long, device=device
            )
            self.decision_mask[start:end] = torch.tensor(
                parsed.decision_mask, dtype=torch.bool, device=device
            )
            self.uses_none_head[start:end] = torch.tensor(
                parsed.uses_none_head, dtype=torch.bool, device=device
            )

        self.decision_start[step_indices] = torch.tensor(
            decision_starts,
            dtype=torch.long,
            device=device,
        )
        self.decision_count[step_indices] = torch.tensor(
            decision_counts,
            dtype=torch.long,
            device=device,
        )

        return BufferWrite(
            step_indices=step_indices,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )
