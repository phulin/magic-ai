"""Backend-neutral replay-buffer storage helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DenseDecisionBatch:
    """Per-step decision groups gathered into a dense minibatch layout."""

    decision_start: Tensor
    decision_count: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    selected_indices: Tensor


@dataclass(frozen=True)
class ReplayCommonBatch:
    """Shared rollout metadata gathered from replay rows."""

    trace_kind_id: Tensor
    may_selected: Tensor
    old_log_prob: Tensor
    value: Tensor
    perspective_player_idx: Tensor
    lstm_h_in: Tensor | None
    lstm_c_in: Tensor | None


class ReplayCore(nn.Module):
    """Shared replay row, rollout metadata, target, and decision storage."""

    def __init__(
        self,
        *,
        capacity: int,
        decision_capacity: int,
        max_decision_groups: int,
        max_cached_choices: int,
        device: torch.device | str,
        recurrent_layers: int = 0,
        recurrent_hidden_dim: int = 0,
        index_dtype: torch.dtype = torch.long,
        trace_dtype: torch.dtype = torch.long,
        perspective_dtype: torch.dtype = torch.long,
    ) -> None:
        super().__init__()
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if decision_capacity < 1:
            raise ValueError("decision_capacity must be at least 1")
        if max_decision_groups < 1:
            raise ValueError("max_decision_groups must be at least 1")
        if max_cached_choices < 1:
            raise ValueError("max_cached_choices must be at least 1")
        if (recurrent_layers == 0) != (recurrent_hidden_dim == 0):
            raise ValueError(
                "recurrent_layers and recurrent_hidden_dim must both be zero or nonzero"
            )
        self.capacity = int(capacity)
        self.decision_capacity = int(decision_capacity)
        self.max_decision_groups = int(max_decision_groups)
        self.max_cached_choices = int(max_cached_choices)
        self.recurrent_layers = int(recurrent_layers)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.index_dtype = index_dtype
        device_t = torch.device(device)

        self.register_buffer(
            "trace_kind_id",
            torch.zeros(self.capacity, dtype=trace_dtype, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "may_selected",
            torch.zeros(self.capacity, dtype=torch.float32, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "old_log_prob",
            torch.zeros(self.capacity, dtype=torch.float32, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "value",
            torch.zeros(self.capacity, dtype=torch.float32, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "perspective_player_idx",
            torch.zeros(self.capacity, dtype=perspective_dtype, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "ppo_return",
            torch.zeros(self.capacity, dtype=torch.float32, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "ppo_advantage",
            torch.zeros(self.capacity, dtype=torch.float32, device=device_t),
            persistent=False,
        )
        if self.recurrent_layers > 0:
            recurrent_shape = (
                self.capacity,
                self.recurrent_layers,
                self.recurrent_hidden_dim,
            )
            self.register_buffer(
                "lstm_h_in",
                torch.zeros(recurrent_shape, dtype=torch.float32, device=device_t),
                persistent=False,
            )
            self.register_buffer(
                "lstm_c_in",
                torch.zeros(recurrent_shape, dtype=torch.float32, device=device_t),
                persistent=False,
            )
        else:
            self.register_buffer("lstm_h_in", None, persistent=False)
            self.register_buffer("lstm_c_in", None, persistent=False)

        self.register_buffer(
            "decision_start",
            torch.zeros(self.capacity, dtype=torch.long, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "decision_count",
            torch.zeros(self.capacity, dtype=torch.long, device=device_t),
            persistent=False,
        )
        decision_shape = (self.decision_capacity, self.max_cached_choices)
        self.register_buffer(
            "decision_option_idx",
            torch.full(decision_shape, -1, dtype=self.index_dtype, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "decision_target_idx",
            torch.full(decision_shape, -1, dtype=self.index_dtype, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "decision_mask",
            torch.zeros(decision_shape, dtype=torch.bool, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "uses_none_head",
            torch.zeros(self.decision_capacity, dtype=torch.bool, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "selected_indices",
            torch.full((self.decision_capacity,), -1, dtype=self.index_dtype, device=device_t),
            persistent=False,
        )
        self.register_buffer(
            "_occupied",
            torch.zeros(self.capacity, dtype=torch.bool, device=device_t),
            persistent=False,
        )
        self._free_rows = list(range(self.capacity - 1, -1, -1))
        self._decision_cursor = 0

    @property
    def device(self) -> torch.device:
        return self.trace_kind_id.device

    @property
    def size(self) -> int:
        return int(self._occupied.sum().item())

    trace_kind_id: Tensor
    may_selected: Tensor
    old_log_prob: Tensor
    value: Tensor
    perspective_player_idx: Tensor
    ppo_return: Tensor
    ppo_advantage: Tensor
    decision_start: Tensor
    decision_count: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    selected_indices: Tensor
    lstm_h_in: Tensor | None
    lstm_c_in: Tensor | None
    _occupied: Tensor

    def reset(self) -> None:
        self._occupied.zero_()
        self._free_rows = list(range(self.capacity - 1, -1, -1))
        self.decision_start.zero_()
        self.decision_count.zero_()
        self._decision_cursor = 0

    def allocate_row(self) -> int:
        if not self._free_rows:
            raise RuntimeError("replay buffer is full")
        row = self._free_rows.pop()
        self._occupied[row] = True
        return int(row)

    def allocate_rows(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if len(self._free_rows) < count:
            raise RuntimeError("replay buffer is full")
        rows = [self._free_rows.pop() for _ in range(count)]
        if rows:
            self._occupied[torch.tensor(rows, device=self.device)] = True
        return [int(row) for row in rows]

    def release_rows(self, replay_rows: list[int]) -> None:
        for row in replay_rows:
            self.validate_row(row)
            if bool(self._occupied[row].item()):
                self._occupied[row] = False
                self._free_rows.append(int(row))

    def validate_row(self, row: int) -> None:
        if row < 0 or row >= self.capacity:
            raise IndexError("replay row out of range")

    def validate_occupied(self, rows: Tensor) -> None:
        idx = rows.to(device=self.device)
        in_range = (idx >= 0) & (idx < self.capacity)
        if not bool(in_range.all().item()):
            bad = int(idx[~in_range][0].item())
            raise ValueError(f"replay row {bad} out of range")
        occupied = self._occupied[idx]
        if not bool(occupied.all().item()):
            bad = int(idx[~occupied][0].item())
            raise ValueError(f"replay row {bad} is not occupied")

    def write_common_row(
        self,
        row: int,
        *,
        trace_kind_id: int,
        may_selected: float,
        old_log_prob: float,
        value: float,
        perspective_player_idx: int,
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> None:
        self.trace_kind_id[row] = int(trace_kind_id)
        self.may_selected[row] = float(may_selected)
        self.old_log_prob[row] = float(old_log_prob)
        self.value[row] = float(value)
        self.perspective_player_idx[row] = int(perspective_player_idx)
        self.write_recurrent_row(row, lstm_h_in, lstm_c_in)

    def write_common_batch(
        self,
        rows: Tensor,
        *,
        trace_kind_id: Tensor,
        may_selected: Tensor,
        old_log_prob: Tensor,
        value: Tensor,
        perspective_player_idx: Tensor,
    ) -> None:
        idx = rows.to(device=self.device)
        self.trace_kind_id[idx] = trace_kind_id.to(
            device=self.device, dtype=self.trace_kind_id.dtype
        )
        self.may_selected[idx] = may_selected.to(device=self.device, dtype=torch.float32)
        self.old_log_prob[idx] = old_log_prob.to(device=self.device, dtype=torch.float32)
        self.value[idx] = value.to(device=self.device, dtype=torch.float32)
        self.perspective_player_idx[idx] = perspective_player_idx.to(
            device=self.device, dtype=self.perspective_player_idx.dtype
        )

    def gather_common(self, rows: Tensor) -> ReplayCommonBatch:
        idx = rows.to(device=self.device)
        h_in = self.lstm_h_in
        c_in = self.lstm_c_in
        return ReplayCommonBatch(
            trace_kind_id=self.trace_kind_id[idx],
            may_selected=self.may_selected[idx],
            old_log_prob=self.old_log_prob[idx],
            value=self.value[idx],
            perspective_player_idx=self.perspective_player_idx[idx],
            lstm_h_in=h_in[idx] if h_in is not None else None,
            lstm_c_in=c_in[idx] if c_in is not None else None,
        )

    def write_recurrent_row(
        self,
        row: int,
        h_in: Tensor | None,
        c_in: Tensor | None,
    ) -> None:
        if self.lstm_h_in is None or self.lstm_c_in is None:
            if h_in is not None or c_in is not None:
                raise ValueError("buffer was constructed without recurrent state storage")
            return
        if h_in is None or c_in is None:
            raise ValueError("h_in and c_in are required for recurrent replay")
        expected = (self.recurrent_layers, self.recurrent_hidden_dim)
        if tuple(h_in.shape) != expected or tuple(c_in.shape) != expected:
            raise ValueError(
                f"recurrent state must have shape {expected}, got {tuple(h_in.shape)} "
                f"and {tuple(c_in.shape)}"
            )
        self.lstm_h_in[row].copy_(h_in.to(device=self.device, dtype=torch.float32))
        self.lstm_c_in[row].copy_(c_in.to(device=self.device, dtype=torch.float32))

    def write_ppo_targets(
        self,
        replay_rows: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> None:
        idx = replay_rows.to(device=self.device)
        self.old_log_prob[idx] = old_log_probs.to(device=self.device, dtype=torch.float32)
        self.ppo_return[idx] = returns.to(device=self.device, dtype=torch.float32)
        self.ppo_advantage[idx] = advantages.to(device=self.device, dtype=torch.float32)

    def gather_ppo_targets(self, replay_rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        idx = replay_rows.to(device=self.device)
        return self.old_log_prob[idx], self.ppo_return[idx], self.ppo_advantage[idx]

    def write_decision_row(
        self,
        row: int,
        *,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
    ) -> int:
        count = int(decision_option_idx.shape[0])
        stored_count = min(count, self.max_decision_groups)
        start = self.allocate_decision_range(stored_count)
        self.decision_start[row] = start
        self.decision_count[row] = stored_count
        if stored_count == 0:
            return stored_count
        self._validate_group_shapes(
            stored_count,
            decision_option_idx[:stored_count],
            decision_target_idx[:stored_count],
            decision_mask[:stored_count],
            uses_none_head[:stored_count],
            selected_indices[:stored_count],
        )
        end = start + stored_count
        self.decision_option_idx[start:end] = decision_option_idx[:stored_count].to(
            device=self.device, dtype=self.index_dtype
        )
        self.decision_target_idx[start:end] = decision_target_idx[:stored_count].to(
            device=self.device, dtype=self.index_dtype
        )
        self.decision_mask[start:end] = decision_mask[:stored_count].to(
            device=self.device, dtype=torch.bool
        )
        self.uses_none_head[start:end] = uses_none_head[:stored_count].to(
            device=self.device, dtype=torch.bool
        )
        self.selected_indices[start:end] = selected_indices[:stored_count].to(
            device=self.device, dtype=self.index_dtype
        )
        return stored_count

    def write_decision_batch(
        self,
        rows: Tensor,
        *,
        decision_count: Tensor,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
    ) -> Tensor:
        rows = rows.to(device=self.device)
        counts = decision_count.to(device=self.device)
        stored_counts = counts.clamp(max=self.max_decision_groups)
        total_stored = int(stored_counts.sum().item())
        start = self.allocate_decision_range(total_stored)
        starts = torch.cumsum(stored_counts, dim=0) - stored_counts + start
        self.decision_start[rows] = starts
        self.decision_count[rows] = stored_counts
        if total_stored == 0:
            return stored_counts

        total_source = int(decision_option_idx.shape[0])
        if total_source < int(counts.sum().item()):
            raise ValueError("decision tensors do not contain decision_count groups")
        step_for_source = torch.repeat_interleave(
            torch.arange(rows.numel(), device=self.device), counts
        )
        source_group_starts = torch.cumsum(counts, dim=0) - counts
        group_in_step = (
            torch.arange(total_source, device=self.device) - source_group_starts[step_for_source]
        )
        keep = group_in_step < self.max_decision_groups
        if not bool(keep.any().item()):
            return stored_counts
        kept_source = keep.nonzero(as_tuple=False).squeeze(-1)
        kept_steps = step_for_source[kept_source]
        kept_groups = group_in_step[kept_source]
        dest = starts[kept_steps] + kept_groups

        self.decision_option_idx[dest] = decision_option_idx[kept_source].to(
            device=self.device, dtype=self.index_dtype
        )
        self.decision_target_idx[dest] = decision_target_idx[kept_source].to(
            device=self.device, dtype=self.index_dtype
        )
        self.decision_mask[dest] = decision_mask[kept_source].to(
            device=self.device, dtype=torch.bool
        )
        self.uses_none_head[dest] = uses_none_head[kept_source].to(
            device=self.device, dtype=torch.bool
        )
        self.selected_indices[dest] = selected_indices[kept_source].to(
            device=self.device, dtype=self.index_dtype
        )
        return stored_counts

    def allocate_decision_range(self, count: int) -> int:
        if count == 0:
            return self._decision_cursor
        if count < 0:
            raise ValueError("count must be non-negative")
        if self._decision_cursor + count > self.decision_capacity:
            raise RuntimeError(
                f"decision buffer capacity {self.decision_capacity} exceeded "
                f"(active={self._decision_cursor}, add={count})"
            )
        start = self._decision_cursor
        self._decision_cursor += count
        return start

    def gather_dense_decisions(self, rows: Tensor) -> DenseDecisionBatch:
        rows = rows.to(device=self.device)
        batch_size = int(rows.numel())
        dense_shape = (batch_size, self.max_decision_groups, self.max_cached_choices)
        option_idx = torch.full(dense_shape, -1, dtype=self.index_dtype, device=self.device)
        target_idx = torch.full_like(option_idx, -1)
        mask = torch.zeros(dense_shape, dtype=torch.bool, device=self.device)
        uses_none = torch.zeros(
            (batch_size, self.max_decision_groups), dtype=torch.bool, device=self.device
        )
        selected = torch.full(
            (batch_size, self.max_decision_groups),
            -1,
            dtype=self.index_dtype,
            device=self.device,
        )
        starts = self.decision_start[rows]
        counts = self.decision_count[rows]
        total = int(counts.sum().item())
        if total:
            step_pos = torch.repeat_interleave(torch.arange(batch_size, device=self.device), counts)
            group_starts = torch.cumsum(counts, dim=0) - counts
            group_pos = torch.arange(total, device=self.device) - group_starts[step_pos]
            source = starts[step_pos] + group_pos
            option_idx[step_pos, group_pos] = self.decision_option_idx[source]
            target_idx[step_pos, group_pos] = self.decision_target_idx[source]
            mask[step_pos, group_pos] = self.decision_mask[source]
            uses_none[step_pos, group_pos] = self.uses_none_head[source]
            selected[step_pos, group_pos] = self.selected_indices[source]
        return DenseDecisionBatch(
            decision_start=starts,
            decision_count=counts,
            decision_option_idx=option_idx,
            decision_target_idx=target_idx,
            decision_mask=mask,
            uses_none_head=uses_none,
            selected_indices=selected,
        )

    def valid_choice_count(self, rows: Tensor) -> Tensor:
        rows = rows.to(device=self.device)
        starts = self.decision_start[rows]
        counts = self.decision_count[rows]
        total = int(counts.sum().item())
        if total == 0:
            return torch.zeros((), dtype=torch.long, device=self.device)
        step_pos = torch.repeat_interleave(torch.arange(rows.numel(), device=self.device), counts)
        group_starts = torch.cumsum(counts, dim=0) - counts
        group_pos = torch.arange(total, device=self.device) - group_starts[step_pos]
        source = starts[step_pos] + group_pos
        return self.decision_mask[source].sum()

    def _validate_group_shapes(
        self,
        count: int,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
    ) -> None:
        expected = (count, self.max_cached_choices)
        if tuple(decision_option_idx.shape) != expected:
            raise ValueError(f"decision_option_idx must have shape {expected}")
        if tuple(decision_target_idx.shape) != expected:
            raise ValueError(f"decision_target_idx must have shape {expected}")
        if tuple(decision_mask.shape) != expected:
            raise ValueError(f"decision_mask must have shape {expected}")
        if tuple(uses_none_head.shape) != (count,):
            raise ValueError(f"uses_none_head must have shape {(count,)}")
        if tuple(selected_indices.shape) != (count,):
            raise ValueError(f"selected_indices must have shape {(count,)}")


FlatDecisionStorage = ReplayCore

__all__ = ["DenseDecisionBatch", "ReplayCommonBatch", "ReplayCore"]
