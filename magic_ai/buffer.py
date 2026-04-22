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
    from magic_ai.model import ParsedBatch, ParsedStep
    from magic_ai.native_encoder import NativeEncodedBatch


@dataclass(frozen=True)
class BufferWrite:
    """Indices and decision offsets returned from one ingest pass."""

    step_indices: Tensor  # Long [n] — buffer rows written this call
    decision_starts: list[int]  # per-step start row in decision buffer
    decision_counts: list[int]  # per-step number of decision rows


class NativeTrajectoryBuffer(nn.Module):
    """GPU staging storage for live native-rollout trajectories."""

    def __init__(
        self,
        *,
        num_envs: int,
        max_steps_per_trajectory: int,
        decision_capacity_per_env: int,
        max_options: int,
        max_targets_per_option: int,
        max_cached_choices: int,
        zone_slot_count: int,
        game_info_dim: int,
        option_scalar_dim: int,
        target_scalar_dim: int,
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.decision_capacity_per_env = decision_capacity_per_env

        def _reg(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
            self.register_buffer(name, torch.zeros(shape, dtype=dtype), persistent=False)

        step_shape = (num_envs, max_steps_per_trajectory)
        _reg("slot_card_rows", step_shape + (zone_slot_count,), torch.long)
        _reg("slot_occupied", step_shape + (zone_slot_count,), torch.float32)
        _reg("slot_tapped", step_shape + (zone_slot_count,), torch.float32)
        _reg("game_info", step_shape + (game_info_dim,), torch.float32)
        _reg("trace_kind_id", step_shape, torch.long)
        _reg("decision_start", step_shape, torch.long)
        _reg("decision_count", step_shape, torch.long)
        _reg("pending_kind_id", step_shape, torch.long)
        _reg("option_kind_ids", step_shape + (max_options,), torch.long)
        _reg("option_scalars", step_shape + (max_options, option_scalar_dim), torch.float32)
        _reg("option_mask", step_shape + (max_options,), torch.float32)
        _reg("option_ref_slot_idx", step_shape + (max_options,), torch.long)
        _reg("option_ref_card_row", step_shape + (max_options,), torch.long)
        _reg("target_mask", step_shape + (max_options, max_targets_per_option), torch.float32)
        _reg("target_type_ids", step_shape + (max_options, max_targets_per_option), torch.long)
        _reg(
            "target_scalars",
            step_shape + (max_options, max_targets_per_option, target_scalar_dim),
            torch.float32,
        )
        _reg("target_overflow", step_shape + (max_options,), torch.float32)
        _reg("target_ref_slot_idx", step_shape + (max_options, max_targets_per_option), torch.long)
        _reg(
            "target_ref_is_player",
            step_shape + (max_options, max_targets_per_option),
            torch.bool,
        )
        _reg(
            "target_ref_is_self",
            step_shape + (max_options, max_targets_per_option),
            torch.bool,
        )
        _reg("may_selected", step_shape, torch.float32)
        _reg("old_log_prob", step_shape, torch.float32)
        _reg("value", step_shape, torch.float32)
        _reg("perspective_player_idx", step_shape, torch.long)

        decision_shape = (num_envs, decision_capacity_per_env)
        _reg("decision_option_idx", decision_shape + (max_cached_choices,), torch.long)
        _reg("decision_target_idx", decision_shape + (max_cached_choices,), torch.long)
        _reg("decision_mask", decision_shape + (max_cached_choices,), torch.bool)
        _reg("uses_none_head", decision_shape, torch.bool)
        _reg("selected_indices", decision_shape, torch.long)

        _reg("step_count", (num_envs,), torch.long)
        _reg("decision_cursor", (num_envs,), torch.long)

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
    old_log_prob: Tensor
    value: Tensor
    perspective_player_idx: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    selected_indices: Tensor
    step_count: Tensor
    decision_cursor: Tensor

    @property
    def device(self) -> torch.device:
        return self.slot_card_rows.device

    def active_step_count(self, env_idx: int) -> int:
        return int(self.step_count[env_idx].item())

    def reset_env(self, env_idx: int) -> None:
        self.step_count[env_idx] = 0
        self.decision_cursor[env_idx] = 0

    def stage_batch(
        self,
        env_indices: list[int],
        native_batch: NativeEncodedBatch,
        *,
        selected_choice_cols: list[tuple[int, ...]],
        may_selected: list[int],
        old_log_probs: Tensor,
        values: Tensor,
        perspective_player_indices: list[int],
    ) -> None:
        if int(native_batch.trace_kind_id.shape[0]) != len(env_indices):
            raise ValueError("env_indices length must match native batch length")
        if len(env_indices) != len(selected_choice_cols):
            raise ValueError("selected_choice_cols length must match native batch length")
        if len(env_indices) != len(may_selected):
            raise ValueError("may_selected length must match native batch length")
        if int(old_log_probs.numel()) != len(env_indices):
            raise ValueError("old_log_probs length must match native batch length")
        if int(values.numel()) != len(env_indices):
            raise ValueError("values length must match native batch length")
        if len(env_indices) != len(perspective_player_indices):
            raise ValueError("perspective_player_indices length must match native batch length")

        device = self.device
        player_idx_t = torch.tensor(perspective_player_indices, dtype=torch.long, device=device)

        for batch_idx, env_idx in enumerate(env_indices):
            step_idx = int(self.step_count[env_idx].item())
            if step_idx >= self.max_steps_per_trajectory:
                raise RuntimeError(
                    "staging step capacity "
                    f"{self.max_steps_per_trajectory} exceeded for env {env_idx}"
                )

            self.slot_card_rows[env_idx, step_idx] = native_batch.slot_card_rows[batch_idx].to(
                device
            )
            self.slot_occupied[env_idx, step_idx] = native_batch.slot_occupied[batch_idx].to(device)
            self.slot_tapped[env_idx, step_idx] = native_batch.slot_tapped[batch_idx].to(device)
            self.game_info[env_idx, step_idx] = native_batch.game_info[batch_idx].to(device)
            self.trace_kind_id[env_idx, step_idx] = native_batch.trace_kind_id[batch_idx].to(device)
            self.pending_kind_id[env_idx, step_idx] = native_batch.pending_kind_id[batch_idx].to(
                device
            )
            self.option_kind_ids[env_idx, step_idx] = native_batch.option_kind_ids[batch_idx].to(
                device
            )
            self.option_scalars[env_idx, step_idx] = native_batch.option_scalars[batch_idx].to(
                device
            )
            self.option_mask[env_idx, step_idx] = native_batch.option_mask[batch_idx].to(device)
            self.option_ref_slot_idx[env_idx, step_idx] = native_batch.option_ref_slot_idx[
                batch_idx
            ].to(device)
            self.option_ref_card_row[env_idx, step_idx] = native_batch.option_ref_card_row[
                batch_idx
            ].to(device)
            self.target_mask[env_idx, step_idx] = native_batch.target_mask[batch_idx].to(device)
            self.target_type_ids[env_idx, step_idx] = native_batch.target_type_ids[batch_idx].to(
                device
            )
            self.target_scalars[env_idx, step_idx] = native_batch.target_scalars[batch_idx].to(
                device
            )
            self.target_overflow[env_idx, step_idx] = native_batch.target_overflow[batch_idx].to(
                device
            )
            self.target_ref_slot_idx[env_idx, step_idx] = native_batch.target_ref_slot_idx[
                batch_idx
            ].to(device)
            self.target_ref_is_player[env_idx, step_idx] = native_batch.target_ref_is_player[
                batch_idx
            ].to(device)
            self.target_ref_is_self[env_idx, step_idx] = native_batch.target_ref_is_self[
                batch_idx
            ].to(device)
            self.may_selected[env_idx, step_idx] = may_selected[batch_idx]
            self.old_log_prob[env_idx, step_idx] = old_log_probs[batch_idx].to(device)
            self.value[env_idx, step_idx] = values[batch_idx].to(device)
            self.perspective_player_idx[env_idx, step_idx] = player_idx_t[batch_idx]

            count = int(native_batch.decision_count[batch_idx].item())
            start = int(self.decision_cursor[env_idx].item())
            if start + count > self.decision_capacity_per_env:
                raise RuntimeError(
                    "staging decision capacity "
                    f"{self.decision_capacity_per_env} exceeded for env {env_idx}"
                )
            self.decision_start[env_idx, step_idx] = start
            self.decision_count[env_idx, step_idx] = count
            if count:
                source_start = int(native_batch.decision_start[batch_idx].item())
                source_end = source_start + count
                dest_end = start + count
                self.decision_option_idx[env_idx, start:dest_end] = (
                    native_batch.decision_option_idx[source_start:source_end].to(device)
                )
                self.decision_target_idx[env_idx, start:dest_end] = (
                    native_batch.decision_target_idx[source_start:source_end].to(device)
                )
                self.decision_mask[env_idx, start:dest_end] = native_batch.decision_mask[
                    source_start:source_end
                ].to(device)
                self.uses_none_head[env_idx, start:dest_end] = native_batch.uses_none_head[
                    source_start:source_end
                ].to(device)
                cols = selected_choice_cols[batch_idx]
                if len(cols) != count:
                    raise ValueError("selected_choice_cols entry must match decision_count")
                self.selected_indices[env_idx, start:dest_end] = torch.tensor(
                    cols,
                    dtype=torch.long,
                    device=device,
                )
            self.step_count[env_idx] = step_idx + 1
            self.decision_cursor[env_idx] = start + count


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

        self._step_cursor = 0
        self._decision_cursor = 0

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
        self._step_cursor = 0
        self._decision_cursor = 0

    @property
    def active_step_count(self) -> int:
        return self._step_cursor

    def _allocate_step_rows(self, count: int) -> list[int]:
        if self._step_cursor + count > self.capacity:
            raise RuntimeError(
                f"rollout buffer capacity {self.capacity} exceeded "
                f"(active={self.active_step_count}, add={count})"
            )
        start = self._step_cursor
        self._step_cursor += count
        return list(range(start, start + count))

    def _allocate_decision_range(self, count: int) -> int:
        if count == 0:
            return 0
        if self._decision_cursor + count > self.decision_capacity:
            raise RuntimeError(
                f"decision buffer capacity {self.decision_capacity} exceeded "
                f"(active={self._decision_cursor}, add={count})"
            )
        start = self._decision_cursor
        self._decision_cursor += count
        return start

    def release_steps(self, step_indices: list[int]) -> None:
        return

    def ingest_batch(self, parsed_steps: list[ParsedStep]) -> BufferWrite:
        return self.ingest_batch_legacy(parsed_steps)

    def ingest_parsed_batch(self, parsed_batch: ParsedBatch) -> BufferWrite:
        n = int(parsed_batch.trace_kind_ids.shape[0])
        device = self.device
        if n == 0:
            return BufferWrite(
                step_indices=torch.zeros(0, dtype=torch.long, device=device),
                decision_starts=[],
                decision_counts=[],
            )

        step_rows = self._allocate_step_rows(n)
        step_indices = torch.tensor(step_rows, dtype=torch.long, device=device)

        self.slot_card_rows[step_indices] = parsed_batch.parsed_state.slot_card_rows.to(device)
        self.slot_occupied[step_indices] = parsed_batch.parsed_state.slot_occupied.to(device)
        self.slot_tapped[step_indices] = parsed_batch.parsed_state.slot_tapped.to(device)
        self.game_info[step_indices] = parsed_batch.parsed_state.game_info.to(device)
        self.trace_kind_id[step_indices] = parsed_batch.trace_kind_ids.to(device)
        self.pending_kind_id[step_indices] = parsed_batch.parsed_action.pending_kind_id.to(device)
        self.option_kind_ids[step_indices] = parsed_batch.parsed_action.option_kind_ids.to(device)
        self.option_scalars[step_indices] = parsed_batch.parsed_action.option_scalars.to(device)
        self.option_mask[step_indices] = parsed_batch.parsed_action.option_mask.to(device)
        self.option_ref_slot_idx[step_indices] = parsed_batch.parsed_action.option_ref_slot_idx.to(
            device
        )
        self.option_ref_card_row[step_indices] = parsed_batch.parsed_action.option_ref_card_row.to(
            device
        )
        self.target_mask[step_indices] = parsed_batch.parsed_action.target_mask.to(device)
        self.target_type_ids[step_indices] = parsed_batch.parsed_action.target_type_ids.to(device)
        self.target_scalars[step_indices] = parsed_batch.parsed_action.target_scalars.to(device)
        self.target_overflow[step_indices] = parsed_batch.parsed_action.target_overflow.to(device)
        self.target_ref_slot_idx[step_indices] = parsed_batch.parsed_action.target_ref_slot_idx.to(
            device
        )
        self.target_ref_is_player[step_indices] = (
            parsed_batch.parsed_action.target_ref_is_player.to(device)
        )
        self.target_ref_is_self[step_indices] = parsed_batch.parsed_action.target_ref_is_self.to(
            device
        )

        decision_counts = parsed_batch.decision_counts
        decision_starts: list[int] = []
        flat_cursor = 0
        for count in decision_counts:
            start = self._allocate_decision_range(count)
            decision_starts.append(start)
            if count == 0:
                continue
            end = start + count
            flat_end = flat_cursor + count
            self.decision_option_idx[start:end] = parsed_batch.decision_option_idx[
                flat_cursor:flat_end
            ].to(device)
            self.decision_target_idx[start:end] = parsed_batch.decision_target_idx[
                flat_cursor:flat_end
            ].to(device)
            self.decision_mask[start:end] = parsed_batch.decision_mask[flat_cursor:flat_end].to(
                device
            )
            self.uses_none_head[start:end] = parsed_batch.uses_none_head[flat_cursor:flat_end].to(
                device
            )
            flat_cursor = flat_end

        self.decision_start[step_indices] = torch.tensor(
            decision_starts, dtype=torch.long, device=device
        )
        self.decision_count[step_indices] = torch.tensor(
            decision_counts, dtype=torch.long, device=device
        )

        return BufferWrite(
            step_indices=step_indices,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )

    def ingest_native_batch(self, native_batch: NativeEncodedBatch) -> BufferWrite:
        n = int(native_batch.trace_kind_id.shape[0])
        device = self.device
        if n == 0:
            return BufferWrite(
                step_indices=torch.zeros(0, dtype=torch.long, device=device),
                decision_starts=[],
                decision_counts=[],
            )

        step_rows = self._allocate_step_rows(n)
        step_indices = torch.tensor(step_rows, dtype=torch.long, device=device)

        self.slot_card_rows[step_indices] = native_batch.slot_card_rows.to(device)
        self.slot_occupied[step_indices] = native_batch.slot_occupied.to(device)
        self.slot_tapped[step_indices] = native_batch.slot_tapped.to(device)
        self.game_info[step_indices] = native_batch.game_info.to(device)
        self.trace_kind_id[step_indices] = native_batch.trace_kind_id.to(device)
        self.pending_kind_id[step_indices] = native_batch.pending_kind_id.to(device)
        self.option_kind_ids[step_indices] = native_batch.option_kind_ids.to(device)
        self.option_scalars[step_indices] = native_batch.option_scalars.to(device)
        self.option_mask[step_indices] = native_batch.option_mask.to(device)
        self.option_ref_slot_idx[step_indices] = native_batch.option_ref_slot_idx.to(device)
        self.option_ref_card_row[step_indices] = native_batch.option_ref_card_row.to(device)
        self.target_mask[step_indices] = native_batch.target_mask.to(device)
        self.target_type_ids[step_indices] = native_batch.target_type_ids.to(device)
        self.target_scalars[step_indices] = native_batch.target_scalars.to(device)
        self.target_overflow[step_indices] = native_batch.target_overflow.to(device)
        self.target_ref_slot_idx[step_indices] = native_batch.target_ref_slot_idx.to(device)
        self.target_ref_is_player[step_indices] = native_batch.target_ref_is_player.to(device)
        self.target_ref_is_self[step_indices] = native_batch.target_ref_is_self.to(device)

        decision_counts = native_batch.decision_count.detach().cpu().tolist()
        decision_starts: list[int] = []
        flat_cursor = 0
        for count in decision_counts:
            start = self._allocate_decision_range(count)
            decision_starts.append(start)
            if count == 0:
                continue
            end = start + count
            flat_end = flat_cursor + count
            self.decision_option_idx[start:end] = native_batch.decision_option_idx[
                flat_cursor:flat_end
            ].to(device)
            self.decision_target_idx[start:end] = native_batch.decision_target_idx[
                flat_cursor:flat_end
            ].to(device)
            self.decision_mask[start:end] = native_batch.decision_mask[flat_cursor:flat_end].to(
                device
            )
            self.uses_none_head[start:end] = native_batch.uses_none_head[flat_cursor:flat_end].to(
                device
            )
            flat_cursor = flat_end

        self.decision_start[step_indices] = torch.tensor(
            decision_starts, dtype=torch.long, device=device
        )
        self.decision_count[step_indices] = torch.tensor(
            decision_counts, dtype=torch.long, device=device
        )

        return BufferWrite(
            step_indices=step_indices,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )

    def ingest_staged_episode(
        self,
        staging: NativeTrajectoryBuffer,
        env_idx: int,
    ) -> BufferWrite:
        n = staging.active_step_count(env_idx)
        device = self.device
        if n == 0:
            return BufferWrite(
                step_indices=torch.zeros(0, dtype=torch.long, device=device),
                decision_starts=[],
                decision_counts=[],
            )

        step_rows = self._allocate_step_rows(n)
        step_indices = torch.tensor(step_rows, dtype=torch.long, device=device)

        self.slot_card_rows[step_indices] = staging.slot_card_rows[env_idx, :n].to(device)
        self.slot_occupied[step_indices] = staging.slot_occupied[env_idx, :n].to(device)
        self.slot_tapped[step_indices] = staging.slot_tapped[env_idx, :n].to(device)
        self.game_info[step_indices] = staging.game_info[env_idx, :n].to(device)
        self.trace_kind_id[step_indices] = staging.trace_kind_id[env_idx, :n].to(device)
        self.pending_kind_id[step_indices] = staging.pending_kind_id[env_idx, :n].to(device)
        self.option_kind_ids[step_indices] = staging.option_kind_ids[env_idx, :n].to(device)
        self.option_scalars[step_indices] = staging.option_scalars[env_idx, :n].to(device)
        self.option_mask[step_indices] = staging.option_mask[env_idx, :n].to(device)
        self.option_ref_slot_idx[step_indices] = staging.option_ref_slot_idx[env_idx, :n].to(device)
        self.option_ref_card_row[step_indices] = staging.option_ref_card_row[env_idx, :n].to(device)
        self.target_mask[step_indices] = staging.target_mask[env_idx, :n].to(device)
        self.target_type_ids[step_indices] = staging.target_type_ids[env_idx, :n].to(device)
        self.target_scalars[step_indices] = staging.target_scalars[env_idx, :n].to(device)
        self.target_overflow[step_indices] = staging.target_overflow[env_idx, :n].to(device)
        self.target_ref_slot_idx[step_indices] = staging.target_ref_slot_idx[env_idx, :n].to(device)
        self.target_ref_is_player[step_indices] = staging.target_ref_is_player[env_idx, :n].to(
            device
        )
        self.target_ref_is_self[step_indices] = staging.target_ref_is_self[env_idx, :n].to(device)
        self.may_selected[step_indices] = staging.may_selected[env_idx, :n].to(device)

        decision_counts = staging.decision_count[env_idx, :n].detach().cpu().tolist()
        source_starts = staging.decision_start[env_idx, :n].detach().cpu().tolist()
        decision_starts: list[int] = []
        for source_start, count in zip(source_starts, decision_counts, strict=True):
            start = self._allocate_decision_range(int(count))
            decision_starts.append(start)
            if count == 0:
                continue
            source_end = int(source_start) + int(count)
            end = start + int(count)
            self.decision_option_idx[start:end] = staging.decision_option_idx[
                env_idx, int(source_start) : source_end
            ].to(device)
            self.decision_target_idx[start:end] = staging.decision_target_idx[
                env_idx, int(source_start) : source_end
            ].to(device)
            self.decision_mask[start:end] = staging.decision_mask[
                env_idx, int(source_start) : source_end
            ].to(device)
            self.uses_none_head[start:end] = staging.uses_none_head[
                env_idx, int(source_start) : source_end
            ].to(device)
            self.selected_indices[start:end] = staging.selected_indices[
                env_idx, int(source_start) : source_end
            ].to(device)

        self.decision_start[step_indices] = torch.tensor(
            decision_starts, dtype=torch.long, device=device
        )
        self.decision_count[step_indices] = torch.tensor(
            decision_counts, dtype=torch.long, device=device
        )

        return BufferWrite(
            step_indices=step_indices,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )

    def ingest_batch_legacy(self, parsed_steps: list[ParsedStep]) -> BufferWrite:
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
