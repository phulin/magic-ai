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
        recurrent_layers: int = 0,
        recurrent_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.decision_capacity_per_env = decision_capacity_per_env
        self.recurrent_layers = recurrent_layers
        self.recurrent_hidden_dim = recurrent_hidden_dim

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
        recurrent_shape = step_shape + (recurrent_layers, recurrent_hidden_dim)
        _reg("lstm_h_in", recurrent_shape, torch.float32)
        _reg("lstm_c_in", recurrent_shape, torch.float32)

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
    lstm_h_in: Tensor
    lstm_c_in: Tensor
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
        selected_choice_cols_flat: Tensor,
        may_selected: list[int],
        old_log_probs: Tensor,
        values: Tensor,
        perspective_player_indices: list[int],
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> None:
        if int(native_batch.trace_kind_id.shape[0]) != len(env_indices):
            raise ValueError("env_indices length must match native batch length")
        if len(env_indices) != len(may_selected):
            raise ValueError("may_selected length must match native batch length")
        if int(old_log_probs.numel()) != len(env_indices):
            raise ValueError("old_log_probs length must match native batch length")
        if int(values.numel()) != len(env_indices):
            raise ValueError("values length must match native batch length")
        if len(env_indices) != len(perspective_player_indices):
            raise ValueError("perspective_player_indices length must match native batch length")

        device = self.device
        env_idx_t = torch.tensor(env_indices, dtype=torch.long, device=device)
        step_idx_t = self.step_count[env_idx_t]
        if bool((step_idx_t >= self.max_steps_per_trajectory).any()):
            bad_env = env_indices[
                int((step_idx_t >= self.max_steps_per_trajectory).nonzero()[0].item())
            ]
            raise RuntimeError(
                f"staging step capacity {self.max_steps_per_trajectory} exceeded for env {bad_env}"
            )

        player_idx_t = torch.tensor(perspective_player_indices, dtype=torch.long, device=device)
        may_selected_t = torch.tensor(may_selected, dtype=torch.float32, device=device)
        counts_t = native_batch.decision_count.to(device=device, dtype=torch.long)
        starts_t = self.decision_cursor[env_idx_t]
        selected_choice_cols_flat = selected_choice_cols_flat.to(device=device, dtype=torch.long)
        if int(selected_choice_cols_flat.numel()) != int(counts_t.sum().item()):
            raise ValueError("selected_choice_cols_flat length must match total decision_count")
        if bool(((starts_t + counts_t) > self.decision_capacity_per_env).any()):
            bad_env = env_indices[
                int((((starts_t + counts_t) > self.decision_capacity_per_env).nonzero())[0].item())
            ]
            raise RuntimeError(
                "staging decision capacity "
                f"{self.decision_capacity_per_env} exceeded for env {bad_env}"
            )

        self.slot_card_rows[env_idx_t, step_idx_t] = native_batch.slot_card_rows.to(device)
        self.slot_occupied[env_idx_t, step_idx_t] = native_batch.slot_occupied.to(device)
        self.slot_tapped[env_idx_t, step_idx_t] = native_batch.slot_tapped.to(device)
        self.game_info[env_idx_t, step_idx_t] = native_batch.game_info.to(device)
        self.trace_kind_id[env_idx_t, step_idx_t] = native_batch.trace_kind_id.to(device)
        self.pending_kind_id[env_idx_t, step_idx_t] = native_batch.pending_kind_id.to(device)
        self.option_kind_ids[env_idx_t, step_idx_t] = native_batch.option_kind_ids.to(device)
        self.option_scalars[env_idx_t, step_idx_t] = native_batch.option_scalars.to(device)
        self.option_mask[env_idx_t, step_idx_t] = native_batch.option_mask.to(device)
        self.option_ref_slot_idx[env_idx_t, step_idx_t] = native_batch.option_ref_slot_idx.to(
            device
        )
        self.option_ref_card_row[env_idx_t, step_idx_t] = native_batch.option_ref_card_row.to(
            device
        )
        self.target_mask[env_idx_t, step_idx_t] = native_batch.target_mask.to(device)
        self.target_type_ids[env_idx_t, step_idx_t] = native_batch.target_type_ids.to(device)
        self.target_scalars[env_idx_t, step_idx_t] = native_batch.target_scalars.to(device)
        self.target_overflow[env_idx_t, step_idx_t] = native_batch.target_overflow.to(device)
        self.target_ref_slot_idx[env_idx_t, step_idx_t] = native_batch.target_ref_slot_idx.to(
            device
        )
        self.target_ref_is_player[env_idx_t, step_idx_t] = native_batch.target_ref_is_player.to(
            device=device, dtype=torch.bool
        )
        self.target_ref_is_self[env_idx_t, step_idx_t] = native_batch.target_ref_is_self.to(
            device=device, dtype=torch.bool
        )
        self.may_selected[env_idx_t, step_idx_t] = may_selected_t
        self.old_log_prob[env_idx_t, step_idx_t] = old_log_probs.to(device)
        self.value[env_idx_t, step_idx_t] = values.to(device)
        self.perspective_player_idx[env_idx_t, step_idx_t] = player_idx_t
        if self.recurrent_layers and self.recurrent_hidden_dim:
            if lstm_h_in is None or lstm_c_in is None:
                raise ValueError("lstm_h_in and lstm_c_in are required for recurrent staging")
            expected_shape = (
                len(env_indices),
                self.recurrent_layers,
                self.recurrent_hidden_dim,
            )
            if tuple(lstm_h_in.shape) != expected_shape or tuple(lstm_c_in.shape) != expected_shape:
                raise ValueError(
                    "lstm_h_in/lstm_c_in shape must be "
                    f"{expected_shape}, got {tuple(lstm_h_in.shape)} and {tuple(lstm_c_in.shape)}"
                )
            self.lstm_h_in[env_idx_t, step_idx_t] = lstm_h_in.to(device)
            self.lstm_c_in[env_idx_t, step_idx_t] = lstm_c_in.to(device)
        self.decision_start[env_idx_t, step_idx_t] = starts_t
        self.decision_count[env_idx_t, step_idx_t] = counts_t

        source_starts_t = native_batch.decision_start.to(device=device, dtype=torch.long)
        total_decisions = int(counts_t.sum().item())
        if total_decisions:
            max_decisions = int(counts_t.max().item())
            decision_pos = torch.arange(max_decisions, device=device).expand(
                len(env_indices), max_decisions
            )
            valid_decision_mask = decision_pos < counts_t[:, None]
            decision_env = env_idx_t[:, None].expand(len(env_indices), max_decisions)[
                valid_decision_mask
            ]
            source_rows = (source_starts_t[:, None] + decision_pos)[valid_decision_mask]
            dest_rows = (starts_t[:, None] + decision_pos)[valid_decision_mask]

            native_decision_option_idx = native_batch.decision_option_idx.to(device)
            native_decision_target_idx = native_batch.decision_target_idx.to(device)
            native_decision_mask = native_batch.decision_mask.to(device=device, dtype=torch.bool)
            native_uses_none_head = native_batch.uses_none_head.to(device=device, dtype=torch.bool)

            self.decision_option_idx[decision_env, dest_rows] = native_decision_option_idx[
                source_rows
            ]
            self.decision_target_idx[decision_env, dest_rows] = native_decision_target_idx[
                source_rows
            ]
            self.decision_mask[decision_env, dest_rows] = native_decision_mask[source_rows]
            self.uses_none_head[decision_env, dest_rows] = native_uses_none_head[source_rows]
            self.selected_indices[decision_env, dest_rows] = selected_choice_cols_flat

        self.step_count[env_idx_t] = step_idx_t + 1
        self.decision_cursor[env_idx_t] = starts_t + counts_t


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
        recurrent_layers: int = 0,
        recurrent_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.decision_capacity = decision_capacity
        self.recurrent_layers = recurrent_layers
        self.recurrent_hidden_dim = recurrent_hidden_dim

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
            "lstm_h_in",
            (capacity, recurrent_layers, recurrent_hidden_dim),
            torch.float32,
        )
        _reg(
            "lstm_c_in",
            (capacity, recurrent_layers, recurrent_hidden_dim),
            torch.float32,
        )
        _reg("next_step_idx", (capacity,), torch.long)
        _reg("has_next", (capacity,), torch.float32)
        _reg("next_same_perspective_step_idx", (capacity,), torch.long)
        _reg("has_next_same_perspective", (capacity,), torch.float32)

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
    lstm_h_in: Tensor
    lstm_c_in: Tensor
    next_step_idx: Tensor
    has_next: Tensor
    next_same_perspective_step_idx: Tensor
    has_next_same_perspective: Tensor
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
        self.target_ref_is_player[step_indices] = native_batch.target_ref_is_player.to(
            device=device, dtype=torch.bool
        )
        self.target_ref_is_self[step_indices] = native_batch.target_ref_is_self.to(
            device=device, dtype=torch.bool
        )

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
                device=device, dtype=torch.bool
            )
            self.uses_none_head[start:end] = native_batch.uses_none_head[flat_cursor:flat_end].to(
                device=device, dtype=torch.bool
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
        return self.ingest_staged_episodes(staging, [env_idx])

    def ingest_staged_episodes(
        self,
        staging: NativeTrajectoryBuffer,
        env_indices: list[int],
    ) -> BufferWrite:
        if not env_indices:
            device = self.device
            return BufferWrite(
                step_indices=torch.zeros(0, dtype=torch.long, device=device),
                decision_starts=[],
                decision_counts=[],
            )

        device = self.device
        env_idx_t = torch.tensor(env_indices, dtype=torch.long, device=device)
        step_counts_t = staging.step_count[env_idx_t].to(dtype=torch.long)
        n = int(step_counts_t.sum().item())
        device = self.device
        if n == 0:
            return BufferWrite(
                step_indices=torch.zeros(0, dtype=torch.long, device=device),
                decision_starts=[],
                decision_counts=[],
            )

        step_rows = self._allocate_step_rows(n)
        step_indices = torch.tensor(step_rows, dtype=torch.long, device=device)

        max_steps = int(step_counts_t.max().item())
        step_pos = torch.arange(max_steps, device=device).expand(len(env_indices), max_steps)
        valid_step_mask = step_pos < step_counts_t[:, None]
        src_env = env_idx_t[:, None].expand(len(env_indices), max_steps)[valid_step_mask]
        src_step = step_pos[valid_step_mask]

        self.slot_card_rows[step_indices] = staging.slot_card_rows[src_env, src_step]
        self.slot_occupied[step_indices] = staging.slot_occupied[src_env, src_step]
        self.slot_tapped[step_indices] = staging.slot_tapped[src_env, src_step]
        self.game_info[step_indices] = staging.game_info[src_env, src_step]
        self.trace_kind_id[step_indices] = staging.trace_kind_id[src_env, src_step]
        self.pending_kind_id[step_indices] = staging.pending_kind_id[src_env, src_step]
        self.option_kind_ids[step_indices] = staging.option_kind_ids[src_env, src_step]
        self.option_scalars[step_indices] = staging.option_scalars[src_env, src_step]
        self.option_mask[step_indices] = staging.option_mask[src_env, src_step]
        self.option_ref_slot_idx[step_indices] = staging.option_ref_slot_idx[src_env, src_step]
        self.option_ref_card_row[step_indices] = staging.option_ref_card_row[src_env, src_step]
        self.target_mask[step_indices] = staging.target_mask[src_env, src_step]
        self.target_type_ids[step_indices] = staging.target_type_ids[src_env, src_step]
        self.target_scalars[step_indices] = staging.target_scalars[src_env, src_step]
        self.target_overflow[step_indices] = staging.target_overflow[src_env, src_step]
        self.target_ref_slot_idx[step_indices] = staging.target_ref_slot_idx[src_env, src_step]
        self.target_ref_is_player[step_indices] = staging.target_ref_is_player[src_env, src_step]
        self.target_ref_is_self[step_indices] = staging.target_ref_is_self[src_env, src_step]
        self.may_selected[step_indices] = staging.may_selected[src_env, src_step]
        self.lstm_h_in[step_indices] = staging.lstm_h_in[src_env, src_step]
        self.lstm_c_in[step_indices] = staging.lstm_c_in[src_env, src_step]

        step_counts_per_row = step_counts_t[:, None].expand(len(env_indices), max_steps)[
            valid_step_mask
        ]
        is_last = (src_step + 1) >= step_counts_per_row
        has_next_vals = (~is_last).to(torch.float32)
        next_rows = step_indices.clone()
        not_last_mask = ~is_last
        next_rows[not_last_mask] = step_indices[not_last_mask] + 1
        self.next_step_idx[step_indices] = next_rows
        self.has_next[step_indices] = has_next_vals

        # Compute next-same-perspective chain for SPR. For each row, find the
        # next row in the same env whose perspective_player_idx matches; if
        # none exists before the episode ends, self-reference with has_next=0.
        # Vectorized via a per-perspective suffix-min of valid positions.
        num_envs_committed = len(env_indices)
        persp_full = staging.perspective_player_idx[env_idx_t].to(device)[:, :max_steps]
        env_pos = torch.arange(num_envs_committed, device=device)
        src_env_pos = env_pos[:, None].expand(num_envs_committed, max_steps)[valid_step_mask]
        flat_2d = torch.full((num_envs_committed, max_steps), -1, dtype=torch.long, device=device)
        flat_2d[src_env_pos, src_step] = step_indices

        pos_row = torch.arange(max_steps, device=device).expand(num_envs_committed, max_steps)
        sentinel = torch.full_like(pos_row, max_steps)
        next_s_per_persp: list[Tensor] = []
        # 2-player zero-sum; compute for each unique perspective value actually present.
        unique_persp = torch.unique(persp_full[valid_step_mask]).tolist()
        for v in unique_persp:
            mask_v = (persp_full == v) & valid_step_mask
            pos_masked = torch.where(mask_v, pos_row, sentinel)
            # suffix min along step axis: min over s' >= s.
            rev_cummin, _ = pos_masked.flip(dims=[1]).cummin(dim=1)
            suffix_min = rev_cummin.flip(dims=[1])
            next_s = torch.full_like(suffix_min, max_steps)
            if max_steps > 1:
                next_s[:, :-1] = suffix_min[:, 1:]
            next_s_per_persp.append(next_s)

        if unique_persp:
            stacked = torch.stack(next_s_per_persp, dim=0)  # [V, P, S]
            persp_to_slot = torch.full(
                (int(persp_full.max().item()) + 1,),
                -1,
                dtype=torch.long,
                device=device,
            )
            for slot, v in enumerate(unique_persp):
                persp_to_slot[v] = slot
            persp_slot = persp_to_slot[persp_full]
            # Gather along V using persp_slot.
            gather_idx = persp_slot.unsqueeze(0).clamp_min(0)
            next_s_chosen = stacked.gather(0, gather_idx).squeeze(0)  # [P, S]
        else:
            next_s_chosen = torch.full(
                (num_envs_committed, max_steps), max_steps, dtype=torch.long, device=device
            )

        valid_next_2d = next_s_chosen < max_steps
        safe_s = next_s_chosen.clamp(max=max(max_steps - 1, 0))
        next_flat_2d = flat_2d.gather(1, safe_s)
        next_flat_2d = torch.where(valid_next_2d, next_flat_2d, flat_2d)
        has_next_same_2d = valid_next_2d.to(torch.float32)
        self.next_same_perspective_step_idx[step_indices] = next_flat_2d[valid_step_mask]
        self.has_next_same_perspective[step_indices] = has_next_same_2d[valid_step_mask]

        decision_counts_t = staging.decision_count[src_env, src_step].to(dtype=torch.long)
        source_starts_t = staging.decision_start[src_env, src_step].to(dtype=torch.long)
        total_decisions = int(decision_counts_t.sum().item())
        if total_decisions:
            decision_base = self._allocate_decision_range(total_decisions)
            decision_offsets = torch.cumsum(decision_counts_t, dim=0) - decision_counts_t
            decision_starts_t = decision_base + decision_offsets
            max_decisions = int(decision_counts_t.max().item())
            decision_pos = torch.arange(max_decisions, device=device).expand(n, max_decisions)
            valid_decision_mask = decision_pos < decision_counts_t[:, None]
            decision_env = src_env[:, None].expand(n, max_decisions)[valid_decision_mask]
            source_rows = (source_starts_t[:, None] + decision_pos)[valid_decision_mask]
            dest_rows = (decision_starts_t[:, None] + decision_pos)[valid_decision_mask]
            self.decision_option_idx[dest_rows] = staging.decision_option_idx[
                decision_env, source_rows
            ]
            self.decision_target_idx[dest_rows] = staging.decision_target_idx[
                decision_env, source_rows
            ]
            self.decision_mask[dest_rows] = staging.decision_mask[decision_env, source_rows]
            self.uses_none_head[dest_rows] = staging.uses_none_head[decision_env, source_rows]
            self.selected_indices[dest_rows] = staging.selected_indices[decision_env, source_rows]
        else:
            decision_starts_t = torch.zeros(n, dtype=torch.long, device=device)

        self.decision_start[step_indices] = decision_starts_t
        self.decision_count[step_indices] = decision_counts_t

        decision_starts = [int(v) for v in decision_starts_t.detach().cpu().tolist()]
        decision_counts = [int(v) for v in decision_counts_t.detach().cpu().tolist()]

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
