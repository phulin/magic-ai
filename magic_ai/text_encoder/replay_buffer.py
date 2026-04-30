"""Replay storage for text-encoded training rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch
from magic_ai.text_encoder.replay_triton import (
    append_batch_encoded_triton,
    clear_append_decisions_triton,
    write_append_decisions_triton,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


@dataclass(frozen=True)
class TextReplayBatch:
    encoded: PackedTextBatch
    trace_kind_id: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    decision_count: Tensor
    selected_indices: Tensor
    may_selected: Tensor
    old_log_prob: Tensor
    value: Tensor
    perspective_player_idx: Tensor
    lstm_h_in: Tensor | None
    lstm_c_in: Tensor | None


class TextReplayBuffer:
    """Fixed-width replay buffer for assembled text encoder rows."""

    def __init__(
        self,
        *,
        capacity: int,
        max_tokens: int,
        max_options: int,
        max_targets_per_option: int,
        max_decision_groups: int,
        max_cached_choices: int,
        recurrent_layers: int = 0,
        recurrent_hidden_dim: int = 0,
        max_card_refs: int = MAX_CARD_REFS,
        device: torch.device | str = "cpu",
        validate: bool = True,
        use_triton_append: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if max_options < 1:
            raise ValueError("max_options must be at least 1")
        if max_targets_per_option < 1:
            raise ValueError("max_targets_per_option must be at least 1")
        if max_decision_groups < 1:
            raise ValueError("max_decision_groups must be at least 1")
        if max_cached_choices < 1:
            raise ValueError("max_cached_choices must be at least 1")
        if (recurrent_layers == 0) != (recurrent_hidden_dim == 0):
            raise ValueError(
                "recurrent_layers and recurrent_hidden_dim must both be zero or nonzero"
            )

        self.capacity = int(capacity)
        self.max_tokens = int(max_tokens)
        self.max_options = int(max_options)
        self.max_targets_per_option = int(max_targets_per_option)
        self.max_decision_groups = int(max_decision_groups)
        self.max_cached_choices = int(max_cached_choices)
        self.recurrent_layers = int(recurrent_layers)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.max_card_refs = int(max_card_refs)
        self.device = torch.device(device)
        self.validate = bool(validate)
        self.use_triton_append = bool(use_triton_append)

        # Storage dtypes are sized to the value ranges they hold, then cast to
        # int64 at the few consumption sites that actually require Long
        # (nn.Embedding, torch.gather indices). The buffer is the dominant GPU
        # consumer at training scale, so the dtype cuts compound.
        self.token_ids = torch.zeros(
            self.capacity, self.max_tokens, dtype=torch.int32, device=self.device
        )
        self.packed_token_ids = torch.zeros(
            self.capacity * self.max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.row_token_start = torch.full(
            (self.capacity,), -1, dtype=torch.int64, device=self.device
        )
        self.row_token_length = torch.zeros(self.capacity, dtype=torch.int32, device=self.device)
        self._token_cursor = 0
        self.attention_mask = torch.zeros(
            self.capacity, self.max_tokens, dtype=torch.bool, device=self.device
        )
        self.card_ref_positions = torch.full(
            (self.capacity, self.max_card_refs), -1, dtype=torch.int32, device=self.device
        )
        self.option_positions = torch.full(
            (self.capacity, self.max_options), -1, dtype=torch.int32, device=self.device
        )
        self.option_mask = torch.zeros(
            self.capacity, self.max_options, dtype=torch.bool, device=self.device
        )
        self.target_positions = torch.full(
            (
                self.capacity,
                self.max_options,
                self.max_targets_per_option,
            ),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.target_mask = torch.zeros(
            self.capacity,
            self.max_options,
            self.max_targets_per_option,
            dtype=torch.bool,
            device=self.device,
        )
        self.seq_lengths = torch.zeros(self.capacity, dtype=torch.int32, device=self.device)
        self.trace_kind_id = torch.zeros(self.capacity, dtype=torch.int8, device=self.device)
        self.decision_option_idx = torch.full(
            (self.capacity, self.max_decision_groups, self.max_cached_choices),
            -1,
            dtype=torch.int16,
            device=self.device,
        )
        self.decision_target_idx = torch.full_like(self.decision_option_idx, -1)
        self.decision_mask = torch.zeros(
            self.capacity,
            self.max_decision_groups,
            self.max_cached_choices,
            dtype=torch.bool,
            device=self.device,
        )
        self.uses_none_head = torch.zeros(
            self.capacity, self.max_decision_groups, dtype=torch.bool, device=self.device
        )
        self.decision_count = torch.zeros(self.capacity, dtype=torch.int16, device=self.device)
        self.selected_indices = torch.full(
            (self.capacity, self.max_decision_groups), -1, dtype=torch.int16, device=self.device
        )
        self.may_selected = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.old_log_prob = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.value = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.perspective_player_idx = torch.zeros(
            self.capacity, dtype=torch.int8, device=self.device
        )
        self.lstm_h_in: Tensor | None = None
        self.lstm_c_in: Tensor | None = None
        if self.recurrent_layers > 0:
            recurrent_shape = (
                self.capacity,
                self.recurrent_layers,
                self.recurrent_hidden_dim,
            )
            self.lstm_h_in = torch.zeros(recurrent_shape, dtype=torch.float32, device=self.device)
            self.lstm_c_in = torch.zeros_like(self.lstm_h_in)

        self._occupied = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self._free_rows = list(range(self.capacity - 1, -1, -1))

    @property
    def size(self) -> int:
        return int(self._occupied.sum().item())

    def reset(self) -> None:
        self._occupied.zero_()
        self._free_rows = list(range(self.capacity - 1, -1, -1))
        self.row_token_start.fill_(-1)
        self.row_token_length.zero_()
        self._token_cursor = 0

    def release_replay_rows(self, replay_rows: list[int]) -> None:
        for row in replay_rows:
            self._validate_row(row)
            if bool(self._occupied[row].item()):
                self._occupied[row] = False
                self._free_rows.append(int(row))

    def append(
        self,
        *,
        encoded: TextEncodedBatch,
        batch_index: int,
        trace_kind_id: int,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
        may_selected: float,
        old_log_prob: float,
        value: float,
        perspective_player_idx: int,
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> int:
        if not self._free_rows:
            raise RuntimeError("TextReplayBuffer is full")
        row = self._free_rows.pop()
        self._write_encoded_row(row, encoded, batch_index)
        decision_count = int(decision_option_idx.shape[0])
        # Silently truncate decisions that exceed the buffer's per-row width.
        # Combat steps (declare attackers / blockers) on dense mid-game boards
        # routinely emit more than ``max_decision_groups`` groups; a hard
        # raise was previously crashing training. Keep the leading prefix —
        # decision group order is deterministic by the engine, so dropping
        # the tail is reproducible across replays of the same trace.
        truncated_count = min(decision_count, self.max_decision_groups)
        self.decision_option_idx[row].fill_(-1)
        self.decision_target_idx[row].fill_(-1)
        self.decision_mask[row].zero_()
        self.uses_none_head[row].zero_()
        self.selected_indices[row].fill_(-1)
        if truncated_count > 0:
            self._validate_decision_shapes(
                decision_option_idx[:truncated_count],
                decision_target_idx[:truncated_count],
                decision_mask[:truncated_count],
                uses_none_head[:truncated_count],
                selected_indices[:truncated_count],
            )
            self.decision_option_idx[row, :truncated_count].copy_(
                decision_option_idx[:truncated_count].to(device=self.device, dtype=torch.long)
            )
            self.decision_target_idx[row, :truncated_count].copy_(
                decision_target_idx[:truncated_count].to(device=self.device, dtype=torch.long)
            )
            self.decision_mask[row, :truncated_count].copy_(
                decision_mask[:truncated_count].to(device=self.device, dtype=torch.bool)
            )
            self.uses_none_head[row, :truncated_count].copy_(
                uses_none_head[:truncated_count].to(device=self.device, dtype=torch.bool)
            )
            self.selected_indices[row, :truncated_count].copy_(
                selected_indices[:truncated_count].to(device=self.device, dtype=torch.long)
            )
        decision_count = truncated_count

        self.trace_kind_id[row] = int(trace_kind_id)
        self.decision_count[row] = decision_count
        self.may_selected[row] = float(may_selected)
        self.old_log_prob[row] = float(old_log_prob)
        self.value[row] = float(value)
        self.perspective_player_idx[row] = int(perspective_player_idx)
        self._write_recurrent_state(row, lstm_h_in, lstm_c_in)
        self._occupied[row] = True
        return row

    def append_packed(
        self,
        *,
        encoded: PackedTextBatch,
        batch_index: int,
        trace_kind_id: int,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
        may_selected: float,
        old_log_prob: float,
        value: float,
        perspective_player_idx: int,
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> int:
        if not self._free_rows:
            raise RuntimeError("TextReplayBuffer is full")
        row = self._free_rows.pop()
        self._write_packed_encoded_row(row, encoded, batch_index)
        decision_count = int(decision_option_idx.shape[0])
        truncated_count = min(decision_count, self.max_decision_groups)
        self.decision_option_idx[row].fill_(-1)
        self.decision_target_idx[row].fill_(-1)
        self.decision_mask[row].zero_()
        self.uses_none_head[row].zero_()
        self.selected_indices[row].fill_(-1)
        if truncated_count > 0:
            self._validate_decision_shapes(
                decision_option_idx[:truncated_count],
                decision_target_idx[:truncated_count],
                decision_mask[:truncated_count],
                uses_none_head[:truncated_count],
                selected_indices[:truncated_count],
            )
            self.decision_option_idx[row, :truncated_count].copy_(
                decision_option_idx[:truncated_count].to(device=self.device, dtype=torch.long)
            )
            self.decision_target_idx[row, :truncated_count].copy_(
                decision_target_idx[:truncated_count].to(device=self.device, dtype=torch.long)
            )
            self.decision_mask[row, :truncated_count].copy_(
                decision_mask[:truncated_count].to(device=self.device, dtype=torch.bool)
            )
            self.uses_none_head[row, :truncated_count].copy_(
                uses_none_head[:truncated_count].to(device=self.device, dtype=torch.bool)
            )
            self.selected_indices[row, :truncated_count].copy_(
                selected_indices[:truncated_count].to(device=self.device, dtype=torch.long)
            )

        self.trace_kind_id[row] = int(trace_kind_id)
        self.decision_count[row] = truncated_count
        self.may_selected[row] = float(may_selected)
        self.old_log_prob[row] = float(old_log_prob)
        self.value[row] = float(value)
        self.perspective_player_idx[row] = int(perspective_player_idx)
        self._write_recurrent_state(row, lstm_h_in, lstm_c_in)
        self._occupied[row] = True
        return row

    def append_batch(
        self,
        *,
        encoded: PackedTextBatch,
        trace_kind_id: Tensor,
        decision_count: Tensor,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
        may_selected: Tensor,
        old_log_prob: Tensor,
        value: Tensor,
        perspective_player_idx: Tensor,
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> Tensor:
        batch_size = int(encoded.seq_lengths.shape[0])
        if batch_size == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if len(self._free_rows) < batch_size:
            raise RuntimeError("TextReplayBuffer is full")
        rows_list = [self._free_rows.pop() for _ in range(batch_size)]
        rows = torch.tensor(rows_list, dtype=torch.long, device=self.device)

        seq_lengths = encoded.seq_lengths.to(device=self.device)
        if self.validate:
            max_seq_length = int(seq_lengths.max().item()) if batch_size > 0 else 0
            if max_seq_length > self.max_tokens:
                raise ValueError("encoded packed row token width exceeds buffer max_tokens")
        total_tokens = int(encoded.token_ids.numel())
        token_start = self._token_cursor
        token_end = token_start + total_tokens
        if token_end > int(self.packed_token_ids.numel()):
            raise RuntimeError("TextReplayBuffer packed token arena is full")
        self._token_cursor = token_end

        wrote_encoded_with_triton = False
        if self.use_triton_append:
            wrote_encoded_with_triton = append_batch_encoded_triton(
                token_ids=encoded.token_ids,
                cu_seqlens=encoded.cu_seqlens,
                seq_lengths=seq_lengths,
                state_positions=encoded.state_positions,
                card_ref_positions=encoded.card_ref_positions,
                option_positions=encoded.option_positions,
                option_mask=encoded.option_mask,
                target_positions=encoded.target_positions,
                target_mask=encoded.target_mask,
                rows=rows,
                packed_token_ids=self.packed_token_ids,
                row_token_start=self.row_token_start,
                row_token_length=self.row_token_length,
                dst_card_ref_positions=self.card_ref_positions,
                dst_option_positions=self.option_positions,
                dst_option_mask=self.option_mask,
                dst_target_positions=self.target_positions,
                dst_target_mask=self.target_mask,
                dst_seq_lengths=self.seq_lengths,
                token_start=token_start,
            )
        option_width = min(int(encoded.option_positions.shape[1]), self.max_options)
        target_width = min(int(encoded.target_positions.shape[2]), self.max_targets_per_option)
        if not wrote_encoded_with_triton:
            self.row_token_start[rows] = -1
            self.row_token_length[rows] = 0
            self.card_ref_positions.index_fill_(0, rows, -1)
            self.option_positions.index_fill_(0, rows, -1)
            self.option_mask.index_fill_(0, rows, 0)
            self.target_positions.index_fill_(0, rows, -1)
            self.target_mask.index_fill_(0, rows, 0)
            token_ids = encoded.token_ids.to(device=self.device, dtype=torch.int32)
            if total_tokens > 0:
                self.packed_token_ids[token_start:token_end] = token_ids
            self.row_token_start[rows] = token_start + encoded.cu_seqlens[:-1].to(
                device=self.device, dtype=torch.int64
            )
            seq_lengths_i32 = seq_lengths.to(dtype=torch.int32)
            self.row_token_length[rows] = seq_lengths_i32

            base = encoded.state_positions.to(device=self.device, dtype=torch.int32)

            def rebase_positions(pos: Tensor, view_shape: tuple[int, ...]) -> Tensor:
                pos_dev = pos.to(device=self.device, dtype=torch.int32)
                valid = pos_dev >= 0
                shifted = pos_dev - base.view(view_shape)
                return torch.where(valid, shifted, pos_dev)

            self.card_ref_positions[rows] = rebase_positions(
                encoded.card_ref_positions,
                (batch_size, 1),
            )
            if option_width > 0:
                self.option_positions[rows, :option_width] = rebase_positions(
                    encoded.option_positions[:, :option_width],
                    (batch_size, 1),
                )
                self.option_mask[rows, :option_width] = encoded.option_mask[:, :option_width].to(
                    device=self.device, dtype=torch.bool
                )
            if option_width > 0 and target_width > 0:
                self.target_positions[rows, :option_width, :target_width] = rebase_positions(
                    encoded.target_positions[:, :option_width, :target_width],
                    (batch_size, 1, 1),
                )
                self.target_mask[rows, :option_width, :target_width] = encoded.target_mask[
                    :, :option_width, :target_width
                ].to(device=self.device, dtype=torch.bool)
            self.seq_lengths[rows] = seq_lengths_i32

        decision_count_dev = decision_count.to(device=self.device, dtype=torch.long)
        wrote_decisions_with_triton = False
        if self.use_triton_append:
            wrote_decisions_with_triton = write_append_decisions_triton(
                rows=rows,
                decision_count=decision_count_dev,
                decision_option_idx=decision_option_idx,
                decision_target_idx=decision_target_idx,
                decision_mask=decision_mask,
                uses_none_head=uses_none_head,
                selected_indices=selected_indices,
                trace_kind_id=trace_kind_id,
                may_selected=may_selected,
                old_log_prob=old_log_prob,
                value=value,
                perspective_player_idx=perspective_player_idx,
                dst_decision_option_idx=self.decision_option_idx,
                dst_decision_target_idx=self.decision_target_idx,
                dst_decision_mask=self.decision_mask,
                dst_uses_none_head=self.uses_none_head,
                dst_selected_indices=self.selected_indices,
                dst_decision_count=self.decision_count,
                dst_trace_kind_id=self.trace_kind_id,
                dst_may_selected=self.may_selected,
                dst_old_log_prob=self.old_log_prob,
                dst_value=self.value,
                dst_perspective_player_idx=self.perspective_player_idx,
            )
        if not wrote_decisions_with_triton:
            stored_count = decision_count_dev.clamp(max=self.max_decision_groups)
            cleared_decisions_with_triton = False
            if self.use_triton_append:
                cleared_decisions_with_triton = clear_append_decisions_triton(
                    rows=rows,
                    decision_option_idx=self.decision_option_idx,
                    decision_target_idx=self.decision_target_idx,
                    decision_mask=self.decision_mask,
                    uses_none_head=self.uses_none_head,
                    selected_indices=self.selected_indices,
                )
            if not cleared_decisions_with_triton:
                self.decision_option_idx.index_fill_(0, rows, -1)
                self.decision_target_idx.index_fill_(0, rows, -1)
                self.decision_mask.index_fill_(0, rows, 0)
                self.uses_none_head.index_fill_(0, rows, 0)
                self.selected_indices.index_fill_(0, rows, -1)
            g_total = int(decision_option_idx.shape[0])
            if g_total > 0:
                step_for_group = torch.repeat_interleave(
                    torch.arange(batch_size, device=self.device), decision_count_dev
                )
                group_starts = torch.cumsum(decision_count_dev, dim=0) - decision_count_dev
                group_in_step = (
                    torch.arange(g_total, device=self.device) - group_starts[step_for_group]
                )
                keep = group_in_step < self.max_decision_groups
                kept_steps = step_for_group[keep]
                kept_groups = group_in_step[keep]
                flat_keep = keep.nonzero(as_tuple=False).squeeze(-1)
                replay_rows = rows[kept_steps]
                self.decision_option_idx[replay_rows, kept_groups] = decision_option_idx[
                    flat_keep
                ].to(device=self.device, dtype=torch.int16)
                self.decision_target_idx[replay_rows, kept_groups] = decision_target_idx[
                    flat_keep
                ].to(device=self.device, dtype=torch.int16)
                self.decision_mask[replay_rows, kept_groups] = decision_mask[flat_keep].to(
                    device=self.device, dtype=torch.bool
                )
                self.uses_none_head[replay_rows, kept_groups] = uses_none_head[flat_keep].to(
                    device=self.device, dtype=torch.bool
                )
                self.selected_indices[replay_rows, kept_groups] = selected_indices[flat_keep].to(
                    device=self.device, dtype=torch.int16
                )
            self.decision_count[rows] = stored_count.to(dtype=torch.int16)
            self.trace_kind_id[rows] = trace_kind_id.to(device=self.device, dtype=torch.int8)
            self.may_selected[rows] = may_selected.to(device=self.device, dtype=torch.float32)
            self.old_log_prob[rows] = old_log_prob.to(device=self.device, dtype=torch.float32)
            self.value[rows] = value.to(device=self.device, dtype=torch.float32)
            self.perspective_player_idx[rows] = perspective_player_idx.to(
                device=self.device, dtype=torch.int8
            )
        if self.lstm_h_in is not None and self.lstm_c_in is not None:
            if lstm_h_in is None or lstm_c_in is None:
                raise ValueError("h_in and c_in are required for recurrent text replay")
            self.lstm_h_in[rows] = lstm_h_in.permute(1, 0, 2).to(
                device=self.device, dtype=torch.float32
            )
            self.lstm_c_in[rows] = lstm_c_in.permute(1, 0, 2).to(
                device=self.device, dtype=torch.float32
            )
        elif lstm_h_in is not None or lstm_c_in is not None:
            raise ValueError("buffer was constructed without recurrent state storage")
        self._occupied[rows] = True
        return rows

    def _write_row_packed_tokens(self, row: int, token_ids: Tensor) -> None:
        token_count = int(token_ids.numel())
        token_start = self._token_cursor
        token_end = token_start + token_count
        if token_end > int(self.packed_token_ids.numel()):
            raise RuntimeError("TextReplayBuffer packed token arena is full")
        self._token_cursor = token_end
        self.row_token_start[row] = token_start
        self.row_token_length[row] = token_count
        if token_count > 0:
            self.packed_token_ids[token_start:token_end] = token_ids

    def gather(self, replay_rows: list[int] | Tensor) -> TextReplayBatch:
        if isinstance(replay_rows, Tensor):
            if int(replay_rows.numel()) == 0:
                raise ValueError("replay_rows must not be empty")
            idx = replay_rows.to(device=self.device, dtype=torch.long)
        else:
            if not replay_rows:
                raise ValueError("replay_rows must not be empty")
            idx = torch.tensor(replay_rows, dtype=torch.long, device=self.device)
        if self.validate:
            in_range = (idx >= 0) & (idx < self.capacity)
            if not bool(in_range.all().item()):
                bad = int(idx[~in_range][0].item())
                raise ValueError(f"replay row {bad} out of range")
            occupied = self._occupied[idx]
            if not bool(occupied.all().item()):
                bad = int(idx[~occupied][0].item())
                raise ValueError(f"replay row {bad} is not occupied")
        seq_lengths = self.seq_lengths[idx].to(torch.long)
        batch_size = int(seq_lengths.numel())
        token_starts = self.row_token_start[idx]
        if self.validate and bool((token_starts < 0).any().item()):
            bad = int(idx[token_starts < 0][0].item())
            raise ValueError(f"replay row {bad} has no packed token span")
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        cu_seqlens[1:] = seq_lengths.cumsum(0)
        state_positions = cu_seqlens[:-1]
        seq_id = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=self.device),
            seq_lengths,
        )
        total_tokens = int(cu_seqlens[-1].item())
        pos_in_seq = torch.arange(
            total_tokens, dtype=torch.long, device=self.device
        ) - torch.repeat_interleave(
            state_positions,
            seq_lengths,
        )
        token_offsets = token_starts[seq_id] + pos_in_seq
        token_ids = self.packed_token_ids[token_offsets]

        def pack_positions(pos: Tensor, view_shape: tuple[int, ...]) -> Tensor:
            valid = pos >= 0
            shifted = pos.to(torch.long) + state_positions.view(view_shape)
            return torch.where(valid, shifted, pos.to(torch.long))

        card_ref_positions = pack_positions(self.card_ref_positions[idx], (batch_size, 1))
        option_positions = pack_positions(self.option_positions[idx], (batch_size, 1))
        target_positions = pack_positions(self.target_positions[idx], (batch_size, 1, 1))
        encoded = PackedTextBatch(
            token_ids=token_ids,
            seq_id=seq_id,
            pos_in_seq=pos_in_seq,
            cu_seqlens=cu_seqlens,
            seq_lengths=seq_lengths,
            state_positions=state_positions,
            card_ref_positions=card_ref_positions,
            option_positions=option_positions,
            option_mask=self.option_mask[idx],
            target_positions=target_positions,
            target_mask=self.target_mask[idx],
        )
        h_in = self.lstm_h_in[idx] if self.lstm_h_in is not None else None
        c_in = self.lstm_c_in[idx] if self.lstm_c_in is not None else None
        return TextReplayBatch(
            encoded=encoded,
            trace_kind_id=self.trace_kind_id[idx],
            decision_option_idx=self.decision_option_idx[idx],
            decision_target_idx=self.decision_target_idx[idx],
            decision_mask=self.decision_mask[idx],
            uses_none_head=self.uses_none_head[idx],
            decision_count=self.decision_count[idx],
            selected_indices=self.selected_indices[idx],
            may_selected=self.may_selected[idx],
            old_log_prob=self.old_log_prob[idx],
            value=self.value[idx],
            perspective_player_idx=self.perspective_player_idx[idx],
            lstm_h_in=h_in,
            lstm_c_in=c_in,
        )

    def _write_encoded_row(
        self,
        row: int,
        encoded: TextEncodedBatch,
        batch_index: int,
    ) -> None:
        self._validate_batch_index(encoded, batch_index)
        # Encoded tensors carry their per-batch dense shape; ``token_ids``
        # has shape (B, T_b) where T_b <= self.max_tokens. Copying the
        # whole row uses the static tensor shape (a Python int, no GPU
        # sync) instead of materializing the per-row ``seq_lengths`` value
        # via ``.item()``. The buffer is zeroed past T_b, which matches
        # the prior behavior (encoded tensors are pad-id padded past
        # seq_lengths anyway, so the trailing slots either way are
        # ignored at training time via ``attention_mask``).
        token_width = int(encoded.token_ids.shape[1])
        attention_width = int(encoded.attention_mask.shape[1])
        seq_length = int(encoded.seq_lengths[batch_index].item())
        self.token_ids[row].zero_()
        self.attention_mask[row].zero_()
        self.card_ref_positions[row].fill_(-1)
        self.option_positions[row].fill_(-1)
        self.option_mask[row].zero_()
        self.target_positions[row].fill_(-1)
        self.target_mask[row].zero_()
        self.token_ids[row, :token_width].copy_(
            encoded.token_ids[batch_index].to(device=self.device, dtype=torch.int32)
        )
        self._write_row_packed_tokens(
            row,
            encoded.token_ids[batch_index, :seq_length].to(device=self.device, dtype=torch.int32),
        )
        self.attention_mask[row, :attention_width].copy_(
            encoded.attention_mask[batch_index].to(device=self.device, dtype=torch.bool)
        )
        self.card_ref_positions[row].copy_(
            encoded.card_ref_positions[batch_index].to(device=self.device, dtype=torch.int32)
        )
        option_width = min(encoded.option_positions.shape[1], self.max_options)
        target_width = min(encoded.target_positions.shape[2], self.max_targets_per_option)
        self.option_positions[row, :option_width].copy_(
            encoded.option_positions[batch_index, :option_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        self.option_mask[row, :option_width].copy_(
            encoded.option_mask[batch_index, :option_width].to(device=self.device)
        )
        self.target_positions[row, :option_width, :target_width].copy_(
            encoded.target_positions[batch_index, :option_width, :target_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        self.target_mask[row, :option_width, :target_width].copy_(
            encoded.target_mask[batch_index, :option_width, :target_width].to(device=self.device)
        )
        # Device-side copy: no .item() sync, just a 0-d tensor write.
        self.seq_lengths[row] = encoded.seq_lengths[batch_index]

    def _write_packed_encoded_row(
        self,
        row: int,
        encoded: PackedTextBatch,
        batch_index: int,
    ) -> None:
        self._validate_packed_batch_index(encoded, batch_index)
        start = int(encoded.cu_seqlens[batch_index].item())
        end = int(encoded.cu_seqlens[batch_index + 1].item())
        token_width = end - start
        if token_width > self.max_tokens:
            raise ValueError("encoded packed row token width exceeds buffer max_tokens")

        self.token_ids[row].zero_()
        self.attention_mask[row].zero_()
        self.card_ref_positions[row].fill_(-1)
        self.option_positions[row].fill_(-1)
        self.option_mask[row].zero_()
        self.target_positions[row].fill_(-1)
        self.target_mask[row].zero_()

        self.token_ids[row, :token_width].copy_(
            encoded.token_ids[start:end].to(device=self.device, dtype=torch.int32)
        )
        self._write_row_packed_tokens(
            row,
            encoded.token_ids[start:end].to(device=self.device, dtype=torch.int32),
        )
        self.attention_mask[row, :token_width] = True

        base = int(encoded.state_positions[batch_index].item())

        def rebase(pos: Tensor) -> Tensor:
            valid = pos >= 0
            shifted = pos - base
            return torch.where(valid, shifted, pos)

        self.card_ref_positions[row].copy_(
            rebase(encoded.card_ref_positions[batch_index]).to(
                device=self.device, dtype=torch.int32
            )
        )
        option_width = min(encoded.option_positions.shape[1], self.max_options)
        target_width = min(encoded.target_positions.shape[2], self.max_targets_per_option)
        self.option_positions[row, :option_width].copy_(
            rebase(encoded.option_positions[batch_index, :option_width]).to(
                device=self.device, dtype=torch.int32
            )
        )
        self.option_mask[row, :option_width].copy_(
            encoded.option_mask[batch_index, :option_width].to(device=self.device)
        )
        self.target_positions[row, :option_width, :target_width].copy_(
            rebase(encoded.target_positions[batch_index, :option_width, :target_width]).to(
                device=self.device, dtype=torch.int32
            )
        )
        self.target_mask[row, :option_width, :target_width].copy_(
            encoded.target_mask[batch_index, :option_width, :target_width].to(device=self.device)
        )
        self.seq_lengths[row] = encoded.seq_lengths[batch_index].to(
            device=self.device, dtype=torch.int32
        )

    def _write_recurrent_state(
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
            raise ValueError("h_in and c_in are required for recurrent text replay")
        expected = (self.recurrent_layers, self.recurrent_hidden_dim)
        if tuple(h_in.shape) != expected or tuple(c_in.shape) != expected:
            raise ValueError(
                f"recurrent state must have shape {expected}, got {tuple(h_in.shape)} "
                f"and {tuple(c_in.shape)}"
            )
        self.lstm_h_in[row].copy_(h_in.to(device=self.device, dtype=torch.float32))
        self.lstm_c_in[row].copy_(c_in.to(device=self.device, dtype=torch.float32))

    def _validate_batch_index(self, encoded: TextEncodedBatch, batch_index: int) -> None:
        batch_size = int(encoded.token_ids.shape[0])
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError("batch_index out of range")
        if encoded.token_ids.shape[1] > self.max_tokens:
            raise ValueError("encoded batch token width exceeds buffer max_tokens")
        if encoded.card_ref_positions.shape[1] != self.max_card_refs:
            raise ValueError("encoded card_ref_positions width does not match buffer")
        if encoded.option_positions.shape[1] > self.max_options:
            raise ValueError("encoded option width exceeds buffer max_options")
        if encoded.target_positions.shape[2] > self.max_targets_per_option:
            raise ValueError("encoded target width exceeds buffer max_targets_per_option")

    def _validate_packed_batch_index(self, encoded: PackedTextBatch, batch_index: int) -> None:
        batch_size = int(encoded.seq_lengths.shape[0])
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError("batch_index out of range")
        if encoded.card_ref_positions.shape[1] != self.max_card_refs:
            raise ValueError("encoded card_ref_positions width does not match buffer")
        if encoded.option_positions.shape[1] > self.max_options:
            raise ValueError("encoded option width exceeds buffer max_options")
        if encoded.target_positions.shape[2] > self.max_targets_per_option:
            raise ValueError("encoded target width exceeds buffer max_targets_per_option")

    def _validate_decision_shapes(
        self,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
    ) -> None:
        expected_groups = int(decision_option_idx.shape[0])
        expected = (expected_groups, self.max_cached_choices)
        if tuple(decision_option_idx.shape) != expected:
            raise ValueError(f"decision_option_idx must have shape {expected}")
        if tuple(decision_target_idx.shape) != expected:
            raise ValueError(f"decision_target_idx must have shape {expected}")
        if tuple(decision_mask.shape) != expected:
            raise ValueError(f"decision_mask must have shape {expected}")
        if tuple(uses_none_head.shape) != (expected_groups,):
            raise ValueError(f"uses_none_head must have shape {(expected_groups,)}")
        if tuple(selected_indices.shape) != (expected_groups,):
            raise ValueError(f"selected_indices must have shape {(expected_groups,)}")

    def _validate_row(self, row: int) -> None:
        if row < 0 or row >= self.capacity:
            raise IndexError("replay row out of range")

    def _validate_occupied_row(self, row: int) -> None:
        self._validate_row(row)
        if not bool(self._occupied[row].item()):
            raise ValueError(f"replay row {row} is not occupied")


__all__ = ["TextReplayBatch", "TextReplayBuffer"]
