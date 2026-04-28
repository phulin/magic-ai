"""Replay storage for text-encoded training rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


@dataclass(frozen=True)
class TextReplayBatch:
    encoded: TextEncodedBatch
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

        self.token_ids = torch.zeros(
            self.capacity, self.max_tokens, dtype=torch.long, device=self.device
        )
        self.attention_mask = torch.zeros_like(self.token_ids)
        self.card_ref_positions = torch.full(
            (self.capacity, self.max_card_refs), -1, dtype=torch.long, device=self.device
        )
        self.option_positions = torch.full(
            (self.capacity, self.max_options), -1, dtype=torch.long, device=self.device
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
            dtype=torch.long,
            device=self.device,
        )
        self.target_mask = torch.zeros(
            self.capacity,
            self.max_options,
            self.max_targets_per_option,
            dtype=torch.bool,
            device=self.device,
        )
        self.seq_lengths = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.trace_kind_id = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.decision_option_idx = torch.full(
            (self.capacity, self.max_decision_groups, self.max_cached_choices),
            -1,
            dtype=torch.long,
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
        self.decision_count = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.selected_indices = torch.full(
            (self.capacity, self.max_decision_groups), -1, dtype=torch.long, device=self.device
        )
        self.may_selected = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.old_log_prob = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.value = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.perspective_player_idx = torch.zeros(
            self.capacity, dtype=torch.long, device=self.device
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
        if decision_count > self.max_decision_groups:
            raise ValueError("decision group count exceeds buffer width")
        self.decision_option_idx[row].fill_(-1)
        self.decision_target_idx[row].fill_(-1)
        self.decision_mask[row].zero_()
        self.uses_none_head[row].zero_()
        self.selected_indices[row].fill_(-1)
        if decision_count > 0:
            self._validate_decision_shapes(
                decision_option_idx,
                decision_target_idx,
                decision_mask,
                uses_none_head,
                selected_indices,
            )
            self.decision_option_idx[row, :decision_count].copy_(
                decision_option_idx.to(device=self.device, dtype=torch.long)
            )
            self.decision_target_idx[row, :decision_count].copy_(
                decision_target_idx.to(device=self.device, dtype=torch.long)
            )
            self.decision_mask[row, :decision_count].copy_(
                decision_mask.to(device=self.device, dtype=torch.bool)
            )
            self.uses_none_head[row, :decision_count].copy_(
                uses_none_head.to(device=self.device, dtype=torch.bool)
            )
            self.selected_indices[row, :decision_count].copy_(
                selected_indices.to(device=self.device, dtype=torch.long)
            )

        self.trace_kind_id[row] = int(trace_kind_id)
        self.decision_count[row] = decision_count
        self.may_selected[row] = float(may_selected)
        self.old_log_prob[row] = float(old_log_prob)
        self.value[row] = float(value)
        self.perspective_player_idx[row] = int(perspective_player_idx)
        self._write_recurrent_state(row, lstm_h_in, lstm_c_in)
        self._occupied[row] = True
        return row

    def gather(self, replay_rows: list[int]) -> TextReplayBatch:
        if not replay_rows:
            raise ValueError("replay_rows must not be empty")
        for row in replay_rows:
            self._validate_occupied_row(row)
        idx = torch.tensor(replay_rows, dtype=torch.long, device=self.device)
        encoded = TextEncodedBatch(
            token_ids=self.token_ids[idx],
            attention_mask=self.attention_mask[idx],
            card_ref_positions=self.card_ref_positions[idx],
            option_positions=self.option_positions[idx],
            option_mask=self.option_mask[idx],
            target_positions=self.target_positions[idx],
            target_mask=self.target_mask[idx],
            seq_lengths=self.seq_lengths[idx],
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
        seq_len = int(encoded.seq_lengths[batch_index].item())
        if seq_len > self.max_tokens:
            raise ValueError("encoded row length exceeds buffer max_tokens")
        self.token_ids[row].zero_()
        self.attention_mask[row].zero_()
        self.card_ref_positions[row].fill_(-1)
        self.option_positions[row].fill_(-1)
        self.option_mask[row].zero_()
        self.target_positions[row].fill_(-1)
        self.target_mask[row].zero_()
        self.token_ids[row, :seq_len].copy_(
            encoded.token_ids[batch_index, :seq_len].to(device=self.device, dtype=torch.long)
        )
        self.attention_mask[row, :seq_len].copy_(
            encoded.attention_mask[batch_index, :seq_len].to(device=self.device, dtype=torch.long)
        )
        self.card_ref_positions[row].copy_(
            encoded.card_ref_positions[batch_index].to(device=self.device, dtype=torch.long)
        )
        option_width = min(encoded.option_positions.shape[1], self.max_options)
        target_width = min(encoded.target_positions.shape[2], self.max_targets_per_option)
        self.option_positions[row, :option_width].copy_(
            encoded.option_positions[batch_index, :option_width].to(
                device=self.device, dtype=torch.long
            )
        )
        self.option_mask[row, :option_width].copy_(
            encoded.option_mask[batch_index, :option_width].to(device=self.device)
        )
        self.target_positions[row, :option_width, :target_width].copy_(
            encoded.target_positions[batch_index, :option_width, :target_width].to(
                device=self.device, dtype=torch.long
            )
        )
        self.target_mask[row, :option_width, :target_width].copy_(
            encoded.target_mask[batch_index, :option_width, :target_width].to(device=self.device)
        )
        self.seq_lengths[row] = seq_len

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
