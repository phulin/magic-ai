"""Replay storage for text-encoded training rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from magic_ai.replay_buffer import ReplayCore
from magic_ai.text_encoder.batch import (
    PackedTextBatch,
    TextEncodedBatch,
    add_packed_offsets,
    packed_sequence_layout,
    subtract_packed_offsets,
)
from magic_ai.text_encoder.replay_triton import (
    append_batch_encoded_triton,
    append_staged_encoded_triton,
    gather_decisions_triton,
    gather_encoded_triton,
    gather_sequence_layout_triton,
    gather_text_fields_triton,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


@dataclass(frozen=True)
class TextReplayBatch:
    encoded: PackedTextBatch
    trace_kind_id: Tensor
    decision_start: Tensor
    decision_count: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    selected_indices: Tensor
    behavior_action_log_prob: Tensor
    step_for_decision_group: Tensor
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
        lstm_proj_hidden: int = 0,
        max_card_refs: int = MAX_CARD_REFS,
        device: torch.device | str = "cpu",
        validate: bool = True,
        use_triton_append: bool = True,
        use_triton_gather: bool = True,
        materialize_gather_seq_id: bool | None = None,
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
        if lstm_proj_hidden < 0:
            raise ValueError("lstm_proj_hidden must be >= 0")

        self.capacity = int(capacity)
        self.max_tokens = int(max_tokens)
        self.max_options = int(max_options)
        self.max_targets_per_option = int(max_targets_per_option)
        self.max_decision_groups = int(max_decision_groups)
        self.max_cached_choices = int(max_cached_choices)
        self.recurrent_layers = int(recurrent_layers)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.lstm_proj_hidden = int(lstm_proj_hidden)
        self.max_card_refs = int(max_card_refs)
        self.device = torch.device(device)
        self.validate = bool(validate)
        self.use_triton_append = bool(use_triton_append)
        self.use_triton_gather = bool(use_triton_gather)
        self.materialize_gather_seq_id = (
            self.device.type != "cuda"
            if materialize_gather_seq_id is None
            else bool(materialize_gather_seq_id)
        )

        # Storage dtypes are sized to the value ranges they hold, then cast to
        # int64 at the few consumption sites that actually require Long
        # (nn.Embedding, torch.gather indices). The buffer is the dominant GPU
        # consumer at training scale, so the dtype cuts compound.
        self.packed_token_ids = torch.zeros(
            self.capacity * self.max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.row_token_start = torch.full(
            (self.capacity,), -1, dtype=torch.int32, device=self.device
        )
        self.row_token_length = torch.zeros(self.capacity, dtype=torch.int32, device=self.device)
        self.row_token_length_host = [0] * self.capacity
        self._token_cursor = 0
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
        self.projected_state: Tensor | None = None
        if self.lstm_proj_hidden > 0:
            self.projected_state = torch.zeros(
                self.capacity, self.lstm_proj_hidden, dtype=torch.bfloat16, device=self.device
            )
        self.core = ReplayCore(
            capacity=self.capacity,
            decision_capacity=self.capacity * self.max_decision_groups,
            max_decision_groups=self.max_decision_groups,
            max_cached_choices=self.max_cached_choices,
            device=self.device,
            recurrent_layers=self.recurrent_layers,
            recurrent_hidden_dim=self.recurrent_hidden_dim,
            index_dtype=torch.int16,
            trace_dtype=torch.int8,
            perspective_dtype=torch.int8,
        )
        self.trace_kind_id = self.core.trace_kind_id
        self.may_selected = self.core.may_selected
        self.old_log_prob = self.core.old_log_prob
        self.value = self.core.value
        self.ppo_return = self.core.ppo_return
        self.ppo_advantage = self.core.ppo_advantage
        self.perspective_player_idx = self.core.perspective_player_idx
        self.lstm_h_in = self.core.lstm_h_in
        self.lstm_c_in = self.core.lstm_c_in
        self.decision_start = self.core.decision_start
        self.decision_count = self.core.decision_count
        self.decision_option_idx = self.core.decision_option_idx
        self.decision_target_idx = self.core.decision_target_idx
        self.decision_mask = self.core.decision_mask
        self.uses_none_head = self.core.uses_none_head
        self.selected_indices = self.core.selected_indices
        self.behavior_action_log_prob = self.core.behavior_action_log_prob

    @property
    def size(self) -> int:
        return self.core.size

    def reset(self) -> None:
        self.core.reset()
        self.row_token_start.fill_(-1)
        self.row_token_length.zero_()
        self.row_token_length_host[:] = [0] * self.capacity
        self._token_cursor = 0

    def write_ppo_targets(
        self,
        replay_rows: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> None:
        self.core.write_ppo_targets(replay_rows, old_log_probs, returns, advantages)

    def gather_ppo_targets(self, replay_rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.core.gather_ppo_targets(replay_rows)

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
        behavior_action_log_prob: Tensor | None = None,
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> int:
        self._validate_batch_index(encoded, batch_index)
        try:
            row = self.core.append_row(
                trace_kind_id=trace_kind_id,
                decision_option_idx=decision_option_idx,
                decision_target_idx=decision_target_idx,
                decision_mask=decision_mask,
                uses_none_head=uses_none_head,
                selected_indices=selected_indices,
                behavior_action_log_prob=behavior_action_log_prob,
                may_selected=may_selected,
                old_log_prob=old_log_prob,
                value=value,
                perspective_player_idx=perspective_player_idx,
                lstm_h_in=lstm_h_in,
                lstm_c_in=lstm_c_in,
            )
        except RuntimeError as exc:
            raise RuntimeError("TextReplayBuffer is full") from exc
        self._write_encoded_row(row, encoded, batch_index)
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
        behavior_action_log_prob: Tensor | None = None,
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> int:
        self._validate_packed_batch_index(encoded, batch_index)
        try:
            row = self.core.append_row(
                trace_kind_id=trace_kind_id,
                decision_option_idx=decision_option_idx,
                decision_target_idx=decision_target_idx,
                decision_mask=decision_mask,
                uses_none_head=uses_none_head,
                selected_indices=selected_indices,
                behavior_action_log_prob=behavior_action_log_prob,
                may_selected=may_selected,
                old_log_prob=old_log_prob,
                value=value,
                perspective_player_idx=perspective_player_idx,
                lstm_h_in=lstm_h_in,
                lstm_c_in=lstm_c_in,
            )
        except RuntimeError as exc:
            raise RuntimeError("TextReplayBuffer is full") from exc
        self._write_packed_encoded_row(row, encoded, batch_index)
        return row

    def _packed_seq_lengths_host(self, encoded: PackedTextBatch) -> list[int]:
        batch_size = int(encoded.seq_lengths.shape[0])
        if encoded.seq_lengths_host is not None:
            lengths = list(encoded.seq_lengths_host)
            if len(lengths) != batch_size:
                raise ValueError("encoded.seq_lengths_host length must match batch size")
            return [int(x) for x in lengths]
        if encoded.seq_lengths.device.type == "cpu":
            return [int(x) for x in encoded.seq_lengths.to(dtype=torch.long).tolist()]
        return [int(x) for x in encoded.seq_lengths.detach().cpu().to(dtype=torch.long).tolist()]

    def append_batch(
        self,
        *,
        encoded: PackedTextBatch,
        trace_kind_id: Tensor,
        decision_count: Tensor,
        decision_count_host: tuple[int, ...] | None = None,
        total_decision_groups: int | None = None,
        total_stored_decision_groups: int | None = None,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
        may_selected: Tensor,
        old_log_prob: Tensor,
        value: Tensor,
        perspective_player_idx: Tensor,
        behavior_action_log_prob: Tensor | None = None,
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> Tensor:
        batch_size = int(encoded.seq_lengths.shape[0])
        if batch_size == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if self.capacity - self.size < batch_size:
            raise RuntimeError("TextReplayBuffer is full")
        seq_lengths = encoded.seq_lengths.to(device=self.device)
        if self.validate:
            max_seq_length = int(seq_lengths.max().item()) if batch_size > 0 else 0
            if max_seq_length > self.max_tokens:
                raise ValueError("encoded packed row token width exceeds buffer max_tokens")
        total_tokens = int(encoded.token_ids.numel())
        seq_lengths_host = self._packed_seq_lengths_host(encoded)
        token_start = self._token_cursor
        token_end = token_start + total_tokens
        if token_end > int(self.packed_token_ids.numel()):
            raise RuntimeError("TextReplayBuffer packed token arena is full")
        self._token_cursor = token_end
        if self.lstm_h_in is not None and self.lstm_c_in is not None:
            if lstm_h_in is None or lstm_c_in is None:
                raise ValueError("h_in and c_in are required for recurrent text replay")
            h_store = lstm_h_in.permute(1, 0, 2)
            c_store = lstm_c_in.permute(1, 0, 2)
        elif lstm_h_in is not None or lstm_c_in is not None:
            raise ValueError("buffer was constructed without recurrent state storage")
        else:
            h_store = None
            c_store = None

        try:
            rows = self.core.append_batch(
                trace_kind_id=trace_kind_id,
                decision_count=decision_count,
                decision_count_host=decision_count_host,
                total_decision_groups=total_decision_groups,
                total_stored_decision_groups=total_stored_decision_groups,
                decision_option_idx=decision_option_idx,
                decision_target_idx=decision_target_idx,
                decision_mask=decision_mask,
                uses_none_head=uses_none_head,
                selected_indices=selected_indices,
                behavior_action_log_prob=behavior_action_log_prob,
                may_selected=may_selected,
                old_log_prob=old_log_prob,
                value=value,
                perspective_player_idx=perspective_player_idx,
                lstm_h_in=h_store,
                lstm_c_in=c_store,
            )
        except RuntimeError as exc:
            raise RuntimeError("TextReplayBuffer is full") from exc

        row_start = self.size - batch_size
        self.row_token_length_host[row_start : row_start + batch_size] = seq_lengths_host

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
                device=self.device, dtype=torch.int32
            )
            seq_lengths_i32 = seq_lengths.to(dtype=torch.int32)
            self.row_token_length[rows] = seq_lengths_i32

            state_positions = encoded.state_positions.to(device=self.device, dtype=torch.int32)

            self.card_ref_positions[rows] = subtract_packed_offsets(
                encoded.card_ref_positions.to(device=self.device),
                state_positions,
            )
            if option_width > 0:
                self.option_positions[rows, :option_width] = subtract_packed_offsets(
                    encoded.option_positions[:, :option_width].to(device=self.device),
                    state_positions,
                )
                self.option_mask[rows, :option_width] = encoded.option_mask[:, :option_width].to(
                    device=self.device, dtype=torch.bool
                )
            if option_width > 0 and target_width > 0:
                self.target_positions[rows, :option_width, :target_width] = subtract_packed_offsets(
                    encoded.target_positions[:, :option_width, :target_width].to(
                        device=self.device
                    ),
                    state_positions,
                )
                self.target_mask[rows, :option_width, :target_width] = encoded.target_mask[
                    :, :option_width, :target_width
                ].to(device=self.device, dtype=torch.bool)
            self.seq_lengths[rows] = seq_lengths_i32

        return rows

    def append_staged_batch(
        self,
        *,
        flat_env: Tensor,
        flat_step: Tensor,
        token_ids: Tensor,
        seq_lengths: Tensor,
        seq_lengths_host: tuple[int, ...],
        card_ref_positions: Tensor,
        option_positions: Tensor,
        option_mask: Tensor,
        target_positions: Tensor,
        target_mask: Tensor,
        trace_kind_id: Tensor,
        decision_count: Tensor,
        decision_count_host: tuple[int, ...] | None = None,
        total_decision_groups: int | None = None,
        total_stored_decision_groups: int | None = None,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
        may_selected: Tensor,
        old_log_prob: Tensor,
        value: Tensor,
        perspective_player_idx: Tensor,
        behavior_action_log_prob: Tensor | None = None,
        lstm_h_in: Tensor | None = None,
        lstm_c_in: Tensor | None = None,
    ) -> Tensor:
        """Append replay rows directly from padded native staging tensors."""

        batch_size = int(seq_lengths.shape[0])
        if batch_size == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if len(seq_lengths_host) != batch_size:
            raise ValueError("seq_lengths_host length must match batch size")
        if self.capacity - self.size < batch_size:
            raise RuntimeError("TextReplayBuffer is full")
        seq_lengths = seq_lengths.to(device=self.device)
        if self.validate:
            max_seq_length = max(seq_lengths_host, default=0)
            if max_seq_length > self.max_tokens:
                raise ValueError("encoded staged row token width exceeds buffer max_tokens")
        total_tokens = sum(int(x) for x in seq_lengths_host)
        token_start = self._token_cursor
        token_end = token_start + total_tokens
        if token_end > int(self.packed_token_ids.numel()):
            raise RuntimeError("TextReplayBuffer packed token arena is full")
        self._token_cursor = token_end
        if self.lstm_h_in is not None and self.lstm_c_in is not None:
            if lstm_h_in is None or lstm_c_in is None:
                raise ValueError("h_in and c_in are required for recurrent text replay")
            h_store = lstm_h_in.permute(1, 0, 2)
            c_store = lstm_c_in.permute(1, 0, 2)
        elif lstm_h_in is not None or lstm_c_in is not None:
            raise ValueError("buffer was constructed without recurrent state storage")
        else:
            h_store = None
            c_store = None

        try:
            rows = self.core.append_batch(
                trace_kind_id=trace_kind_id,
                decision_count=decision_count,
                decision_count_host=decision_count_host,
                total_decision_groups=total_decision_groups,
                total_stored_decision_groups=total_stored_decision_groups,
                decision_option_idx=decision_option_idx,
                decision_target_idx=decision_target_idx,
                decision_mask=decision_mask,
                uses_none_head=uses_none_head,
                selected_indices=selected_indices,
                behavior_action_log_prob=behavior_action_log_prob,
                may_selected=may_selected,
                old_log_prob=old_log_prob,
                value=value,
                perspective_player_idx=perspective_player_idx,
                lstm_h_in=h_store,
                lstm_c_in=c_store,
            )
        except RuntimeError as exc:
            raise RuntimeError("TextReplayBuffer is full") from exc

        row_start = self.size - batch_size
        self.row_token_length_host[row_start : row_start + batch_size] = [
            int(x) for x in seq_lengths_host
        ]

        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        cu_seqlens[1:] = seq_lengths.cumsum(0).to(dtype=torch.int32)
        wrote_encoded_with_triton = False
        if self.use_triton_append:
            wrote_encoded_with_triton = append_staged_encoded_triton(
                flat_env=flat_env.to(device=self.device),
                flat_step=flat_step.to(device=self.device),
                token_ids=token_ids,
                seq_lengths=seq_lengths,
                cu_seqlens=cu_seqlens,
                card_ref_positions=card_ref_positions,
                option_positions=option_positions,
                option_mask=option_mask,
                target_positions=target_positions,
                target_mask=target_mask,
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
                max_seq_length=max(seq_lengths_host, default=0),
            )

        if not wrote_encoded_with_triton:
            flat_env = flat_env.to(device=self.device, dtype=torch.long)
            flat_step = flat_step.to(device=self.device, dtype=torch.long)
            self.row_token_start[rows] = -1
            self.row_token_length[rows] = 0
            self.card_ref_positions.index_fill_(0, rows, -1)
            self.option_positions.index_fill_(0, rows, -1)
            self.option_mask.index_fill_(0, rows, 0)
            self.target_positions.index_fill_(0, rows, -1)
            self.target_mask.index_fill_(0, rows, 0)
            row_tokens = token_ids[flat_env, flat_step]
            if total_tokens > 0:
                seq_id = torch.repeat_interleave(
                    torch.arange(batch_size, dtype=torch.int32, device=self.device),
                    seq_lengths,
                    output_size=total_tokens,
                )
                repeated_starts = torch.repeat_interleave(
                    cu_seqlens[:-1],
                    seq_lengths,
                    output_size=total_tokens,
                )
                pos_in_seq = (
                    torch.arange(total_tokens, dtype=torch.int32, device=self.device)
                    - repeated_starts
                )
                self.packed_token_ids[token_start:token_end] = row_tokens[
                    seq_id.to(dtype=torch.long), pos_in_seq.to(dtype=torch.long)
                ].to(dtype=torch.int32)
            self.row_token_start[rows] = token_start + cu_seqlens[:-1]
            seq_lengths_i32 = seq_lengths.to(dtype=torch.int32)
            self.row_token_length[rows] = seq_lengths_i32
            self.seq_lengths[rows] = seq_lengths_i32
            self.card_ref_positions[rows] = card_ref_positions[flat_env, flat_step].to(
                device=self.device, dtype=torch.int32
            )
            option_width = min(int(option_positions.shape[2]), self.max_options)
            target_width = min(int(target_positions.shape[3]), self.max_targets_per_option)
            if option_width > 0:
                self.option_positions[rows, :option_width] = option_positions[
                    flat_env, flat_step, :option_width
                ].to(device=self.device, dtype=torch.int32)
                self.option_mask[rows, :option_width] = option_mask[
                    flat_env, flat_step, :option_width
                ].to(device=self.device, dtype=torch.bool)
            if option_width > 0 and target_width > 0:
                self.target_positions[rows, :option_width, :target_width] = target_positions[
                    flat_env, flat_step, :option_width, :target_width
                ].to(device=self.device, dtype=torch.int32)
                self.target_mask[rows, :option_width, :target_width] = target_mask[
                    flat_env, flat_step, :option_width, :target_width
                ].to(device=self.device, dtype=torch.bool)

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
        self.row_token_length_host[row] = token_count
        if token_count > 0:
            self.packed_token_ids[token_start:token_end] = token_ids

    def gather(self, replay_rows: list[int] | Tensor) -> TextReplayBatch:
        if isinstance(replay_rows, Tensor):
            if int(replay_rows.numel()) == 0:
                raise ValueError("replay_rows must not be empty")
            idx = replay_rows.to(device=self.device)
            idx_host = (
                [int(x) for x in replay_rows.to(dtype=torch.long).tolist()]
                if replay_rows.device.type == "cpu"
                else None
            )
        else:
            if not replay_rows:
                raise ValueError("replay_rows must not be empty")
            idx = torch.tensor(replay_rows, dtype=torch.long, device=self.device)
            idx_host = [int(x) for x in replay_rows]
        total_tokens_host = (
            sum(self.row_token_length_host[row] for row in idx_host)
            if idx_host is not None
            else None
        )
        if self.validate:
            in_range = (idx >= 0) & (idx < self.capacity)
            if not bool(in_range.all().item()):
                bad = int(idx[~in_range][0].item())
                raise ValueError(f"replay row {bad} out of range")
            self.core.validate_occupied(idx)
        gathered_layout = None
        if self.use_triton_gather:
            gathered_layout = gather_sequence_layout_triton(
                rows=idx,
                row_token_length=self.row_token_length,
            )
        if gathered_layout is None:
            seq_lengths = self.row_token_length[idx]
            cu_seqlens, state_positions, seq_id, pos_in_seq = packed_sequence_layout(
                seq_lengths,
                total_tokens=total_tokens_host,
            )
        else:
            seq_lengths, cu_seqlens = gathered_layout
            state_positions = cu_seqlens[:-1]
            seq_id = torch.empty(0, dtype=torch.int32, device=self.device)
            pos_in_seq = torch.empty(0, dtype=torch.int32, device=self.device)
        if self.validate:
            token_starts = self.row_token_start[idx]
            if bool((token_starts < 0).any().item()):
                bad = int(idx[token_starts < 0][0].item())
                raise ValueError(f"replay row {bad} has no packed token span")
        gathered_encoded = None
        if self.use_triton_gather:
            gathered_encoded = gather_encoded_triton(
                rows=idx,
                seq_lengths=seq_lengths,
                cu_seqlens=cu_seqlens,
                packed_token_ids=self.packed_token_ids,
                row_token_start=self.row_token_start,
                total_tokens=total_tokens_host,
                include_seq_id=self.materialize_gather_seq_id,
            )

        if gathered_encoded is None:
            if seq_id.numel() == 0 and pos_in_seq.numel() == 0:
                _, _, seq_id, pos_in_seq = packed_sequence_layout(
                    seq_lengths,
                    total_tokens=total_tokens_host,
                )
            token_starts = self.row_token_start[idx]
            token_offsets = token_starts[seq_id] + pos_in_seq
            token_ids = self.packed_token_ids[token_offsets]
        else:
            token_ids, seq_id, pos_in_seq = gathered_encoded

        gathered_fields = None
        if self.use_triton_gather:
            gathered_fields = gather_text_fields_triton(
                rows=idx,
                state_positions=state_positions,
                card_ref_positions=self.card_ref_positions,
                option_positions=self.option_positions,
                option_mask=self.option_mask,
                target_positions=self.target_positions,
                target_mask=self.target_mask,
                trace_kind_id=self.trace_kind_id,
                may_selected=self.may_selected,
                old_log_prob=self.old_log_prob,
                value=self.value,
                perspective_player_idx=self.perspective_player_idx,
                lstm_h_in=self.lstm_h_in,
                lstm_c_in=self.lstm_c_in,
            )
        if gathered_fields is None:
            card_ref_positions = add_packed_offsets(self.card_ref_positions[idx], state_positions)
            option_positions = add_packed_offsets(self.option_positions[idx], state_positions)
            option_mask = self.option_mask[idx]
            target_positions = add_packed_offsets(self.target_positions[idx], state_positions)
            target_mask = self.target_mask[idx]
            common = self.core.gather_common(idx)
            trace_kind_id = common.trace_kind_id
            may_selected = common.may_selected
            old_log_prob = common.old_log_prob
            value = common.value
            perspective_player_idx = common.perspective_player_idx
            h_in = common.lstm_h_in
            c_in = common.lstm_c_in
        else:
            (
                card_ref_positions,
                option_positions,
                option_mask,
                target_positions,
                target_mask,
                trace_kind_id,
                may_selected,
                old_log_prob,
                value,
                perspective_player_idx,
                h_in,
                c_in,
            ) = gathered_fields

        gathered_decisions = None
        if self.use_triton_gather:
            gathered_decisions = gather_decisions_triton(
                rows=idx,
                decision_start=self.decision_start,
                decision_count=self.decision_count,
                decision_option_idx=self.decision_option_idx,
                decision_target_idx=self.decision_target_idx,
                decision_mask=self.decision_mask,
                uses_none_head=self.uses_none_head,
                selected_indices=self.selected_indices,
                max_decision_groups=self.max_decision_groups,
            )
        if gathered_decisions is None:
            decision_batch = self.core.gather_decisions(idx)
            decision_start = decision_batch.decision_start
            decision_count = decision_batch.decision_count
            decision_option_idx = decision_batch.decision_option_idx
            decision_target_idx = decision_batch.decision_target_idx
            decision_mask = decision_batch.decision_mask
            uses_none_head = decision_batch.uses_none_head
            selected_indices = decision_batch.selected_indices
            behavior_action_log_prob = decision_batch.behavior_action_log_prob
            step_for_decision_group = decision_batch.step_for_group
        else:
            (
                decision_start,
                decision_count,
                decision_option_idx,
                decision_target_idx,
                decision_mask,
                uses_none_head,
                selected_indices,
                step_for_decision_group,
            ) = gathered_decisions
            behavior_action_log_prob = self.core.gather_decision_behavior_action_log_prob(idx)
        encoded = PackedTextBatch(
            token_ids=token_ids,
            seq_id=seq_id,
            pos_in_seq=pos_in_seq,
            cu_seqlens=cu_seqlens,
            seq_lengths=seq_lengths,
            state_positions=state_positions,
            card_ref_positions=card_ref_positions,
            option_positions=option_positions,
            option_mask=option_mask,
            target_positions=target_positions,
            target_mask=target_mask,
            total_tokens=int(token_ids.numel()) if total_tokens_host is None else total_tokens_host,
            seq_lengths_host=tuple(self.row_token_length_host[row] for row in idx_host)
            if idx_host is not None
            else None,
        )
        return TextReplayBatch(
            encoded=encoded,
            trace_kind_id=trace_kind_id,
            decision_start=decision_start,
            decision_count=decision_count,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            selected_indices=selected_indices,
            behavior_action_log_prob=behavior_action_log_prob,
            step_for_decision_group=step_for_decision_group,
            may_selected=may_selected,
            old_log_prob=old_log_prob,
            value=value,
            perspective_player_idx=perspective_player_idx,
            lstm_h_in=h_in,
            lstm_c_in=c_in,
        )

    def write_projected_state(self, rows: Tensor, projected: Tensor) -> None:
        """Write cached LSTM input projections for the given rows.

        ``rows``: ``[N]`` long tensor of row indices.
        ``projected``: ``[N, lstm_proj_hidden]`` tensor; stored as bfloat16.
        """
        if self.projected_state is None:
            raise ValueError("buffer was constructed without projected_state storage")
        self.projected_state[rows.to(device=self.device)] = projected.to(
            device=self.device, dtype=torch.bfloat16
        )

    def _write_encoded_row(
        self,
        row: int,
        encoded: TextEncodedBatch,
        batch_index: int,
    ) -> None:
        self._validate_batch_index(encoded, batch_index)
        seq_length = int(encoded.seq_lengths[batch_index].item())
        self.card_ref_positions[row].fill_(-1)
        self.option_positions[row].fill_(-1)
        self.option_mask[row].zero_()
        self.target_positions[row].fill_(-1)
        self.target_mask[row].zero_()
        self._write_row_packed_tokens(
            row,
            encoded.token_ids[batch_index, :seq_length].to(device=self.device, dtype=torch.int32),
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

        self.card_ref_positions[row].fill_(-1)
        self.option_positions[row].fill_(-1)
        self.option_mask[row].zero_()
        self.target_positions[row].fill_(-1)
        self.target_mask[row].zero_()

        self._write_row_packed_tokens(
            row,
            encoded.token_ids[start:end].to(device=self.device, dtype=torch.int32),
        )

        state_positions = encoded.state_positions[batch_index : batch_index + 1].to(
            device=self.device, dtype=torch.int32
        )

        self.card_ref_positions[row].copy_(
            subtract_packed_offsets(
                encoded.card_ref_positions[batch_index : batch_index + 1].to(device=self.device),
                state_positions,
            )
            .squeeze(0)
            .to(device=self.device, dtype=torch.int32)
        )
        option_width = min(encoded.option_positions.shape[1], self.max_options)
        target_width = min(encoded.target_positions.shape[2], self.max_targets_per_option)
        self.option_positions[row, :option_width].copy_(
            subtract_packed_offsets(
                encoded.option_positions[batch_index : batch_index + 1, :option_width].to(
                    device=self.device
                ),
                state_positions,
            )
            .squeeze(0)
            .to(device=self.device, dtype=torch.int32)
        )
        self.option_mask[row, :option_width].copy_(
            encoded.option_mask[batch_index, :option_width].to(device=self.device)
        )
        self.target_positions[row, :option_width, :target_width].copy_(
            subtract_packed_offsets(
                encoded.target_positions[
                    batch_index : batch_index + 1, :option_width, :target_width
                ].to(device=self.device),
                state_positions,
            )
            .squeeze(0)
            .to(device=self.device, dtype=torch.int32)
        )
        self.target_mask[row, :option_width, :target_width].copy_(
            encoded.target_mask[batch_index, :option_width, :target_width].to(device=self.device)
        )
        self.seq_lengths[row] = encoded.seq_lengths[batch_index].to(
            device=self.device, dtype=torch.int32
        )

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

    def _validate_row(self, row: int) -> None:
        if row < 0 or row >= self.capacity:
            raise IndexError("replay row out of range")


__all__ = ["TextReplayBatch", "TextReplayBuffer"]
