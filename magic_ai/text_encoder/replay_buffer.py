"""Replay storage for text-encoded training rows."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from magic_ai.ppo import gae_returns_batched
from magic_ai.replay_buffer import ReplayCore
from magic_ai.text_encoder.batch import (
    PackedTextBatch,
    TextEncodedBatch,
    add_packed_offsets,
    packed_sequence_layout,
    subtract_packed_offsets,
)
from magic_ai.text_encoder.replay_triton import (
    gather_decisions_triton,
    gather_encoded_triton,
    gather_sequence_layout_triton,
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


@dataclass
class _ReplayReservation:
    reservation_id: int
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    decision_start: int
    decision_end: int
    complete: bool = False


@dataclass(frozen=True)
class TextReplayTrainWindow:
    row_start: int
    row_end: int
    rows: Tensor
    ready_events: tuple[Any, ...] = ()


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
        max_blanks_per_row: int | None = None,
        max_legal_per_blank: int | None = None,
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
        if max_blanks_per_row is None:
            max_blanks_per_row = max_options
        if max_legal_per_blank is None:
            max_legal_per_blank = max_targets_per_option + 1
        if max_blanks_per_row < 0:
            raise ValueError("max_blanks_per_row must be >= 0")
        if max_legal_per_blank < 0:
            raise ValueError("max_legal_per_blank must be >= 0")
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
        self.max_blanks_per_row = int(max_blanks_per_row)
        self.max_legal_per_blank = int(max_legal_per_blank)
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
        self.blank_positions = torch.full(
            (self.capacity, self.max_blanks_per_row),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.blank_kind = torch.zeros(
            self.capacity, self.max_blanks_per_row, dtype=torch.int32, device=self.device
        )
        self.blank_group = torch.full(
            (self.capacity, self.max_blanks_per_row),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.blank_group_kind = torch.zeros(
            self.capacity, self.max_blanks_per_row, dtype=torch.int32, device=self.device
        )
        self.blank_option_index = torch.full(
            (self.capacity, self.max_blanks_per_row),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.blank_legal_ids = torch.zeros(
            self.capacity,
            self.max_blanks_per_row,
            self.max_legal_per_blank,
            dtype=torch.int32,
            device=self.device,
        )
        self.blank_legal_mask = torch.zeros(
            self.capacity,
            self.max_blanks_per_row,
            self.max_legal_per_blank,
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
        self.episode_id = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.step_idx = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.is_terminal = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.terminal_reward_p0 = torch.zeros(
            self.capacity, dtype=torch.float32, device=self.device
        )
        self.zero_sum = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.actor_id = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.behavior_policy_version = torch.zeros(
            self.capacity, dtype=torch.long, device=self.device
        )
        self.inference_policy_version = torch.zeros(
            self.capacity, dtype=torch.long, device=self.device
        )
        self.target_policy_version = torch.full(
            (self.capacity,), -1, dtype=torch.long, device=self.device
        )
        self._reserve_lock = threading.RLock()
        self._reserve_cond = threading.Condition(self._reserve_lock)
        self._next_reservation_id = 0
        self._commit_reservation_id = 0
        self._committed_row_cursor = 0
        self._train_claim_cursor = 0
        self._row_ring_start = 0
        self._row_ring_used = 0
        self._token_ring_start = 0
        self._token_ring_used = 0
        self._decision_ring_start = 0
        self._decision_ring_used = 0
        self._ring_active = False
        self._committed_windows: list[tuple[int, int]] = []
        self._reservations: dict[int, _ReplayReservation] = {}
        self._row_complete_host = [False] * self.capacity
        self._row_ready_events: list[Any | None] = [None] * self.capacity

    @property
    def size(self) -> int:
        with self._reserve_lock:
            return int(self._row_ring_used) if self._ring_active else self.core.size

    def reset(self) -> None:
        self.core.reset()
        self.row_token_start.fill_(-1)
        self.row_token_length.zero_()
        self.row_token_length_host[:] = [0] * self.capacity
        self.blank_positions.fill_(-1)
        self.blank_kind.zero_()
        self.blank_group.fill_(-1)
        self.blank_group_kind.zero_()
        self.blank_option_index.fill_(-1)
        self.blank_legal_ids.zero_()
        self.blank_legal_mask.zero_()
        self.episode_id.fill_(-1)
        self.step_idx.fill_(-1)
        self.is_terminal.zero_()
        self.terminal_reward_p0.zero_()
        self.zero_sum.zero_()
        self.actor_id.fill_(-1)
        self.behavior_policy_version.zero_()
        self.inference_policy_version.zero_()
        self.target_policy_version.fill_(-1)
        self._token_cursor = 0
        with self._reserve_lock:
            self._next_reservation_id = 0
            self._commit_reservation_id = 0
            self._committed_row_cursor = 0
            self._train_claim_cursor = 0
            self._row_ring_start = 0
            self._row_ring_used = 0
            self._token_ring_start = 0
            self._token_ring_used = 0
            self._decision_ring_start = 0
            self._decision_ring_used = 0
            self._ring_active = False
            self._committed_windows.clear()
            self._reservations.clear()
            self._row_complete_host[:] = [False] * self.capacity
            self._row_ready_events[:] = [None] * self.capacity
            self._reserve_cond.notify_all()

    def write_ppo_targets(
        self,
        replay_rows: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> None:
        self.core.write_ppo_targets(replay_rows, old_log_probs, returns, advantages)

    def write_episode_metadata(
        self,
        replay_rows: Tensor,
        *,
        episode_id: int,
        terminal_reward_p0: float,
        zero_sum: bool,
        actor_id: int = -1,
        behavior_policy_version: int = 0,
        inference_policy_version: int = 0,
        target_policy_version: int = -1,
    ) -> None:
        """Attach episode/window metadata to already-appended replay rows."""

        rows = replay_rows.to(device=self.device)
        if int(rows.numel()) == 0:
            return
        rows_host = rows.detach().cpu().tolist()
        steps = torch.arange(int(rows.numel()), dtype=torch.long, device=self.device)
        self.episode_id[rows] = int(episode_id)
        self.step_idx[rows] = steps
        self.is_terminal[rows] = False
        self.is_terminal[rows[-1]] = True
        self.terminal_reward_p0[rows] = float(terminal_reward_p0)
        self.zero_sum[rows] = bool(zero_sum)
        self.actor_id[rows] = int(actor_id)
        self.behavior_policy_version[rows] = int(behavior_policy_version)
        self.inference_policy_version[rows] = int(inference_policy_version)
        self.target_policy_version[rows] = int(target_policy_version)
        ready_event: Any | None = None
        if self.device.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))
        for row in rows_host:
            row_i = int(row)
            self._row_ready_events[row_i % self.capacity] = ready_event
            self._row_complete_host[row_i % self.capacity] = True
        with self._reserve_cond:
            self._reserve_cond.notify_all()

    @property
    def committed_size(self) -> int:
        with self._reserve_lock:
            return int(self._committed_row_cursor)

    def reserve_append(
        self,
        *,
        row_count: int,
        token_count: int,
        decision_count: int,
    ) -> _ReplayReservation:
        return self._reserve_append(
            row_count=row_count,
            token_count=token_count,
            decision_count=decision_count,
        )

    def commit(self, reservation: _ReplayReservation) -> None:
        self._seal_reservation(reservation)

    def claim_train_window(
        self,
        *,
        min_rows: int,
        max_rows: int,
        target_rows: int | None = None,
        allow_partial: bool = False,
    ) -> TextReplayTrainWindow | None:
        if min_rows < 1:
            raise ValueError("min_rows must be at least 1")
        if max_rows < min_rows:
            raise ValueError("max_rows must be >= min_rows")
        target = int(min_rows if target_rows is None else target_rows)
        if target < min_rows:
            raise ValueError("target_rows must be >= min_rows")
        if target > max_rows:
            raise ValueError("target_rows must be <= max_rows")
        with self._reserve_lock:
            if not self._committed_windows:
                return None
            row_start, completed_limit = self._completed_window_prefix_locked()
            available = int(completed_limit - row_start)
            if available < int(min_rows):
                return None
            if not allow_partial and available < target:
                return None
            row_end = row_start + min(available, int(max_rows))
            ready_events = self._ready_events_for_rows_locked(row_start, row_end)
            self._consume_committed_prefix_locked(row_end)
        return TextReplayTrainWindow(
            row_start=row_start,
            row_end=row_end,
            rows=torch.arange(row_start, row_end, dtype=torch.long, device=self.device),
            ready_events=ready_events,
        )

    def wait_for_train_window(
        self,
        *,
        min_rows: int,
        max_rows: int,
        target_rows: int | None = None,
        allow_partial: bool = False,
        timeout: float | None = None,
    ) -> TextReplayTrainWindow | None:
        deadline = time.monotonic() + float(timeout) if timeout is not None else 0.0
        with self._reserve_cond:
            while True:
                window = self.claim_train_window(
                    min_rows=min_rows,
                    max_rows=max_rows,
                    target_rows=target_rows,
                    allow_partial=allow_partial,
                )
                if window is not None:
                    return window
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return None
                    self._reserve_cond.wait(timeout=remaining)
                else:
                    self._reserve_cond.wait()

    @torch.no_grad()
    def build_ppo_returns_for_rows(
        self,
        rows: Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> Tensor:
        rows = rows.to(device=self.device)
        if int(rows.numel()) == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        episode_ids = self.episode_id[rows]
        if bool((episode_ids < 0).any().item()):
            raise ValueError("cannot build returns for incomplete replay rows")
        unique_ids, inverse = torch.unique(episode_ids, sorted=True, return_inverse=True)
        batch_size = int(unique_ids.numel())
        step_idx = self.step_idx[rows].to(dtype=torch.int32)
        step_count = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        step_count.scatter_reduce_(
            0,
            inverse,
            step_idx + 1,
            reduce="amax",
            include_self=True,
        )
        max_steps = int(step_count.max().item())
        values = torch.zeros(batch_size, max_steps, dtype=torch.float32, device=self.device)
        players = torch.zeros(batch_size, max_steps, dtype=torch.int32, device=self.device)
        flat_dest = inverse * int(max_steps) + step_idx
        values.view(-1)[flat_dest] = self.value[rows].to(dtype=torch.float32)
        players.view(-1)[flat_dest] = self.perspective_player_idx[rows].to(dtype=torch.int32)
        terminal_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        zero_sum = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        terminal_reward[inverse] = self.terminal_reward_p0[rows].to(dtype=torch.float32)
        zero_sum[inverse] = self.zero_sum[rows]
        returns = gae_returns_batched(
            values,
            players,
            step_count,
            terminal_reward_p0=terminal_reward,
            zero_sum=zero_sum,
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
        )
        return returns.view(-1)[flat_dest]

    def release_train_window(self, window: TextReplayTrainWindow) -> None:
        self.release_rows(window.rows)

    @property
    def available_rows(self) -> int:
        with self._reserve_lock:
            return int(self.capacity - self._row_ring_used)

    @property
    def available_tokens(self) -> int:
        with self._reserve_lock:
            return int(int(self.packed_token_ids.numel()) - self._token_ring_used)

    def can_reserve(self, *, row_count: int, token_count: int, decision_count: int = 0) -> bool:
        with self._reserve_lock:
            return self._can_reserve_locked(
                row_count=int(row_count),
                token_count=int(token_count),
                decision_count=int(decision_count),
            )

    def release_rows(self, replay_rows: Tensor) -> None:
        rows = replay_rows.to(device=self.device)
        if int(rows.numel()) == 0:
            return
        row_count = int(rows.numel())
        token_lengths = self.row_token_length[rows]
        decision_counts = self.decision_count[rows]
        token_count = int(token_lengths.sum().item())
        decision_count = int(decision_counts.sum().item())
        rows_host = rows.detach().cpu().tolist()
        first_row = int(rows_host[0])
        last_row = int(rows_host[-1])
        token_starts = self.row_token_start[rows]
        decision_starts = self.decision_start[rows]
        last_token_len = int(token_lengths[-1].item())
        last_decision_len = int(decision_counts[-1].item())
        last_token_start = int(token_starts[-1].item()) if token_count > 0 else 0
        last_decision_start = int(decision_starts[-1].item()) if decision_count > 0 else 0
        with self._reserve_cond:
            if first_row == self._row_ring_start:
                self._row_ring_start = (last_row + 1) % self.capacity
            if token_count > 0:
                self._token_ring_start = (last_token_start + last_token_len) % int(
                    self.packed_token_ids.numel()
                )
            if decision_count > 0:
                self._decision_ring_start = (
                    last_decision_start + last_decision_len
                ) % self.core.decision_capacity
            self._row_ring_used = max(0, self._row_ring_used - row_count)
            self._token_ring_used = max(0, self._token_ring_used - token_count)
            self._decision_ring_used = max(0, self._decision_ring_used - decision_count)
            if self._row_ring_used == 0:
                self._row_ring_start = 0
                self._token_ring_start = 0
                self._decision_ring_start = 0
                self.core._row_cursor = 0
                self._token_cursor = 0
                self.core._decision_cursor = 0
            self.row_token_start[rows] = -1
            self.row_token_length[rows] = 0
            self.decision_count[rows] = 0
            tuple(map(lambda row: self.row_token_length_host.__setitem__(int(row), 0), rows_host))
            tuple(map(lambda row: self._row_complete_host.__setitem__(int(row), False), rows_host))
            self._reserve_cond.notify_all()

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
            return [int(x) for x in encoded.seq_lengths.tolist()]
        return [int(x) for x in encoded.seq_lengths.detach().cpu().tolist()]

    def _reserve_append(
        self,
        *,
        row_count: int,
        token_count: int,
        decision_count: int,
    ) -> _ReplayReservation:
        row_count = int(row_count)
        token_count = int(token_count)
        decision_count = int(decision_count)
        if row_count > self.capacity:
            raise RuntimeError("TextReplayBuffer row request exceeds capacity")
        if token_count > int(self.packed_token_ids.numel()):
            raise RuntimeError("TextReplayBuffer token request exceeds capacity")
        if decision_count > self.core.decision_capacity:
            raise RuntimeError("TextReplayBuffer decision request exceeds capacity")
        with self._reserve_cond:
            while not self._can_reserve_locked(
                row_count=row_count,
                token_count=token_count,
                decision_count=decision_count,
            ):
                self._reserve_cond.wait()
            row_start, row_end = self._reserve_ring_span_locked(
                cursor=self.core._row_cursor,
                start=self._row_ring_start,
                used=self._row_ring_used,
                capacity=self.capacity,
                count=row_count,
            )
            token_start, token_end = self._reserve_ring_span_locked(
                cursor=self._token_cursor,
                start=self._token_ring_start,
                used=self._token_ring_used,
                capacity=int(self.packed_token_ids.numel()),
                count=token_count,
            )
            decision_start, decision_end = self._reserve_ring_span_locked(
                cursor=self.core._decision_cursor,
                start=self._decision_ring_start,
                used=self._decision_ring_used,
                capacity=self.core.decision_capacity,
                count=decision_count,
            )
            self.core._row_cursor = row_end % self.capacity
            self._token_cursor = token_end % int(self.packed_token_ids.numel())
            self.core._decision_cursor = decision_end % self.core.decision_capacity
            self._row_ring_used += row_count
            self._token_ring_used += token_count
            self._decision_ring_used += decision_count
            self._ring_active = True
            reservation = _ReplayReservation(
                reservation_id=self._next_reservation_id,
                row_start=row_start,
                row_end=row_end,
                token_start=token_start,
                token_end=token_end,
                decision_start=decision_start,
                decision_end=decision_end,
            )
            self._reservations[reservation.reservation_id] = reservation
            self._next_reservation_id += 1
            return reservation

    def _seal_reservation(self, reservation: _ReplayReservation) -> None:
        with self._reserve_cond:
            stored = self._reservations[reservation.reservation_id]
            stored.complete = True
            while True:
                head = self._reservations.get(self._commit_reservation_id)
                if head is None or not head.complete:
                    break
                self._committed_row_cursor = head.row_end
                self._committed_windows.append((head.row_start, head.row_end))
                del self._reservations[self._commit_reservation_id]
                self._commit_reservation_id += 1
            self._reserve_cond.notify_all()

    def _can_reserve_locked(
        self,
        *,
        row_count: int,
        token_count: int,
        decision_count: int,
    ) -> bool:
        return (
            row_count <= self.capacity - self._row_ring_used
            and token_count <= int(self.packed_token_ids.numel()) - self._token_ring_used
            and decision_count <= self.core.decision_capacity - self._decision_ring_used
            and self._ring_span_fits(
                cursor=self.core._row_cursor % self.capacity,
                start=self._row_ring_start,
                used=self._row_ring_used,
                capacity=self.capacity,
                count=row_count,
            )
            and self._ring_span_fits(
                cursor=self._token_cursor % int(self.packed_token_ids.numel()),
                start=self._token_ring_start,
                used=self._token_ring_used,
                capacity=int(self.packed_token_ids.numel()),
                count=token_count,
            )
            and self._ring_span_fits(
                cursor=self.core._decision_cursor % self.core.decision_capacity,
                start=self._decision_ring_start,
                used=self._decision_ring_used,
                capacity=self.core.decision_capacity,
                count=decision_count,
            )
        )

    @staticmethod
    def _ring_span_fits(
        *,
        cursor: int,
        start: int,
        used: int,
        capacity: int,
        count: int,
    ) -> bool:
        if count == 0:
            return True
        if used == 0:
            return count <= capacity
        return cursor + count <= capacity or count <= start

    def _reserve_ring_span_locked(
        self,
        *,
        cursor: int,
        start: int,
        used: int,
        capacity: int,
        count: int,
    ) -> tuple[int, int]:
        cursor = int(cursor) % int(capacity)
        if count == 0:
            return cursor, cursor
        if used == 0 and cursor + count > capacity:
            return 0, count
        if cursor + count <= capacity:
            return cursor, cursor + count
        if count <= start:
            return 0, count
        raise RuntimeError("ring span does not fit")

    def _completed_prefix_end_locked(self, row_start: int, row_limit: int) -> int:
        row = int(row_start)
        limit = int(row_limit)
        while row < limit and self._row_complete_host[row % self.capacity]:
            row += 1
        return row

    def _completed_window_prefix_locked(self) -> tuple[int, int]:
        if not self._committed_windows:
            return 0, 0
        row_start, row_limit = self._committed_windows[0]
        completed_limit = self._completed_prefix_end_locked(row_start, row_limit)
        idx = 1
        while completed_limit == row_limit and idx < len(self._committed_windows):
            next_start, next_limit = self._committed_windows[idx]
            if int(next_start) != int(completed_limit):
                break
            row_limit = next_limit
            completed_limit = self._completed_prefix_end_locked(next_start, next_limit)
            idx += 1
        return int(row_start), int(completed_limit)

    def _ready_events_for_rows_locked(self, row_start: int, row_end: int) -> tuple[Any, ...]:
        events: list[Any] = []
        seen: set[int] = set()
        for row in range(int(row_start), int(row_end)):
            event = self._row_ready_events[row % self.capacity]
            if event is None:
                continue
            event_id = id(event)
            if event_id in seen:
                continue
            seen.add(event_id)
            events.append(event)
        return tuple(events)

    def _consume_committed_prefix_locked(self, row_end: int) -> None:
        row_end = int(row_end)
        if self._committed_windows:
            start = int(self._committed_windows[0][0])
            for row in range(start, row_end):
                self._row_ready_events[row % self.capacity] = None
        while self._committed_windows and self._committed_windows[0][1] <= row_end:
            self._committed_windows.pop(0)
        if self._committed_windows and self._committed_windows[0][0] < row_end:
            _start, limit = self._committed_windows[0]
            self._committed_windows[0] = (row_end, limit)

    def _write_decision_batch_at(
        self,
        *,
        row_start: int,
        row_end: int,
        decision_start: int,
        decision_count: Tensor,
        decision_count_host: tuple[int, ...] | None = None,
        total_decision_groups: int | None = None,
        total_stored_decision_groups: int | None = None,
        decision_option_idx: Tensor,
        decision_target_idx: Tensor,
        decision_mask: Tensor,
        uses_none_head: Tensor,
        selected_indices: Tensor,
        behavior_action_log_prob: Tensor | None,
    ) -> Tensor:
        counts = decision_count.to(device=self.device)
        batch_size = row_end - row_start
        if int(counts.shape[0]) != batch_size:
            raise ValueError("decision_count length must match appended row count")
        if decision_count_host is not None and len(decision_count_host) != batch_size:
            raise ValueError("decision_count_host length must match appended row count")
        stored_counts = counts.clamp(max=self.max_decision_groups)
        total_stored = (
            int(total_stored_decision_groups)
            if total_stored_decision_groups is not None
            else (
                sum(min(int(c), self.max_decision_groups) for c in decision_count_host)
                if decision_count_host is not None
                else int(stored_counts.sum().item())
            )
        )
        starts = torch.cumsum(stored_counts, dim=0) - stored_counts + int(decision_start)
        self.decision_start[row_start:row_end] = starts
        self.decision_count[row_start:row_end] = stored_counts
        if total_stored == 0:
            return stored_counts

        total_source = (
            int(total_decision_groups)
            if total_decision_groups is not None
            else int(decision_option_idx.shape[0])
        )
        required_source = (
            sum(int(c) for c in decision_count_host)
            if decision_count_host is not None
            else int(counts.sum().item())
        )
        if total_source < required_source:
            raise ValueError("decision tensors do not contain decision_count groups")
        if int(decision_option_idx.shape[0]) < total_source:
            raise ValueError("decision_option_idx has fewer rows than total_decision_groups")
        step_for_source = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device),
            counts,
            output_size=total_source,
        )
        source_group_starts = torch.cumsum(counts, dim=0) - counts
        group_in_step = (
            torch.arange(total_source, device=self.device) - source_group_starts[step_for_source]
        )
        valid = group_in_step < self.max_decision_groups
        relative_dest = starts[step_for_source] - int(decision_start) + group_in_step
        dummy_dest = relative_dest.new_full((), total_stored)
        relative_dest = torch.where(valid, relative_dest, dummy_dest)

        option_tmp = torch.full(
            (total_stored + 1, self.max_cached_choices),
            -1,
            dtype=self.core.index_dtype,
            device=self.device,
        )
        target_tmp = torch.full_like(option_tmp, -1)
        mask_tmp = torch.zeros(
            total_stored + 1,
            self.max_cached_choices,
            dtype=torch.bool,
            device=self.device,
        )
        none_tmp = torch.zeros(total_stored + 1, dtype=torch.bool, device=self.device)
        selected_tmp = torch.full(
            (total_stored + 1,),
            -1,
            dtype=self.core.index_dtype,
            device=self.device,
        )
        behavior_lp_tmp = torch.zeros(total_stored + 1, dtype=torch.float32, device=self.device)

        option_tmp[relative_dest] = decision_option_idx[:total_source].to(
            device=self.device, dtype=self.core.index_dtype
        )
        target_tmp[relative_dest] = decision_target_idx[:total_source].to(
            device=self.device, dtype=self.core.index_dtype
        )
        mask_tmp[relative_dest] = decision_mask[:total_source].to(
            device=self.device, dtype=torch.bool
        )
        none_tmp[relative_dest] = uses_none_head[:total_source].to(
            device=self.device, dtype=torch.bool
        )
        selected_tmp[relative_dest] = selected_indices[:total_source].to(
            device=self.device, dtype=self.core.index_dtype
        )
        if behavior_action_log_prob is not None:
            behavior_lp_tmp[relative_dest] = behavior_action_log_prob[:total_source].to(
                device=self.device, dtype=torch.float32
            )

        decision_end = int(decision_start) + total_stored
        self.decision_option_idx[decision_start:decision_end] = option_tmp[:-1]
        self.decision_target_idx[decision_start:decision_end] = target_tmp[:-1]
        self.decision_mask[decision_start:decision_end] = mask_tmp[:-1]
        self.uses_none_head[decision_start:decision_end] = none_tmp[:-1]
        self.selected_indices[decision_start:decision_end] = selected_tmp[:-1]
        self.behavior_action_log_prob[decision_start:decision_end] = behavior_lp_tmp[:-1]
        return stored_counts

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
        seq_lengths = encoded.seq_lengths.to(device=self.device)
        if self.validate:
            max_seq_length = int(seq_lengths.max().item()) if batch_size > 0 else 0
            if max_seq_length > self.max_tokens:
                raise ValueError("encoded packed row token width exceeds buffer max_tokens")
            self._validate_packed_blank_widths(encoded)
        total_tokens = int(encoded.token_ids.numel())
        seq_lengths_host = self._packed_seq_lengths_host(encoded)
        if decision_count_host is not None and len(decision_count_host) != batch_size:
            raise ValueError("decision_count_host length must match batch size")
        total_stored = (
            int(total_stored_decision_groups)
            if total_stored_decision_groups is not None
            else (
                sum(map(lambda c: min(int(c), self.max_decision_groups), decision_count_host))
                if decision_count_host is not None
                else int(
                    decision_count.to(device=self.device)
                    .clamp(max=self.max_decision_groups)
                    .sum()
                    .item()
                )
            )
        )
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
            reservation = self._reserve_append(
                row_count=batch_size,
                token_count=total_tokens,
                decision_count=total_stored,
            )
        except RuntimeError as exc:
            raise RuntimeError("TextReplayBuffer is full") from exc

        row_start = reservation.row_start
        row_end = reservation.row_end
        token_start = reservation.token_start
        token_end = reservation.token_end
        rows = torch.arange(row_start, row_end, dtype=torch.long, device=self.device)
        self._write_decision_batch_at(
            row_start=row_start,
            row_end=row_end,
            decision_start=reservation.decision_start,
            decision_count=decision_count,
            decision_count_host=decision_count_host,
            total_decision_groups=total_decision_groups,
            total_stored_decision_groups=total_stored,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            selected_indices=selected_indices,
            behavior_action_log_prob=behavior_action_log_prob,
        )
        self.trace_kind_id[rows] = trace_kind_id.to(
            device=self.device, dtype=self.trace_kind_id.dtype
        )
        self.may_selected[rows] = may_selected.to(device=self.device, dtype=torch.float32)
        self.old_log_prob[rows] = old_log_prob.to(device=self.device, dtype=torch.float32)
        self.value[rows] = value.to(device=self.device, dtype=torch.float32)
        self.perspective_player_idx[rows] = perspective_player_idx.to(
            device=self.device, dtype=self.perspective_player_idx.dtype
        )
        if self.lstm_h_in is not None and self.lstm_c_in is not None:
            if h_store is None or c_store is None:
                raise ValueError("h_in and c_in are required for recurrent text replay")
            self.lstm_h_in[rows] = h_store.to(device=self.device, dtype=torch.float32)
            self.lstm_c_in[rows] = c_store.to(device=self.device, dtype=torch.float32)
        self.row_token_length_host[row_start : row_start + batch_size] = seq_lengths_host
        blank_width = min(int(encoded.blank_positions.shape[1]), self.max_blanks_per_row)
        blank_legal_width = min(int(encoded.blank_legal_ids.shape[2]), self.max_legal_per_blank)
        seq_lengths_i32 = seq_lengths.to(dtype=torch.int32)
        self.row_token_start[rows] = -1
        self.row_token_length[rows] = 0
        self.card_ref_positions.index_fill_(0, rows, -1)
        self.seq_lengths[rows] = seq_lengths_i32

        self.blank_positions.index_fill_(0, rows, -1)
        self.blank_kind.index_fill_(0, rows, 0)
        self.blank_group.index_fill_(0, rows, -1)
        self.blank_group_kind.index_fill_(0, rows, 0)
        self.blank_option_index.index_fill_(0, rows, -1)
        self.blank_legal_ids.index_fill_(0, rows, 0)
        self.blank_legal_mask.index_fill_(0, rows, 0)
        token_ids = encoded.token_ids.to(device=self.device, dtype=torch.int32)
        if total_tokens > 0:
            self.packed_token_ids[token_start:token_end] = token_ids
        self.row_token_start[rows] = token_start + encoded.cu_seqlens[:-1].to(
            device=self.device, dtype=torch.int32
        )
        self.row_token_length[rows] = seq_lengths_i32

        state_positions = encoded.state_positions.to(device=self.device, dtype=torch.int32)

        self.card_ref_positions[rows] = subtract_packed_offsets(
            encoded.card_ref_positions.to(device=self.device),
            state_positions,
        )
        if blank_width > 0:
            state_positions = encoded.state_positions.to(device=self.device, dtype=torch.int32)
            self.blank_positions[rows, :blank_width] = subtract_packed_offsets(
                encoded.blank_positions[:, :blank_width].to(device=self.device),
                state_positions,
            )
            self.blank_kind[rows, :blank_width] = encoded.blank_kind[:, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
            self.blank_group[rows, :blank_width] = encoded.blank_group[:, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
            self.blank_group_kind[rows, :blank_width] = encoded.blank_group_kind[
                :, :blank_width
            ].to(device=self.device, dtype=torch.int32)
            self.blank_option_index[rows, :blank_width] = encoded.blank_option_index[
                :, :blank_width
            ].to(device=self.device, dtype=torch.int32)
            if blank_legal_width > 0:
                self.blank_legal_ids[rows, :blank_width, :blank_legal_width] = (
                    encoded.blank_legal_ids[:, :blank_width, :blank_legal_width].to(
                        device=self.device, dtype=torch.int32
                    )
                )
                self.blank_legal_mask[rows, :blank_width, :blank_legal_width] = (
                    encoded.blank_legal_mask[:, :blank_width, :blank_legal_width].to(
                        device=self.device, dtype=torch.bool
                    )
                )

        self._seal_reservation(reservation)
        return rows

    def write_batch_at(
        self,
        *,
        rows: Tensor,
        rows_host: tuple[int, ...],
        encoded: PackedTextBatch,
        trace_kind_id: Tensor,
        decision_count: Tensor,
        decision_count_host: tuple[int, ...] | None = None,
        total_decision_groups: int | None = None,
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
        """Write replay rows at pre-reserved fixed row ids.

        This is the native actor hot path: row ids are claimed by the caller
        with host-side slot/step bookkeeping, then this method performs the
        tensor writes directly into final replay storage.
        """

        batch_size = int(encoded.seq_lengths.shape[0])
        if batch_size == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if len(rows_host) != batch_size:
            raise ValueError("rows_host length must match batch size")
        if max(rows_host) >= self.capacity or min(rows_host) < 0:
            raise RuntimeError("TextReplayBuffer fixed replay row is out of range")
        rows = rows.to(device=self.device)
        seq_lengths = encoded.seq_lengths.to(device=self.device)
        seq_lengths_host = self._packed_seq_lengths_host(encoded)
        if self.validate:
            max_seq_length = max(seq_lengths_host, default=0)
            if max_seq_length > self.max_tokens:
                raise ValueError("encoded packed row token width exceeds buffer max_tokens")
            self._validate_packed_blank_widths(encoded)
        if decision_count_host is not None and len(decision_count_host) != batch_size:
            raise ValueError("decision_count_host length must match batch size")

        # Mark sparse fixed rows as occupied for gather-time validation. The
        # caller only samples explicit row ids, so holes below this cursor are
        # harmless.
        self.core._row_cursor = max(self.core._row_cursor, max(rows_host) + 1)
        for row, seq_len in zip(rows_host, seq_lengths_host, strict=True):
            self.row_token_length_host[int(row)] = int(seq_len)

        total_tokens = int(encoded.token_ids.numel())
        token_start = self._token_cursor
        token_end = token_start + total_tokens
        if token_end > int(self.packed_token_ids.numel()):
            raise RuntimeError("TextReplayBuffer packed token arena is full")
        self._token_cursor = token_end
        seq_lengths_i32 = seq_lengths.to(dtype=torch.int32)
        self.row_token_length[rows] = seq_lengths_i32
        self.seq_lengths[rows] = seq_lengths_i32
        if total_tokens > 0:
            self.packed_token_ids[token_start:token_end] = encoded.token_ids.to(
                device=self.device, dtype=torch.int32
            )
        self.row_token_start[rows] = token_start + encoded.cu_seqlens[:-1].to(
            device=self.device, dtype=torch.int32
        )

        state_positions = encoded.state_positions.to(device=self.device, dtype=torch.int32)
        self.card_ref_positions.index_fill_(0, rows, -1)
        self.card_ref_positions[rows] = subtract_packed_offsets(
            encoded.card_ref_positions.to(device=self.device),
            state_positions,
        )
        self.blank_positions.index_fill_(0, rows, -1)
        self.blank_kind.index_fill_(0, rows, 0)
        self.blank_group.index_fill_(0, rows, -1)
        self.blank_group_kind.index_fill_(0, rows, 0)
        self.blank_option_index.index_fill_(0, rows, -1)
        self.blank_legal_ids.index_fill_(0, rows, 0)
        self.blank_legal_mask.index_fill_(0, rows, 0)
        blank_width = min(int(encoded.blank_positions.shape[1]), self.max_blanks_per_row)
        blank_legal_width = min(int(encoded.blank_legal_ids.shape[2]), self.max_legal_per_blank)
        if blank_width > 0:
            self.blank_positions[rows, :blank_width] = subtract_packed_offsets(
                encoded.blank_positions[:, :blank_width].to(device=self.device),
                state_positions,
            )
            self.blank_kind[rows, :blank_width] = encoded.blank_kind[:, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
            self.blank_group[rows, :blank_width] = encoded.blank_group[:, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
            self.blank_group_kind[rows, :blank_width] = encoded.blank_group_kind[
                :, :blank_width
            ].to(device=self.device, dtype=torch.int32)
            self.blank_option_index[rows, :blank_width] = encoded.blank_option_index[
                :, :blank_width
            ].to(device=self.device, dtype=torch.int32)
            if blank_legal_width > 0:
                self.blank_legal_ids[rows, :blank_width, :blank_legal_width] = (
                    encoded.blank_legal_ids[:, :blank_width, :blank_legal_width].to(
                        device=self.device, dtype=torch.int32
                    )
                )
                self.blank_legal_mask[rows, :blank_width, :blank_legal_width] = (
                    encoded.blank_legal_mask[:, :blank_width, :blank_legal_width].to(
                        device=self.device, dtype=torch.bool
                    )
                )

        counts = decision_count.to(device=self.device)
        stored_counts = counts.clamp(max=self.max_decision_groups)
        decision_starts = rows * int(self.max_decision_groups)
        self.decision_start[rows] = decision_starts.to(dtype=self.decision_start.dtype)
        self.decision_count[rows] = stored_counts.to(dtype=self.decision_count.dtype)
        decision_slots = rows.unsqueeze(1) * int(self.max_decision_groups) + torch.arange(
            self.max_decision_groups, dtype=torch.long, device=self.device
        )
        flat_slots = decision_slots.reshape(-1)
        self.decision_option_idx.index_fill_(0, flat_slots, -1)
        self.decision_target_idx.index_fill_(0, flat_slots, -1)
        self.decision_mask.index_fill_(0, flat_slots, 0)
        self.uses_none_head.index_fill_(0, flat_slots, 0)
        self.selected_indices.index_fill_(0, flat_slots, -1)
        self.behavior_action_log_prob.index_fill_(0, flat_slots, 0)

        total_source = (
            int(total_decision_groups)
            if total_decision_groups is not None
            else int(decision_option_idx.shape[0])
        )
        if total_source > 0:
            step_for_source = torch.repeat_interleave(
                torch.arange(batch_size, device=self.device),
                counts,
                output_size=total_source,
            )
            source_group_starts = torch.cumsum(counts, dim=0) - counts
            group_in_step = (
                torch.arange(total_source, device=self.device)
                - source_group_starts[step_for_source]
            )
            valid = group_in_step < self.max_decision_groups
            dest = decision_starts[step_for_source] + group_in_step
            dest = dest[valid]
            src = valid
            self.decision_option_idx[dest] = decision_option_idx[:total_source][src].to(
                device=self.device, dtype=self.core.index_dtype
            )
            self.decision_target_idx[dest] = decision_target_idx[:total_source][src].to(
                device=self.device, dtype=self.core.index_dtype
            )
            self.decision_mask[dest] = decision_mask[:total_source][src].to(
                device=self.device, dtype=torch.bool
            )
            self.uses_none_head[dest] = uses_none_head[:total_source][src].to(
                device=self.device, dtype=torch.bool
            )
            self.selected_indices[dest] = selected_indices[:total_source][src].to(
                device=self.device, dtype=self.core.index_dtype
            )
            if behavior_action_log_prob is not None:
                self.behavior_action_log_prob[dest] = behavior_action_log_prob[:total_source][
                    src
                ].to(device=self.device, dtype=torch.float32)

        self.trace_kind_id[rows] = trace_kind_id.to(
            device=self.device, dtype=self.trace_kind_id.dtype
        )
        self.may_selected[rows] = may_selected.to(device=self.device, dtype=torch.float32)
        self.old_log_prob[rows] = old_log_prob.to(device=self.device, dtype=torch.float32)
        self.value[rows] = value.to(device=self.device, dtype=torch.float32)
        self.perspective_player_idx[rows] = perspective_player_idx.to(
            device=self.device, dtype=self.perspective_player_idx.dtype
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
        blank_positions: Tensor,
        blank_kind: Tensor,
        blank_group: Tensor,
        blank_group_kind: Tensor,
        blank_option_index: Tensor,
        blank_legal_ids: Tensor,
        blank_legal_mask: Tensor,
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
        seq_lengths = seq_lengths.to(device=self.device)
        if self.validate:
            max_seq_length = max(seq_lengths_host, default=0)
            if max_seq_length > self.max_tokens:
                raise ValueError("encoded staged row token width exceeds buffer max_tokens")
        total_tokens = sum(int(x) for x in seq_lengths_host)
        if decision_count_host is not None and len(decision_count_host) != batch_size:
            raise ValueError("decision_count_host length must match batch size")
        total_stored = (
            int(total_stored_decision_groups)
            if total_stored_decision_groups is not None
            else (
                sum(min(int(c), self.max_decision_groups) for c in decision_count_host)
                if decision_count_host is not None
                else int(
                    decision_count.to(device=self.device)
                    .clamp(max=self.max_decision_groups)
                    .sum()
                    .item()
                )
            )
        )
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
            reservation = self._reserve_append(
                row_count=batch_size,
                token_count=total_tokens,
                decision_count=total_stored,
            )
        except RuntimeError as exc:
            raise RuntimeError("TextReplayBuffer is full") from exc

        row_start = reservation.row_start
        row_end = reservation.row_end
        token_start = reservation.token_start
        token_end = reservation.token_end
        rows = torch.arange(row_start, row_end, dtype=torch.long, device=self.device)
        self.row_token_length_host[row_start : row_start + batch_size] = [
            int(x) for x in seq_lengths_host
        ]

        self._write_decision_batch_at(
            row_start=row_start,
            row_end=row_end,
            decision_start=reservation.decision_start,
            decision_count=decision_count,
            decision_count_host=decision_count_host,
            total_decision_groups=total_decision_groups,
            total_stored_decision_groups=total_stored,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            selected_indices=selected_indices,
            behavior_action_log_prob=behavior_action_log_prob,
        )
        self.trace_kind_id[row_start:row_end] = trace_kind_id.to(
            device=self.device, dtype=self.trace_kind_id.dtype
        )
        self.may_selected[row_start:row_end] = may_selected.to(
            device=self.device, dtype=torch.float32
        )
        self.old_log_prob[row_start:row_end] = old_log_prob.to(
            device=self.device, dtype=torch.float32
        )
        self.value[row_start:row_end] = value.to(device=self.device, dtype=torch.float32)
        self.perspective_player_idx[row_start:row_end] = perspective_player_idx.to(
            device=self.device, dtype=self.perspective_player_idx.dtype
        )
        self.core._write_recurrent_batch(row_start, row_end, h_store, c_store)

        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        cu_seqlens[1:] = seq_lengths.cumsum(0).to(dtype=torch.int32)
        flat_env = flat_env.to(device=self.device)
        flat_step = flat_step.to(device=self.device)
        self.row_token_start[row_start:row_end] = -1
        self.row_token_length[row_start:row_end] = 0
        self.card_ref_positions[row_start:row_end].fill_(-1)
        self.blank_positions[row_start:row_end].fill_(-1)
        self.blank_kind[row_start:row_end].zero_()
        self.blank_group[row_start:row_end].fill_(-1)
        self.blank_group_kind[row_start:row_end].zero_()
        self.blank_option_index[row_start:row_end].fill_(-1)
        self.blank_legal_ids[row_start:row_end].zero_()
        self.blank_legal_mask[row_start:row_end].zero_()
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
                torch.arange(total_tokens, dtype=torch.int32, device=self.device) - repeated_starts
            )
            self.packed_token_ids[token_start:token_end] = row_tokens[seq_id, pos_in_seq].to(
                dtype=torch.int32
            )
        self.row_token_start[rows] = token_start + cu_seqlens[:-1]
        seq_lengths_i32 = seq_lengths.to(dtype=torch.int32)
        self.row_token_length[rows] = seq_lengths_i32
        self.seq_lengths[rows] = seq_lengths_i32
        self.card_ref_positions[rows] = card_ref_positions[flat_env, flat_step].to(
            device=self.device, dtype=torch.int32
        )
        blank_width = min(int(blank_positions.shape[2]), self.max_blanks_per_row)
        blank_legal_width = min(int(blank_legal_ids.shape[3]), self.max_legal_per_blank)
        if blank_width > 0:
            self.blank_positions[rows, :blank_width] = blank_positions[
                flat_env, flat_step, :blank_width
            ].to(device=self.device, dtype=torch.int32)
            self.blank_kind[rows, :blank_width] = blank_kind[flat_env, flat_step, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
            self.blank_group[rows, :blank_width] = blank_group[
                flat_env, flat_step, :blank_width
            ].to(device=self.device, dtype=torch.int32)
            self.blank_group_kind[rows, :blank_width] = blank_group_kind[
                flat_env, flat_step, :blank_width
            ].to(device=self.device, dtype=torch.int32)
            self.blank_option_index[rows, :blank_width] = blank_option_index[
                flat_env, flat_step, :blank_width
            ].to(device=self.device, dtype=torch.int32)
            if blank_legal_width > 0:
                self.blank_legal_ids[rows, :blank_width, :blank_legal_width] = blank_legal_ids[
                    flat_env, flat_step, :blank_width, :blank_legal_width
                ].to(device=self.device, dtype=torch.int32)
                self.blank_legal_mask[rows, :blank_width, :blank_legal_width] = blank_legal_mask[
                    flat_env, flat_step, :blank_width, :blank_legal_width
                ].to(device=self.device, dtype=torch.bool)

        self._seal_reservation(reservation)
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
                [int(x) for x in replay_rows.tolist()] if replay_rows.device.type == "cpu" else None
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
                raise ValueError(f"replay row {bad} is not occupied")
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

        card_ref_positions = add_packed_offsets(self.card_ref_positions[idx], state_positions)
        common = self.core.gather_common(idx)
        trace_kind_id = common.trace_kind_id
        may_selected = common.may_selected
        old_log_prob = common.old_log_prob
        value = common.value
        perspective_player_idx = common.perspective_player_idx
        h_in = common.lstm_h_in
        c_in = common.lstm_c_in
        blank_positions = add_packed_offsets(self.blank_positions[idx], state_positions)
        blank_kind = self.blank_kind[idx]
        blank_group = self.blank_group[idx]
        blank_group_kind = self.blank_group_kind[idx]
        blank_option_index = self.blank_option_index[idx]
        blank_legal_ids = self.blank_legal_ids[idx]
        blank_legal_mask = self.blank_legal_mask[idx]

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
            decision_start = gathered_decisions.decision_start
            decision_count = gathered_decisions.decision_count
            decision_option_idx = gathered_decisions.decision_option_idx
            decision_target_idx = gathered_decisions.decision_target_idx
            decision_mask = gathered_decisions.decision_mask
            uses_none_head = gathered_decisions.uses_none_head
            selected_indices = gathered_decisions.selected_indices
            step_for_decision_group = gathered_decisions.step_for_group
            behavior_action_log_prob = self.core.gather_decision_behavior_action_log_prob(idx)
        encoded = PackedTextBatch(
            token_ids=token_ids,
            seq_id=seq_id,
            pos_in_seq=pos_in_seq,
            cu_seqlens=cu_seqlens,
            seq_lengths=seq_lengths,
            state_positions=state_positions,
            card_ref_positions=card_ref_positions,
            total_tokens=int(token_ids.numel()) if total_tokens_host is None else total_tokens_host,
            seq_lengths_host=tuple(self.row_token_length_host[row] for row in idx_host)
            if idx_host is not None
            else None,
            blank_positions=blank_positions,
            blank_kind=blank_kind,
            blank_group=blank_group,
            blank_group_kind=blank_group_kind,
            blank_option_index=blank_option_index,
            blank_legal_ids=blank_legal_ids,
            blank_legal_mask=blank_legal_mask,
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
        self._clear_blank_row(row)
        self._write_row_packed_tokens(
            row,
            encoded.token_ids[batch_index, :seq_length].to(device=self.device, dtype=torch.int32),
        )
        self.card_ref_positions[row].copy_(
            encoded.card_ref_positions[batch_index].to(device=self.device, dtype=torch.int32)
        )
        blank_width = min(encoded.blank_positions.shape[1], self.max_blanks_per_row)
        blank_legal_width = min(encoded.blank_legal_ids.shape[2], self.max_legal_per_blank)
        self._copy_padded_blank_row(
            row,
            encoded,
            batch_index,
            blank_width=blank_width,
            blank_legal_width=blank_legal_width,
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
        self._clear_blank_row(row)

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
        blank_width = min(encoded.blank_positions.shape[1], self.max_blanks_per_row)
        blank_legal_width = min(encoded.blank_legal_ids.shape[2], self.max_legal_per_blank)
        self._copy_packed_blank_row(
            row,
            encoded,
            batch_index,
            state_positions,
            blank_width=blank_width,
            blank_legal_width=blank_legal_width,
        )
        self.seq_lengths[row] = encoded.seq_lengths[batch_index].to(
            device=self.device, dtype=torch.int32
        )

    def _clear_blank_row(self, row: int) -> None:
        self.blank_positions[row].fill_(-1)
        self.blank_kind[row].zero_()
        self.blank_group[row].fill_(-1)
        self.blank_group_kind[row].zero_()
        self.blank_option_index[row].fill_(-1)
        self.blank_legal_ids[row].zero_()
        self.blank_legal_mask[row].zero_()

    def _copy_padded_blank_row(
        self,
        row: int,
        encoded: TextEncodedBatch,
        batch_index: int,
        *,
        blank_width: int,
        blank_legal_width: int,
    ) -> None:
        if blank_width == 0:
            return
        self.blank_positions[row, :blank_width].copy_(
            encoded.blank_positions[batch_index, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        self.blank_kind[row, :blank_width].copy_(
            encoded.blank_kind[batch_index, :blank_width].to(device=self.device, dtype=torch.int32)
        )
        self.blank_group[row, :blank_width].copy_(
            encoded.blank_group[batch_index, :blank_width].to(device=self.device, dtype=torch.int32)
        )
        self.blank_group_kind[row, :blank_width].copy_(
            encoded.blank_group_kind[batch_index, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        self.blank_option_index[row, :blank_width].copy_(
            encoded.blank_option_index[batch_index, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        if blank_legal_width == 0:
            return
        self.blank_legal_ids[row, :blank_width, :blank_legal_width].copy_(
            encoded.blank_legal_ids[batch_index, :blank_width, :blank_legal_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        self.blank_legal_mask[row, :blank_width, :blank_legal_width].copy_(
            encoded.blank_legal_mask[batch_index, :blank_width, :blank_legal_width].to(
                device=self.device
            )
        )

    def _copy_packed_blank_row(
        self,
        row: int,
        encoded: PackedTextBatch,
        batch_index: int,
        state_positions: Tensor,
        *,
        blank_width: int,
        blank_legal_width: int,
    ) -> None:
        if blank_width == 0:
            return
        self.blank_positions[row, :blank_width].copy_(
            subtract_packed_offsets(
                encoded.blank_positions[batch_index : batch_index + 1, :blank_width].to(
                    device=self.device
                ),
                state_positions,
            )
            .squeeze(0)
            .to(device=self.device, dtype=torch.int32)
        )
        self.blank_kind[row, :blank_width].copy_(
            encoded.blank_kind[batch_index, :blank_width].to(device=self.device, dtype=torch.int32)
        )
        self.blank_group[row, :blank_width].copy_(
            encoded.blank_group[batch_index, :blank_width].to(device=self.device, dtype=torch.int32)
        )
        self.blank_group_kind[row, :blank_width].copy_(
            encoded.blank_group_kind[batch_index, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        self.blank_option_index[row, :blank_width].copy_(
            encoded.blank_option_index[batch_index, :blank_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        if blank_legal_width == 0:
            return
        self.blank_legal_ids[row, :blank_width, :blank_legal_width].copy_(
            encoded.blank_legal_ids[batch_index, :blank_width, :blank_legal_width].to(
                device=self.device, dtype=torch.int32
            )
        )
        self.blank_legal_mask[row, :blank_width, :blank_legal_width].copy_(
            encoded.blank_legal_mask[batch_index, :blank_width, :blank_legal_width].to(
                device=self.device
            )
        )

    def _validate_batch_index(self, encoded: TextEncodedBatch, batch_index: int) -> None:
        batch_size = int(encoded.token_ids.shape[0])
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError("batch_index out of range")
        if encoded.token_ids.shape[1] > self.max_tokens:
            raise ValueError("encoded batch token width exceeds buffer max_tokens")
        if encoded.card_ref_positions.shape[1] != self.max_card_refs:
            raise ValueError("encoded card_ref_positions width does not match buffer")
        self._validate_blank_widths(encoded)

    def _validate_packed_batch_index(self, encoded: PackedTextBatch, batch_index: int) -> None:
        batch_size = int(encoded.seq_lengths.shape[0])
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError("batch_index out of range")
        if encoded.card_ref_positions.shape[1] != self.max_card_refs:
            raise ValueError("encoded card_ref_positions width does not match buffer")
        self._validate_packed_blank_widths(encoded)

    def _validate_blank_widths(self, encoded: TextEncodedBatch) -> None:
        if encoded.blank_positions.shape[1] > self.max_blanks_per_row:
            raise ValueError("encoded blank width exceeds buffer max_blanks_per_row")
        if encoded.blank_legal_ids.shape[2] > self.max_legal_per_blank:
            raise ValueError("encoded blank legal width exceeds buffer max_legal_per_blank")

    def _validate_packed_blank_widths(self, encoded: PackedTextBatch) -> None:
        if encoded.blank_positions.shape[1] > self.max_blanks_per_row:
            raise ValueError("encoded blank width exceeds buffer max_blanks_per_row")
        if encoded.blank_legal_ids.shape[2] > self.max_legal_per_blank:
            raise ValueError("encoded blank legal width exceeds buffer max_legal_per_blank")

    def _validate_row(self, row: int) -> None:
        if row < 0 or row >= self.capacity:
            raise IndexError("replay row out of range")


ConcurrentTextReplayRing = TextReplayBuffer


__all__ = [
    "ConcurrentTextReplayRing",
    "TextReplayBatch",
    "TextReplayBuffer",
    "TextReplayTrainWindow",
]
