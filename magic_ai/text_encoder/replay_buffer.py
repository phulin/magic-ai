"""Replay storage for text-encoded training rows.

V2 layout: stores autoregressive-decoder targets per row (output token
ids, pointer positions, anchor metadata, legal-edge bitmaps) in place of
the V1 inline-blank fields. V1 rows are not readable by V2 code.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from magic_ai.aggregate_tensor import AggregateTensor, Field
from magic_ai.replay_buffer import ReplayCore
from magic_ai.returns import gae_returns_batched
from magic_ai.text_encoder.batch import (
    PackedTextBatch,
    TextEncodedBatch,
    packed_sequence_layout,
)
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, GrammarVocab
from magic_ai.text_encoder.replay_triton import (
    gather_decisions_triton,
    gather_encoded_triton,
    gather_sequence_layout_triton,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS

_REPLAY_FORMAT_VERSION = 2


@dataclass(frozen=True)
class DecoderDecisionPayload:
    """Per-row decoder targets written into the replay ring at commit time.

    Sizes may be smaller than the buffer's ``max_decoder_len`` /
    ``max_anchors`` / ``max_blockers`` / ``max_attackers``; the buffer
    pads with the appropriate fill values when scattering into the ring.
    """

    output_token_ids: Tensor  # [B, L] int32 (GrammarVocab.PAD = 0)
    output_pointer_pos: Tensor  # [B, L] int32 (-1 fill)
    output_is_pointer: Tensor  # [B, L] bool
    output_pad_mask: Tensor  # [B, L] bool
    output_log_prob: Tensor  # [B, L] float32 — rollout-time log p of chosen step
    decision_type: Tensor  # [B] int32
    pointer_anchor_positions: Tensor  # [B, N] int32 (-1 fill)
    pointer_anchor_kinds: Tensor  # [B, N] int32 (-1 fill)
    pointer_anchor_subjects: Tensor  # [B, N] int32 (-1 fill)
    pointer_anchor_handles: Tensor  # [B, N] int32 (-1 fill)
    pointer_anchor_count: Tensor  # [B] int32
    legal_edge_bitmap: Tensor  # [B, max_blockers, max_attackers] bool
    legal_edge_n_blockers: Tensor  # [B] int32
    legal_edge_n_attackers: Tensor  # [B] int32
    # Per-step grammar masks captured by the live sampler.
    vocab_mask: Tensor  # [B, L, V_vocab] bool
    pointer_mask: Tensor  # [B, L, T_enc] bool — col width may be < buffer.max_tokens


@dataclass(frozen=True)
class DecoderGatherOutput:
    """Decoder targets gathered from the replay ring for training."""

    output_token_ids: Tensor
    output_pointer_pos: Tensor
    output_is_pointer: Tensor
    output_pad_mask: Tensor
    output_log_prob: Tensor
    decision_type: Tensor
    pointer_anchor_positions: Tensor
    pointer_anchor_kinds: Tensor
    pointer_anchor_subjects: Tensor
    pointer_anchor_handles: Tensor
    pointer_anchor_count: Tensor
    legal_edge_bitmap: Tensor
    legal_edge_n_blockers: Tensor
    legal_edge_n_attackers: Tensor
    vocab_mask: Tensor  # [B, L, V_vocab] bool
    pointer_mask: Tensor  # [B, L, max_tokens] bool


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
    decoder: DecoderGatherOutput


@dataclass
class _RingTracker:
    """Cursor/start/used tracking for one ring buffer dimension."""

    capacity: int
    start: int = 0
    used: int = 0
    cursor: int = 0

    @property
    def available(self) -> int:
        return self.capacity - self.used

    def fits(self, count: int) -> bool:
        if count == 0:
            return True
        if count > self.available:
            return False
        if self.used == 0:
            return True
        c = self.cursor % self.capacity
        return c + count <= self.capacity or count <= self.start

    def reserve(self, count: int) -> tuple[int, int]:
        c = self.cursor % self.capacity
        if count == 0:
            return c, c
        if self.used == 0 and c + count > self.capacity:
            s, e = 0, count
        elif c + count <= self.capacity:
            s, e = c, c + count
        elif count <= self.start:
            s, e = 0, count
        else:
            raise RuntimeError("ring span does not fit")
        self.cursor = e % self.capacity
        self.used += count
        return s, e

    def release(self, count: int, new_start: int | None = None) -> None:
        if count <= 0:
            return
        if new_start is not None:
            self.start = new_start % self.capacity
        self.used = max(0, self.used - count)
        if self.used == 0:
            self.start = 0
            self.cursor = 0

    def reset(self) -> None:
        self.start = 0
        self.used = 0
        self.cursor = 0


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
class _CommittedWindow:
    row_start: int
    row_end: int
    append_ready_events: tuple[Any | None, ...]
    metadata_ready_events: tuple[Any | None, ...]


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
        max_decoder_len: int = 32,
        max_anchors: int = 64,
        max_blockers: int = 16,
        max_attackers: int = 16,
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
        if max_decoder_len < 1:
            raise ValueError("max_decoder_len must be at least 1")
        if max_anchors < 0:
            raise ValueError("max_anchors must be >= 0")
        if max_blockers < 0:
            raise ValueError("max_blockers must be >= 0")
        if max_attackers < 0:
            raise ValueError("max_attackers must be >= 0")
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
        self.max_decoder_len = int(max_decoder_len)
        self.max_anchors = int(max_anchors)
        self.max_blockers = int(max_blockers)
        self.max_attackers = int(max_attackers)
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
        self.card_ref_positions = torch.full(
            (self.capacity, self.max_card_refs), -1, dtype=torch.int32, device=self.device
        )
        self.decoder = AggregateTensor(
            length=self.capacity,
            fields=(
                Field(
                    "output_token_ids",
                    torch.int32,
                    fill=int(GrammarVocab.PAD),
                    inner_shape=(self.max_decoder_len,),
                ),
                Field(
                    "output_pointer_pos",
                    torch.int32,
                    fill=-1,
                    inner_shape=(self.max_decoder_len,),
                ),
                Field(
                    "output_is_pointer",
                    torch.bool,
                    fill=False,
                    inner_shape=(self.max_decoder_len,),
                ),
                Field(
                    "output_pad_mask",
                    torch.bool,
                    fill=False,
                    inner_shape=(self.max_decoder_len,),
                ),
                Field(
                    "output_log_prob",
                    torch.float32,
                    fill=0.0,
                    inner_shape=(self.max_decoder_len,),
                ),
                Field("decision_type", torch.int32, fill=-1),
                Field(
                    "pointer_anchor_positions",
                    torch.int32,
                    fill=-1,
                    inner_shape=(self.max_anchors,),
                ),
                Field(
                    "pointer_anchor_kinds",
                    torch.int32,
                    fill=-1,
                    inner_shape=(self.max_anchors,),
                ),
                Field(
                    "pointer_anchor_subjects",
                    torch.int32,
                    fill=-1,
                    inner_shape=(self.max_anchors,),
                ),
                Field(
                    "pointer_anchor_handles",
                    torch.int32,
                    fill=-1,
                    inner_shape=(self.max_anchors,),
                ),
                Field("pointer_anchor_count", torch.int32, fill=0),
                Field(
                    "legal_edge_bitmap",
                    torch.bool,
                    fill=False,
                    inner_shape=(self.max_blockers, self.max_attackers),
                ),
                Field("legal_edge_n_blockers", torch.int32, fill=0),
                Field("legal_edge_n_attackers", torch.int32, fill=0),
                Field(
                    "vocab_mask",
                    torch.bool,
                    fill=False,
                    inner_shape=(self.max_decoder_len, GRAMMAR_VOCAB_SIZE),
                ),
                Field(
                    "pointer_mask",
                    torch.bool,
                    fill=False,
                    inner_shape=(self.max_decoder_len, self.max_tokens),
                ),
            ),
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
        self.episode_meta = AggregateTensor(
            length=self.capacity,
            fields=(
                Field("episode_id", torch.long, fill=-1),
                Field("step_idx", torch.long, fill=-1),
                Field("is_terminal", torch.bool, fill=False),
                Field("terminal_reward_p0", torch.float32, fill=0.0),
                Field("zero_sum", torch.bool, fill=False),
                Field("actor_id", torch.long, fill=-1),
                Field("behavior_policy_version", torch.long, fill=0),
                Field("inference_policy_version", torch.long, fill=0),
                Field("target_policy_version", torch.long, fill=-1),
            ),
            device=self.device,
        )
        self._reserve_lock = threading.RLock()
        self._reserve_cond = threading.Condition(self._reserve_lock)
        self._next_reservation_id = 0
        self._commit_reservation_id = 0
        self._committed_row_cursor = 0
        self._train_claim_cursor = 0
        self.row_ring = _RingTracker(capacity=self.capacity)
        self.token_ring = _RingTracker(capacity=int(self.packed_token_ids.numel()))
        self.decision_ring = _RingTracker(capacity=self.core.decision_capacity)
        self._ring_active = False
        self._committed_windows: list[_CommittedWindow] = []
        self._claimed_windows: list[tuple[int, int]] = []
        self._reservations: dict[int, _ReplayReservation] = {}
        self._unsealed_staged_reservations: dict[tuple[int, int], int] = {}
        self._row_complete_host = [False] * self.capacity
        self._row_append_ready_events: list[Any | None] = [None] * self.capacity
        self._row_ready_events: list[Any | None] = [None] * self.capacity

    # ------------------------------------------------------------------
    # Lifecycle / size accessors
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._reserve_lock:
            return int(self.row_ring.used) if self._ring_active else self.core.size

    def reset(self) -> None:
        with self._reserve_lock:
            self.core.reset()
            self.row_token_start.fill_(-1)
            self.row_token_length.zero_()
            self.row_token_length_host[:] = [0] * self.capacity
            self.decoder.reset()
            self.episode_meta.reset()
            self._next_reservation_id = 0
            self._commit_reservation_id = 0
            self._committed_row_cursor = 0
            self._train_claim_cursor = 0
            self.row_ring.reset()
            self.token_ring.reset()
            self.decision_ring.reset()
            self._ring_active = False
            self._committed_windows.clear()
            self._claimed_windows.clear()
            self._reservations.clear()
            self._unsealed_staged_reservations.clear()
            self._row_complete_host[:] = [False] * self.capacity
            self._row_append_ready_events[:] = [None] * self.capacity
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
        rows = replay_rows.to(device=self.device)
        if int(rows.numel()) == 0:
            return
        rows_host = rows.detach().cpu().tolist()
        steps = torch.arange(int(rows.numel()), dtype=torch.long, device=self.device)
        self.episode_meta.write(
            rows,
            episode_id=int(episode_id),
            step_idx=steps,
            is_terminal=False,
            terminal_reward_p0=float(terminal_reward_p0),
            zero_sum=bool(zero_sum),
            actor_id=int(actor_id),
            behavior_policy_version=int(behavior_policy_version),
            inference_policy_version=int(inference_policy_version),
            target_policy_version=int(target_policy_version),
        )
        self.episode_meta.is_terminal[rows[-1]] = True
        ready_event: Any | None = None
        if self.device.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))
        with self._reserve_cond:
            for row in rows_host:
                row_i = int(row)
                self._row_ready_events[row_i % self.capacity] = ready_event
                self._row_complete_host[row_i % self.capacity] = True
            self._publish_committable_locked()
            self._reserve_cond.notify_all()

    @property
    def committed_size(self) -> int:
        with self._reserve_lock:
            return int(self._committed_row_cursor)

    # ------------------------------------------------------------------
    # Reservations / commits / windows
    # ------------------------------------------------------------------

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

    def commit_decoder_decision(
        self,
        reservation: _ReplayReservation,
        payload: DecoderDecisionPayload,
    ) -> None:
        """Write decoder targets for the rows covered by ``reservation``.

        Sizes in ``payload`` may be smaller than the buffer's
        ``max_decoder_len`` / ``max_anchors`` / ``max_blockers`` /
        ``max_attackers``; the buffer pads with the field's fill value.
        Does not mark the reservation complete — caller still calls
        :meth:`commit` after writing tokens / decisions / decoder.
        """

        row_start = int(reservation.row_start)
        row_end = int(reservation.row_end)
        row_count = row_end - row_start
        if row_count == 0:
            return
        rows = torch.arange(row_start, row_end, dtype=torch.long, device=self.device)
        self._scatter_decoder(rows, payload)

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
            row_start, claimable_limit = self._claimable_window_prefix_locked()
            available = int(claimable_limit - row_start)
            if available < int(min_rows):
                return None
            if not allow_partial and available < target:
                return None
            row_end = row_start + min(available, int(max_rows))
            self._consume_committed_prefix_locked(row_end)
            self._claimed_windows.append((row_start, row_end))
        return TextReplayTrainWindow(
            row_start=row_start,
            row_end=row_end,
            rows=torch.arange(row_start, row_end, dtype=torch.long, device=self.device),
            ready_events=(),
        )

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
        episode_ids = self.episode_meta.episode_id[rows]
        if bool((episode_ids < 0).any().item()):
            raise ValueError("cannot build returns for incomplete replay rows")
        unique_ids, inverse = torch.unique(episode_ids, sorted=True, return_inverse=True)
        batch_size = int(unique_ids.numel())
        step_idx = self.episode_meta.step_idx[rows].to(dtype=torch.int32)
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
        terminal_reward[inverse] = self.episode_meta.terminal_reward_p0[rows].to(
            dtype=torch.float32
        )
        zero_sum[inverse] = self.episode_meta.zero_sum[rows]
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
            return self.row_ring.available

    @property
    def available_tokens(self) -> int:
        with self._reserve_lock:
            return self.token_ring.available

    def debug_snapshot(self) -> dict[str, int]:
        with self._reserve_lock:
            committed_start = -1
            committed_limit = -1
            completed_limit = -1
            claimable_limit = -1
            claimed_start = -1
            claimed_limit = -1
            if self._committed_windows:
                head = self._committed_windows[0]
                committed_start, committed_limit = head.row_start, head.row_end
                _row_start, completed_limit = self._completed_window_prefix_locked()
                _row_start, claimable_limit = self._claimable_window_prefix_locked()
            if self._claimed_windows:
                claimed_start, claimed_limit = self._claimed_windows[0]
            return {
                "committed_windows": len(self._committed_windows),
                "committed_start": int(committed_start),
                "committed_limit": int(committed_limit),
                "completed_limit": int(completed_limit),
                "claimable_limit": int(claimable_limit),
                "claimed_windows": len(self._claimed_windows),
                "claimed_start": int(claimed_start),
                "claimed_limit": int(claimed_limit),
                "row_ring_start": int(self.row_ring.start),
                "row_ring_used": int(self.row_ring.used),
                "token_ring_used": int(self.token_ring.used),
                "decision_ring_used": int(self.decision_ring.used),
                "reservations": len(self._reservations),
            }

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
        rows_host = rows.detach().cpu().tolist()
        first_row = int(rows_host[0])
        last_row = int(rows_host[-1])
        if any(
            int(row) != expected
            for row, expected in zip(rows_host, range(first_row, first_row + row_count))
        ):
            raise RuntimeError("replay rows must be released as a contiguous FIFO window")
        token_lengths = self.row_token_length[rows]
        decision_counts = self.decision_count[rows]
        token_count = int(token_lengths.sum().item())
        decision_count = int(decision_counts.sum().item())
        token_starts = self.row_token_start[rows]
        decision_starts = self.decision_start[rows]
        last_token_len = int(token_lengths[-1].item())
        last_decision_len = int(decision_counts[-1].item())
        last_token_start = int(token_starts[-1].item()) if token_count > 0 else 0
        last_decision_start = int(decision_starts[-1].item()) if decision_count > 0 else 0
        with self._reserve_cond:
            if not self._claimed_windows:
                raise RuntimeError("replay rows must be claimed before release")
            claimed_start, claimed_end = self._claimed_windows[0]
            if first_row != claimed_start or last_row + 1 != claimed_end:
                raise RuntimeError("claimed replay windows must be released in FIFO order")
            if first_row != self.row_ring.start:
                raise RuntimeError("claimed replay window must begin at the row ring head")
            self._claimed_windows.pop(0)
            # The new ring head is the oldest still-held row, not naively
            # ``last_row + 1``: a wrap-around reservation can sit "behind"
            # the just-released window, so the row-ring start needs to
            # follow whichever entry is now the oldest (next claim → next
            # committed window → oldest uncommitted reservation). Falling
            # back to ``last_row + 1`` keeps the contiguous case correct;
            # if everything has drained, ``_RingTracker.release`` resets
            # start/cursor to 0 once ``used`` hits 0.
            next_row, next_token, next_decision = self._oldest_held_starts_locked(
                fallback_row=last_row + 1,
                fallback_token=last_token_start + last_token_len if token_count > 0 else None,
                fallback_decision=(
                    last_decision_start + last_decision_len if decision_count > 0 else None
                ),
            )
            self.row_ring.release(row_count, new_start=next_row)
            self.token_ring.release(token_count, new_start=next_token)
            self.decision_ring.release(decision_count, new_start=next_decision)
            self.core._row_cursor = self.row_ring.cursor
            self.core._decision_cursor = self.decision_ring.cursor
            for row in rows_host:
                self.row_token_length_host[int(row)] = 0
                self._row_complete_host[int(row)] = False
                self._row_append_ready_events[int(row)] = None
                self._row_ready_events[int(row)] = None
            self._reserve_cond.notify_all()

    def gather_ppo_targets(self, replay_rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.core.gather_ppo_targets(replay_rows)

    # ------------------------------------------------------------------
    # Append paths (state tokens + decision groups + decoder targets)
    # ------------------------------------------------------------------

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
        seq_length = int(encoded.seq_lengths[batch_index].item())
        decision_groups = int(decision_option_idx.shape[0])
        stored_decisions = min(decision_groups, self.max_decision_groups)
        try:
            reservation = self._reserve_append(
                row_count=1,
                token_count=seq_length,
                decision_count=stored_decisions,
                block=False,
            )
        except RuntimeError as exc:
            raise RuntimeError("TextReplayBuffer is full") from exc
        row = int(reservation.row_start)
        self._write_decision_batch_at(
            row_start=row,
            row_end=row + 1,
            decision_start=reservation.decision_start,
            decision_count=torch.tensor([decision_groups], dtype=torch.long, device=self.device),
            decision_count_host=(decision_groups,),
            total_decision_groups=decision_groups,
            total_stored_decision_groups=stored_decisions,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            selected_indices=selected_indices,
            behavior_action_log_prob=behavior_action_log_prob,
        )
        self.core._write_common_row(
            row,
            trace_kind_id=int(trace_kind_id),
            may_selected=float(may_selected),
            old_log_prob=float(old_log_prob),
            value=float(value),
            perspective_player_idx=int(perspective_player_idx),
            lstm_h_in=lstm_h_in,
            lstm_c_in=lstm_c_in,
        )
        self._write_encoded_row(row, reservation.token_start, encoded, batch_index)
        self._mark_append_ready_range(row, row + 1)
        self._seal_reservation(reservation)
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
        if encoded.seq_lengths_host is None:
            raise ValueError("packed replay append requires seq_lengths_host")
        start = sum(int(n) for n in encoded.seq_lengths_host[:batch_index])
        end = start + int(encoded.seq_lengths_host[batch_index])
        token_width = end - start
        decision_groups = int(decision_option_idx.shape[0])
        stored_decisions = min(decision_groups, self.max_decision_groups)
        try:
            reservation = self._reserve_append(
                row_count=1,
                token_count=token_width,
                decision_count=stored_decisions,
                block=False,
            )
        except RuntimeError as exc:
            raise RuntimeError("TextReplayBuffer is full") from exc
        row = int(reservation.row_start)
        self._write_decision_batch_at(
            row_start=row,
            row_end=row + 1,
            decision_start=reservation.decision_start,
            decision_count=torch.tensor([decision_groups], dtype=torch.long, device=self.device),
            decision_count_host=(decision_groups,),
            total_decision_groups=decision_groups,
            total_stored_decision_groups=stored_decisions,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            selected_indices=selected_indices,
            behavior_action_log_prob=behavior_action_log_prob,
        )
        self.core._write_common_row(
            row,
            trace_kind_id=int(trace_kind_id),
            may_selected=float(may_selected),
            old_log_prob=float(old_log_prob),
            value=float(value),
            perspective_player_idx=int(perspective_player_idx),
            lstm_h_in=lstm_h_in,
            lstm_c_in=lstm_c_in,
        )
        self._write_packed_encoded_row(row, reservation.token_start, encoded, batch_index)
        self._mark_append_ready_range(row, row + 1)
        self._seal_reservation(reservation)
        return row

    def _packed_seq_lengths_host(self, encoded: PackedTextBatch) -> list[int]:
        batch_size = int(encoded.seq_lengths.shape[0])
        if encoded.seq_lengths_host is not None:
            lengths = list(encoded.seq_lengths_host)
            if len(lengths) != batch_size:
                raise ValueError("encoded.seq_lengths_host length must match batch size")
            return [int(x) for x in lengths]
        raise ValueError("packed replay append requires seq_lengths_host")

    def _reserve_append(
        self,
        *,
        row_count: int,
        token_count: int,
        decision_count: int,
        block: bool = True,
    ) -> _ReplayReservation:
        row_count = int(row_count)
        token_count = int(token_count)
        decision_count = int(decision_count)
        if row_count > self.row_ring.capacity:
            raise RuntimeError("TextReplayBuffer row request exceeds capacity")
        if token_count > self.token_ring.capacity:
            raise RuntimeError("TextReplayBuffer token request exceeds capacity")
        if decision_count > self.decision_ring.capacity:
            raise RuntimeError("TextReplayBuffer decision request exceeds capacity")
        with self._reserve_cond:
            if not block and not self._can_reserve_locked(
                row_count=row_count,
                token_count=token_count,
                decision_count=decision_count,
            ):
                raise RuntimeError("TextReplayBuffer is full")
            while not self._can_reserve_locked(
                row_count=row_count,
                token_count=token_count,
                decision_count=decision_count,
            ):
                self._reserve_cond.wait()
            row_start, row_end = self.row_ring.reserve(row_count)
            token_start, token_end = self.token_ring.reserve(token_count)
            decision_start, decision_end = self.decision_ring.reserve(decision_count)
            self.core._row_cursor = self.row_ring.cursor
            self.core._decision_cursor = self.decision_ring.cursor
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

    def _reservation_rows_complete_locked(self, reservation: _ReplayReservation) -> bool:
        return all(
            self._row_complete_host[row % self.capacity]
            for row in range(int(reservation.row_start), int(reservation.row_end))
        )

    def _publish_committable_locked(self) -> None:
        while True:
            head = self._reservations.get(self._commit_reservation_id)
            if (
                head is None
                or not head.complete
                or not self._reservation_rows_complete_locked(head)
            ):
                break
            self._committed_row_cursor = head.row_end
            append_events = tuple(
                self._row_append_ready_events[row % self.capacity]
                for row in range(int(head.row_start), int(head.row_end))
            )
            metadata_events = tuple(
                self._row_ready_events[row % self.capacity]
                for row in range(int(head.row_start), int(head.row_end))
            )
            self._committed_windows.append(
                _CommittedWindow(
                    row_start=head.row_start,
                    row_end=head.row_end,
                    append_ready_events=append_events,
                    metadata_ready_events=metadata_events,
                )
            )
            del self._reservations[self._commit_reservation_id]
            self._commit_reservation_id += 1

    def _seal_reservation(self, reservation: _ReplayReservation) -> None:
        with self._reserve_cond:
            stored = self._reservations[reservation.reservation_id]
            stored.complete = True
            self._publish_committable_locked()
            self._reserve_cond.notify_all()

    def _mark_append_ready_range(self, row_start: int, row_end: int) -> None:
        ready_event: Any | None = None
        if self.device.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))
        with self._reserve_cond:
            for row in range(int(row_start), int(row_end)):
                self._row_append_ready_events[row % self.capacity] = ready_event
            self._reserve_cond.notify_all()

    def seal_staged_rows(self, rows: Tensor) -> None:
        if int(rows.numel()) == 0:
            return
        rows_host = rows.detach().cpu().tolist()
        row_start = int(rows_host[0])
        row_end = row_start + len(rows_host)
        if any(int(row) != expected for row, expected in zip(rows_host, range(row_start, row_end))):
            raise RuntimeError("staged replay rows must be sealed as one contiguous reservation")
        key = (row_start, row_end)
        with self._reserve_cond:
            reservation_id = self._unsealed_staged_reservations.pop(key)
            reservation = self._reservations[reservation_id]
        self._seal_reservation(reservation)

    def _can_reserve_locked(
        self,
        *,
        row_count: int,
        token_count: int,
        decision_count: int,
    ) -> bool:
        return (
            self.row_ring.fits(row_count)
            and self.token_ring.fits(token_count)
            and self.decision_ring.fits(decision_count)
        )

    def _completed_prefix_end_locked(self, row_start: int, row_limit: int) -> int:
        del row_start
        return int(row_limit)

    @staticmethod
    def _ready_event_complete(event: Any | None) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        if query is None:
            return True
        return bool(query())

    def _claimable_prefix_end_for_window_locked(
        self,
        window: _CommittedWindow,
        row_start: int,
    ) -> int:
        offset = int(row_start) - int(window.row_start)
        row = int(row_start)
        while row < int(window.row_end):
            if not self._ready_event_complete(window.append_ready_events[offset]):
                break
            if not self._ready_event_complete(window.metadata_ready_events[offset]):
                break
            row += 1
            offset += 1
        return row

    def _window_prefix_locked(self, end_finder: Callable[[int, int], int]) -> tuple[int, int]:
        if not self._committed_windows:
            return 0, 0
        head = self._committed_windows[0]
        row_start, row_limit = head.row_start, head.row_end
        limit = end_finder(row_start, row_limit)
        idx = 1
        while limit == row_limit and idx < len(self._committed_windows):
            next_window = self._committed_windows[idx]
            next_start, next_limit = next_window.row_start, next_window.row_end
            if int(next_start) != int(limit):
                break
            row_limit = next_limit
            limit = end_finder(next_start, next_limit)
            idx += 1
        return int(row_start), int(limit)

    def _completed_window_prefix_locked(self) -> tuple[int, int]:
        return self._window_prefix_locked(self._completed_prefix_end_locked)

    def _claimable_window_prefix_locked(self) -> tuple[int, int]:
        if not self._committed_windows:
            return 0, 0
        head = self._committed_windows[0]
        row_start = head.row_start
        row_limit = head.row_end
        limit = self._claimable_prefix_end_for_window_locked(head, row_start)
        idx = 1
        while limit == row_limit and idx < len(self._committed_windows):
            next_window = self._committed_windows[idx]
            if int(next_window.row_start) != int(limit):
                break
            row_limit = next_window.row_end
            limit = self._claimable_prefix_end_for_window_locked(next_window, next_window.row_start)
            idx += 1
        return int(row_start), int(limit)

    def _oldest_held_starts_locked(
        self,
        *,
        fallback_row: int,
        fallback_token: int | None,
        fallback_decision: int | None,
    ) -> tuple[int, int | None, int | None]:
        """Return (row_start, token_start, decision_start) of the oldest still-held
        reservation/window after a release, or the fallbacks if nothing is still
        held (in which case ``_RingTracker.release`` will reset start/cursor to 0
        once ``used`` hits zero).

        "Held" means anything that has reserved ring space but isn't yet
        released: the next pending claim, then the next committed window,
        then the oldest uncommitted reservation. The fallbacks reproduce
        the historical "advance to ``last_row + 1``" behavior, which is
        only correct in the contiguous (non-wrap) case.
        """
        if self._claimed_windows:
            first_row = int(self._claimed_windows[0][0])
        elif self._committed_windows:
            first_row = int(self._committed_windows[0].row_start)
        else:
            head = self._reservations.get(self._commit_reservation_id)
            if head is None:
                return fallback_row, fallback_token, fallback_decision
            return (
                int(head.row_start),
                int(head.token_start),
                int(head.decision_start),
            )
        row_idx = first_row % self.capacity
        token_start = int(self.row_token_start[row_idx].item())
        decision_start = int(self.decision_start[row_idx].item())
        return first_row, token_start, decision_start

    def _consume_committed_prefix_locked(self, row_end: int) -> None:
        row_end = int(row_end)
        if self._committed_windows:
            start = int(self._committed_windows[0].row_start)
            for row in range(start, row_end):
                self._row_append_ready_events[row % self.capacity] = None
                self._row_ready_events[row % self.capacity] = None
        while self._committed_windows and self._committed_windows[0].row_end <= row_end:
            self._committed_windows.pop(0)
        if self._committed_windows and self._committed_windows[0].row_start < row_end:
            window = self._committed_windows[0]
            offset = int(row_end) - int(window.row_start)
            self._committed_windows[0] = _CommittedWindow(
                row_start=row_end,
                row_end=window.row_end,
                append_ready_events=window.append_ready_events[offset:],
                metadata_ready_events=window.metadata_ready_events[offset:],
            )

    def _split_recurrent(
        self,
        lstm_h_in: Tensor | None,
        lstm_c_in: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.lstm_h_in is not None and self.lstm_c_in is not None:
            if lstm_h_in is None or lstm_c_in is None:
                raise ValueError("h_in and c_in are required for recurrent text replay")
            return lstm_h_in.permute(1, 0, 2), lstm_c_in.permute(1, 0, 2)
        if lstm_h_in is not None or lstm_c_in is not None:
            raise ValueError("buffer was constructed without recurrent state storage")
        return None, None

    def _compute_total_stored_decisions(
        self,
        decision_count: Tensor,
        decision_count_host: tuple[int, ...] | None,
        total_stored_decision_groups: int | None,
    ) -> int:
        if total_stored_decision_groups is not None:
            return int(total_stored_decision_groups)
        if decision_count_host is not None:
            return sum(min(int(c), self.max_decision_groups) for c in decision_count_host)
        return int(
            decision_count.to(device=self.device).clamp(max=self.max_decision_groups).sum().item()
        )

    def _write_common_metadata(
        self,
        rows: Tensor,
        *,
        trace_kind_id: Tensor,
        may_selected: Tensor,
        old_log_prob: Tensor,
        value: Tensor,
        perspective_player_idx: Tensor,
        h_store: Tensor | None,
        c_store: Tensor | None,
    ) -> None:
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
            assert h_store is not None and c_store is not None
            self.lstm_h_in[rows] = h_store.to(device=self.device, dtype=torch.float32)
            self.lstm_c_in[rows] = c_store.to(device=self.device, dtype=torch.float32)

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
        seq_lengths_host = self._packed_seq_lengths_host(encoded)
        seq_lengths = encoded.seq_lengths.to(device=self.device)
        if self.validate:
            max_seq_length = max(seq_lengths_host, default=0)
            if max_seq_length > self.max_tokens:
                raise ValueError("encoded packed row token width exceeds buffer max_tokens")
        total_tokens = int(encoded.token_ids.numel())
        if decision_count_host is not None and len(decision_count_host) != batch_size:
            raise ValueError("decision_count_host length must match batch size")
        total_stored = self._compute_total_stored_decisions(
            decision_count, decision_count_host, total_stored_decision_groups
        )
        h_store, c_store = self._split_recurrent(lstm_h_in, lstm_c_in)

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
        self._write_common_metadata(
            rows,
            trace_kind_id=trace_kind_id,
            may_selected=may_selected,
            old_log_prob=old_log_prob,
            value=value,
            perspective_player_idx=perspective_player_idx,
            h_store=h_store,
            c_store=c_store,
        )
        self.row_token_length_host[row_start : row_start + batch_size] = seq_lengths_host
        seq_lengths_i32 = seq_lengths.to(dtype=torch.int32)
        self.card_ref_positions.index_fill_(0, rows, -1)
        self.seq_lengths[rows] = seq_lengths_i32
        self._clear_decoder_rows(rows)
        token_ids = encoded.token_ids.to(device=self.device, dtype=torch.int32)
        if total_tokens > 0:
            self.packed_token_ids[token_start:token_end] = token_ids
        self.row_token_start[rows] = token_start + encoded.cu_seqlens[:-1].to(
            device=self.device, dtype=torch.int32
        )
        self.row_token_length[rows] = seq_lengths_i32

        # card_ref_positions are stored row-local both in the live
        # PackedTextBatch and in the replay buffer; copy through.
        self.card_ref_positions[rows] = encoded.card_ref_positions.to(
            device=self.device, dtype=self.card_ref_positions.dtype
        )

        self._mark_append_ready_range(row_start, row_end)
        self._seal_reservation(reservation)
        return rows

    def append_native_payload(self, payload: Any, ready_event: Any | None = None) -> Tensor:
        """Append a sampled native text replay payload directly into replay."""

        if ready_event is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_event(ready_event)
        rows = self.append_batch(
            encoded=payload.encoded,
            trace_kind_id=payload.trace_kind_id,
            decision_count=payload.decision_count,
            decision_count_host=payload.decision_count_host,
            total_decision_groups=payload.total_decision_groups,
            total_stored_decision_groups=payload.total_stored_decision_groups,
            decision_option_idx=payload.decision_option_idx,
            decision_target_idx=payload.decision_target_idx,
            decision_mask=payload.decision_mask,
            uses_none_head=payload.uses_none_head,
            selected_indices=payload.selected_indices,
            behavior_action_log_prob=payload.behavior_action_log_prob,
            may_selected=payload.may_selected,
            old_log_prob=payload.old_log_prob,
            value=payload.value,
            perspective_player_idx=payload.perspective_player_idx,
            lstm_h_in=payload.lstm_h_in,
            lstm_c_in=payload.lstm_c_in,
        )
        if self.projected_state is not None and payload.projected_state is not None:
            self.write_projected_state(rows, payload.projected_state)
        return rows

    def _write_row_packed_tokens(self, row: int, token_start: int, token_ids: Tensor) -> None:
        token_count = int(token_ids.numel())
        token_end = token_start + token_count
        if token_end > self.token_ring.capacity:
            raise RuntimeError("TextReplayBuffer packed token arena is full")
        self.row_token_start[row] = token_start
        self.row_token_length[row] = token_count
        self.row_token_length_host[row] = token_count
        if token_count > 0:
            self.packed_token_ids[token_start:token_end] = token_ids

    # ------------------------------------------------------------------
    # Decoder targets
    # ------------------------------------------------------------------

    def _clear_decoder_rows(self, rows: Tensor) -> None:
        self.decoder.output_token_ids.index_fill_(0, rows, int(GrammarVocab.PAD))
        self.decoder.output_pointer_pos.index_fill_(0, rows, -1)
        self.decoder.output_is_pointer.index_fill_(0, rows, 0)
        self.decoder.output_pad_mask.index_fill_(0, rows, 0)
        self.decoder.output_log_prob.index_fill_(0, rows, 0.0)
        self.decoder.decision_type.index_fill_(0, rows, -1)
        self.decoder.pointer_anchor_positions.index_fill_(0, rows, -1)
        self.decoder.pointer_anchor_kinds.index_fill_(0, rows, -1)
        self.decoder.pointer_anchor_subjects.index_fill_(0, rows, -1)
        self.decoder.pointer_anchor_handles.index_fill_(0, rows, -1)
        self.decoder.pointer_anchor_count.index_fill_(0, rows, 0)
        self.decoder.legal_edge_bitmap.index_fill_(0, rows, 0)
        self.decoder.legal_edge_n_blockers.index_fill_(0, rows, 0)
        self.decoder.legal_edge_n_attackers.index_fill_(0, rows, 0)
        self.decoder.vocab_mask.index_fill_(0, rows, 0)
        self.decoder.pointer_mask.index_fill_(0, rows, 0)

    def _scatter_decoder(self, rows: Tensor, payload: DecoderDecisionPayload) -> None:
        b = int(rows.shape[0])
        if int(payload.output_token_ids.shape[0]) != b:
            raise ValueError("decoder payload batch size does not match reservation row count")

        self._clear_decoder_rows(rows)

        l_payload = int(payload.output_token_ids.shape[1])
        if l_payload > self.max_decoder_len:
            raise ValueError(
                f"decoder payload length {l_payload} exceeds max_decoder_len {self.max_decoder_len}"
            )
        n_payload = int(payload.pointer_anchor_positions.shape[1])
        if n_payload > self.max_anchors:
            raise ValueError(
                f"pointer anchor count {n_payload} exceeds max_anchors {self.max_anchors}"
            )
        n_block = int(payload.legal_edge_bitmap.shape[1])
        n_attk = int(payload.legal_edge_bitmap.shape[2])
        if n_block > self.max_blockers:
            raise ValueError(
                f"legal_edge n_blockers {n_block} exceeds max_blockers {self.max_blockers}"
            )
        if n_attk > self.max_attackers:
            raise ValueError(
                f"legal_edge n_attackers {n_attk} exceeds max_attackers {self.max_attackers}"
            )

        dev = self.device
        if l_payload > 0:
            row_view = rows.view(-1, 1).expand(b, l_payload)
            col = torch.arange(l_payload, device=dev).view(1, -1).expand(b, l_payload)
            self.decoder.output_token_ids[row_view, col] = payload.output_token_ids.to(
                device=dev, dtype=torch.int32
            )
            self.decoder.output_pointer_pos[row_view, col] = payload.output_pointer_pos.to(
                device=dev, dtype=torch.int32
            )
            self.decoder.output_is_pointer[row_view, col] = payload.output_is_pointer.to(
                device=dev, dtype=torch.bool
            )
            self.decoder.output_pad_mask[row_view, col] = payload.output_pad_mask.to(
                device=dev, dtype=torch.bool
            )
            self.decoder.output_log_prob[row_view, col] = payload.output_log_prob.to(
                device=dev, dtype=torch.float32
            )

        self.decoder.decision_type[rows] = payload.decision_type.to(device=dev, dtype=torch.int32)

        if n_payload > 0:
            row_view = rows.view(-1, 1).expand(b, n_payload)
            col = torch.arange(n_payload, device=dev).view(1, -1).expand(b, n_payload)
            self.decoder.pointer_anchor_positions[row_view, col] = (
                payload.pointer_anchor_positions.to(device=dev, dtype=torch.int32)
            )
            self.decoder.pointer_anchor_kinds[row_view, col] = payload.pointer_anchor_kinds.to(
                device=dev, dtype=torch.int32
            )
            self.decoder.pointer_anchor_subjects[row_view, col] = (
                payload.pointer_anchor_subjects.to(device=dev, dtype=torch.int32)
            )
            self.decoder.pointer_anchor_handles[row_view, col] = payload.pointer_anchor_handles.to(
                device=dev, dtype=torch.int32
            )
        self.decoder.pointer_anchor_count[rows] = payload.pointer_anchor_count.to(
            device=dev, dtype=torch.int32
        )

        if n_block > 0 and n_attk > 0:
            self.decoder.legal_edge_bitmap[rows, :n_block, :n_attk] = payload.legal_edge_bitmap.to(
                device=dev, dtype=torch.bool
            )
        self.decoder.legal_edge_n_blockers[rows] = payload.legal_edge_n_blockers.to(
            device=dev, dtype=torch.int32
        )
        self.decoder.legal_edge_n_attackers[rows] = payload.legal_edge_n_attackers.to(
            device=dev, dtype=torch.int32
        )

        # Per-step grammar masks captured by the live sampler. ``vocab_mask``
        # is fixed-width along the vocab axis; ``pointer_mask``'s column
        # axis is ``T_enc`` at sample time and can be < buffer.max_tokens
        # (encoder padding for that batch). Pad with False; replay-time
        # readers truncate to the current ``T_enc``.
        v_w = int(payload.vocab_mask.shape[2])
        p_w = int(payload.pointer_mask.shape[2])
        if v_w > GRAMMAR_VOCAB_SIZE:
            raise ValueError(
                f"vocab_mask width {v_w} exceeds GRAMMAR_VOCAB_SIZE {GRAMMAR_VOCAB_SIZE}"
            )
        if p_w > self.max_tokens:
            raise ValueError(
                f"pointer_mask T_enc {p_w} exceeds buffer max_tokens {self.max_tokens}"
            )
        if l_payload > 0 and v_w > 0:
            self.decoder.vocab_mask[rows, :l_payload, :v_w] = payload.vocab_mask.to(
                device=dev, dtype=torch.bool
            )
        if l_payload > 0 and p_w > 0:
            self.decoder.pointer_mask[rows, :l_payload, :p_w] = payload.pointer_mask.to(
                device=dev, dtype=torch.bool
            )

    def gather_decoder(self, indices: Tensor) -> DecoderGatherOutput:
        idx = indices.to(device=self.device, dtype=torch.long)
        return DecoderGatherOutput(
            output_token_ids=self.decoder.output_token_ids[idx],
            output_pointer_pos=self.decoder.output_pointer_pos[idx],
            output_is_pointer=self.decoder.output_is_pointer[idx],
            output_pad_mask=self.decoder.output_pad_mask[idx],
            output_log_prob=self.decoder.output_log_prob[idx],
            decision_type=self.decoder.decision_type[idx],
            pointer_anchor_positions=self.decoder.pointer_anchor_positions[idx],
            pointer_anchor_kinds=self.decoder.pointer_anchor_kinds[idx],
            pointer_anchor_subjects=self.decoder.pointer_anchor_subjects[idx],
            pointer_anchor_handles=self.decoder.pointer_anchor_handles[idx],
            pointer_anchor_count=self.decoder.pointer_anchor_count[idx],
            legal_edge_bitmap=self.decoder.legal_edge_bitmap[idx],
            legal_edge_n_blockers=self.decoder.legal_edge_n_blockers[idx],
            legal_edge_n_attackers=self.decoder.legal_edge_n_attackers[idx],
            vocab_mask=self.decoder.vocab_mask[idx],
            pointer_mask=self.decoder.pointer_mask[idx],
        )

    # ------------------------------------------------------------------
    # Gather (full training batch)
    # ------------------------------------------------------------------

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
        if idx_host is None:
            raise ValueError("replay gather requires host replay row ids")
        if self.validate:
            bad_host = next((row for row in idx_host if self.row_token_length_host[row] <= 0), None)
            if bad_host is not None:
                raise ValueError(f"replay row {bad_host} is not occupied")
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
        max_seqlen = max((self.row_token_length_host[row] for row in idx_host), default=0)
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

        common = self.core.gather_common(idx)
        if self.use_triton_gather:
            decisions = gather_decisions_triton(
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
        else:
            decisions = None
        if decisions is None:
            decisions = self.core.gather_decisions(idx)
            behavior_action_log_prob = decisions.behavior_action_log_prob
        else:
            behavior_action_log_prob = self.core.gather_decision_behavior_action_log_prob(idx)
        # PackedTextBatch's spec / pointer fields are not stored in V2 replay
        # — actor_critic rebuilds them at gather time from the decoder fields.
        b = int(idx.shape[0])
        empty_i32 = torch.zeros(b, dtype=torch.int32, device=self.device)
        empty_dec_type = torch.full((b,), -1, dtype=torch.int32, device=self.device)
        empty_anchor = torch.full((b, 0), -1, dtype=torch.int32, device=self.device)
        encoded = PackedTextBatch(
            token_ids=token_ids,
            seq_id=seq_id,
            pos_in_seq=pos_in_seq,
            cu_seqlens=cu_seqlens,
            seq_lengths=seq_lengths,
            state_positions=state_positions,
            card_ref_positions=self.card_ref_positions[idx],
            spec_lens=empty_i32,
            decision_type=empty_dec_type,
            pointer_anchor_positions=empty_anchor,
            pointer_anchor_kinds=empty_anchor,
            pointer_anchor_subjects=empty_anchor,
            pointer_anchor_handles=empty_anchor,
            legal_edge_bitmap=None,
            total_tokens=int(token_ids.numel()) if total_tokens_host is None else total_tokens_host,
            seq_lengths_host=tuple(self.row_token_length_host[row] for row in idx_host),
            max_seqlen=max_seqlen,
        )
        decoder = self.gather_decoder(idx)
        return TextReplayBatch(
            encoded=encoded,
            trace_kind_id=common.trace_kind_id,
            decision_start=decisions.decision_start,
            decision_count=decisions.decision_count,
            decision_option_idx=decisions.decision_option_idx,
            decision_target_idx=decisions.decision_target_idx,
            decision_mask=decisions.decision_mask,
            uses_none_head=decisions.uses_none_head,
            selected_indices=decisions.selected_indices,
            behavior_action_log_prob=behavior_action_log_prob,
            step_for_decision_group=decisions.step_for_group,
            may_selected=common.may_selected,
            old_log_prob=common.old_log_prob,
            value=common.value,
            perspective_player_idx=common.perspective_player_idx,
            lstm_h_in=common.lstm_h_in,
            lstm_c_in=common.lstm_c_in,
            decoder=decoder,
        )

    def write_projected_state(self, rows: Tensor, projected: Tensor) -> None:
        if self.projected_state is None:
            raise ValueError("buffer was constructed without projected_state storage")
        self.projected_state[rows.to(device=self.device)] = projected.to(
            device=self.device, dtype=torch.bfloat16
        )

    def _write_encoded_row(
        self,
        row: int,
        token_start: int,
        encoded: TextEncodedBatch,
        batch_index: int,
    ) -> None:
        self._validate_batch_index(encoded, batch_index)
        seq_length = int(encoded.seq_lengths[batch_index].item())
        self.card_ref_positions[row].fill_(-1)
        self._clear_decoder_rows(torch.tensor([row], dtype=torch.long, device=self.device))
        self._write_row_packed_tokens(
            row,
            token_start,
            encoded.token_ids[batch_index, :seq_length].to(device=self.device, dtype=torch.int32),
        )
        self.card_ref_positions[row].copy_(
            encoded.card_ref_positions[batch_index].to(device=self.device, dtype=torch.int32)
        )
        self.seq_lengths[row] = encoded.seq_lengths[batch_index]

    def _write_packed_encoded_row(
        self,
        row: int,
        token_start: int,
        encoded: PackedTextBatch,
        batch_index: int,
    ) -> None:
        self._validate_packed_batch_index(encoded, batch_index)
        # Avoid syncing on cu_seqlens for every staged row at finish-time.
        # seq_lengths_host is required by both call paths and is already host data.
        if encoded.seq_lengths_host is None:
            raise ValueError("packed replay write requires seq_lengths_host")
        start = sum(int(n) for n in encoded.seq_lengths_host[:batch_index])
        end = start + int(encoded.seq_lengths_host[batch_index])
        token_width = end - start
        if token_width > self.max_tokens:
            raise ValueError("encoded packed row token width exceeds buffer max_tokens")

        self.card_ref_positions[row].fill_(-1)
        self._clear_decoder_rows(torch.tensor([row], dtype=torch.long, device=self.device))

        self._write_row_packed_tokens(
            row,
            token_start,
            encoded.token_ids[start:end].to(device=self.device, dtype=torch.int32),
        )

        # card_ref_positions are row-local end-to-end; just slice + copy.
        self.card_ref_positions[row].copy_(
            encoded.card_ref_positions[batch_index].to(device=self.device, dtype=torch.int32)
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

    def _validate_packed_batch_index(self, encoded: PackedTextBatch, batch_index: int) -> None:
        batch_size = int(encoded.seq_lengths.shape[0])
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError("batch_index out of range")
        if encoded.card_ref_positions.shape[1] != self.max_card_refs:
            raise ValueError("encoded card_ref_positions width does not match buffer")


__all__ = [
    "DecoderDecisionPayload",
    "DecoderGatherOutput",
    "TextReplayBatch",
    "TextReplayBuffer",
    "TextReplayTrainWindow",
    "_REPLAY_FORMAT_VERSION",
]
