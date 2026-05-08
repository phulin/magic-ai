"""IMPALA-style inference server for the text-encoder rollout path.

A single background thread owns the GPU policy: it drains encoded-batch
requests submitted by N actor threads, dynamically batches them into one
forward pass, calls ``TextActorCritic.sample_native_tensor_batch`` on the
merged batch, and scatters per-request replay payload slices plus host-side
scalars (decision counts, selected choices, may-bit, log-prob, value,
trace-kind) back to the actors via per-request ``Future`` objects.

All concat / scatter operations are vectorized — no Python ``for`` loops over
tensor rows. The server is the only thread that touches the policy's
parameters or its live LSTM env-state cache during normal rollout, so no
GPU lock is required. ``pause()`` / ``resume()`` quiesce the server for the
PPO/R-NaD update window or for snapshot eval.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, fields
from queue import Empty
from typing import Any, cast

import torch

from magic_ai.slot_encoder.native_encoder import NativeEncodedBatch
from magic_ai.text_encoder.actor_critic import NativeTextReplayPayload
from magic_ai.text_encoder.batch import PackedTextBatch

_PROFILE_SAMPLE_COMPONENTS = os.environ.get("MAGIC_AI_PROFILE_SAMPLE_COMPONENTS") == "1"


def _infer_policy_device(policy: Any) -> torch.device:
    device = getattr(policy, "device", None)
    if device is not None:
        return torch.device(device)
    parameters = getattr(policy, "parameters", None)
    if parameters is not None:
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _arena_empty(*shape: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


class _HeadItemDoesNotFit(Exception):
    pass


@dataclass(frozen=True)
class TextInferenceRequest:
    """One actor's encoded batch + per-row routing metadata."""

    native_batch: Any  # NativeEncodedBatch
    packed_batch: PackedTextBatch
    env_indices: list[int]
    perspective_player_indices: list[int]


@dataclass(frozen=True)
class TextInferenceReply:
    """Host-side per-row scalars an actor needs for stepping + transcripts."""

    decision_counts: list[int]
    selected_choice_cols: list[int]  # flat across the request's rows
    may_selected: list[int]
    old_log_prob: list[float]
    value: list[float]
    trace_kind_id: list[int]
    replay_payload: NativeTextReplayPayload
    replay_rows: list[int] | None = None
    inference_policy_version: int = 0
    ready_event: Any | None = None
    release_item: Any | None = None


class RolloutTimingStats:
    """Thread-safe low-overhead timing accumulator for actor rollouts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._rows = 0
        self._tokens = 0
        self._max_seq = 0

    def add(self, name: str, elapsed: float) -> None:
        with self._lock:
            self._totals[name] = self._totals.get(name, 0.0) + float(elapsed)
            self._counts[name] = self._counts.get(name, 0) + 1

    def add_batch(self, *, rows: int, tokens: int, max_seq: int) -> None:
        with self._lock:
            self._rows += int(rows)
            self._tokens += int(tokens)
            self._max_seq = max(self._max_seq, int(max_seq))

    def snapshot_and_reset(self) -> dict[str, float | int]:
        with self._lock:
            totals = dict(self._totals)
            counts = dict(self._counts)
            rows = self._rows
            tokens = self._tokens
            max_seq = self._max_seq
            self._totals.clear()
            self._counts.clear()
            self._rows = 0
            self._tokens = 0
            self._max_seq = 0
        out: dict[str, float | int] = {
            "rows": rows,
            "tokens": tokens,
            "max_seq": max_seq,
            "avg_tokens_per_row": (tokens / rows) if rows else 0.0,
        }
        for name, total in totals.items():
            count = max(1, counts.get(name, 0))
            out[f"{name}_total_s"] = total
            out[f"{name}_mean_ms"] = 1000.0 * total / count
            out[f"{name}_count"] = count
        return out


def _concat_packed_text_batches(batches: list[PackedTextBatch]) -> PackedTextBatch:
    """Concatenate ``PackedTextBatch`` objects along the row axis.

    Vectorized: every per-shard offset (token, batch, anchor) is rolled into
    one tensor op. ``-1`` sentinels in anchor positions are preserved.
    """

    for b in batches:
        token_count = int(b.token_ids.shape[0])
        if int(b.seq_id.shape[0]) != token_count or int(b.pos_in_seq.shape[0]) != token_count:
            raise ValueError(
                "packed batch token metadata length must match token_ids length "
                f"(tokens={token_count}, seq_id={int(b.seq_id.shape[0])}, "
                f"pos_in_seq={int(b.pos_in_seq.shape[0])})"
            )

    if len(batches) == 1:
        return batches[0]

    seq_lengths = torch.cat([b.seq_lengths for b in batches], dim=0)
    token_ids = torch.cat([b.token_ids for b in batches], dim=0)
    seq_lengths_host: tuple[int, ...] | None = None
    if all(b.seq_lengths_host is not None for b in batches):
        seq_lengths_host = tuple(
            n for b in batches for n in cast(tuple[int, ...], b.seq_lengths_host)
        )

    # Per-batch token offset (cumulative live-token count of preceding shards).
    token_totals = torch.tensor(
        [int(b.token_ids.shape[0]) for b in batches],
        dtype=torch.int32,
        device=seq_lengths.device,
    )
    token_offsets = torch.zeros_like(token_totals)
    token_offsets[1:] = token_totals.cumsum(0)[:-1]

    # cu_seqlens: drop each shard's leading 0, shift by token offset, prepend 0.
    parts: list[torch.Tensor] = [seq_lengths.new_zeros(1, dtype=torch.int32)]
    for b, off in zip(batches, token_offsets.tolist(), strict=True):
        parts.append(b.cu_seqlens[1:].to(torch.int32) + int(off))
    cu_seqlens = torch.cat(parts, dim=0)

    seq_id_parts: list[torch.Tensor] = []
    row_offset = 0
    for b in batches:
        seq_id_parts.append(b.seq_id.to(torch.int32) + int(row_offset))
        row_offset += int(b.seq_lengths.shape[0])
    seq_id = torch.cat(seq_id_parts, dim=0)
    pos_in_seq = torch.cat([b.pos_in_seq.to(torch.int32) for b in batches], dim=0)

    state_parts: list[torch.Tensor] = []
    for b, off in zip(batches, token_offsets.tolist(), strict=True):
        state_parts.append(b.state_positions.to(torch.int32) + int(off))
    state_positions = torch.cat(state_parts, dim=0)

    def _shift_anchor(name: str) -> torch.Tensor:
        out_parts: list[torch.Tensor] = []
        for b, off in zip(batches, token_offsets.tolist(), strict=True):
            t = cast(torch.Tensor, getattr(b, name)).to(torch.int32)
            valid = t >= 0
            out_parts.append(torch.where(valid, t + int(off), t))
        return torch.cat(out_parts, dim=0)

    max_blanks = max((int(b.blank_positions.shape[1]) for b in batches), default=0)
    max_legal = max((int(b.blank_legal_ids.shape[2]) for b in batches), default=0)

    def _pad_blank_2d(name: str, *, fill: int = 0, shift: bool = False) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for b, off in zip(batches, token_offsets.tolist(), strict=True):
            t = cast(torch.Tensor, getattr(b, name)).to(torch.int32)
            rows = int(t.shape[0])
            out = torch.full((rows, max_blanks), fill, dtype=torch.int32, device=t.device)
            cols = int(t.shape[1])
            if cols > 0:
                src = t[:, :cols]
                if shift:
                    valid = src >= 0
                    src = torch.where(valid, src + int(off), src)
                out[:, :cols] = src
            parts.append(out)
        return torch.cat(parts, dim=0)

    def _pad_blank_legal(name: str, *, dtype: torch.dtype) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for b in batches:
            t = cast(torch.Tensor, getattr(b, name)).to(dtype)
            rows = int(t.shape[0])
            out = torch.zeros((rows, max_blanks, max_legal), dtype=dtype, device=t.device)
            k = int(t.shape[1])
            v = int(t.shape[2])
            if k > 0 and v > 0:
                out[:, :k, :v] = t
            parts.append(out)
        return torch.cat(parts, dim=0)

    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu_seqlens,
        seq_lengths=seq_lengths,
        state_positions=state_positions,
        card_ref_positions=_shift_anchor("card_ref_positions"),
        total_tokens=int(token_ids.shape[0]),
        seq_lengths_host=seq_lengths_host,
        blank_positions=_pad_blank_2d("blank_positions", fill=-1, shift=True),
        blank_kind=_pad_blank_2d("blank_kind"),
        blank_group=_pad_blank_2d("blank_group", fill=-1),
        blank_group_kind=_pad_blank_2d("blank_group_kind"),
        blank_option_index=_pad_blank_2d("blank_option_index", fill=-1),
        blank_legal_ids=_pad_blank_legal("blank_legal_ids", dtype=torch.int32),
        blank_legal_mask=_pad_blank_legal("blank_legal_mask", dtype=torch.bool),
    )


def _shift_packed_positions(pos: torch.Tensor, token_start: int) -> torch.Tensor:
    valid = pos >= 0
    return torch.where(valid, pos - int(token_start), pos)


def _slice_packed_text_batch(
    batch: PackedTextBatch,
    *,
    row_start: int,
    row_end: int,
    token_start: int,
    token_end: int,
) -> PackedTextBatch:
    seq_lengths = batch.seq_lengths[row_start:row_end]
    seq_lengths_host = (
        batch.seq_lengths_host[row_start:row_end] if batch.seq_lengths_host is not None else None
    )
    token_ids = batch.token_ids[token_start:token_end]
    cu_seqlens = batch.cu_seqlens[row_start : row_end + 1].to(torch.int32) - int(token_start)
    total_tokens = int(token_end - token_start)
    seq_id = batch.seq_id[token_start:token_end] - int(row_start)
    pos_in_seq = batch.pos_in_seq[token_start:token_end]
    state_positions = batch.state_positions[row_start:row_end].to(torch.int32) - int(token_start)
    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu_seqlens,
        seq_lengths=seq_lengths,
        state_positions=state_positions,
        card_ref_positions=_shift_packed_positions(
            batch.card_ref_positions[row_start:row_end],
            token_start,
        ),
        total_tokens=total_tokens,
        seq_lengths_host=seq_lengths_host,
        blank_positions=_shift_packed_positions(
            batch.blank_positions[row_start:row_end],
            token_start,
        ),
        blank_kind=batch.blank_kind[row_start:row_end],
        blank_group=batch.blank_group[row_start:row_end],
        blank_group_kind=batch.blank_group_kind[row_start:row_end],
        blank_option_index=batch.blank_option_index[row_start:row_end],
        blank_legal_ids=batch.blank_legal_ids[row_start:row_end],
        blank_legal_mask=batch.blank_legal_mask[row_start:row_end],
        max_seqlen=max(seq_lengths_host, default=0) if seq_lengths_host is not None else None,
    )


# The protocol the server calls. Concretely this is
# ``TextActorCritic.sample_native_tensor_batch`` but parameterizing keeps the
# server testable against fakes.
ForwardCallable = Any


@dataclass
class _PendingItem:
    future: Future[TextInferenceReply]
    rows: int
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    decision_start: int
    decision_end: int
    env_indices: list[int]
    perspective_player_indices: list[int]
    seq_lengths_host: tuple[int, ...]
    blank_cols: int
    legal_cols: int
    copy_event: torch.cuda.Event | None = None
    released: bool = False


class _InferenceWorkRing:
    """Thread-safe FIFO backed by contiguous arena storage.

    Producers reserve row/token/decision spans, copy their encoded microbatch
    into shared arena tensors, then publish only a small descriptor. The server
    drains descriptors in FIFO order and slices one contiguous arena range for
    the policy call, avoiding per-batch ``torch.cat`` work on the critical path.
    """

    def __init__(self, *, capacity_rows: int, arena_device: torch.device) -> None:
        if capacity_rows < 1:
            raise ValueError("capacity_rows must be >= 1")
        self.capacity_rows = int(capacity_rows)
        self.arena_device = torch.device(arena_device)
        self.token_capacity = self.capacity_rows * 1024
        self.decision_capacity = self.capacity_rows * 128
        self._cond = threading.Condition()
        self._items: list[_PendingItem | None] = []
        self._active_rows = 0
        self._active_tokens = 0
        self._active_decisions = 0
        self._reserved_rows = 0
        self._reserved_tokens = 0
        self._reserved_decisions = 0
        self._waiting_producers = 0
        self._row_cursor = 0
        self._token_cursor = 0
        self._decision_cursor = 0
        self._next_reservation_seq = 0
        self._next_publish_seq = 0
        self._pending_publish: dict[int, _PendingItem] = {}
        self._closed = False
        self._force_flush = False
        self._native_arena: dict[str, torch.Tensor] | None = None
        self._packed_arena: dict[str, torch.Tensor] | None = None

    def put(
        self,
        request: TextInferenceRequest,
        future: Future[TextInferenceReply],
        *,
        block: bool = True,
    ) -> bool:
        rows = len(request.env_indices)
        if rows > self.capacity_rows:
            raise RuntimeError(
                f"inference work item has {rows} rows, exceeds ring capacity {self.capacity_rows}"
            )
        packed = request.packed_batch
        token_count = int(packed.token_ids.shape[0])
        native = cast(NativeEncodedBatch, request.native_batch)
        decision_count = int(native.decision_rows_written)
        self._validate_item_capacity(rows=rows, tokens=token_count, decisions=decision_count)
        with self._cond:
            self._ensure_arenas(native, packed)
            while not self._closed:
                self._reset_if_empty_locked()
                if self._has_tail_capacity_locked(
                    rows=rows,
                    tokens=token_count,
                    decisions=decision_count,
                ):
                    break
                if not block:
                    self._force_flush = True
                    self._cond.notify_all()
                    return False
                self._waiting_producers += 1
                self._cond.notify_all()
                try:
                    self._cond.wait()
                finally:
                    self._waiting_producers -= 1
            if self._closed:
                raise RuntimeError("inference work ring is closed")
            row_start = self._row_cursor
            row_end = row_start + rows
            token_start = self._token_cursor
            token_end = token_start + token_count
            decision_start = self._decision_cursor
            decision_end = decision_start + decision_count
            publish_seq = self._next_reservation_seq
            self._next_reservation_seq += 1
            native_arena = self._native_arena
            packed_arena = self._packed_arena
            self._row_cursor = row_end
            self._token_cursor = token_end
            self._decision_cursor = decision_end
            self._reserved_rows += rows
            self._reserved_tokens += token_count
            self._reserved_decisions += decision_count

        copy_event: torch.cuda.Event | None = None
        try:
            self._copy_into_arena(
                native=native,
                packed=packed,
                native_arena=cast(dict[str, torch.Tensor], native_arena),
                packed_arena=cast(dict[str, torch.Tensor], packed_arena),
                row_start=row_start,
                row_end=row_end,
                token_start=token_start,
                token_end=token_end,
                decision_start=decision_start,
                decision_end=decision_end,
            )
            if self.arena_device.type == "cuda":
                copy_event = torch.cuda.Event()
                copy_event.record(torch.cuda.current_stream(self.arena_device))
        except BaseException:
            with self._cond:
                self._reserved_rows -= rows
                self._reserved_tokens -= token_count
                self._reserved_decisions -= decision_count
                self._reset_if_empty_locked()
                self._cond.notify_all()
            raise

        with self._cond:
            self._reserved_rows -= rows
            self._reserved_tokens -= token_count
            self._reserved_decisions -= decision_count
            if self._closed:
                self._reset_if_empty_locked()
                self._cond.notify_all()
                raise RuntimeError("inference work ring is closed")
            seq_lengths_host = packed.seq_lengths_host
            if seq_lengths_host is None:
                seq_lengths_host = tuple(int(x) for x in packed.seq_lengths.cpu().tolist())
            item = _PendingItem(
                future=future,
                rows=rows,
                row_start=row_start,
                row_end=row_end,
                token_start=token_start,
                token_end=token_end,
                decision_start=decision_start,
                decision_end=decision_end,
                env_indices=list(request.env_indices),
                perspective_player_indices=list(request.perspective_player_indices),
                seq_lengths_host=seq_lengths_host,
                blank_cols=int(packed.blank_positions.shape[1]),
                legal_cols=int(packed.blank_legal_ids.shape[2]),
                copy_event=copy_event,
            )
            self._pending_publish[publish_seq] = item
            while self._next_publish_seq in self._pending_publish:
                published = self._pending_publish.pop(self._next_publish_seq)
                self._items.append(published)
                self._next_publish_seq += 1
            self._cond.notify_all()
        return True

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._force_flush = True
            self._items.append(None)
            self._cond.notify_all()

    def request_flush(self) -> None:
        with self._cond:
            self._force_flush = True
            self._cond.notify_all()

    def get_first(self, *, timeout: float) -> _PendingItem | None | object:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise Empty
                self._cond.wait(timeout=remaining)
            item = self._items.pop(0)
            if item is not None:
                self._active_rows += item.rows
                self._active_tokens += item.token_end - item.token_start
                self._active_decisions += item.decision_end - item.decision_start
            self._cond.notify_all()
            return item

    def get_nowait_fit(self, *, remaining_rows: int) -> _PendingItem | None | object:
        with self._cond:
            if not self._items:
                raise Empty
            item = self._items[0]
            if item is None:
                self._items.pop(0)
                self._cond.notify_all()
                return item
            if item.rows > remaining_rows:
                raise _HeadItemDoesNotFit
            self._items.pop(0)
            self._active_rows += item.rows
            self._active_tokens += item.token_end - item.token_start
            self._active_decisions += item.decision_end - item.decision_start
            self._cond.notify_all()
            return item

    def get_fit(self, *, remaining_rows: int, timeout: float) -> _PendingItem | None | object:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._force_flush or self._closed:
                    raise Empty
                if self._items:
                    item = self._items[0]
                    if item is None:
                        self._items.pop(0)
                        self._cond.notify_all()
                        return item
                    if item.rows <= remaining_rows:
                        self._items.pop(0)
                        self._active_rows += item.rows
                        self._active_tokens += item.token_end - item.token_start
                        self._active_decisions += item.decision_end - item.decision_start
                        self._cond.notify_all()
                        return item
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise Empty
                self._cond.wait(timeout=remaining)

    def wait_for_item_or_flush(self) -> bool:
        with self._cond:
            while (
                not self._items
                and not self._force_flush
                and not self._closed
                and self._waiting_producers <= 0
            ):
                self._cond.wait()
            return self._force_flush or self._closed or self._waiting_producers > 0

    def clear_flush(self) -> None:
        with self._cond:
            self._force_flush = False

    def drain(self) -> list[_PendingItem]:
        with self._cond:
            items = [item for item in self._items if item is not None]
            self._items.clear()
            self._pending_publish.clear()
            self._reset_if_empty_locked()
            self._cond.notify_all()
            return items

    def qsize(self) -> int:
        with self._cond:
            return len([item for item in self._items if item is not None])

    def queued_rows(self) -> int:
        with self._cond:
            return sum(item.rows for item in self._items if item is not None)

    def has_tail_capacity_for_another_item(self) -> bool:
        with self._cond:
            return (
                self._row_cursor < self.capacity_rows
                and self._token_cursor < self.token_capacity
                and self._decision_cursor < self.decision_capacity
            )

    def has_capacity_blocked_producers(self) -> bool:
        with self._cond:
            return self._waiting_producers > 0

    def finish_items(self, items: list[_PendingItem]) -> None:
        with self._cond:
            for item in items:
                if item.released:
                    continue
                item.released = True
                self._active_rows -= item.rows
                self._active_tokens -= item.token_end - item.token_start
                self._active_decisions -= item.decision_end - item.decision_start
            self._reset_if_empty_locked()
            self._cond.notify_all()

    def native_batch_for(self, items: list[_PendingItem]) -> NativeEncodedBatch:
        arena = self._native_arena
        if arena is None:
            raise RuntimeError("native arena is not initialized")
        row_start = items[0].row_start
        row_end = items[-1].row_end
        decision_start = items[0].decision_start
        decision_end = items[-1].decision_end

        def row(name: str) -> torch.Tensor:
            return arena[name][row_start:row_end]

        def decisions(name: str) -> torch.Tensor:
            return arena[name][decision_start:decision_end]

        return NativeEncodedBatch(
            trace_kind_id=row("trace_kind_id"),
            slot_card_rows=row("slot_card_rows"),
            slot_occupied=row("slot_occupied"),
            slot_tapped=row("slot_tapped"),
            game_info=row("game_info"),
            pending_kind_id=row("pending_kind_id"),
            num_present_options=row("num_present_options"),
            option_kind_ids=row("option_kind_ids"),
            option_scalars=row("option_scalars"),
            option_mask=row("option_mask"),
            option_ref_slot_idx=row("option_ref_slot_idx"),
            option_ref_card_row=row("option_ref_card_row"),
            target_mask=row("target_mask"),
            target_type_ids=row("target_type_ids"),
            target_scalars=row("target_scalars"),
            target_overflow=row("target_overflow"),
            target_ref_slot_idx=row("target_ref_slot_idx"),
            target_ref_is_player=row("target_ref_is_player"),
            target_ref_is_self=row("target_ref_is_self"),
            may_mask=row("may_mask"),
            decision_start=row("decision_start") - int(decision_start),
            decision_count=row("decision_count"),
            decision_option_idx=decisions("decision_option_idx"),
            decision_target_idx=decisions("decision_target_idx"),
            decision_mask=decisions("decision_mask"),
            uses_none_head=decisions("uses_none_head"),
            decision_rows_written=int(decision_end - decision_start),
            pendings=cast(Any, [None] * int(row_end - row_start)),
            trace_kinds=[""] * int(row_end - row_start),
        )

    def packed_batch_for(self, items: list[_PendingItem]) -> PackedTextBatch:
        if self._packed_arena is None:
            raise RuntimeError("packed arena is not initialized")
        row_start = items[0].row_start
        row_end = items[-1].row_end
        token_start = items[0].token_start
        token_end = items[-1].token_end
        seq_lengths_host = tuple(n for item in items for n in item.seq_lengths_host)
        arena_batch = PackedTextBatch(
            token_ids=self._packed_arena["token_ids"],
            seq_id=self._packed_arena["seq_id"],
            pos_in_seq=self._packed_arena["pos_in_seq"],
            cu_seqlens=self._packed_arena["cu_seqlens"],
            seq_lengths=self._packed_arena["seq_lengths"],
            state_positions=self._packed_arena["state_positions"],
            card_ref_positions=self._packed_arena["card_ref_positions"],
            total_tokens=self._token_cursor,
            seq_lengths_host=None,
            blank_positions=self._packed_arena["blank_positions"],
            blank_kind=self._packed_arena["blank_kind"],
            blank_group=self._packed_arena["blank_group"],
            blank_group_kind=self._packed_arena["blank_group_kind"],
            blank_option_index=self._packed_arena["blank_option_index"],
            blank_legal_ids=self._packed_arena["blank_legal_ids"],
            blank_legal_mask=self._packed_arena["blank_legal_mask"],
        )
        sliced = _slice_packed_text_batch(
            arena_batch,
            row_start=row_start,
            row_end=row_end,
            token_start=token_start,
            token_end=token_end,
        )
        sliced.seq_lengths_host = seq_lengths_host
        sliced.max_seqlen = max(seq_lengths_host, default=0)
        blank_cols = max((item.blank_cols for item in items), default=0)
        legal_cols = max((item.legal_cols for item in items), default=0) if blank_cols > 0 else 0
        sliced.blank_positions = sliced.blank_positions[:, :blank_cols]
        sliced.blank_kind = sliced.blank_kind[:, :blank_cols]
        sliced.blank_group = sliced.blank_group[:, :blank_cols]
        sliced.blank_group_kind = sliced.blank_group_kind[:, :blank_cols]
        sliced.blank_option_index = sliced.blank_option_index[:, :blank_cols]
        sliced.blank_legal_ids = sliced.blank_legal_ids[:, :blank_cols, :legal_cols]
        sliced.blank_legal_mask = sliced.blank_legal_mask[:, :blank_cols, :legal_cols]
        return sliced

    def wait_for_item_copies(self, items: list[_PendingItem]) -> None:
        if self.arena_device.type != "cuda":
            return
        stream = torch.cuda.current_stream(self.arena_device)
        for item in items:
            if item.copy_event is not None:
                stream.wait_event(item.copy_event)

    def _validate_item_capacity(self, *, rows: int, tokens: int, decisions: int) -> None:
        if tokens > self.token_capacity:
            raise RuntimeError(
                f"inference work item has {tokens} tokens, exceeds arena token capacity "
                f"{self.token_capacity}"
            )
        if decisions > self.decision_capacity:
            raise RuntimeError(
                f"inference work item has {decisions} decision rows, exceeds arena decision "
                f"capacity {self.decision_capacity}"
            )

    def _has_tail_capacity_locked(self, *, rows: int, tokens: int, decisions: int) -> bool:
        return (
            self._row_cursor + rows <= self.capacity_rows
            and self._token_cursor + tokens <= self.token_capacity
            and self._decision_cursor + decisions <= self.decision_capacity
        )

    def _reset_if_empty_locked(self) -> None:
        if (
            not any(item is not None for item in self._items)
            and self._active_rows == 0
            and self._active_tokens == 0
            and self._active_decisions == 0
            and self._reserved_rows == 0
            and self._reserved_tokens == 0
            and self._reserved_decisions == 0
            and not self._pending_publish
        ):
            self._row_cursor = 0
            self._token_cursor = 0
            self._decision_cursor = 0
            self._next_reservation_seq = 0
            self._next_publish_seq = 0

    def _ensure_arenas(self, native: NativeEncodedBatch, packed: PackedTextBatch) -> None:
        if self._native_arena is None:
            self._native_arena = {}
            row_names = {
                "trace_kind_id",
                "slot_card_rows",
                "slot_occupied",
                "slot_tapped",
                "game_info",
                "pending_kind_id",
                "num_present_options",
                "option_kind_ids",
                "option_scalars",
                "option_mask",
                "option_ref_slot_idx",
                "option_ref_card_row",
                "target_mask",
                "target_type_ids",
                "target_scalars",
                "target_overflow",
                "target_ref_slot_idx",
                "target_ref_is_player",
                "target_ref_is_self",
                "may_mask",
                "decision_start",
                "decision_count",
            }
            decision_names = {
                "decision_option_idx",
                "decision_target_idx",
                "decision_mask",
                "uses_none_head",
            }
            for field in fields(NativeEncodedBatch):
                name = field.name
                if name in row_names:
                    src = cast(torch.Tensor, getattr(native, name))
                    self._native_arena[name] = _arena_empty(
                        self.capacity_rows,
                        *src.shape[1:],
                        dtype=src.dtype,
                        device=self.arena_device,
                    )
                elif name in decision_names:
                    src = cast(torch.Tensor, getattr(native, name))
                    self._native_arena[name] = _arena_empty(
                        self.decision_capacity,
                        *src.shape[1:],
                        dtype=src.dtype,
                        device=self.arena_device,
                    )
        if self._packed_arena is None:
            blank_capacity = max(64, int(packed.blank_positions.shape[1]))
            legal_capacity = max(64, int(packed.blank_legal_ids.shape[2]))
            self._packed_arena = {
                "token_ids": _arena_empty(
                    self.token_capacity,
                    dtype=packed.token_ids.dtype,
                    device=self.arena_device,
                ),
                "seq_id": _arena_empty(
                    self.token_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "pos_in_seq": _arena_empty(
                    self.token_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "cu_seqlens": _arena_empty(
                    self.capacity_rows + 1,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "seq_lengths": _arena_empty(
                    self.capacity_rows,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "state_positions": _arena_empty(
                    self.capacity_rows,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "card_ref_positions": _arena_empty(
                    self.capacity_rows,
                    *packed.card_ref_positions.shape[1:],
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "blank_positions": _arena_empty(
                    self.capacity_rows,
                    blank_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "blank_kind": _arena_empty(
                    self.capacity_rows,
                    blank_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "blank_group": _arena_empty(
                    self.capacity_rows,
                    blank_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "blank_group_kind": _arena_empty(
                    self.capacity_rows,
                    blank_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "blank_option_index": _arena_empty(
                    self.capacity_rows,
                    blank_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "blank_legal_ids": _arena_empty(
                    self.capacity_rows,
                    blank_capacity,
                    legal_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "blank_legal_mask": _arena_empty(
                    self.capacity_rows,
                    blank_capacity,
                    legal_capacity,
                    dtype=torch.bool,
                    device=self.arena_device,
                ),
            }

    def _copy_into_arena(
        self,
        *,
        native: NativeEncodedBatch,
        packed: PackedTextBatch,
        native_arena: dict[str, torch.Tensor],
        packed_arena: dict[str, torch.Tensor],
        row_start: int,
        row_end: int,
        token_start: int,
        token_end: int,
        decision_start: int,
        decision_end: int,
    ) -> None:
        rows = row_end - row_start
        decision_rows = decision_end - decision_start
        for name, dst in native_arena.items():
            if name in {
                "decision_option_idx",
                "decision_target_idx",
                "decision_mask",
                "uses_none_head",
            }:
                dst[decision_start:decision_end].copy_(
                    cast(torch.Tensor, getattr(native, name))[:decision_rows]
                )
            elif name == "decision_start":
                dst[row_start:row_end].copy_(native.decision_start[:rows] + int(decision_start))
            else:
                dst[row_start:row_end].copy_(cast(torch.Tensor, getattr(native, name))[:rows])

        packed_arena["token_ids"][token_start:token_end].copy_(packed.token_ids)
        packed_arena["seq_id"][token_start:token_end].copy_(
            packed.seq_id.to(torch.int32) + int(row_start)
        )
        packed_arena["pos_in_seq"][token_start:token_end].copy_(packed.pos_in_seq.to(torch.int32))
        packed_arena["cu_seqlens"][row_start : row_end + 1].copy_(
            packed.cu_seqlens.to(torch.int32) + int(token_start)
        )
        packed_arena["seq_lengths"][row_start:row_end].copy_(packed.seq_lengths.to(torch.int32))
        packed_arena["state_positions"][row_start:row_end].copy_(
            packed.state_positions.to(torch.int32) + int(token_start)
        )
        src_refs = packed.card_ref_positions.to(torch.int32)
        packed_arena["card_ref_positions"][row_start:row_end].copy_(
            torch.where(src_refs >= 0, src_refs + int(token_start), src_refs)
        )

        self._copy_blank_2d(
            name="blank_positions",
            src=packed.blank_positions.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=-1,
            token_start=token_start,
        )
        self._copy_blank_2d(
            name="blank_kind",
            src=packed.blank_kind.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=0,
        )
        self._copy_blank_2d(
            name="blank_group",
            src=packed.blank_group.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=-1,
        )
        self._copy_blank_2d(
            name="blank_group_kind",
            src=packed.blank_group_kind.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=0,
        )
        self._copy_blank_2d(
            name="blank_option_index",
            src=packed.blank_option_index.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=-1,
        )
        self._copy_blank_3d(
            name="blank_legal_ids",
            src=packed.blank_legal_ids.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
        )
        self._copy_blank_3d(
            name="blank_legal_mask",
            src=packed.blank_legal_mask.bool(),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
        )

    def _copy_blank_2d(
        self,
        *,
        name: str,
        src: torch.Tensor,
        packed_arena: dict[str, torch.Tensor],
        row_start: int,
        row_end: int,
        fill: int,
        token_start: int | None = None,
    ) -> None:
        dst = packed_arena[name][row_start:row_end]
        src_cols = int(src.shape[1])
        if src_cols > int(dst.shape[1]):
            raise RuntimeError(
                f"packed arena field {name} has width {int(dst.shape[1])}, request needs {src_cols}"
            )
        dst.fill_(fill)
        if src_cols == 0:
            return
        if token_start is not None:
            src = torch.where(src >= 0, src + int(token_start), src)
        dst[:, :src_cols].copy_(src)

    def _copy_blank_3d(
        self,
        *,
        name: str,
        src: torch.Tensor,
        packed_arena: dict[str, torch.Tensor],
        row_start: int,
        row_end: int,
    ) -> None:
        dst = packed_arena[name][row_start:row_end]
        src_k = int(src.shape[1])
        src_v = int(src.shape[2])
        if src_k > int(dst.shape[1]) or src_v > int(dst.shape[2]):
            raise RuntimeError(
                f"packed arena field {name} has shape {tuple(dst.shape[1:])}, "
                f"request needs {(src_k, src_v)}"
            )
        dst.zero_()
        if src_k == 0 or src_v == 0:
            return
        dst[:, :src_k, :src_v].copy_(src)


class TextInferenceServer:
    """Dynamic-batching inference server for the native text rollout path.

    Submitting actors block on their per-request ``Future``; the server thread
    drains the queue once the configured row threshold is reached (or an
    explicit flush/stop arrives), merges descriptors into a single GPU batch,
    runs the policy forward, and resolves each future with a sliced
    ``TextInferenceReply``. Actors own trajectory staging and commit finished
    games into replay after returns can be computed.
    """

    def __init__(
        self,
        *,
        sampling_policy: Any,  # TextActorCritic-compatible
        max_batch: int,
        deterministic: bool = False,
        name: str = "text-inference",
        timing_stats: RolloutTimingStats | None = None,
        ring_capacity_rows: int | None = None,
        min_batch_rows: int = 1,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        if min_batch_rows < 1:
            raise ValueError("min_batch_rows must be >= 1")
        self._policy = sampling_policy
        self._policy_version_manager = (
            sampling_policy if hasattr(sampling_policy, "acquire_inference_policy") else None
        )
        self._max_batch = int(max_batch)
        self._min_batch_rows = min(int(min_batch_rows), int(max_batch))
        self._deterministic = bool(deterministic)
        self._timing = timing_stats
        self._queue = _InferenceWorkRing(
            arena_device=_infer_policy_device(sampling_policy),
            capacity_rows=(
                int(ring_capacity_rows)
                if ring_capacity_rows is not None
                else max(int(max_batch) * 8, int(max_batch))
            ),
        )
        self._stop_event = threading.Event()
        # Pause/resume coordination. ``_paused`` is the desired state; the
        # server checks it (and waits on the condition) between batches. The
        # condition also publishes ``_idle`` (True when the server is not
        # currently running a forward pass), which ``pause()`` waits on so
        # the in-flight batch always completes before pause returns.
        self._cond = threading.Condition()
        self._paused = False
        self._idle = True
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._exc: BaseException | None = None

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def submit(self, request: TextInferenceRequest) -> Future[TextInferenceReply]:
        """Submit one actor's encoded batch; returns a future for host scalars.

        Fails fast if the server thread has died — without this, actors would
        block forever on the future.
        """

        if not request.env_indices:
            raise ValueError("submit() requires a non-empty request")
        if self._stop_event.is_set() or not self._thread.is_alive():
            fut: Future[TextInferenceReply] = Future()
            fut.set_exception(
                self._exc
                if self._exc is not None
                else RuntimeError("inference server is not running")
            )
            return fut
        fut2: Future[TextInferenceReply] = Future()
        try:
            self._queue.put(request, fut2)
        except BaseException as exc:  # noqa: BLE001
            fut2.set_exception(exc)
        return fut2

    def try_submit(self, request: TextInferenceRequest) -> Future[TextInferenceReply] | None:
        """Submit without waiting for ring capacity.

        ``None`` means the request was not published because the arena tail is
        currently full. The caller still owns the encoded tensors and should
        drain completed replies before retrying.
        """

        if not request.env_indices:
            raise ValueError("try_submit() requires a non-empty request")
        if self._stop_event.is_set() or not self._thread.is_alive():
            fut: Future[TextInferenceReply] = Future()
            fut.set_exception(
                self._exc
                if self._exc is not None
                else RuntimeError("inference server is not running")
            )
            return fut
        fut2: Future[TextInferenceReply] = Future()
        try:
            published = self._queue.put(request, fut2, block=False)
        except BaseException as exc:  # noqa: BLE001
            fut2.set_exception(exc)
            return fut2
        if not published:
            return None
        return fut2

    def pause(self) -> None:
        """Block new batches from running. Waits for the in-flight batch."""

        with self._cond:
            self._paused = True
            while not self._idle:
                self._cond.wait()

    def resume(self) -> None:
        with self._cond:
            self._paused = False
            self._cond.notify_all()

    def flush(self) -> None:
        """Force the next queued partial batch to run below min_batch_rows."""

        self._queue.request_flush()

    def stop(self) -> None:
        with self._cond:
            self._stop_event.set()
            self._paused = False
            self._cond.notify_all()
        self._queue.close()
        # Reject any queued requests so actors blocked in ``future.result()``
        # wake up immediately instead of waiting for the server thread to
        # drain. The server thread itself returns on the None sentinel.
        self._fail_pending(RuntimeError("inference server is stopping"))
        self._thread.join(timeout=2.0)
        if self._exc is not None:
            raise self._exc

    # --------------------------------------------------------------- internal

    def _run(self) -> None:
        # Catch *per-batch* failures and propagate to that batch's futures
        # only; the server keeps running so other actors aren't deadlocked
        # by one bad batch. A batch failure still records ``self._exc`` so
        # ``stop()`` re-raises it on the learner thread.
        while not self._stop_event.is_set():
            try:
                batch = self._collect_batch()
            except BaseException as exc:  # noqa: BLE001
                self._exc = exc
                self._fail_pending(exc)
                return
            if batch is None:  # stop sentinel
                return
            if not batch:
                continue
            # Gate: if paused was requested between the queue.get() above
            # and now, hold the items here (not yet processing) until
            # resume. ``idle`` stays True throughout the wait so pause()
            # can return immediately.
            with self._cond:
                while self._paused and not self._stop_event.is_set():
                    self._cond.wait()
                if self._stop_event.is_set():
                    # Re-queue items so they get a clean exception below.
                    self._fail_items(batch, RuntimeError("inference server stopped"))
                    self._queue.finish_items(batch)
                    return
                self._idle = False
            try:
                processed = False
                try:
                    self._process(batch)
                    processed = True
                except BaseException as exc:  # noqa: BLE001
                    self._exc = exc
                    # Resolve only the in-flight batch's futures; do NOT
                    # tear down the server. Pending submits are independent.
                    self._fail_items(batch, exc)
            finally:
                if not processed:
                    self._queue.finish_items(batch)
                with self._cond:
                    self._idle = True
                    self._cond.notify_all()

    @staticmethod
    def _fail_items(items: list[_PendingItem], exc: BaseException) -> None:
        for it in items:
            if not it.future.done():
                it.future.set_exception(exc)

    def _fail_pending(self, exc: BaseException) -> None:
        """Drain remaining queued items and reject their futures."""

        for item in self._queue.drain():
            if not item.future.done():
                item.future.set_exception(exc)
        with self._cond:
            self._idle = True
            self._cond.notify_all()

    def _collect_batch(self) -> list[_PendingItem] | None:
        """Block until a launch condition is met, then coalesce up to max_batch."""
        start = time.perf_counter()
        try:
            try:
                first = self._queue.get_first(timeout=0.1)
            except Empty:
                return []
            if first is None:
                return None
            if not isinstance(first, _PendingItem):
                return []
            items: list[_PendingItem] = [first]
            rows = first.rows
            while rows < self._max_batch:
                remaining_rows = self._max_batch - rows
                try:
                    nxt = self._queue.get_nowait_fit(remaining_rows=remaining_rows)
                except _HeadItemDoesNotFit:
                    break
                except Empty:
                    if rows >= self._min_batch_rows:
                        break
                    if (
                        not self._queue.has_tail_capacity_for_another_item()
                        or self._queue.has_capacity_blocked_producers()
                    ):
                        break
                    if self._queue.wait_for_item_or_flush():
                        break
                    continue
                if nxt is None:
                    break
                if not isinstance(nxt, _PendingItem):
                    break
                items.append(nxt)
                rows += nxt.rows
            self._queue.clear_flush()
            return items
        finally:
            if self._timing is not None:
                self._timing.add("server_collect", time.perf_counter() - start)

    def _process(self, items: list[_PendingItem]) -> None:
        start = time.perf_counter()
        self._queue.wait_for_item_copies(items)
        merged_native = self._queue.native_batch_for(items)
        merged_packed = self._queue.packed_batch_for(items)
        env_indices: list[int] = [i for it in items for i in it.env_indices]
        perspective: list[int] = [p for it in items for p in it.perspective_player_indices]
        if self._timing is not None:
            self._timing.add("server_concat", time.perf_counter() - start)
            max_seq = (
                int(merged_packed.seq_lengths.max().item())
                if int(merged_packed.seq_lengths.numel()) > 0
                else 0
            )
            self._timing.add_batch(
                rows=len(env_indices),
                tokens=int(merged_packed.token_ids.shape[0]),
                max_seq=max_seq,
            )

        start = time.perf_counter()
        policy_version = 0
        sample_profile: dict[str, float] | None = (
            {} if self._timing is not None and _PROFILE_SAMPLE_COMPONENTS else None
        )
        with torch.no_grad():
            if self._policy_version_manager is not None:
                with self._policy_version_manager.acquire_inference_policy() as (
                    policy,
                    policy_version,
                ):
                    sample = policy.sample_native_tensor_batch(
                        native_batch=merged_native,
                        env_indices=env_indices,
                        perspective_player_indices=perspective,
                        packed_batch=merged_packed,
                        deterministic=self._deterministic,
                        append_replay=False,
                        return_replay_payload=True,
                        profile_timings=sample_profile,
                    )
            else:
                sample = self._policy.sample_native_tensor_batch(
                    native_batch=merged_native,
                    env_indices=env_indices,
                    perspective_player_indices=perspective,
                    packed_batch=merged_packed,
                    deterministic=self._deterministic,
                    append_replay=False,
                    return_replay_payload=True,
                    profile_timings=sample_profile,
                )
        if self._timing is not None:
            self._timing.add("server_sample", time.perf_counter() - start)
            if sample_profile is not None:
                for name, elapsed in sample_profile.items():
                    self._timing.add(f"server_sample_{name}", elapsed)
        if sample.replay_payload is None:
            raise RuntimeError("inference server expected a replay payload")
        ready_event: Any | None = None
        if sample.replay_payload.value.device.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(sample.replay_payload.value.device))

        # Scatter per-request host-side slices.
        start = time.perf_counter()
        row_cursor = 0
        col_cursor = 0
        token_cursor = 0
        decision_counts = sample.decision_counts
        sel_cols = sample.selected_choice_cols
        may_sel = sample.may_selected
        log_prob = sample.old_log_prob
        value = sample.value
        trace_kind_ids = merged_native.trace_kind_id.tolist()
        seq_lengths_host = merged_packed.seq_lengths_host
        if seq_lengths_host is None:
            seq_lengths_host = tuple(int(x) for x in merged_packed.seq_lengths.cpu().tolist())
        for it in items:
            n = len(it.env_indices)
            row_end = row_cursor + n
            req_counts = decision_counts[row_cursor:row_end]
            req_cols_total = sum(req_counts)
            token_end = token_cursor + sum(int(x) for x in seq_lengths_host[row_cursor:row_end])
            replay_payload = NativeTextReplayPayload(
                encoded=_slice_packed_text_batch(
                    merged_packed,
                    row_start=row_cursor,
                    row_end=row_end,
                    token_start=token_cursor,
                    token_end=token_end,
                ),
                trace_kind_id=sample.replay_payload.trace_kind_id[row_cursor:row_end],
                decision_count=sample.replay_payload.decision_count[row_cursor:row_end],
                decision_count_host=tuple(int(x) for x in req_counts),
                total_decision_groups=req_cols_total,
                total_stored_decision_groups=req_cols_total,
                decision_option_idx=sample.replay_payload.decision_option_idx[
                    col_cursor : col_cursor + req_cols_total
                ],
                decision_target_idx=sample.replay_payload.decision_target_idx[
                    col_cursor : col_cursor + req_cols_total
                ],
                decision_mask=sample.replay_payload.decision_mask[
                    col_cursor : col_cursor + req_cols_total
                ],
                uses_none_head=sample.replay_payload.uses_none_head[
                    col_cursor : col_cursor + req_cols_total
                ],
                selected_indices=sample.replay_payload.selected_indices[
                    col_cursor : col_cursor + req_cols_total
                ],
                behavior_action_log_prob=sample.replay_payload.behavior_action_log_prob[
                    col_cursor : col_cursor + req_cols_total
                ]
                if sample.replay_payload.behavior_action_log_prob is not None
                else None,
                may_selected=sample.replay_payload.may_selected[row_cursor:row_end],
                old_log_prob=sample.replay_payload.old_log_prob[row_cursor:row_end],
                value=sample.replay_payload.value[row_cursor:row_end],
                perspective_player_idx=sample.replay_payload.perspective_player_idx[
                    row_cursor:row_end
                ],
                lstm_h_in=sample.replay_payload.lstm_h_in[:, row_cursor:row_end],
                lstm_c_in=sample.replay_payload.lstm_c_in[:, row_cursor:row_end],
                projected_state=sample.replay_payload.projected_state[row_cursor:row_end]
                if sample.replay_payload.projected_state is not None
                else None,
            )
            reply = TextInferenceReply(
                decision_counts=list(req_counts),
                selected_choice_cols=list(sel_cols[col_cursor : col_cursor + req_cols_total]),
                may_selected=list(may_sel[row_cursor:row_end]),
                old_log_prob=list(log_prob[row_cursor:row_end]),
                value=list(value[row_cursor:row_end]),
                trace_kind_id=list(trace_kind_ids[row_cursor:row_end]),
                replay_payload=replay_payload,
                replay_rows=None,
                inference_policy_version=int(policy_version),
                ready_event=ready_event,
                release_item=lambda item=it: self._queue.finish_items([item]),
            )
            row_cursor = row_end
            col_cursor += req_cols_total
            token_cursor = token_end
            it.future.set_result(reply)
        if self._timing is not None:
            self._timing.add("server_scatter", time.perf_counter() - start)


__all__ = [
    "TextInferenceRequest",
    "TextInferenceReply",
    "TextInferenceServer",
    "RolloutTimingStats",
    "_concat_packed_text_batches",
]
