"""IMPALA-style inference server for the text-encoder rollout path.

A single background thread owns the GPU policy: it drains encoded-batch
requests submitted by N actor threads, dynamically batches them into one
forward pass, runs the policy's decoder forward on the merged batch, and
scatters per-request replay payload slices plus host-side scalars
(decision counts, selected tokens, log-prob, value, trace-kind) back to
the actors via per-request ``Future`` objects.

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
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, fields
from queue import Empty
from typing import Any, cast

import torch

from magic_ai.slot_encoder.native_encoder import NativeEncodedBatch
from magic_ai.text_encoder.batch import (
    PackedTextBatch,
    scatter_packed_to_padded,
)
from magic_ai.text_encoder.decoder_batch import (
    NativeTextDecoderBatch,
    native_decoder_batch_from_sample,
)
from magic_ai.text_encoder.decoder_inference import decoder_sample

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
class DecoderHostView:
    """Host-side decoder fields the actor needs after inference.

    Pre-staged on the server (one D→H pass per merged batch, sliced
    per-actor in scatter) so the actor's post-inference critical path is
    GPU-free — transcript bookkeeping, the cgo ``batch_step_by_decoder_action``
    call, and per-env staging into the replay buffer all read from these
    host tensors directly.

    The view carries the full decoder batch (everything an episode-commit
    write into the CPU replay buffer needs), not just the cgo-step subset.
    ``output_is_pointer_u8`` is the uint8 form cgo expects;
    ``output_is_pointer`` is the bool view used by staging / training
    paths. ``log_probs_sum`` is precomputed from ``log_probs`` for
    actor record_step.

    All fields are CPU tensors with leading dim equal to the per-actor row
    count.
    """

    decision_type: torch.Tensor  # [n] int
    output_token_ids: torch.Tensor  # [n, L_max] int32
    output_pointer_pos: torch.Tensor  # [n, L_max] int32
    output_pointer_subjects: torch.Tensor  # [n, L_max] int32
    output_is_pointer: torch.Tensor  # [n, L_max] bool
    output_is_pointer_u8: torch.Tensor  # [n, L_max] uint8 (cgo)
    output_pad_mask: torch.Tensor  # [n, L_max] bool
    output_lens: torch.Tensor  # [n] int
    pointer_anchor_handles: torch.Tensor  # [n, N_max] int
    pointer_anchor_count: torch.Tensor  # [n] int
    log_probs: torch.Tensor  # [n, L_max] float32 (zero at pad)
    log_probs_sum: torch.Tensor  # [n] float32 (sum over L_max)
    value: torch.Tensor  # [n] float32
    vocab_mask: torch.Tensor  # [n, L_max, V_vocab] bool
    pointer_mask: torch.Tensor  # [n, L_max, T_enc] bool


@dataclass(frozen=True)
class TextInferenceReply:
    """Per-actor inference reply for the decoder-shaped IMPALA pipeline.

    The actor uses ``host_decoder`` to step its envs via
    ``mage.batch_step_by_decoder_action`` and to populate transcripts /
    record_step. ``decoder_batch`` is the GPU-resident form, kept around
    for staging clones into the replay buffer. ``packed_rows`` carries the
    per-env encoded snapshots needed at episode-commit time for replay
    scoring.
    """

    decoder_batch: NativeTextDecoderBatch
    host_decoder: DecoderHostView
    packed_rows: list[PackedTextBatch]
    perspective_player_indices: list[int]
    # Per-row LSTM state captured before encode_with_history advanced it.
    # Replay scoring re-runs the encoder under this state to match sample-time
    # log-π for PPO/R-NaD importance ratios. Layout: [layers, n, hidden].
    lstm_h_in: torch.Tensor | None = None
    lstm_c_in: torch.Tensor | None = None
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
    if not all(b.seq_lengths_host is not None for b in batches):
        raise ValueError("packed batch concat requires seq_lengths_host")
    seq_lengths_host = tuple(n for b in batches for n in cast(tuple[int, ...], b.seq_lengths_host))

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

    max_anchors = max((int(b.pointer_anchor_positions.shape[1]) for b in batches), default=0)

    def _pad_anchor_2d(name: str, *, fill: int = -1) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for b in batches:
            t = cast(torch.Tensor, getattr(b, name)).to(torch.int32)
            rows = int(t.shape[0])
            out = torch.full((rows, max_anchors), fill, dtype=torch.int32, device=t.device)
            cols = int(t.shape[1])
            if cols > 0:
                out[:, :cols] = t[:, :cols]
            parts.append(out)
        return torch.cat(parts, dim=0)

    legal_edge_bitmap: torch.Tensor | None = None
    if any(b.legal_edge_bitmap is not None for b in batches):
        max_blockers = max(
            (int(b.legal_edge_bitmap.shape[1]) for b in batches if b.legal_edge_bitmap is not None),
            default=0,
        )
        max_attackers = max(
            (int(b.legal_edge_bitmap.shape[2]) for b in batches if b.legal_edge_bitmap is not None),
            default=0,
        )
        parts: list[torch.Tensor] = []
        for b in batches:
            rows = int(b.seq_lengths.shape[0])
            device = b.seq_lengths.device
            out = torch.zeros((rows, max_blockers, max_attackers), dtype=torch.bool, device=device)
            if b.legal_edge_bitmap is not None:
                bk = int(b.legal_edge_bitmap.shape[1])
                ak = int(b.legal_edge_bitmap.shape[2])
                if bk > 0 and ak > 0:
                    out[:, :bk, :ak] = b.legal_edge_bitmap
            parts.append(out)
        legal_edge_bitmap = torch.cat(parts, dim=0)

    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu_seqlens,
        seq_lengths=seq_lengths,
        state_positions=state_positions,
        card_ref_positions=torch.cat(
            [b.card_ref_positions.to(torch.int32) for b in batches], dim=0
        ),
        total_tokens=int(token_ids.shape[0]),
        seq_lengths_host=seq_lengths_host,
        spec_lens=torch.cat([b.spec_lens.to(torch.int32) for b in batches], dim=0),
        decision_type=torch.cat([b.decision_type.to(torch.int32) for b in batches], dim=0),
        pointer_anchor_positions=_pad_anchor_2d("pointer_anchor_positions"),
        pointer_anchor_kinds=_pad_anchor_2d("pointer_anchor_kinds"),
        pointer_anchor_subjects=_pad_anchor_2d("pointer_anchor_subjects"),
        pointer_anchor_handles=_pad_anchor_2d("pointer_anchor_handles"),
        legal_edge_bitmap=legal_edge_bitmap,
        max_seqlen=max(seq_lengths_host, default=0),
    )


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
    if seq_lengths_host is None:
        raise ValueError("packed batch slicing requires seq_lengths_host")
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
        # card_ref / anchor positions are row-local end-to-end; slicing
        # the row range is enough — no token_start shift needed.
        card_ref_positions=batch.card_ref_positions[row_start:row_end],
        total_tokens=total_tokens,
        seq_lengths_host=seq_lengths_host,
        spec_lens=batch.spec_lens[row_start:row_end],
        decision_type=batch.decision_type[row_start:row_end],
        pointer_anchor_positions=batch.pointer_anchor_positions[row_start:row_end],
        pointer_anchor_kinds=batch.pointer_anchor_kinds[row_start:row_end],
        pointer_anchor_subjects=batch.pointer_anchor_subjects[row_start:row_end],
        pointer_anchor_handles=batch.pointer_anchor_handles[row_start:row_end],
        legal_edge_bitmap=(
            batch.legal_edge_bitmap[row_start:row_end]
            if batch.legal_edge_bitmap is not None
            else None
        ),
        max_seqlen=max(seq_lengths_host, default=0) if seq_lengths_host is not None else None,
    )


# The protocol the server calls. Concretely this is the decoder-pipeline
# forward path on ``LSTMStatefulTextPolicy`` but parameterizing keeps the server
# testable against fakes.
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
                raise ValueError("inference requests require packed seq_lengths_host")
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
                blank_cols=int(packed.pointer_anchor_positions.shape[1]),
                legal_cols=0,
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

    def wait_for_release(self, timeout: float) -> None:
        """Block briefly until the ring's state changes.

        Wakes on any item finish, publish, flush, or close. Callers re-check
        their condition after waking; this is a notification primitive, not a
        capacity check. Bounded by ``timeout`` so a stuck server can't park
        the caller forever.
        """

        with self._cond:
            if self._closed:
                return
            self._cond.wait(timeout=timeout)

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
            seq_lengths_host=seq_lengths_host,
            spec_lens=self._packed_arena["spec_lens"],
            decision_type=self._packed_arena["decision_type"],
            pointer_anchor_positions=self._packed_arena["pointer_anchor_positions"],
            pointer_anchor_kinds=self._packed_arena["pointer_anchor_kinds"],
            pointer_anchor_subjects=self._packed_arena["pointer_anchor_subjects"],
            pointer_anchor_handles=self._packed_arena["pointer_anchor_handles"],
            legal_edge_bitmap=None,
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
        anchor_cols = max((item.blank_cols for item in items), default=0)
        sliced.pointer_anchor_positions = sliced.pointer_anchor_positions[:, :anchor_cols]
        sliced.pointer_anchor_kinds = sliced.pointer_anchor_kinds[:, :anchor_cols]
        sliced.pointer_anchor_subjects = sliced.pointer_anchor_subjects[:, :anchor_cols]
        sliced.pointer_anchor_handles = sliced.pointer_anchor_handles[:, :anchor_cols]
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
            anchor_capacity = max(64, int(packed.pointer_anchor_positions.shape[1]))
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
                "spec_lens": _arena_empty(
                    self.capacity_rows,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "decision_type": _arena_empty(
                    self.capacity_rows,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "pointer_anchor_positions": _arena_empty(
                    self.capacity_rows,
                    anchor_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "pointer_anchor_kinds": _arena_empty(
                    self.capacity_rows,
                    anchor_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "pointer_anchor_subjects": _arena_empty(
                    self.capacity_rows,
                    anchor_capacity,
                    dtype=torch.int32,
                    device=self.arena_device,
                ),
                "pointer_anchor_handles": _arena_empty(
                    self.capacity_rows,
                    anchor_capacity,
                    dtype=torch.int32,
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
        # card_ref / pointer_anchor positions are row-local end-to-end —
        # no per-row offset add when staging into the arena.
        packed_arena["card_ref_positions"][row_start:row_end].copy_(
            packed.card_ref_positions.to(torch.int32)
        )

        packed_arena["spec_lens"][row_start:row_end].copy_(packed.spec_lens.to(torch.int32))
        packed_arena["decision_type"][row_start:row_end].copy_(packed.decision_type.to(torch.int32))
        self._copy_blank_2d(
            name="pointer_anchor_positions",
            src=packed.pointer_anchor_positions.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=-1,
        )
        self._copy_blank_2d(
            name="pointer_anchor_kinds",
            src=packed.pointer_anchor_kinds.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=-1,
        )
        self._copy_blank_2d(
            name="pointer_anchor_subjects",
            src=packed.pointer_anchor_subjects.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=-1,
        )
        self._copy_blank_2d(
            name="pointer_anchor_handles",
            src=packed.pointer_anchor_handles.to(torch.int32),
            packed_arena=packed_arena,
            row_start=row_start,
            row_end=row_end,
            fill=-1,
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
        sampling_policy: Any,  # LSTMStatefulTextPolicy-compatible
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

    def wait_for_capacity(self, timeout: float) -> None:
        """Park the caller until the ring's state changes or ``timeout`` elapses.

        Used by actor producers when ``try_submit`` returned ``None``. Wakes on
        any release/publish/flush/close so the caller can retry promptly.
        """

        self._queue.wait_for_release(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive() and not self._stop_event.is_set()

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
        merged_packed = self._queue.packed_batch_for(items)
        env_indices: list[int] = [i for it in items for i in it.env_indices]
        perspective: list[int] = [p for it in items for p in it.perspective_player_indices]
        if self._timing is not None:
            self._timing.add("server_concat", time.perf_counter() - start)
            seq_lengths_host = merged_packed.seq_lengths_host
            if seq_lengths_host is None:
                raise ValueError("merged packed batch missing seq_lengths_host")
            max_seq = max(seq_lengths_host, default=0)
            self._timing.add_batch(
                rows=len(env_indices),
                tokens=int(merged_packed.token_ids.shape[0]),
                max_seq=max_seq,
            )

        start = time.perf_counter()
        policy_version = 0
        with torch.no_grad():
            if self._policy_version_manager is not None:
                with self._policy_version_manager.acquire_inference_policy() as (
                    policy,
                    policy_version,
                ):
                    decoder_batch, h_in_replay, c_in_replay = self._sample_decoder(
                        policy, merged_packed, env_indices, perspective
                    )
            else:
                decoder_batch, h_in_replay, c_in_replay = self._sample_decoder(
                    self._policy, merged_packed, env_indices, perspective
                )
        if self._timing is not None:
            self._timing.add("server_sample", time.perf_counter() - start)

        ready_event: Any | None = None
        if decoder_batch.value.device.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(decoder_batch.value.device))

        # Stage host copies of the full decoder batch + LSTM state in a
        # single D→H pass per merged batch. Replaces N actors × many
        # ``.cpu()`` syncs and lets the actors stage rows for replay
        # without holding GPU clones.
        h_in_host = h_in_replay.detach().cpu() if h_in_replay is not None else None
        c_in_host = c_in_replay.detach().cpu() if c_in_replay is not None else None
        d = decoder_batch
        log_probs_host = d.log_probs.detach().cpu()
        is_pointer_u8 = d.output_is_pointer.to(dtype=torch.uint8).detach().cpu()
        host_decoder_full = DecoderHostView(
            decision_type=d.decision_type.detach().cpu(),
            output_token_ids=d.output_token_ids.detach().cpu(),
            output_pointer_pos=d.output_pointer_pos.detach().cpu(),
            output_pointer_subjects=d.output_pointer_subjects.detach().cpu(),
            output_is_pointer=is_pointer_u8.bool(),
            output_is_pointer_u8=is_pointer_u8,
            output_pad_mask=d.output_pad_mask.detach().cpu(),
            output_lens=d.output_lens.detach().cpu(),
            pointer_anchor_handles=d.pointer_anchor_handles.detach().cpu(),
            pointer_anchor_count=d.pointer_anchor_count.detach().cpu(),
            log_probs=log_probs_host,
            log_probs_sum=log_probs_host.sum(dim=-1),
            value=d.value.detach().cpu(),
            vocab_mask=d.vocab_mask.detach().cpu(),
            pointer_mask=d.pointer_mask.detach().cpu(),
        )

        # Scatter per-actor decoder-batch slices and per-row encoded snapshots.
        start = time.perf_counter()
        row_cursor = 0
        token_cursor = 0
        seq_lengths_host = merged_packed.seq_lengths_host
        if seq_lengths_host is None:
            raise ValueError("merged packed batch missing seq_lengths_host")
        for it in items:
            n = len(it.env_indices)
            row_end = row_cursor + n
            row_lengths = [int(x) for x in seq_lengths_host[row_cursor:row_end]]
            packed_rows: list[PackedTextBatch] = []
            tcur = token_cursor
            for n_tokens in row_lengths:
                tend = tcur + n_tokens
                packed_rows.append(
                    _slice_packed_text_batch(
                        merged_packed,
                        row_start=row_cursor + len(packed_rows),
                        row_end=row_cursor + len(packed_rows) + 1,
                        token_start=tcur,
                        token_end=tend,
                    )
                )
                tcur = tend
            actor_decoder = decoder_batch[row_cursor:row_end]
            # LSTM state is host-side now; per-actor slice is a free view.
            actor_h_in = (
                h_in_host[:, row_cursor:row_end].contiguous() if h_in_host is not None else None
            )
            actor_c_in = (
                c_in_host[:, row_cursor:row_end].contiguous() if c_in_host is not None else None
            )
            sl = slice(row_cursor, row_end)
            actor_host_decoder = DecoderHostView(
                decision_type=host_decoder_full.decision_type[sl],
                output_token_ids=host_decoder_full.output_token_ids[sl],
                output_pointer_pos=host_decoder_full.output_pointer_pos[sl],
                output_pointer_subjects=host_decoder_full.output_pointer_subjects[sl],
                output_is_pointer=host_decoder_full.output_is_pointer[sl],
                output_is_pointer_u8=host_decoder_full.output_is_pointer_u8[sl],
                output_pad_mask=host_decoder_full.output_pad_mask[sl],
                output_lens=host_decoder_full.output_lens[sl],
                pointer_anchor_handles=host_decoder_full.pointer_anchor_handles[sl],
                pointer_anchor_count=host_decoder_full.pointer_anchor_count[sl],
                log_probs=host_decoder_full.log_probs[sl],
                log_probs_sum=host_decoder_full.log_probs_sum[sl],
                value=host_decoder_full.value[sl],
                vocab_mask=host_decoder_full.vocab_mask[sl],
                pointer_mask=host_decoder_full.pointer_mask[sl],
            )
            reply = TextInferenceReply(
                decoder_batch=actor_decoder,
                host_decoder=actor_host_decoder,
                packed_rows=packed_rows,
                perspective_player_indices=list(perspective[row_cursor:row_end]),
                lstm_h_in=actor_h_in,
                lstm_c_in=actor_c_in,
                inference_policy_version=int(policy_version),
                ready_event=ready_event,
                release_item=lambda item=it: self._queue.finish_items([item]),
            )
            row_cursor = row_end
            token_cursor += sum(row_lengths)
            it.future.set_result(reply)
        if self._timing is not None:
            self._timing.add("server_scatter", time.perf_counter() - start)

    def _sample_decoder(
        self,
        policy: Any,
        merged_packed: PackedTextBatch,
        env_indices: Sequence[int],
        perspective_player_indices: Sequence[int],
    ) -> tuple[NativeTextDecoderBatch, torch.Tensor | None, torch.Tensor | None]:
        """Run encode_with_history + decoder_sample on the merged packed batch.

        Per-env LSTM state is fetched from the acquired ``policy``'s live state,
        injected into the encoder via the additive HIST embedding, advanced
        by one step, then scattered back. Train-time and sample-time share
        the identical encode-with-history path.
        """

        text_policy = policy.policy.text_policy if hasattr(policy, "policy") else policy.text_policy
        recurrent_policy = policy.policy if hasattr(policy, "policy") else None
        h_in_replay: torch.Tensor | None = None
        c_in_replay: torch.Tensor | None = None
        if recurrent_policy is not None:
            h_in, c_in = policy.lstm_env_state_inputs(env_indices, perspective_player_indices)
            # Capture the pre-update state so the actor can pin it onto each
            # staged replay row; replay scoring re-runs encode_with_history
            # under the same recurrent input to match sample-time log-π.
            h_in_replay = h_in.detach().clone()
            c_in_replay = c_in.detach().clone()
            encoded_snaps, h_out, c_out = recurrent_policy.encode_with_history(
                merged_packed, h_in=h_in, c_in=c_in
            )
            policy.scatter_lstm_env_states(env_indices, perspective_player_indices, h_out, c_out)
        else:
            encoded_snaps = text_policy.encode_packed_only(merged_packed)
        # encoder.forward_packed returns rank-2 [T_packed, D]; decoder cross-attn
        # needs rank-3 [B, T_max, D]. After unpacking, pointer anchor positions
        # must be row-local to index into the padded encoded tensor — strip
        # the per-row packed offset.
        encoded, attn_mask = scatter_packed_to_padded(encoded_snaps.encoded, merged_packed)
        device = encoded.device
        # ``pointer_anchor_positions`` is row-local end-to-end (see
        # native_assembler.to_packed_text_batch); the decoder consumes it
        # directly as an index into the [B, T_max, D] padded tensor.
        anchor_positions_rowlocal = merged_packed.pointer_anchor_positions

        sample = decoder_sample(
            text_policy,
            encoded,
            attn_mask,
            merged_packed.decision_type.to(device=device, dtype=torch.long),
            anchor_positions_rowlocal.to(device=device, dtype=torch.long),
            merged_packed.pointer_anchor_kinds.to(device=device, dtype=torch.long),
            merged_packed.pointer_anchor_subjects.to(device=device, dtype=torch.long),
            merged_packed.pointer_anchor_handles.to(device=device, dtype=torch.long),
            legal_edge_bitmap=merged_packed.legal_edge_bitmap,
            greedy=self._deterministic,
        )
        # ``ValueHead.forward`` already squeezes the trailing 1-dim axis;
        # an extra squeeze(-1) collapses ``[B=1]`` to a 0-dim tensor and
        # breaks the per-actor slicing of ``decoder_batch.value`` below.
        value = text_policy.run_heads(encoded_snaps)
        return (
            native_decoder_batch_from_sample(sample, value=value),
            h_in_replay,
            c_in_replay,
        )


__all__ = [
    "DecoderHostView",
    "TextInferenceRequest",
    "TextInferenceReply",
    "TextInferenceServer",
    "RolloutTimingStats",
    "_concat_packed_text_batches",
]
