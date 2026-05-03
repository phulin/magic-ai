"""IMPALA-style inference server for the text-encoder rollout path.

A single background thread owns the GPU policy: it drains encoded-batch
requests submitted by N actor threads, dynamically batches them into one
forward pass, calls ``TextActorCritic.sample_native_tensor_batch`` on the
merged batch, stages the resulting replay payload into the shared
``NativeTextTrajectoryBuffer``, and scatters per-request host-side scalars
(decision counts, selected choices, may-bit, log-prob, value, trace-kind)
back to the actors via per-request ``Future`` objects.

All concat / scatter operations are vectorized — no Python ``for`` loops over
tensor rows. The server is the only thread that touches the policy's
parameters or its live LSTM env-state cache during normal rollout, so no
GPU lock is required. ``pause()`` / ``resume()`` quiesce the server for the
PPO/R-NaD update window or for snapshot eval.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, cast

import torch

from magic_ai.native.sharded import _concat_encoded_batches
from magic_ai.text_encoder.batch import PackedTextBatch

if TYPE_CHECKING:
    from magic_ai.slot_encoder.native_encoder import NativeEncodedBatch


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


def _concat_packed_text_batches(batches: list[PackedTextBatch]) -> PackedTextBatch:
    """Concatenate ``PackedTextBatch`` objects along the row axis.

    Vectorized: every per-shard offset (token, batch, anchor) is rolled into
    one tensor op. ``-1`` sentinels in anchor positions are preserved.
    """

    if len(batches) == 1:
        return batches[0]

    seq_lengths = torch.cat([b.seq_lengths for b in batches], dim=0)
    token_ids = torch.cat([b.token_ids for b in batches], dim=0)

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

    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu_seqlens,
        seq_lengths=seq_lengths,
        state_positions=state_positions,
        card_ref_positions=_shift_anchor("card_ref_positions"),
        option_positions=_shift_anchor("option_positions"),
        option_mask=torch.cat([b.option_mask for b in batches], dim=0),
        target_positions=_shift_anchor("target_positions"),
        target_mask=torch.cat([b.target_mask for b in batches], dim=0),
    )


# The protocol the server calls. Concretely this is
# ``TextActorCritic.sample_native_tensor_batch`` but parameterizing keeps the
# server testable against fakes.
ForwardCallable = Any


@dataclass
class _PendingItem:
    request: TextInferenceRequest
    future: Future[TextInferenceReply]


class TextInferenceServer:
    """Dynamic-batching inference server for the native text rollout path.

    Submitting actors block on their per-request ``Future``; the server thread
    drains the queue (waiting up to ``max_wait_ms`` for additional requests
    once at least one is in hand), merges them into a single GPU batch, runs
    the policy forward, stages the replay payload, and resolves each future
    with a sliced ``TextInferenceReply``.
    """

    def __init__(
        self,
        *,
        sampling_policy: Any,  # TextActorCritic-compatible
        staging_buffer: Any,
        max_batch: int,
        max_wait_ms: float,
        deterministic: bool = False,
        name: str = "text-inference",
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self._policy = sampling_policy
        self._staging = staging_buffer
        self._max_batch = int(max_batch)
        self._max_wait_s = float(max_wait_ms) / 1000.0
        self._deterministic = bool(deterministic)
        self._queue: Queue[_PendingItem | None] = Queue()
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
        self._queue.put(_PendingItem(request=request, future=fut2))
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

    def stop(self) -> None:
        with self._cond:
            self._stop_event.set()
            self._paused = False
            self._cond.notify_all()
        self._queue.put(None)
        self._thread.join(timeout=10.0)
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
                    return
                self._idle = False
            try:
                try:
                    self._process(batch)
                except BaseException as exc:  # noqa: BLE001
                    self._exc = exc
                    # Resolve only the in-flight batch's futures; do NOT
                    # tear down the server. Pending submits are independent.
                    self._fail_items(batch, exc)
            finally:
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

        try:
            while True:
                item = self._queue.get_nowait()
                if item is None:
                    continue
                if not item.future.done():
                    item.future.set_exception(exc)
        except Empty:
            pass
        with self._cond:
            self._idle = True
            self._cond.notify_all()

    def _collect_batch(self) -> list[_PendingItem] | None:
        """Block until at least one item is available, then opportunistically
        coalesce up to max_batch within max_wait_ms.
        """
        try:
            first = self._queue.get(timeout=0.1)
        except Empty:
            return []
        if first is None:
            return None
        items: list[_PendingItem] = [first]
        rows = len(first.request.env_indices)
        deadline = None
        if self._max_wait_s > 0.0 and rows < self._max_batch:
            import time as _time

            deadline = _time.monotonic() + self._max_wait_s
        while rows < self._max_batch:
            timeout = None
            if deadline is not None:
                import time as _time

                timeout = max(0.0, deadline - _time.monotonic())
                if timeout == 0.0:
                    break
            try:
                nxt = self._queue.get(timeout=timeout) if timeout else self._queue.get_nowait()
            except Empty:
                break
            if nxt is None:
                # Stop sentinel encountered mid-coalesce. Put it back and stop
                # collecting — the next loop iteration will see stop_event.
                self._queue.put(None)
                break
            items.append(nxt)
            rows += len(nxt.request.env_indices)
        return items

    def _process(self, items: list[_PendingItem]) -> None:
        merged_native = _concat_encoded_batches(
            [cast("NativeEncodedBatch", it.request.native_batch) for it in items]
        )
        merged_packed = _concat_packed_text_batches([it.request.packed_batch for it in items])
        env_indices: list[int] = [i for it in items for i in it.request.env_indices]
        perspective: list[int] = [p for it in items for p in it.request.perspective_player_indices]

        with torch.no_grad():
            sample = self._policy.sample_native_tensor_batch(
                native_batch=merged_native,
                env_indices=env_indices,
                perspective_player_indices=perspective,
                packed_batch=merged_packed,
                deterministic=self._deterministic,
                append_replay=False,
                return_replay_payload=True,
            )
        if sample.replay_payload is None:
            raise RuntimeError("inference server expected a replay payload")

        # Stage the merged payload directly. Single-threaded since the server
        # owns the staging buffer during normal operation.
        self._staging.stage_batch(env_indices, sample.replay_payload)

        # Scatter per-request host-side slices.
        row_cursor = 0
        col_cursor = 0
        decision_counts = sample.decision_counts
        sel_cols = sample.selected_choice_cols
        may_sel = sample.may_selected
        log_prob = sample.old_log_prob
        value = sample.value
        trace_kind_ids = merged_native.trace_kind_id.tolist()
        for it in items:
            n = len(it.request.env_indices)
            row_end = row_cursor + n
            req_counts = decision_counts[row_cursor:row_end]
            req_cols_total = sum(req_counts)
            reply = TextInferenceReply(
                decision_counts=list(req_counts),
                selected_choice_cols=list(sel_cols[col_cursor : col_cursor + req_cols_total]),
                may_selected=list(may_sel[row_cursor:row_end]),
                old_log_prob=list(log_prob[row_cursor:row_end]),
                value=list(value[row_cursor:row_end]),
                trace_kind_id=list(trace_kind_ids[row_cursor:row_end]),
            )
            row_cursor = row_end
            col_cursor += req_cols_total
            it.future.set_result(reply)


__all__ = [
    "TextInferenceRequest",
    "TextInferenceReply",
    "TextInferenceServer",
    "_concat_packed_text_batches",
]
