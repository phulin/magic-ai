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
from dataclasses import dataclass
from queue import Empty
from typing import Any, cast

import torch

from magic_ai.text_encoder.batch import PackedTextBatch
from magic_ai.text_encoder.decoder_batch import NativeTextDecoderBatch
from magic_ai.text_encoder.inference_pipeline import TextInferencePipeline

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


class _HeadItemDoesNotFit(Exception):
    pass


@dataclass(frozen=True)
class TextInferenceRequest:
    """One actor's encoded batch + per-row routing metadata."""

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

    The actor reads everything from ``host_decoder``: decoder fields
    needed for cgo step, transcript bookkeeping, and per-env staging into
    the replay buffer. Per-env encoded snapshots live on the actor side —
    the actor saves its pre-submit host packed batch in the in-flight
    record.
    """

    host_decoder: DecoderHostView
    perspective_player_indices: list[int]
    # Per-row LSTM state captured before encode_with_history advanced it.
    # Replay scoring re-runs the encoder under this state to match sample-time
    # log-π for PPO/R-NaD importance ratios. Layout: [layers, n, hidden].
    lstm_h_in: torch.Tensor | None = None
    lstm_c_in: torch.Tensor | None = None
    inference_policy_version: int = 0
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


class _HostPackedArena:
    """Pre-allocated (pinned where available) host buffers for the merged
    packed batch.

    Replaces ``_concat_packed_text_batches``'s per-merge ``torch.cat`` +
    ``torch.full`` allocation pattern with a single set of long-lived
    buffers that the merge writes into in place. Buffers grow on demand
    (doubling) and are kept across merges so steady-state inference
    allocates nothing.

    The arena holds buffers for the *next* H→D's input. The server does
    a blocking ``host.to(device)`` after the merge so reuse on the next
    merge is safe; Phase C will switch to ``non_blocking=True`` and a
    second slot to keep the H→D async.
    """

    def __init__(self, *, use_pinned: bool) -> None:
        self._use_pinned = bool(use_pinned)
        self._max_rows = 0
        self._max_tokens = 0
        self._max_anchors = 0
        self._max_blockers = 0
        self._max_attackers = 0
        self._has_bitmap = False
        # Allocated lazily on first merge.
        self.token_ids: torch.Tensor | None = None
        self.seq_id: torch.Tensor | None = None
        self.pos_in_seq: torch.Tensor | None = None
        self.cu_seqlens: torch.Tensor | None = None
        self.seq_lengths: torch.Tensor | None = None
        self.state_positions: torch.Tensor | None = None
        self.card_ref_positions: torch.Tensor | None = None
        self.spec_lens: torch.Tensor | None = None
        self.decision_type: torch.Tensor | None = None
        self.pointer_anchor_positions: torch.Tensor | None = None
        self.pointer_anchor_kinds: torch.Tensor | None = None
        self.pointer_anchor_subjects: torch.Tensor | None = None
        self.pointer_anchor_handles: torch.Tensor | None = None
        self.legal_edge_bitmap: torch.Tensor | None = None

    def _empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        fill: int | bool | None = None,
    ) -> torch.Tensor:
        kw: dict[str, Any] = {"dtype": dtype}
        if self._use_pinned:
            kw["pin_memory"] = True
        if fill is None:
            return torch.empty(shape, **kw)
        return torch.full(shape, fill, **kw)

    def _ensure_capacity(
        self,
        *,
        rows: int,
        tokens: int,
        anchors: int,
        blockers: int,
        attackers: int,
        need_bitmap: bool,
        card_refs: int,
    ) -> None:
        grow_rows = rows > self._max_rows
        grow_tokens = tokens > self._max_tokens
        grow_anchors = anchors > self._max_anchors
        # ``grow_rows`` must also trigger ``grow_bitmap`` because the
        # bitmap's first dim is the arena row capacity — if rows grew but
        # the bitmap didn't, the bitmap view returned in ``merge`` would
        # be shorter (in dim 0) than ``seq_lengths`` for the same merge.
        grow_bitmap = need_bitmap and (
            not self._has_bitmap
            or grow_rows
            or blockers > self._max_blockers
            or attackers > self._max_attackers
        )

        if grow_tokens:
            new_tokens = max(tokens, self._max_tokens * 2, 1024)
            self.token_ids = self._empty((new_tokens,), dtype=torch.int32)
            self.seq_id = self._empty((new_tokens,), dtype=torch.int32)
            self.pos_in_seq = self._empty((new_tokens,), dtype=torch.int32)
            self._max_tokens = new_tokens
        if grow_rows:
            new_rows = max(rows, self._max_rows * 2, 8)
            self.cu_seqlens = self._empty((new_rows + 1,), dtype=torch.int32)
            self.seq_lengths = self._empty((new_rows,), dtype=torch.int32)
            self.state_positions = self._empty((new_rows,), dtype=torch.int32)
            self.spec_lens = self._empty((new_rows,), dtype=torch.int32)
            self.decision_type = self._empty((new_rows,), dtype=torch.int32)
            self._max_rows = new_rows
        if (
            grow_rows
            or self.card_ref_positions is None
            or card_refs > int(self.card_ref_positions.shape[1])
        ):
            cap_refs = max(
                card_refs,
                int(self.card_ref_positions.shape[1]) if self.card_ref_positions is not None else 0,
            )
            self.card_ref_positions = self._empty((self._max_rows, cap_refs), dtype=torch.int32)
        if grow_rows or grow_anchors:
            new_anchors = max(anchors, self._max_anchors * 2, 4)
            shape = (self._max_rows, new_anchors)
            # Sentinel -1 across the whole buffer; per-merge writes only
            # touch the live portion of each row, so columns past each
            # shard's live width stay at -1 (matches concat-with-pad).
            self.pointer_anchor_positions = self._empty(shape, dtype=torch.int32, fill=-1)
            self.pointer_anchor_kinds = self._empty(shape, dtype=torch.int32, fill=-1)
            self.pointer_anchor_subjects = self._empty(shape, dtype=torch.int32, fill=-1)
            self.pointer_anchor_handles = self._empty(shape, dtype=torch.int32, fill=-1)
            self._max_anchors = new_anchors
        if grow_bitmap:
            # Only double the dim that's actually growing; reusing the
            # existing dim for "rows grew but blockers/attackers didn't"
            # avoids reallocating the bitmap any larger than necessary.
            new_blockers = (
                max(blockers, self._max_blockers * 2, 1)
                if blockers > self._max_blockers or self._max_blockers == 0
                else self._max_blockers
            )
            new_attackers = (
                max(attackers, self._max_attackers * 2, 1)
                if attackers > self._max_attackers or self._max_attackers == 0
                else self._max_attackers
            )
            self.legal_edge_bitmap = self._empty(
                (self._max_rows, new_blockers, new_attackers), dtype=torch.bool, fill=False
            )
            self._max_blockers = new_blockers
            self._max_attackers = new_attackers
            self._has_bitmap = True

    def merge(self, batches: list[PackedTextBatch]) -> PackedTextBatch:
        """Write ``batches`` into the arena and return a sliced view.

        The returned ``PackedTextBatch`` shares storage with the arena's
        long-lived buffers — callers must consume it (typically via
        ``.to(device)``) before the next ``merge`` overwrites them.
        """
        if not batches:
            raise ValueError("empty batches")
        # Discover target capacities.
        total_rows = 0
        total_tokens = 0
        max_anchors = 0
        max_blockers = 0
        max_attackers = 0
        need_bitmap = False
        max_card_refs = 0
        for b in batches:
            total_rows += int(b.seq_lengths.shape[0])
            total_tokens += int(b.token_ids.shape[0])
            max_anchors = max(max_anchors, int(b.pointer_anchor_positions.shape[1]))
            max_card_refs = max(max_card_refs, int(b.card_ref_positions.shape[1]))
            if b.legal_edge_bitmap is not None:
                need_bitmap = True
                max_blockers = max(max_blockers, int(b.legal_edge_bitmap.shape[1]))
                max_attackers = max(max_attackers, int(b.legal_edge_bitmap.shape[2]))

        self._ensure_capacity(
            rows=total_rows,
            tokens=total_tokens,
            anchors=max_anchors,
            blockers=max_blockers,
            attackers=max_attackers,
            need_bitmap=need_bitmap,
            card_refs=max_card_refs,
        )

        # The typecheck-asserting helper keeps line-noise out of the loop.
        assert self.token_ids is not None and self.seq_id is not None
        assert self.pos_in_seq is not None and self.cu_seqlens is not None
        assert self.seq_lengths is not None and self.state_positions is not None
        assert self.card_ref_positions is not None and self.spec_lens is not None
        assert self.decision_type is not None
        assert self.pointer_anchor_positions is not None
        assert self.pointer_anchor_kinds is not None
        assert self.pointer_anchor_subjects is not None
        assert self.pointer_anchor_handles is not None

        # First cu_seqlens entry is always 0; per-shard writes fill (row+1:row_end+1).
        self.cu_seqlens[0] = 0

        row_cursor = 0
        token_cursor = 0
        seq_lengths_host: list[int] = []
        for b in batches:
            n_rows = int(b.seq_lengths.shape[0])
            n_tokens = int(b.token_ids.shape[0])
            row_end = row_cursor + n_rows
            token_end = token_cursor + n_tokens

            # Token-axis fields. seq_id needs row_offset shift; the rest copy.
            self.token_ids[token_cursor:token_end].copy_(b.token_ids.to(torch.int32))
            shifted_seq_id = b.seq_id.to(torch.int32) + int(row_cursor)
            self.seq_id[token_cursor:token_end].copy_(shifted_seq_id)
            self.pos_in_seq[token_cursor:token_end].copy_(b.pos_in_seq.to(torch.int32))

            # Row-axis fields. cu_seqlens needs token_offset; state_positions too.
            self.seq_lengths[row_cursor:row_end].copy_(b.seq_lengths.to(torch.int32))
            # b.cu_seqlens is [n_rows + 1]; we want [n_rows] entries starting from
            # cu_seqlens[1:] shifted by token_cursor, written into
            # arena.cu_seqlens[row_cursor + 1 : row_end + 1].
            self.cu_seqlens[row_cursor + 1 : row_end + 1].copy_(
                b.cu_seqlens[1:].to(torch.int32) + int(token_cursor)
            )
            self.state_positions[row_cursor:row_end].copy_(
                b.state_positions.to(torch.int32) + int(token_cursor)
            )
            self.spec_lens[row_cursor:row_end].copy_(b.spec_lens.to(torch.int32))
            self.decision_type[row_cursor:row_end].copy_(b.decision_type.to(torch.int32))

            # 2-D row-aligned fields. Width may be smaller than arena's; we
            # only write the live portion (rest stays at sentinel from init).
            cref_w = int(b.card_ref_positions.shape[1])
            if cref_w > 0:
                self.card_ref_positions[row_cursor:row_end, :cref_w].copy_(
                    b.card_ref_positions.to(torch.int32)
                )
            # NOTE: arena.card_ref_positions for cols past cref_w may carry
            # stale data from previous merges. _concat fills them with the
            # shard's own values (which would already be -1 if absent). We
            # need to reset past-live cols here to match — see explicit fill.
            if cref_w < int(self.card_ref_positions.shape[1]):
                self.card_ref_positions[row_cursor:row_end, cref_w:].fill_(-1)

            a_w = int(b.pointer_anchor_positions.shape[1])
            if a_w > 0:
                self.pointer_anchor_positions[row_cursor:row_end, :a_w].copy_(
                    b.pointer_anchor_positions.to(torch.int32)
                )
                self.pointer_anchor_kinds[row_cursor:row_end, :a_w].copy_(
                    b.pointer_anchor_kinds.to(torch.int32)
                )
                self.pointer_anchor_subjects[row_cursor:row_end, :a_w].copy_(
                    b.pointer_anchor_subjects.to(torch.int32)
                )
                self.pointer_anchor_handles[row_cursor:row_end, :a_w].copy_(
                    b.pointer_anchor_handles.to(torch.int32)
                )
            if a_w < self._max_anchors:
                # Reset stale cols to -1 sentinel for this shard's rows.
                self.pointer_anchor_positions[row_cursor:row_end, a_w:].fill_(-1)
                self.pointer_anchor_kinds[row_cursor:row_end, a_w:].fill_(-1)
                self.pointer_anchor_subjects[row_cursor:row_end, a_w:].fill_(-1)
                self.pointer_anchor_handles[row_cursor:row_end, a_w:].fill_(-1)

            if need_bitmap:
                assert self.legal_edge_bitmap is not None
                if b.legal_edge_bitmap is not None:
                    bk = int(b.legal_edge_bitmap.shape[1])
                    ak = int(b.legal_edge_bitmap.shape[2])
                    if bk > 0 and ak > 0:
                        self.legal_edge_bitmap[row_cursor:row_end, :bk, :ak].copy_(
                            b.legal_edge_bitmap
                        )
                    # Zero stale cells outside this shard's live region.
                    if bk < self._max_blockers:
                        self.legal_edge_bitmap[row_cursor:row_end, bk:, :].fill_(False)
                    if ak < self._max_attackers:
                        self.legal_edge_bitmap[row_cursor:row_end, :bk, ak:].fill_(False)
                else:
                    self.legal_edge_bitmap[row_cursor:row_end].fill_(False)

            sl_host = b.seq_lengths_host
            if sl_host is None:
                raise ValueError("packed batch arena merge requires seq_lengths_host")
            seq_lengths_host.extend(int(x) for x in sl_host)

            row_cursor = row_end
            token_cursor = token_end

        cref_view_w = int(self.card_ref_positions.shape[1])
        bitmap_view: torch.Tensor | None = None
        if need_bitmap:
            assert self.legal_edge_bitmap is not None
            bitmap_view = self.legal_edge_bitmap[:total_rows, :max_blockers, :max_attackers]

        return PackedTextBatch(
            token_ids=self.token_ids[:total_tokens],
            seq_id=self.seq_id[:total_tokens],
            pos_in_seq=self.pos_in_seq[:total_tokens],
            cu_seqlens=self.cu_seqlens[: total_rows + 1],
            seq_lengths=self.seq_lengths[:total_rows],
            state_positions=self.state_positions[:total_rows],
            card_ref_positions=self.card_ref_positions[:total_rows, :cref_view_w],
            spec_lens=self.spec_lens[:total_rows],
            decision_type=self.decision_type[:total_rows],
            pointer_anchor_positions=self.pointer_anchor_positions[:total_rows, :max_anchors],
            pointer_anchor_kinds=self.pointer_anchor_kinds[:total_rows, :max_anchors],
            pointer_anchor_subjects=self.pointer_anchor_subjects[:total_rows, :max_anchors],
            pointer_anchor_handles=self.pointer_anchor_handles[:total_rows, :max_anchors],
            legal_edge_bitmap=bitmap_view,
            total_tokens=total_tokens,
            seq_lengths_host=tuple(seq_lengths_host),
            max_seqlen=max(seq_lengths_host, default=0),
        )


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


# The protocol the server calls. Concretely this is the decoder-pipeline
# forward path on ``LSTMStatefulTextPolicy`` but parameterizing keeps the server
# testable against fakes.
ForwardCallable = Any


@dataclass
class _PendingItem:
    """One actor's submitted batch waiting on the inference queue.

    The ring holds the item by reference (no arena copy). Per-actor host
    tensors live inside ``request.packed_batch``; the dispatch loop
    concatenates them on host right before the single ``merged.to(device)``
    H→D that feeds the policy forward.
    """

    future: Future[TextInferenceReply]
    request: TextInferenceRequest
    rows: int
    seq_lengths_host: tuple[int, ...]
    released: bool = False

    @property
    def env_indices(self) -> list[int]:
        return self.request.env_indices

    @property
    def perspective_player_indices(self) -> list[int]:
        return self.request.perspective_player_indices


class _InferenceWorkRing:
    """Thread-safe bounded FIFO of pending inference items.

    Producers (actors) push :class:`_PendingItem` records carrying their
    own pinned host tensors; the server thread pops one or more items per
    dispatch group, merges them on host, then issues a single H→D for the
    forward pass. Capacity is in rows: the queue stops accepting new items
    once the total queued row count would exceed ``capacity_rows``,
    bounding pinned-host pressure across actors.
    """

    def __init__(self, *, capacity_rows: int) -> None:
        if capacity_rows < 1:
            raise ValueError("capacity_rows must be >= 1")
        self.capacity_rows = int(capacity_rows)
        self._cond = threading.Condition()
        self._items: list[_PendingItem | None] = []
        self._queued_rows = 0
        self._active_rows = 0
        self._waiting_producers = 0
        self._closed = False
        self._force_flush = False

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
        seq_lengths_host = packed.seq_lengths_host
        if seq_lengths_host is None:
            raise ValueError("inference requests require packed seq_lengths_host")
        with self._cond:
            while not self._closed:
                if self._queued_rows + self._active_rows + rows <= self.capacity_rows:
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
            item = _PendingItem(
                future=future,
                request=request,
                rows=rows,
                seq_lengths_host=seq_lengths_host,
            )
            self._items.append(item)
            self._queued_rows += rows
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
                self._queued_rows -= item.rows
                self._active_rows += item.rows
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
            self._queued_rows -= item.rows
            self._active_rows += item.rows
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
                        self._queued_rows -= item.rows
                        self._active_rows += item.rows
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
            self._queued_rows = 0
            self._cond.notify_all()
            return items

    def qsize(self) -> int:
        with self._cond:
            return len([item for item in self._items if item is not None])

    def queued_rows(self) -> int:
        with self._cond:
            return self._queued_rows

    def has_tail_capacity_for_another_item(self) -> bool:
        with self._cond:
            return self._queued_rows + self._active_rows < self.capacity_rows

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
            self._cond.notify_all()


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
        bucketed_inference: bool = False,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        if min_batch_rows < 1:
            raise ValueError("min_batch_rows must be >= 1")
        self._policy = sampling_policy
        self._policy_version_manager = (
            sampling_policy if hasattr(sampling_policy, "acquire_inference_policy") else None
        )
        self._device = _infer_policy_device(sampling_policy)
        self._max_batch = int(max_batch)
        self._min_batch_rows = min(int(min_batch_rows), int(max_batch))
        self._deterministic = bool(deterministic)
        # Phase C/F compile knobs are off by default; the caller opts in via
        # ``bucketed_inference=True`` (typically when ``--torch-compile`` is
        # set so the policy stack's eager fallbacks line up).
        self._pipeline = TextInferencePipeline(
            deterministic=self._deterministic,
            bucketed=bucketed_inference and self._device.type == "cuda",
            compile_decoder=bucketed_inference and self._device.type == "cuda",
        )
        self._arena = _HostPackedArena(use_pinned=self._device.type == "cuda")
        self._timing = timing_stats
        self._queue = _InferenceWorkRing(
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
        # Merge actor batches on host first via the long-lived arena (no
        # per-merge allocs), then a single H→D transfer feeds the forward.
        host_packed = self._arena.merge([it.request.packed_batch for it in items])
        env_indices: list[int] = [i for it in items for i in it.env_indices]
        perspective: list[int] = [p for it in items for p in it.perspective_player_indices]
        if self._timing is not None:
            self._timing.add("server_concat", time.perf_counter() - start)
            seq_lengths_host = host_packed.seq_lengths_host
            if seq_lengths_host is None:
                raise ValueError("merged packed batch missing seq_lengths_host")
            max_seq = max(seq_lengths_host, default=0)
            self._timing.add_batch(
                rows=len(env_indices),
                tokens=int(host_packed.token_ids.shape[0]),
                max_seq=max_seq,
            )

        # Hand the host batch to the pipeline; bucketed Phase C wants to
        # pad on host before the H→D, and the non-bucketed path moves
        # to device internally.
        start = time.perf_counter()
        policy_version = 0
        with torch.no_grad():
            if self._policy_version_manager is not None:
                with self._policy_version_manager.acquire_inference_policy() as (
                    policy,
                    policy_version,
                ):
                    decoder_batch, h_in_replay, c_in_replay = self._sample_decoder(
                        policy, host_packed, env_indices, perspective
                    )
            else:
                decoder_batch, h_in_replay, c_in_replay = self._sample_decoder(
                    self._policy, host_packed, env_indices, perspective
                )
        if self._timing is not None:
            self._timing.add("server_sample", time.perf_counter() - start)

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

        # Scatter per-actor decoder-batch slices. The actor uses its
        # pre-submit host packed_batch (saved in the in-flight record) for
        # staging — no need to slice merged_packed per-actor here.
        start = time.perf_counter()
        row_cursor = 0
        seq_lengths_host = host_packed.seq_lengths_host
        if seq_lengths_host is None:
            raise ValueError("merged packed batch missing seq_lengths_host")
        for it in items:
            n = len(it.env_indices)
            row_end = row_cursor + n
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
                host_decoder=actor_host_decoder,
                perspective_player_indices=list(perspective[row_cursor:row_end]),
                lstm_h_in=actor_h_in,
                lstm_c_in=actor_c_in,
                inference_policy_version=int(policy_version),
                release_item=lambda item=it: self._queue.finish_items([item]),
            )
            row_cursor = row_end
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
        """Fetch env LSTM state, run the pipeline, scatter state back.

        The model forward lives in ``TextInferencePipeline``. The server
        only owns env-state cache plumbing (LSTM h/c per env, captured
        pre-update for replay scoring).
        """
        recurrent_policy = policy.policy if hasattr(policy, "policy") else None
        h_in: torch.Tensor | None = None
        c_in: torch.Tensor | None = None
        h_in_replay: torch.Tensor | None = None
        c_in_replay: torch.Tensor | None = None
        if recurrent_policy is not None:
            h_in, c_in = policy.lstm_env_state_inputs(env_indices, perspective_player_indices)
            # Capture pre-update state for replay scoring: it pins each row's
            # log-π under the same recurrent input the actor saw.
            h_in_replay = h_in.detach().clone()
            c_in_replay = c_in.detach().clone()

        out = self._pipeline.encode_and_sample(policy, merged_packed, h_in, c_in)

        if recurrent_policy is not None and out.h_out is not None and out.c_out is not None:
            policy.scatter_lstm_env_states(
                env_indices, perspective_player_indices, out.h_out, out.c_out
            )
        return out.decoder, h_in_replay, c_in_replay


__all__ = [
    "DecoderHostView",
    "TextInferenceRequest",
    "TextInferenceReply",
    "TextInferenceServer",
    "RolloutTimingStats",
    "_concat_packed_text_batches",
]
