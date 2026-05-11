"""Reusable text-policy inference forward.

``TextInferencePipeline`` packages the encoder + decoder forward used by
the inference server and offline eval (opponent pool, smoke tests). It is
intentionally stateless w.r.t. the policy itself: callers pass the
policy as a per-call argument so the pipeline plays nicely with the
inference server's policy-version manager (the live policy reference
swaps under the server's feet on policy updates).

Phase C adds optional shape bucketing + per-bucket ``torch.compile``
with ``mode="reduce-overhead"``. When enabled, the pipeline:

* Selects the smallest bucket ``(rows, tokens)`` that fits the merged
  batch.
* Pads the host packed batch up to bucket shape with dummy rows holding
  PAD tokens. The dummy rows' encoder outputs are discarded.
* Runs the compiled encoder forward — same shape every call, so
  inductor's reduce-overhead path captures CUDA Graphs.

The decoder autoregressive loop and value head stay eager: their Python
control flow is hard to fold into a single graph. Phase F handles
the decoder separately.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from magic_ai.text_encoder.batch import PackedTextBatch, scatter_packed_to_padded
from magic_ai.text_encoder.decoder_batch import (
    NativeTextDecoderBatch,
    native_decoder_batch_from_sample,
)
from magic_ai.text_encoder.decoder_inference import decoder_sample
from magic_ai.text_encoder.policy import EncodedSnapshots


@dataclass(frozen=True)
class InferenceOutput:
    decoder: NativeTextDecoderBatch
    h_out: Tensor | None
    c_out: Tensor | None


# Default bucket table. ``(R_b, T_b)`` — picked so that
# ``T_b - R_b`` ≥ typical (tokens - rows) of the merged batch. Override
# via ``TextInferencePipeline(buckets=...)``.
DEFAULT_BUCKETS: tuple[tuple[int, int], ...] = (
    (16, 2048),
    (64, 8192),
    (192, 24576),
    (384, 49152),
    (768, 98304),
)

# PAD token id for dummy rows. Any valid token id works — the dummy rows'
# outputs are sliced away before the decoder runs.
_PAD_TOKEN_ID = 0


def _select_bucket(
    rows: int, tokens: int, buckets: tuple[tuple[int, int], ...]
) -> tuple[int, int] | None:
    """Smallest bucket whose row/slack capacity fits the batch.

    A bucket ``(R_b, T_b)`` fits ``(rows, tokens)`` iff there is at least
    one dummy row available (``rows < R_b``) and the surplus tokens fit
    in the dummy rows (``T_b - R_b ≥ tokens - rows``).
    """
    for r, t in buckets:
        if rows < r and (t - r) >= (tokens - rows):
            return (r, t)
    return None


def _pad_packed_to_bucket(
    packed: PackedTextBatch,
    *,
    bucket_rows: int,
    bucket_tokens: int,
) -> PackedTextBatch:
    """Pad ``packed`` up to ``(bucket_rows, bucket_tokens)``.

    Adds ``D = bucket_rows - R`` dummy rows holding ``T_b - T`` PAD
    tokens between them. Each dummy row gets ≥ 1 token so the resulting
    cu_seqlens has no zero-length sequences (flash_attn rejects those).
    """
    real_rows = int(packed.seq_lengths.shape[0])
    real_tokens = int(packed.token_ids.shape[0])
    dummy_rows = bucket_rows - real_rows
    pad_tokens = bucket_tokens - real_tokens
    if dummy_rows < 1 or pad_tokens < dummy_rows:
        raise ValueError(
            f"bucket ({bucket_rows},{bucket_tokens}) cannot fit batch "
            f"({real_rows},{real_tokens}); need ≥1 dummy row and "
            f"pad_tokens ≥ dummy_rows"
        )

    base = pad_tokens // dummy_rows
    extra = pad_tokens % dummy_rows
    dummy_lens = torch.full((dummy_rows,), base, dtype=torch.int32)
    if extra > 0:
        dummy_lens[:extra] += 1

    # Row-axis fields. spec_lens/decision_type/seq_lengths: pad with 0/-1.
    seq_lengths = torch.cat([packed.seq_lengths.to(torch.int32), dummy_lens])
    cu_seqlens = torch.zeros(bucket_rows + 1, dtype=torch.int32)
    cu_seqlens[1:] = seq_lengths.cumsum(0)
    state_positions = cu_seqlens[:bucket_rows].clone()

    # Token-axis fields.
    token_ids = torch.full((bucket_tokens,), _PAD_TOKEN_ID, dtype=torch.int32)
    token_ids[:real_tokens].copy_(packed.token_ids.to(torch.int32))

    seq_id = torch.empty(bucket_tokens, dtype=torch.int32)
    seq_id[:real_tokens].copy_(packed.seq_id.to(torch.int32))
    # Each dummy row's tokens get seq_id = real_rows + i.
    cursor = real_tokens
    for i, n in enumerate(dummy_lens.tolist()):
        seq_id[cursor : cursor + n] = real_rows + i
        cursor += n

    pos_in_seq = torch.empty(bucket_tokens, dtype=torch.int32)
    pos_in_seq[:real_tokens].copy_(packed.pos_in_seq.to(torch.int32))
    cursor = real_tokens
    for n in dummy_lens.tolist():
        pos_in_seq[cursor : cursor + n] = torch.arange(n, dtype=torch.int32)
        cursor += n

    # 2-D row fields: card_ref_positions and pointer_anchor_*. Dummy
    # rows fill with -1 sentinel.
    n_refs = int(packed.card_ref_positions.shape[1])
    card_ref_positions = torch.full((bucket_rows, n_refs), -1, dtype=torch.int32)
    card_ref_positions[:real_rows].copy_(packed.card_ref_positions.to(torch.int32))

    n_anchors = int(packed.pointer_anchor_positions.shape[1])

    def _pad_anchor_2d(t: Tensor) -> Tensor:
        out = torch.full((bucket_rows, n_anchors), -1, dtype=torch.int32)
        if real_rows > 0 and n_anchors > 0:
            out[:real_rows].copy_(t.to(torch.int32))
        return out

    spec_lens = torch.zeros(bucket_rows, dtype=torch.int32)
    spec_lens[:real_rows].copy_(packed.spec_lens.to(torch.int32))
    decision_type = torch.full((bucket_rows,), -1, dtype=torch.int32)
    decision_type[:real_rows].copy_(packed.decision_type.to(torch.int32))

    seq_lengths_host = (
        tuple(int(x) for x in packed.seq_lengths_host)
        if packed.seq_lengths_host is not None
        else tuple(int(x) for x in packed.seq_lengths.tolist())
    )
    seq_lengths_host = seq_lengths_host + tuple(int(x) for x in dummy_lens.tolist())

    bitmap: Tensor | None = None
    if packed.legal_edge_bitmap is not None:
        bk = int(packed.legal_edge_bitmap.shape[1])
        ak = int(packed.legal_edge_bitmap.shape[2])
        bitmap = torch.zeros((bucket_rows, bk, ak), dtype=torch.bool)
        bitmap[:real_rows].copy_(packed.legal_edge_bitmap)

    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu_seqlens,
        seq_lengths=seq_lengths,
        state_positions=state_positions,
        card_ref_positions=card_ref_positions,
        spec_lens=spec_lens,
        decision_type=decision_type,
        pointer_anchor_positions=_pad_anchor_2d(packed.pointer_anchor_positions),
        pointer_anchor_kinds=_pad_anchor_2d(packed.pointer_anchor_kinds),
        pointer_anchor_subjects=_pad_anchor_2d(packed.pointer_anchor_subjects),
        pointer_anchor_handles=_pad_anchor_2d(packed.pointer_anchor_handles),
        legal_edge_bitmap=bitmap,
        total_tokens=bucket_tokens,
        seq_lengths_host=seq_lengths_host,
        # Pin to the bucket's total tokens — a safe upper bound (no single
        # sequence can exceed the total). flash_attn uses ``max_seqlen``
        # for scratch sizing; over-shooting is correct, just slightly more
        # memory. Critically, holding it constant per bucket stops
        # Dynamo from specializing on a different Python int per call.
        max_seqlen=bucket_tokens,
    )


def _slice_padded_packed_to_live(
    padded: PackedTextBatch, *, real_rows: int, real_tokens: int
) -> PackedTextBatch:
    """View the live region of a padded packed batch.

    The padded layout places real tokens at indices ``[0, real_tokens)``
    and real rows at ``[0, real_rows)``; slicing those prefixes gives
    a coherent unpadded PackedTextBatch for downstream consumers.
    """
    sl_host = padded.seq_lengths_host[:real_rows] if padded.seq_lengths_host is not None else None
    return PackedTextBatch(
        token_ids=padded.token_ids[:real_tokens],
        seq_id=padded.seq_id[:real_tokens],
        pos_in_seq=padded.pos_in_seq[:real_tokens],
        cu_seqlens=padded.cu_seqlens[: real_rows + 1],
        seq_lengths=padded.seq_lengths[:real_rows],
        state_positions=padded.state_positions[:real_rows],
        card_ref_positions=padded.card_ref_positions[:real_rows],
        spec_lens=padded.spec_lens[:real_rows],
        decision_type=padded.decision_type[:real_rows],
        pointer_anchor_positions=padded.pointer_anchor_positions[:real_rows],
        pointer_anchor_kinds=padded.pointer_anchor_kinds[:real_rows],
        pointer_anchor_subjects=padded.pointer_anchor_subjects[:real_rows],
        pointer_anchor_handles=padded.pointer_anchor_handles[:real_rows],
        legal_edge_bitmap=(
            padded.legal_edge_bitmap[:real_rows] if padded.legal_edge_bitmap is not None else None
        ),
        total_tokens=real_tokens,
        seq_lengths_host=sl_host,
        max_seqlen=max(sl_host, default=0) if sl_host is not None else None,
    )


def _slice_encoded_to_live(
    encoded: EncodedSnapshots,
    h_out: Tensor | None,
    c_out: Tensor | None,
    *,
    real_rows: int,
    real_tokens: int,
    live_packed: PackedTextBatch,
) -> tuple[EncodedSnapshots, Tensor | None, Tensor | None]:
    """Slice padded encoder outputs back to the live row/token region.

    ``live_packed`` is the unpadded device-side batch the downstream
    decoder + scatter_packed_to_padded consume.
    """
    sliced = EncodedSnapshots(
        state_vector=encoded.state_vector[:real_rows],
        encoded=encoded.encoded[:real_tokens],
        card_vectors=encoded.card_vectors[:real_rows],
        card_mask=encoded.card_mask[:real_rows],
        packed=live_packed,
    )
    sliced_h = h_out[:, :real_rows].contiguous() if h_out is not None else None
    sliced_c = c_out[:, :real_rows].contiguous() if c_out is not None else None
    return sliced, sliced_h, sliced_c


class TextInferencePipeline:
    """Encoder + decoder forward for inference. Stateless w.r.t. policy.

    When ``bucketed=True``, the encoder portion is compiled per bucket
    with ``mode="reduce-overhead"`` so inductor can capture a CUDA graph.
    The decoder loop + value head stay eager.

    Defaults to ``bucketed=False`` so existing call sites (opponent-pool
    eval, unit tests) keep the simple dynamic-shape path.
    """

    def __init__(
        self,
        *,
        deterministic: bool = False,
        bucketed: bool = False,
        buckets: tuple[tuple[int, int], ...] = DEFAULT_BUCKETS,
        compile_decoder: bool = False,
    ) -> None:
        self._deterministic = bool(deterministic)
        self._bucketed = bool(bucketed)
        self._buckets = tuple(sorted(buckets, key=lambda rt: (rt[0], rt[1])))
        self._compiled_encoders: dict[tuple[int, int], Callable[..., Any]] = {}
        # Phase F: lazily compile decoder_sample with dynamic=True. Per-bucket
        # decoder bucketing would also need a fixed S_max (scatter_padded output)
        # — deferred. dynamic=True doesn't get CUDA Graphs but inductor can
        # still fuse the per-step ops, cutting per-iteration launches.
        self._compile_decoder = bool(compile_decoder)
        self._compiled_decoder_sample: Callable[..., Any] | None = None

    # --- Bucketed encoder forward -------------------------------------

    def _compiled_encoder_for_bucket(
        self, recurrent_policy: Any, bucket: tuple[int, int]
    ) -> Callable[..., Any]:
        existing = self._compiled_encoders.get(bucket)
        if existing is not None:
            return existing
        compiled = cast(
            Callable[..., Any],
            torch.compile(
                recurrent_policy._encode_with_history_impl,
                mode="reduce-overhead",
                dynamic=False,
                fullgraph=False,
            ),
        )
        self._compiled_encoders[bucket] = compiled
        return compiled

    def _bucketed_encoder_forward(
        self,
        recurrent_policy: Any,
        host_packed: PackedTextBatch,
        h_in: Tensor,
        c_in: Tensor,
        device: torch.device,
    ) -> tuple[EncodedSnapshots, Tensor, Tensor, PackedTextBatch] | None:
        """Bucketed encoder forward path.

        Returns ``(encoded, h_out, c_out, live_device_batch)`` or ``None``
        if no bucket fits / inputs are not host-side.
        """
        real_rows = int(host_packed.seq_lengths.shape[0])
        real_tokens = int(host_packed.token_ids.shape[0])
        bucket = _select_bucket(real_rows, real_tokens, self._buckets)
        if bucket is None:
            return None
        bucket_rows, bucket_tokens = bucket

        if host_packed.token_ids.device.type != "cpu":
            return None
        padded_host = _pad_packed_to_bucket(
            host_packed, bucket_rows=bucket_rows, bucket_tokens=bucket_tokens
        )
        padded_device = padded_host.to(device)
        live_device_batch = _slice_padded_packed_to_live(
            padded_device, real_rows=real_rows, real_tokens=real_tokens
        )

        h_padded = torch.zeros(
            (h_in.shape[0], bucket_rows, h_in.shape[2]), dtype=h_in.dtype, device=device
        )
        c_padded = torch.zeros(
            (c_in.shape[0], bucket_rows, c_in.shape[2]), dtype=c_in.dtype, device=device
        )
        h_padded[:, :real_rows].copy_(h_in.to(device))
        c_padded[:, :real_rows].copy_(c_in.to(device))

        compiled = self._compiled_encoder_for_bucket(recurrent_policy, bucket)
        encoded_padded, h_out_padded, c_out_padded = compiled(padded_device, h_padded, c_padded)

        encoded, h_out, c_out = _slice_encoded_to_live(
            encoded_padded,
            h_out_padded,
            c_out_padded,
            real_rows=real_rows,
            real_tokens=real_tokens,
            live_packed=live_device_batch,
        )
        assert h_out is not None and c_out is not None
        return encoded, h_out, c_out, live_device_batch

    # --- Main entry point ---------------------------------------------

    def encode_and_sample(
        self,
        policy: Any,
        merged_packed: PackedTextBatch,
        h_in: Tensor | None,
        c_in: Tensor | None,
    ) -> InferenceOutput:
        """Encode + sample one merged batch.

        ``h_in`` / ``c_in`` carry the per-row LSTM input state (one entry
        per row of ``merged_packed``). They may be ``None`` for the
        non-recurrent path (RecurrentTextPolicy absent), in which case
        the recurrent encoder is skipped.
        """

        text_policy = policy.policy.text_policy if hasattr(policy, "policy") else policy.text_policy
        recurrent_policy = policy.policy if hasattr(policy, "policy") else None

        device = next(text_policy.parameters()).device
        encoded_snaps: EncodedSnapshots
        h_out: Tensor | None = None
        c_out: Tensor | None = None
        encoded_device_batch: PackedTextBatch

        def _move_to_device(b: PackedTextBatch) -> PackedTextBatch:
            return b if b.token_ids.device == device else b.to(device)

        if recurrent_policy is not None and h_in is not None and c_in is not None:
            bucketed_result = None
            if self._bucketed and device.type == "cuda":
                bucketed_result = self._bucketed_encoder_forward(
                    recurrent_policy, merged_packed, h_in, c_in, device
                )
            if bucketed_result is not None:
                encoded_snaps, h_out, c_out, encoded_device_batch = bucketed_result
            else:
                encoded_device_batch = _move_to_device(merged_packed)
                encoded_snaps, h_out, c_out = recurrent_policy.encode_with_history(
                    encoded_device_batch, h_in=h_in.to(device), c_in=c_in.to(device)
                )
        else:
            encoded_device_batch = _move_to_device(merged_packed)
            encoded_snaps = text_policy.encode_packed_only(encoded_device_batch)

        # encoder.forward_packed returns rank-2 [T_packed, D]; decoder
        # cross-attn needs rank-3 [B, T_max, D].
        encoded, attn_mask = scatter_packed_to_padded(encoded_snaps.encoded, encoded_device_batch)
        # ``pointer_anchor_positions`` is row-local end-to-end (see
        # native_assembler.to_packed_text_batch); the decoder consumes
        # it directly as an index into the [B, T_max, D] padded tensor.
        anchor_positions_rowlocal = encoded_device_batch.pointer_anchor_positions

        decoder_fn: Callable[..., Any] = decoder_sample
        if self._compile_decoder and device.type == "cuda":
            if self._compiled_decoder_sample is None:
                self._compiled_decoder_sample = cast(
                    Callable[..., Any],
                    torch.compile(decoder_sample, dynamic=True),
                )
            decoder_fn = self._compiled_decoder_sample
        sample = decoder_fn(
            text_policy,
            encoded,
            attn_mask,
            encoded_device_batch.decision_type.to(device=device, dtype=torch.long),
            anchor_positions_rowlocal.to(device=device, dtype=torch.long),
            encoded_device_batch.pointer_anchor_kinds.to(device=device, dtype=torch.long),
            encoded_device_batch.pointer_anchor_subjects.to(device=device, dtype=torch.long),
            encoded_device_batch.pointer_anchor_handles.to(device=device, dtype=torch.long),
            legal_edge_bitmap=encoded_device_batch.legal_edge_bitmap,
            greedy=self._deterministic,
        )
        value = text_policy.run_heads(encoded_snaps)
        return InferenceOutput(
            decoder=native_decoder_batch_from_sample(sample, value=value),
            h_out=h_out,
            c_out=c_out,
        )


__all__ = [
    "DEFAULT_BUCKETS",
    "InferenceOutput",
    "TextInferencePipeline",
]
