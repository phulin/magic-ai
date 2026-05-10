"""Reusable text-policy inference forward.

``TextInferencePipeline`` packages the encoder + decoder forward used by
the inference server and offline eval (opponent pool, smoke tests). It is
intentionally stateless w.r.t. the policy itself: callers pass the
policy as a per-call argument so the pipeline plays nicely with the
inference server's policy-version manager (the live policy reference
swaps under the server's feet on policy updates).

Phase B of the inference-pipeline refactor extracts ``_sample_decoder``
out of the server class into this module unchanged. Phase C will add
per-bucket static buffers + CUDA-graph capture here; that state is
correctly scoped to the pipeline (not the server's queue management).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from magic_ai.text_encoder.batch import PackedTextBatch, scatter_packed_to_padded
from magic_ai.text_encoder.decoder_batch import (
    NativeTextDecoderBatch,
    native_decoder_batch_from_sample,
)
from magic_ai.text_encoder.decoder_inference import decoder_sample


@dataclass(frozen=True)
class InferenceOutput:
    decoder: NativeTextDecoderBatch
    h_out: Tensor | None
    c_out: Tensor | None


class TextInferencePipeline:
    """Encoder + decoder forward for inference. Stateless w.r.t. policy.

    The pipeline owns no policy reference — callers supply an already-
    acquired policy per call. This keeps it composable with both the
    inference server's policy-version manager and the offline eval path.

    Phase B placeholder: forwards verbatim to the previous
    ``_sample_decoder`` body. Phase C will add bucketed static buffers +
    CUDA graphs here.
    """

    def __init__(self, *, deterministic: bool = False) -> None:
        self._deterministic = bool(deterministic)

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

        if recurrent_policy is not None and h_in is not None and c_in is not None:
            encoded_snaps, h_out, c_out = recurrent_policy.encode_with_history(
                merged_packed, h_in=h_in, c_in=c_in
            )
        else:
            encoded_snaps = text_policy.encode_packed_only(merged_packed)
            h_out = None
            c_out = None

        # encoder.forward_packed returns rank-2 [T_packed, D]; decoder
        # cross-attn needs rank-3 [B, T_max, D].
        encoded, attn_mask = scatter_packed_to_padded(encoded_snaps.encoded, merged_packed)
        device = encoded.device
        # ``pointer_anchor_positions`` is row-local end-to-end (see
        # native_assembler.to_packed_text_batch); the decoder consumes
        # it directly as an index into the [B, T_max, D] padded tensor.
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
        value = text_policy.run_heads(encoded_snaps)
        return InferenceOutput(
            decoder=native_decoder_batch_from_sample(sample, value=value),
            h_out=h_out,
            c_out=c_out,
        )


__all__ = ["InferenceOutput", "TextInferencePipeline"]
