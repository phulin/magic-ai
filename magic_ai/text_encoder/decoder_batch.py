"""Wire-shape dataclasses for the grammar-decoder pipeline.

Pure data + one sample→wire conversion helper. No policy or engine deps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from magic_ai.text_encoder.batch import PackedTextBatch


@dataclass(frozen=True)
class DecoderSampleOutput:
    """Result of one batched ``decoder_sample`` call.

    Lengths are right-padded to ``max_decode_len``. ``output_pad_mask``
    is False for steps after END (and for rows that ended early).
    """

    output_token_ids: Tensor  # [B, L] int64 (PAD = 0)
    output_pointer_pos: Tensor  # [B, L] int64 (-1 fill, encoder position)
    output_pointer_subjects: Tensor  # [B, L] int64 (-1 fill, anchor subject_index)
    output_is_pointer: Tensor  # [B, L] bool
    output_pad_mask: Tensor  # [B, L] bool (True = valid, False = post-END/pad)
    log_probs: Tensor  # [B, L] float — per-step log p of chosen action
    decision_type: Tensor  # [B] int64 (passed through)
    pointer_anchor_handles: Tensor  # [B, N_max] int64 (passed through)
    pointer_anchor_count: Tensor  # [B] int64 (passed through)


@dataclass(frozen=True)
class DecoderReplayScores:
    """Result of teacher-forced replay scoring for the grammar decoder."""

    per_row_log_pi: Tensor  # [B] sum of per-step log p of stored target
    per_row_entropy: Tensor  # [B] sum of per-step entropy of the stored decision
    per_step_log_pi: Tensor  # [B, L] per-step log p (zeroed at pad positions)


@dataclass(frozen=True)
class DecoderDecisionLayout:
    """Per-row decoded action returned by the sampler (replaces inline-blank
    ``TextDecisionLayout``).
    """

    output_token_ids: Tensor  # [L] int64
    output_pointer_pos: Tensor  # [L] int64 (encoder position, -1 fill)
    output_pointer_subjects: Tensor  # [L] int64 (anchor subject_index, -1 fill)
    output_is_pointer: Tensor  # [L] bool
    output_pad_mask: Tensor  # [L] bool
    decision_type: int
    pointer_anchor_handles: Tensor  # [N_max] int64
    pointer_anchor_count: int


@dataclass(frozen=True)
class NativeTextDecoderBatch:
    """Wire shape for the IMPALA inference-server reply / actor step input.

    Mirrors the Go-side ``MageDecoderStepRequest`` struct: a flat int
    rectangle that the Go entrypoint slices and dispatches per-decision-
    type. Plus the IMPALA-side bookkeeping (log_probs, value, pad_mask)
    used by replay scoring and importance ratios.
    """

    decision_type: Tensor  # [B] int32
    output_token_ids: Tensor  # [B, L_max] int32 (PAD = 0)
    output_pointer_subjects: Tensor  # [B, L_max] int32 (anchor subject_index, -1 fill)
    output_is_pointer: Tensor  # [B, L_max] bool
    output_lens: Tensor  # [B] int32 (number of valid steps per row)
    pointer_anchor_handles: Tensor  # [B, N_max] int32
    pointer_anchor_count: Tensor  # [B] int32
    log_probs: Tensor  # [B, L_max] float (zero at pad)
    value: Tensor  # [B] float
    output_pad_mask: Tensor  # [B, L_max] bool

    def __len__(self) -> int:
        return int(self.decision_type.shape[0])

    def __getitem__(self, key: slice | int) -> NativeTextDecoderBatch:
        """Row slice. Integer keys are wrapped to a length-1 batch."""
        if isinstance(key, int):
            key = slice(key, key + 1)
        return NativeTextDecoderBatch(
            decision_type=self.decision_type[key],
            output_token_ids=self.output_token_ids[key],
            output_pointer_subjects=self.output_pointer_subjects[key],
            output_is_pointer=self.output_is_pointer[key],
            output_lens=self.output_lens[key],
            pointer_anchor_handles=self.pointer_anchor_handles[key],
            pointer_anchor_count=self.pointer_anchor_count[key],
            log_probs=self.log_probs[key],
            value=self.value[key],
            output_pad_mask=self.output_pad_mask[key],
        )

    @classmethod
    def concat(cls, parts: Sequence[NativeTextDecoderBatch]) -> NativeTextDecoderBatch:
        """Row-axis concatenation; left-pads ragged L_max / N_max with fill."""
        if len(parts) == 1:
            return parts[0]
        l_max = max(int(p.output_token_ids.shape[1]) for p in parts)
        n_max = max(int(p.pointer_anchor_handles.shape[1]) for p in parts)

        def _pad2d(t: Tensor, width: int, *, fill: int | float) -> Tensor:
            cur = int(t.shape[1])
            if cur == width:
                return t
            rows = int(t.shape[0])
            out = torch.full((rows, width), fill, dtype=t.dtype, device=t.device)
            out[:, :cur] = t
            return out

        return cls(
            decision_type=torch.cat([p.decision_type for p in parts], dim=0),
            output_token_ids=torch.cat(
                [_pad2d(p.output_token_ids, l_max, fill=0) for p in parts], dim=0
            ),
            output_pointer_subjects=torch.cat(
                [_pad2d(p.output_pointer_subjects, l_max, fill=-1) for p in parts], dim=0
            ),
            output_is_pointer=torch.cat(
                [_pad2d(p.output_is_pointer, l_max, fill=0) for p in parts], dim=0
            ),
            output_lens=torch.cat([p.output_lens for p in parts], dim=0),
            pointer_anchor_handles=torch.cat(
                [_pad2d(p.pointer_anchor_handles, n_max, fill=0) for p in parts], dim=0
            ),
            pointer_anchor_count=torch.cat([p.pointer_anchor_count for p in parts], dim=0),
            log_probs=torch.cat([_pad2d(p.log_probs, l_max, fill=0.0) for p in parts], dim=0),
            value=torch.cat([p.value for p in parts], dim=0),
            output_pad_mask=torch.cat(
                [_pad2d(p.output_pad_mask, l_max, fill=0) for p in parts], dim=0
            ),
        )


# Kept for backwards-compatible imports while the rest of the cutover lands.
@dataclass(frozen=True)
class NativeTextReplayPayload:
    """Placeholder for the legacy native replay payload struct.

    Phase 5 stripped the inline-blank training path. The native replay
    pipeline will be re-wired around :class:`DecoderDecisionPayload` in
    a later phase. Symbol kept so unrelated import sites don't blow up
    mid-cutover; constructing one will fail on access.
    """

    encoded: PackedTextBatch | None = None


@dataclass(frozen=True)
class NativeTextSampleBatch:
    """Result of ``LSTMStatefulTextPolicy.sample_batch`` — a list of decoded actions.

    ``replay_payload`` will be wired up in a later phase; for now the field
    is kept ``None`` so call-sites that pass it through don't crash.
    """

    decoded: list[DecoderDecisionLayout]
    log_probs: Tensor  # [B, L]
    replay_rows: list[int]
    replay_payload: NativeTextReplayPayload | None = None


def native_decoder_batch_from_sample(
    sample: DecoderSampleOutput,
    *,
    value: Tensor,
) -> NativeTextDecoderBatch:
    """Convert a :class:`DecoderSampleOutput` (+ per-row value head outputs)
    into the IMPALA wire shape consumed by the inference-server reply and
    the actor-side ``mage.batch_step_by_decoder_action`` call.
    """

    output_lens = sample.output_pad_mask.sum(dim=-1).to(dtype=torch.int32)
    return NativeTextDecoderBatch(
        decision_type=sample.decision_type.to(dtype=torch.int32),
        output_token_ids=sample.output_token_ids.to(dtype=torch.int32),
        output_pointer_subjects=sample.output_pointer_subjects.to(dtype=torch.int32),
        output_is_pointer=sample.output_is_pointer.to(dtype=torch.bool),
        output_lens=output_lens,
        pointer_anchor_handles=sample.pointer_anchor_handles.to(dtype=torch.int32),
        pointer_anchor_count=sample.pointer_anchor_count.to(dtype=torch.int32),
        log_probs=sample.log_probs.to(dtype=torch.float32),
        value=value.to(dtype=torch.float32),
        output_pad_mask=sample.output_pad_mask.to(dtype=torch.bool),
    )


__all__ = [
    "DecoderDecisionLayout",
    "DecoderReplayScores",
    "DecoderSampleOutput",
    "NativeTextDecoderBatch",
    "NativeTextReplayPayload",
    "NativeTextSampleBatch",
    "native_decoder_batch_from_sample",
]
