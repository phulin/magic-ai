"""Tests for ``NativeTextDecoderBatch`` slice/concat round-trip and the
``DecoderSampleOutput → NativeTextDecoderBatch`` translation.
"""

from __future__ import annotations

import torch
from magic_ai.text_encoder.decoder_batch import (
    DecoderSampleOutput,
    NativeTextDecoderBatch,
    native_decoder_batch_from_sample,
)


def _make_sample(
    b: int, l_max: int, n_max: int, t_enc: int = 8, v_vocab: int = 4
) -> DecoderSampleOutput:
    """Construct a synthetic DecoderSampleOutput."""

    return DecoderSampleOutput(
        output_token_ids=torch.arange(b * l_max, dtype=torch.long).view(b, l_max),
        output_pointer_pos=torch.full((b, l_max), -1, dtype=torch.long),
        output_pointer_subjects=torch.arange(b * l_max, dtype=torch.long).view(b, l_max) % n_max,
        output_is_pointer=torch.zeros((b, l_max), dtype=torch.bool),
        output_pad_mask=torch.ones((b, l_max), dtype=torch.bool),
        log_probs=torch.full((b, l_max), -0.25),
        vocab_mask=torch.ones((b, l_max, v_vocab), dtype=torch.bool),
        pointer_mask=torch.ones((b, l_max, t_enc), dtype=torch.bool),
        decision_type=torch.arange(b, dtype=torch.long),
        pointer_anchor_handles=torch.arange(b * n_max, dtype=torch.long).view(b, n_max),
        pointer_anchor_count=torch.full((b,), n_max, dtype=torch.long),
    )


def test_native_decoder_batch_round_trip_from_sample() -> None:
    sample = _make_sample(b=4, l_max=6, n_max=3)
    value = torch.linspace(-1.0, 1.0, 4)
    batch = native_decoder_batch_from_sample(sample, value=value)

    assert isinstance(batch, NativeTextDecoderBatch)
    assert int(len(batch)) == 4
    assert batch.decision_type.dtype == torch.int32
    assert batch.output_token_ids.dtype == torch.int32
    assert batch.output_pointer_subjects.dtype == torch.int32
    assert batch.output_is_pointer.dtype == torch.bool
    assert batch.output_lens.dtype == torch.int32
    assert batch.pointer_anchor_handles.dtype == torch.int32
    assert batch.pointer_anchor_count.dtype == torch.int32
    assert batch.log_probs.dtype == torch.float32
    assert batch.value.dtype == torch.float32
    assert tuple(batch.output_token_ids.shape) == (4, 6)
    assert tuple(batch.pointer_anchor_handles.shape) == (4, 3)
    # output_lens = sum of pad mask
    torch.testing.assert_close(batch.output_lens, torch.full((4,), 6, dtype=torch.int32))


def test_native_decoder_batch_slice_returns_consistent_subbatch() -> None:
    sample = _make_sample(b=6, l_max=4, n_max=2)
    value = torch.zeros(6)
    batch = native_decoder_batch_from_sample(sample, value=value)

    sub = batch[2:5]
    assert int(len(sub)) == 3
    assert tuple(sub.output_token_ids.shape) == (3, 4)
    assert tuple(sub.pointer_anchor_handles.shape) == (3, 2)
    torch.testing.assert_close(sub.decision_type, batch.decision_type[2:5])
    torch.testing.assert_close(sub.output_token_ids, batch.output_token_ids[2:5])
    torch.testing.assert_close(sub.pointer_anchor_handles, batch.pointer_anchor_handles[2:5])


def test_native_decoder_batch_concat_pads_ragged_widths() -> None:
    sample_a = _make_sample(b=2, l_max=4, n_max=2)
    sample_b = _make_sample(b=3, l_max=6, n_max=3)
    a = native_decoder_batch_from_sample(sample_a, value=torch.zeros(2))
    b = native_decoder_batch_from_sample(sample_b, value=torch.zeros(3))

    merged = NativeTextDecoderBatch.concat([a, b])
    assert int(len(merged)) == 5
    # Should pad to max widths of the parts.
    assert tuple(merged.output_token_ids.shape) == (5, 6)
    assert tuple(merged.pointer_anchor_handles.shape) == (5, 3)
    # First 2 rows should match a's data padded with 0/-1.
    torch.testing.assert_close(merged.output_token_ids[:2, :4], a.output_token_ids)
    assert bool((merged.output_token_ids[:2, 4:] == 0).all())
    assert bool((merged.output_pointer_subjects[:2, 4:] == -1).all())
    # Anchor handles padded with 0 fill on dim 1 (per concat fill=0).
    torch.testing.assert_close(merged.pointer_anchor_handles[:2, :2], a.pointer_anchor_handles)
    # Last 3 rows match b directly.
    torch.testing.assert_close(merged.output_token_ids[2:, :], b.output_token_ids)


def test_native_decoder_batch_single_row_index() -> None:
    sample = _make_sample(b=3, l_max=2, n_max=2)
    batch = native_decoder_batch_from_sample(sample, value=torch.tensor([0.0, 1.0, 2.0]))
    row = batch[1]
    assert int(len(row)) == 1
    torch.testing.assert_close(row.value, torch.tensor([1.0]))
