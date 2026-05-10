"""Parity: padded vs packed forward over the text-state encoder.

The packed path concatenates rows along one sequence axis and uses a
flex_attention document mask plus per-token RoPE. After gathering at
the same anchors, the resulting per-card / state vectors must agree with the
padded path to fp tolerance.
"""

from __future__ import annotations

import torch
from magic_ai.text_encoder.batch import TextEncodedBatch, pack_batch
from magic_ai.text_encoder.model import (
    TextEncoderConfig,
    TextStateEncoder,
    gather_card_vectors,
    gather_card_vectors_packed,
    gather_state_vector,
    gather_state_vector_packed,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


def _make_batch(vocab_size: int = 1000) -> TextEncodedBatch:
    torch.manual_seed(0)
    b, t = 3, 20
    seq_lens = torch.tensor([20, 14, 9], dtype=torch.int64)
    token_ids = torch.randint(low=1, high=vocab_size, size=(b, t), dtype=torch.int64)
    attention_mask = torch.zeros(b, t, dtype=torch.int64)
    for i, n in enumerate(seq_lens.tolist()):
        attention_mask[i, :n] = 1
        token_ids[i, n:] = 0  # pad

    card_ref_positions = torch.full((b, MAX_CARD_REFS), -1, dtype=torch.int64)
    card_ref_positions[0, :3] = torch.tensor([2, 5, 8])
    card_ref_positions[1, :2] = torch.tensor([3, 6])
    card_ref_positions[2, :1] = torch.tensor([4])

    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lens,
        seq_lengths_host=tuple(int(n) for n in seq_lens.tolist()),
    )


def test_pack_batch_shapes_and_anchors() -> None:
    padded = _make_batch()
    packed = pack_batch(padded)

    total = int(padded.seq_lengths.sum().item())
    assert packed.token_ids.shape == (total,)
    assert packed.seq_id.shape == (total,)
    assert packed.pos_in_seq.shape == (total,)
    assert packed.cu_seqlens.shape == (4,)

    # cu_seqlens is exclusive-prefix-sum.
    expected_cu = torch.tensor([0, 20, 34, 43], dtype=torch.int64)
    assert torch.equal(packed.cu_seqlens, expected_cu)
    assert torch.equal(packed.state_positions, expected_cu[:-1])

    # pos_in_seq resets per document and ranges 0..len-1.
    for i, (start, end) in enumerate(zip(expected_cu[:-1].tolist(), expected_cu[1:].tolist())):
        seg = packed.pos_in_seq[start:end]
        assert torch.equal(seg, torch.arange(end - start, dtype=torch.int64))
        assert (packed.seq_id[start:end] == i).all()

    # Card anchors stay row-local end-to-end; pack_batch is a no-op for them
    # (just dtype cast). gather_card_vectors_packed converts to packed coords
    # internally where it actually needs absolute offsets into [T_packed, D].
    assert torch.equal(packed.card_ref_positions, padded.card_ref_positions.to(torch.int32))


def test_padded_vs_packed_parity() -> None:
    vocab_size = 1000
    cfg = TextEncoderConfig(vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=4, d_ff=64)
    encoder = TextStateEncoder(cfg).eval()

    padded = _make_batch(vocab_size=vocab_size)
    packed = pack_batch(padded)

    with torch.no_grad():
        hidden_dense = encoder(padded)  # [B, T_max, D]
        hidden_packed = encoder.forward_packed(packed)  # [T_packed, D]

    state_dense = gather_state_vector(hidden_dense, padded)
    state_packed = gather_state_vector_packed(hidden_packed, packed)
    assert torch.allclose(state_dense, state_packed, atol=1e-5, rtol=1e-4)

    card_dense, card_mask_dense = gather_card_vectors(hidden_dense, padded)
    card_packed, card_mask_packed = gather_card_vectors_packed(hidden_packed, packed)
    assert torch.equal(card_mask_dense, card_mask_packed)
    assert torch.allclose(card_dense, card_packed, atol=1e-5, rtol=1e-4)


def test_packed_backward_smoke() -> None:
    cfg = TextEncoderConfig(vocab_size=1000, d_model=32, n_layers=1, n_heads=4, d_ff=64)
    encoder = TextStateEncoder(cfg)
    packed = pack_batch(_make_batch())

    hidden = encoder.forward_packed(packed)
    state = gather_state_vector_packed(hidden, packed)
    state.sum().backward()

    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
