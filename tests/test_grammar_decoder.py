"""Tests for the autoregressive GrammarDecoder module."""

from __future__ import annotations

import torch
from magic_ai.text_encoder.decoder import (
    GRAMMAR_VOCAB_SIZE,
    GrammarDecoder,
    GrammarDecoderConfig,
    combined_sample,
)


def _make_decoder(*, n_layers: int = 2, d_model: int = 32, n_heads: int = 4) -> GrammarDecoder:
    cfg = GrammarDecoderConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=64,
        max_decode_len=16,
    )
    torch.manual_seed(0)
    return GrammarDecoder(cfg).eval()


def test_forward_teacher_forced_shapes() -> None:
    decoder = _make_decoder()
    b, seq_len, t_enc, d = 2, 5, 10, decoder.cfg.d_model
    target_tokens = torch.randint(0, GRAMMAR_VOCAB_SIZE, (b, seq_len))
    encoded = torch.randn(b, t_enc, d)
    enc_mask = torch.ones(b, t_enc, dtype=torch.bool)
    vocab_logits, pointer_logits = decoder.forward_teacher_forced(target_tokens, encoded, enc_mask)
    assert vocab_logits.shape == (b, seq_len, GRAMMAR_VOCAB_SIZE)
    assert pointer_logits.shape == (b, seq_len, t_enc)


def test_teacher_forced_step_parity() -> None:
    decoder = _make_decoder()
    b, seq_len, t_enc, d = 2, 4, 8, decoder.cfg.d_model
    torch.manual_seed(42)
    target_tokens = torch.randint(0, GRAMMAR_VOCAB_SIZE, (b, seq_len))
    encoded = torch.randn(b, t_enc, d)
    enc_mask = torch.ones(b, t_enc, dtype=torch.bool)

    with torch.no_grad():
        tf_vocab, tf_pointer = decoder.forward_teacher_forced(target_tokens, encoded, enc_mask)
        state = decoder.init_state(encoded)
        step_vocab = []
        step_pointer = []
        for i in range(seq_len):
            v, p, state = decoder.step(
                target_tokens[:, i],
                torch.full((b,), -1, dtype=torch.long),
                encoded,
                enc_mask,
                state,
            )
            step_vocab.append(v)
            step_pointer.append(p)
        sv = torch.stack(step_vocab, dim=1)
        sp = torch.stack(step_pointer, dim=1)
    assert torch.allclose(tf_vocab, sv, atol=1e-5)
    assert torch.allclose(tf_pointer, sp, atol=1e-5)


def test_causality() -> None:
    decoder = _make_decoder()
    b, seq_len, t_enc, d = 2, 6, 8, decoder.cfg.d_model
    torch.manual_seed(7)
    target = torch.randint(0, GRAMMAR_VOCAB_SIZE, (b, seq_len))
    encoded = torch.randn(b, t_enc, d)
    enc_mask = torch.ones(b, t_enc, dtype=torch.bool)
    with torch.no_grad():
        v1, _ = decoder.forward_teacher_forced(target, encoded, enc_mask)
        target2 = target.clone()
        target2[:, 3:] = (target2[:, 3:] + 1) % GRAMMAR_VOCAB_SIZE
        v2, _ = decoder.forward_teacher_forced(target2, encoded, enc_mask)
    assert torch.allclose(v1[:, :3], v2[:, :3], atol=1e-5)
    assert not torch.allclose(v1[:, 3:], v2[:, 3:], atol=1e-5)


def test_kv_cache_shape_growth() -> None:
    decoder = _make_decoder(n_layers=2)
    b, t_enc, d = 2, 5, decoder.cfg.d_model
    encoded = torch.randn(b, t_enc, d)
    enc_mask = torch.ones(b, t_enc, dtype=torch.bool)
    state = decoder.init_state(encoded)
    head_dim = decoder.cfg.d_model // decoder.cfg.n_heads
    h = decoder.cfg.n_heads
    for step_idx in range(3):
        prev = torch.randint(0, GRAMMAR_VOCAB_SIZE, (b,))
        _, _, state = decoder.step(
            prev, torch.full((b,), -1, dtype=torch.long), encoded, enc_mask, state
        )
        assert state is not None
        for layer_idx in range(decoder.cfg.n_layers):
            sk = state.self_k[layer_idx]
            sv = state.self_v[layer_idx]
            assert sk is not None and sv is not None
            assert sk.shape == (b, h, step_idx + 1, head_dim)
            assert sv.shape == (b, h, step_idx + 1, head_dim)
            assert state.cross_k[layer_idx].shape == (b, h, t_enc, head_dim)
            assert state.cross_v[layer_idx].shape == (b, h, t_enc, head_dim)


def test_combined_sample_pointer_singleton() -> None:
    b, v, t_enc = 4, GRAMMAR_VOCAB_SIZE, 7
    vocab_logits = torch.zeros(b, v)
    pointer_logits = torch.zeros(b, t_enc)
    vocab_mask = torch.ones(b, v, dtype=torch.bool)
    pointer_mask = torch.zeros(b, t_enc, dtype=torch.bool)
    chosen = torch.tensor([0, 3, 5, 6])
    pointer_mask[torch.arange(b), chosen] = True
    is_pointer = torch.ones(b, dtype=torch.bool)
    sampled_vocab, sampled_pointer = combined_sample(
        vocab_logits, pointer_logits, vocab_mask, pointer_mask, is_pointer
    )
    assert torch.equal(sampled_pointer, chosen)
    assert torch.equal(sampled_vocab, torch.full((b,), -1))


def test_combined_sample_mixed_batch_greedy() -> None:
    b, v, t_enc = 3, GRAMMAR_VOCAB_SIZE, 5
    vocab_logits = torch.zeros(b, v)
    vocab_logits[:, 2] = 10.0
    pointer_logits = torch.zeros(b, t_enc)
    pointer_logits[:, 4] = 10.0
    vocab_mask = torch.ones(b, v, dtype=torch.bool)
    pointer_mask = torch.ones(b, t_enc, dtype=torch.bool)
    is_pointer = torch.tensor([False, True, False])
    sv, sp = combined_sample(
        vocab_logits, pointer_logits, vocab_mask, pointer_mask, is_pointer, greedy=True
    )
    assert torch.equal(sv, torch.tensor([2, -1, 2]))
    assert torch.equal(sp, torch.tensor([-1, 4, -1]))
