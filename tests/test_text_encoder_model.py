"""Forward / pooling / heads smoke test for the text-encoder model."""

from __future__ import annotations

import torch
from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.model import (
    InlineBlankPolicy,
    TextEncoderConfig,
    TextStateEncoder,
    ValueHead,
    gather_card_vectors,
    gather_state_vector,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


def _make_batch(
    b: int = 2,
    t: int = 24,
    max_opts: int = 4,
    max_targets: int = 3,
    vocab_size: int = 1000,
) -> TextEncodedBatch:
    torch.manual_seed(0)
    token_ids = torch.randint(low=1, high=vocab_size, size=(b, t), dtype=torch.int64)
    # Right-pad varying amounts.
    seq_lens = torch.tensor([t, t - 8], dtype=torch.int64)
    attention_mask = torch.zeros(b, t, dtype=torch.int64)
    for i, n in enumerate(seq_lens.tolist()):
        attention_mask[i, :n] = 1
        token_ids[i, n:] = 0  # pad_id

    # Card-ref positions: a few real anchors per row, rest -1.
    card_ref_positions = torch.full((b, MAX_CARD_REFS), -1, dtype=torch.int64)
    card_ref_positions[0, :3] = torch.tensor([2, 5, 8])
    card_ref_positions[1, :2] = torch.tensor([3, 6])

    # Options: 3 valid in row 0, 2 valid in row 1.
    option_positions = torch.full((b, max_opts), -1, dtype=torch.int64)
    option_positions[0, :3] = torch.tensor([10, 14, 18])
    option_positions[1, :2] = torch.tensor([9, 13])
    option_mask = option_positions >= 0

    # Targets per option.
    target_positions = torch.full((b, max_opts, max_targets), -1, dtype=torch.int64)
    target_positions[0, 0, :2] = torch.tensor([11, 12])
    target_positions[0, 1, :1] = torch.tensor([15])
    target_positions[1, 0, :3] = torch.tensor([10, 11, 12])
    target_mask = target_positions >= 0

    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        option_positions=option_positions,
        option_mask=option_mask,
        target_positions=target_positions,
        target_mask=target_mask,
        seq_lengths=seq_lens,
    )


def test_text_encoder_forward_pooling_heads_backward() -> None:
    vocab_size = 1000
    cfg = TextEncoderConfig(vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=4, d_ff=64)
    encoder = TextStateEncoder(cfg)
    batch = _make_batch(vocab_size=vocab_size)

    hidden = encoder(batch)
    assert hidden.shape == (2, 24, cfg.d_model)
    assert torch.isfinite(hidden).all()

    # Card vectors.
    card_vecs, card_mask = gather_card_vectors(hidden, batch)
    assert card_vecs.shape == (2, MAX_CARD_REFS, cfg.d_model)
    assert card_mask.shape == (2, MAX_CARD_REFS)
    assert (card_vecs[~card_mask] == 0).all()
    assert card_mask[0, :3].all() and not card_mask[0, 3:].any()

    # State.
    state_vec = gather_state_vector(hidden, batch)
    assert state_vec.shape == (2, cfg.d_model)
    assert torch.allclose(state_vec, hidden[:, 0, :])

    # Value head.
    value = ValueHead(cfg.d_model)

    values = value(state_vec)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()

    # Backward smoke.
    loss = values.sum() + card_vecs[card_mask].sum()
    loss.backward()

    grad_norms = [
        p.grad.detach().abs().sum().item()
        for p in list(encoder.parameters()) + list(value.parameters())
        if p.grad is not None
    ]
    assert any(g > 0 for g in grad_norms)


def test_inline_blank_policy_scores_legal_candidates_and_masks_padding() -> None:
    torch.manual_seed(0)
    vocab_size = 32
    d_model = 8
    embed = torch.nn.Embedding(vocab_size, d_model)
    dense = torch.nn.Linear(d_model, d_model)
    norm = torch.nn.LayerNorm(d_model)
    head = InlineBlankPolicy(embed, dense, norm, num_kinds=vocab_size)

    hidden = torch.randn(2, 5, d_model, requires_grad=True)
    blank_positions = torch.tensor([[1, 3], [2, -1]], dtype=torch.int32)
    blank_kind = torch.tensor([[4, 5], [6, 0]], dtype=torch.int32)
    blank_legal_ids = torch.tensor(
        [
            [[7, 8, 0], [9, 10, 11]],
            [[12, 13, 0], [0, 0, 0]],
        ],
        dtype=torch.int32,
    )
    blank_legal_mask = torch.tensor(
        [
            [[True, True, False], [True, True, True]],
            [[True, True, False], [False, False, False]],
        ]
    )

    logits = head(hidden, blank_positions, blank_kind, blank_legal_ids, blank_legal_mask)

    assert logits.shape == (2, 2, 3)
    assert torch.isfinite(logits[blank_legal_mask]).all()
    assert (logits[~blank_legal_mask] == float("-inf")).all()

    loss = logits[blank_legal_mask].sum()
    loss.backward()
    assert hidden.grad is not None
    assert float(hidden.grad.detach().abs().sum().item()) > 0


def test_local_window_collapses_to_global_when_oversized() -> None:
    """A local layer with window >= seq_len must equal a global layer."""

    vocab_size = 200
    t = 16
    cfg_global = TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=t,
        rope_theta_global=10000.0,
        rope_theta_local=10000.0,
        local_attention_window=2 * t,
        global_attn_every_n_layers=1,  # all layers global
    )
    cfg_local = TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=t,
        rope_theta_global=10000.0,
        rope_theta_local=10000.0,
        local_attention_window=2 * t,
        global_attn_every_n_layers=999,  # all layers local
    )

    torch.manual_seed(0)
    enc_global = TextStateEncoder(cfg_global)
    enc_local = TextStateEncoder(cfg_local)
    enc_local.load_state_dict(enc_global.state_dict(), strict=False)

    batch = _make_batch(b=2, t=t, vocab_size=vocab_size)
    enc_global.eval()
    enc_local.eval()
    with torch.no_grad():
        h_g = enc_global(batch)
        h_l = enc_local(batch)
    assert torch.allclose(h_g, h_l, atol=1e-5, rtol=1e-5)


def test_rope_theta_changes_outputs() -> None:
    """Different RoPE bases should produce different hidden states."""

    vocab_size = 200
    t = 16
    cfg_a = TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        max_seq_len=t,
        rope_theta_global=10000.0,
        rope_theta_local=10000.0,
        global_attn_every_n_layers=1,
    )
    cfg_b = TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        max_seq_len=t,
        rope_theta_global=160000.0,
        rope_theta_local=160000.0,
        global_attn_every_n_layers=1,
    )

    torch.manual_seed(0)
    enc_a = TextStateEncoder(cfg_a)
    enc_b = TextStateEncoder(cfg_b)
    enc_b.load_state_dict(enc_a.state_dict(), strict=False)
    batch = _make_batch(b=2, t=t, vocab_size=vocab_size)
    enc_a.eval()
    enc_b.eval()
    with torch.no_grad():
        h_a = enc_a(batch)
        h_b = enc_b(batch)
    # Same weights, same inputs, but rope theta differs ⇒ outputs differ.
    assert not torch.allclose(h_a, h_b, atol=1e-4)


def test_is_global_layer_pattern() -> None:
    from typing import cast

    from magic_ai.text_encoder.model import EncoderBlock

    cfg = TextEncoderConfig(
        vocab_size=100,
        d_model=32,
        n_layers=7,
        n_heads=4,
        d_ff=64,
        max_seq_len=32,
        global_attn_every_n_layers=3,
    )
    enc = TextStateEncoder(cfg)
    assert enc.is_global_layer == [True, False, False, True, False, False, True]
    # Per-block window is None on global layers, the configured window otherwise.
    windows = [cast(EncoderBlock, block).attn.window for block in enc.blocks]
    assert windows == [None, 128, 128, None, 128, 128, None]
