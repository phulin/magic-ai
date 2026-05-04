"""Smoke tests for the v1 history adapter (LSTM around state_vector)."""

from __future__ import annotations

import torch
from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


def _make_batch(
    b: int = 2,
    t: int = 20,
    max_opts: int = 4,
    max_targets: int = 3,
    vocab_size: int = 1000,
    seed: int = 0,
) -> TextEncodedBatch:
    torch.manual_seed(seed)
    token_ids = torch.randint(low=1, high=vocab_size, size=(b, t), dtype=torch.int64)
    seq_lens = torch.tensor([t, t - 8], dtype=torch.int64)
    attention_mask = torch.zeros(b, t, dtype=torch.int64)
    for i, n in enumerate(seq_lens.tolist()):
        attention_mask[i, :n] = 1
        token_ids[i, n:] = 0  # pad_id

    card_ref_positions = torch.full((b, MAX_CARD_REFS), -1, dtype=torch.int64)
    card_ref_positions[0, :3] = torch.tensor([2, 5, 8])
    card_ref_positions[1, :2] = torch.tensor([3, 6])

    option_positions = torch.full((b, max_opts), -1, dtype=torch.int64)
    option_positions[0, :3] = torch.tensor([10, 13, 16])
    option_positions[1, :2] = torch.tensor([4, 8])
    option_mask = option_positions >= 0

    target_positions = torch.full((b, max_opts, max_targets), -1, dtype=torch.int64)
    target_positions[0, 0, :2] = torch.tensor([11, 12])
    target_positions[0, 1, :1] = torch.tensor([14])
    target_positions[1, 0, :3] = torch.tensor([5, 6, 7])
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


def _make_policy(
    vocab_size: int = 1000, lstm_hidden: int = 32, lstm_layers: int = 1
) -> RecurrentTextPolicy:
    enc = TextEncoderConfig(
        vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=4, d_ff=64, max_seq_len=20
    )
    cfg = RecurrentTextPolicyConfig(encoder=enc, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers)
    return RecurrentTextPolicy(cfg)


def test_recurrent_forward_and_state_shape() -> None:
    vocab_size = 1000
    lstm_hidden = 32
    lstm_layers = 1
    policy = _make_policy(vocab_size=vocab_size, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers)
    batch = _make_batch(vocab_size=vocab_size)

    out, (h_out, c_out) = policy(batch, h_in=None, c_in=None)

    b = batch.token_ids.shape[0]
    d = policy.cfg.encoder.d_model

    assert out.values.shape == (b,)
    assert out.state_hidden.shape == (b, lstm_hidden)
    assert out.card_vectors.shape == (b, MAX_CARD_REFS, d)
    assert out.card_mask.shape == (b, MAX_CARD_REFS)

    assert h_out.shape == (lstm_layers, b, lstm_hidden)
    assert c_out.shape == (lstm_layers, b, lstm_hidden)

    assert torch.isfinite(out.values).all()
    assert torch.isfinite(out.state_hidden).all()


def test_recurrent_state_persists_across_calls() -> None:
    vocab_size = 1000
    policy = _make_policy(vocab_size=vocab_size)
    policy.eval()  # disable dropout determinism noise (none today, but safe)

    batch1 = _make_batch(vocab_size=vocab_size, seed=0)
    batch2 = _make_batch(vocab_size=vocab_size, seed=1)

    out1, (h1, c1) = policy(batch1, h_in=None, c_in=None)
    out2, (h2, c2) = policy(batch2, h_in=h1, c_in=c1)

    # Step-2 hidden must differ from step-1 (the LSTM is doing work) and from
    # what we would have got with a zero initial state.
    out2_zero, _ = policy(batch2, h_in=None, c_in=None)
    assert not torch.allclose(out2.state_hidden, out1.state_hidden)
    assert not torch.allclose(out2.state_hidden, out2_zero.state_hidden)

    # Hidden state should evolve.
    assert not torch.allclose(h2, h1)


def test_recurrent_backward_smoke() -> None:
    vocab_size = 1000
    policy = _make_policy(vocab_size=vocab_size)
    batch = _make_batch(vocab_size=vocab_size)

    out, _ = policy(batch)
    loss = out.values.sum()
    loss.backward()

    lstm_grads = [
        p.grad.detach().abs().sum().item() for p in policy.lstm.parameters() if p.grad is not None
    ]
    encoder_grads = [
        p.grad.detach().abs().sum().item()
        for p in policy.text_policy.encoder.parameters()
        if p.grad is not None
    ]
    assert lstm_grads and any(g > 0 for g in lstm_grads), "no LSTM gradient"
    assert encoder_grads and any(g > 0 for g in encoder_grads), "no encoder gradient"


def test_recurrent_init_state_zeros() -> None:
    policy = _make_policy(lstm_hidden=128, lstm_layers=2)
    device = torch.device("cpu")
    h, c = policy.init_state(batch_size=5, device=device)
    assert h.shape == (2, 5, 128)
    assert c.shape == (2, 5, 128)
    assert torch.all(h == 0)
    assert torch.all(c == 0)
    assert h.device == device
    assert c.device == device
