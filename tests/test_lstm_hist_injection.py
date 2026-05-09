"""Tests for the LSTM-history additive injection at encoder BOS positions.

Verifies that ``RecurrentTextPolicy.encode_with_history`` actually conditions
the encoder hidden states on the carried LSTM state — fixing the train-time
vs sample-time forward-pass mismatch where the LSTM's output never reached
the encoder.
"""

from __future__ import annotations

import torch
from magic_ai.text_encoder.batch import TextEncodedBatch, pack_batch
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


def _make_batch(b: int = 2, t: int = 12, vocab_size: int = 200, seed: int = 0) -> TextEncodedBatch:
    torch.manual_seed(seed)
    token_ids = torch.randint(low=1, high=vocab_size, size=(b, t), dtype=torch.int64)
    seq_lens = torch.tensor([t, t - 4][:b], dtype=torch.int64)
    attention_mask = torch.zeros(b, t, dtype=torch.int64)
    for i, n in enumerate(seq_lens.tolist()):
        attention_mask[i, :n] = 1
        token_ids[i, n:] = 0
    card_ref_positions = torch.full((b, MAX_CARD_REFS), -1, dtype=torch.int64)
    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lens,
    )


def _make_policy(
    vocab_size: int = 200, lstm_hidden: int = 32, lstm_layers: int = 1
) -> RecurrentTextPolicy:
    enc = TextEncoderConfig(
        vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=12
    )
    cfg = RecurrentTextPolicyConfig(encoder=enc, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers)
    return RecurrentTextPolicy(cfg)


def _force_nontrivial_hist_proj(policy: RecurrentTextPolicy) -> None:
    """The hist_proj is zero-initialized so existing checkpoints aren't broken;
    re-initialize it to a non-trivial value for tests that want to observe the
    injection effect."""
    torch.nn.init.normal_(policy.hist_proj.weight, std=0.5)
    torch.nn.init.normal_(policy.hist_proj.bias, std=0.1)


def test_zero_init_hist_proj_means_zero_hist_at_construction() -> None:
    # Sanity: with the default zero-init hist_proj, encode_with_history with a
    # random h_in should give the SAME encoded as a no-hist forward.
    policy = _make_policy()
    policy.eval()
    batch = _make_batch()
    packed = pack_batch(batch)
    h_in = torch.randn(policy.lstm_layers, batch.token_ids.shape[0], policy.lstm_hidden)
    c_in = torch.zeros_like(h_in)

    enc_with, _, _ = policy.encode_with_history(packed, h_in=h_in, c_in=c_in)
    # Compare against encode_packed_only directly (no hist).
    enc_baseline = policy.text_policy.encode_packed_only(packed)
    assert torch.allclose(enc_with.encoded, enc_baseline.encoded)


def test_hist_injection_changes_encoded_states() -> None:
    policy = _make_policy()
    policy.eval()
    _force_nontrivial_hist_proj(policy)
    batch = _make_batch()
    packed = pack_batch(batch)
    b = batch.token_ids.shape[0]
    h_zero = torch.zeros(policy.lstm_layers, b, policy.lstm_hidden)
    c_zero = torch.zeros_like(h_zero)
    h_ones = torch.ones_like(h_zero)

    enc_zero, h_out_zero, _ = policy.encode_with_history(packed, h_in=h_zero, c_in=c_zero)
    enc_ones, h_out_ones, _ = policy.encode_with_history(packed, h_in=h_ones, c_in=c_zero)

    # Encoder hidden states differ → hist actually conditioned the encoder.
    assert not torch.allclose(enc_zero.encoded, enc_ones.encoded)
    # LSTM outputs also differ since (a) state_vector now differs and (b) h_in
    # itself differs.
    assert not torch.allclose(h_out_zero, h_out_ones)


def test_encoded_shape_unchanged_by_hist_injection() -> None:
    policy = _make_policy()
    policy.eval()
    _force_nontrivial_hist_proj(policy)
    batch = _make_batch()
    packed = pack_batch(batch)
    b = batch.token_ids.shape[0]
    h_in = torch.randn(policy.lstm_layers, b, policy.lstm_hidden)
    c_in = torch.zeros_like(h_in)

    enc_with, _, _ = policy.encode_with_history(packed, h_in=h_in, c_in=c_in)
    enc_baseline = policy.text_policy.encode_packed_only(packed)
    # Hist is an additive modulation, NOT a new token → packed shape preserved.
    assert enc_with.encoded.shape == enc_baseline.encoded.shape
    assert enc_with.state_vector.shape == enc_baseline.state_vector.shape


def test_padded_path_matches_packed_path_with_hist() -> None:
    # encoder_forward_padded_with_history (sample-time) must use the same
    # additive injection as encode_with_history (train-time replay scoring).
    policy = _make_policy()
    policy.eval()
    _force_nontrivial_hist_proj(policy)
    batch = _make_batch()
    packed = pack_batch(batch)
    b = batch.token_ids.shape[0]
    h_in = torch.randn(policy.lstm_layers, b, policy.lstm_hidden)
    c_in = torch.zeros_like(h_in)

    encoded_padded, h_out_padded, _ = policy.encoder_forward_padded_with_history(
        batch, h_in=h_in, c_in=c_in
    )
    encoded_packed, h_out_packed, _ = policy.encode_with_history(packed, h_in=h_in, c_in=c_in)
    # The state_vector (BOS) hidden must agree between padded and packed paths
    # — that's the exact column that drove the LSTM in both, so h_out matches.
    assert torch.allclose(h_out_padded, h_out_packed, atol=1e-5)
    # The padded encoded[:, 0, :] equals packed encoded[state_positions].
    bos_padded = encoded_padded[:, 0, :]
    bos_packed = encoded_packed.encoded.index_select(0, packed.state_positions)
    assert torch.allclose(bos_padded, bos_packed, atol=1e-5)
