"""Tests for the autoregressive grammar-decoder sampler / replay scorer.

Scoped to the helpers added in Phase 5 of the inline-blank cutover. The
end-to-end ``LSTMStatefulTextPolicy`` integration lives in Phase 6's tests.
"""

from __future__ import annotations

from typing import cast

import torch
from magic_ai.text_encoder.decision_spec import (
    AnchorKind,
    DecisionType,
)
from magic_ai.text_encoder.decoder import (
    GrammarDecoder,
    GrammarDecoderConfig,
)
from magic_ai.text_encoder.decoder_inference import (
    decoder_sample,
    decoder_score_replay,
)
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE
from magic_ai.text_encoder.policy import TextPolicy


class _StubTextPolicy:
    """Minimal stand-in for ``TextPolicy`` exposing the grammar decoder only.

    The sampler and scorer only ever touch ``text_policy.grammar_decoder``;
    constructing a full ``TextPolicy`` requires the ModernBERT trunk + HF
    weights, which is overkill for grammar tests.
    """

    def __init__(self, decoder: GrammarDecoder) -> None:
        self.grammar_decoder = decoder


def _make_decoder(d_model: int = 16) -> GrammarDecoder:
    cfg = GrammarDecoderConfig(
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0.0,
        max_decode_len=8,
    )
    torch.manual_seed(0)
    return GrammarDecoder(cfg)


def _priority_batch(b: int, t_enc: int, n_actions: int):
    """Build encoder-side tensors for a batch of PRIORITY decisions.

    Anchors live at encoder positions 1..n_actions inclusive (position 0 is
    treated as a generic non-anchor token).
    """

    decision_type = torch.full((b,), int(DecisionType.PRIORITY), dtype=torch.long)
    n_max = n_actions
    pointer_anchor_positions = torch.full((b, n_max), -1, dtype=torch.long)
    pointer_anchor_kinds = torch.full((b, n_max), -1, dtype=torch.long)
    pointer_anchor_subjects = torch.full((b, n_max), -1, dtype=torch.long)
    pointer_anchor_handles = torch.full((b, n_max), -1, dtype=torch.long)
    for i in range(b):
        for j in range(n_actions):
            pointer_anchor_positions[i, j] = j + 1  # 1..n_actions
            pointer_anchor_kinds[i, j] = int(AnchorKind.LEGAL_ACTION)
            pointer_anchor_subjects[i, j] = j
            pointer_anchor_handles[i, j] = j  # handle == option index
    encoder_attention_mask = torch.ones((b, t_enc), dtype=torch.bool)
    return (
        decision_type,
        pointer_anchor_positions,
        pointer_anchor_kinds,
        pointer_anchor_subjects,
        pointer_anchor_handles,
        encoder_attention_mask,
    )


def test_decoder_sample_priority_shapes_and_pad_mask():
    b, t_enc, d_model, n_actions = 3, 6, 16, 2
    decoder = _make_decoder(d_model)
    text_policy = _StubTextPolicy(decoder)
    encoded = torch.randn(b, t_enc, d_model)
    (
        decision_type,
        positions,
        kinds,
        subjects,
        handles,
        attn_mask,
    ) = _priority_batch(b, t_enc, n_actions)

    out = decoder_sample(
        cast(TextPolicy, text_policy),
        encoded,
        attn_mask,
        decision_type,
        positions,
        kinds,
        subjects,
        handles,
        max_decode_len=8,
        greedy=True,
    )

    assert out.output_token_ids.shape == (b, 8)
    assert out.output_pointer_pos.shape == (b, 8)
    assert out.output_is_pointer.shape == (b, 8)
    assert out.output_pad_mask.shape == (b, 8)
    assert out.log_probs.shape == (b, 8)

    # PRIORITY produces exactly 3 supervised steps: OPEN, pointer, END.
    # Steps after END must be padded (False).
    for i in range(b):
        valid_steps = int(out.output_pad_mask[i].sum().item())
        assert valid_steps == 3, f"row {i}: expected 3 valid steps, got {valid_steps}"
        # Tail steps are padding.
        assert not bool(out.output_pad_mask[i, 3:].any())
        # Log probs at pad steps are zeroed.
        assert torch.all(out.log_probs[i, 3:] == 0.0)
        # Pointer step is step index 1.
        assert bool(out.output_is_pointer[i, 1])
        # Pointer position lands on one of the legal anchors.
        chosen_pos = int(out.output_pointer_pos[i, 1].item())
        assert chosen_pos in {1, 2}, f"row {i}: pointer outside anchors, got {chosen_pos}"


def test_decoder_score_replay_returns_finite_scalar_per_row():
    b, t_enc, d_model, L = 2, 5, 16, 4
    decoder = _make_decoder(d_model)
    text_policy = _StubTextPolicy(decoder)
    encoded = torch.randn(b, t_enc, d_model)
    encoder_attention_mask = torch.ones((b, t_enc), dtype=torch.bool)

    # Construct a tiny made-up target sequence: 3 steps (token, pointer, token)
    # then a padded step. Masks are all-True so the log-softmax falls back to
    # the unmasked distribution, which is enough to verify the reduction
    # plumbing.
    target_tokens = torch.zeros((b, L), dtype=torch.long)
    target_tokens[:, 0] = 2  # any non-pad token
    target_tokens[:, 2] = 1  # END
    target_pointer_pos = torch.zeros((b, L), dtype=torch.long)
    target_pointer_pos[:, 1] = 3  # an encoder position
    is_pointer_step = torch.tensor([[False, True, False, False]] * b, dtype=torch.bool)
    pad_mask = torch.tensor([[True, True, True, False]] * b, dtype=torch.bool)
    vocab_mask = torch.ones((b, L, GRAMMAR_VOCAB_SIZE), dtype=torch.bool)
    pointer_mask = torch.ones((b, L, t_enc), dtype=torch.bool)

    scores = decoder_score_replay(
        cast(TextPolicy, text_policy),
        encoded,
        encoder_attention_mask,
        target_tokens,
        target_pointer_pos,
        is_pointer_step,
        pad_mask,
        vocab_mask,
        pointer_mask,
    )

    assert scores.per_row_log_pi.shape == (b,)
    assert scores.per_row_entropy.shape == (b,)
    assert scores.per_step_log_pi.shape == (b, L)
    assert torch.all(torch.isfinite(scores.per_row_log_pi))
    assert torch.all(torch.isfinite(scores.per_row_entropy))
    # Log-probs should be non-positive under any well-defined log-softmax.
    assert torch.all(scores.per_row_log_pi <= 1e-5)
    # Pad-step log probs should be zero.
    assert torch.all(scores.per_step_log_pi[:, 3] == 0.0)
