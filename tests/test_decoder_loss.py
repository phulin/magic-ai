"""Smoke tests for ``decoder_cross_entropy_loss`` and per-step accuracy."""

from __future__ import annotations

import torch
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, GrammarVocab
from magic_ai.text_encoder.training import (
    decoder_cross_entropy_loss,
    decoder_per_step_accuracy,
)


def _make_inputs(B: int = 2, L: int = 3, T_enc: int = 8) -> dict[str, torch.Tensor]:
    V = GRAMMAR_VOCAB_SIZE
    torch.manual_seed(0)
    vocab_logits = torch.randn(B, L, V, requires_grad=True)
    pointer_logits = torch.randn(B, L, T_enc, requires_grad=True)
    # Mixed steps: row 0 = [vocab, vocab, pointer], row 1 = [vocab, pointer, pad]
    target_tokens = torch.tensor(
        [
            [GrammarVocab.PRIORITY_OPEN, GrammarVocab.END, 0],
            [GrammarVocab.MAY_OPEN, 0, 0],
        ],
        dtype=torch.int32,
    )
    target_pointer_pos = torch.tensor([[0, 0, 3], [0, 5, 0]], dtype=torch.int32)
    is_pointer_step = torch.tensor([[False, False, True], [False, True, False]])
    pad_mask = torch.tensor([[True, True, True], [True, True, False]])
    vocab_mask = torch.ones(B, L, V, dtype=torch.bool)
    pointer_mask = torch.ones(B, L, T_enc, dtype=torch.bool)
    return {
        "vocab_logits": vocab_logits,
        "pointer_logits": pointer_logits,
        "target_tokens": target_tokens,
        "target_pointer_pos": target_pointer_pos,
        "is_pointer_step": is_pointer_step,
        "vocab_mask": vocab_mask,
        "pointer_mask": pointer_mask,
        "pad_mask": pad_mask,
    }


def test_decoder_loss_finite_and_grads_flow() -> None:
    inputs = _make_inputs()
    loss = decoder_cross_entropy_loss(**inputs)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert inputs["vocab_logits"].grad is not None
    assert inputs["pointer_logits"].grad is not None


def test_decoder_loss_zero_when_only_pad() -> None:
    inputs = _make_inputs()
    inputs["pad_mask"] = torch.zeros_like(inputs["pad_mask"])
    loss = decoder_cross_entropy_loss(**inputs)
    assert float(loss) == 0.0


def test_decoder_per_step_accuracy_perfect() -> None:
    inputs = _make_inputs()
    # Build logits that perfectly score the targets (one-hot)
    inputs["vocab_logits"] = torch.full_like(inputs["vocab_logits"], -1e6)
    inputs["pointer_logits"] = torch.full_like(inputs["pointer_logits"], -1e6)
    B, L, _ = inputs["vocab_logits"].shape
    for b in range(B):
        for t in range(L):
            if not inputs["pad_mask"][b, t]:
                continue
            if inputs["is_pointer_step"][b, t]:
                inputs["pointer_logits"][b, t, int(inputs["target_pointer_pos"][b, t])] = 1e6
            else:
                inputs["vocab_logits"][b, t, int(inputs["target_tokens"][b, t])] = 1e6
    metrics = decoder_per_step_accuracy(**inputs)
    assert metrics["accuracy"] == 1.0
    assert metrics["total"] == int(inputs["pad_mask"].sum().item())


def test_decoder_loss_respects_legality_mask() -> None:
    """Masking the correct vocab id should NOT make CE explode to inf
    if no other legal column is True (loss returns -inf on -inf logsumexp).
    Realistic invariant: mask always includes at least the target column."""
    inputs = _make_inputs()
    # Zero out one illegal column on every step — loss should still be finite.
    inputs["vocab_mask"] = torch.ones_like(inputs["vocab_mask"])
    inputs["vocab_mask"][..., 0] = False  # PAD = 0 is never the target above
    loss = decoder_cross_entropy_loss(**inputs)
    assert torch.isfinite(loss)
