"""Supervised value pretrain + grammar-decoder BC training scaffolding.

Implements pure-PyTorch loss functions plus a thin
:class:`TextEncoderTrainer` that drives a :class:`RecurrentTextPolicy`
through a single optimizer step.

Out of scope here: data loading. Real data sources are plugged in via
``Dataset`` objects that yield ``TextEncodedBatch`` plus the relevant labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def value_loss(values: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """Masked MSE between predicted ``values`` and scalar ``targets``."""

    if values.shape != targets.shape:
        raise ValueError(
            f"value_loss: shape mismatch values={tuple(values.shape)} "
            f"targets={tuple(targets.shape)}"
        )
    sq = (values - targets).pow(2)
    if mask is None:
        return sq.mean()
    m = mask.to(sq.dtype)
    denom = m.sum().clamp(min=1.0)
    return (sq * m).sum() / denom


# ---------------------------------------------------------------------------
# Grammar-decoder loss (decoder-pipeline path; see docs/decoder_grammar_plan.md).
# ---------------------------------------------------------------------------


def decoder_cross_entropy_loss(
    vocab_logits: Tensor,
    pointer_logits: Tensor,
    target_tokens: Tensor,
    target_pointer_pos: Tensor,
    is_pointer_step: Tensor,
    vocab_mask: Tensor,
    pointer_mask: Tensor,
    pad_mask: Tensor,
) -> Tensor:
    """Teacher-forced CE over the autoregressive decoder output."""

    B, L, V = vocab_logits.shape
    T_enc = pointer_logits.shape[-1]

    neg_inf = torch.finfo(vocab_logits.dtype).min
    vocab_logits_masked = vocab_logits.masked_fill(~vocab_mask, neg_inf)
    pointer_logits_masked = pointer_logits.masked_fill(~pointer_mask, neg_inf)

    valid = pad_mask.reshape(-1)
    vocab_step = (~is_pointer_step) & pad_mask
    pointer_step = is_pointer_step & pad_mask

    vocab_flat = vocab_logits_masked.reshape(B * L, V)
    pointer_flat = pointer_logits_masked.reshape(B * L, T_enc)
    vocab_target = target_tokens.reshape(-1).to(torch.long)
    pointer_target = target_pointer_pos.reshape(-1).to(torch.long)

    vocab_sel = vocab_step.reshape(-1)
    pointer_sel = pointer_step.reshape(-1)

    losses = []
    if vocab_sel.any():
        losses.append(
            F.cross_entropy(vocab_flat[vocab_sel], vocab_target[vocab_sel], reduction="sum")
        )
    if pointer_sel.any():
        losses.append(
            F.cross_entropy(pointer_flat[pointer_sel], pointer_target[pointer_sel], reduction="sum")
        )
    if not losses:
        return vocab_logits.new_zeros(())
    n_valid = valid.sum().clamp_min(1)
    return torch.stack(losses).sum() / n_valid


def decoder_per_step_accuracy(
    vocab_logits: Tensor,
    pointer_logits: Tensor,
    target_tokens: Tensor,
    target_pointer_pos: Tensor,
    is_pointer_step: Tensor,
    vocab_mask: Tensor,
    pointer_mask: Tensor,
    pad_mask: Tensor,
) -> dict[str, float | int]:
    """Per-step argmax accuracy over the masked logits."""

    neg_inf = torch.finfo(vocab_logits.dtype).min
    vocab_pred = vocab_logits.masked_fill(~vocab_mask, neg_inf).argmax(dim=-1)
    pointer_pred = pointer_logits.masked_fill(~pointer_mask, neg_inf).argmax(dim=-1)

    correct_per_step = torch.where(
        is_pointer_step,
        pointer_pred == target_pointer_pos.to(torch.long),
        vocab_pred == target_tokens.to(torch.long),
    )
    valid = pad_mask
    correct = int((correct_per_step & valid).sum().item())
    total = int(valid.sum().item())
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TextEncoderTrainer:
    """Wraps a :class:`RecurrentTextPolicy` with AdamW + clip + step plumbing."""

    def __init__(
        self,
        policy: RecurrentTextPolicy,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        grad_clip: float | None = 1.0,
    ) -> None:
        self.policy = policy
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    def _step_with_clip(self, loss: Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.grad_clip)
        else:
            total_sq = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    total_sq += float(p.grad.detach().pow(2).sum().item())
            grad_norm = torch.tensor(total_sq**0.5)
        self.optimizer.step()
        return float(grad_norm)

    def value_step(self, batch: TextEncodedBatch, value_targets: Tensor) -> dict[str, float]:
        self.policy.train()
        out, _ = self.policy(batch, h_in=None, c_in=None)
        loss = value_loss(out.values, value_targets)
        gn = self._step_with_clip(loss)
        return {"value_loss": float(loss.detach().item()), "grad_norm": gn}


__all__ = [
    "TextEncoderTrainer",
    "decoder_cross_entropy_loss",
    "decoder_per_step_accuracy",
    "value_loss",
]
