"""Supervised value pretrain + inline-blank BC training scaffolding.

Implements pure-PyTorch loss functions plus a thin
:class:`TextEncoderTrainer` that drives a :class:`RecurrentTextPolicy`
through a single optimizer step.

Out of scope here: data loading. Real data sources are plugged in via
``Dataset`` objects that yield ``TextEncodedBatch`` plus the relevant labels.

Conventions:

* Each call to :meth:`TextEncoderTrainer.value_step` resets the LSTM state.
  We treat every batch as an independent episode tail; multi-step BPTT is a
  follow-up.
* Inline-blank losses honour blank metadata emitted by
  :class:`magic_ai.text_encoder.batch.TextEncodedBatch`; padded blanks do not
  contribute gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy
from magic_ai.text_encoder.render_plan import (
    BLANK_GROUP_CONSTRAINED,
    BLANK_GROUP_CROSS_BLANK,
    BLANK_GROUP_PER_BLANK,
)

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def value_loss(values: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """Masked MSE between predicted ``values`` and scalar ``targets``.

    ``values`` and ``targets`` are ``[B]``. ``mask`` (optional) is a bool or
    float ``[B]`` selecting which examples contribute. If ``mask`` is omitted
    every example contributes equally.
    """

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


def _validate_inline_priority_shapes(
    blank_logits: Tensor,
    blank_group: Tensor,
    blank_group_kind: Tensor,
    blank_legal_mask: Tensor,
    target_blank_index: Tensor,
) -> None:
    if blank_logits.ndim != 3:
        raise ValueError(
            "inline_blank_priority_loss: blank_logits must be [B, K, V], "
            f"got {tuple(blank_logits.shape)}"
        )
    if blank_logits.shape[-1] < 1:
        raise ValueError("inline_blank_priority_loss: blank_logits needs at least one legal slot")
    if blank_group.shape != blank_logits.shape[:2]:
        raise ValueError(
            "inline_blank_priority_loss: blank_group shape mismatch "
            f"{tuple(blank_group.shape)} vs {tuple(blank_logits.shape[:2])}"
        )
    if blank_group_kind.shape != blank_logits.shape[:2]:
        raise ValueError(
            "inline_blank_priority_loss: blank_group_kind shape mismatch "
            f"{tuple(blank_group_kind.shape)} vs {tuple(blank_logits.shape[:2])}"
        )
    if blank_legal_mask.shape != blank_logits.shape:
        raise ValueError(
            "inline_blank_priority_loss: blank_legal_mask shape mismatch "
            f"{tuple(blank_legal_mask.shape)} vs {tuple(blank_logits.shape)}"
        )
    if target_blank_index.shape != blank_logits.shape[:1]:
        raise ValueError(
            "inline_blank_priority_loss: target_blank_index must be [B], "
            f"got {tuple(target_blank_index.shape)}"
        )


def _inline_priority_support_mask(
    blank_group: Tensor,
    blank_group_kind: Tensor,
    blank_legal_mask: Tensor,
    group_id: int,
) -> Tensor:
    """Return ``[B, K]`` mask for anchors participating in the priority softmax."""

    return (
        (blank_group == group_id)
        & (blank_group_kind == BLANK_GROUP_CROSS_BLANK)
        & blank_legal_mask[..., 0]
    )


def inline_blank_priority_loss(
    blank_logits: Tensor,
    blank_group: Tensor,
    blank_group_kind: Tensor,
    blank_legal_mask: Tensor,
    target_blank_index: Tensor,
    *,
    group_id: int = 0,
    ignore_index: int = -1,
) -> Tensor:
    """Cross-blank CE for priority inline blanks.

    Priority anchors are rendered as one ``CROSS_BLANK`` group. Each anchor has
    a singleton legal vocabulary headed by ``<chosen>``, so slot 0 of
    ``blank_logits`` is the score for selecting that anchor. Targets are blank
    ordinals within ``K``; rows with ``ignore_index`` targets or invalid target
    anchors are skipped.
    """

    _validate_inline_priority_shapes(
        blank_logits,
        blank_group,
        blank_group_kind,
        blank_legal_mask,
        target_blank_index,
    )

    support = _inline_priority_support_mask(
        blank_group,
        blank_group_kind,
        blank_legal_mask,
        group_id,
    )
    scores = blank_logits[..., 0]
    target = target_blank_index.to(device=scores.device, dtype=torch.long)

    in_range = (target >= 0) & (target < scores.shape[1]) & (target != ignore_index)
    safe_target = target.clamp(min=0, max=scores.shape[1] - 1)
    target_is_supported = support.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    valid_rows = in_range & target_is_supported & support.any(dim=1)
    if not valid_rows.any():
        return scores.sum() * 0.0

    neg_inf = torch.full_like(scores, float("-inf"))
    masked_scores = torch.where(support, scores, neg_inf)
    return F.cross_entropy(masked_scores[valid_rows], target[valid_rows])


@torch.no_grad()
def inline_blank_priority_accuracy(
    blank_logits: Tensor,
    blank_group: Tensor,
    blank_group_kind: Tensor,
    blank_legal_mask: Tensor,
    target_blank_index: Tensor,
    *,
    group_id: int = 0,
    ignore_index: int = -1,
) -> dict[str, float | int]:
    """Accuracy for the priority cross-blank categorical."""

    _validate_inline_priority_shapes(
        blank_logits,
        blank_group,
        blank_group_kind,
        blank_legal_mask,
        target_blank_index,
    )

    support = _inline_priority_support_mask(
        blank_group,
        blank_group_kind,
        blank_legal_mask,
        group_id,
    )
    scores = blank_logits[..., 0]
    target = target_blank_index.to(device=scores.device, dtype=torch.long)
    in_range = (target >= 0) & (target < scores.shape[1]) & (target != ignore_index)
    safe_target = target.clamp(min=0, max=scores.shape[1] - 1)
    target_is_supported = support.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    valid_rows = in_range & target_is_supported & support.any(dim=1)
    total = int(valid_rows.sum().item())
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    neg_inf = torch.full_like(scores, float("-inf"))
    masked_scores = torch.where(support, scores, neg_inf)
    pred = masked_scores[valid_rows].argmax(dim=-1)
    correct = int((pred == target[valid_rows]).sum().item())
    return {"accuracy": correct / total, "correct": correct, "total": total}


def _validate_inline_per_blank_shapes(
    blank_logits: Tensor,
    blank_group_kind: Tensor,
    blank_legal_mask: Tensor,
    target_legal_index: Tensor,
) -> None:
    if blank_logits.ndim != 3:
        raise ValueError(
            "inline_blank_per_blank_loss: blank_logits must be [B, K, V], "
            f"got {tuple(blank_logits.shape)}"
        )
    if blank_group_kind.shape != blank_logits.shape[:2]:
        raise ValueError(
            "inline_blank_per_blank_loss: blank_group_kind shape mismatch "
            f"{tuple(blank_group_kind.shape)} vs {tuple(blank_logits.shape[:2])}"
        )
    if blank_legal_mask.shape != blank_logits.shape:
        raise ValueError(
            "inline_blank_per_blank_loss: blank_legal_mask shape mismatch "
            f"{tuple(blank_legal_mask.shape)} vs {tuple(blank_logits.shape)}"
        )
    if target_legal_index.shape != blank_logits.shape[:2]:
        raise ValueError(
            "inline_blank_per_blank_loss: target_legal_index must be [B, K], "
            f"got {tuple(target_legal_index.shape)}"
        )


def _inline_per_blank_valid_slots(
    blank_group_kind: Tensor,
    blank_legal_mask: Tensor,
    target_legal_index: Tensor,
    ignore_index: int,
) -> Tensor:
    target = target_legal_index.to(device=blank_legal_mask.device, dtype=torch.long)
    group_kind = blank_group_kind.to(device=blank_legal_mask.device)
    support = (
        (group_kind == BLANK_GROUP_PER_BLANK) | (group_kind == BLANK_GROUP_CONSTRAINED)
    ) & blank_legal_mask.any(dim=-1)
    in_range = (target >= 0) & (target < blank_legal_mask.shape[-1]) & (target != ignore_index)
    safe_target = target.clamp(min=0, max=max(0, blank_legal_mask.shape[-1] - 1))
    target_is_legal = blank_legal_mask.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return support & in_range & target_is_legal


def inline_blank_per_blank_loss(
    blank_logits: Tensor,
    blank_group_kind: Tensor,
    blank_legal_mask: Tensor,
    target_legal_index: Tensor,
    *,
    ignore_index: int = -1,
) -> Tensor:
    """Per-blank CE for ``PER_BLANK`` and ``CONSTRAINED`` inline groups.

    ``target_legal_index`` stores the chosen legal-slot ordinal per blank
    (for blockers: 0 is ``<none>``, 1..N are legal attackers). Cross-blank
    groups are ignored; they are handled by :func:`inline_blank_priority_loss`.
    """

    _validate_inline_per_blank_shapes(
        blank_logits,
        blank_group_kind,
        blank_legal_mask,
        target_legal_index,
    )
    target = target_legal_index.to(device=blank_logits.device, dtype=torch.long)
    valid = _inline_per_blank_valid_slots(
        blank_group_kind,
        blank_legal_mask,
        target,
        ignore_index,
    )
    if not valid.any():
        return blank_logits.sum() * 0.0

    masked_logits = blank_logits.masked_fill(~blank_legal_mask, float("-inf"))
    return F.cross_entropy(masked_logits[valid], target[valid])


@torch.no_grad()
def inline_blank_per_blank_accuracy(
    blank_logits: Tensor,
    blank_group_kind: Tensor,
    blank_legal_mask: Tensor,
    target_legal_index: Tensor,
    *,
    ignore_index: int = -1,
) -> dict[str, float | int]:
    """Accuracy for ``PER_BLANK`` and ``CONSTRAINED`` inline groups."""

    _validate_inline_per_blank_shapes(
        blank_logits,
        blank_group_kind,
        blank_legal_mask,
        target_legal_index,
    )
    target = target_legal_index.to(device=blank_logits.device, dtype=torch.long)
    valid = _inline_per_blank_valid_slots(
        blank_group_kind,
        blank_legal_mask,
        target,
        ignore_index,
    )
    total = int(valid.sum().item())
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    pred = blank_logits.masked_fill(~blank_legal_mask, float("-inf"))[valid].argmax(dim=-1)
    correct = int((pred == target[valid]).sum().item())
    return {"accuracy": correct / total, "correct": correct, "total": total}


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
            # Compute global grad norm for logging even when not clipping.
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
    "inline_blank_per_blank_accuracy",
    "inline_blank_per_blank_loss",
    "inline_blank_priority_accuracy",
    "inline_blank_priority_loss",
    "value_loss",
]
