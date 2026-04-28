"""Supervised value pretrain + BC distillation training scaffolding.

Implements §7 steps 3 & 4 of ``docs/text_encoder_plan.md`` at the *plumbing*
level: pure-PyTorch loss functions plus a thin :class:`TextEncoderTrainer`
that drives a :class:`RecurrentTextPolicy` through a single optimizer step.

Out of scope here: data loading. Real data sources (rollout buffer for value
pretrain, slot-policy teacher for BC distillation) are plugged in later via
``Dataset`` objects that yield ``TextEncodedBatch`` plus the relevant labels;
the trainer API is shaped so swapping in real data is purely a Dataset-level
change.

Conventions:

* Each call to :meth:`TextEncoderTrainer.value_step` /
  :meth:`TextEncoderTrainer.distill_step` resets the LSTM state. We treat
  every batch as an independent episode tail; multi-step BPTT is a follow-up.
* All losses honour the option / target masks emitted by
  :class:`magic_ai.text_encoder.batch.TextEncodedBatch`. Padding rows do not
  contribute gradient.
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


def _masked_log_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    """log_softmax restricted to positions where ``mask`` is True.

    Positions where ``mask`` is False are forced to ``-inf`` *before* the
    softmax, so they contribute zero probability and zero gradient.
    """

    neg_inf = torch.full_like(logits, float("-inf"))
    masked = torch.where(mask, logits, neg_inf)
    return F.log_softmax(masked, dim=dim)


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    neg_inf = torch.full_like(logits, float("-inf"))
    masked = torch.where(mask, logits, neg_inf)
    return F.softmax(masked, dim=dim)


def policy_distillation_loss(
    student_logits: Tensor, teacher_logits: Tensor, mask: Tensor
) -> Tensor:
    """Masked KL(teacher || student) over the option dimension.

    Shapes: ``student_logits``, ``teacher_logits`` both ``[B, max_opts]``;
    ``mask`` ``[B, max_opts]`` bool. Rows where no option is valid contribute
    zero loss; padded option slots within a row contribute zero (their
    softmax probability is zero on both sides).
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "policy_distillation_loss: student/teacher logits shape mismatch "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if mask.shape != student_logits.shape:
        raise ValueError(
            "policy_distillation_loss: mask shape mismatch "
            f"{tuple(mask.shape)} vs {tuple(student_logits.shape)}"
        )

    log_student = _masked_log_softmax(student_logits, mask, dim=-1)
    log_teacher = _masked_log_softmax(teacher_logits, mask, dim=-1)
    teacher_p = log_teacher.exp()

    # KL(teacher || student) = sum_i teacher_p_i * (log_teacher_i - log_student_i)
    # Masked positions have teacher_p = 0 so they drop out cleanly. Replace any
    # NaNs introduced by 0 * -inf with zeros.
    diff = log_teacher - log_student
    contrib = teacher_p * diff
    contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)

    per_row_kl = contrib.sum(dim=-1)
    has_any = mask.any(dim=-1)
    if not has_any.any():
        return per_row_kl.sum() * 0.0
    return per_row_kl[has_any].mean()


def target_distillation_loss(
    student_logits: Tensor, teacher_logits: Tensor, mask: Tensor
) -> Tensor:
    """Per-option masked KL on the target axis.

    Shapes: ``student_logits``, ``teacher_logits`` both
    ``[B, max_opts, max_targets]``; ``mask`` ``[B, max_opts, max_targets]``
    bool. KL is computed along the targets axis for each (batch, option),
    then averaged over options where at least one target is valid.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "target_distillation_loss: student/teacher logits shape mismatch "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if mask.shape != student_logits.shape:
        raise ValueError(
            "target_distillation_loss: mask shape mismatch "
            f"{tuple(mask.shape)} vs {tuple(student_logits.shape)}"
        )

    log_student = _masked_log_softmax(student_logits, mask, dim=-1)
    log_teacher = _masked_log_softmax(teacher_logits, mask, dim=-1)
    teacher_p = log_teacher.exp()

    contrib = teacher_p * (log_teacher - log_student)
    contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
    per_option_kl = contrib.sum(dim=-1)  # [B, max_opts]

    valid_options = mask.any(dim=-1)  # [B, max_opts]
    if not valid_options.any():
        return per_option_kl.sum() * 0.0
    return per_option_kl[valid_options].mean()


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

    def distill_step(
        self,
        batch: TextEncodedBatch,
        teacher_policy_logits: Tensor,
        teacher_target_logits: Tensor,
        value_targets: Tensor | None = None,
        policy_weight: float = 1.0,
        target_weight: float = 1.0,
        value_weight: float = 0.5,
    ) -> dict[str, float]:
        self.policy.train()
        out, _ = self.policy(batch, h_in=None, c_in=None)

        p_loss = policy_distillation_loss(
            out.policy_logits, teacher_policy_logits, batch.option_mask
        )
        t_loss = target_distillation_loss(
            out.target_logits, teacher_target_logits, batch.target_mask
        )
        total = policy_weight * p_loss + target_weight * t_loss

        v_loss_val = 0.0
        if value_targets is not None:
            v_loss = value_loss(out.values, value_targets)
            total = total + value_weight * v_loss
            v_loss_val = float(v_loss.detach().item())

        gn = self._step_with_clip(total)
        return {
            "loss": float(total.detach().item()),
            "policy_loss": float(p_loss.detach().item()),
            "target_loss": float(t_loss.detach().item()),
            "value_loss": v_loss_val,
            "grad_norm": gn,
        }


__all__ = [
    "TextEncoderTrainer",
    "policy_distillation_loss",
    "target_distillation_loss",
    "value_loss",
]
