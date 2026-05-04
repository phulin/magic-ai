"""Tests for ``magic_ai.text_encoder.training``.

Covers loss correctness (KL=0 when student==teacher, KL>0 when distributions
disagree), mask correctness (padding-only positions don't contribute), and a
single-step smoke that verifies :class:`TextEncoderTrainer` actually backprops
through :class:`RecurrentTextPolicy`.
"""

from __future__ import annotations

import torch
from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
)
from magic_ai.text_encoder.render_plan import (
    BLANK_GROUP_CONSTRAINED,
    BLANK_GROUP_CROSS_BLANK,
    BLANK_GROUP_PER_BLANK,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS
from magic_ai.text_encoder.training import (
    TextEncoderTrainer,
    inline_blank_per_blank_accuracy,
    inline_blank_per_blank_loss,
    inline_blank_priority_accuracy,
    inline_blank_priority_loss,
    policy_distillation_loss,
    target_distillation_loss,
    value_loss,
)

# ---------------------------------------------------------------------------
# Loss correctness
# ---------------------------------------------------------------------------


def test_policy_distillation_zero_when_student_equals_teacher() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    loss = policy_distillation_loss(logits.clone(), logits.clone(), mask)
    assert loss.item() < 1e-6


def test_policy_distillation_positive_when_disagreeing() -> None:
    student = torch.tensor([[10.0, -10.0, -10.0]])
    teacher = torch.tensor([[-10.0, -10.0, 10.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss = policy_distillation_loss(student, teacher, mask)
    assert loss.item() > 1.0


def test_target_distillation_zero_when_equal() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    loss = target_distillation_loss(logits.clone(), logits.clone(), mask)
    assert loss.item() < 1e-6


def test_target_distillation_positive_when_disagreeing() -> None:
    student = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])
    teacher = torch.tensor([[[-10.0, 10.0], [10.0, -10.0]]])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    loss = target_distillation_loss(student, teacher, mask)
    assert loss.item() > 1.0


def test_value_loss_basic() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    targ = torch.tensor([0.0, 2.0, 4.0])
    loss = value_loss(pred, targ)
    # (1 + 0 + 1) / 3 == 2/3
    assert abs(loss.item() - 2.0 / 3.0) < 1e-6


def test_value_loss_mask() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    targ = torch.tensor([0.0, 2.0, 4.0])
    mask = torch.tensor([1.0, 0.0, 1.0])
    # Only first and third contribute: (1 + 1) / 2 = 1.0
    loss = value_loss(pred, targ, mask)
    assert abs(loss.item() - 1.0) < 1e-6


def test_inline_blank_priority_loss_low_for_correct_anchor() -> None:
    logits = torch.tensor([[[0.0], [5.0], [-2.0]]], requires_grad=True)
    group = torch.zeros((1, 3), dtype=torch.int64)
    group_kind = torch.full((1, 3), BLANK_GROUP_CROSS_BLANK, dtype=torch.int64)
    legal_mask = torch.ones((1, 3, 1), dtype=torch.bool)
    target = torch.tensor([1])

    loss = inline_blank_priority_loss(logits, group, group_kind, legal_mask, target)

    assert loss.item() < 0.01
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, :, 0].abs().sum().item() > 0.0


def test_inline_blank_priority_loss_positive_for_wrong_anchor() -> None:
    logits = torch.tensor([[[0.0], [5.0], [-2.0]]])
    group = torch.zeros((1, 3), dtype=torch.int64)
    group_kind = torch.full((1, 3), BLANK_GROUP_CROSS_BLANK, dtype=torch.int64)
    legal_mask = torch.ones((1, 3, 1), dtype=torch.bool)
    target = torch.tensor([2])

    loss = inline_blank_priority_loss(logits, group, group_kind, legal_mask, target)

    assert loss.item() > 6.0


def test_inline_blank_priority_ignores_padding_and_non_cross_blanks() -> None:
    logits = torch.tensor([[[0.0], [4.0], [100.0], [200.0]]], requires_grad=True)
    group = torch.zeros((1, 4), dtype=torch.int64)
    group_kind = torch.tensor(
        [
            [
                BLANK_GROUP_CROSS_BLANK,
                BLANK_GROUP_CROSS_BLANK,
                BLANK_GROUP_PER_BLANK,
                BLANK_GROUP_CROSS_BLANK,
            ]
        ]
    )
    legal_mask = torch.tensor([[[True], [True], [True], [False]]])
    target = torch.tensor([1])

    loss = inline_blank_priority_loss(logits, group, group_kind, legal_mask, target)
    loss.backward()

    assert loss.item() < 0.02
    assert logits.grad is not None
    assert logits.grad[0, 0, 0].abs().item() > 0.0
    assert logits.grad[0, 1, 0].abs().item() > 0.0
    assert logits.grad[0, 2, 0].item() == 0.0
    assert logits.grad[0, 3, 0].item() == 0.0


def test_inline_blank_priority_ignores_invalid_target_rows() -> None:
    logits = torch.tensor([[[0.0], [5.0]], [[1.0], [2.0]]], requires_grad=True)
    group = torch.zeros((2, 2), dtype=torch.int64)
    group_kind = torch.full((2, 2), BLANK_GROUP_CROSS_BLANK, dtype=torch.int64)
    legal_mask = torch.ones((2, 2, 1), dtype=torch.bool)
    target = torch.tensor([-1, 1])

    loss = inline_blank_priority_loss(logits, group, group_kind, legal_mask, target)
    loss.backward()

    assert loss.item() < 0.32
    assert logits.grad is not None
    assert logits.grad[0].abs().sum().item() == 0.0
    assert logits.grad[1].abs().sum().item() > 0.0


def test_inline_blank_priority_all_ignored_returns_zero_loss() -> None:
    logits = torch.randn(2, 3, 1, requires_grad=True)
    group = torch.zeros((2, 3), dtype=torch.int64)
    group_kind = torch.full((2, 3), BLANK_GROUP_CROSS_BLANK, dtype=torch.int64)
    legal_mask = torch.ones((2, 3, 1), dtype=torch.bool)
    target = torch.tensor([-1, 99])

    loss = inline_blank_priority_loss(logits, group, group_kind, legal_mask, target)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0


def test_inline_blank_priority_accuracy_counts_valid_rows_only() -> None:
    logits = torch.tensor([[[0.0], [4.0]], [[3.0], [1.0]], [[7.0], [8.0]]])
    group = torch.zeros((3, 2), dtype=torch.int64)
    group_kind = torch.full((3, 2), BLANK_GROUP_CROSS_BLANK, dtype=torch.int64)
    legal_mask = torch.ones((3, 2, 1), dtype=torch.bool)
    target = torch.tensor([1, 1, -1])

    stats = inline_blank_priority_accuracy(logits, group, group_kind, legal_mask, target)

    assert stats == {"accuracy": 0.5, "correct": 1, "total": 2}


def test_inline_blank_per_blank_loss_for_constrained_block_choices() -> None:
    logits = torch.tensor([[[0.0, 5.0, -1.0], [4.0, 0.0, -2.0]]], requires_grad=True)
    group_kind = torch.full((1, 2), BLANK_GROUP_CONSTRAINED, dtype=torch.int64)
    legal_mask = torch.tensor([[[True, True, True], [True, True, False]]])
    target = torch.tensor([[1, 0]])

    loss = inline_blank_per_blank_loss(logits, group_kind, legal_mask, target)
    loss.backward()

    assert loss.item() < 0.03
    assert logits.grad is not None
    assert logits.grad[0, 0].abs().sum().item() > 0.0
    assert logits.grad[0, 1, :2].abs().sum().item() > 0.0
    assert logits.grad[0, 1, 2].item() == 0.0


def test_inline_blank_per_blank_ignores_cross_and_invalid_targets() -> None:
    logits = torch.tensor(
        [
            [[100.0, -100.0], [0.0, 3.0], [9.0, 0.0]],
            [[1.0, 2.0], [5.0, 0.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )
    group_kind = torch.tensor(
        [
            [BLANK_GROUP_CROSS_BLANK, BLANK_GROUP_PER_BLANK, BLANK_GROUP_CONSTRAINED],
            [BLANK_GROUP_CONSTRAINED, BLANK_GROUP_CONSTRAINED, BLANK_GROUP_CONSTRAINED],
        ]
    )
    legal_mask = torch.tensor(
        [
            [[True, True], [True, True], [True, False]],
            [[True, True], [True, False], [False, False]],
        ]
    )
    target = torch.tensor([[0, 1, 1], [-1, 1, 0]])

    loss = inline_blank_per_blank_loss(logits, group_kind, legal_mask, target)
    loss.backward()

    assert loss.item() < 0.05
    assert logits.grad is not None
    assert logits.grad[0, 0].abs().sum().item() == 0.0
    assert logits.grad[0, 1].abs().sum().item() > 0.0
    assert logits.grad[0, 2].abs().sum().item() == 0.0
    assert logits.grad[1].abs().sum().item() == 0.0


def test_inline_blank_per_blank_accuracy_counts_valid_slots_only() -> None:
    logits = torch.tensor([[[0.0, 3.0], [4.0, 0.0], [7.0, 8.0]]])
    group_kind = torch.tensor(
        [[BLANK_GROUP_CONSTRAINED, BLANK_GROUP_PER_BLANK, BLANK_GROUP_CROSS_BLANK]]
    )
    legal_mask = torch.ones((1, 3, 2), dtype=torch.bool)
    target = torch.tensor([[1, 1, 1]])

    stats = inline_blank_per_blank_accuracy(logits, group_kind, legal_mask, target)

    assert stats == {"accuracy": 0.5, "correct": 1, "total": 2}


# ---------------------------------------------------------------------------
# Mask correctness — padded positions must not contribute.
# ---------------------------------------------------------------------------


def test_policy_distillation_padding_invariance() -> None:
    """Loss is unchanged whether padded positions hold zeros or garbage."""

    torch.manual_seed(2)
    student_real = torch.randn(2, 3)
    teacher_real = torch.randn(2, 3)
    pad_zeros = torch.zeros(2, 2)
    pad_garbage = torch.randn(2, 2) * 100.0

    student_a = torch.cat([student_real, pad_zeros], dim=-1)
    teacher_a = torch.cat([teacher_real, pad_zeros], dim=-1)
    student_b = torch.cat([student_real, pad_garbage], dim=-1)
    teacher_b = torch.cat([teacher_real, pad_garbage * -1.0], dim=-1)

    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, :3] = True

    loss_a = policy_distillation_loss(student_a, teacher_a, mask)
    loss_b = policy_distillation_loss(student_b, teacher_b, mask)
    assert abs(loss_a.item() - loss_b.item()) < 1e-5


def test_target_distillation_padding_invariance() -> None:
    torch.manual_seed(3)
    student_real = torch.randn(1, 2, 2)
    teacher_real = torch.randn(1, 2, 2)
    pad_zeros = torch.zeros(1, 2, 2)
    pad_garbage = torch.randn(1, 2, 2) * 50.0

    student_a = torch.cat([student_real, pad_zeros], dim=-1)
    teacher_a = torch.cat([teacher_real, pad_zeros], dim=-1)
    student_b = torch.cat([student_real, pad_garbage], dim=-1)
    teacher_b = torch.cat([teacher_real, -pad_garbage], dim=-1)

    mask = torch.zeros(1, 2, 4, dtype=torch.bool)
    mask[..., :2] = True

    loss_a = target_distillation_loss(student_a, teacher_a, mask)
    loss_b = target_distillation_loss(student_b, teacher_b, mask)
    assert abs(loss_a.item() - loss_b.item()) < 1e-5


def test_target_distillation_skips_options_with_no_valid_target() -> None:
    """Options whose target axis is fully masked don't contribute."""

    torch.manual_seed(4)
    s = torch.randn(1, 3, 2)
    t = torch.randn(1, 3, 2)
    mask = torch.zeros(1, 3, 2, dtype=torch.bool)
    mask[0, 0, :] = True  # only first option has any targets
    loss_one = target_distillation_loss(s, t, mask)

    # Now make option 2's logits totally different — should not affect loss
    # because option 2's targets are all masked out.
    s2 = s.clone()
    s2[0, 2, :] = 1000.0
    loss_two = target_distillation_loss(s2, t, mask)
    assert abs(loss_one.item() - loss_two.item()) < 1e-5


# ---------------------------------------------------------------------------
# Single-step trainer smoke
# ---------------------------------------------------------------------------


def _make_batch(
    vocab_size: int = 200,
    seq_len: int = 32,
    batch_size: int = 2,
    max_opts: int = 3,
    max_targets: int = 2,
    pad_id: int = 0,
) -> tuple[TextEncodedBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(42)
    token_ids = torch.randint(1, vocab_size - 4, (batch_size, seq_len), generator=g)
    option_id = vocab_size - 2
    target_id = vocab_size - 3

    option_positions = torch.full((batch_size, max_opts), -1, dtype=torch.int64)
    option_mask = torch.zeros((batch_size, max_opts), dtype=torch.bool)
    target_positions = torch.full((batch_size, max_opts, max_targets), -1, dtype=torch.int64)
    target_mask = torch.zeros((batch_size, max_opts, max_targets), dtype=torch.bool)
    stride = seq_len // (max_opts + 1)
    for b in range(batch_size):
        for o in range(max_opts):
            opt_pos = (o + 1) * stride
            token_ids[b, opt_pos] = option_id
            option_positions[b, o] = opt_pos
            option_mask[b, o] = True
            for t in range(max_targets):
                tpos = opt_pos + 1 + t
                token_ids[b, tpos] = target_id
                target_positions[b, o, t] = tpos
                target_mask[b, o, t] = True

    batch = TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=torch.ones_like(token_ids, dtype=torch.int64),
        card_ref_positions=torch.full((batch_size, MAX_CARD_REFS), -1, dtype=torch.int64),
        option_positions=option_positions,
        option_mask=option_mask,
        target_positions=target_positions,
        target_mask=target_mask,
        seq_lengths=torch.full((batch_size,), seq_len, dtype=torch.int64),
    )

    teacher_p = torch.randn(batch_size, max_opts)
    teacher_t = torch.randn(batch_size, max_opts, max_targets)
    value_targets = torch.rand(batch_size)
    return batch, teacher_p, teacher_t, value_targets


def _build_small_policy(vocab_size: int = 200) -> RecurrentTextPolicy:
    cfg = TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        pad_id=0,
    )
    rcfg = RecurrentTextPolicyConfig(encoder=cfg, lstm_hidden=64, lstm_layers=1)
    return RecurrentTextPolicy(rcfg)


def test_distill_step_returns_expected_keys_and_grad() -> None:
    torch.manual_seed(0)
    policy = _build_small_policy()
    trainer = TextEncoderTrainer(policy, lr=1e-3)
    batch, teacher_p, teacher_t, value_targets = _make_batch()

    stats = trainer.distill_step(
        batch,
        teacher_policy_logits=teacher_p,
        teacher_target_logits=teacher_t,
        value_targets=value_targets,
    )
    expected = {"loss", "policy_loss", "target_loss", "value_loss", "grad_norm"}
    assert expected.issubset(stats.keys())
    assert stats["grad_norm"] > 0.0
    assert stats["loss"] > 0.0


def test_value_step_returns_expected_keys() -> None:
    torch.manual_seed(0)
    policy = _build_small_policy()
    trainer = TextEncoderTrainer(policy, lr=1e-3)
    batch, _, _, value_targets = _make_batch()

    stats = trainer.value_step(batch, value_targets)
    assert "value_loss" in stats
    assert "grad_norm" in stats
    assert stats["grad_norm"] > 0.0
