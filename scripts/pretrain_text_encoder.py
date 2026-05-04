"""Synthetic-data smoke for the text-encoder training loop.

Stage 3/4 of ``docs/text_encoder_plan.md`` §7 calls for two pretraining phases:
supervised value pretrain on rollout outcomes, and BC distillation from the
slot-encoder teacher. Both share the same :class:`TextEncoderTrainer` plumbing
in ``magic_ai.text_encoder.training``.

This script exists to *prove the loop trains*: it builds a tiny encoder, draws
random :class:`TextEncodedBatch` instances whose labels are deterministic
functions of the token-id statistics, and asserts the loss in the last 20
steps is at least 25% below the loss in the first 20 steps. Real data sources
are plugged in later by replacing the synthetic generator with a Dataset that
yields the same ``(batch, labels)`` tuple.

CLI:

    uv run python scripts/pretrain_text_encoder.py --mode value
    uv run python scripts/pretrain_text_encoder.py --mode distill

Use ``--vocab-size 1000`` for the smoke; loading the production 50k tokenizer
is unnecessary here (the encoder does not care about token *meaning*, only
that the labels are a learnable function of the inputs).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS
from magic_ai.text_encoder.training import TextEncoderTrainer
from torch import Tensor


@dataclass
class SyntheticLabels:
    value_targets: Tensor  # [B]
    teacher_policy_logits: Tensor  # [B, max_opts]
    teacher_target_logits: Tensor  # [B, max_opts, max_targets]


def make_synthetic_batch(
    batch_size: int,
    seq_len: int,
    max_opts: int,
    max_targets: int,
    vocab_size: int,
    pad_id: int,
    rng: torch.Generator,
    device: torch.device,
) -> tuple[TextEncodedBatch, SyntheticLabels]:
    """Random token sequences + deterministic labels.

    The labels are computed from the *token ids* (not from random noise) so
    the encoder has something to learn:

    * ``value_target = (token_ids == K).float().mean()`` for a fixed token K.
    * ``teacher_policy_logits[b, o]`` = sum of marker-token indicators in a
      window around the option position. An option is "good" iff its
      surrounding context contains the marker token; the trainer just needs
      SOMETHING the encoder can learn.
    * ``teacher_target_logits`` analogous, per (option, target) slot.
    """

    if pad_id >= vocab_size:
        raise ValueError(f"pad_id ({pad_id}) must be < vocab_size ({vocab_size})")

    # Reserve special token ids — they must be valid embeddings, just kept
    # distinct from "regular" ids so anchor positions don't collide.
    option_id = max(1, vocab_size - 2)
    target_id = max(1, vocab_size - 3)
    marker_id = max(1, vocab_size - 4)
    if min(option_id, target_id, marker_id) <= pad_id:
        raise ValueError("vocab_size too small to host pad+special ids")

    # Sample regular tokens uniformly over [1, vocab_size - 4) so we don't
    # accidentally generate the special ids and confuse the anchor recovery.
    regular_low = 1
    regular_high = max(2, vocab_size - 4)
    token_ids = torch.randint(
        regular_low,
        regular_high,
        (batch_size, seq_len),
        generator=rng,
        device=device,
    )

    # Stamp <option> / <target> tokens at fixed-ish positions so anchors are
    # in-bounds and non-overlapping. Each example gets `max_opts` options;
    # each option gets `max_targets` target tokens immediately after it.
    option_positions = torch.full((batch_size, max_opts), -1, dtype=torch.int64, device=device)
    option_mask = torch.zeros((batch_size, max_opts), dtype=torch.bool, device=device)
    target_positions = torch.full(
        (batch_size, max_opts, max_targets), -1, dtype=torch.int64, device=device
    )
    target_mask = torch.zeros((batch_size, max_opts, max_targets), dtype=torch.bool, device=device)
    stride = seq_len // (max_opts + 1)
    for b in range(batch_size):
        # Per-example variable count of options to exercise the mask.
        n_opts = int(torch.randint(1, max_opts + 1, (1,), generator=rng).item())
        for o in range(n_opts):
            opt_pos = (o + 1) * stride
            token_ids[b, opt_pos] = option_id
            option_positions[b, o] = opt_pos
            option_mask[b, o] = True
            n_t = int(torch.randint(1, max_targets + 1, (1,), generator=rng).item())
            for t in range(n_t):
                tpos = opt_pos + 1 + t
                if tpos >= seq_len:
                    break
                token_ids[b, tpos] = target_id
                target_positions[b, o, t] = tpos
                target_mask[b, o, t] = True

    # Pick one "winning" option per example and stamp the marker token
    # immediately before it. This is a deterministic, easily-attentionable
    # signal for the policy distillation: the teacher prefers the option that
    # has a marker token at position-1.
    winning_option = torch.zeros(batch_size, dtype=torch.int64, device=device)
    for b in range(batch_size):
        valid = int(option_mask[b].sum().item())
        winning_option[b] = int(torch.randint(0, valid, (1,), generator=rng).item())
        opt_pos = int(option_positions[b, int(winning_option[b].item())].item())
        if opt_pos - 1 >= 0:
            token_ids[b, opt_pos - 1] = marker_id

    # Also sprinkle a low-density background marker so the value target is a
    # learnable scalar (count of markers) in addition to the policy signal.
    marker_bg = torch.rand((batch_size, seq_len), generator=rng, device=device) < 0.05
    token_ids = torch.where(marker_bg, torch.full_like(token_ids, marker_id), token_ids)

    attention_mask = torch.ones_like(token_ids, dtype=torch.int64)
    seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)
    card_ref_positions = torch.full(
        (batch_size, MAX_CARD_REFS), -1, dtype=torch.int64, device=device
    )

    batch = TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lengths,
    )

    # Labels --------------------------------------------------------------
    is_marker = (token_ids == marker_id).float()
    value_targets = is_marker.mean(dim=-1)  # [B], in [0, 1]

    # Teacher policy logits: peaked at the winning option (strong, one-hot-ish
    # distribution). The encoder learns to look at position-1 of each option
    # for the marker token to identify which is winning.
    del is_marker  # only used to compute value_targets above
    teacher_policy_logits = torch.zeros((batch_size, max_opts), device=device)
    for b in range(batch_size):
        teacher_policy_logits[b, int(winning_option[b].item())] = 4.0

    # Teacher target logits: prefer target slot 0 within the winning option.
    # Simple but exercises the per-(b, o) target KL.
    teacher_target_logits = torch.zeros((batch_size, max_opts, max_targets), device=device)
    for b in range(batch_size):
        wo = int(winning_option[b].item())
        teacher_target_logits[b, wo, 0] = 4.0

    labels = SyntheticLabels(
        value_targets=value_targets,
        teacher_policy_logits=teacher_policy_logits,
        teacher_target_logits=teacher_target_logits,
    )
    return batch, labels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("value", "distill"), default="value")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Synthetic vocab size; keep small for the smoke run.",
    )
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--max-opts", type=int, default=6)
    p.add_argument("--max-targets", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    cfg = TextEncoderConfig(
        vocab_size=args.vocab_size,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        max_seq_len=args.seq_len,
        pad_id=0,
    )
    rcfg = RecurrentTextPolicyConfig(encoder=cfg, lstm_hidden=64, lstm_layers=1)
    policy = RecurrentTextPolicy(rcfg).to(device)
    trainer = TextEncoderTrainer(policy, lr=args.lr, grad_clip=1.0)

    losses: list[float] = []
    log_every = 10

    for step in range(args.steps):
        batch, labels = make_synthetic_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_opts=args.max_opts,
            max_targets=args.max_targets,
            vocab_size=args.vocab_size,
            pad_id=cfg.pad_id,
            rng=rng,
            device=device,
        )
        if args.mode == "value":
            stats = trainer.value_step(batch, labels.value_targets)
            loss_val = stats["value_loss"]
        else:
            stats = trainer.distill_step(
                batch,
                teacher_policy_logits=labels.teacher_policy_logits,
                teacher_target_logits=labels.teacher_target_logits,
                value_targets=labels.value_targets,
            )
            loss_val = stats["loss"]
        losses.append(loss_val)

        if step % log_every == 0:
            sys.stdout.write(
                f"[{args.mode}] step={step:4d} loss={loss_val:.4f} "
                f"grad_norm={stats['grad_norm']:.3f}\n"
            )
            sys.stdout.flush()

    # Convergence assertion: last-20 mean is at least 25% below first-20 mean.
    if len(losses) < 40:
        raise RuntimeError("Need at least 40 steps to evaluate convergence assertion.")
    first_mean = sum(losses[:20]) / 20.0
    last_mean = sum(losses[-20:]) / 20.0
    sys.stdout.write(
        f"[{args.mode}] first20_mean={first_mean:.4f} "
        f"last20_mean={last_mean:.4f} "
        f"reduction={(1.0 - last_mean / max(first_mean, 1e-12)):.2%}\n"
    )
    if last_mean > 0.75 * first_mean:
        raise AssertionError(
            f"Convergence smoke failed: last20_mean={last_mean:.4f} "
            f"is not <= 75% of first20_mean={first_mean:.4f}."
        )
    sys.stdout.write("convergence OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
