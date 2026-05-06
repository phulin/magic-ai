"""Synthetic-data smoke for the text-encoder value training loop.

This script exists to *prove the loop trains*: it builds a tiny encoder, draws
random :class:`TextEncodedBatch` instances whose labels are deterministic
functions of the token-id statistics, and asserts the loss in the last 20
steps is at least 25% below the loss in the first 20 steps. Real data sources
are plugged in later by replacing the synthetic generator with a Dataset that
yields the same ``(batch, labels)`` tuple.

CLI:

    uv run python scripts/pretrain_text_encoder.py

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


def make_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    pad_id: int,
    rng: torch.Generator,
    device: torch.device,
) -> tuple[TextEncodedBatch, SyntheticLabels]:
    """Random token sequences + deterministic labels.

    The labels are computed from the *token ids* (not from random noise) so
    the encoder has something to learn:

    * ``value_target = (token_ids == K).float().mean()`` for a fixed token K.
    """

    if pad_id >= vocab_size:
        raise ValueError(f"pad_id ({pad_id}) must be < vocab_size ({vocab_size})")

    marker_id = max(1, vocab_size - 2)
    if marker_id <= pad_id:
        raise ValueError("vocab_size too small to host pad+special ids")

    # Sample regular tokens below marker_id so the value target is controlled
    # by explicit marker placement.
    regular_low = 1
    regular_high = max(2, marker_id)
    token_ids = torch.randint(
        regular_low,
        regular_high,
        (batch_size, seq_len),
        generator=rng,
        device=device,
    )

    # Sprinkle a low-density marker so the value target is a learnable scalar.
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

    labels = SyntheticLabels(value_targets=value_targets)
    return batch, labels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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
            vocab_size=args.vocab_size,
            pad_id=cfg.pad_id,
            rng=rng,
            device=device,
        )
        stats = trainer.value_step(batch, labels.value_targets)
        loss_val = stats["value_loss"]
        losses.append(loss_val)

        if step % log_every == 0:
            sys.stdout.write(
                f"[value] step={step:4d} loss={loss_val:.4f} grad_norm={stats['grad_norm']:.3f}\n"
            )
            sys.stdout.flush()

    # Convergence assertion: last-20 mean is at least 25% below first-20 mean.
    if len(losses) < 40:
        raise RuntimeError("Need at least 40 steps to evaluate convergence assertion.")
    first_mean = sum(losses[:20]) / 20.0
    last_mean = sum(losses[-20:]) / 20.0
    sys.stdout.write(
        f"[value] first20_mean={first_mean:.4f} "
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
