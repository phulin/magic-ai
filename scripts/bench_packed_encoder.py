"""Benchmark padded vs packed forward+backward for the text-state encoder.

Sweeps a few length-skew regimes (every-row-full ... very-skewed) and
reports per-iter wall time + forward output throughput. Runs on CUDA
when available, otherwise CPU. Backward is exercised so the per-doc
RoPE gather and document mask_mod's gradient paths are timed too.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
from magic_ai.text_encoder.batch import TextEncodedBatch, pack_batch
from magic_ai.text_encoder.model import (
    TextEncoderConfig,
    TextStateEncoder,
    gather_state_vector,
    gather_state_vector_packed,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


@dataclass
class Regime:
    name: str
    seq_lengths: list[int]


def _build_padded(seq_lengths: list[int], vocab_size: int) -> TextEncodedBatch:
    b = len(seq_lengths)
    t_max = max(seq_lengths)
    token_ids = torch.randint(low=1, high=vocab_size, size=(b, t_max), dtype=torch.int64)
    attention_mask = torch.zeros(b, t_max, dtype=torch.int64)
    for i, n in enumerate(seq_lengths):
        attention_mask[i, :n] = 1
        token_ids[i, n:] = 0
    card_ref_positions = torch.full((b, MAX_CARD_REFS), -1, dtype=torch.int64)
    option_positions = torch.full((b, 1), -1, dtype=torch.int64)
    target_positions = torch.full((b, 1, 1), -1, dtype=torch.int64)
    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        option_positions=option_positions,
        option_mask=option_positions >= 0,
        target_positions=target_positions,
        target_mask=target_positions >= 0,
        seq_lengths=torch.as_tensor(seq_lengths, dtype=torch.int64),
    )


def _move_padded(batch: TextEncodedBatch, device: torch.device) -> TextEncodedBatch:
    def to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device, non_blocking=True)

    return TextEncodedBatch(
        token_ids=to(batch.token_ids),
        attention_mask=to(batch.attention_mask),
        card_ref_positions=to(batch.card_ref_positions),
        option_positions=to(batch.option_positions),
        option_mask=to(batch.option_mask),
        target_positions=to(batch.target_positions),
        target_mask=to(batch.target_mask),
        seq_lengths=to(batch.seq_lengths),
    )


def _move_packed(batch, device):  # type: ignore[no-untyped-def]
    from magic_ai.text_encoder.batch import PackedTextBatch

    def to(t):  # type: ignore[no-untyped-def]
        return t.to(device, non_blocking=True)

    return PackedTextBatch(
        token_ids=to(batch.token_ids),
        seq_id=to(batch.seq_id),
        pos_in_seq=to(batch.pos_in_seq),
        cu_seqlens=to(batch.cu_seqlens),
        seq_lengths=to(batch.seq_lengths),
        state_positions=to(batch.state_positions),
        card_ref_positions=to(batch.card_ref_positions),
        option_positions=to(batch.option_positions),
        option_mask=to(batch.option_mask),
        target_positions=to(batch.target_positions),
        target_mask=to(batch.target_mask),
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench_padded(
    encoder: TextStateEncoder, batch: TextEncodedBatch, device: torch.device, *, iters: int
) -> float:
    batch = _move_padded(batch, device)
    encoder.zero_grad(set_to_none=True)
    # Warmup.
    for _ in range(3):
        h = encoder(batch)
        loss = gather_state_vector(h, batch).sum()
        loss.backward()
        encoder.zero_grad(set_to_none=True)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        h = encoder(batch)
        loss = gather_state_vector(h, batch).sum()
        loss.backward()
        encoder.zero_grad(set_to_none=True)
    _sync(device)
    return (time.perf_counter() - start) / iters


def _bench_packed(encoder: TextStateEncoder, batch, device: torch.device, *, iters: int) -> float:
    batch = _move_packed(batch, device)
    encoder.zero_grad(set_to_none=True)
    for _ in range(3):
        h = encoder.forward_packed(batch)
        loss = gather_state_vector_packed(h, batch).sum()
        loss.backward()
        encoder.zero_grad(set_to_none=True)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        h = encoder.forward_packed(batch)
        loss = gather_state_vector_packed(h, batch).sum()
        loss.backward()
        encoder.zero_grad(set_to_none=True)
    _sync(device)
    return (time.perf_counter() - start) / iters


def _regimes(batch_size: int, t_max: int) -> list[Regime]:
    full = [t_max] * batch_size
    half = [t_max if i % 2 == 0 else t_max // 2 for i in range(batch_size)]
    skew = [max(8, t_max - i * (t_max // batch_size)) for i in range(batch_size)]
    extreme = [t_max] + [16] * (batch_size - 1)
    return [
        Regime("uniform-full", full),
        Regime("half-half", half),
        Regime("linear-skew", skew),
        Regime("one-long", extreme),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--t-max", type=int, default=512)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=4000)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    cfg = TextEncoderConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=max(2048, args.t_max),
    )
    encoder = TextStateEncoder(cfg).to(device)

    print(f"device={device}  d_model={cfg.d_model}  layers={cfg.n_layers}")
    print(f"batch_size={args.batch_size}  t_max={args.t_max}  iters={args.iters}\n")
    header = f"{'regime':<14}{'live%':>8}{'padded ms':>12}{'packed ms':>12}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    for regime in _regimes(args.batch_size, args.t_max):
        padded = _build_padded(regime.seq_lengths, args.vocab_size)
        packed = pack_batch(padded)
        live = sum(regime.seq_lengths) / (len(regime.seq_lengths) * args.t_max)
        t_dense = _bench_padded(encoder, padded, device, iters=args.iters)
        t_pack = _bench_packed(encoder, packed, device, iters=args.iters)
        print(
            f"{regime.name:<14}{live * 100:>7.1f}%{t_dense * 1000:>12.2f}"
            f"{t_pack * 1000:>12.2f}{t_dense / t_pack:>9.2f}x"
        )


if __name__ == "__main__":
    main()
