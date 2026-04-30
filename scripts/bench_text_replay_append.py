"""Benchmark text replay-buffer append/gather with and without Triton."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magic_ai.text_encoder.batch import PackedTextBatch  # noqa: E402
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer  # noqa: E402
from magic_ai.text_encoder.replay_triton import TRITON_AVAILABLE  # noqa: E402
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS  # noqa: E402


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _random_abs_positions(
    shape: tuple[int, ...],
    *,
    state_positions: torch.Tensor,
    seq_lengths: torch.Tensor,
    absent_p: float,
) -> torch.Tensor:
    trailing = (1,) * (len(shape) - 1)
    rel = (torch.rand(shape, device=seq_lengths.device) * seq_lengths.view(-1, *trailing)).to(
        torch.int64
    )
    pos = rel + state_positions.view(-1, *trailing)
    absent = torch.rand(shape, device=seq_lengths.device) < absent_p
    return torch.where(absent, torch.full_like(pos, -1), pos)


def _make_packed_batch(
    *,
    batch_size: int,
    min_tokens: int,
    max_tokens: int,
    max_options: int,
    max_targets_per_option: int,
    device: torch.device,
) -> PackedTextBatch:
    seq_lengths = torch.randint(min_tokens, max_tokens + 1, (batch_size,), device=device)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int64, device=device)
    cu_seqlens[1:] = seq_lengths.cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())
    token_ids = torch.randint(1, 50_000, (total_tokens,), dtype=torch.int64, device=device)
    seq_id = torch.repeat_interleave(torch.arange(batch_size, device=device), seq_lengths)
    state_positions = cu_seqlens[:-1]
    pos_in_seq = torch.arange(total_tokens, device=device) - torch.repeat_interleave(
        state_positions, seq_lengths
    )
    card_ref_positions = _random_abs_positions(
        (batch_size, MAX_CARD_REFS),
        state_positions=state_positions,
        seq_lengths=seq_lengths,
        absent_p=0.70,
    )
    option_positions = _random_abs_positions(
        (batch_size, max_options),
        state_positions=state_positions,
        seq_lengths=seq_lengths,
        absent_p=0.20,
    )
    option_mask = option_positions >= 0
    target_positions = _random_abs_positions(
        (batch_size, max_options, max_targets_per_option),
        state_positions=state_positions,
        seq_lengths=seq_lengths,
        absent_p=0.55,
    )
    target_mask = target_positions >= 0
    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu_seqlens,
        seq_lengths=seq_lengths,
        state_positions=state_positions,
        card_ref_positions=card_ref_positions,
        option_positions=option_positions,
        option_mask=option_mask,
        target_positions=target_positions,
        target_mask=target_mask,
    )


def _metadata(
    *,
    batch_size: int,
    decision_groups: int,
    max_cached_choices: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    total_groups = batch_size * decision_groups
    return {
        "trace_kind_id": torch.randint(0, 4, (batch_size,), device=device),
        "decision_count": torch.full((batch_size,), decision_groups, device=device),
        "decision_option_idx": torch.randint(
            -1, 8, (total_groups, max_cached_choices), device=device
        ),
        "decision_target_idx": torch.randint(
            -1, 8, (total_groups, max_cached_choices), device=device
        ),
        "decision_mask": torch.rand(total_groups, max_cached_choices, device=device) > 0.25,
        "uses_none_head": torch.rand(total_groups, device=device) > 0.5,
        "selected_indices": torch.randint(0, max_cached_choices, (total_groups,), device=device),
        "may_selected": torch.rand(batch_size, device=device),
        "old_log_prob": torch.randn(batch_size, device=device),
        "value": torch.randn(batch_size, device=device),
        "perspective_player_idx": torch.randint(0, 2, (batch_size,), device=device),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run(
    *,
    use_triton: bool,
    encoded: PackedTextBatch,
    meta: dict[str, torch.Tensor],
    iterations: int,
    max_tokens: int,
    max_options: int,
    max_targets_per_option: int,
    max_decision_groups: int,
    max_cached_choices: int,
    device: torch.device,
    profile_range: bool = False,
) -> float:
    batch_size = int(encoded.seq_lengths.shape[0])
    buffer = TextReplayBuffer(
        capacity=batch_size * iterations,
        max_tokens=max_tokens,
        max_options=max_options,
        max_targets_per_option=max_targets_per_option,
        max_decision_groups=max_decision_groups,
        max_cached_choices=max_cached_choices,
        device=device,
        validate=False,
        use_triton_append=use_triton,
    )
    _sync(device)
    if profile_range and device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    for _ in range(iterations):
        buffer.append_batch(encoded=encoded, **meta)
    _sync(device)
    if profile_range and device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStop()
    return time.perf_counter() - start


def _run_gather(
    *,
    use_triton: bool,
    encoded: PackedTextBatch,
    meta: dict[str, torch.Tensor],
    iterations: int,
    max_tokens: int,
    max_options: int,
    max_targets_per_option: int,
    max_decision_groups: int,
    max_cached_choices: int,
    device: torch.device,
    profile_range: bool = False,
) -> float:
    batch_size = int(encoded.seq_lengths.shape[0])
    buffer = TextReplayBuffer(
        capacity=batch_size,
        max_tokens=max_tokens,
        max_options=max_options,
        max_targets_per_option=max_targets_per_option,
        max_decision_groups=max_decision_groups,
        max_cached_choices=max_cached_choices,
        device=device,
        validate=False,
        use_triton_append=True,
        use_triton_gather=use_triton,
    )
    rows = buffer.append_batch(encoded=encoded, **meta)
    _sync(device)
    if profile_range and device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    for _ in range(iterations):
        buffer.gather(rows)
    _sync(device)
    if profile_range and device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStop()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-tokens", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--max-targets-per-option", type=int, default=4)
    parser.add_argument("--max-decision-groups", type=int, default=16)
    parser.add_argument("--max-cached-choices", type=int, default=8)
    parser.add_argument("--decision-groups", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mode", choices=("both", "torch", "triton"), default="both")
    parser.add_argument("--op", choices=("append", "gather"), default="append")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    args = parser.parse_args()

    device = _device(args.device)
    torch.manual_seed(0)
    encoded = _make_packed_batch(
        batch_size=args.batch_size,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        device=device,
    )
    meta = _metadata(
        batch_size=args.batch_size,
        decision_groups=args.decision_groups,
        max_cached_choices=args.max_cached_choices,
        device=device,
    )

    modes = (False, True)
    if args.mode == "torch":
        modes = (False,)
    elif args.mode == "triton":
        modes = (True,)

    for use_triton in modes:
        if use_triton and (device.type != "cuda" or not TRITON_AVAILABLE):
            continue
        run_fn = _run_gather if args.op == "gather" else _run
        run_fn(
            use_triton=use_triton,
            encoded=encoded,
            meta=meta,
            iterations=args.warmup,
            max_tokens=args.max_tokens,
            max_options=args.max_options,
            max_targets_per_option=args.max_targets_per_option,
            max_decision_groups=args.max_decision_groups,
            max_cached_choices=args.max_cached_choices,
            device=device,
        )

    torch_seconds = None
    if args.mode in ("both", "torch"):
        run_fn = _run_gather if args.op == "gather" else _run
        torch_seconds = run_fn(
            use_triton=False,
            encoded=encoded,
            meta=meta,
            iterations=args.iterations,
            max_tokens=args.max_tokens,
            max_options=args.max_options,
            max_targets_per_option=args.max_targets_per_option,
            max_decision_groups=args.max_decision_groups,
            max_cached_choices=args.max_cached_choices,
            device=device,
            profile_range=args.cuda_profiler_range and args.mode == "torch",
        )
        print(
            f"torch {args.op}: "
            f"{torch_seconds * 1_000 / args.iterations:.3f} ms/{args.op} "
            f"({args.batch_size} rows)"
        )
    if args.mode == "torch":
        return
    if device.type != "cuda" or not TRITON_AVAILABLE:
        print("triton append_batch: skipped (requires CUDA and Triton)")
        return

    run_fn = _run_gather if args.op == "gather" else _run
    triton_seconds = run_fn(
        use_triton=True,
        encoded=encoded,
        meta=meta,
        iterations=args.iterations,
        max_tokens=args.max_tokens,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        max_decision_groups=args.max_decision_groups,
        max_cached_choices=args.max_cached_choices,
        device=device,
        profile_range=args.cuda_profiler_range,
    )
    speed = f"{torch_seconds / triton_seconds:.2f}x torch" if torch_seconds is not None else "only"
    print(
        f"triton {args.op}: {triton_seconds * 1_000 / args.iterations:.3f} ms/{args.op} ({speed})"
    )


if __name__ == "__main__":
    main()
