"""Correctness + performance comparison for LSTM recompute strategies.

Each strategy in :mod:`magic_ai.lstm_recompute` is exercised on synthetic
projected features with a 2-layer ``nn.LSTM`` matching the production
config (``hidden_dim=512``, ``hidden_layers=2``). Correctness asserts the
three batched strategies (``pad``/``gather``/``packed``) match the
``legacy`` per-episode reference loop step-for-step. The benchmark prints
wall-time per strategy so the speed comparison is visible in test output.

Run a quick correctness check with:

    uv run python -m unittest tests.test_lstm_recompute_strategies -v

Run with ``RNAD_LSTM_BENCH=1`` to print the benchmark summary; the timing
test is skipped by default to keep the suite fast.
"""

from __future__ import annotations

import os
import random
import time
import unittest

import torch
from magic_ai.lstm_recompute import (
    STRATEGIES,
    lstm_recompute_per_step_h_out,
    lstm_recompute_per_step_states,
)
from torch import nn


def _make_lstm(hidden: int, num_layers: int, dtype: torch.dtype, device: torch.device) -> nn.LSTM:
    lstm = nn.LSTM(
        input_size=hidden,
        hidden_size=hidden,
        num_layers=num_layers,
        batch_first=True,
    ).to(dtype=dtype, device=device)
    lstm.eval()
    return lstm


def _make_workload(
    n_episodes: int,
    t_max: int,
    t_min: int,
    hidden: int,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Synthetic ``(T_max, N, hidden)`` + per-episode lengths."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    py_rng = random.Random(seed)
    lengths = [py_rng.randint(t_min, t_max) for _ in range(n_episodes)]
    actual_t_max = max(lengths)
    projected = torch.randn(actual_t_max, n_episodes, hidden, generator=g, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    # Zero out the padding positions so ``pad`` (which doesn't mask) cannot
    # accidentally pass by reading non-zero garbage that happens to match.
    for i, t_i in enumerate(lengths):
        projected[t_i:, i] = 0.0
    return projected, lengths


def _per_episode(projected: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
    """Slice a ``(T_max, N, hidden)`` workload into per-episode tensors."""
    return [projected[: lengths[i], i, :].contiguous() for i in range(len(lengths))]


class LstmRecomputeStrategiesCorrectness(unittest.TestCase):
    """All strategies must agree with the legacy per-episode reference."""

    def test_strategies_match_legacy_small(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 16
        num_layers = 2
        torch.manual_seed(0)
        lstm = _make_lstm(hidden, num_layers, dtype, device)

        # Mix of lengths including the boundary cases (T==1, T==T_max, ties).
        projected, lengths = _make_workload(
            n_episodes=4, t_max=8, t_min=1, hidden=hidden, seed=42, dtype=dtype, device=device
        )

        ref = lstm_recompute_per_step_states(lstm, projected, lengths, strategy="legacy")
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                got = lstm_recompute_per_step_states(lstm, projected, lengths, strategy=strategy)
                self.assertEqual(len(got), len(ref))
                for i, ((h_g, c_g), (h_r, c_r)) in enumerate(zip(got, ref, strict=True)):
                    self.assertEqual(h_g.shape, h_r.shape, msg=f"episode {i} h shape")
                    self.assertEqual(c_g.shape, c_r.shape, msg=f"episode {i} c shape")
                    torch.testing.assert_close(h_g, h_r, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(c_g, c_r, rtol=1e-5, atol=1e-6)

    def test_first_state_is_zero(self) -> None:
        # ``h_in[:, 0, :]`` is the state *before* step 0 -- must be zero by
        # construction in every strategy.
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 8
        num_layers = 2
        torch.manual_seed(1)
        lstm = _make_lstm(hidden, num_layers, dtype, device)
        projected, lengths = _make_workload(
            n_episodes=3, t_max=5, t_min=2, hidden=hidden, seed=7, dtype=dtype, device=device
        )
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                got = lstm_recompute_per_step_states(lstm, projected, lengths, strategy=strategy)
                for h, c in got:
                    self.assertTrue(torch.equal(h[:, 0, :], torch.zeros_like(h[:, 0, :])))
                    self.assertTrue(torch.equal(c[:, 0, :], torch.zeros_like(c[:, 0, :])))

    def test_rejects_bad_inputs(self) -> None:
        device = torch.device("cpu")
        lstm = _make_lstm(8, 1, torch.float32, device)
        projected = torch.zeros(3, 2, 8)
        with self.assertRaises(ValueError):
            lstm_recompute_per_step_states(lstm, projected, [], strategy="legacy")
        with self.assertRaises(ValueError):
            lstm_recompute_per_step_states(lstm, projected, [0, 2], strategy="legacy")
        with self.assertRaises(ValueError):
            lstm_recompute_per_step_states(lstm, projected, [3, 3, 3], strategy="legacy")
        with self.assertRaises(ValueError):
            lstm_recompute_per_step_states(lstm, projected, [4, 1], strategy="legacy")
        from typing import cast as _cast

        from magic_ai.lstm_recompute import LstmRecomputeStrategy

        with self.assertRaises(ValueError):
            lstm_recompute_per_step_states(
                lstm,
                projected,
                [2, 1],
                strategy=_cast(LstmRecomputeStrategy, "bogus"),
            )


class LstmRecomputeChunkedBptt(unittest.TestCase):
    """Chunked-BPTT semantics: forward output is invariant to chunk size."""

    def test_forward_invariant_to_chunk_size(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 12
        num_layers = 2
        torch.manual_seed(2)
        lstm = _make_lstm(hidden, num_layers, dtype, device)
        projected, lengths = _make_workload(
            n_episodes=3,
            t_max=10,
            t_min=3,
            hidden=hidden,
            seed=11,
            dtype=dtype,
            device=device,
        )
        # Reference: chunk_size = T_max (single chunk, full BPTT).
        per_ep = _per_episode(projected, lengths)
        ref = lstm_recompute_per_step_h_out(lstm, per_ep, chunk_size=max(lengths))
        # All other chunk sizes -- including ones that don't evenly divide
        # T_max -- must produce identical forward output. Gradient flow is
        # truncated at chunk boundaries, but the forward computation is
        # exact.
        for chunk_size in (1, 2, 3, 7, 9, 100):
            with self.subTest(chunk_size=chunk_size):
                got = lstm_recompute_per_step_h_out(lstm, per_ep, chunk_size=chunk_size)
                for g, r in zip(got, ref, strict=True):
                    torch.testing.assert_close(g, r, rtol=1e-5, atol=1e-6)

    def test_chunk_size_one_truncates_gradient(self) -> None:
        # chunk_size=1 detaches state every step, so a gradient on h_out[t]
        # cannot reach LSTM weights via h_out[t-1]. Verify the gradient is
        # nonzero (the in-chunk one-step path still flows) but smaller than
        # full BPTT.
        device = torch.device("cpu")
        torch.manual_seed(3)
        lstm = _make_lstm(8, 2, torch.float32, device)
        projected, lengths = _make_workload(
            n_episodes=2,
            t_max=6,
            t_min=6,
            hidden=8,
            seed=5,
            dtype=torch.float32,
            device=device,
        )
        projected = projected.detach().requires_grad_(False)

        per_ep = _per_episode(projected, lengths)

        def grad_norm(chunk_size: int) -> float:
            for p in lstm.parameters():
                if p.grad is not None:
                    p.grad = None
            outputs = lstm_recompute_per_step_h_out(lstm, per_ep, chunk_size=chunk_size)
            # Take the loss only at the LAST step of each episode -- this is
            # where chunk-1 truncation differs most from full BPTT.
            loss = sum(out[-1].sum() for out in outputs)
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            total = 0.0
            for p in lstm.parameters():
                if p.grad is not None:
                    total += float(p.grad.detach().pow(2).sum())
            return total**0.5

        full_bptt_norm = grad_norm(chunk_size=max(lengths))
        truncated_norm = grad_norm(chunk_size=1)
        self.assertGreater(full_bptt_norm, 0.0)
        self.assertGreater(truncated_norm, 0.0)
        # Chunk=1 must lose gradient signal compared to full BPTT.
        self.assertLess(truncated_norm, full_bptt_norm)


class LstmRecomputeStrategiesBenchmark(unittest.TestCase):
    """Wall-time comparison; opt in with ``RNAD_LSTM_BENCH=1``."""

    @unittest.skipUnless(
        os.environ.get("RNAD_LSTM_BENCH"),
        "set RNAD_LSTM_BENCH=1 to run the benchmark",
    )
    def test_benchmark_strategies(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        # Match production: hidden_dim=512, hidden_layers=2.
        hidden = 512
        num_layers = 2
        n_episodes = int(os.environ.get("RNAD_LSTM_BENCH_N", "32"))
        t_max = int(os.environ.get("RNAD_LSTM_BENCH_T", "200"))
        t_min = int(os.environ.get("RNAD_LSTM_BENCH_T_MIN", str(max(1, t_max // 4))))
        n_iters = int(os.environ.get("RNAD_LSTM_BENCH_ITERS", "5"))
        warmup = int(os.environ.get("RNAD_LSTM_BENCH_WARMUP", "2"))

        torch.manual_seed(0)
        lstm = _make_lstm(hidden, num_layers, dtype, device)
        projected, lengths = _make_workload(
            n_episodes=n_episodes,
            t_max=t_max,
            t_min=t_min,
            hidden=hidden,
            seed=123,
            dtype=dtype,
            device=device,
        )

        def _sync() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Correctness check on the benchmark workload before we trust the
        # speed numbers -- caught more than one impl bug during development.
        ref = lstm_recompute_per_step_states(lstm, projected, lengths, strategy="legacy")
        # Loose tolerance: the batched paths use cuDNN/oneDNN with a
        # different batch dimension than the legacy ref, so per-step
        # rounding accumulates over a long sequence. We just want a sanity
        # check that they're computing the same thing -- the small-T
        # correctness test in this file enforces tight bounds.
        for strategy in STRATEGIES:
            got = lstm_recompute_per_step_states(lstm, projected, lengths, strategy=strategy)
            for (h_g, c_g), (h_r, c_r) in zip(got, ref, strict=True):
                torch.testing.assert_close(h_g, h_r, rtol=1e-2, atol=1e-3)
                torch.testing.assert_close(c_g, c_r, rtol=1e-2, atol=1e-3)

        results: dict[str, float] = {}
        for strategy in STRATEGIES:
            for _ in range(warmup):
                lstm_recompute_per_step_states(lstm, projected, lengths, strategy=strategy)
            _sync()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                lstm_recompute_per_step_states(lstm, projected, lengths, strategy=strategy)
            _sync()
            elapsed = (time.perf_counter() - t0) / n_iters
            results[strategy] = elapsed

        # Fused single-cuDNN-call path (override-interface change: returns
        # per-step top-layer h_out, not (h_in, c_in)). Correctness here is a
        # sanity check that h_out matches what legacy would produce as the
        # post-step hidden output -- compute it on-the-fly from legacy.
        h_out_ref: list[torch.Tensor] = []
        for i, t_i in enumerate(lengths):
            h_in_i, c_in_i = ref[i]  # (num_layers, T_i, hidden)
            # Run an extra step from each (h_in, c_in) to get h_out per step.
            outs: list[torch.Tensor] = []
            for t in range(t_i):
                step_input = projected[t : t + 1, i : i + 1, :].contiguous()
                _o, (h_post, _c_post) = lstm(
                    step_input,
                    (h_in_i[:, t : t + 1, :].contiguous(), c_in_i[:, t : t + 1, :].contiguous()),
                )
                outs.append(h_post[-1, 0, :])  # top layer
            h_out_ref.append(torch.stack(outs, dim=0))
        per_ep = _per_episode(projected, lengths)
        h_out_fused = lstm_recompute_per_step_h_out(lstm, per_ep)
        for got, want in zip(h_out_fused, h_out_ref, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-3)

        # Bench the fused path uncompiled (cross-episode batched).
        with torch.no_grad():
            for _ in range(warmup):
                lstm_recompute_per_step_h_out(lstm, per_ep)
            _sync()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                lstm_recompute_per_step_h_out(lstm, per_ep)
            _sync()
            results["fused"] = (time.perf_counter() - t0) / n_iters

        # Also bench the *per-episode* fused path -- the previous wiring
        # in rnad_trajectory_loss called the fused recompute once per
        # episode (batch=1). Lifting the call up to run_rnad_update lets a
        # single fused call cover the whole rollout batch; this row shows
        # what we recover by doing that.
        with torch.no_grad():
            for _ in range(warmup):
                for i in range(n_episodes):
                    lstm_recompute_per_step_h_out(lstm, [per_ep[i]])
            _sync()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                for i in range(n_episodes):
                    lstm_recompute_per_step_h_out(lstm, [per_ep[i]])
            _sync()
            results["fused_per_episode"] = (time.perf_counter() - t0) / n_iters

        # Bench the fused path with torch.compile around the LSTM forward.
        compile_disabled = os.environ.get("RNAD_LSTM_BENCH_NO_COMPILE")
        if not compile_disabled:
            try:
                compiled_lstm = torch.compile(lstm, dynamic=False, mode="reduce-overhead")
                with torch.no_grad():
                    for _ in range(warmup + 2):  # extra warmup for compile
                        lstm_recompute_per_step_h_out(lstm, per_ep, compiled_lstm=compiled_lstm)
                    _sync()
                    t0 = time.perf_counter()
                    for _ in range(n_iters):
                        lstm_recompute_per_step_h_out(lstm, per_ep, compiled_lstm=compiled_lstm)
                    _sync()
                    results["fused_compiled"] = (time.perf_counter() - t0) / n_iters
            except Exception as exc:  # pragma: no cover - compile env-dependent
                print(f"  (torch.compile failed: {type(exc).__name__}: {exc})")

        baseline = results["legacy"]
        print()
        print(
            f"LSTM recompute benchmark: device={device} hidden={hidden} layers={num_layers} "
            f"N={n_episodes} T_min={t_min} T_max={t_max} iters={n_iters}"
        )
        print(f"  total padded steps  = {n_episodes * max(lengths)}")
        print(f"  total real steps    = {sum(lengths)}")
        ordered = list(STRATEGIES) + ["fused_per_episode", "fused"]
        if "fused_compiled" in results:
            ordered.append("fused_compiled")
        for strategy in ordered:
            elapsed = results[strategy]
            speedup = baseline / elapsed if elapsed > 0 else float("inf")
            print(f"  {strategy:>14s}: {elapsed * 1000:8.2f} ms  ({speedup:6.2f}x vs legacy)")


if __name__ == "__main__":
    unittest.main()
