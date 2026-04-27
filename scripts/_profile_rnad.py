"""Throwaway: profile a few full training iterations (rollout + update).

Usage: uv run python scripts/_profile_rnad.py [extra train.py args ...]

Wraps run_rnad_update so a single torch.profiler captures everything
between updates (so the rollout/sampling/native-step path is included),
plus the update itself. After N profiled iterations, dumps key-averages
tables and exits.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch.profiler import ProfilerActivity, profile, record_function

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util  # noqa: E402
import threading  # noqa: E402

import magic_ai.rnad_trainer as rt  # noqa: E402
from magic_ai.model import PPOPolicy  # noqa: E402
from magic_ai.sharded_native import (  # noqa: E402
    ShardedNativeBatchEncoder,
    ShardedNativeRolloutDriver,
)

# --- rollout phase timing -------------------------------------------------
_phase_lock = threading.Lock()
_phase_totals: dict[str, float] = {"poll": 0.0, "encode": 0.0, "sample": 0.0, "step": 0.0}
_phase_counts: dict[str, int] = {"poll": 0, "encode": 0, "sample": 0, "step": 0}


def _reset_phase_counters() -> None:
    with _phase_lock:
        for k in _phase_totals:
            _phase_totals[k] = 0.0
            _phase_counts[k] = 0


def _add_phase(name: str, dt: float) -> None:
    with _phase_lock:
        _phase_totals[name] += dt
        _phase_counts[name] += 1


def _phase_summary(rollout_wall: float) -> str:
    with _phase_lock:
        parts = [f"rollout {rollout_wall:.3f}s"]
        for name in ("poll", "encode", "sample", "step"):
            parts.append(f"{name}={_phase_totals[name]:.3f}s/{_phase_counts[name]}c")
        unaccounted = rollout_wall - sum(_phase_totals.values())
        parts.append(f"other={unaccounted:.3f}s")
        return " ".join(parts)


def _wrap_phase(obj: Any, attr: str, name: str) -> None:
    orig = getattr(obj, attr)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            _add_phase(name, time.perf_counter() - t0)

    setattr(obj, attr, wrapper)


_wrap_phase(ShardedNativeRolloutDriver, "poll", "poll")
_wrap_phase(ShardedNativeRolloutDriver, "step_by_choice", "step")
_wrap_phase(ShardedNativeBatchEncoder, "encode_handles", "encode")
_wrap_phase(PPOPolicy, "sample_native_batch", "sample")
# --------------------------------------------------------------------------

_train_spec = importlib.util.spec_from_file_location("train_mod", _ROOT / "scripts" / "train.py")
assert _train_spec is not None and _train_spec.loader is not None
train_mod = importlib.util.module_from_spec(_train_spec)
sys.modules["train_mod"] = train_mod
_train_spec.loader.exec_module(train_mod)

N_PROFILE = 4
WARMUP = 1
WITH_STACK = False


@dataclass
class ProfileState:
    n: int = 0
    wall_iter: float = 0.0
    wall_update: float = 0.0
    iter_start: float | None = None
    prof: Any | None = None
    captures: list[Any] = field(default_factory=list)


def main() -> None:
    orig = rt.run_rnad_update
    state = ProfileState()

    def start_prof() -> Any:
        p = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=WITH_STACK,
        )
        p.__enter__()
        return p

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        n = state.n
        # First call: there's no live profiler yet; nothing to capture for the
        # rollout that produced this batch (it ran before profiling started).
        # Treat this as the warmup boundary.
        if n == 0:
            t_u0 = time.perf_counter()
            out = orig(*args, **kwargs)
            torch.cuda.synchronize()
            update_wall = time.perf_counter() - t_u0
            print(f"[profile] warmup update: {update_wall:.3f}s wall (skipped)", flush=True)
            state.n = 1
            state.iter_start = time.perf_counter()
            state.prof = start_prof()
            _reset_phase_counters()
            return out

        # Profiler is active — it has been running since the previous wrapped
        # return, capturing the rollout work that built this batch.
        prof = state.prof
        iter_start = state.iter_start
        if prof is None or iter_start is None:
            raise RuntimeError("profile state was not initialized")
        with record_function(f"rnad_update_{n - 1}"):
            t_u0 = time.perf_counter()
            out = orig(*args, **kwargs)
            torch.cuda.synchronize()
            update_wall = time.perf_counter() - t_u0
        prof.__exit__(None, None, None)

        iter_wall = time.perf_counter() - iter_start
        state.wall_iter += iter_wall
        state.wall_update += update_wall
        state.captures.append(prof)
        rollout_wall = iter_wall - update_wall
        print(
            f"[profile] iter #{n}: total {iter_wall:.3f}s "
            f"(rollout {rollout_wall:.3f}s + update {update_wall:.3f}s)",
            flush=True,
        )
        print(f"[profile]    phases: {_phase_summary(rollout_wall)}", flush=True)
        _reset_phase_counters()

        state.n = n + 1
        if n >= N_PROFILE:
            print(
                f"\n[profile] captured {N_PROFILE} iterations "
                f"(plus {WARMUP} warmup). total wall={state.wall_iter:.3f}s, "
                f"update wall={state.wall_update:.3f}s, "
                f"rollout wall={state.wall_iter - state.wall_update:.3f}s",
                flush=True,
            )
            for i, p in enumerate(state.captures):
                print(f"\n========== ITER #{i + 1} (rollout + update) ==========", flush=True)
                ka = p.key_averages()
                print(ka.table(sort_by="cuda_time_total", row_limit=25), flush=True)
                print("\n--- by CPU time ---", flush=True)
                print(ka.table(sort_by="cpu_time_total", row_limit=25), flush=True)
                if WITH_STACK:
                    print(
                        "\n--- by CPU time, grouped by source stack (top-3 frames) ---", flush=True
                    )
                    ka_stack = p.key_averages(group_by_stack_n=3)
                    print(ka_stack.table(sort_by="cpu_time_total", row_limit=30), flush=True)
                    out_path = Path(f"/tmp/prof_iter_{i + 1}.json")
                    p.export_chrome_trace(str(out_path))
                    print(f"[profile] chrome trace -> {out_path}", flush=True)
            sys.exit(0)

        # Open a fresh profiler for the next iteration boundary.
        state.iter_start = time.perf_counter()
        state.prof = start_prof()
        return out

    rt.run_rnad_update = cast(Any, wrapped)
    setattr(train_mod, "run_rnad_update", wrapped)
    train_mod.main()


if __name__ == "__main__":
    main()
