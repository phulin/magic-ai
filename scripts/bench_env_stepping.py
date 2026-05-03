"""Benchmark mage-go env stepping throughput.

Drives the same poll → encode → policy-sample → step_by_choice loop the PPO
trainer uses, but skips staging/transcript/advantages so the measurement
reflects just the environment-stepping side. Sweeps `--num-envs` x
`--batch-workers` so we can see where CPU stepping saturates.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from magic_ai.native.sharded import ShardedNativeBatchEncoder, ShardedNativeRolloutDriver
from magic_ai.slot_encoder.game_state import GameStateEncoder
from magic_ai.slot_encoder.model import PPOPolicy
from scripts.train import load_deck_pool, sample_decks


def build_policy(args: argparse.Namespace, device: torch.device) -> PPOPolicy:
    encoder = GameStateEncoder.from_embedding_json(args.embeddings, d_model=args.d_model)
    policy = PPOPolicy(
        encoder,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        rollout_capacity=4096,
        use_lstm=False,
        spr_enabled=False,
        validate=False,
        compile_forward=False,
    ).to(device)
    return policy


def run_one_config(
    *,
    num_envs: int,
    workers: int,
    seconds: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    mage = importlib.import_module("mage")

    policy = build_policy(args, device)
    pool = (
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bench") if workers > 1 else None
    )
    encoder = ShardedNativeBatchEncoder.for_policy(policy, workers=workers, pool=pool)
    rollout = ShardedNativeRolloutDriver.for_mage(mage, workers=workers, pool=pool)
    deck_pool = load_deck_pool(args.deck_json, args.deck_dir)

    def new_game(seed: int):
        deck_a, deck_b = sample_decks(deck_pool, seed)
        return mage.new_game(
            deck_a, deck_b, name_a="A", name_b="B", seed=seed, shuffle=True, hand_size=7
        )

    games = [new_game(args.seed + i) for i in range(num_envs)]
    action_counts = [0] * num_envs
    next_seed = args.seed + num_envs

    # Warmup: a short run so JIT/cuDNN/cgo stabilise.
    warmup_deadline = time.monotonic() + max(1.0, seconds * 0.2)
    measure_steps = 0
    measure_iters = 0
    measure_ready_total = 0
    finished_games = 0
    finished_lengths = 0
    measuring = False
    measure_start = 0.0

    deadline = time.monotonic() + seconds + (warmup_deadline - time.monotonic())

    while True:
        now = time.monotonic()
        if not measuring and now >= warmup_deadline:
            measuring = True
            measure_start = now
            measure_steps = 0
            measure_iters = 0
            measure_ready_total = 0
            finished_games = 0
            finished_lengths = 0
            deadline = now + seconds
        if measuring and now >= deadline:
            break

        ready_t, over_t, _player_t, _winner_t = rollout.poll(games)
        ready_l = ready_t.tolist()
        over_l = over_t.tolist()

        # Recycle finished or stuck games.
        for i, g in enumerate(games):
            if over_l[i] or action_counts[i] >= args.max_steps_per_game:
                try:
                    g.close()
                except Exception:
                    pass
                if measuring:
                    finished_games += 1
                    finished_lengths += action_counts[i]
                games[i] = new_game(next_seed)
                next_seed += 1
                action_counts[i] = 0

        ready_envs = [g for g, r, o in zip(games, ready_l, over_l, strict=True) if r and not o]
        ready_idxs = [
            i for i, (r, o) in enumerate(zip(ready_l, over_l, strict=True)) if r and not o
        ]
        if not ready_envs:
            continue
        ready_players = [0] * len(ready_envs)  # perspective doesn't matter for stepping cost
        parsed_batch = encoder.encode_handles(ready_envs, perspective_player_indices=ready_players)
        with torch.no_grad():
            steps = policy.sample_native_batch(
                parsed_batch, env_indices=ready_idxs, deterministic=False
            )
        counts = [len(s.selected_choice_cols) for s in steps]
        starts = list(itertools.accumulate(counts, initial=0))[:-1]
        selected = [c for s in steps for c in s.selected_choice_cols]
        may_selected = [s.may_selected for s in steps]
        rollout.step_by_choice(
            ready_envs,
            decision_starts=starts,
            decision_counts=counts,
            selected_choice_cols=selected,
            may_selected=may_selected,
            max_options=args.max_options,
            max_targets_per_option=args.max_targets_per_option,
        )
        for idx in ready_idxs:
            action_counts[idx] += 1
        if measuring:
            measure_steps += len(ready_envs)
            measure_iters += 1
            measure_ready_total += len(ready_envs)

    elapsed = max(1e-6, time.monotonic() - measure_start)
    for g in games:
        try:
            g.close()
        except Exception:
            pass
    if pool is not None:
        pool.shutdown(wait=False)
    del policy, encoder, rollout
    if device.type == "cuda":
        torch.cuda.empty_cache()

    avg_finished_len = finished_lengths / finished_games if finished_games else float("nan")
    return {
        "num_envs": num_envs,
        "workers": workers,
        "elapsed_s": elapsed,
        "iters": measure_iters,
        "env_steps": measure_steps,
        "iters_per_s": measure_iters / elapsed,
        "env_steps_per_s": measure_steps / elapsed,
        "avg_ready_batch": measure_ready_total / max(1, measure_iters),
        "finished_games": finished_games,
        "avg_finished_game_len": avg_finished_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.json"))
    parser.add_argument("--deck-json", type=Path, default=None)
    parser.add_argument("--deck-dir", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--batch-workers", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--max-steps-per-game", type=int, default=200)
    parser.add_argument("--max-options", type=int, default=64)
    parser.add_argument("--max-targets-per-option", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(
        f"device={device} torch_threads={torch.get_num_threads()} "
        f"max_options={args.max_options} max_targets={args.max_targets_per_option}"
    )
    print(
        f"{'envs':>5} {'workers':>7} {'iters/s':>9} {'env_steps/s':>12} "
        f"{'avg_ready':>10} {'avg_game_len':>13} {'finished':>9}"
    )
    rows = []
    for n_env, n_workers in itertools.product(args.num_envs, args.batch_workers):
        if n_workers > n_env:
            continue
        result = run_one_config(
            num_envs=n_env, workers=n_workers, seconds=args.seconds, args=args, device=device
        )
        rows.append(result)
        print(
            f"{result['num_envs']:>5d} {result['workers']:>7d} "
            f"{result['iters_per_s']:>9.1f} {result['env_steps_per_s']:>12.0f} "
            f"{result['avg_ready_batch']:>10.1f} {result['avg_finished_game_len']:>13.1f} "
            f"{result['finished_games']:>9d}",
            flush=True,
        )

    if rows:
        best = max(rows, key=lambda r: r["env_steps_per_s"])
        print(
            f"\nbest: envs={best['num_envs']} workers={best['workers']} "
            f"-> {best['env_steps_per_s']:.0f} env-steps/s, "
            f"{best['env_steps_per_s'] / max(1, best['workers']):.0f} per worker"
        )


if __name__ == "__main__":
    main()
