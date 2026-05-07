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
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from magic_ai.native.sharded import (
    ShardedNativeBatchEncoder,
    ShardedNativeRolloutDriver,
    _collect,
    _shard_ranges,
)
from magic_ai.slot_encoder.game_state import GameStateEncoder
from magic_ai.slot_encoder.native_encoder import NativeBatchEncoder

DEFAULT_DECK = {
    "name": "bolt-mountain",
    "cards": [
        {"name": "Mountain", "count": 24},
        {"name": "Lightning Bolt", "count": 36},
    ],
}


def load_decks(path: Path | None) -> tuple[dict, dict]:
    if path is None:
        return dict(DEFAULT_DECK), dict(DEFAULT_DECK)
    payload = json.loads(path.read_text())
    if "player_a" in payload or "player_b" in payload:
        return dict(payload.get("player_a", DEFAULT_DECK)), dict(
            payload.get("player_b", DEFAULT_DECK)
        )
    return dict(payload), dict(payload)


def load_deck_pool(deck_json: Path | None, deck_dir: Path | None) -> list[dict]:
    if deck_dir is None:
        return list(load_decks(deck_json))
    if not deck_dir.exists():
        raise FileNotFoundError(f"deck directory does not exist: {deck_dir}")
    if not deck_dir.is_dir():
        raise NotADirectoryError(f"deck directory path is not a directory: {deck_dir}")
    decks = list(map(lambda p: dict(json.loads(p.read_text())), sorted(deck_dir.glob("*.json"))))
    if not decks:
        raise ValueError(f"deck directory contains no JSON decks: {deck_dir}")
    return decks


def sample_decks(deck_pool: list[dict], seed: int) -> tuple[dict, dict]:
    if not deck_pool:
        raise ValueError("deck pool must contain at least one deck")
    rng = random.Random(seed)
    return rng.choice(deck_pool), rng.choice(deck_pool)


def build_policy(args: argparse.Namespace, device: torch.device) -> Any:
    from magic_ai.slot_encoder.model import PPOPolicy

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


def build_random_action_encoder(
    args: argparse.Namespace,
    *,
    workers: int,
    pool: ThreadPoolExecutor | None,
) -> ShardedNativeBatchEncoder:
    mage = importlib.import_module("mage")
    if mage._lib is None or mage._ffi is None:
        mage.load()
    game_encoder = GameStateEncoder.from_embedding_json(args.embeddings, d_model=args.d_model)
    encoders = list(
        map(
            lambda idx: NativeBatchEncoder(
                max_options=args.max_options,
                max_targets_per_option=args.max_targets_per_option,
                max_cached_choices=64,
                zone_slot_count=50,
                game_info_dim=90,
                option_scalar_dim=14,
                target_scalar_dim=2,
                lib=mage._lib,
                ffi=mage._ffi,
                card_name_to_row=game_encoder._card_name_to_row if idx == 0 else None,
                validate=False,
            ),
            range(workers),
        )
    )
    return ShardedNativeBatchEncoder(encoders, pool=pool if workers > 1 else None)


def sample_random_native_actions(parsed_batch) -> tuple[list[int], list[int], list[int]]:
    counts = parsed_batch.decision_count.tolist()
    total = int(parsed_batch.decision_count.sum().item())
    may_selected = [0] * len(counts)
    if total <= 0:
        return counts, [], may_selected
    valid_counts = parsed_batch.decision_mask[:total].sum(dim=1).clamp_min(1)
    selected = (torch.rand((total,)) * valid_counts).to(dtype=torch.long).tolist()
    return counts, selected, may_selected


def sample_random_actions_sharded(
    encoder: ShardedNativeBatchEncoder,
    pool: ThreadPoolExecutor | None,
    games: list,
    players: list[int],
) -> tuple[list[int], list[int], list[int]]:
    encoders = encoder.encoders
    if pool is None or len(games) <= 1 or len(encoders) == 1:
        return sample_random_native_actions(
            encoders[0].encode_handles(games, perspective_player_indices=players)
        )

    shards = _shard_ranges(len(games), len(encoders))
    results = [None] * len(shards)

    def run(idx: int, a: int, b: int) -> None:
        results[idx] = sample_random_native_actions(
            encoders[idx].encode_handles(games[a:b], perspective_player_indices=players[a:b])
        )

    futures = list(
        map(lambda item: pool.submit(run, item[0], item[1][0], item[1][1]), enumerate(shards))
    )
    _collect(futures)
    triples = list(results)
    return (
        list(itertools.chain.from_iterable(map(lambda x: x[0], triples))),
        list(itertools.chain.from_iterable(map(lambda x: x[1], triples))),
        list(itertools.chain.from_iterable(map(lambda x: x[2], triples))),
    )


def run_one_config(
    *,
    num_envs: int,
    workers: int,
    seconds: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    mage = importlib.import_module("mage")

    pool = (
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bench") if workers > 1 else None
    )
    policy = None if args.random_actions else build_policy(args, device)
    encoder = (
        build_random_action_encoder(args, workers=workers, pool=pool)
        if args.random_actions
        else ShardedNativeBatchEncoder.for_policy(policy, workers=workers, pool=pool)
    )
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
            if args.native_timing and hasattr(mage, "native_timing_summary"):
                mage.native_timing_summary(reset=True)
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
        if args.random_actions:
            counts, selected, may_selected = sample_random_actions_sharded(
                encoder,
                pool,
                ready_envs,
                ready_players,
            )
        else:
            assert policy is not None
            parsed_batch = encoder.encode_handles(
                ready_envs, perspective_player_indices=ready_players
            )
            with torch.no_grad():
                steps = policy.sample_native_batch(
                    parsed_batch, env_indices=ready_idxs, deterministic=False
                )
            counts = list(map(lambda s: len(s.selected_choice_cols), steps))
            selected = list(
                itertools.chain.from_iterable(map(lambda s: s.selected_choice_cols, steps))
            )
            may_selected = list(map(lambda s: s.may_selected, steps))
        starts = list(itertools.accumulate(counts, initial=0))[:-1]
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
    native_timing = None
    if args.native_timing and hasattr(mage, "native_timing_summary"):
        native_timing = mage.native_timing_summary(reset=False)

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
        "native_timing": native_timing,
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
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="sample legal native decision columns randomly instead of running the policy",
    )
    parser.add_argument(
        "--native-timing",
        action="store_true",
        help="print mage-go native encode/step timing counters for the measured window",
    )
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
        timing = result.get("native_timing")
        if timing:
            loop = timing.get("loop") or {}
            engine = timing.get("engine") or {}
            print(
                "      native "
                f"encode={timing['encode_total_per_row_ms'] * 1000:.1f}us/row "
                f"(state={timing['encode_state_action_per_row_ms'] * 1000:.1f}, "
                f"decision={timing['encode_decision_per_row_ms'] * 1000:.1f}) "
                f"step={timing['step_total_per_row_ms'] * 1000:.1f}us/row "
                f"(pending={timing['step_pending_action_per_row_ms'] * 1000:.1f}, "
                f"route={timing['step_route_per_row_ms'] * 1000:.1f}, "
                f"wait={timing['step_wait_per_row_ms'] * 1000:.1f})",
                flush=True,
            )
            if loop:
                print(
                    "      loop "
                    f"run_step={loop['loop_step_per_call_ms'] * 1000:.1f}us/call "
                    f"get_actions={loop['loop_get_available_per_call_us']:.1f}us/call "
                    f"send_prompt={loop['loop_send_prompt_per_call_us']:.1f}us/call "
                    f"read_wait={loop['loop_read_action_per_call_us']:.1f}us/call "
                    f"auto_passes={loop['loop_auto_passes']} "
                    f"decision_prompts={loop['loop_prompt_decision_sends']} "
                    f"prompt_none={loop['loop_prompt_none_sends']}",
                    flush=True,
                )
            if engine:
                print(
                    "      engine "
                    f"sba={engine['engine_sba_per_iteration_us']:.1f}us/iter "
                    f"triggers={engine['engine_triggers_per_iteration_us']:.1f}us/iter "
                    f"on_priority={engine['engine_on_priority_per_iteration_us']:.1f}us/iter "
                    f"execute={engine['engine_execute_per_action_us']:.1f}us/action "
                    f"rounds={engine['engine_priority_rounds']} "
                    f"iters={engine['engine_priority_iterations']}",
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
