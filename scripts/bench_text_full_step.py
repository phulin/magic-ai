"""Benchmark full native text-encoder rollout steps with phase timings.

Measures the hot poll -> native token encode -> packed batch construction ->
policy sample -> native step_by_choice loop. This is intentionally thinner
than train.py so individual phase costs are visible.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mage
import torch
from magic_ai.game_state import PendingState
from magic_ai.native.sharded import ShardedNativeBatchEncoder, ShardedNativeRolloutDriver
from magic_ai.slot_encoder.native_encoder import TRACE_KIND_VALUES
from magic_ai.text_encoder.actor_critic import TextActorCritic
from magic_ai.text_encoder.card_cache import build_card_cache, load_card_cache
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.native_token_tables import register_native_token_tables
from magic_ai.text_encoder.recurrent import RecurrentTextPolicyConfig
from magic_ai.text_encoder.render import load_oracle_text
from magic_ai.text_encoder.rollout import _default_action_for, _translate_action
from magic_ai.text_encoder.token_tables import build_token_tables
from magic_ai.text_encoder.tokenizer import load_tokenizer

DEFAULT_DECK = Path("decks/bears.json")
DEFAULT_CACHE = Path("data/text_encoder_card_tokens.pt")


class PhaseTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.samples: dict[str, list[float]] = {}

    def add(self, name: str, elapsed: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1
        self.samples.setdefault(name, []).append(1000.0 * elapsed)

    def mean_ms(self, name: str) -> float:
        count = max(1, self.counts.get(name, 0))
        return 1000.0 * self.totals.get(name, 0.0) / count

    def percentile_ms(self, name: str, q: float) -> float:
        return percentile(self.samples.get(name, []), q)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((q / 100.0) * len(ordered)) - 1))
    return float(ordered[idx])


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_decks(path: Path) -> tuple[Any, Any]:
    payload = json.loads(path.read_text())
    return payload.get("player_a", payload), payload.get("player_b", payload)


def build_policy(args: argparse.Namespace, tokenizer_len: int, pad_id: int) -> TextActorCritic:
    cfg = TextEncoderConfig(
        vocab_size=tokenizer_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_tokens,
        pad_id=pad_id,
        dropout=0.0,
    )
    return TextActorCritic(
        RecurrentTextPolicyConfig(
            encoder=cfg,
            lstm_hidden=args.d_model,
            lstm_layers=args.lstm_layers,
            compile_forward=not args.no_compile,
        )
    )


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=4)
    parser.add_argument("--measure-iters", type=int, default=16)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--max-card-refs", type=int, default=256)
    parser.add_argument("--max-cached-choices", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--derive-token-metadata", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--shard-packed-tokens",
        action="store_true",
        help="also shard native packed token assembly; off by default because merge can dominate",
    )
    parser.add_argument(
        "--reset-each-iter",
        action="store_true",
        help="recreate all games after each measured step to keep batches at initial priority",
    )
    parser.add_argument(
        "--card-body-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "emit each unique card body once and reference per-zone occurrences by "
            "compact dict slot"
        ),
    )
    parser.add_argument(
        "--profile-sample-components",
        action="store_true",
        help="synchronize inside sample_native_tensor_batch and report sub-stage timings",
    )
    args = parser.parse_args()

    mage.load()
    tokenizer = load_tokenizer()
    oracle = load_oracle_text()
    cache = (
        load_card_cache(args.cache)
        if args.cache.exists()
        else build_card_cache(sorted(oracle.keys()), oracle, tokenizer, missing_policy="warn")
    )
    register_native_token_tables(build_token_tables(tokenizer, cache))
    deck_a, deck_b = load_decks(args.deck)

    device = torch.device(args.device)
    policy = build_policy(
        args,
        tokenizer_len=len(tokenizer),
        pad_id=int(tokenizer.pad_token_id or 0),
    ).to(device)
    policy.eval()
    policy.init_lstm_env_states(args.num_envs)

    pool = (
        ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="text-step-bench")
        if args.workers > 1
        else None
    )
    native_encoder = ShardedNativeBatchEncoder.for_text(
        max_options=args.max_options,
        max_targets_per_option=args.max_targets,
        max_cached_choices=args.max_cached_choices,
        zone_slot_count=50,
        game_info_dim=90,
        option_scalar_dim=14,
        target_scalar_dim=2,
        card_name_to_row={name: idx for idx, name in enumerate(cache.row_to_name)},
        emit_render_plan=False,
        render_plan_capacity=4096,
        validate=False,
        workers=args.workers,
        pool=pool,
        dedup_card_bodies=args.card_body_dedup,
        shard_packed_tokens=args.shard_packed_tokens,
    )
    native_rollout = ShardedNativeRolloutDriver.for_mage(mage, workers=args.workers, pool=pool)

    def new_game(seed: int) -> Any:
        game = mage.new_game(deck_a, deck_b, seed=seed, shuffle=True, hand_size=7)
        if not args.reset_each_iter:
            return game
        for _ in range(400):
            game.refresh_state()
            if game.is_over:
                break
            pending = game.pending or game.legal()
            if pending is None:
                game.step({"kind": "pass"})
                continue
            pending_t = cast(PendingState, pending)
            kind = pending.get("kind", "") or ""
            options = list(pending.get("options", []) or [])
            if kind == "priority" and options:
                return game
            if not options:
                action = _default_action_for(pending_t)
            elif kind == "priority":
                action = _translate_action(pending_t, 0, None)
            else:
                action = _default_action_for(pending_t)
            game.step(dict(action))
        return game

    games = [new_game(args.seed + i) for i in range(args.num_envs)]
    next_seed = args.seed + args.num_envs
    action_counts = [0] * args.num_envs
    phase = PhaseTimer()
    measured_steps = 0
    measured_ready = 0
    measured_iters = 0
    measured_tokens: list[float] = []
    measured_max_seq: list[float] = []
    measured_trace_counts = {name: 0 for name in TRACE_KIND_VALUES}
    sample_component = PhaseTimer()

    try:
        total_iters = args.warmup_iters + args.measure_iters
        for iter_idx in range(total_iters):
            measuring = iter_idx >= args.warmup_iters
            iter_start = time.perf_counter()

            start = time.perf_counter()
            ready_t, over_t, player_t, _winner_t = native_rollout.poll(games)
            phase.add("poll", time.perf_counter() - start) if measuring else None
            ready_l = ready_t.tolist()
            over_l = over_t.tolist()
            player_l = player_t.tolist()

            start = time.perf_counter()
            for i, is_over in enumerate(over_l):
                if is_over:
                    try:
                        games[i].close()
                    except Exception:
                        pass
                    games[i] = new_game(next_seed)
                    next_seed += 1
                    action_counts[i] = 0
            if args.reset_each_iter and iter_idx > 0:
                for i, game in enumerate(games):
                    try:
                        game.close()
                    except Exception:
                        pass
                    games[i] = new_game(next_seed)
                    next_seed += 1
                    action_counts[i] = 0
                policy.reset_lstm_env_states(list(range(args.num_envs)))
            phase.add("recycle", time.perf_counter() - start) if measuring else None

            if args.reset_each_iter and iter_idx > 0:
                start = time.perf_counter()
                ready_t, over_t, player_t, _winner_t = native_rollout.poll(games)
                phase.add("poll", time.perf_counter() - start) if measuring else None
                ready_l = ready_t.tolist()
                over_l = over_t.tolist()
                player_l = player_t.tolist()

            ready_indices = [
                i
                for i, (ready, is_over) in enumerate(zip(ready_l, over_l, strict=True))
                if ready and not is_over
            ]
            if not ready_indices:
                continue
            ready_games = [games[i] for i in ready_indices]
            ready_players = [
                int(player_l[i]) if int(player_l[i]) in (0, 1) else 0 for i in ready_indices
            ]

            start = time.perf_counter()
            native_batch, nat_outputs = native_encoder.encode_tokens_packed(
                ready_games,
                perspective_player_indices=ready_players,
                max_tokens=args.max_tokens,
                max_options=args.max_options,
                max_targets=args.max_targets,
                max_card_refs=args.max_card_refs,
            )
            phase.add("encode", time.perf_counter() - start) if measuring else None
            if measuring:
                trace_ids = native_batch.trace_kind_id[: len(ready_indices)].tolist()
                for trace_id in trace_ids:
                    idx = int(trace_id)
                    if 0 <= idx < len(TRACE_KIND_VALUES):
                        measured_trace_counts[TRACE_KIND_VALUES[idx]] += 1

            start = time.perf_counter()
            packed = nat_outputs.to_packed_text_batch(
                trim=True,
                derive_token_metadata=args.derive_token_metadata,
            )
            phase.add("pack", time.perf_counter() - start) if measuring else None
            if measuring:
                measured_tokens.append(float(packed.total_tokens))
                measured_max_seq.append(float(packed.max_seqlen or 0))

            sync(device)
            start = time.perf_counter()
            sample_profile: dict[str, float] | None = (
                {} if measuring and args.profile_sample_components else None
            )
            with torch.inference_mode():
                policy_batch = policy.sample_native_tensor_batch(
                    native_batch=native_batch,
                    env_indices=ready_indices,
                    perspective_player_indices=ready_players,
                    packed_batch=packed,
                    deterministic=False,
                    append_replay=False,
                    return_replay_payload=False,
                    profile_timings=sample_profile,
                )
            sync(device)
            phase.add("sample", time.perf_counter() - start) if measuring else None
            if sample_profile is not None:
                for name, elapsed in sample_profile.items():
                    sample_component.add(name, elapsed)

            start = time.perf_counter()
            counts = policy_batch.decision_counts
            starts = list(itertools.accumulate(counts, initial=0))[:-1]
            phase.add("prepare_step", time.perf_counter() - start) if measuring else None

            start = time.perf_counter()
            native_rollout.step_by_choice(
                ready_games,
                decision_starts=starts,
                decision_counts=counts,
                selected_choice_cols=policy_batch.selected_choice_cols,
                may_selected=policy_batch.may_selected,
                max_options=args.max_options,
                max_targets_per_option=args.max_targets,
            )
            phase.add("step", time.perf_counter() - start) if measuring else None
            for i in ready_indices:
                action_counts[i] += 1

            if measuring:
                phase.add("total_iter", time.perf_counter() - iter_start)
                measured_steps += len(ready_indices)
                measured_ready += len(ready_indices)
                measured_iters += 1
    finally:
        for game in games:
            try:
                game.close()
            except Exception:
                pass
        if pool is not None:
            pool.shutdown(wait=False)

    total_s = phase.totals.get("total_iter", 0.0)
    print(
        {
            "num_envs": args.num_envs,
            "workers": args.workers,
            "measure_iters": measured_iters,
            "env_steps": measured_steps,
            "env_steps_per_s": round(measured_steps / max(total_s, 1e-9), 1),
            "avg_ready_batch": round(measured_ready / max(measured_iters, 1), 1),
            "device": str(device),
            "layers": args.layers,
            "d_model": args.d_model,
            "derive_token_metadata": args.derive_token_metadata,
            "reset_each_iter": args.reset_each_iter,
            "card_body_dedup": args.card_body_dedup,
            "shard_packed_tokens": args.shard_packed_tokens,
        }
    )
    print(
        {
            "tokens_p50": round(percentile(measured_tokens, 50.0), 1),
            "tokens_p90": round(percentile(measured_tokens, 90.0), 1),
            "tokens_max": round(max(measured_tokens, default=0.0), 1),
            "max_seq_p50": round(percentile(measured_max_seq, 50.0), 1),
            "max_seq_p90": round(percentile(measured_max_seq, 90.0), 1),
            "max_seq_max": round(max(measured_max_seq, default=0.0), 1),
        }
    )
    print({"trace_counts": measured_trace_counts})
    print(f"{'phase':>14} {'mean_ms':>10} {'p50_ms':>10} {'p90_ms':>10} {'pct_iter':>9}")
    for name in ("poll", "recycle", "encode", "pack", "sample", "prepare_step", "step"):
        mean = phase.mean_ms(name)
        p50 = phase.percentile_ms(name, 50.0)
        p90 = phase.percentile_ms(name, 90.0)
        pct = 100.0 * phase.totals.get(name, 0.0) / max(total_s, 1e-9)
        print(f"{name:>14} {mean:>10.3f} {p50:>10.3f} {p90:>10.3f} {pct:>8.1f}%")
    print(
        f"{'total_iter':>14} {phase.mean_ms('total_iter'):>10.3f} "
        f"{phase.percentile_ms('total_iter', 50.0):>10.3f} "
        f"{phase.percentile_ms('total_iter', 90.0):>10.3f} {100.0:>8.1f}%"
    )
    if args.profile_sample_components:
        print(f"{'sample_part':>24} {'mean_ms':>10} {'p50_ms':>10} {'p90_ms':>10}")
        for name in (
            "move_text",
            "lstm_state_in",
            "forward",
            "lstm_state_out",
            "native_metadata_to_device",
            "decision_init",
            "decision_tensors_to_device",
            "decision_full_layout",
            "decision_inline_fast",
            "decision_sampling",
            "decision_priority_batch",
            "decision_choice_batch",
            "decision_binary_batch",
            "decision_merge_batch",
            "decision_accept_check",
            "decision_post_decision",
            "decision_may",
            "replay_payload",
            "append_replay",
            "host_return",
        ):
            print(
                f"{name:>24} {sample_component.mean_ms(name):>10.3f} "
                f"{sample_component.percentile_ms(name, 50.0):>10.3f} "
                f"{sample_component.percentile_ms(name, 90.0):>10.3f}"
            )


if __name__ == "__main__":
    run()
