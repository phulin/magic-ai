"""Thin CLI runner for ``TextRolloutWorker`` — pipeline smoke (PR 13-D).

Plays N episodes with a freshly-initialized :class:`RecurrentTextPolicy`
against the mage engine via ``magic_ai.text_encoder.rollout``. The goal is
"the cache + emitter + assembler + recurrent policy run end-to-end on real
game states without exception" — convergence is *not* the goal here.

Examples
--------

    PYTHONPATH=. uv run python scripts/play_text_rollout.py \
        --n-episodes 2 --max-turns 50 --device cpu --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from magic_ai.text_encoder.card_cache import (  # noqa: E402
    build_card_cache,
    load_card_cache,
)
from magic_ai.text_encoder.model import TextEncoderConfig  # noqa: E402
from magic_ai.text_encoder.recurrent import (  # noqa: E402
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
)
from magic_ai.text_encoder.render import load_oracle_text  # noqa: E402
from magic_ai.text_encoder.rollout import TextRolloutWorker  # noqa: E402
from magic_ai.text_encoder.tokenizer import load_tokenizer  # noqa: E402

DEFAULT_CACHE = REPO_ROOT / "data" / "text_encoder_card_tokens.npz"
DEFAULT_DECK = REPO_ROOT / "decks" / "bears.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-episodes", type=int, default=2)
    p.add_argument("--max-turns", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--decks",
        type=Path,
        default=DEFAULT_DECK,
        help="Path to a deck JSON ({player_a, player_b} or single deck).",
    )
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Encoder d_model (smoke only — random init is fine).",
    )
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument(
        "--native-render-plan",
        action="store_true",
        help="Use mage-go's native render-plan emitter instead of the Python parity emitter.",
    )
    p.add_argument("--render-plan-capacity", type=int, default=4096)
    return p.parse_args()


def _load_decks(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text())
    if "player_a" in payload or "player_b" in payload:
        return payload.get("player_a", payload), payload.get("player_b", payload)
    return payload, payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    tokenizer = load_tokenizer()
    oracle = load_oracle_text()

    if args.cache.exists():
        cache = load_card_cache(args.cache)
        print(f"Loaded card cache from {args.cache} ({cache.num_rows - 1} cards).")
    else:
        names = sorted(oracle.keys())
        cache = build_card_cache(names, oracle, tokenizer, missing_policy="warn")
        print(
            f"Built in-memory card cache from oracle ({cache.num_rows - 1} cards); "
            f"{args.cache} not found — pass --cache or run "
            "scripts/build_text_encoder_card_cache.py to persist."
        )

    cfg = TextEncoderConfig(
        vocab_size=len(tokenizer),
        pad_id=int(tokenizer.pad_token_id or 0),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    rcfg = RecurrentTextPolicyConfig(encoder=cfg, lstm_hidden=args.d_model)
    policy = RecurrentTextPolicy(rcfg)
    policy.eval()

    deck_a, deck_b = _load_decks(args.decks)
    base_seed = int(args.seed)

    worker = TextRolloutWorker(
        policy=policy,
        cache=cache,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        device=args.device,
        sampling_temperature=args.temperature,
        oracle=oracle,
        seed=base_seed,
        use_native_render_plan=bool(args.native_render_plan),
        render_plan_capacity=int(args.render_plan_capacity),
    )

    total_steps = 0
    total_wall = 0.0
    for ep_i in range(int(args.n_episodes)):
        cfg_dict = {
            "deck_a": deck_a,
            "deck_b": deck_b,
            "seed": base_seed + ep_i,
            "shuffle": True,
        }
        t0 = time.perf_counter()
        episode = worker.play_episode(cfg_dict, max_turns=int(args.max_turns))
        dt = time.perf_counter() - t0
        total_steps += len(episode.steps)
        total_wall += dt
        total_reward = sum(s.reward for s in episode.steps)
        print(
            f"[ep {ep_i}] turns={episode.turns} steps={len(episode.steps)} "
            f"winner={episode.winner_player_idx} total_reward={total_reward:.1f} "
            f"wall={dt:.2f}s "
            f"step_ms={(dt / max(1, len(episode.steps)) * 1000):.2f}"
        )

    if total_steps:
        print(
            f"[mean] steps/ep={total_steps / max(1, args.n_episodes):.1f} "
            f"step_ms={total_wall / total_steps * 1000:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
