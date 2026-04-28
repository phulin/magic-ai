"""Build the pre-tokenized card-body cache (``data/text_encoder_card_tokens.pt``).

PR 13-A from ``docs/text_encoder_plan.md``. Run once at startup (or whenever
the engine card set changes); the hot-path assembler memcpys slices from this
cache instead of running BPE per step.

Examples
--------

    # Use the engine's card list (requires libmage to be built):
    python scripts/build_text_encoder_card_cache.py

    # Use a custom Scryfall oracle-cards bulk dump:
    python scripts/build_text_encoder_card_cache.py --oracle-db /path/oracle-cards.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from magic_ai.text_encoder.card_cache import (  # noqa: E402
    DEFAULT_ORACLE_DB_PATH,
    build_card_cache,
    cache_length_stats,
    fetch_registered_card_names_from_engine,
    load_oracle_db,
    save_card_cache,
)
from magic_ai.text_encoder.tokenizer import load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oracle-db",
        type=Path,
        default=DEFAULT_ORACLE_DB_PATH,
        help="Scryfall oracle-cards.json bulk dump (default: data/oracle-cards.json).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "text_encoder_card_tokens.pt",
        help="Where to write the PyTorch cache.",
    )
    p.add_argument(
        "--missing",
        choices=("raise", "warn", "skip"),
        default="raise",
        help="How to handle registered cards with no oracle entry.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer()
    names = fetch_registered_card_names_from_engine()
    oracle = load_oracle_db(args.oracle_db, names=names)

    cache = build_card_cache(
        names,
        oracle,
        tokenizer,
        oracle_db_path=args.oracle_db,
        missing_policy=args.missing,
    )
    save_card_cache(cache, args.output)
    stats = cache_length_stats(cache)
    file_size = args.output.stat().st_size

    print(f"Wrote {args.output} ({file_size:,} bytes).")
    print(f"  cards (rows excl. unknown): {stats['count']}")
    print(f"  total tokens cached:        {int(cache.token_buffer.numel()):,}")
    print(
        "  body length: mean={mean:.1f}  p50={p50}  p90={p90}  max={max}".format(
            mean=float(stats["mean"]),
            p50=int(stats["p50"]),
            p90=int(stats["p90"]),
            max=int(stats["max"]),
        )
    )
    print(f"  engine_card_set_hash:       {cache.engine_card_set_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
