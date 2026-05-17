"""Export Python-owned text-encoder token tables for Rust Forge extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from magic_ai.text_encoder.token_table_export import (  # noqa: E402
    export_token_tables,
    load_card_cache_for_export,
)
from magic_ai.text_encoder.tokenizer import TOKENIZER_DIR, load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=TOKENIZER_DIR,
        help="Augmented tokenizer directory.",
    )
    parser.add_argument(
        "--card-cache",
        type=Path,
        default=REPO_ROOT / "data" / "text_encoder_card_tokens.pt",
        help="CardTokenCache checkpoint produced by build_text_encoder_card_cache.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "text_encoder_token_tables" / "token_tables.json",
        help="Token-table artifact consumed by rust/forge_extract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    cache = load_card_cache_for_export(args.card_cache)
    path = export_token_tables(args.output, tokenizer, cache)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes).")
    print(f"  vocab_size: {len(tokenizer):,}")
    print(f"  card_rows:  {cache.num_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
