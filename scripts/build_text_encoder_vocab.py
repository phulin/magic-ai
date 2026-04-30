"""Build the augmented ModernBERT tokenizer for the text encoder.

Loads the upstream ModernBERT tokenizer at a pinned revision, registers every
custom token string defined in :mod:`magic_ai.text_encoder.tokenizer` as an
``additional_special_token`` (so each maps to a single id and is never
BPE-split), and saves the result via ``save_pretrained``.

Idempotent: running again overwrites the existing tokenizer directory.

Usage::

    uv run scripts/build_text_encoder_vocab.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow direct invocation as ``uv run scripts/build_text_encoder_vocab.py``
# without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.text_encoder.tokenizer import (  # noqa: E402
    ALL_CUSTOM_TOKENS,
    MODERNBERT_REPO,
    MODERNBERT_REVISION,
    TOKENIZER_DIR,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-tokenizer",
        default=MODERNBERT_REPO,
        help="HF tokenizer/model repo to augment with text-encoder custom tokens",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="optional HF revision for --base-tokenizer; defaults to the pinned "
        "ModernBERT revision when using the built-in base tokenizer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = args.revision
    if revision is None and args.base_tokenizer == MODERNBERT_REPO:
        revision = MODERNBERT_REVISION
    rev_label = revision[:12] if revision else "default"
    print(f"Loading base tokenizer {args.base_tokenizer}@{rev_label}…")
    tok = AutoTokenizer.from_pretrained(args.base_tokenizer, revision=revision)
    if not isinstance(tok, PreTrainedTokenizerFast):
        raise TypeError(f"Expected PreTrainedTokenizerFast, got {type(tok).__name__}")

    base_size = len(tok)
    print(f"  base vocab size: {base_size}")
    print(f"  registering {len(ALL_CUSTOM_TOKENS)} custom tokens")

    # `additional_special_tokens` ensures each entry is treated as a single
    # atomic token (never split by the BPE merges) and is excluded from
    # normalization. This is the contract the renderer relies on.
    added = tok.add_special_tokens(
        {"additional_special_tokens": list(ALL_CUSTOM_TOKENS)},
        replace_extra_special_tokens=True,
    )
    print(f"  add_special_tokens reported {added} newly-added tokens")

    # Idempotent overwrite of the artifact directory.
    if TOKENIZER_DIR.exists():
        shutil.rmtree(TOKENIZER_DIR)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(TOKENIZER_DIR))
    print(f"  saved tokenizer to {TOKENIZER_DIR}")

    final_size = len(tok)
    print(f"final vocab size: {final_size} (base {base_size} + added {added})")

    # Sanity-check every custom token is exactly one id.
    bad: list[tuple[str, list[int]]] = []
    for s in ALL_CUSTOM_TOKENS:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            bad.append((s, ids))
    if bad:
        raise RuntimeError(f"{len(bad)} custom tokens did not map to a single id, e.g. {bad[:5]}")
    print("all custom tokens map to a single id")


if __name__ == "__main__":
    main()
