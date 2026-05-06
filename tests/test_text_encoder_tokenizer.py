"""Round-trip tests for the augmented ModernBERT tokenizer.

The build script ``scripts/build_text_encoder_vocab.py`` must have run first
so ``data/text_encoder_tokenizer/`` exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from magic_ai.text_encoder.tokenizer import (
    ALL_CUSTOM_TOKENS,
    BLANK_ANSWER_TOKENS,
    CHOOSE_KIND_TOKENS,
    INLINE_BLANK_TOKENS,
    MAX_NUM,
    NUM_TOKENS,
    TOKENIZER_DIR,
    load_tokenizer,
    num_token,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS_JSON = REPO_ROOT / "data" / "card_oracle_embeddings.json"


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR.exists():
        pytest.skip(
            f"tokenizer artifact missing at {TOKENIZER_DIR}; "
            "run scripts/build_text_encoder_vocab.py first"
        )
    return load_tokenizer()


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def test_every_custom_token_is_single_id(tokenizer) -> None:
    unk_id = tokenizer.unk_token_id
    for tok_str in ALL_CUSTOM_TOKENS:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        assert len(ids) == 1, f"{tok_str!r} tokenized to {len(ids)} ids: {ids}"
        assert ids[0] != unk_id, f"{tok_str!r} mapped to <unk>"


def test_inline_blank_tokens_are_distinct_singletons(tokenizer) -> None:
    # Every blank-kind / answer / num token resolves to a unique single id,
    # and the set is disjoint from the priority-anchor ``<pass>`` token
    # (which is reused, not redeclared).
    seen: dict[int, str] = {}
    for tok_str in INLINE_BLANK_TOKENS:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        assert len(ids) == 1, f"{tok_str!r} -> {ids}"
        tid = ids[0]
        assert tid not in seen, f"{tok_str!r} collides with {seen[tid]!r}"
        seen[tid] = tok_str
    assert len(CHOOSE_KIND_TOKENS) == 9
    assert len(BLANK_ANSWER_TOKENS) == 5
    assert len(NUM_TOKENS) == MAX_NUM == 16
    assert num_token(0) == "<num:0>"
    assert num_token(MAX_NUM - 1) == f"<num:{MAX_NUM - 1}>"
    with pytest.raises(ValueError):
        num_token(MAX_NUM)
    with pytest.raises(ValueError):
        num_token(-1)


def test_pass_token_not_duplicated_in_inline_blanks() -> None:
    # ``<pass>`` lives in ACTION_KIND_TOKENS as the priority-anchor; the
    # inline-blank plan reuses it rather than redeclaring it.
    assert "<pass>" not in INLINE_BLANK_TOKENS


@pytest.mark.parametrize("symbol", ["{W}", "{2/W}"])
def test_mana_symbol_single_id(tokenizer, symbol: str) -> None:
    ids = tokenizer.encode(symbol, add_special_tokens=False)
    assert len(ids) == 1, f"{symbol!r} -> {ids}"


def test_oracle_text_round_trips(tokenizer) -> None:
    if not CARDS_JSON.exists():
        pytest.skip(f"missing {CARDS_JSON}")
    payload: dict[str, Any] = json.loads(CARDS_JSON.read_text())
    raw_cards = payload["cards"]
    assert raw_cards, "no cards in card_oracle_embeddings.json"
    card_iter = raw_cards.values() if isinstance(raw_cards, dict) else raw_cards
    cards: list[dict[str, Any]] = [c for c in card_iter if isinstance(c, dict)]

    unk_id = tokenizer.unk_token_id

    for card in cards:
        name = card.get("name")
        for field in ("name", "oracle_text"):
            text = card.get(field) or ""
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert unk_id not in ids, f"<unk> appeared while encoding {field} of {name!r}: {text!r}"
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            assert _normalize_ws(decoded) == _normalize_ws(text), (
                f"round-trip mismatch for {field} of {name!r}:\n  in:  {text!r}\n  out: {decoded!r}"
            )
