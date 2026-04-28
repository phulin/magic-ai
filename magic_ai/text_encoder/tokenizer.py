"""Tokenizer for the text-based state/action encoder.

This module owns the canonical custom-token vocabulary that augments the
ModernBERT BPE tokenizer (see ``docs/text_encoder_plan.md`` §2). The set of
strings defined here is the single source of truth — both the build script
(``scripts/build_text_encoder_vocab.py``) and the renderer (PR #2) import
from this module so token strings cannot drift via typo.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Upstream pin — keep tokenizer reproducible across HF revisions.
# ---------------------------------------------------------------------------

MODERNBERT_REPO = "answerdotai/ModernBERT-base"
# `main` of answerdotai/ModernBERT-base as of 2026-04-28. Pinned so a future
# upstream update cannot silently shift base-vocab ids out from under a
# trained checkpoint.
MODERNBERT_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"

# Where the augmented tokenizer is persisted via ``save_pretrained``.
TOKENIZER_DIR = Path(__file__).resolve().parents[2] / "data" / "text_encoder_tokenizer"

# ---------------------------------------------------------------------------
# Custom token strings.
# ---------------------------------------------------------------------------

# Structural delimiters: snapshot framing, player scope, zones, card / action
# wrappers, and the field separator inside a card record.
STRUCTURAL_TOKENS: tuple[str, ...] = (
    "<bos>",
    "<eos>",
    "<pad>",
    "<state>",
    "</state>",
    "<self>",
    "</self>",
    "<opp>",
    "</opp>",
    "<hand>",
    "</hand>",
    "<battlefield>",
    "</battlefield>",
    "<graveyard>",
    "</graveyard>",
    "<exile>",
    "</exile>",
    "<library>",
    "</library>",
    "<stack>",
    "</stack>",
    "<command>",
    "</command>",
    "<card>",
    "</card>",
    "<actions>",
    "</actions>",
    "<option>",
    "</option>",
    "<target>",
    "</target>",
    "<sep>",
)

# Status flags — bare tokens emitted inside a ``<card>`` block.
STATUS_TOKENS: tuple[str, ...] = (
    "<tapped>",
    "<untapped>",
    "<sick>",
    "<attacking>",
    "<blocking>",
    "<monstrous>",
    "<flipped>",
    "<facedown>",
    "<+1+1>",
    "<-1-1>",
    "<counter>",
    "<attached-to>",
)

# ---------------------------------------------------------------------------
# Mana / cost symbols — one token per Scryfall-canonical bracketed symbol.
# ---------------------------------------------------------------------------


def _mana_tokens() -> tuple[str, ...]:
    out: list[str] = []
    # Single-color, colorless, snow, X, and generic 0..20.
    for sym in ("W", "U", "B", "R", "G", "C", "S", "X"):
        out.append(f"{{{sym}}}")
    for n in range(0, 21):
        out.append(f"{{{n}}}")
    # Hybrid (two-color), Phyrexian, 2-hybrid.
    hybrid_pairs = [
        ("W", "U"),
        ("U", "B"),
        ("B", "R"),
        ("R", "G"),
        ("G", "W"),
        ("W", "B"),
        ("U", "R"),
        ("B", "G"),
        ("R", "W"),
        ("G", "U"),
    ]
    for a, b in hybrid_pairs:
        out.append(f"{{{a}/{b}}}")
    for c in ("W", "U", "B", "R", "G"):
        out.append(f"{{{c}/P}}")
    for c in ("W", "U", "B", "R", "G"):
        out.append(f"{{2/{c}}}")
    # Tap, untap, energy.
    out.extend(("{T}", "{Q}", "{E}"))
    return tuple(out)


MANA_TOKENS: tuple[str, ...] = _mana_tokens()

# ---------------------------------------------------------------------------
# Loyalty: ``[+N]`` and ``[-N]`` for N = 0..MAX_LOYALTY.
# ``[+0]``/``[-0]`` collapse to ``[0]`` to match plain-text planeswalker conv.
# ---------------------------------------------------------------------------

MAX_LOYALTY = 20


def _loyalty_tokens() -> tuple[str, ...]:
    out: list[str] = ["[0]"]
    for n in range(1, MAX_LOYALTY + 1):
        out.append(f"[+{n}]")
        out.append(f"[-{n}]")
    return tuple(out)


LOYALTY_TOKENS: tuple[str, ...] = _loyalty_tokens()

# ---------------------------------------------------------------------------
# Intra-snapshot card references.
# ---------------------------------------------------------------------------

MAX_CARD_REFS = 64


def card_ref_token(k: int) -> str:
    """Return the intra-snapshot card-ref token string for index ``k``."""

    if not 0 <= k < MAX_CARD_REFS:
        raise ValueError(f"card-ref index {k} out of range [0, {MAX_CARD_REFS})")
    return f"<card-ref:{k}>"


CARD_REF_TOKENS: tuple[str, ...] = tuple(card_ref_token(k) for k in range(MAX_CARD_REFS))

# Union of every custom token, ordered (structural, status, mana, loyalty,
# card-ref). The build script feeds this whole tuple to ``add_special_tokens``
# so each entry is guaranteed a single id and is never BPE-split.
ALL_CUSTOM_TOKENS: tuple[str, ...] = (
    STRUCTURAL_TOKENS + STATUS_TOKENS + MANA_TOKENS + LOYALTY_TOKENS + CARD_REF_TOKENS
)


# ---------------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------------


def load_tokenizer(
    tokenizer_dir: str | Path = TOKENIZER_DIR,
) -> PreTrainedTokenizerFast:
    """Load the persisted augmented ModernBERT tokenizer.

    Run ``scripts/build_text_encoder_vocab.py`` first to materialize the
    artifact at ``tokenizer_dir`` (default: ``data/text_encoder_tokenizer``).
    """

    path = Path(tokenizer_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer dir {path} does not exist; run "
            "scripts/build_text_encoder_vocab.py to build it."
        )
    tok = AutoTokenizer.from_pretrained(str(path))
    if not isinstance(tok, PreTrainedTokenizerFast):
        raise TypeError(f"Expected PreTrainedTokenizerFast, got {type(tok).__name__}")
    return tok
