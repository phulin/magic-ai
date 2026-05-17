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
    "<dict>",
    "</dict>",
    # Self-reference token: stands in for any occurrence of this card's
    # printed name (or any face name) inside its own oracle text. The card's
    # printed name is *not* emitted at the top of the body, so the encoder
    # has no string-matchable identity handle and must read the rules text.
    "<card-name>",
)

# ---------------------------------------------------------------------------
# Card-type / supertype tokens — one atomic id per word that may appear before
# the em-dash in a Scryfall ``type_line``. Subtypes (after the em-dash) are
# rendered as plain text so the encoder can read them as words.
# ---------------------------------------------------------------------------

# Canonical lowercase form -> token string. The full set of MTG supertypes and
# card types per the Comprehensive Rules; closed enough to be a one-time list.
_CARD_TYPE_WORDS: tuple[str, ...] = (
    # Supertypes
    "basic",
    "legendary",
    "ongoing",
    "snow",
    "world",
    "host",
    # Card types
    "artifact",
    "battle",
    "conspiracy",
    "creature",
    "dungeon",
    "enchantment",
    "instant",
    "kindred",
    "land",
    "phenomenon",
    "plane",
    "planeswalker",
    "scheme",
    "sorcery",
    "tribal",
    "vanguard",
)

CARD_TYPE_TOKENS: tuple[str, ...] = tuple(f"<{w}>" for w in _CARD_TYPE_WORDS)


def card_type_token(word: str) -> str:
    """Return the canonical ``<word>`` token for a type-line word.

    ``word`` is matched case-insensitively against the closed set of MTG
    supertypes and card types; raises ``KeyError`` for unknown words.
    """

    canon = word.strip().lower()
    if canon not in _CARD_TYPE_WORDS:
        raise KeyError(f"unknown card-type word {word!r}")
    return f"<{canon}>"


# ---------------------------------------------------------------------------
# Field-delimiter tokens — one paired open/close per body field.
# ---------------------------------------------------------------------------

CARD_FIELD_TOKENS: tuple[str, ...] = (
    "<subtypes>",
    "</subtypes>",
    "<mana-cost>",
    "</mana-cost>",
    "<rules-text>",
    "</rules-text>",
    "<pt>",
    "</pt>",
    "<loyalty>",
    "</loyalty>",
    "<face>",
    "</face>",
)

# ---------------------------------------------------------------------------
# Action-kind answer tokens used by inline blank legal vocabularies.
# The legacy ``<option>`` blocks are gone, but these ids remain stable because
# snapshots and checkpoints may still refer to the tokens.
# ---------------------------------------------------------------------------

ACTION_KIND_TOKENS: tuple[str, ...] = (
    "<cast>",
    "<play>",
    "<pass>",
    "<attack>",
    "<block>",
    "<mulligan>",
    "<keep>",
    "<activate>",
)

# ---------------------------------------------------------------------------
# Step / phase tokens — one atomic id per game-step value the engine emits.
# ---------------------------------------------------------------------------

# (Engine step name, canonical token suffix). The order matches
# ``magic_ai.text_encoder.token_tables.STEP_NAMES`` so that step ids on the
# Go side line up with these strings.
_STEP_NAME_TO_SUFFIX: tuple[tuple[str, str], ...] = (
    ("Untap", "untap"),
    ("Upkeep", "upkeep"),
    ("Draw", "draw"),
    ("Precombat Main", "precombat-main"),
    ("Begin Combat", "begin-combat"),
    ("Declare Attackers", "declare-attackers"),
    ("Declare Blockers", "declare-blockers"),
    ("Combat Damage", "combat-damage"),
    ("End Combat", "end-combat"),
    ("Postcombat Main", "postcombat-main"),
    ("End", "end"),
    ("Cleanup", "cleanup"),
)

STEP_TOKENS: tuple[str, ...] = tuple(f"<step:{suffix}>" for _name, suffix in _STEP_NAME_TO_SUFFIX)


def step_token(step_name: str) -> str:
    """Return the ``<step:...>`` token for an engine step name."""

    for name, suffix in _STEP_NAME_TO_SUFFIX:
        if name == step_name:
            return f"<step:{suffix}>"
    raise KeyError(f"unknown step name {step_name!r}")


# ---------------------------------------------------------------------------
# Top-level scalar wrappers + mana-pool tokens.
# ---------------------------------------------------------------------------

SCALAR_WRAPPER_TOKENS: tuple[str, ...] = (
    "<turn>",
    "</turn>",
    "<life>",
    "</life>",
    "<mana-pool>",
    "</mana-pool>",
)

# Pool mana symbols, distinct from the bracketed cost glyphs (``{G}`` etc.).
# Pool tokens denote "this color is currently floating in the player's pool";
# cost glyphs denote "this color appears in a cost description". Keeping the
# namespaces separate lets the encoder learn them as different roles.
POOL_MANA_TOKENS: tuple[str, ...] = (
    "<mana:W>",
    "<mana:U>",
    "<mana:B>",
    "<mana:R>",
    "<mana:G>",
    "<mana:C>",
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

MAX_CARD_REFS = 256


def card_ref_token(k: int) -> str:
    """Return the intra-snapshot card-ref token string for index ``k``."""

    if not 0 <= k < MAX_CARD_REFS:
        raise ValueError(f"card-ref index {k} out of range [0, {MAX_CARD_REFS})")
    return f"<card-ref:{k}>"


CARD_REF_TOKENS: tuple[str, ...] = tuple(card_ref_token(k) for k in range(MAX_CARD_REFS))

# ---------------------------------------------------------------------------
# Per-snapshot card-body dictionary references.
#
# The native packed-token assembler can emit each unique card body once inside
# a ``<dict>...</dict>`` block, prefixed by a sequence-local ``<dict-entry:D>``
# token. Each occurrence of a card in any zone is then a short
# ``<card-ref:K> <card> <dict-entry:D> ... </card>`` reference instead of a
# full body splice. ``<card-ref:K>`` is the per-permanent identity handle
# (range MAX_CARD_REFS); ``<dict-entry:D>`` is the per-sequence body handle
# (range MAX_DICT_ENTRIES). The two namespaces are deliberately disjoint so the
# encoder can learn distinct roles.
# ---------------------------------------------------------------------------

MAX_DICT_ENTRIES = 512


def dict_entry_token(r: int) -> str:
    """Return the per-snapshot dictionary-entry token string for cache row ``r``."""

    if not 0 <= r < MAX_DICT_ENTRIES:
        raise ValueError(f"dict-entry index {r} out of range [0, {MAX_DICT_ENTRIES})")
    return f"<dict-entry:{r}>"


DICT_ENTRY_TOKENS: tuple[str, ...] = tuple(dict_entry_token(r) for r in range(MAX_DICT_ENTRIES))

# ---------------------------------------------------------------------------
# Decision-spec tokens (see docs/decoder_grammar_plan.md).
#
# Encoder-side tokens for the spec section the renderer appends to state
# text. The decoder's grammar vocabulary (<DECLARE_ATTACKERS>, <ATTACK>,
# <BLOCK>, <END>, …) is *not* registered here — it lives only on the
# decoder side.
# ---------------------------------------------------------------------------

MAX_STACK_REFS = 16


def stack_ref_token(k: int) -> str:
    if not 0 <= k < MAX_STACK_REFS:
        raise ValueError(f"stack-ref index {k} out of range [0, {MAX_STACK_REFS})")
    return f"<stack-ref:{k}>"


SPEC_STRUCTURAL_TOKENS: tuple[str, ...] = (
    "<spec-open>",
    "<spec-close>",
    "<decision-type>",
    "<legal-attacker>",
    "<legal-blocker>",
    "<legal-target>",
    "<legal-action>",
    "<for-action>",
    "<max-value>",
    "</max-value>",
    "<player-ref:0>",
    "<player-ref:1>",
)

# Decision-type-name tokens. Prefixed with `<dt-` to disambiguate from the
# pre-existing `<choose-mode>` / `<choose-may>` inline-blank kind tokens.
DECISION_TYPE_NAME_TOKENS: tuple[str, ...] = (
    "<dt-priority>",
    "<dt-declare-attackers>",
    "<dt-declare-blockers>",
    "<dt-choose-targets>",
    "<dt-may>",
    "<dt-choose-mode>",
    "<dt-choose-x>",
)

STACK_REF_TOKENS: tuple[str, ...] = tuple(stack_ref_token(k) for k in range(MAX_STACK_REFS))

DECISION_SPEC_TOKENS: tuple[str, ...] = (
    SPEC_STRUCTURAL_TOKENS + DECISION_TYPE_NAME_TOKENS + STACK_REF_TOKENS
)

# Union of every custom token, ordered (structural, status, mana, loyalty,
# card-ref, dict-entry). The build script feeds this whole tuple to
# ``add_special_tokens`` so each entry is guaranteed a single id and is
# never BPE-split.
ALL_CUSTOM_TOKENS: tuple[str, ...] = (
    STRUCTURAL_TOKENS
    + STATUS_TOKENS
    + MANA_TOKENS
    + LOYALTY_TOKENS
    + CARD_REF_TOKENS
    + DICT_ENTRY_TOKENS
    + CARD_TYPE_TOKENS
    + CARD_FIELD_TOKENS
    + ACTION_KIND_TOKENS
    + STEP_TOKENS
    + SCALAR_WRAPPER_TOKENS
    + POOL_MANA_TOKENS
    + DECISION_SPEC_TOKENS
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
