"""Precomputed token-id tables covering every string the assembler emits.

Phase 1 of the option-(c) cutover (see chat history): the assembler's runtime
string vocabulary is closed, so we can replace every ``tokenizer.encode``
call with a table lookup. Once all emissions go through this module, the
assembler stops needing the live tokenizer at runtime, which lets us port
the dispatch loop to native code that just does table lookups + memcpy.

This module is the **single source of truth** for which (kind, scalar)
pairs the assembler is allowed to emit. Phase 4 (Go-side native assembler)
serializes this same table over the FFI; the parity test in
``tests/test_token_tables.py`` guarantees the table matches what the live
HF tokenizer would produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.card_cache import CardTokenCache
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS

# ---------------------------------------------------------------------------
# Closed vocabulary constants (mirrored from assembler.py / render_plan.py).
# ---------------------------------------------------------------------------

CARD_CLOSER_TEXT = " </card>"
STATUS_TAPPED_TEXT = " <sep> <tapped>"
STATUS_UNTAPPED_TEXT = " <sep> <untapped>"

STEP_NAMES: tuple[str, ...] = (
    "Untap",
    "Upkeep",
    "Draw",
    "Precombat Main",
    "Begin Combat",
    "Declare Attackers",
    "Declare Blockers",
    "Combat Damage",
    "End Combat",
    "Postcombat Main",
    "End",
    "Cleanup",
    "Unknown",
)

MANA_SYMBOLS: tuple[str, ...] = ("W", "U", "B", "R", "G", "C")

ZONE_TAGS_BY_ID: dict[int, str] = {
    0: "hand",
    1: "battlefield",
    2: "graveyard",
    3: "exile",
    4: "library",
    5: "stack",
    6: "command",
}

ACTION_VERBS_BY_ID: dict[int, str] = {
    0: "pass",
    1: "play",
    2: "cast",
    3: "activate",
    4: "attack with",
    5: "block with",
    6: "choice",
}

OWNER_NAMES: tuple[str, ...] = ("self", "opp")

# Bounded scalar ranges. Out-of-range inputs raise — the assembler is
# expected to surface the bound mismatch upstream rather than fall back to
# a live encode (that would defeat the whole point of the table).
TURN_MIN = 0
TURN_MAX = 200
LIFE_MIN = -30
LIFE_MAX = 300
COUNT_MIN = 0
COUNT_MAX = 200
ABILITY_MIN = 0
ABILITY_MAX = 16


# ---------------------------------------------------------------------------
# Symbolic keys for static structural fragments.
# ---------------------------------------------------------------------------


class Frag(IntEnum):
    """Static structural fragments emitted by the assembler.

    Integer values are stable across runs (used as the FFI table index in
    Phase 3). Append-only; never reorder.
    """

    BOS_STATE = 0
    CLOSE_STATE_EOS = 1
    CLOSE_SELF = 2  # " </self>"
    CLOSE_OPP = 3  # " </opp>"
    CLOSE_OPTION = 4  # " </option>"
    OPEN_ACTIONS = 5  # "<actions>"
    CLOSE_ACTIONS = 6  # "</actions>"
    OPEN_TARGET = 7  # " <target>"
    SPACE = 8  # " "
    TARGET_FALLBACK = 9  # "target"
    SELF_MANA = 10  # "<self> mana="
    OPP_MANA = 11  # "<opp> mana="


_FRAG_TEXT: dict[Frag, str] = {
    Frag.BOS_STATE: "<bos><state>",
    Frag.CLOSE_STATE_EOS: "</state><eos>",
    Frag.CLOSE_SELF: " </self>",
    Frag.CLOSE_OPP: " </opp>",
    Frag.CLOSE_OPTION: " </option>",
    Frag.OPEN_ACTIONS: "<actions>",
    Frag.CLOSE_ACTIONS: "</actions>",
    Frag.OPEN_TARGET: " <target>",
    Frag.SPACE: " ",
    Frag.TARGET_FALLBACK: "target",
    Frag.SELF_MANA: "<self> mana=",
    Frag.OPP_MANA: "<opp> mana=",
}


def fragment_text(frag: Frag) -> str:
    """Return the ground-truth string a Frag is supposed to encode to."""
    return _FRAG_TEXT[frag]


# ---------------------------------------------------------------------------
# Table layout.
# ---------------------------------------------------------------------------


@dataclass
class TokenTables:
    """All precomputed token-id sequences the assembler can emit at runtime.

    Indexing conventions
    --------------------
    ``structural[Frag]`` -> token-id list for the given symbolic fragment.
    ``zone_open[(zone_id, owner_id)]``, ``zone_close[(zone_id, owner_id)]``.
    ``action_verb[kind_id]`` (no leading space — caller emits SPACE first).
    ``mana_glyph[color_id]`` -> tokens for ``"{C}"``.
    ``turn_step[(turn, step_id)]`` -> tokens for ``f" turn={turn} step={step} "``.
    ``life_owner[(life, owner_id)]`` -> tokens for ``f"<self> life={life} mana="`` (or ``<opp>``).
    ``ability[N]`` -> tokens for ``f" ability {N}"``.
    ``count[N]`` -> tokens for ``f" count={N}"``  (reserved; not currently emitted).
    ``card_body[row]`` -> body tokens with trailing CARD_CLOSER stripped.
    ``card_name[row]`` -> tokens for the row's display name.
    ``card_ref[K]`` -> single-id list for ``<card-ref:K>``.
    """

    # Single-id specials.
    pad_id: int
    option_id: int
    target_open_id: int
    target_close_id: int
    tapped_id: int
    untapped_id: int

    # Multi-token sequences.
    card_closer: list[int]
    status_tapped: list[int]
    status_untapped: list[int]

    structural: dict[Frag, list[int]] = field(default_factory=dict)
    zone_open: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    zone_close: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    action_verb: dict[int, list[int]] = field(default_factory=dict)
    mana_glyph: list[list[int]] = field(default_factory=list)
    turn_step: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    life_owner: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    ability: dict[int, list[int]] = field(default_factory=dict)
    count: dict[int, list[int]] = field(default_factory=dict)
    card_ref: list[int] = field(default_factory=list)
    card_body: list[list[int]] = field(default_factory=list)
    card_name: list[list[int]] = field(default_factory=list)

    # Bounds (echoed for downstream consumers / FFI).
    turn_min: int = TURN_MIN
    turn_max: int = TURN_MAX
    life_min: int = LIFE_MIN
    life_max: int = LIFE_MAX
    count_min: int = COUNT_MIN
    count_max: int = COUNT_MAX
    ability_min: int = ABILITY_MIN
    ability_max: int = ABILITY_MAX


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _encode(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]


def _single(tokenizer: PreTrainedTokenizerFast, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if isinstance(tid, list):
        raise TypeError(f"convert_tokens_to_ids({token!r}) returned a list")
    return int(tid)


def _strip_card_closer(body: list[int], closer: list[int]) -> list[int]:
    if closer and len(body) >= len(closer) and body[-len(closer) :] == closer:
        return body[: -len(closer)]
    return body


def build_token_tables(
    tokenizer: PreTrainedTokenizerFast,
    cache: CardTokenCache,
) -> TokenTables:
    """Precompute every token-id sequence the assembler may emit at runtime.

    ``cache`` provides the per-card body tokens and display names; everything
    else is derived from the tokenizer plus the closed vocabulary defined
    above.
    """

    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    card_closer = _encode(tokenizer, CARD_CLOSER_TEXT)
    status_tapped = _encode(tokenizer, STATUS_TAPPED_TEXT)
    status_untapped = _encode(tokenizer, STATUS_UNTAPPED_TEXT)

    tables = TokenTables(
        pad_id=pad_id,
        option_id=_single(tokenizer, "<option>"),
        target_open_id=_single(tokenizer, "<target>"),
        target_close_id=_single(tokenizer, "</target>"),
        tapped_id=_single(tokenizer, "<tapped>"),
        untapped_id=_single(tokenizer, "<untapped>"),
        card_closer=card_closer,
        status_tapped=status_tapped,
        status_untapped=status_untapped,
    )

    # Static structural fragments.
    for frag, text in _FRAG_TEXT.items():
        tables.structural[frag] = _encode(tokenizer, text)

    # Zone open/close pairs by (zone_id, owner_id).
    for zone_id, tag in ZONE_TAGS_BY_ID.items():
        for owner_id, owner in enumerate(OWNER_NAMES):
            tables.zone_open[(zone_id, owner_id)] = _encode(tokenizer, f"<{owner}><{tag}>")
            tables.zone_close[(zone_id, owner_id)] = _encode(tokenizer, f"</{tag}></{owner}>")

    # Action verbs (no leading space; assembler emits SPACE first).
    for kind_id, verb in ACTION_VERBS_BY_ID.items():
        tables.action_verb[kind_id] = _encode(tokenizer, verb)

    # Mana glyphs per color id.
    tables.mana_glyph = [_encode(tokenizer, f"{{{sym}}}") for sym in MANA_SYMBOLS]

    # turn × step (single fragment so BPE merges across the boundary stay
    # exact). Keyed by (turn, step_id).
    for turn in range(TURN_MIN, TURN_MAX + 1):
        for step_id, step in enumerate(STEP_NAMES):
            tables.turn_step[(turn, step_id)] = _encode(tokenizer, f" turn={turn} step={step} ")

    # life × owner. The assembler emits "<self>" / "<opp>" + " life=N mana="
    # as a single string, so the table folds them together.
    for life in range(LIFE_MIN, LIFE_MAX + 1):
        for owner_id, owner in enumerate(OWNER_NAMES):
            tables.life_owner[(life, owner_id)] = _encode(tokenizer, f"<{owner}> life={life} mana=")

    for n in range(ABILITY_MIN, ABILITY_MAX + 1):
        tables.ability[n] = _encode(tokenizer, f" ability {n}")

    for n in range(COUNT_MIN, COUNT_MAX + 1):
        tables.count[n] = _encode(tokenizer, f" count={n}")

    # card-ref single-id table.
    tables.card_ref = [_single(tokenizer, f"<card-ref:{k}>") for k in range(MAX_CARD_REFS)]

    # Per-row card body (trailing " </card>" stripped) and display-name tokens.
    num_rows = len(cache.row_to_name)
    body_tokens_buf = cache.token_buffer
    body_offsets = cache.offsets
    tables.card_body = []
    tables.card_name = []
    for row in range(num_rows):
        start = int(body_offsets[row])
        end = int(body_offsets[row + 1])
        body = body_tokens_buf[start:end].tolist()
        tables.card_body.append(_strip_card_closer(body, card_closer))
        tables.card_name.append(_encode(tokenizer, cache.row_to_name[row]))

    return tables


__all__ = [
    "ABILITY_MAX",
    "ABILITY_MIN",
    "ACTION_VERBS_BY_ID",
    "CARD_CLOSER_TEXT",
    "COUNT_MAX",
    "COUNT_MIN",
    "Frag",
    "LIFE_MAX",
    "LIFE_MIN",
    "MANA_SYMBOLS",
    "OWNER_NAMES",
    "STATUS_TAPPED_TEXT",
    "STATUS_UNTAPPED_TEXT",
    "STEP_NAMES",
    "TURN_MAX",
    "TURN_MIN",
    "TokenTables",
    "ZONE_TAGS_BY_ID",
    "build_token_tables",
    "fragment_text",
]
