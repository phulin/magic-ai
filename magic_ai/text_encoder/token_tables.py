"""Precomputed token-id tables covering native text assembly emissions.

The runtime string vocabulary is closed, so native assembly can use table
lookups instead of calling ``tokenizer.encode`` while stepping environments.

This module is the **single source of truth** for which (kind, scalar)
pairs the assembler is allowed to emit. Phase 4 (Go-side native assembler)
serializes this same table over the FFI; the parity test in
``tests/test_text_token_tables.py`` guarantees the table matches what the live
HF tokenizer would produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.card_cache import CardTokenCache
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS, MAX_NUM

# ---------------------------------------------------------------------------
# Closed vocabulary constants mirrored by mage-go.
# ---------------------------------------------------------------------------

CARD_CLOSER_TEXT = "</card>"
STATUS_TAPPED_TEXT = "<tapped>"
STATUS_UNTAPPED_TEXT = "<untapped>"

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
    # CLOSE_SELF / CLOSE_OPP also close the ``<mana-pool>`` opened by
    # ``life_owner`` so the Go assembler produces matched tags.
    Frag.CLOSE_SELF: "</mana-pool></self>",
    Frag.CLOSE_OPP: "</mana-pool></opp>",
    Frag.CLOSE_OPTION: "</option>",
    Frag.OPEN_ACTIONS: "<actions>",
    Frag.CLOSE_ACTIONS: "</actions>",
    Frag.OPEN_TARGET: "<target>",
    Frag.SPACE: " ",
    Frag.TARGET_FALLBACK: "target",
    # SELF_MANA / OPP_MANA: emitted when an opMana opcode arrives without a
    # preceding opLife (i.e. mana-only block). Open ``<self><mana-pool>`` so
    # the closer ``</mana-pool></self>`` matches.
    Frag.SELF_MANA: "<self><mana-pool>",
    Frag.OPP_MANA: "<opp><mana-pool>",
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
    ``action_verb[kind_id]`` -> tokens for ``f" {verb}"`` (leading space folded in).
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

    # v2 card-body dedup specials. ``card_open_id`` is ``<card>``, used when
    # an occurrence references a dict entry. Default zero so existing call
    # sites that don't fill these still construct the dataclass.
    dict_open_id: int = 0
    dict_close_id: int = 0
    card_open_id: int = 0

    # Per-player + shared-zone single-id specials (Go-parity additions).
    # ``self_id`` / ``opp_id`` are emitted inside ``<target>`` blocks when
    # an option targets a player. ``stack_*`` / ``command_*`` open and close
    # the shared (non-per-player) stack and command zones once per snapshot.
    self_id: int = 0
    opp_id: int = 0
    stack_open_id: int = 0
    stack_close_id: int = 0
    command_open_id: int = 0
    command_close_id: int = 0

    # Inline-blank specials (Step 1 of text_encoder_inline_blanks_plan.md).
    # Single-id ``<choose-*>`` blank-kind markers, ``<chosen>`` scoring
    # token, ``<yes>``/``<no>``/``<none>`` / ``<x-end>`` answer tokens, and
    # the ``<num:k>`` small-integer answer vocab. ``<pass>`` is reused from
    # the priority-anchor vocab and is not redeclared here. Default-init to
    # 0 so existing call sites that don't fill these still construct.
    choose_target_id: int = 0
    choose_block_id: int = 0
    choose_damage_order_id: int = 0
    choose_mode_id: int = 0
    choose_may_id: int = 0
    choose_x_digit_id: int = 0
    choose_mana_source_id: int = 0
    choose_play_id: int = 0
    use_ability_id: int = 0
    chosen_id: int = 0
    yes_id: int = 0
    no_id: int = 0
    none_id: int = 0
    x_end_id: int = 0
    mulligan_id: int = 0
    keep_id: int = 0
    num_ids: list[int] = field(default_factory=list)

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
    # v2 card-body dedup: dict_entry[row] -> single-id ``<dict-entry:row>``.
    # Aligned with ``card_body`` (one entry per cache row).
    dict_entry: list[int] = field(default_factory=list)

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
    cache: CardTokenCache | None = None,
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
        dict_open_id=_single(tokenizer, "<dict>"),
        dict_close_id=_single(tokenizer, "</dict>"),
        card_open_id=_single(tokenizer, "<card>"),
        self_id=_single(tokenizer, "<self>"),
        opp_id=_single(tokenizer, "<opp>"),
        stack_open_id=_single(tokenizer, "<stack>"),
        stack_close_id=_single(tokenizer, "</stack>"),
        command_open_id=_single(tokenizer, "<command>"),
        command_close_id=_single(tokenizer, "</command>"),
        choose_target_id=_single(tokenizer, "<choose-target>"),
        choose_block_id=_single(tokenizer, "<choose-block>"),
        choose_damage_order_id=_single(tokenizer, "<choose-damage-order>"),
        choose_mode_id=_single(tokenizer, "<choose-mode>"),
        choose_may_id=_single(tokenizer, "<choose-may>"),
        choose_x_digit_id=_single(tokenizer, "<choose-x-digit>"),
        choose_mana_source_id=_single(tokenizer, "<choose-mana-source>"),
        choose_play_id=_single(tokenizer, "<choose-play>"),
        use_ability_id=_single(tokenizer, "<use-ability>"),
        chosen_id=_single(tokenizer, "<chosen>"),
        yes_id=_single(tokenizer, "<yes>"),
        no_id=_single(tokenizer, "<no>"),
        none_id=_single(tokenizer, "<none>"),
        x_end_id=_single(tokenizer, "<x-end>"),
        mulligan_id=_single(tokenizer, "<mulligan>"),
        keep_id=_single(tokenizer, "<keep>"),
        num_ids=[_single(tokenizer, f"<num:{k}>") for k in range(MAX_NUM)],
    )

    # Static structural fragments.
    for frag, text in _FRAG_TEXT.items():
        tables.structural[frag] = _encode(tokenizer, text)

    # Zone open/close pairs by (zone_id, owner_id).
    for zone_id, tag in ZONE_TAGS_BY_ID.items():
        for owner_id, owner in enumerate(OWNER_NAMES):
            tables.zone_open[(zone_id, owner_id)] = _encode(tokenizer, f"<{owner}><{tag}>")
            tables.zone_close[(zone_id, owner_id)] = _encode(tokenizer, f"</{tag}></{owner}>")

    # Action verbs — one atomic kind token per kind id (the leading space the
    # old encoding folded in is gone since the kind is now a single special
    # token immediately following ``<option>``).
    _ACTION_KIND_TOKEN_BY_ID: dict[int, str] = {
        0: "<pass>",
        1: "<play>",
        2: "<cast>",
        3: "<activate>",
        4: "<attack>",
        5: "<block>",
        6: "<choice>",  # not in vocab — falls back to literal text
    }
    for kind_id, _verb in ACTION_VERBS_BY_ID.items():
        tok_str = _ACTION_KIND_TOKEN_BY_ID.get(kind_id, "")
        tables.action_verb[kind_id] = _encode(tokenizer, tok_str) if tok_str else []

    # Pool mana — one ``<mana:X>`` token per color (used by the opMana opcode,
    # which emits floating mana inside ``<mana-pool>...</mana-pool>``). The
    # cost-glyph counterpart ``{X}`` is used only inside ``<mana-cost>`` and
    # ``<rules-text>`` and lives in its own namespace.
    _POOL_MANA_BY_SYMBOL: dict[str, str] = {
        "W": "<mana:W>",
        "U": "<mana:U>",
        "B": "<mana:B>",
        "R": "<mana:R>",
        "G": "<mana:G>",
        "C": "<mana:C>",
    }
    tables.mana_glyph = [_encode(tokenizer, _POOL_MANA_BY_SYMBOL[sym]) for sym in MANA_SYMBOLS]

    # turn × step → ``<turn>{turn}</turn><step:...>``. Step ids past the named
    # set (e.g. STEP_NAMES "Unknown") drop the step token.
    _STEP_NAME_TO_TOKEN: dict[str, str] = {
        "Untap": "<step:untap>",
        "Upkeep": "<step:upkeep>",
        "Draw": "<step:draw>",
        "Precombat Main": "<step:precombat-main>",
        "Begin Combat": "<step:begin-combat>",
        "Declare Attackers": "<step:declare-attackers>",
        "Declare Blockers": "<step:declare-blockers>",
        "Combat Damage": "<step:combat-damage>",
        "End Combat": "<step:end-combat>",
        "Postcombat Main": "<step:postcombat-main>",
        "End": "<step:end>",
        "Cleanup": "<step:cleanup>",
    }
    for turn in range(TURN_MIN, TURN_MAX + 1):
        for step_id, step in enumerate(STEP_NAMES):
            step_tok = _STEP_NAME_TO_TOKEN.get(step, "")
            tables.turn_step[(turn, step_id)] = _encode(tokenizer, f"<turn>{turn}</turn>{step_tok}")

    # life × owner → ``<{owner}><life>{life}</life><mana-pool>``. The matching
    # ``</mana-pool></{owner}>`` is emitted by Frag.CLOSE_SELF / CLOSE_OPP.
    for life in range(LIFE_MIN, LIFE_MAX + 1):
        for owner_id, owner in enumerate(OWNER_NAMES):
            tables.life_owner[(life, owner_id)] = _encode(
                tokenizer, f"<{owner}><life>{life}</life><mana-pool>"
            )

    for n in range(ABILITY_MIN, ABILITY_MAX + 1):
        tables.ability[n] = _encode(tokenizer, f" ability {n}")

    # count[N] → just the bare integer; library blocks emit ``<library>{N}</library>``
    # with the integer slotted in by the assembler.
    for n in range(COUNT_MIN, COUNT_MAX + 1):
        tables.count[n] = _encode(tokenizer, str(n))

    # card-ref single-id table.
    tables.card_ref = [_single(tokenizer, f"<card-ref:{k}>") for k in range(MAX_CARD_REFS)]

    # Per-row card body (trailing " </card>" stripped) and display-name tokens.
    if cache is not None:
        num_rows = len(cache.row_to_name)
        body_tokens_buf = cache.token_buffer
        body_offsets = cache.offsets
        tables.card_body = []
        tables.card_name = []
        tables.dict_entry = []
        for row in range(num_rows):
            start = int(body_offsets[row])
            end = int(body_offsets[row + 1])
            body = body_tokens_buf[start:end].tolist()
            tables.card_body.append(_strip_card_closer(body, card_closer))
            tables.card_name.append(_encode(tokenizer, cache.row_to_name[row]))
            tables.dict_entry.append(_single(tokenizer, f"<dict-entry:{row}>"))

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
