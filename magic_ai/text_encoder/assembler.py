"""Render-plan -> ``TextEncodedBatch`` assembler.

PR 13-C from ``docs/text_encoder_plan.md`` §13. Walks the int32 render-plan
stream produced by either the Python parity emitter or the structured Go
emitter, memcpys card-body token slices from the :class:`CardTokenCache`,
and writes the result into a preallocated ``[B, max_tokens]`` token buffer
alongside ``card_ref_positions`` and inline-blank metadata.

The Python parity route carries pre-tokenized literal slices. The structured
Go route tokenizes small scalar/tag fragments in Python until the Go side
emits equivalent literal-token slices or the assembler grows precomputed
numeric/string tables for every scalar.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.card_cache import CardTokenCache
from magic_ai.text_encoder.render_plan import (
    OP_ATTACHED_TO,
    OP_CLOSE_DICT,
    OP_CLOSE_PLAYER,
    OP_CLOSE_RAW_CARD,
    OP_CLOSE_STATE,
    OP_CLOSE_ZONE,
    OP_COMMAND_CLOSE,
    OP_COMMAND_OPEN,
    OP_COUNT,
    OP_COUNTER,
    OP_DICT_ENTRY,
    OP_EMIT_BLANK,
    OP_EMIT_BLANK_LEGAL,
    OP_END_CARD,
    OP_LIFE,
    OP_LITERAL_TOKENS,
    OP_MANA,
    OP_OPEN_DICT,
    OP_OPEN_PLAYER,
    OP_OPEN_RAW_CARD,
    OP_OPEN_STATE,
    OP_OPEN_ZONE,
    OP_PLACE_CARD,
    OP_PLACE_CARD_REF,
    OP_STACK_CLOSE,
    OP_STACK_OPEN,
    OP_TURN,
    OPCODE_ARITY,
    STATUS_TAPPED,
    STATUS_TAPPED_KNOWN,
    ZONE_BATTLEFIELD,
    ZONE_COMMAND,
    ZONE_EXILE,
    ZONE_GRAVEYARD,
    ZONE_HAND,
    ZONE_LIBRARY,
    ZONE_STACK,
)
from magic_ai.text_encoder.token_tables import (
    CARD_CLOSER_TEXT as _TT_CARD_CLOSER_TEXT,
)
from magic_ai.text_encoder.token_tables import (
    Frag,
    TokenTables,
    build_token_tables,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS, MAX_DICT_ENTRIES

logger = logging.getLogger(__name__)

CARD_CLOSER_TEXT = _TT_CARD_CLOSER_TEXT

_STEP_NAMES: tuple[str, ...] = (
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
_MANA_SYMBOLS: tuple[str, ...] = ("W", "U", "B", "R", "G", "C")
_ZONE_TAGS: dict[int, str] = {
    ZONE_HAND: "hand",
    ZONE_BATTLEFIELD: "battlefield",
    ZONE_GRAVEYARD: "graveyard",
    ZONE_EXILE: "exile",
    ZONE_LIBRARY: "library",
    ZONE_STACK: "stack",
    ZONE_COMMAND: "command",
}


@dataclass
class AssemblerTokens:
    """Pre-resolved structural-token ids the assembler emits at runtime.

    Looked up once at init time from the tokenizer's added-tokens map.
    """

    pad_id: int
    tapped_id: int
    untapped_id: int
    card_ref_ids: list[int]  # length MAX_CARD_REFS
    card_closer_ids: list[int]  # tokens of " </card>"
    # v2 card-body-dedup token ids. ``<dict>`` / ``</dict>`` are
    # structural tokens; ``<dict-entry:R>`` is one id per cache row in the
    # MAX_DICT_ENTRIES namespace. ``card_open_id`` is the ``<card>``
    # structural token id, used by the per-occurrence ref form
    # ``<card-ref:K> <card> <dict-entry:R> [status] </card>``.
    dict_open_id: int = 0
    dict_close_id: int = 0
    dict_entry_ids: list[int] = field(default_factory=list)
    card_open_id: int = 0
    stack_open_id: int = 0
    stack_close_id: int = 0
    command_open_id: int = 0
    command_close_id: int = 0
    # ref_id -> K reverse map for the literal-tokens walker, which would
    # otherwise scan card_ref_ids per token.
    card_ref_id_to_k: dict[int, int] = field(default_factory=dict)
    _status_tapped: list[int] = field(default_factory=list)
    _status_untapped: list[int] = field(default_factory=list)
    # Memoized per-fragment token-id lists. Populated at init for all known
    # static fragments and small bounded vocabularies; dynamic-but-low-arity
    # strings (turn=N, life=N) fill in lazily on first encounter.
    fragment_ids: dict[str, list[int]] = field(default_factory=dict)
    # Cached mana glyph per color id (e.g. "{W}" -> [tok_ids...]).
    mana_glyph_ids: list[list[int]] = field(default_factory=list)
    _tokenizer: PreTrainedTokenizerFast | None = None
    # Per-(cache id) memo of card-body Python lists with the trailing
    # ``" </card>"`` already stripped. Built once per CardTokenCache instance
    # the first time the assembler sees it, then reused across all subsequent
    # snapshots — replaces the per-card .tolist()/tail-compare/.tolist() in
    # the OP_PLACE_CARD hot path.
    _body_lists_cache: dict[int, list[list[int]]] = field(default_factory=dict)
    # Per-(cache id) memo of card-name display-string token lists, mirroring
    # ``_body_lists_cache``. Built lazily from the tokenizer the first time the
    # assembler sees a given cache.
    _name_lists_cache: dict[int, list[list[int]]] = field(default_factory=dict)
    # Closed-vocabulary token tables (Phase 1 of the assembler-port). Populated
    # at build time from the tokenizer. The walker dispatches every emit-text
    # call through these tables, so the live tokenizer never fires on the hot
    # path. ``tables.card_body`` / ``tables.card_name`` are *not* populated
    # here — those live in the per-cache ``_body_lists_cache`` /
    # ``_name_lists_cache`` memos so a single AssemblerTokens can be reused
    # across caches.
    tables: TokenTables | None = None


def _single_id(tokenizer: PreTrainedTokenizerFast, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if isinstance(tid, list):
        raise TypeError(f"convert_tokens_to_ids({token!r}) returned a list")
    return int(tid)


def _encode(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]


def build_assembler_tokens(tokenizer: PreTrainedTokenizerFast) -> AssemblerTokens:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    toks = AssemblerTokens(
        pad_id=int(pad),
        tapped_id=_single_id(tokenizer, "<tapped>"),
        untapped_id=_single_id(tokenizer, "<untapped>"),
        card_ref_ids=[_single_id(tokenizer, f"<card-ref:{k}>") for k in range(MAX_CARD_REFS)],
        card_closer_ids=list(tokenizer.encode(CARD_CLOSER_TEXT, add_special_tokens=False)),
        dict_open_id=_single_id(tokenizer, "<dict>"),
        dict_close_id=_single_id(tokenizer, "</dict>"),
        dict_entry_ids=[
            _single_id(tokenizer, f"<dict-entry:{r}>") for r in range(MAX_DICT_ENTRIES)
        ],
        card_open_id=_single_id(tokenizer, "<card>"),
        stack_open_id=_single_id(tokenizer, "<stack>"),
        stack_close_id=_single_id(tokenizer, "</stack>"),
        command_open_id=_single_id(tokenizer, "<command>"),
        command_close_id=_single_id(tokenizer, "</command>"),
    )
    toks.card_ref_id_to_k = {ref_id: k for k, ref_id in enumerate(toks.card_ref_ids)}
    toks._tokenizer = tokenizer
    # Build the closed-vocabulary token table. ``cache=None`` skips the
    # per-row card_body / card_name fields; those live in the per-cache memos
    # below. Anything tokenizer-only is now precomputed here.
    toks.tables = build_token_tables(tokenizer, cache=None)
    frags = toks.fragment_ids
    # Static structural fragments.
    for s in (
        "<bos><state>",
        "</state><eos>",
        " </self>",
        " </opp>",
        " ",
        "<self> mana=",
        "<opp> mana=",
    ):
        frags[s] = _encode(tokenizer, s)
    # Step-name fragments: " turn=" handled lazily; step strings as a unit
    # within the f" turn={turn} step={step} " template are dynamic on `turn`,
    # so we cache the f" step={step} " portion and emit `turn=N` lazily.
    for step in _STEP_NAMES:
        frags[f" step={step} "] = _encode(tokenizer, f" step={step} ")
    # Owner+zone open/close tags (2 × 7 combos).
    for tag in _ZONE_TAGS.values():
        for owner_tag in ("self", "opp"):
            open_s = f"<{owner_tag}><{tag}>"
            close_s = f"</{tag}></{owner_tag}>"
            frags[open_s] = _encode(tokenizer, open_s)
            frags[close_s] = _encode(tokenizer, close_s)
    # Mana glyph per color id.
    toks.mana_glyph_ids = [_encode(tokenizer, f"{{{sym}}}") for sym in _MANA_SYMBOLS]
    return toks


def _body_lists(toks: AssemblerTokens, cache: CardTokenCache) -> list[list[int]]:
    """Return per-row tail-stripped token-id lists for ``cache``.

    Built once per cache and memoized on ``toks`` keyed by ``id(cache)``. The
    trailing ``" </card>"`` token sequence is removed up front so the assembler
    never re-checks it inside the per-card hot path.
    """
    key = id(cache)
    cached = toks._body_lists_cache.get(key)
    if cached is not None:
        return cached
    tail = toks.card_closer_ids
    tail_len = len(tail)
    offsets = cache.offsets
    buf = cache.token_buffer
    rows: list[list[int]] = []
    num_rows = len(cache.row_to_name)
    for row in range(num_rows):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        body = buf[start:end].tolist()
        if tail_len and len(body) >= tail_len and body[-tail_len:] == tail:
            del body[-tail_len:]
        rows.append(body)
    toks._body_lists_cache[key] = rows
    return rows


def _name_lists(toks: AssemblerTokens, cache: CardTokenCache) -> list[list[int]]:
    """Return per-row tokenized display-name lists for ``cache``.

    Built once per cache and memoized on ``toks`` keyed by ``id(cache)``. The
    assembler emits ``cache.row_to_name[row]`` as a single fragment via
    ``emit_text``; precomputing the BPE for each row lets the hot path do a
    single ``out.extend(name_lists[row])`` instead of a tokenizer call.
    """
    key = id(cache)
    cached = toks._name_lists_cache.get(key)
    if cached is not None:
        return cached
    tok = toks._tokenizer
    if tok is None:
        raise RuntimeError("AssemblerTokens has no tokenizer attached")
    rows: list[list[int]] = [_encode(tok, name) for name in cache.row_to_name]
    toks._name_lists_cache[key] = rows
    return rows


def _fragment(toks: AssemblerTokens, text: str) -> list[int]:
    """Return the cached token-id list for ``text``, encoding lazily on miss."""
    cached = toks.fragment_ids.get(text)
    if cached is not None:
        return cached
    tok = toks._tokenizer
    if tok is None:
        raise RuntimeError("AssemblerTokens has no tokenizer attached")
    encoded = _encode(tok, text)
    toks.fragment_ids[text] = encoded
    return encoded


# ---------------------------------------------------------------------------
# Single-plan assembler
# ---------------------------------------------------------------------------


@dataclass
class _AssembledExample:
    token_ids: list[int]
    card_ref_positions: dict[int, int]
    # Inline-blank metadata (Step 3 of inline-blanks plan). Populated when the
    # render plan contains OP_EMIT_BLANK opcodes; otherwise these stay empty.
    blank_positions: list[int] = field(default_factory=list)
    blank_kind_ids: list[int] = field(default_factory=list)
    blank_group_ids: list[int] = field(default_factory=list)
    blank_group_kinds: list[int] = field(default_factory=list)
    blank_legal_ids: list[list[int]] = field(default_factory=list)


def _assemble_one(
    plan: torch.Tensor,
    cache: CardTokenCache,
    toks: AssemblerTokens,
    body_lists: list[list[int]],
    name_lists: list[list[int]],
) -> _AssembledExample:
    # Materialize the int32 plan as a Python list once so the inner
    # walker does plain int indexing instead of paying ``.item()`` cost
    # on each tensor element. ``plan`` is a small CPU int32 tensor.
    plan_list: list[int] = plan.tolist() if isinstance(plan, torch.Tensor) else list(plan)
    plan_len = len(plan_list)
    ref_id_to_k = toks.card_ref_id_to_k
    if toks.tables is None:
        raise RuntimeError("AssemblerTokens.tables not populated")
    tables = toks.tables
    frag_table = tables.structural
    zone_open_table = tables.zone_open
    zone_close_table = tables.zone_close
    turn_step_table = tables.turn_step
    life_owner_table = tables.life_owner
    out: list[int] = []
    card_ref_positions: dict[int, int] = {}

    scalar_owner_open: int | None = None
    blank_positions: list[int] = []
    blank_kind_ids: list[int] = []
    blank_group_ids: list[int] = []
    blank_group_kinds: list[int] = []
    blank_legal: list[list[int]] = []
    # (zone_id, owner) pushed on OPEN_ZONE / popped on CLOSE_ZONE so we can
    # recover the matching open in O(1) instead of rescanning the plan.
    zone_stack: list[tuple[int, int]] = []
    # Detect literal-tokens mode by walking the opcode stream — naive
    # ``np.any(plan == OP_LITERAL_TOKENS)`` scans payload ints too and
    # spuriously flips when any payload happens to equal the opcode id
    # (slot indices, card-row ids, and mana amounts routinely do).
    structured_plan = True
    _scan_i = 0
    while _scan_i < plan_len:
        _scan_op = plan_list[_scan_i]
        if _scan_op == OP_LITERAL_TOKENS:
            structured_plan = False
            break
        _scan_arity = OPCODE_ARITY.get(_scan_op)
        if _scan_arity is None:
            # Unknown opcode — let the main loop raise with a real error.
            break
        _scan_i += 1 + _scan_arity

    def emit_card_ref(uuid_idx: int) -> bool:
        # Returns True iff a ``<card-ref:K>`` token was emitted. Caller may
        # fall back to a row-name on False. ``uuid_idx >= MAX_CARD_REFS``
        # happens in cluttered states where Go assigned a per-snapshot ref
        # past the tokenizer's 64-slot cap; treat it like a missing ref.
        if uuid_idx < 0 or uuid_idx >= len(toks.card_ref_ids):
            return False
        ref_id = toks.card_ref_ids[uuid_idx]
        pos = len(out)
        out.append(ref_id)
        if uuid_idx not in card_ref_positions:
            card_ref_positions[uuid_idx] = pos
        return True

    def close_scalar_owner() -> None:
        nonlocal scalar_owner_open
        if scalar_owner_open is None:
            return
        out.extend(frag_table[Frag.CLOSE_SELF if scalar_owner_open == 0 else Frag.CLOSE_OPP])
        scalar_owner_open = None

    i = 0
    n = plan_len
    while i < n:
        op = plan_list[i]
        arity = OPCODE_ARITY.get(op)
        if arity is None:
            raise ValueError(f"unknown opcode {op} at position {i}")

        if op == OP_LITERAL_TOKENS:
            close_scalar_owner()
            length = plan_list[i + 1]
            slice_start = i + 2
            slice_end = slice_start + length
            for j in range(slice_start, slice_end):
                tid = plan_list[j]
                pos = len(out)
                out.append(tid)
                # card-ref ids: record first-occurrence position per K.
                # O(1) reverse-map lookup; the linear scan over
                # MAX_CARD_REFS=256 here used to dominate literal-heavy
                # plans.
                k = ref_id_to_k.get(tid)
                if k is not None and k not in card_ref_positions:
                    card_ref_positions[k] = pos
            i = slice_end
            continue

        if structured_plan:
            if scalar_owner_open is not None:
                keep_open_for_mana = op == OP_MANA and plan_list[i + 1] == scalar_owner_open
                if not keep_open_for_mana:
                    close_scalar_owner()

            if op == OP_OPEN_STATE:
                out.extend(frag_table[Frag.BOS_STATE])
                i += 1
                continue

            if op == OP_CLOSE_STATE:
                close_scalar_owner()
                out.extend(frag_table[Frag.CLOSE_STATE_EOS])
                i += 1
                continue

            if op == OP_TURN:
                turn = plan_list[i + 1]
                step_id = plan_list[i + 2]
                if not (0 <= step_id < len(_STEP_NAMES)):
                    step_id = len(_STEP_NAMES) - 1  # "Unknown"
                ts = turn_step_table.get((turn, step_id))
                if ts is None:
                    raise ValueError(
                        f"OP_TURN out of bounds: turn={turn} step_id={step_id} "
                        f"(allowed turn range {tables.turn_min}..{tables.turn_max})"
                    )
                out.extend(ts)
                i += 1 + arity
                continue

            if op == OP_LIFE:
                close_scalar_owner()
                owner = plan_list[i + 1]
                life = plan_list[i + 2]
                lo = life_owner_table.get((life, owner))
                if lo is None:
                    raise ValueError(
                        f"OP_LIFE out of bounds: life={life} owner={owner} "
                        f"(allowed life range {tables.life_min}..{tables.life_max})"
                    )
                out.extend(lo)
                scalar_owner_open = owner
                i += 1 + arity
                continue

            if op == OP_MANA:
                owner = plan_list[i + 1]
                color_id = plan_list[i + 2]
                amount = plan_list[i + 3]
                if scalar_owner_open is None:
                    out.extend(frag_table[Frag.SELF_MANA if owner == 0 else Frag.OPP_MANA])
                    scalar_owner_open = owner
                if 0 <= color_id < len(toks.mana_glyph_ids) and amount > 0:
                    glyph_ids = toks.mana_glyph_ids[color_id]
                    for _ in range(amount):
                        out.extend(glyph_ids)
                i += 1 + arity
                continue

            if op == OP_OPEN_ZONE:
                zone = plan_list[i + 1]
                owner = plan_list[i + 2]
                zone_stack.append((zone, owner))
                out.extend(zone_open_table[(zone, owner)])
                i += 1 + arity
                continue

            if op == OP_CLOSE_ZONE:
                if zone_stack:
                    zone, owner = zone_stack.pop()
                    out.extend(zone_close_table[(zone, owner)])
                i += 1
                continue

        if op == OP_COUNT:
            n_val = plan_list[i + 1]
            count_table = toks.tables.count if toks.tables is not None else None
            span = count_table.get(n_val) if count_table is not None else None
            if span is None:
                raise ValueError(f"OP_COUNT out of bounds: amount={n_val}")
            out.extend(span)
            i += 1 + arity
            continue

        if op == OP_STACK_OPEN:
            out.append(toks.stack_open_id)
            i += 1
            continue

        if op == OP_STACK_CLOSE:
            out.append(toks.stack_close_id)
            i += 1
            continue

        if op == OP_COMMAND_OPEN:
            out.append(toks.command_open_id)
            i += 1
            continue

        if op == OP_COMMAND_CLOSE:
            out.append(toks.command_close_id)
            i += 1
            continue

        if op == OP_OPEN_DICT:
            out.append(toks.dict_open_id)
            i += 1
            continue

        if op == OP_CLOSE_DICT:
            out.append(toks.dict_close_id)
            i += 1
            continue

        if op == OP_DICT_ENTRY:
            row = plan_list[i + 1]
            if 0 <= row < len(toks.dict_entry_ids):
                out.append(toks.dict_entry_ids[row])
            if 0 <= row < len(body_lists):
                out.extend(body_lists[row])
            # Bodies have ``" </card>"`` stripped at memo time; the dict
            # entry stands on its own (no status flag inside the dict), so
            # we reattach the closer immediately.
            out.extend(toks.card_closer_ids)
            i += 1 + arity
            continue

        if op == OP_PLACE_CARD_REF:
            _slot_idx = plan_list[i + 1]
            row = plan_list[i + 2]
            status = plan_list[i + 3]
            uuid_idx = plan_list[i + 4]
            # Per-occurrence reference form:
            #   <card-ref:K> <card> <dict-entry:R> [<sep> <tapped>] </card>
            emit_card_ref(uuid_idx)
            out.append(toks.card_open_id)
            if 0 <= row < len(toks.dict_entry_ids):
                out.append(toks.dict_entry_ids[row])
            if status & STATUS_TAPPED_KNOWN:
                if status & STATUS_TAPPED:
                    out.extend(_status_prefix_tapped(toks))
                else:
                    out.extend(_status_prefix_untapped(toks))
            elif status & STATUS_TAPPED:
                out.extend(_status_prefix_tapped(toks))
            out.extend(toks.card_closer_ids)
            i += 1 + arity
            continue

        if op == OP_PLACE_CARD:
            _slot_idx = plan_list[i + 1]
            row = plan_list[i + 2]
            status = plan_list[i + 3]
            uuid_idx = plan_list[i + 4]
            # Emit <card-ref:K> if attached.
            emit_card_ref(uuid_idx)
            # Memcpy the card body — the trailing ``" </card>"`` was already
            # stripped when ``body_lists`` was built. The END_CARD opcode (or
            # the status decoder) will replace it.
            if 0 <= row < len(body_lists):
                out.extend(body_lists[row])
            # Status flags — replicate the renderer's behavior:
            # "<sep> <tapped>" or "<sep> <untapped>" appended before " </card>".
            # The cache's body already had " </card>" stripped above, so we
            # write the sep+flag here. Note the renderer prepends a space and
            # `<sep>` token only when a tapped flag exists. We model that
            # exactly by tokenizing " <sep> " once at init... but we don't
            # have <sep> as an id stored. Simpler: store the literal-token
            # equivalent of " <sep> <tapped>" / " <sep> <untapped>" at init.
            # (See AssemblerTokens — but we cache the encoded sequence here
            # via the tokenizer's `<sep>` id and a leading whitespace token.)
            # For PR 13-C, the renderer's exact whitespace " <sep> <tapped>"
            # has been stable enough to encode at init time; we delegate to
            # the precomputed lists held on `toks`.
            if status & STATUS_TAPPED_KNOWN:
                if status & STATUS_TAPPED:
                    out.extend(_status_prefix_tapped(toks))
                else:
                    out.extend(_status_prefix_untapped(toks))
            elif structured_plan and status & STATUS_TAPPED:
                out.extend(_status_prefix_tapped(toks))
            if structured_plan:
                out.extend(toks.card_closer_ids)
            i += 1 + arity
            continue

        if op == OP_EMIT_BLANK:
            kind_id = plan_list[i + 1]
            group_id = plan_list[i + 2]
            group_kind = plan_list[i + 3]
            legal_count = plan_list[i + 4]
            pos = len(out)
            out.append(int(kind_id))
            blank_positions.append(pos)
            blank_kind_ids.append(int(kind_id))
            blank_group_ids.append(int(group_id))
            blank_group_kinds.append(int(group_kind))
            legal: list[int] = []
            i += 1 + arity
            for _ in range(legal_count):
                if i >= n or plan_list[i] != OP_EMIT_BLANK_LEGAL:
                    raise ValueError(
                        f"OP_EMIT_BLANK legal_count={legal_count} but "
                        f"OP_EMIT_BLANK_LEGAL stream truncated at i={i}"
                    )
                legal.append(int(plan_list[i + 1]))
                i += 1 + OPCODE_ARITY[OP_EMIT_BLANK_LEGAL]
            blank_legal.append(legal)
            continue

        if op == OP_EMIT_BLANK_LEGAL:
            # Stray OP_EMIT_BLANK_LEGAL (without a preceding OP_EMIT_BLANK):
            # treat as a wire-format error.
            raise ValueError(
                f"unexpected OP_EMIT_BLANK_LEGAL at position {i}; "
                "must follow an OP_EMIT_BLANK header"
            )

        if op == OP_END_CARD:
            out.extend(toks.card_closer_ids)
            i += 1
            continue

        if op == OP_OPEN_RAW_CARD:
            uuid_idx = plan_list[i + 1]
            emit_card_ref(uuid_idx)
            i += 1 + arity
            continue

        # Bookkeeping-only opcodes: skip past header + payload, no tokens emitted.
        if op in (
            OP_OPEN_STATE,
            OP_CLOSE_STATE,
            OP_OPEN_ZONE,
            OP_CLOSE_ZONE,
            OP_OPEN_PLAYER,
            OP_CLOSE_PLAYER,
            OP_COUNTER,
            OP_ATTACHED_TO,
            OP_TURN,
            OP_LIFE,
            OP_MANA,
            OP_CLOSE_RAW_CARD,
        ):
            i += 1 + arity
            continue

        raise ValueError(f"unhandled opcode {op} at position {i}")

    return _AssembledExample(
        token_ids=out,
        card_ref_positions=card_ref_positions,
        blank_positions=blank_positions,
        blank_kind_ids=blank_kind_ids,
        blank_group_ids=blank_group_ids,
        blank_group_kinds=blank_group_kinds,
        blank_legal_ids=blank_legal,
    )


# ---------------------------------------------------------------------------
# Status-flag prelude tokens — computed lazily from the tokenizer the
# AssemblerTokens was built from. We stash them on the dataclass at init.
# ---------------------------------------------------------------------------


def _status_prefix_tapped(toks: AssemblerTokens) -> list[int]:
    return toks._status_tapped


def _status_prefix_untapped(toks: AssemblerTokens) -> list[int]:
    return toks._status_untapped


def _attach_status_prefixes(toks: AssemblerTokens, tokenizer: PreTrainedTokenizerFast) -> None:
    # Idempotent: status-prefix token-ids are static for a given tokenizer,
    # so once they've been attached we can skip re-encoding on every call.
    if toks._status_tapped and toks._status_untapped:
        return
    toks._status_tapped = list(tokenizer.encode("<tapped>", add_special_tokens=False))
    toks._status_untapped = list(tokenizer.encode("<untapped>", add_special_tokens=False))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def assemble_batch(
    plans: Sequence[torch.Tensor],
    cache: CardTokenCache,
    tokenizer: PreTrainedTokenizerFast,
    *,
    max_tokens: int,
    on_overflow: str = "raise",
    assembler_tokens: AssemblerTokens | None = None,
) -> TextEncodedBatch:
    """Assemble a batch of render plans into a :class:`TextEncodedBatch`.

    Parameters
    ----------
    plans:
        Sequence of 1-D int32 CPU render-plan tensors, one per env.
    cache:
        Pre-tokenized card-body cache.
    tokenizer:
        Used for structural-token lookup and, for structured Go plans,
        scalar/tag fragment encoding.
    max_tokens:
        Per-example token-buffer width. Plans whose decoded length exceeds
        this raise (``on_overflow="raise"``) or are truncated
        (``on_overflow="truncate"``).
    """

    if len(plans) == 0:
        raise ValueError("assemble_batch() requires at least one plan")
    if on_overflow not in ("raise", "truncate"):
        raise ValueError(f"on_overflow must be 'raise' or 'truncate', got {on_overflow!r}")

    toks = assembler_tokens or build_assembler_tokens(tokenizer)
    _attach_status_prefixes(toks, tokenizer)
    body_lists = _body_lists(toks, cache)

    name_lists = _name_lists(toks, cache)
    assembled: list[_AssembledExample] = [
        _assemble_one(p, cache, toks, body_lists, name_lists) for p in plans
    ]

    seq_lengths = [len(ex.token_ids) for ex in assembled]
    actual_max = max(seq_lengths)
    if actual_max > max_tokens:
        if on_overflow == "raise":
            raise ValueError(f"assembled token length {actual_max} exceeds max_tokens={max_tokens}")
        # Truncate card-ref and blank positions that fall outside the retained
        # prefix. card_ref_positions is keyed by K so it can stay a filtered
        # dict; blank tensor shapes preserve the original blank ordinal space.
        truncated_blanks = 0
        truncated_examples = 0
        for ex in assembled:
            if len(ex.token_ids) > max_tokens:
                ex.token_ids = ex.token_ids[:max_tokens]
                ex.card_ref_positions = {
                    k: p for k, p in ex.card_ref_positions.items() if p < max_tokens
                }
                ex.blank_positions = [pos if pos < max_tokens else -1 for pos in ex.blank_positions]
                truncated_blanks += sum(1 for pos in ex.blank_positions if pos < 0)
                truncated_examples += 1
        seq_lengths = [len(ex.token_ids) for ex in assembled]
        logger.warning(
            "assemble_batch: truncated %d/%d example(s) to max_tokens=%d "
            "(masked %d blank(s) past the budget)",
            truncated_examples,
            len(assembled),
            max_tokens,
            truncated_blanks,
        )

    batch_size = len(assembled)
    width = max(seq_lengths)
    max_blanks = max((len(ex.blank_positions) for ex in assembled), default=0)
    max_legal = max(
        (len(legal) for ex in assembled for legal in ex.blank_legal_ids),
        default=0,
    )

    token_ids = torch.full((batch_size, width), toks.pad_id, dtype=torch.int64)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.int64)
    card_ref_positions = torch.full((batch_size, MAX_CARD_REFS), -1, dtype=torch.int64)
    blank_positions = torch.full((batch_size, max_blanks), -1, dtype=torch.int32)
    blank_kind = torch.zeros((batch_size, max_blanks), dtype=torch.int32)
    blank_group = torch.full((batch_size, max_blanks), -1, dtype=torch.int32)
    blank_group_kind = torch.zeros((batch_size, max_blanks), dtype=torch.int32)
    blank_legal_ids = torch.zeros((batch_size, max_blanks, max_legal), dtype=torch.int32)
    blank_legal_mask = torch.zeros((batch_size, max_blanks, max_legal), dtype=torch.bool)

    for b, ex in enumerate(assembled):
        t_i = len(ex.token_ids)
        if t_i:
            token_ids[b, :t_i] = torch.as_tensor(ex.token_ids, dtype=torch.int64)
            attention_mask[b, :t_i] = 1
        for k, pos in ex.card_ref_positions.items():
            if 0 <= k < MAX_CARD_REFS:
                card_ref_positions[b, k] = int(pos)
        for k_idx, pos in enumerate(ex.blank_positions):
            blank_positions[b, k_idx] = int(pos)
            blank_kind[b, k_idx] = int(ex.blank_kind_ids[k_idx])
            blank_group[b, k_idx] = int(ex.blank_group_ids[k_idx])
            blank_group_kind[b, k_idx] = int(ex.blank_group_kinds[k_idx])
            for v_idx, tid in enumerate(ex.blank_legal_ids[k_idx]):
                blank_legal_ids[b, k_idx, v_idx] = int(tid)
                blank_legal_mask[b, k_idx, v_idx] = True

    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=torch.as_tensor(seq_lengths, dtype=torch.int64),
        blank_positions=blank_positions,
        blank_kind=blank_kind,
        blank_group=blank_group,
        blank_group_kind=blank_group_kind,
        blank_legal_ids=blank_legal_ids,
        blank_legal_mask=blank_legal_mask,
    )
