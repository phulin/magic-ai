"""Render-plan -> ``TextEncodedBatch`` assembler.

PR 13-C from ``docs/text_encoder_plan.md`` §13. Walks the int32 render-plan
stream produced by either the Python parity emitter or the structured Go
emitter, memcpys card-body token slices from the :class:`CardTokenCache`,
and writes the result into a preallocated ``[B, max_tokens]`` token buffer
alongside ``card_ref_positions`` / ``option_positions`` / ``target_positions``.

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
    OP_CLOSE_ACTIONS,
    OP_CLOSE_PLAYER,
    OP_CLOSE_RAW_CARD,
    OP_CLOSE_STATE,
    OP_CLOSE_ZONE,
    OP_COUNTER,
    OP_END_CARD,
    OP_LIFE,
    OP_LITERAL_TOKENS,
    OP_MANA,
    OP_OPEN_ACTIONS,
    OP_OPEN_PLAYER,
    OP_OPEN_RAW_CARD,
    OP_OPEN_STATE,
    OP_OPEN_ZONE,
    OP_OPTION,
    OP_PLACE_CARD,
    OP_TARGET,
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
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS

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
_ACTION_KINDS: dict[int, str] = {
    0: "pass",
    1: "play",
    2: "cast",
    3: "activate",
    4: "attack with",
    5: "block with",
    6: "choice",
}


@dataclass
class AssemblerTokens:
    """Pre-resolved structural-token ids the assembler emits at runtime.

    Looked up once at init time from the tokenizer's added-tokens map.
    """

    pad_id: int
    option_id: int
    target_open_id: int
    target_close_id: int
    tapped_id: int
    untapped_id: int
    card_ref_ids: list[int]  # length MAX_CARD_REFS
    card_closer_ids: list[int]  # tokens of " </card>"
    # ref_id -> K reverse map for the literal-tokens walker, which would
    # otherwise scan card_ref_ids per token.
    card_ref_id_to_k: dict[int, int] = field(default_factory=dict)
    _status_tapped: list[int] = field(default_factory=list)
    _status_untapped: list[int] = field(default_factory=list)
    # Memoized per-fragment token-id lists. Populated at init for all known
    # static fragments and small bounded vocabularies; dynamic-but-low-arity
    # strings (turn=N, life=N, ability N) fill in lazily on first encounter.
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
        option_id=_single_id(tokenizer, "<option>"),
        target_open_id=_single_id(tokenizer, "<target>"),
        target_close_id=_single_id(tokenizer, "</target>"),
        tapped_id=_single_id(tokenizer, "<tapped>"),
        untapped_id=_single_id(tokenizer, "<untapped>"),
        card_ref_ids=[_single_id(tokenizer, f"<card-ref:{k}>") for k in range(MAX_CARD_REFS)],
        card_closer_ids=list(tokenizer.encode(CARD_CLOSER_TEXT, add_special_tokens=False)),
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
        " </option>",
        "<actions>",
        "</actions>",
        " <target>",
        " ",
        "target",
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
    # Action verbs (with leading space, as emitted).
    for verb in _ACTION_KINDS.values():
        frags[f" {verb}"] = _encode(tokenizer, f" {verb}")
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
    option_positions: list[int]
    target_positions: list[list[int]]


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
    action_verb_table = tables.action_verb
    turn_step_table = tables.turn_step
    life_owner_table = tables.life_owner
    ability_table = tables.ability
    out: list[int] = []
    card_ref_positions: dict[int, int] = {}
    option_positions: list[int] = []
    target_positions: list[list[int]] = []

    cur_target_bucket: list[int] | None = None
    scalar_owner_open: int | None = None
    option_open = False
    # (zone_id, owner) pushed on OPEN_ZONE / popped on CLOSE_ZONE so we can
    # recover the matching open in O(1) instead of rescanning the plan.
    zone_stack: list[tuple[int, int]] = []
    # Detect literal-tokens mode by walking the opcode stream — naive
    # ``np.any(plan == OP_LITERAL_TOKENS)`` scans payload ints too and
    # spuriously flips when any payload happens to equal the opcode id
    # (slot indices, card-row ids, and mana amounts routinely do). When
    # the flag mis-fires, OP_OPTION / OP_TARGET / OP_OPEN_ACTIONS fall
    # through to the bookkeeping fallback and the encoded batch ends up
    # with empty option/target positions, which then crashes downstream
    # gather (or returns CUDA-garbage NaN at sample time).
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

    def close_option() -> None:
        nonlocal option_open
        if option_open:
            out.extend(frag_table[Frag.CLOSE_OPTION])
            option_open = False

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
                # Anchor recovery while walking literal slices.
                if tid == toks.option_id:
                    option_positions.append(pos)
                    cur_target_bucket = []
                    target_positions.append(cur_target_bucket)
                elif tid == toks.target_open_id and cur_target_bucket is not None:
                    cur_target_bucket.append(pos)
                else:
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
                close_option()
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
                close_option()
                zone = plan_list[i + 1]
                owner = plan_list[i + 2]
                zone_stack.append((zone, owner))
                out.extend(zone_open_table[(zone, owner)])
                i += 1 + arity
                continue

            if op == OP_CLOSE_ZONE:
                close_option()
                if zone_stack:
                    zone, owner = zone_stack.pop()
                    out.extend(zone_close_table[(zone, owner)])
                i += 1
                continue

            if op == OP_OPEN_ACTIONS:
                out.extend(frag_table[Frag.OPEN_ACTIONS])
                i += 1
                continue

            if op == OP_CLOSE_ACTIONS:
                close_option()
                out.extend(frag_table[Frag.CLOSE_ACTIONS])
                i += 1
                continue

            if op == OP_OPTION:
                close_option()
                kind_id = plan_list[i + 1]
                source_row = plan_list[i + 2]
                source_uuid_idx = plan_list[i + 3]
                _mana_cost_id = plan_list[i + 4]
                ability_idx = plan_list[i + 5]
                pos = len(out)
                out.append(toks.option_id)
                option_positions.append(pos)
                cur_target_bucket = []
                target_positions.append(cur_target_bucket)
                option_open = True
                # ``action_verb_table`` already folds in the leading space, so
                # the dispatch is one ``out.extend`` per action verb. An
                # unknown ``kind_id`` falls through to a one-shot encode of
                # the literal " unknown" string — not reachable on a well-
                # formed plan but kept for legacy parity.
                verb_ids_opt = action_verb_table.get(kind_id)
                kind_known = verb_ids_opt is not None
                verb_ids = verb_ids_opt if verb_ids_opt is not None else _fragment(toks, " unknown")
                out.extend(verb_ids)
                # ``kind_id`` 0 (pass) and 6 (choice) skip the source-row /
                # ability suffix, matching the legacy "verb in (pass, choice,
                # unknown)" guard.
                if kind_known and kind_id not in (0, 6):
                    out.extend(frag_table[Frag.SPACE])
                    if not emit_card_ref(source_uuid_idx) and (0 <= source_row < len(name_lists)):
                        out.extend(name_lists[source_row])
                if ability_idx >= 0 and kind_id == 3:
                    ab = ability_table.get(ability_idx)
                    if ab is None:
                        raise ValueError(
                            f"OP_OPTION out of bounds: ability_idx={ability_idx} "
                            f"(allowed range {tables.ability_min}..{tables.ability_max})"
                        )
                    out.extend(ab)
                i += 1 + arity
                continue

            if op == OP_TARGET:
                target_row = plan_list[i + 1]
                target_uuid_idx = plan_list[i + 2]
                pos = len(out)
                out.append(toks.target_open_id)
                if cur_target_bucket is not None:
                    cur_target_bucket.append(pos)
                out.extend(frag_table[Frag.SPACE])
                if not emit_card_ref(target_uuid_idx):
                    if 0 <= target_row < len(name_lists):
                        out.extend(name_lists[target_row])
                    else:
                        out.extend(frag_table[Frag.TARGET_FALLBACK])
                out.extend(frag_table[Frag.SPACE])
                out.append(toks.target_close_id)
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
            OP_OPEN_ACTIONS,
            OP_CLOSE_ACTIONS,
            OP_OPEN_PLAYER,
            OP_CLOSE_PLAYER,
            OP_COUNTER,
            OP_ATTACHED_TO,
            OP_OPTION,
            OP_TARGET,
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
        option_positions=option_positions,
        target_positions=target_positions,
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
    toks._status_tapped = list(tokenizer.encode(" <sep> <tapped>", add_special_tokens=False))
    toks._status_untapped = list(tokenizer.encode(" <sep> <untapped>", add_special_tokens=False))


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
        # Truncate but preserve the option/target index space. Setting
        # truncated positions to -1 (instead of dropping the entries) keeps
        # the K↔K invariant the upstream layout relies on: a layout col that
        # references "the K'th option" still finds slot K — option_mask just
        # ends up False there, so the model can't score it but the indexing
        # contract holds. card_ref_positions is keyed by K so it can stay a
        # filtered dict.
        truncated_options = 0
        truncated_targets = 0
        truncated_examples = 0
        for ex in assembled:
            if len(ex.token_ids) > max_tokens:
                ex.token_ids = ex.token_ids[:max_tokens]
                ex.card_ref_positions = {
                    k: p for k, p in ex.card_ref_positions.items() if p < max_tokens
                }
                new_options: list[int] = []
                new_targets: list[list[int]] = []
                ex_truncated_opts = 0
                ex_truncated_tgts = 0
                for opt_pos, tp in zip(ex.option_positions, ex.target_positions, strict=True):
                    if opt_pos < max_tokens:
                        new_options.append(opt_pos)
                    else:
                        new_options.append(-1)
                        ex_truncated_opts += 1
                    new_targets.append([tpos if tpos < max_tokens else -1 for tpos in tp])
                    ex_truncated_tgts += sum(1 for tpos in tp if tpos >= max_tokens)
                ex.option_positions = new_options
                ex.target_positions = new_targets
                truncated_options += ex_truncated_opts
                truncated_targets += ex_truncated_tgts
                truncated_examples += 1
        seq_lengths = [len(ex.token_ids) for ex in assembled]
        logger.warning(
            "assemble_batch: truncated %d/%d example(s) to max_tokens=%d "
            "(masked %d option(s) and %d target(s) past the budget)",
            truncated_examples,
            len(assembled),
            max_tokens,
            truncated_options,
            truncated_targets,
        )

    batch_size = len(assembled)
    width = max(seq_lengths)
    max_opts = max((len(ex.option_positions) for ex in assembled), default=0)
    max_targets = max(
        (len(t) for ex in assembled for t in ex.target_positions),
        default=0,
    )

    token_ids = torch.full((batch_size, width), toks.pad_id, dtype=torch.int64)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.int64)
    card_ref_positions = torch.full((batch_size, MAX_CARD_REFS), -1, dtype=torch.int64)
    option_positions = torch.full((batch_size, max_opts), -1, dtype=torch.int64)
    option_mask = torch.zeros((batch_size, max_opts), dtype=torch.bool)
    target_positions = torch.full((batch_size, max_opts, max_targets), -1, dtype=torch.int64)
    target_mask = torch.zeros((batch_size, max_opts, max_targets), dtype=torch.bool)

    for b, ex in enumerate(assembled):
        t_i = len(ex.token_ids)
        if t_i:
            token_ids[b, :t_i] = torch.as_tensor(ex.token_ids, dtype=torch.int64)
            attention_mask[b, :t_i] = 1
        for k, pos in ex.card_ref_positions.items():
            if 0 <= k < MAX_CARD_REFS:
                card_ref_positions[b, k] = int(pos)
        for o, pos in enumerate(ex.option_positions):
            if pos >= 0:
                option_positions[b, o] = int(pos)
                option_mask[b, o] = True
            for t, tpos in enumerate(ex.target_positions[o]):
                if tpos >= 0:
                    target_positions[b, o, t] = int(tpos)
                    target_mask[b, o, t] = True

    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        option_positions=option_positions,
        option_mask=option_mask,
        target_positions=target_positions,
        target_mask=target_mask,
        seq_lengths=torch.as_tensor(seq_lengths, dtype=torch.int64),
    )
