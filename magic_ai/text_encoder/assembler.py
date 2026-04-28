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

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
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
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS

CARD_CLOSER_TEXT = " </card>"

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
    toks._tokenizer = tokenizer
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
    plan: np.ndarray,
    cache: CardTokenCache,
    toks: AssemblerTokens,
    body_lists: list[list[int]],
) -> _AssembledExample:
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
    _scan_n = int(plan.shape[0])
    while _scan_i < _scan_n:
        _scan_op = int(plan[_scan_i])
        if _scan_op == OP_LITERAL_TOKENS:
            structured_plan = False
            break
        _scan_arity = OPCODE_ARITY.get(_scan_op)
        if _scan_arity is None:
            # Unknown opcode — let the main loop raise with a real error.
            break
        _scan_i += 1 + _scan_arity

    def emit_text(text: str) -> None:
        out.extend(_fragment(toks, text))

    def emit_card_ref(uuid_idx: int) -> None:
        if uuid_idx < 0:
            return
        ref_id = toks.card_ref_ids[uuid_idx]
        pos = len(out)
        out.append(ref_id)
        if uuid_idx not in card_ref_positions:
            card_ref_positions[uuid_idx] = pos

    def close_scalar_owner() -> None:
        nonlocal scalar_owner_open
        if scalar_owner_open is None:
            return
        emit_text(" </self>" if scalar_owner_open == 0 else " </opp>")
        scalar_owner_open = None

    def close_option() -> None:
        nonlocal option_open
        if option_open:
            emit_text(" </option>")
            option_open = False

    i = 0
    n = int(plan.shape[0])
    while i < n:
        op = int(plan[i])
        arity = OPCODE_ARITY.get(op)
        if arity is None:
            raise ValueError(f"unknown opcode {op} at position {i}")

        if op == OP_LITERAL_TOKENS:
            close_scalar_owner()
            length = int(plan[i + 1])
            slice_start = i + 2
            slice_end = slice_start + length
            for j in range(slice_start, slice_end):
                tid = int(plan[j])
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
                    # Linear lookup is fine — MAX_CARD_REFS = 64 and most
                    # literal slices contain 0–1 card-refs.
                    for k, ref_id in enumerate(toks.card_ref_ids):
                        if tid == ref_id and k not in card_ref_positions:
                            card_ref_positions[k] = pos
                            break
            i = slice_end
            continue

        if structured_plan:
            if scalar_owner_open is not None:
                keep_open_for_mana = op == OP_MANA and int(plan[i + 1]) == scalar_owner_open
                if not keep_open_for_mana:
                    close_scalar_owner()

            if op == OP_OPEN_STATE:
                emit_text("<bos><state>")
                i += 1
                continue

            if op == OP_CLOSE_STATE:
                close_option()
                close_scalar_owner()
                emit_text("</state><eos>")
                i += 1
                continue

            if op == OP_TURN:
                turn = int(plan[i + 1])
                step_id = int(plan[i + 2])
                step = _STEP_NAMES[step_id] if 0 <= step_id < len(_STEP_NAMES) else "Unknown"
                emit_text(f" turn={turn} step={step} ")
                i += 1 + arity
                continue

            if op == OP_LIFE:
                close_scalar_owner()
                owner = int(plan[i + 1])
                life = int(plan[i + 2])
                emit_text(("<self>" if owner == 0 else "<opp>") + f" life={life} mana=")
                scalar_owner_open = owner
                i += 1 + arity
                continue

            if op == OP_MANA:
                owner = int(plan[i + 1])
                color_id = int(plan[i + 2])
                amount = int(plan[i + 3])
                if scalar_owner_open is None:
                    emit_text("<self> mana=" if owner == 0 else "<opp> mana=")
                    scalar_owner_open = owner
                if 0 <= color_id < len(toks.mana_glyph_ids) and amount > 0:
                    glyph_ids = toks.mana_glyph_ids[color_id]
                    for _ in range(amount):
                        out.extend(glyph_ids)
                i += 1 + arity
                continue

            if op == OP_OPEN_ZONE:
                close_option()
                zone = int(plan[i + 1])
                owner = int(plan[i + 2])
                zone_stack.append((zone, owner))
                tag = _ZONE_TAGS.get(zone, "zone")
                owner_tag = "self" if owner == 0 else "opp"
                emit_text(f"<{owner_tag}><{tag}>")
                i += 1 + arity
                continue

            if op == OP_CLOSE_ZONE:
                close_option()
                if zone_stack:
                    zone, owner = zone_stack.pop()
                    tag = _ZONE_TAGS.get(zone, "zone")
                    owner_tag = "self" if owner == 0 else "opp"
                    emit_text(f"</{tag}></{owner_tag}>")
                i += 1
                continue

            if op == OP_OPEN_ACTIONS:
                emit_text("<actions>")
                i += 1
                continue

            if op == OP_CLOSE_ACTIONS:
                close_option()
                emit_text("</actions>")
                i += 1
                continue

            if op == OP_OPTION:
                close_option()
                kind_id = int(plan[i + 1])
                source_row = int(plan[i + 2])
                source_uuid_idx = int(plan[i + 3])
                _mana_cost_id = int(plan[i + 4])
                ability_idx = int(plan[i + 5])
                pos = len(out)
                out.append(toks.option_id)
                option_positions.append(pos)
                cur_target_bucket = []
                target_positions.append(cur_target_bucket)
                option_open = True
                verb = _ACTION_KINDS.get(kind_id, "unknown")
                emit_text(f" {verb}")
                if verb not in ("pass", "choice", "unknown"):
                    emit_text(" ")
                    if source_uuid_idx >= 0:
                        emit_card_ref(source_uuid_idx)
                    elif source_row >= 0 and source_row < len(cache.row_to_name):
                        emit_text(cache.row_to_name[source_row])
                if ability_idx >= 0 and kind_id == 3:
                    emit_text(f" ability {ability_idx}")
                i += 1 + arity
                continue

            if op == OP_TARGET:
                target_row = int(plan[i + 1])
                target_uuid_idx = int(plan[i + 2])
                pos = len(out)
                out.append(toks.target_open_id)
                if cur_target_bucket is not None:
                    cur_target_bucket.append(pos)
                emit_text(" ")
                if target_uuid_idx >= 0:
                    emit_card_ref(target_uuid_idx)
                elif target_row >= 0 and target_row < len(cache.row_to_name):
                    emit_text(cache.row_to_name[target_row])
                else:
                    emit_text("target")
                emit_text(" ")
                out.append(toks.target_close_id)
                i += 1 + arity
                continue

        if op == OP_PLACE_CARD:
            _slot_idx = int(plan[i + 1])
            row = int(plan[i + 2])
            status = int(plan[i + 3])
            uuid_idx = int(plan[i + 4])
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
            uuid_idx = int(plan[i + 1])
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
    toks._status_tapped = list(tokenizer.encode(" <sep> <tapped>", add_special_tokens=False))
    toks._status_untapped = list(tokenizer.encode(" <sep> <untapped>", add_special_tokens=False))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def assemble_batch(
    plans: Sequence[np.ndarray],
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
        Sequence of int32 render-plan arrays, one per env.
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

    assembled: list[_AssembledExample] = [_assemble_one(p, cache, toks, body_lists) for p in plans]

    seq_lengths = [len(ex.token_ids) for ex in assembled]
    actual_max = max(seq_lengths)
    if actual_max > max_tokens:
        if on_overflow == "raise":
            raise ValueError(f"assembled token length {actual_max} exceeds max_tokens={max_tokens}")
        # Truncate.
        for ex in assembled:
            if len(ex.token_ids) > max_tokens:
                ex.token_ids = ex.token_ids[:max_tokens]
                ex.card_ref_positions = {
                    k: p for k, p in ex.card_ref_positions.items() if p < max_tokens
                }
                ex.option_positions = [p for p in ex.option_positions if p < max_tokens]
                ex.target_positions = [
                    [p for p in tp if p < max_tokens] for tp in ex.target_positions
                ]
        seq_lengths = [len(ex.token_ids) for ex in assembled]

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
            option_positions[b, o] = int(pos)
            option_mask[b, o] = True
            for t, tpos in enumerate(ex.target_positions[o]):
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
