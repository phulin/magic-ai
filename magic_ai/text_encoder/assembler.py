"""Render-plan -> ``TextEncodedBatch`` assembler.

PR 13-C from ``docs/text_encoder_plan.md`` §13. Walks the int32 render-plan
stream produced by :mod:`magic_ai.text_encoder.render_plan` (the Python
stand-in for the eventual Go emitter), memcpys card-body token slices from
the :class:`CardTokenCache`, and writes the result into a preallocated
``[B, max_tokens]`` token buffer alongside ``card_ref_positions`` /
``option_positions`` / ``target_positions``.

The assembler **never** calls the tokenizer. All tokenizer use is at
init time to look up the int ids of structural and status tokens.
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
    OP_LITERAL_TOKENS,
    OP_OPEN_ACTIONS,
    OP_OPEN_PLAYER,
    OP_OPEN_RAW_CARD,
    OP_OPEN_STATE,
    OP_OPEN_ZONE,
    OP_OPTION,
    OP_PLACE_CARD,
    OP_TARGET,
    OPCODE_ARITY,
    STATUS_TAPPED,
    STATUS_TAPPED_KNOWN,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS

CARD_CLOSER_TEXT = " </card>"


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


def _single_id(tokenizer: PreTrainedTokenizerFast, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if isinstance(tid, list):
        raise TypeError(f"convert_tokens_to_ids({token!r}) returned a list")
    return int(tid)


def build_assembler_tokens(tokenizer: PreTrainedTokenizerFast) -> AssemblerTokens:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return AssemblerTokens(
        pad_id=int(pad),
        option_id=_single_id(tokenizer, "<option>"),
        target_open_id=_single_id(tokenizer, "<target>"),
        target_close_id=_single_id(tokenizer, "</target>"),
        tapped_id=_single_id(tokenizer, "<tapped>"),
        untapped_id=_single_id(tokenizer, "<untapped>"),
        card_ref_ids=[_single_id(tokenizer, f"<card-ref:{k}>") for k in range(MAX_CARD_REFS)],
        card_closer_ids=list(tokenizer.encode(CARD_CLOSER_TEXT, add_special_tokens=False)),
    )


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
) -> _AssembledExample:
    out: list[int] = []
    card_ref_positions: dict[int, int] = {}
    option_positions: list[int] = []
    target_positions: list[list[int]] = []

    cur_target_bucket: list[int] | None = None

    i = 0
    n = int(plan.shape[0])
    while i < n:
        op = int(plan[i])
        arity = OPCODE_ARITY.get(op)
        if arity is None:
            raise ValueError(f"unknown opcode {op} at position {i}")

        if op == OP_LITERAL_TOKENS:
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

        if op == OP_PLACE_CARD:
            _slot_idx = int(plan[i + 1])
            row = int(plan[i + 2])
            status = int(plan[i + 3])
            uuid_idx = int(plan[i + 4])
            # Emit <card-ref:K> if attached.
            if uuid_idx >= 0:
                ref_id = toks.card_ref_ids[uuid_idx]
                pos = len(out)
                out.append(ref_id)
                if uuid_idx not in card_ref_positions:
                    card_ref_positions[uuid_idx] = pos
            # Memcpy the card body, dropping its trailing ` </card>` tail
            # — the END_CARD opcode (or the status decoder) will replace it.
            body = cache.token_buffer[int(cache.offsets[row]) : int(cache.offsets[row + 1])]
            tail = toks.card_closer_ids
            if len(tail) and len(body) >= len(tail) and list(body[-len(tail) :]) == tail:
                trimmed = body[: -len(tail)]
            else:
                trimmed = body
            out.extend(int(t) for t in trimmed.tolist())
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
            i += 1 + arity
            continue

        if op == OP_END_CARD:
            out.extend(toks.card_closer_ids)
            i += 1
            continue

        if op == OP_OPEN_RAW_CARD:
            uuid_idx = int(plan[i + 1])
            if uuid_idx >= 0:
                ref_id = toks.card_ref_ids[uuid_idx]
                pos = len(out)
                out.append(ref_id)
                if uuid_idx not in card_ref_positions:
                    card_ref_positions[uuid_idx] = pos
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
        Used only for one-time structural-token id lookup; never invoked
        per opcode.
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

    assembled: list[_AssembledExample] = [_assemble_one(p, cache, toks) for p in plans]

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
