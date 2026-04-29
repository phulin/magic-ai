"""Render-plan opcode definitions, writer, and Python-side emitter.

PR 13-C from ``docs/text_encoder_plan.md`` §13. The render-plan ABI is
specified in ``docs/text_encoder_render_plan_abi.md``; this module is the
single source of truth for opcode IDs, status-bit layout, zone IDs, and the
fixed-arity table the assembler dispatches on.

Design note (deviates from the v0 ABI doc, see §13/§ABI updates)
----------------------------------------------------------------

The §3 renderer emits free-form text mixed with structural special tokens —
``" turn=3 step=Precombat Main "``, ``" life=20 mana={W}{W} "``,
``" count=53 "``, plus the action verbs ``cast``/``activate``/``attack``/...
None of these are reasonable to assemble at runtime without re-running BPE,
which the §13 hot path forbids.

Resolution: extend the opcode set with ``OP_LITERAL_TOKENS(len, tok0, tok1
…)`` — a generic carrier for a pre-tokenized int32 slice. The Python
emitter (which runs on the slow path during fixture/parity testing) calls
the tokenizer once per opcode to produce the slice. The assembler does pure
memcpy.

When PR 13-D swaps the emitter from Python to Go, the Go side has two
options:
  1. Carry the same BPE table and emit ``OP_LITERAL_TOKENS`` slices for the
     variable scalars (turn/life/count/step/action verbs / mana symbols).
  2. Replace the variable scalars with structured opcodes whose token
     decoding lives entirely in the assembler (e.g. ``OP_LIFE`` already
     gets a structured form per the ABI), and stop emitting
     ``OP_LITERAL_TOKENS`` for those cases.

This module assumes route (1) for v1: simpler, gives byte-for-byte parity
on day one, deferred optimization later.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import torch

from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    PendingOptionState,
    PlayerState,
    StackObjectState,
    TargetState,
)
from magic_ai.text_encoder.render import (
    _MANA_COLOR_TO_SYMBOL,
    _RENDER_ZONES,
    _ZONE_ORDER,
    OracleEntry,
    _resolve_perspective_idx,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS

# ---------------------------------------------------------------------------
# Opcodes (per ABI §5; ``OP_LITERAL_TOKENS`` added in PR 13-C, see module docstring)
# ---------------------------------------------------------------------------

OP_OPEN_STATE: Final = 1
OP_CLOSE_STATE: Final = 2
OP_TURN: Final = 3  # unused in PR 13-C; reserved for the structured-scalar route
OP_LIFE: Final = 4  # ditto
OP_MANA: Final = 5  # ditto
OP_OPEN_PLAYER: Final = 6
OP_CLOSE_PLAYER: Final = 7
OP_OPEN_ZONE: Final = 8
OP_CLOSE_ZONE: Final = 9
OP_PLACE_CARD: Final = 10
OP_COUNTER: Final = 11
OP_ATTACHED_TO: Final = 12
OP_OPEN_ACTIONS: Final = 13
OP_CLOSE_ACTIONS: Final = 14
OP_OPTION: Final = 15
OP_TARGET: Final = 16
# PR 13-C extension: variable-length literal-token slice produced by the
# emitter. Header is ``(opcode, length, tok0, tok1, …, tok_{length-1})``.
OP_LITERAL_TOKENS: Final = 17
# Convenience opcode for emitting the body of a card by row id (memcpy from
# the cache). The renderer needs to splice the card-ref token + the body's
# prefix (everything up to ` </card>`); we model that as a paired sequence:
#   OP_PLACE_CARD(slot, row, status, uuid_idx)
#     -- assembler emits:
#        - <card-ref:K> if uuid_idx >= 0
#        - cache.body_tokens(row) MINUS the trailing " </card>" tail.
#   OP_LITERAL_TOKENS(...) -- status-flag prelude (optional)
#   OP_END_CARD            -- assembler emits the trailing " </card>" tail.
OP_END_CARD: Final = 18
# OP_RAW_CARD: emit a non-cached card (e.g. stack object with name only).
# Payload: ref_idx (-1 if no ref). Followed by OP_LITERAL_TOKENS for the name.
OP_OPEN_RAW_CARD: Final = 19
OP_CLOSE_RAW_CARD: Final = 20

# v2 (card-body-dedup) opcodes. The body of each unique card-cache row is
# emitted once at the top of the snapshot inside an
# ``OP_OPEN_DICT … OP_DICT_ENTRY(row)* … OP_CLOSE_DICT`` block, and each
# per-zone occurrence is emitted as ``OP_PLACE_CARD_REF`` (same payload as
# OP_PLACE_CARD but the assembler emits ``<card-ref:K> <card>
# <dict-entry:row> [status] </card>`` instead of splicing the body). v1
# emitters never produce these opcodes; v1 assembler/v1 plan combinations
# remain byte-equal to the legacy renderer.
OP_OPEN_DICT: Final = 21
OP_CLOSE_DICT: Final = 22
OP_DICT_ENTRY: Final = 23
OP_PLACE_CARD_REF: Final = 24

# Fixed arity per opcode (number of int32 payload slots after the opcode
# header). ``-1`` marks a variable-length opcode whose first payload word is
# ``length`` and whose total in-stream size is ``2 + length``.
OPCODE_ARITY: Final[dict[int, int]] = {
    OP_OPEN_STATE: 0,
    OP_CLOSE_STATE: 0,
    OP_TURN: 2,
    OP_LIFE: 2,
    OP_MANA: 3,
    OP_OPEN_PLAYER: 1,
    OP_CLOSE_PLAYER: 0,
    OP_OPEN_ZONE: 2,
    OP_CLOSE_ZONE: 0,
    OP_PLACE_CARD: 4,
    OP_COUNTER: 2,
    OP_ATTACHED_TO: 1,
    OP_OPEN_ACTIONS: 0,
    OP_CLOSE_ACTIONS: 0,
    OP_OPTION: 5,
    OP_TARGET: 3,
    OP_LITERAL_TOKENS: -1,  # variable
    OP_END_CARD: 0,
    OP_OPEN_RAW_CARD: 1,
    OP_CLOSE_RAW_CARD: 0,
    OP_OPEN_DICT: 0,
    OP_CLOSE_DICT: 0,
    OP_DICT_ENTRY: 1,
    OP_PLACE_CARD_REF: 4,
}

# ---------------------------------------------------------------------------
# Status-bit layout (ABI §6).
# ---------------------------------------------------------------------------

STATUS_TAPPED: Final = 0x0001
STATUS_SICK: Final = 0x0002
STATUS_ATTACKING: Final = 0x0004
STATUS_BLOCKING: Final = 0x0008
STATUS_MONSTROUS: Final = 0x0010
STATUS_FLIPPED: Final = 0x0020
STATUS_FACEDOWN: Final = 0x0040
STATUS_PHASED_OUT: Final = 0x0080
STATUS_IS_CREATURE: Final = 0x0100
STATUS_IS_LAND: Final = 0x0200
STATUS_IS_ARTIFACT: Final = 0x0400
STATUS_IS_ATTACHED: Final = 0x0800

# A second bit ABI extension: distinguish "Tapped == False" (renderer emits
# ``<untapped>``) from "Tapped key absent" (renderer emits nothing). Bit 13
# is reserved-zero per ABI §6 (in the keyword-surfacing band) and we coopt
# bit 13 for "TAPPED_KNOWN" — the assembler only emits a tapped/untapped
# token if this bit is set.
STATUS_TAPPED_KNOWN: Final = 0x2000

# ---------------------------------------------------------------------------
# Zone IDs (ABI §7).
# ---------------------------------------------------------------------------

ZONE_HAND: Final = 0
ZONE_BATTLEFIELD: Final = 1
ZONE_GRAVEYARD: Final = 2
ZONE_EXILE: Final = 3
ZONE_LIBRARY: Final = 4
ZONE_STACK: Final = 5
ZONE_COMMAND: Final = 6

OWNER_SELF: Final = 0
OWNER_OPP: Final = 1


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class RenderPlanWriter:
    """Append-only int32 buffer with one method per opcode.

    All ``emit_*`` methods append the opcode header followed by its fixed
    payload. ``emit_literal_tokens`` is the only variable-length opcode and
    writes ``(opcode, length, tok0, tok1, …)``.
    """

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf: list[int] = []

    # -- structural --------------------------------------------------------

    def emit_open_state(self) -> None:
        self._buf.append(OP_OPEN_STATE)

    def emit_close_state(self) -> None:
        self._buf.append(OP_CLOSE_STATE)

    def emit_open_player(self, owner_id: int) -> None:
        self._buf.extend((OP_OPEN_PLAYER, int(owner_id)))

    def emit_close_player(self) -> None:
        self._buf.append(OP_CLOSE_PLAYER)

    def emit_open_zone(self, zone_id: int, owner_id: int) -> None:
        self._buf.extend((OP_OPEN_ZONE, int(zone_id), int(owner_id)))

    def emit_close_zone(self) -> None:
        self._buf.append(OP_CLOSE_ZONE)

    def emit_open_actions(self) -> None:
        self._buf.append(OP_OPEN_ACTIONS)

    def emit_close_actions(self) -> None:
        self._buf.append(OP_CLOSE_ACTIONS)

    # -- card / option / target -------------------------------------------

    def emit_place_card(
        self, slot_idx: int, card_row_id: int, status_bits: int, uuid_idx: int
    ) -> None:
        self._buf.extend(
            (
                OP_PLACE_CARD,
                int(slot_idx),
                int(card_row_id),
                int(status_bits),
                int(uuid_idx),
            )
        )

    def emit_end_card(self) -> None:
        self._buf.append(OP_END_CARD)

    def emit_counter(self, kind_id: int, count: int) -> None:
        self._buf.extend((OP_COUNTER, int(kind_id), int(count)))

    def emit_attached_to(self, target_uuid_idx: int) -> None:
        self._buf.extend((OP_ATTACHED_TO, int(target_uuid_idx)))

    def emit_option(
        self,
        kind_id: int,
        source_card_row: int,
        source_uuid_idx: int,
        mana_cost_id: int,
        ability_idx: int,
    ) -> None:
        self._buf.extend(
            (
                OP_OPTION,
                int(kind_id),
                int(source_card_row),
                int(source_uuid_idx),
                int(mana_cost_id),
                int(ability_idx),
            )
        )

    def emit_target(self, target_card_row: int, target_uuid_idx: int, target_kind: int) -> None:
        self._buf.extend((OP_TARGET, int(target_card_row), int(target_uuid_idx), int(target_kind)))

    def emit_open_raw_card(self, uuid_idx: int) -> None:
        self._buf.extend((OP_OPEN_RAW_CARD, int(uuid_idx)))

    def emit_close_raw_card(self) -> None:
        self._buf.append(OP_CLOSE_RAW_CARD)

    # -- v2 card-body-dedup -----------------------------------------------

    def emit_open_dict(self) -> None:
        self._buf.append(OP_OPEN_DICT)

    def emit_close_dict(self) -> None:
        self._buf.append(OP_CLOSE_DICT)

    def emit_dict_entry(self, card_row_id: int) -> None:
        self._buf.extend((OP_DICT_ENTRY, int(card_row_id)))

    def emit_place_card_ref(
        self, slot_idx: int, card_row_id: int, status_bits: int, uuid_idx: int
    ) -> None:
        self._buf.extend(
            (
                OP_PLACE_CARD_REF,
                int(slot_idx),
                int(card_row_id),
                int(status_bits),
                int(uuid_idx),
            )
        )

    # -- literal token slice (variable length) ----------------------------

    def emit_literal_tokens(self, tokens: Sequence[int]) -> None:
        self._buf.append(OP_LITERAL_TOKENS)
        self._buf.append(len(tokens))
        self._buf.extend(int(t) for t in tokens)

    # -- finalize ---------------------------------------------------------

    def finalize(self) -> torch.Tensor:
        return torch.tensor(self._buf, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Python-side emitter (stand-in for the eventual Go emitter).
# ---------------------------------------------------------------------------


CardRowLookup = Callable[[str], int]


def _player_zone(player: PlayerState | None, attr: str) -> list[GameCardState]:
    if player is None:
        return []
    cards = player.get(attr)  # type: ignore[misc]
    if not cards:
        return []
    return list(cards)


def _mana_pool_text(player: PlayerState | None) -> str:
    if player is None:
        return ""
    pool = player.get("ManaPool") or {}
    parts: list[str] = []
    for color, symbol in _MANA_COLOR_TO_SYMBOL.items():
        amount = int(pool.get(color, 0) or 0)  # type: ignore[arg-type]
        if amount > 0:
            parts.append(symbol * amount)
    return "".join(parts)


def _assign_card_refs(
    snapshot: GameStateSnapshot, max_card_refs: int = MAX_CARD_REFS
) -> dict[str, int]:
    """Mirror SnapshotRenderer._assign_card_refs exactly (deterministic)."""

    perspective = _resolve_perspective_idx(snapshot)
    players = snapshot["players"]
    self_player = players[perspective]
    opp_player = players[1 - perspective] if len(players) == 2 else None

    refs: dict[str, int] = {}
    for owner, attr in _ZONE_ORDER:
        player = self_player if owner == "self" else opp_player
        for card in _player_zone(player, attr):
            cid = card.get("ID")
            if not cid or cid in refs:
                continue
            if len(refs) >= max_card_refs:
                return refs
            refs[cid] = len(refs)
    for obj in snapshot.get("stack") or []:
        cid = obj.get("id")
        if not cid or cid in refs:
            continue
        if len(refs) >= max_card_refs:
            return refs
        refs[cid] = len(refs)
    return refs


def _status_bits_from_card(card: GameCardState) -> int:
    """Translate the snapshot's card flags into the status bitfield.

    PR 13-C only consumes ``Tapped`` because that's the only flag the
    renderer emits today; the rest of the bits are reserved for when the
    engine plumbs them through.
    """

    bits = 0
    tapped = card.get("Tapped")
    if tapped is True:
        bits |= STATUS_TAPPED | STATUS_TAPPED_KNOWN
    elif tapped is False:
        bits |= STATUS_TAPPED_KNOWN
    return bits


def emit_render_plan(
    snapshot: GameStateSnapshot,
    actions: Sequence[PendingOptionState] | None = None,
    *,
    card_row_lookup: CardRowLookup,
    tokenize: Callable[[str], list[int]],
    oracle: dict[str, OracleEntry] | None = None,
    max_card_refs: int = MAX_CARD_REFS,
    dedup_card_bodies: bool = False,
) -> torch.Tensor:
    """Walk ``snapshot`` (and ``actions``) and emit a render-plan int32 tensor.

    The traversal exactly matches :class:`SnapshotRenderer.render` so that
    the assembler's reconstruction is byte-equal to ``tokenize_snapshot
    (render_snapshot(snapshot, actions))``.

    Parameters
    ----------
    card_row_lookup:
        Maps a card name to its 1-indexed row in the card-token cache.
        Names not in the cache map to row 0 (unknown sentinel).
    tokenize:
        BPE-tokenize a string fragment to a list of int ids
        (``add_special_tokens=False``). The emitter calls this once per
        ``OP_LITERAL_TOKENS`` opcode; the assembler does not call the
        tokenizer at all.
    oracle:
        Optional oracle dict; if absent, ``card_row_lookup`` still works
        because card bodies come from the pre-built cache, not from the
        emitter.
    """

    w = RenderPlanWriter()
    refs = _assign_card_refs(snapshot, max_card_refs=max_card_refs)
    perspective = _resolve_perspective_idx(snapshot)
    players = snapshot["players"]
    self_player = players[perspective]
    opp_player = players[1 - perspective] if len(players) == 2 else None
    oracle = oracle or {}

    w.emit_open_state()
    # Replace `<bos><state>` opener — that's a literal-tokens chunk.
    w.emit_literal_tokens(tokenize("<bos><state>"))

    # v2 card-body dedup: emit the body of every unique cache row that
    # appears in this snapshot, in deterministic row order, as a
    # ``<dict>...</dict>`` block right after ``<bos><state>``. Per-zone
    # occurrences below switch from full-body splice to short references.
    if dedup_card_bodies:
        unique_rows: list[int] = []
        seen_rows: set[int] = set()
        for owner, attr, _open_tag, _close_tag in _RENDER_ZONES:
            if attr == "Exile":
                player = self_player if owner == "self" else opp_player
                if not _player_zone(player, attr):
                    continue
            player = self_player if owner == "self" else opp_player
            for card in _player_zone(player, attr):
                name = card.get("Name", "") or ""
                row = card_row_lookup(name)
                if row in seen_rows:
                    continue
                seen_rows.add(row)
                unique_rows.append(row)
        unique_rows.sort()
        w.emit_open_dict()
        for row in unique_rows:
            w.emit_dict_entry(row)
        w.emit_close_dict()

    # Top-level: " turn=N step=X ".
    turn = snapshot.get("turn", 0)
    step = snapshot.get("step", "") or ""
    w.emit_literal_tokens(tokenize(f" turn={turn} step={step} "))

    # Per-player blocks: <self> life=N mana=... </self> ; <opp> ...
    for scope, player in (("self", self_player), ("opp", opp_player)):
        if player is None:
            life = 0
            mana = ""
        else:
            life = int(player.get("Life", 0) or 0)
            mana = _mana_pool_text(player)
        open_tag = "<self>" if scope == "self" else "<opp>"
        close_tag = "</self>" if scope == "self" else "</opp>"
        w.emit_literal_tokens(tokenize(f"{open_tag} life={life} mana={mana} {close_tag}"))

    # Zone blocks: render each (owner, zone) just like SnapshotRenderer does.
    for owner, attr, open_tag, close_tag in _RENDER_ZONES:
        owner_open = "<self>" if owner == "self" else "<opp>"
        owner_close = "</self>" if owner == "self" else "</opp>"
        player = self_player if owner == "self" else opp_player
        cards = _player_zone(player, attr)
        if attr == "Exile" and not cards:
            continue
        # Write the wrapper opener as one literal slice so the tokenizer
        # produces exactly what the renderer's string would.
        w.emit_literal_tokens(tokenize(f"{owner_open}{open_tag}"))
        zone_id = _ZONE_ATTR_TO_ID[attr]
        owner_id = OWNER_SELF if owner == "self" else OWNER_OPP
        w.emit_open_zone(zone_id, owner_id)  # bookkeeping; emits no tokens
        for slot_idx, card in enumerate(cards):
            cid = card.get("ID", "")
            ref_idx = refs.get(cid, -1) if cid else -1
            name = card.get("Name", "") or ""
            row = card_row_lookup(name)
            status = _status_bits_from_card(card)
            if dedup_card_bodies:
                w.emit_place_card_ref(slot_idx, row, status, ref_idx)
            else:
                w.emit_place_card(slot_idx, row, status, ref_idx)
                w.emit_end_card()
        w.emit_close_zone()
        w.emit_literal_tokens(tokenize(f"{close_tag}{owner_close}"))

    # Library counts.
    self_lib = int(self_player.get("LibraryCount", 0) or 0)
    w.emit_literal_tokens(tokenize(f"<self><library> count={self_lib} </library></self>"))
    if opp_player is not None:
        opp_lib = int(opp_player.get("LibraryCount", 0) or 0)
        w.emit_literal_tokens(tokenize(f"<opp><library> count={opp_lib} </library></opp>"))

    # Stack
    w.emit_literal_tokens(tokenize("<stack>"))
    for stack_obj in snapshot.get("stack") or []:
        _emit_stack_object(w, stack_obj, refs, tokenize)
    w.emit_literal_tokens(tokenize("</stack>"))

    # Command zone.
    w.emit_literal_tokens(tokenize("<command></command>"))

    # Actions.
    if actions is None:
        pending = snapshot.get("pending")
        actions = pending.get("options", []) if pending is not None else []
    w.emit_literal_tokens(tokenize("<actions>"))
    w.emit_open_actions()  # bookkeeping
    for option in actions:
        _emit_option(w, option, refs, tokenize)
    w.emit_close_actions()
    w.emit_literal_tokens(tokenize("</actions>"))

    w.emit_literal_tokens(tokenize("</state><eos>"))
    w.emit_close_state()

    return w.finalize()


_ZONE_ATTR_TO_ID: dict[str, int] = {
    "Hand": ZONE_HAND,
    "Battlefield": ZONE_BATTLEFIELD,
    "Graveyard": ZONE_GRAVEYARD,
    "Exile": ZONE_EXILE,
}


def _emit_stack_object(
    w: RenderPlanWriter,
    obj: StackObjectState,
    refs: dict[str, int],
    tokenize: Callable[[str], list[int]],
) -> None:
    cid = obj.get("id", "")
    name = obj.get("name", "") or ""
    ref_idx = refs.get(cid, -1) if cid else -1
    if ref_idx >= 0:
        w.emit_literal_tokens(tokenize(f"<card> <card-ref:{ref_idx}> {name} </card>"))
    else:
        w.emit_literal_tokens(tokenize(f"<card> {name} </card>"))


def _emit_option(
    w: RenderPlanWriter,
    option: PendingOptionState,
    refs: dict[str, int],
    tokenize: Callable[[str], list[int]],
) -> None:
    """Emit a single option as a literal-tokens block.

    The renderer's per-option formatting (`cast <card-ref:K> cost {R}`,
    `pass`, `attack with <card-ref:K>`, etc.) lives in one helper here so
    the byte-for-byte match is obvious.
    """

    kind = (option.get("kind") or "").lower()
    card_id = option.get("card_id") or option.get("permanent_id") or ""
    card_name = option.get("card_name") or ""
    cost = option.get("mana_cost") or ""
    targets = option.get("valid_targets") or []

    parts: list[str] = ["<option> "]

    def emit_card_token(cid: str, fallback: str) -> None:
        ref = refs.get(cid)
        if ref is not None:
            parts.append(f"<card-ref:{ref}>")
        elif fallback:
            parts.append(fallback)

    if kind in ("cast", "play", "play_land"):
        verb = "play" if kind in ("play", "play_land") else "cast"
        parts.append(f"{verb} ")
        emit_card_token(card_id, card_name)
        if cost:
            parts.append(f" cost {cost}")
    elif kind in ("activate", "activated_ability"):
        parts.append("activate ")
        emit_card_token(card_id, card_name)
        ability_idx = option.get("ability_index")
        if ability_idx is not None:
            parts.append(f" ability {int(ability_idx)}")
        if cost:
            parts.append(f" cost {cost}")
    elif kind == "pass":
        parts.append("pass")
    elif kind == "attack":
        parts.append("attack with ")
        emit_card_token(card_id, card_name)
    elif kind == "block":
        parts.append("block with ")
        emit_card_token(card_id, card_name)
    elif kind == "mulligan":
        parts.append("mulligan")
    elif kind == "keep":
        parts.append("keep")
    else:
        label = option.get("label") or ""
        if kind:
            parts.append(kind)
        if label:
            if kind:
                parts.append(" ")
            parts.append(label)

    for target in targets:
        _append_target_text(parts, target, refs)

    parts.append(" </option>")
    w.emit_literal_tokens(tokenize("".join(parts)))


def _append_target_text(parts: list[str], target: TargetState, refs: dict[str, int]) -> None:
    tid = target.get("id", "")
    ref = refs.get(tid)
    parts.append(" <target>")
    if ref is not None:
        parts.append(f"<card-ref:{ref}>")
    else:
        label = target.get("label") or tid
        if label:
            parts.append(label)
    parts.append("</target>")
