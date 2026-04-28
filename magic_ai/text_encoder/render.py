"""Pure-Python deterministic renderer: GameStateSnapshot -> text + anchor metadata.

This is PR #2 of the text-encoder plan in ``docs/text_encoder_plan.md``. The
renderer turns a structured ``GameStateSnapshot`` plus the legal action options
into a single text string laced with custom tokens (mana symbols, status flags,
zone delimiters, intra-snapshot card references). The output is consumed by the
tokenizer in a follow-up PR; this module deliberately stops at ``str``.

Anchor positions (string indices for ``<card-ref:K>``, ``<option>``,
``<target>``) are returned alongside the text so downstream code can map them
to token ids and to per-card / per-option / per-target pooling slots.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    PendingOptionState,
    PlayerState,
    StackObjectState,
    TargetState,
)
from magic_ai.text_encoder.tokenizer import (
    MAX_CARD_REFS,
    card_ref_token,
)

DEFAULT_ORACLE_PATH = Path(__file__).resolve().parents[2] / "data" / "card_oracle_embeddings.json"


class OracleEntry(TypedDict, total=False):
    """Subset of fields used by the renderer from ``card_oracle_embeddings.json``."""

    name: str
    type_line: str
    mana_cost: str
    oracle_text: str
    power_toughness: str | None
    colors: list[str]


@dataclass(frozen=True)
class CardRefAnchor:
    """Position of a ``<card-ref:K>`` token in the rendered string."""

    ref_index: int
    engine_card_id: str
    name: str
    char_start: int  # offset of '<' in <card-ref:K>
    char_end: int  # offset just past '>'


@dataclass(frozen=True)
class OptionAnchor:
    """Position of an ``<option>`` block."""

    option_index: int
    kind: str
    char_start: int  # offset of '<' in <option>
    char_end: int  # offset just past '>' of </option>
    target_anchors: tuple[TargetAnchor, ...] = ()


@dataclass(frozen=True)
class TargetAnchor:
    """Position of a ``<target>`` block inside an option."""

    option_index: int
    target_index: int
    referenced_card_ref: int | None  # K from <card-ref:K> if the target binds to one
    char_start: int
    char_end: int


@dataclass
class RenderedSnapshot:
    text: str
    card_refs: dict[str, int] = field(default_factory=dict)
    card_ref_anchors: list[CardRefAnchor] = field(default_factory=list)
    option_anchors: list[OptionAnchor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Oracle text loader
# ---------------------------------------------------------------------------


def load_oracle_text(path: str | Path = DEFAULT_ORACLE_PATH) -> dict[str, OracleEntry]:
    """Load ``card_oracle_embeddings.json`` keyed by canonical Scryfall name.

    The renderer accepts the resulting dict via constructor injection so tests
    can pass small fixtures without disk I/O.
    """

    payload = json.loads(Path(path).read_text())
    out: dict[str, OracleEntry] = {}
    for record in payload.get("cards", []):
        name = record.get("name")
        if not name:
            continue
        entry: OracleEntry = {
            "name": name,
            "type_line": record.get("type_line", "") or "",
            "mana_cost": record.get("mana_cost", "") or "",
            "oracle_text": record.get("oracle_text", "") or "",
            "power_toughness": record.get("power_toughness"),
            "colors": list(record.get("colors") or []),
        }
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_oracle_text(text: str) -> str:
    """Render Scryfall oracle text verbatim.

    Mana symbols (``{W}``, ``{2/W}``, ``{T}``, ``{X}`` …) are left untouched —
    the tokenizer adds them as single tokens. Newlines are normalized to a
    single space so the resulting prose flows in the snapshot string.
    """

    if not text:
        return ""
    # Collapse newlines so reminder-text linebreaks don't fight the renderer's
    # own whitespace.
    return " ".join(text.split())


# Stable zone enumeration order for card-ref assignment + rendering.
# Each entry: (owner, zone_attr).
_ZONE_ORDER: tuple[tuple[str, str], ...] = (
    ("self", "Battlefield"),
    ("opp", "Battlefield"),
    ("self", "Hand"),
    ("opp", "Hand"),
    ("self", "Graveyard"),
    ("opp", "Graveyard"),
    ("self", "Exile"),
    ("opp", "Exile"),
)

# Zones rendered as content (not just a count).
_RENDER_ZONES: tuple[tuple[str, str, str, str], ...] = (
    # owner, zone attr, open tag, close tag
    ("self", "Battlefield", "<battlefield>", "</battlefield>"),
    ("opp", "Battlefield", "<battlefield>", "</battlefield>"),
    ("self", "Hand", "<hand>", "</hand>"),
    ("opp", "Hand", "<hand>", "</hand>"),
    ("self", "Graveyard", "<graveyard>", "</graveyard>"),
    ("opp", "Graveyard", "<graveyard>", "</graveyard>"),
    ("self", "Exile", "<exile>", "</exile>"),
    ("opp", "Exile", "<exile>", "</exile>"),
)


_MANA_COLOR_TO_SYMBOL = {
    "White": "{W}",
    "Blue": "{U}",
    "Black": "{B}",
    "Red": "{R}",
    "Green": "{G}",
    "Colorless": "{C}",
}


def _resolve_perspective_idx(snapshot: GameStateSnapshot) -> int:
    pending = snapshot.get("pending")
    if pending is not None:
        return int(pending.get("player_idx", 0))
    active = snapshot.get("active_player")
    for idx, player in enumerate(snapshot["players"]):
        if player.get("Name") == active or player.get("ID") == active:
            return idx
    return 0


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


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class SnapshotRenderer:
    """Stateful (per-snapshot) renderer.

    Construction is cheap; ``render`` is the main entrypoint. Using a class
    keeps the assignment of card-ref indices, the per-anchor metadata, and
    the in-progress text buffer co-located without threading them through
    every helper.
    """

    def __init__(
        self,
        oracle: dict[str, OracleEntry] | None = None,
        *,
        max_card_refs: int = MAX_CARD_REFS,
    ) -> None:
        self._oracle = oracle if oracle is not None else {}
        self._max_card_refs = max_card_refs

    # -- public ------------------------------------------------------------

    def render(
        self,
        snapshot: GameStateSnapshot,
        actions: Sequence[PendingOptionState] | None = None,
    ) -> RenderedSnapshot:
        result = RenderedSnapshot(text="")
        buf: list[str] = []
        # First pass: assign card-ref indices in deterministic traversal order.
        card_refs = self._assign_card_refs(snapshot)
        result.card_refs = card_refs

        perspective = _resolve_perspective_idx(snapshot)
        players = snapshot["players"]
        self_player = players[perspective]
        opp_player = players[1 - perspective] if len(players) == 2 else None

        buf.append("<bos><state>")
        # Top-level game info
        turn = snapshot.get("turn", 0)
        step = snapshot.get("step", "")
        buf.append(f" turn={turn} step={step} ")

        # Player blocks
        self._render_player_block(buf, "self", self_player)
        self._render_player_block(buf, "opp", opp_player)

        # Battlefield / hand / graveyard / exile
        for owner, attr, open_tag, close_tag in _RENDER_ZONES:
            owner_open = "<self>" if owner == "self" else "<opp>"
            owner_close = "</self>" if owner == "self" else "</opp>"
            player = self_player if owner == "self" else opp_player
            cards = _player_zone(player, attr)
            if attr == "Exile" and not cards:
                # Skip empty exile entirely.
                continue
            buf.append(owner_open)
            buf.append(open_tag)
            self._render_zone_cards(buf, result, cards, card_refs)
            buf.append(close_tag)
            buf.append(owner_close)

        # Library counts (count only — hidden information).
        buf.append("<self><library>")
        buf.append(f" count={int(self_player.get('LibraryCount', 0) or 0)} ")
        buf.append("</library></self>")
        if opp_player is not None:
            buf.append("<opp><library>")
            buf.append(f" count={int(opp_player.get('LibraryCount', 0) or 0)} ")
            buf.append("</library></opp>")

        # Stack
        buf.append("<stack>")
        for stack_obj in snapshot.get("stack") or []:
            self._render_stack_object(buf, stack_obj, card_refs)
        buf.append("</stack>")

        # Command zone — not represented in the snapshot today; emit empty for
        # forward compatibility per §3 (closers preserved for empty zones).
        buf.append("<command></command>")

        # Actions
        if actions is None:
            pending = snapshot.get("pending")
            actions = pending.get("options", []) if pending is not None else []
        self._render_actions(buf, result, actions, card_refs)

        buf.append("</state><eos>")
        result.text = "".join(buf)
        # Recompute anchor positions against the final string (we tracked them
        # against running buf-length; that already matches the final string).
        return result

    # -- card-ref assignment ----------------------------------------------

    def _assign_card_refs(self, snapshot: GameStateSnapshot) -> dict[str, int]:
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
                if len(refs) >= self._max_card_refs:
                    return refs
                refs[cid] = len(refs)
        # Stack objects too — they carry an ID.
        for obj in snapshot.get("stack") or []:
            cid = obj.get("id")
            if not cid or cid in refs:
                continue
            if len(refs) >= self._max_card_refs:
                return refs
            refs[cid] = len(refs)
        return refs

    # -- per-element renderers --------------------------------------------

    def _render_player_block(
        self,
        buf: list[str],
        scope: str,
        player: PlayerState | None,
    ) -> None:
        open_tag = "<self>" if scope == "self" else "<opp>"
        close_tag = "</self>" if scope == "self" else "</opp>"
        buf.append(open_tag)
        if player is None:
            buf.append(" life=0 mana= ")
        else:
            life = int(player.get("Life", 0) or 0)
            mana = _mana_pool_text(player)
            buf.append(f" life={life} mana={mana} ")
        buf.append(close_tag)

    def _render_zone_cards(
        self,
        buf: list[str],
        result: RenderedSnapshot,
        cards: Sequence[GameCardState],
        card_refs: dict[str, int],
    ) -> None:
        for card in cards:
            self._render_card(buf, result, card, card_refs)

    def _render_card(
        self,
        buf: list[str],
        result: RenderedSnapshot,
        card: GameCardState,
        card_refs: dict[str, int],
    ) -> None:
        cid = card.get("ID", "")
        ref_idx = card_refs.get(cid)
        name = card.get("Name", "") or ""
        if ref_idx is not None:
            ref_token = card_ref_token(ref_idx)
            char_start = sum(len(s) for s in buf)
            buf.append(ref_token)
            char_end = sum(len(s) for s in buf)
            result.card_ref_anchors.append(
                CardRefAnchor(
                    ref_index=ref_idx,
                    engine_card_id=cid,
                    name=name,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        buf.append("<card> ")
        buf.append(name)
        oracle = self._oracle.get(name)
        if oracle is not None:
            type_line = oracle.get("type_line", "") or ""
            mana_cost = oracle.get("mana_cost", "") or ""
            pt = oracle.get("power_toughness")
            text = render_oracle_text(oracle.get("oracle_text", "") or "")
            if type_line:
                buf.append(f" <sep> {type_line}")
            if mana_cost:
                buf.append(f" <sep> {mana_cost}")
            if pt:
                buf.append(f" <sep> {pt}")
            if text:
                buf.append(f" <sep> {text}")
        # Status flags — only what the snapshot actually carries.
        tapped = card.get("Tapped")
        if tapped is True:
            buf.append(" <sep> <tapped>")
        elif tapped is False:
            buf.append(" <sep> <untapped>")
        buf.append(" </card>")

    def _render_stack_object(
        self,
        buf: list[str],
        obj: StackObjectState,
        card_refs: dict[str, int],
    ) -> None:
        cid = obj.get("id", "")
        name = obj.get("name", "") or ""
        ref_idx = card_refs.get(cid)
        buf.append("<card> ")
        if ref_idx is not None:
            buf.append(card_ref_token(ref_idx) + " ")
        buf.append(name)
        buf.append(" </card>")

    # -- actions -----------------------------------------------------------

    def _render_actions(
        self,
        buf: list[str],
        result: RenderedSnapshot,
        actions: Sequence[PendingOptionState],
        card_refs: dict[str, int],
    ) -> None:
        buf.append("<actions>")
        for opt_idx, option in enumerate(actions):
            self._render_option(buf, result, opt_idx, option, card_refs)
        buf.append("</actions>")

    def _render_option(
        self,
        buf: list[str],
        result: RenderedSnapshot,
        option_index: int,
        option: PendingOptionState,
        card_refs: dict[str, int],
    ) -> None:
        char_start = sum(len(s) for s in buf)
        buf.append("<option> ")

        kind = (option.get("kind") or "").lower()
        card_id = option.get("card_id") or option.get("permanent_id") or ""
        card_name = option.get("card_name") or ""
        cost = option.get("mana_cost") or ""
        targets = option.get("valid_targets") or []
        target_anchors: list[TargetAnchor] = []

        def emit_card(cid: str, fallback_name: str) -> None:
            ref = card_refs.get(cid)
            if ref is not None:
                buf.append(card_ref_token(ref))
            elif fallback_name:
                buf.append(fallback_name)

        if kind in ("cast", "play", "play_land"):
            verb = "play" if kind in ("play", "play_land") else "cast"
            buf.append(f"{verb} ")
            emit_card(card_id, card_name)
            if cost:
                buf.append(f" cost {cost}")
        elif kind in ("activate", "activated_ability"):
            buf.append("activate ")
            emit_card(card_id, card_name)
            ability_idx = option.get("ability_index")
            if ability_idx is not None:
                buf.append(f" ability {int(ability_idx)}")
            if cost:
                buf.append(f" cost {cost}")
        elif kind == "pass":
            buf.append("pass")
        elif kind == "attack":
            buf.append("attack with ")
            emit_card(card_id, card_name)
        elif kind == "block":
            buf.append("block with ")
            emit_card(card_id, card_name)
        elif kind == "mulligan":
            buf.append("mulligan")
        elif kind == "keep":
            buf.append("keep")
        else:
            # Generic fallback: kind + label (label retained verbatim).
            label = option.get("label") or ""
            if kind:
                buf.append(kind)
            if label:
                if kind:
                    buf.append(" ")
                buf.append(label)

        for target_idx, target in enumerate(targets):
            self._render_target(buf, target_anchors, option_index, target_idx, target, card_refs)

        buf.append(" </option>")
        char_end = sum(len(s) for s in buf)
        result.option_anchors.append(
            OptionAnchor(
                option_index=option_index,
                kind=kind,
                char_start=char_start,
                char_end=char_end,
                target_anchors=tuple(target_anchors),
            )
        )

    def _render_target(
        self,
        buf: list[str],
        target_anchors: list[TargetAnchor],
        option_index: int,
        target_index: int,
        target: TargetState,
        card_refs: dict[str, int],
    ) -> None:
        tid = target.get("id", "")
        ref = card_refs.get(tid)
        char_start = sum(len(s) for s in buf)
        buf.append(" <target>")
        if ref is not None:
            buf.append(card_ref_token(ref))
        else:
            label = target.get("label") or tid
            if label:
                buf.append(label)
        buf.append("</target>")
        char_end = sum(len(s) for s in buf)
        target_anchors.append(
            TargetAnchor(
                option_index=option_index,
                target_index=target_index,
                referenced_card_ref=ref,
                char_start=char_start,
                char_end=char_end,
            )
        )


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------


def render_snapshot(
    snapshot: GameStateSnapshot,
    actions: Sequence[PendingOptionState] | None = None,
    *,
    oracle: dict[str, OracleEntry] | None = None,
    max_card_refs: int = MAX_CARD_REFS,
) -> RenderedSnapshot:
    """Render ``snapshot`` (and optional ``actions``) to text + anchor metadata.

    The ``oracle`` dict (loaded once via :func:`load_oracle_text`) supplies card
    type lines, mana costs, and oracle text. If a card name is missing from the
    dict the renderer falls back to emitting only the card name + status flags.
    """

    return SnapshotRenderer(oracle, max_card_refs=max_card_refs).render(snapshot, actions)
