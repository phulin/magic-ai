"""Pure-Python deterministic renderer: GameStateSnapshot -> text + anchor metadata.

This is PR #2 of the text-encoder plan in ``docs/text_encoder_plan.md``. The
renderer turns a structured ``GameStateSnapshot`` into a single text string
laced with custom tokens (mana symbols, status flags, zone delimiters,
intra-snapshot card references). State text is "what is true" only — pending
decisions live in the decision-spec section produced by
:mod:`magic_ai.text_encoder.render_spec`.

Anchor positions for ``<card-ref:K>`` are returned alongside the text so
downstream code can map them to token ids and per-card pooling slots.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    PlayerState,
    StackObjectState,
)
from magic_ai.text_encoder.tokenizer import (
    _CARD_TYPE_WORDS,
    MAX_CARD_REFS,
    card_ref_token,
    step_token,
)


class RenderError(RuntimeError):
    """Raised when a snapshot can't be rendered."""


DEFAULT_ORACLE_PATH = Path(__file__).resolve().parents[2] / "data" / "card_oracle_embeddings.json"


class OracleFace(TypedDict, total=False):
    name: str
    type_line: str
    mana_cost: str
    oracle_text: str
    power_toughness: str | None


class OracleEntry(TypedDict, total=False):
    name: str
    type_line: str
    mana_cost: str
    oracle_text: str
    power_toughness: str | None
    colors: list[str]
    layout: str
    card_faces: list[OracleFace]


@dataclass(frozen=True)
class CardRefAnchor:
    """Position of a ``<card-ref:K>`` token in the rendered string."""

    ref_index: int
    engine_card_id: str
    name: str
    char_start: int
    char_end: int


@dataclass
class RenderedSnapshot:
    text: str
    card_refs: dict[str, int] = field(default_factory=dict)
    card_ref_anchors: list[CardRefAnchor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Oracle text loader
# ---------------------------------------------------------------------------


def load_oracle_text(path: str | Path = DEFAULT_ORACLE_PATH) -> dict[str, OracleEntry]:
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
        layout = record.get("layout")
        if layout:
            entry["layout"] = layout
        faces_raw = record.get("card_faces") or []
        if faces_raw:
            faces: list[OracleFace] = []
            for face in faces_raw:
                face_pt = face.get("power_toughness")
                if face_pt is None:
                    fp = face.get("power")
                    ft = face.get("toughness")
                    if fp is not None and ft is not None:
                        face_pt = f"{fp}/{ft}"
                faces.append(
                    {
                        "name": face.get("name", "") or "",
                        "type_line": face.get("type_line", "") or "",
                        "mana_cost": face.get("mana_cost", "") or "",
                        "oracle_text": face.get("oracle_text", "") or "",
                        "power_toughness": face_pt,
                    }
                )
            entry["card_faces"] = faces
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


CARD_NAME_PLACEHOLDER = "<card-name>"


def _anonymize_self_references(text: str, names: Sequence[str]) -> str:
    if not text:
        return text
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    ordered.sort(key=len, reverse=True)
    out = text
    for name in ordered:
        out = out.replace(name, CARD_NAME_PLACEHOLDER)
    return out


_TYPE_LINE_DASH = "—"


def _split_type_line(type_line: str) -> tuple[list[str], str] | None:
    if not type_line:
        return ([], "")
    pre, _dash, post = type_line.partition(_TYPE_LINE_DASH)
    type_tokens: list[str] = []
    for word in pre.split():
        canon = word.lower()
        if canon not in _CARD_TYPE_WORDS:
            return None
        type_tokens.append(f"<{canon}>")
    return (type_tokens, post.strip())


def _is_planeswalker_face(face: OracleFace | OracleEntry) -> bool:
    type_line = face.get("type_line", "") or ""
    return "Planeswalker" in type_line


def _render_face_fields(
    parts: list[str], face: OracleFace | OracleEntry, names: Sequence[str] = ()
) -> None:
    parsed = _split_type_line(face.get("type_line", "") or "")
    if parsed is None:
        parts.append(face.get("type_line", "") or "")
    else:
        type_tokens, subtype_text = parsed
        parts.extend(type_tokens)
        if subtype_text:
            parts.append(f"<subtypes>{subtype_text}</subtypes>")

    mana_cost = face.get("mana_cost", "") or ""
    if mana_cost:
        parts.append(f"<mana-cost>{mana_cost}</mana-cost>")

    pt = face.get("power_toughness")
    if pt:
        if _is_planeswalker_face(face):
            parts.append(f"<loyalty>{pt}</loyalty>")
        else:
            parts.append(f"<pt>{pt}</pt>")

    text = _anonymize_self_references(render_oracle_text(face.get("oracle_text", "") or ""), names)
    if text:
        parts.append(f"<rules-text>{text}</rules-text>")


_MULTI_FACE_LAYOUTS: frozenset[str] = frozenset(
    {"split", "modal_dfc", "transform", "adventure", "flip"}
)


def _is_multi_face(oracle: OracleEntry) -> bool:
    faces = oracle.get("card_faces") or []
    if faces:
        return True
    layout = oracle.get("layout")
    if layout and layout in _MULTI_FACE_LAYOUTS:
        return True
    return False


def _ordered_faces(name: str, oracle: OracleEntry) -> list[OracleFace]:
    faces: list[OracleFace] = list(oracle.get("card_faces") or [])
    if faces:
        return faces
    fallback: OracleFace = {
        "name": name,
        "type_line": oracle.get("type_line", "") or "",
        "mana_cost": oracle.get("mana_cost", "") or "",
        "oracle_text": oracle.get("oracle_text", "") or "",
        "power_toughness": oracle.get("power_toughness"),
    }
    return [fallback]


def render_card_body(name: str, oracle: OracleEntry | None) -> str:
    parts: list[str] = ["<card>"]
    if oracle is not None and _is_multi_face(oracle):
        faces = _ordered_faces(name, oracle)
        all_names: list[str] = [name]
        for face in faces:
            face_name = face.get("name", "") or ""
            if face_name:
                all_names.append(face_name)
        for face in faces:
            parts.append("<face>")
            _render_face_fields(parts, face, all_names)
            parts.append("</face>")
    elif oracle is not None:
        _render_face_fields(parts, oracle, (name,))
    parts.append("</card>")
    return "".join(parts)


def render_oracle_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())


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

_RENDER_ZONES: tuple[tuple[str, str, str, str], ...] = (
    ("self", "Battlefield", "<battlefield>", "</battlefield>"),
    ("opp", "Battlefield", "<battlefield>", "</battlefield>"),
    ("self", "Hand", "<hand>", "</hand>"),
    ("opp", "Hand", "<hand>", "</hand>"),
    ("self", "Graveyard", "<graveyard>", "</graveyard>"),
    ("opp", "Graveyard", "<graveyard>", "</graveyard>"),
    ("self", "Exile", "<exile>", "</exile>"),
    ("opp", "Exile", "<exile>", "</exile>"),
)


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


_POOL_MANA_TOKEN_BY_COLOR: dict[str, str] = {
    "White": "<mana:W>",
    "Blue": "<mana:U>",
    "Black": "<mana:B>",
    "Red": "<mana:R>",
    "Green": "<mana:G>",
    "Colorless": "<mana:C>",
}


def _mana_pool_tokens(player: PlayerState | None) -> str:
    if player is None:
        return ""
    pool = player.get("ManaPool") or {}
    parts: list[str] = []
    for color, token in _POOL_MANA_TOKEN_BY_COLOR.items():
        amount = int(pool.get(color, 0) or 0)  # type: ignore[arg-type]
        parts.extend(token for _ in range(amount))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class SnapshotRenderer:
    """Stateful (per-snapshot) renderer for state text only."""

    def __init__(
        self,
        oracle: dict[str, OracleEntry] | None = None,
        *,
        max_card_refs: int = MAX_CARD_REFS,
        self_token_id: int | None = None,
        opp_token_id: int | None = None,
        mana_token_ids: Sequence[int] | None = None,
        card_ref_token_ids: Sequence[int] | None = None,
    ) -> None:
        self._oracle = oracle if oracle is not None else {}
        self._max_card_refs = max_card_refs
        self._self_token_id = self_token_id
        self._opp_token_id = opp_token_id
        self._mana_token_ids = tuple(int(tid) for tid in mana_token_ids or ())
        self._card_ref_token_ids = tuple(
            int(tid) for tid in (card_ref_token_ids or range(max_card_refs))
        )

    def render(self, snapshot: GameStateSnapshot) -> RenderedSnapshot:
        result = RenderedSnapshot(text="")
        buf: list[str] = []
        card_refs = self._assign_card_refs(snapshot)
        result.card_refs = card_refs

        perspective = _resolve_perspective_idx(snapshot)
        players = snapshot["players"]
        self_player = players[perspective]
        opp_player = players[1 - perspective] if len(players) == 2 else None

        buf.append("<bos><state>")
        turn = snapshot.get("turn", 0)
        step = snapshot.get("step", "") or ""
        try:
            step_tok = step_token(step)
        except KeyError:
            step_tok = ""
        buf.append(f"<turn>{int(turn)}</turn>{step_tok}")

        self._render_player_block(buf, "self", self_player)
        self._render_player_block(buf, "opp", opp_player)

        for owner, attr, open_tag, close_tag in _RENDER_ZONES:
            if owner == "opp" and attr == "Hand":
                continue
            owner_open = "<self>" if owner == "self" else "<opp>"
            owner_close = "</self>" if owner == "self" else "</opp>"
            player = self_player if owner == "self" else opp_player
            cards = _player_zone(player, attr)
            if attr == "Exile" and not cards:
                continue
            buf.append(owner_open)
            buf.append(open_tag)
            self._render_zone_cards(buf, result, cards, card_refs)
            buf.append(close_tag)
            buf.append(owner_close)

        self_lib = int(self_player.get("LibraryCount", 0) or 0)
        buf.append(f"<self><library>{self_lib}</library></self>")
        if opp_player is not None:
            opp_lib = int(opp_player.get("LibraryCount", 0) or 0)
            buf.append(f"<opp><library>{opp_lib}</library></opp>")

        buf.append("<stack>")
        for stack_obj in snapshot.get("stack") or []:
            self._render_stack_object(buf, result, stack_obj, card_refs)
        buf.append("</stack>")

        has_command = any(
            (p or {}).get("Command") for p in (self_player, opp_player) if p is not None
        )
        if has_command:
            buf.append("<command></command>")

        buf.append("</state><eos>")
        result.text = "".join(buf)
        return result

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
        for obj in snapshot.get("stack") or []:
            cid = obj.get("id")
            if not cid or cid in refs:
                continue
            if len(refs) >= self._max_card_refs:
                return refs
            refs[cid] = len(refs)
        return refs

    def _render_player_block(
        self,
        buf: list[str],
        scope: str,
        player: PlayerState | None,
    ) -> None:
        open_tag = "<self>" if scope == "self" else "<opp>"
        close_tag = "</self>" if scope == "self" else "</opp>"
        if player is None:
            life = 0
            mana_tokens = ""
        else:
            life = int(player.get("Life", 0) or 0)
            mana_tokens = _mana_pool_tokens(player)
        buf.append(f"{open_tag}<life>{life}</life><mana-pool>{mana_tokens}</mana-pool>{close_tag}")

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
        body = render_card_body(name, self._oracle.get(name))
        closing = "</card>"
        assert body.endswith(closing)
        buf.append(body[: -len(closing)])
        tapped = card.get("Tapped")
        if tapped is True:
            buf.append("<tapped>")
        elif tapped is False:
            buf.append("<untapped>")
        buf.append(closing)

    def _render_stack_object(
        self,
        buf: list[str],
        result: RenderedSnapshot,
        obj: StackObjectState,
        card_refs: dict[str, int],
    ) -> None:
        cid = obj.get("id", "")
        name = obj.get("name", "") or ""
        card = cast(GameCardState, {"ID": cid, "Name": name} if cid else {"Name": name})
        self._render_card(buf, result, card, card_refs)


def render_snapshot(
    snapshot: GameStateSnapshot,
    *,
    oracle: dict[str, OracleEntry] | None = None,
    max_card_refs: int = MAX_CARD_REFS,
    self_token_id: int | None = None,
    opp_token_id: int | None = None,
    mana_token_ids: Sequence[int] | None = None,
    card_ref_token_ids: Sequence[int] | None = None,
) -> RenderedSnapshot:
    """Render ``snapshot`` to state text + card-ref anchor metadata.

    Pending decisions are NOT rendered into state text; see
    :mod:`magic_ai.text_encoder.render_spec` for the decision-spec section
    that the encoder consumes alongside state text.
    """

    return SnapshotRenderer(
        oracle,
        max_card_refs=max_card_refs,
        self_token_id=self_token_id,
        opp_token_id=opp_token_id,
        mana_token_ids=mana_token_ids,
        card_ref_token_ids=card_ref_token_ids,
    ).render(snapshot)
