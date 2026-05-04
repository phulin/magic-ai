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
)
from magic_ai.text_encoder.tokenizer import (
    _CARD_TYPE_WORDS,
    MAX_CARD_REFS,
    MAX_NUM,
    card_ref_token,
    step_token,
)


class RenderError(RuntimeError):
    """Raised when a snapshot can't be rendered into a legal blank layout.

    Examples: an inline-blank pass produces a blank with zero legal candidates,
    or a priority option references a card / permanent that isn't in any
    visible zone. Render-time validation per
    ``docs/text_encoder_inline_blanks_plan.md`` "Legality enforcement".
    """


DEFAULT_ORACLE_PATH = Path(__file__).resolve().parents[2] / "data" / "card_oracle_embeddings.json"


class OracleFace(TypedDict, total=False):
    """One face of a multi-face card (split, MDFC, transform, adventure, flip).

    Mirrors the per-face shape Scryfall returns under ``card_faces``: each face
    has its own ``name``, ``type_line``, ``mana_cost``, ``oracle_text``, and
    optional ``power``/``toughness``. We pre-flatten ``power``/``toughness``
    into ``power_toughness`` for renderer convenience, matching the top-level
    convention in :func:`load_oracle_text`.
    """

    name: str
    type_line: str
    mana_cost: str
    oracle_text: str
    power_toughness: str | None


class OracleEntry(TypedDict, total=False):
    """Subset of fields used by the renderer from ``card_oracle_embeddings.json``.

    For single-face cards the top-level ``type_line`` / ``mana_cost`` /
    ``oracle_text`` / ``power_toughness`` carry everything. For multi-face
    cards (split, MDFC, transform, adventure, flip) each face is also exposed
    via ``card_faces``; when present the renderer prefers the per-face fields
    over the top-level ones. The ``layout`` field is preserved when known so
    callers can dispatch on it.
    """

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
    char_start: int  # offset of '<' in <card-ref:K>
    char_end: int  # offset just past '>'


@dataclass(frozen=True)
class BlankAnchor:
    """Position of an inline-blank ``<choose-*>`` token in the rendered string.

    Step 2 of ``docs/text_encoder_inline_blanks_plan.md``. Only priority
    blanks (``<choose-play>`` / ``<use-ability>`` / ``<pass>``) are emitted
    today; combat / target / mode / X / mana-source kinds land in later
    steps. All priority anchors in a snapshot share **one** ``group_id``
    with ``group_kind == "CROSS_BLANK"`` per the plan's cross-blank
    softmax design; the per-anchor "logit" is the score for selecting the
    ``<chosen>`` token at that position, so ``legal_token_ids`` is the
    singleton ``(<chosen>_id,)``.
    """

    blank_index: int  # ordinal across the snapshot (render order)
    kind: str  # "<choose-play>" / "<use-ability>" / "<pass>"
    char_start: int
    char_end: int
    group_id: int
    group_kind: str  # "CROSS_BLANK" | "PER_BLANK" | "CONSTRAINED"
    legal_token_ids: tuple[int, ...]
    # Provenance: index into the engine's options list, so the engine adapter
    # can map a chosen blank back to a concrete action.
    option_index: int


@dataclass
class RenderedSnapshot:
    text: str
    card_refs: dict[str, int] = field(default_factory=dict)
    card_ref_anchors: list[CardRefAnchor] = field(default_factory=list)
    blank_anchors: list[BlankAnchor] = field(default_factory=list)


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
        # Preserve multi-face metadata when present. The current
        # ``card_oracle_embeddings.json`` flattens these fields out (see
        # ``scripts/build_card_embeddings.py``), but the renderer accepts the
        # raw Scryfall shape so test fixtures can opt in without touching the
        # build pipeline.
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
    """Replace every occurrence of any card / face name with ``<card-name>``.

    The encoder is meant to learn from rules-text mechanics, not card-name
    identity. Self-references inside oracle text are masked so that a card's
    abilities cannot be cross-referenced to its name string. ``names`` is
    matched longest-first so e.g. ``"Lightning Bolt"`` is replaced before its
    substring ``"Lightning"`` is considered. Matching is case-sensitive
    because Scryfall canonicalizes printed names.
    """

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


_TYPE_LINE_DASH = "—"  # U+2014 em-dash, the Scryfall canonical separator.


def _split_type_line(type_line: str) -> tuple[list[str], str] | None:
    """Return ``(type_tokens, subtype_text)`` parsed from ``type_line``.

    Returns ``None`` when the pre-dash portion contains a word that isn't a
    recognized MTG supertype / card type — caller falls back to literal text.
    """

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
    """Append a face's fields to ``parts`` as token-delimited records.

    Layout::

        <type1><type2>...<subtypes>SUBTYPES</subtypes>
        <mana-cost>COST</mana-cost>
        (<pt>P/T</pt> | <loyalty>N</loyalty>)?
        <rules-text>ORACLE</rules-text>

    Empty fields are skipped entirely. Only oracle text has self-references
    rewritten to ``<card-name>``; subtype text is kept as-is.
    """

    parsed = _split_type_line(face.get("type_line", "") or "")
    if parsed is None:
        # Unrecognized type word — fall back to the literal type line so we
        # don't lose information.
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


# Layouts whose ``card_faces`` represent two halves the model should see
# together inside one ``<card>`` block. ``flip`` (Kamigawa-block flip cards)
# is included for completeness even though the engine doesn't surface them
# today.
_MULTI_FACE_LAYOUTS: frozenset[str] = frozenset(
    {"split", "modal_dfc", "transform", "adventure", "flip"}
)


def _is_multi_face(oracle: OracleEntry) -> bool:
    """Return True if ``oracle`` describes a multi-face card.

    Multi-face is detected via the presence of ``card_faces`` (most robust:
    the raw Scryfall payload always carries this for multi-face cards) or
    via a ``layout`` value in :data:`_MULTI_FACE_LAYOUTS`. The two are
    redundant by construction, but we accept either so fixtures can be terse.
    """

    faces = oracle.get("card_faces") or []
    if faces:
        return True
    layout = oracle.get("layout")
    if layout and layout in _MULTI_FACE_LAYOUTS:
        return True
    return False


def _ordered_faces(name: str, oracle: OracleEntry) -> list[OracleFace]:
    """Return the faces in render order.

    - ``adventure``: creature half first, adventure half second. Scryfall's
      ``card_faces`` ordering is creature-then-adventure; the printed name on
      the top-level card is the creature side, so we emit faces in their
      Scryfall order which already matches.
    - ``split`` / ``modal_dfc`` / ``transform`` / ``flip``: emit in the
      Scryfall-provided order (left-to-right for split, front-to-back for
      MDFC/transform, original-to-flipped for flip).

    If ``card_faces`` is absent but ``layout`` flags multi-face, fall back to
    a single synthetic face from the top-level fields so the renderer still
    produces the standard structure.
    """

    faces: list[OracleFace] = list(oracle.get("card_faces") or [])
    if faces:
        return faces
    # Fallback: layout claimed multi-face but no faces array; treat the
    # top-level fields as the single face we know about.
    fallback: OracleFace = {
        "name": name,
        "type_line": oracle.get("type_line", "") or "",
        "mana_cost": oracle.get("mana_cost", "") or "",
        "oracle_text": oracle.get("oracle_text", "") or "",
        "power_toughness": oracle.get("power_toughness"),
    }
    return [fallback]


def render_card_body(name: str, oracle: OracleEntry | None) -> str:
    """Render the static ``<card> ...fields... </card>`` fragment for one card.

    For single-face cards the layout is::

        <card> Name <sep> Type <sep> mana <sep> P/T <sep> oracle </card>

    For multi-face cards (split / MDFC / transform / adventure / flip) every
    face's fields are emitted inside the same ``<card>`` block, separated by
    `` <sep> // <sep> ``::

        <card> Name <sep> Type <sep> mana <sep> P/T <sep> oracle
               <sep> // <sep>
               OtherName <sep> Type <sep> mana <sep> P/T <sep> oracle </card>

    The cached body is the same regardless of which face is "active"
    in-game — the model has both faces' rules text always; per-game state
    (which face is currently up) is a separate status concern.

    Status flags, zone wrappers, and ``<card-ref:K>`` are *not* included —
    those are state-dependent and inserted by the snapshot renderer /
    hot-path assembler. This is the unit cached at startup by
    :mod:`magic_ai.text_encoder.card_cache` and reused by
    :class:`SnapshotRenderer` so the cache and slow-path agree byte-for-byte.
    """

    parts: list[str] = ["<card>"]
    if oracle is not None and _is_multi_face(oracle):
        faces = _ordered_faces(name, oracle)
        # Collect every face name (plus the canonical printed name) so each
        # face's oracle text has *all* name variants masked.
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


_POOL_MANA_TOKEN_BY_COLOR: dict[str, str] = {
    "White": "<mana:W>",
    "Blue": "<mana:U>",
    "Black": "<mana:B>",
    "Red": "<mana:R>",
    "Green": "<mana:G>",
    "Colorless": "<mana:C>",
}


def _mana_pool_tokens(player: PlayerState | None) -> str:
    """Concatenated ``<mana:X>`` tokens for a player's floating pool."""

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
        chosen_token_id: int | None = None,
        none_token_id: int | None = None,
        yes_token_id: int | None = None,
        no_token_id: int | None = None,
        num_token_ids: Sequence[int] | None = None,
        mana_token_ids: Sequence[int] | None = None,
        card_ref_token_ids: Sequence[int] | None = None,
    ) -> None:
        self._oracle = oracle if oracle is not None else {}
        self._max_card_refs = max_card_refs
        self._chosen_token_id = 0 if chosen_token_id is None else chosen_token_id
        self._none_token_id = none_token_id
        self._yes_token_id = yes_token_id
        self._no_token_id = no_token_id
        self._num_token_ids = tuple(int(tid) for tid in num_token_ids or ())
        self._mana_token_ids = tuple(int(tid) for tid in mana_token_ids or ())
        self._card_ref_token_ids = tuple(
            int(tid) for tid in (card_ref_token_ids or range(max_card_refs))
        )
        self._cur_self_id: str = ""
        self._cur_opp_id: str = ""
        # Per-render scratch state for inline-blank emission. Reset in
        # ``render`` before each pass so a renderer instance can be reused.
        self._blank_options_by_card: dict[
            str, list[tuple[str, int, tuple[int, ...], str, tuple[object, ...]]]
        ] = {}
        self._pass_options: list[int] = []
        self._blank_group_id: int = 0
        self._blank_index: int = 0
        self._pending_kind: str = ""

    # -- public ------------------------------------------------------------

    def render(
        self,
        snapshot: GameStateSnapshot,
        actions: Sequence[PendingOptionState] | None = None,
    ) -> RenderedSnapshot:
        result = RenderedSnapshot(text="")
        buf: list[str] = []
        # Reset per-render scratch state.
        self._blank_options_by_card = {}
        self._pass_options = []
        self._blank_group_id = 0
        self._blank_index = 0
        self._pending_kind = ""
        # First pass: assign card-ref indices in deterministic traversal order.
        card_refs = self._assign_card_refs(snapshot)
        result.card_refs = card_refs

        perspective = _resolve_perspective_idx(snapshot)
        players = snapshot["players"]
        self_player = players[perspective]
        opp_player = players[1 - perspective] if len(players) == 2 else None
        self._cur_self_id = str(self_player.get("ID", "")) if self_player else ""
        self._cur_opp_id = str(opp_player.get("ID", "")) if opp_player else ""

        # Resolve the pending options once so inline pre-classification and
        # the later action / choices block use the same list.
        pending = snapshot.get("pending")
        self._pending_kind = str((pending or {}).get("kind") or "").lower()
        if actions is None:
            resolved_actions: Sequence[PendingOptionState] = (
                pending.get("options", []) if pending is not None else []
            )
        else:
            resolved_actions = actions

        # Inline-blank pre-classification: bucket pending options by source
        # card / permanent so the zone walk can splice the correct
        # ``<choose-play>`` / ``<use-ability>`` token next to each card.
        # Pass options are buffered for the trailing ``<choices>`` block.
        self._classify_inline_options(resolved_actions, card_refs)

        buf.append("<bos><state>")
        # Top-level game info: <turn>N</turn><step:...>
        turn = snapshot.get("turn", 0)
        step = snapshot.get("step", "") or ""
        try:
            step_tok = step_token(step)
        except KeyError:
            step_tok = ""
        buf.append(f"<turn>{int(turn)}</turn>{step_tok}")

        # Player blocks
        self._render_player_block(buf, "self", self_player)
        self._render_player_block(buf, "opp", opp_player)

        # Battlefield / hand / graveyard / exile.
        # Skip the opponent's hand entirely (fog of war). Skip empty Exile.
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

        # Library counts: <self><library>{N}</library></self> (and opp).
        self_lib = int(self_player.get("LibraryCount", 0) or 0)
        buf.append(f"<self><library>{self_lib}</library></self>")
        if opp_player is not None:
            opp_lib = int(opp_player.get("LibraryCount", 0) or 0)
            buf.append(f"<opp><library>{opp_lib}</library></opp>")

        # Stack
        buf.append("<stack>")
        for stack_obj in snapshot.get("stack") or []:
            self._render_stack_object(buf, stack_obj, card_refs)
        buf.append("</stack>")

        # Command zone — only emitted when at least one player has a non-empty
        # Command zone. Today the snapshot does not surface command-zone
        # contents, so this is rare in practice.
        has_command = any(
            (p or {}).get("Command") for p in (self_player, opp_player) if p is not None
        )
        if has_command:
            buf.append("<command></command>")

        # Decision blanks / choices.
        self._render_choices_inline(buf, result, resolved_actions)

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
        # Reuse the same `<card> ... </card>` fragment that the offline cache
        # builds, then splice state-dependent status flags in front of the
        # closing tag so the cache and slow-path agree byte-for-byte on the
        # static portion.
        body = render_card_body(name, self._oracle.get(name))
        closing = "</card>"
        assert body.endswith(closing)
        buf.append(body[: -len(closing)])
        # Status flags — only what the snapshot actually carries.
        tapped = card.get("Tapped")
        if tapped is True:
            buf.append("<tapped>")
        elif tapped is False:
            buf.append("<untapped>")
        buf.append(closing)
        # Inline-blank emission: any priority anchor whose source is this
        # card / permanent gets emitted immediately after ``</card>``. The
        # per-card option list was sorted at classification time so render
        # order is deterministic across dict-iteration perturbations.
        if cid:
            options_here = self._blank_options_by_card.get(cid)
            if options_here:
                for kind_token, option_index, legal_ids, group_kind, _sort_key in options_here:
                    self._emit_blank(
                        buf,
                        result,
                        kind_token,
                        option_index,
                        legal_token_ids=legal_ids,
                        group_kind=group_kind,
                    )

    def _render_stack_object(
        self,
        buf: list[str],
        obj: StackObjectState,
        card_refs: dict[str, int],
    ) -> None:
        cid = obj.get("id", "")
        name = obj.get("name", "") or ""
        ref_idx = card_refs.get(cid)
        # Mirror _emit_stack_object exactly: spaces around the inner content.
        if ref_idx is not None:
            buf.append(f"<card> {card_ref_token(ref_idx)} {name} </card>")
        else:
            buf.append(f"<card> {name} </card>")

    # -- inline-blank helpers ---------------------------------------------

    def _classify_inline_options(
        self,
        actions: Sequence[PendingOptionState],
        card_refs: dict[str, int],
    ) -> None:
        """Bucket inline options into per-card blanks + a pass-only list.

        Every option is one of:

        - ``play`` / ``cast`` of a hand card → ``<choose-play>`` blank
          adjacent to the hand-card render position; targeted cast options
          also get a following ``<choose-target>`` blank.
        - ``activate`` of an ability → ``<use-ability>`` blank adjacent to
          the source permanent on the battlefield; targeted abilities also
          get a following ``<choose-target>`` blank.
        - ``pass`` → buffered for emission in the trailing ``<choices>``
          block.

        Per-card options are sorted by an engine-stable key so the blank
        ordinal layout is deterministic across re-renders, independent of
        Python dict iteration order. See "Stable blank numbering" in
        ``docs/text_encoder_inline_blanks_plan.md``.
        """

        chosen_id = self._chosen_token_id
        if chosen_id is None:
            raise RenderError("inline priority blanks require chosen_token_id")
        priority_legal_ids = (int(chosen_id),)
        per_card: dict[str, list[tuple[str, int, tuple[int, ...], str, tuple[object, ...]]]] = {}
        pass_options: list[int] = []
        for opt_idx, option in enumerate(actions):
            kind = (option.get("kind") or "").lower()
            if kind in ("play", "play_land", "cast", "cast_spell"):
                source = option.get("card_id") or option.get("permanent_id") or ""
                if not source:
                    raise RenderError(
                        f"play/cast option {opt_idx} ({kind!r}) has no card_id; "
                        "cannot anchor inline blank."
                    )
                # Stable key: (ability_index_or_-1, option-id-string, opt_idx).
                ability_idx = option.get("ability_index")
                key: tuple[object, ...] = (
                    0,
                    "<choose-play>",
                    -1 if ability_idx is None else int(ability_idx),
                    str(option.get("id") or ""),
                    opt_idx,
                )
                per_card.setdefault(source, []).append(
                    ("<choose-play>", opt_idx, priority_legal_ids, "CROSS_BLANK", key)
                )
                self._append_target_blank_option(per_card, source, opt_idx, option, card_refs)
            elif kind in ("activate", "activate_ability", "activated_ability"):
                source = option.get("permanent_id") or option.get("card_id") or ""
                if not source:
                    raise RenderError(
                        f"activate option {opt_idx} has no permanent_id / card_id; "
                        "cannot anchor inline blank."
                    )
                ability_idx = option.get("ability_index")
                key = (
                    0,
                    "<use-ability>",
                    -1 if ability_idx is None else int(ability_idx),
                    str(option.get("id") or ""),
                    opt_idx,
                )
                per_card.setdefault(source, []).append(
                    ("<use-ability>", opt_idx, priority_legal_ids, "CROSS_BLANK", key)
                )
                self._append_target_blank_option(per_card, source, opt_idx, option, card_refs)
            elif kind == "block":
                source = option.get("permanent_id") or option.get("card_id") or ""
                if not source:
                    raise RenderError(
                        f"block option {opt_idx} has no permanent_id / card_id; "
                        "cannot anchor inline blank."
                    )
                legal_ids = self._block_legal_token_ids(option, card_refs)
                key = (0, "<choose-block>", str(option.get("id") or ""), opt_idx)
                per_card.setdefault(source, []).append(
                    ("<choose-block>", opt_idx, legal_ids, "CONSTRAINED", key)
                )
            elif kind == "pass":
                pass_options.append(opt_idx)
            else:
                # Other kinds (mulligan, attack, block, choice, …) aren't
                # in scope for Step 2. Silently drop them; later steps add
                # their dedicated blank kinds. Crucially we don't raise:
                # the legacy path still runs in non-inline mode for them.
                continue

        # Sort each card's options by their stable key so the blank ordinal
        # ordering is independent of input-list permutation.
        for cid in per_card:
            per_card[cid].sort(key=lambda item: item[4])
        self._blank_options_by_card = per_card
        self._pass_options = pass_options

    def _append_target_blank_option(
        self,
        per_card: dict[str, list[tuple[str, int, tuple[int, ...], str, tuple[object, ...]]]],
        source: str,
        opt_idx: int,
        option: PendingOptionState,
        card_refs: dict[str, int],
    ) -> None:
        legal_ids = self._target_legal_token_ids(option, card_refs)
        if not legal_ids:
            return
        key = (1, "<choose-target>", str(option.get("id") or ""), opt_idx)
        per_card.setdefault(source, []).append(
            ("<choose-target>", opt_idx, legal_ids, "PER_BLANK", key)
        )

    def _block_legal_token_ids(
        self,
        option: PendingOptionState,
        card_refs: dict[str, int],
    ) -> tuple[int, ...]:
        none_id = self._none_token_id
        if none_id is None:
            raise RenderError("inline block blanks require none_token_id")
        legal_ids = [int(none_id)]
        for target in option.get("valid_targets") or []:
            tid = target.get("id", "")
            ref = card_refs.get(tid)
            if ref is None:
                continue
            if not 0 <= ref < len(self._card_ref_token_ids):
                raise RenderError(
                    f"block target card-ref:{ref} has no token id in card_ref_token_ids"
                )
            legal_ids.append(int(self._card_ref_token_ids[ref]))
        return tuple(legal_ids)

    def _target_legal_token_ids(
        self,
        option: PendingOptionState,
        card_refs: dict[str, int],
    ) -> tuple[int, ...]:
        legal_ids: list[int] = []
        for target in option.get("valid_targets") or []:
            tid = target.get("id", "")
            ref = card_refs.get(tid)
            if ref is None:
                continue
            if not 0 <= ref < len(self._card_ref_token_ids):
                raise RenderError(f"target card-ref:{ref} has no token id in card_ref_token_ids")
            legal_ids.append(int(self._card_ref_token_ids[ref]))
        return tuple(legal_ids)

    def _emit_blank(
        self,
        buf: list[str],
        result: RenderedSnapshot,
        kind_token: str,
        option_index: int,
        *,
        legal_token_ids: tuple[int, ...],
        group_kind: str,
    ) -> None:
        """Write ``kind_token`` to ``buf`` and record a ``BlankAnchor``.

        An empty legal set raises ``RenderError``; every inline blank must
        expose at least one legal answer token.
        """

        if not legal_token_ids:
            raise RenderError(f"blank {kind_token!r} for option {option_index} has empty legal set")
        char_start = sum(len(s) for s in buf)
        buf.append(kind_token)
        char_end = sum(len(s) for s in buf)
        result.blank_anchors.append(
            BlankAnchor(
                blank_index=self._blank_index,
                kind=kind_token,
                char_start=char_start,
                char_end=char_end,
                group_id=self._blank_group_id,
                group_kind=group_kind,
                legal_token_ids=legal_token_ids,
                option_index=option_index,
            )
        )
        self._blank_index += 1

    def _render_choices_inline(
        self,
        buf: list[str],
        result: RenderedSnapshot,
        actions: Sequence[PendingOptionState],
    ) -> None:
        """Emit the trailing ``<choices>`` block with the ``<pass>`` blank.

        Currently the choices zone carries priority-pass anchors and the
        Step-7 ``<choose-may>`` blank. Later steps add ``<choose-mode>`` /
        ``<choose-x>`` groups whose blanks live here too.
        """

        buf.append("<choices>")
        chosen_id = self._chosen_token_id
        if chosen_id is None:
            raise RenderError("inline pass blanks require chosen_token_id")
        priority_legal_ids = (int(chosen_id),)
        for opt_idx in self._pass_options:
            self._emit_blank(
                buf,
                result,
                "<pass>",
                opt_idx,
                legal_token_ids=priority_legal_ids,
                group_kind="CROSS_BLANK",
            )
        if self._pending_kind == "may":
            yes_id = self._yes_token_id
            no_id = self._no_token_id
            if yes_id is None or no_id is None:
                raise RenderError("inline may blanks require yes_token_id and no_token_id")
            self._emit_blank(
                buf,
                result,
                "<choose-may>",
                -1,
                legal_token_ids=(int(no_id), int(yes_id)),
                group_kind="PER_BLANK",
            )
        if self._pending_kind in ("mode", "number"):
            choice_count = len(actions)
            if choice_count < 1:
                raise RenderError(f"inline {self._pending_kind} blanks require at least one option")
            if choice_count > MAX_NUM:
                raise RenderError(
                    f"inline {self._pending_kind} option count {choice_count} exceeds MAX_NUM"
                )
            if len(self._num_token_ids) < choice_count:
                raise RenderError(f"inline {self._pending_kind} blanks require num_token_ids")
            self._emit_blank(
                buf,
                result,
                "<choose-mode>" if self._pending_kind == "mode" else "<choose-x-digit>",
                -1,
                legal_token_ids=tuple(self._num_token_ids[:choice_count]),
                group_kind="PER_BLANK",
            )
        if self._pending_kind == "mana_color":
            if len(self._mana_token_ids) < 6:
                raise RenderError("inline mana_color blanks require mana_token_ids")
            self._emit_blank(
                buf,
                result,
                "<choose-mana-source>",
                -1,
                legal_token_ids=tuple(self._mana_token_ids[:6]),
                group_kind="PER_BLANK",
            )
        buf.append("</choices>")


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------


def render_snapshot(
    snapshot: GameStateSnapshot,
    actions: Sequence[PendingOptionState] | None = None,
    *,
    oracle: dict[str, OracleEntry] | None = None,
    max_card_refs: int = MAX_CARD_REFS,
    chosen_token_id: int | None = None,
    none_token_id: int | None = None,
    yes_token_id: int | None = None,
    no_token_id: int | None = None,
    num_token_ids: Sequence[int] | None = None,
    mana_token_ids: Sequence[int] | None = None,
    card_ref_token_ids: Sequence[int] | None = None,
) -> RenderedSnapshot:
    """Render ``snapshot`` (and optional ``actions``) to text + anchor metadata.

    The ``oracle`` dict (loaded once via :func:`load_oracle_text`) supplies card
    type lines, mana costs, and oracle text. If a card name is missing from the
    dict the renderer falls back to emitting only the card name + status flags.

    Pending decisions are rendered as inline ``<choose-*>`` blanks adjacent to
    their natural source text plus a trailing ``<choices>`` block for
    standalone choices. Scoring callers should supply ``chosen_token_id`` (the
    tokenizer id of the ``<chosen>`` scoring token); render-only callers may
    omit it.
    """

    return SnapshotRenderer(
        oracle,
        max_card_refs=max_card_refs,
        chosen_token_id=chosen_token_id,
        none_token_id=none_token_id,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        num_token_ids=num_token_ids,
        mana_token_ids=mana_token_ids,
        card_ref_token_ids=card_ref_token_ids,
    ).render(snapshot, actions)
