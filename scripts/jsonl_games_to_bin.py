"""Convert recorded ``*.jsonl.gz`` game logs into uint16 ``.bin`` token streams.

Input shape: each ``.jsonl.gz`` file is one game produced by the recorder
under ``data/games/``. The first line is a ``META`` record; subsequent lines
are ``EVENT`` records with a structured ``snapshot`` field.

Output shape: one ``<gameId>.bin`` per input game, written into ``--out-dir``.
Each file is a flat ``uint16`` stream of tokenizer ids; one rendered
decision-point snapshot is appended per "real" decision (any ``PRIORITY``
event whose ``snapshot.playableActions`` is non-empty), wrapped in
``<bos>...<eos>`` so the result is the same shape
``magic_ai.text_encoder.mlm.BinTokenDataset`` expects.

When ``--with-value-labels`` is passed, an additional ``<gameId>.json``
sidecar is written alongside each ``.bin`` carrying ``winner_id``,
``players``, and a ``spans`` list of
``{offset, length, perspective_id, label, steps_to_end}`` per decision
point. ``label`` is the perspective-signed terminal outcome (``+1`` win,
``-1`` loss, ``0`` draw). ``steps_to_end`` counts decision points from
this span to the terminal so downstream consumers can apply a discount
``γ^steps_to_end`` and treat each span as a Monte-Carlo return-to-go from
that decision (matching the ``gae_returns`` convention in
``magic_ai/ppo.py`` with no bootstrap and λ=1). The bin half stays
byte-identical regardless of the flag, so the same artifact directory can
drive both pretraining phases.

Layout matches the live native Go encoder (``mage-go/cmd/pylib/
direct_token_emitter.go``) with ``dedupCardBodies=true``: a single
``<dict>...</dict>`` block at the top of the snapshot lists each unique card
body via ``<dict-entry:R>``, and each placement on the battlefield / hand /
graveyard / exile is emitted as ``<card-ref:K><card><dict-entry:R>[<tap>]
</card>`` referencing that body. Token-id sequences come from the same
``magic_ai.text_encoder.token_tables.build_token_tables`` tables the live
path registers with libmage, so the offline ``.bin`` is byte-for-byte
compatible with what the encoder sees during rollouts.

Usage:

    uv run python scripts/jsonl_games_to_bin.py \
        --in-dir data/games \
        --out-dir data/games_bin

    uv run python scripts/jsonl_games_to_bin.py \
        --in-dir data/games \
        --out-dir data/games_value_bin \
        --with-value-labels
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from magic_ai.text_encoder.card_cache import CardTokenCache, build_card_cache, load_oracle_db
from magic_ai.text_encoder.token_tables import (
    ACTION_VERBS_BY_ID,
    COUNT_MAX,
    COUNT_MIN,
    LIFE_MAX,
    LIFE_MIN,
    OWNER_NAMES,
    STEP_NAMES,
    TURN_MAX,
    TURN_MIN,
    Frag,
    TokenTables,
    build_token_tables,
)
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS, load_tokenizer


def _resolve_label(winner_id: str | None, priority_id: str) -> float:
    if not winner_id:
        return 0.0
    return 1.0 if winner_id == priority_id else -1.0


def _winner_from_snapshot(snap: dict[str, Any]) -> str | None:
    """Recover the winner from a terminal snapshot.

    The recorder's META ``winnerId`` is null on every game in the current
    corpus even when the game has a clear loser (e.g. life <= 0,
    ``hasLost=True``); META reports a draw whenever the engine doesn't
    set a winner explicitly. We instead trust the per-player
    ``hasWon`` / ``hasLost`` flags on the last snapshot, which are
    populated correctly by the engine. Returns the winning player id, or
    ``None`` if the snapshot is genuinely tied (both lost / neither
    flagged).
    """

    players = snap.get("players") or []
    won = [p for p in players if p.get("hasWon")]
    if len(won) == 1:
        return str(won[0].get("id") or "") or None
    lost = [p for p in players if p.get("hasLost")]
    survivors = [p for p in players if not p.get("hasLost")]
    if len(lost) == 1 and len(survivors) == 1:
        return str(survivors[0].get("id") or "") or None
    return None


# --- recorder field normalization --------------------------------------------

_STEP_NAME_MAP: dict[str, str] = {
    "UNTAP": "Untap",
    "UPKEEP": "Upkeep",
    "DRAW": "Draw",
    "PRECOMBAT_MAIN": "Precombat Main",
    "BEGIN_COMBAT": "Begin Combat",
    "BEGINNING_OF_COMBAT": "Begin Combat",
    "DECLARE_ATTACKERS": "Declare Attackers",
    "DECLARE_BLOCKERS": "Declare Blockers",
    "FIRST_STRIKE_DAMAGE": "Combat Damage",
    "COMBAT_DAMAGE": "Combat Damage",
    "END_COMBAT": "End Combat",
    "END_OF_COMBAT": "End Combat",
    "POSTCOMBAT_MAIN": "Postcombat Main",
    "END": "End",
    "END_TURN": "End",
    "CLEANUP": "Cleanup",
}


def _step_id(raw: str | None) -> int:
    """Index into ``STEP_NAMES`` for a recorder step string. Falls back to "Unknown"."""

    if not raw:
        return len(STEP_NAMES) - 1
    canon = _STEP_NAME_MAP.get(raw.upper(), "")
    if not canon:
        return len(STEP_NAMES) - 1
    try:
        return STEP_NAMES.index(canon)
    except ValueError:
        return len(STEP_NAMES) - 1


_POOL_GLYPH_TO_COLOR_ID: dict[str, int] = {"W": 0, "U": 1, "B": 2, "R": 3, "G": 4, "C": 5}


def _parse_mana_pool(pool: str | None) -> list[int]:
    """Return a per-color amount list keyed by ``MANA_SYMBOLS`` order."""

    out = [0] * 6
    if not pool:
        return out
    for ch in pool:
        i = _POOL_GLYPH_TO_COLOR_ID.get(ch.upper())
        if i is not None:
            out[i] += 1
    return out


# Recorder ``Kind`` strings → ``ACTION_VERBS_BY_ID`` ids. Mirrors the Go
# ``actionKinds = [pass, play_land, cast_spell, activate_ability, attacker,
# blocker, choice, unknown]`` plus normalization (case-insensitive, common
# aliases collapsed).
_KIND_TO_ID: dict[str, int] = {
    "PASS": 0,
    "PASS_PRIORITY": 0,
    "PLAY_LAND": 1,
    "PLAY": 1,
    "CAST": 2,
    "CAST_SPELL": 2,
    "ACTIVATE": 3,
    "ACTIVATE_ABILITY": 3,
    "ACTIVATED_ABILITY": 3,
    "ACTIVATE_MANA": 3,
    "MANA_ABILITY": 3,
    "ATTACK": 4,
    "DECLARE_ATTACK": 4,
    "ATTACKER": 4,
    "BLOCK": 5,
    "DECLARE_BLOCK": 5,
    "BLOCKER": 5,
    "CHOICE": 6,
    "MULLIGAN": 6,
    "KEEP": 6,
}


def _kind_id(raw: str | None) -> int:
    if not raw:
        return 7  # unknown
    return _KIND_TO_ID.get(raw.upper(), 7)


def _kind_has_no_source(kind_id: int) -> bool:
    return kind_id == 0 or kind_id == 6 or kind_id >= len(ACTION_VERBS_BY_ID)


# --- snapshot iteration -------------------------------------------------------


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _is_decision(rec: dict[str, Any]) -> bool:
    if rec.get("type") != "PRIORITY":
        return False
    snap = rec.get("snapshot") or {}
    return bool(snap.get("playableActions"))


def _card_name(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return str(c.get("name") or c.get("Name") or "")
    return ""


def _card_id(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("id") or c.get("ID") or "")
    return ""


# --- card-row resolution -----------------------------------------------------


def collect_card_names(paths: Iterable[Path]) -> set[str]:
    """First pass: every card name observed anywhere in the corpus.

    Includes battlefield / hand / graveyard / exile occupants and option /
    target source names. The result feeds ``build_card_cache`` so we have a
    body span available for every name the emitter might reference.
    """

    names: set[str] = set()
    for path in paths:
        for rec in _iter_records(path):
            if rec.get("record") != "EVENT":
                continue
            snap = rec.get("snapshot") or {}
            for card in snap.get("battlefield") or []:
                n = _card_name(card)
                if n:
                    names.add(n)
            for player in snap.get("players") or []:
                for zone_attr in ("hand", "graveyard", "exile"):
                    for card in player.get(zone_attr) or []:
                        n = _card_name(card)
                        if n:
                            names.add(n)
            for opt in snap.get("playableActions") or []:
                n = str(opt.get("sourceName") or "")
                if n:
                    names.add(n)
                for tgt in opt.get("targets") or []:
                    label = str(tgt.get("label") or "")
                    if label:
                        names.add(label)
    return names


# --- emitter -----------------------------------------------------------------

# Zone ids mirror ``ZONE_TAGS_BY_ID`` in token_tables.py.
_ZONE_HAND = 0
_ZONE_BATTLEFIELD = 1
_ZONE_GRAVEYARD = 2
_ZONE_EXILE = 3
_ZONE_LIBRARY = 4

_OWNER_SELF = 0
_OWNER_OPP = 1


@dataclass
class _PlacedCard:
    zone: int
    owner: int
    card_id: str
    name: str
    row: int
    uuid_idx: int  # -1 if the card has no engine id
    tapped_known: bool
    tapped: bool


class _DirectEmitter:
    """Python port of ``direct_token_emitter.go`` with ``dedupCardBodies=True``."""

    def __init__(self, tables: TokenTables, name_to_row: dict[str, int]) -> None:
        self.tables = tables
        self.name_to_row = name_to_row
        self.out: list[int] = []
        self._scalar_owner_open: int = -1  # -1, 0=self, 1=opp
        self._option_open = False

    # -- low-level writers -------------------------------------------------

    def _write(self, span: list[int]) -> None:
        self.out.extend(span)

    def _write_one(self, tok_id: int) -> None:
        self.out.append(tok_id)

    def _emit_frag(self, frag: Frag) -> None:
        span = self.tables.structural.get(frag)
        if span:
            self._write(span)

    def _close_scalar_owner(self) -> None:
        if self._scalar_owner_open == _OWNER_SELF:
            self._emit_frag(Frag.CLOSE_SELF)
        elif self._scalar_owner_open == _OWNER_OPP:
            self._emit_frag(Frag.CLOSE_OPP)
        self._scalar_owner_open = -1

    def _close_option(self) -> None:
        if self._option_open:
            self._emit_frag(Frag.CLOSE_OPTION)
            self._option_open = False

    # -- per-snapshot driver ----------------------------------------------

    def emit(
        self,
        *,
        turn: int,
        step_id: int,
        self_life: int,
        opp_life: int,
        self_mana: list[int],
        opp_mana: list[int],
        cards_by_zone: dict[tuple[int, int], list[_PlacedCard]],
        row_order: list[int],
        self_lib: int,
        opp_lib: int,
        actions: list[dict[str, Any]],
        target_self_id: str,
        target_opp_id: str,
        card_id_to_uuid_idx: dict[str, int],
    ) -> None:
        # <bos><state>
        self._emit_frag(Frag.BOS_STATE)

        # <dict> ... </dict>
        if row_order:
            self._write_one(self.tables.dict_open_id)
            for row in row_order:
                if 0 <= row < len(self.tables.dict_entry):
                    self._write_one(self.tables.dict_entry[row])
                if 0 <= row < len(self.tables.card_body):
                    self._write(self.tables.card_body[row])
                self._write(self.tables.card_closer)
            self._write_one(self.tables.dict_close_id)

        # turn × step
        clamped_turn = max(TURN_MIN, min(TURN_MAX, turn))
        ts = self.tables.turn_step.get((clamped_turn, step_id))
        if ts is not None:
            self._write(ts)

        # life + mana per owner
        for owner_id, life, pool in (
            (_OWNER_SELF, self_life, self_mana),
            (_OWNER_OPP, opp_life, opp_mana),
        ):
            self._emit_life(owner_id, life)
            for color_id, amount in enumerate(pool):
                if amount > 0:
                    self._emit_mana(owner_id, color_id, amount)
        self._close_scalar_owner()

        # zones (zone-outer, owner-inner) — battlefield, hand (self only),
        # graveyard, exile (skip empty), library, stack
        for zone in (_ZONE_BATTLEFIELD, _ZONE_HAND, _ZONE_GRAVEYARD):
            for owner in (_OWNER_SELF, _OWNER_OPP):
                if zone == _ZONE_HAND and owner == _OWNER_OPP:
                    continue  # fog of war
                self._emit_zone(zone, owner, cards_by_zone.get((zone, owner), []))
        for owner in (_OWNER_SELF, _OWNER_OPP):
            cards = cards_by_zone.get((_ZONE_EXILE, owner), [])
            if cards:
                self._emit_zone(_ZONE_EXILE, owner, cards)
        for owner, lib in ((_OWNER_SELF, self_lib), (_OWNER_OPP, opp_lib)):
            self._emit_open_zone(_ZONE_LIBRARY, owner)
            self._emit_count(lib)
            self._emit_close_zone(_ZONE_LIBRARY, owner)

        # stack: open + close, no contents (matches Go direct emitter)
        self._close_scalar_owner()
        self._write_one(self.tables.stack_open_id)
        self._write_one(self.tables.stack_close_id)

        # actions
        self._emit_actions(actions, card_id_to_uuid_idx, target_self_id, target_opp_id)

        # </state><eos>
        self._close_scalar_owner()
        self._close_option()
        self._emit_frag(Frag.CLOSE_STATE_EOS)

    # -- helpers ----------------------------------------------------------

    def _emit_life(self, owner_id: int, life: int) -> None:
        self._close_scalar_owner()
        clamped = max(LIFE_MIN, min(LIFE_MAX, life))
        span = self.tables.life_owner.get((clamped, owner_id))
        if span is not None:
            self._write(span)
        self._scalar_owner_open = owner_id

    def _emit_mana(self, owner_id: int, color_id: int, amount: int) -> None:
        if self._scalar_owner_open >= 0 and self._scalar_owner_open != owner_id:
            self._close_scalar_owner()
        if self._scalar_owner_open < 0:
            self._emit_frag(Frag.SELF_MANA if owner_id == _OWNER_SELF else Frag.OPP_MANA)
            self._scalar_owner_open = owner_id
        if 0 <= color_id < len(self.tables.mana_glyph) and amount > 0:
            glyph = self.tables.mana_glyph[color_id]
            for _ in range(amount):
                self._write(glyph)

    def _emit_open_zone(self, zone: int, owner: int) -> None:
        self._close_scalar_owner()
        self._close_option()
        span = self.tables.zone_open.get((zone, owner))
        if span is not None:
            self._write(span)

    def _emit_close_zone(self, zone: int, owner: int) -> None:
        self._close_scalar_owner()
        self._close_option()
        span = self.tables.zone_close.get((zone, owner))
        if span is not None:
            self._write(span)

    def _emit_zone(self, zone: int, owner: int, cards: list[_PlacedCard]) -> None:
        self._emit_open_zone(zone, owner)
        for card in cards:
            self._emit_place_card_ref(card)
        self._emit_close_zone(zone, owner)

    def _emit_place_card_ref(self, card: _PlacedCard) -> None:
        self._close_scalar_owner()
        if 0 <= card.uuid_idx < min(MAX_CARD_REFS, len(self.tables.card_ref)):
            self._write_one(self.tables.card_ref[card.uuid_idx])
        self._write_one(self.tables.card_open_id)
        if 0 <= card.row < len(self.tables.dict_entry):
            self._write_one(self.tables.dict_entry[card.row])
        self._emit_status(card)
        self._write(self.tables.card_closer)

    def _emit_status(self, card: _PlacedCard) -> None:
        if card.tapped_known:
            self._write(self.tables.status_tapped if card.tapped else self.tables.status_untapped)
        elif card.tapped:
            self._write(self.tables.status_tapped)

    def _emit_count(self, amount: int) -> None:
        self._close_scalar_owner()
        clamped = max(COUNT_MIN, min(COUNT_MAX, amount))
        span = self.tables.count.get(clamped)
        if span is not None:
            self._write(span)

    def _emit_actions(
        self,
        actions: list[dict[str, Any]],
        card_id_to_uuid_idx: dict[str, int],
        target_self_id: str,
        target_opp_id: str,
    ) -> None:
        self._close_scalar_owner()
        self._emit_frag(Frag.OPEN_ACTIONS)
        for opt in actions:
            self._close_scalar_owner()
            self._close_option()
            self._write_one(self.tables.option_id)
            self._option_open = True

            kind_id = _kind_id(opt.get("kind"))
            verb_span = self.tables.action_verb.get(kind_id)
            kind_known = bool(verb_span)
            if verb_span:
                self._write(verb_span)
            if kind_known and not _kind_has_no_source(kind_id):
                source_id = str(opt.get("sourceId") or "")
                source_name = str(opt.get("sourceName") or "")
                uuid_idx = card_id_to_uuid_idx.get(source_id, -1)
                if 0 <= uuid_idx < min(MAX_CARD_REFS, len(self.tables.card_ref)):
                    self._write_one(self.tables.card_ref[uuid_idx])
                else:
                    row = self.name_to_row.get(source_name, -1)
                    if 0 <= row < len(self.tables.card_name):
                        self._write(self.tables.card_name[row])
            ability_idx = opt.get("abilityIndex")
            if kind_id == 3 and isinstance(ability_idx, int) and ability_idx >= 0:
                ab = self.tables.ability.get(ability_idx)
                if ab is not None:
                    self._write(ab)

            for tgt in opt.get("targets") or []:
                self._close_scalar_owner()
                self._write_one(self.tables.target_open_id)
                tid = str(tgt.get("id") or "")
                if tid and tid == target_self_id:
                    self._write_one(self.tables.self_id)
                elif tid and tid == target_opp_id:
                    self._write_one(self.tables.opp_id)
                else:
                    uuid_idx = card_id_to_uuid_idx.get(tid, -1)
                    if 0 <= uuid_idx < min(MAX_CARD_REFS, len(self.tables.card_ref)):
                        self._write_one(self.tables.card_ref[uuid_idx])
                    else:
                        label = str(tgt.get("label") or "")
                        row = self.name_to_row.get(label, -1)
                        if 0 <= row < len(self.tables.card_name):
                            self._write(self.tables.card_name[row])
                        else:
                            self._emit_frag(Frag.TARGET_FALLBACK)
                self._write_one(self.tables.target_close_id)

        self._close_scalar_owner()
        self._close_option()
        self._emit_frag(Frag.CLOSE_ACTIONS)


# --- snapshot → emitter args -------------------------------------------------


def _build_placement_index(
    snap: dict[str, Any],
    perspective: int,
    name_to_row: dict[str, int],
) -> tuple[dict[tuple[int, int], list[_PlacedCard]], list[int], dict[str, int]]:
    """Return ``(cards_by_zone, row_order, card_id_to_uuid_idx)``.

    UUID indices are assigned in the same order as the live path's
    ``buildRenderPlanIndex``: zone-outer (battlefield, hand, graveyard, exile)
    × owner-inner (self, opp). Row order is sorted ascending — matches the
    Go ``slices.Sort(index.rowOrder)``.
    """

    players = snap.get("players") or []
    if not players:
        return {}, [], {}
    self_p = players[perspective] if perspective < len(players) else None
    opp_p = players[1 - perspective] if len(players) > 1 else None
    self_id = str((self_p or {}).get("id") or "")
    opp_id = str((opp_p or {}).get("id") or "")

    bf_by_owner: dict[int, list[Any]] = {_OWNER_SELF: [], _OWNER_OPP: []}
    for card in snap.get("battlefield") or []:
        ctrl = str((card or {}).get("controllerId") or "")
        if ctrl == self_id:
            bf_by_owner[_OWNER_SELF].append(card)
        elif ctrl == opp_id:
            bf_by_owner[_OWNER_OPP].append(card)

    def zone_cards(player: dict[str, Any] | None, attr: str) -> list[Any]:
        if not player:
            return []
        return list(player.get(attr) or [])

    cards_by_zone: dict[tuple[int, int], list[_PlacedCard]] = {}
    card_id_to_uuid_idx: dict[str, int] = {}
    rows_seen: set[int] = set()
    next_uuid = 0

    # _ZONE_ORDER: bf×{self,opp}, hand×{self,opp}, gy×{self,opp}, exile×{self,opp}
    walk = [
        (_ZONE_BATTLEFIELD, _OWNER_SELF, bf_by_owner[_OWNER_SELF], True),
        (_ZONE_BATTLEFIELD, _OWNER_OPP, bf_by_owner[_OWNER_OPP], True),
        (_ZONE_HAND, _OWNER_SELF, zone_cards(self_p, "hand"), False),
        (_ZONE_HAND, _OWNER_OPP, zone_cards(opp_p, "hand"), False),
        (_ZONE_GRAVEYARD, _OWNER_SELF, zone_cards(self_p, "graveyard"), False),
        (_ZONE_GRAVEYARD, _OWNER_OPP, zone_cards(opp_p, "graveyard"), False),
        (_ZONE_EXILE, _OWNER_SELF, zone_cards(self_p, "exile"), False),
        (_ZONE_EXILE, _OWNER_OPP, zone_cards(opp_p, "exile"), False),
    ]

    for zone, owner, raw_cards, is_battlefield in walk:
        bucket: list[_PlacedCard] = []
        for raw in raw_cards:
            name = _card_name(raw)
            cid = _card_id(raw)
            row = name_to_row.get(name, 0)  # row 0 is unknown sentinel
            if cid:
                uuid_idx = card_id_to_uuid_idx.get(cid)
                if uuid_idx is None and next_uuid < MAX_CARD_REFS:
                    uuid_idx = next_uuid
                    next_uuid += 1
                    card_id_to_uuid_idx[cid] = uuid_idx
                if uuid_idx is None:
                    uuid_idx = -1
            else:
                uuid_idx = -1
            tapped_known = bool(is_battlefield and isinstance(raw, dict))
            tapped = bool(isinstance(raw, dict) and raw.get("tapped"))
            bucket.append(
                _PlacedCard(
                    zone=zone,
                    owner=owner,
                    card_id=cid,
                    name=name,
                    row=row,
                    uuid_idx=uuid_idx,
                    tapped_known=tapped_known,
                    tapped=tapped,
                )
            )
            if row > 0:
                rows_seen.add(row)
        cards_by_zone[(zone, owner)] = bucket

    return cards_by_zone, sorted(rows_seen), card_id_to_uuid_idx


# --- per-file conversion ------------------------------------------------------


def convert_one(
    in_path: Path,
    out_dir: Path,
    tables: TokenTables,
    name_to_row: dict[str, int],
    vocab_size: int,
    *,
    with_value_labels: bool = False,
) -> tuple[Path, int, int]:
    game_id: str | None = None
    meta_winner_id: str | None = None
    last_snapshot: dict[str, Any] | None = None
    players_meta: list[dict[str, str]] = []
    token_chunks: list[np.ndarray] = []
    span_records: list[dict[str, Any]] = []  # offset/length/perspective_id; label filled after
    cursor = 0
    n_decisions = 0

    for rec in _iter_records(in_path):
        if rec.get("record") == "META":
            game_id = rec.get("gameId")
            if with_value_labels:
                raw_winner = rec.get("winnerId")
                meta_winner_id = str(raw_winner) if raw_winner else None
                for p in rec.get("players") or []:
                    players_meta.append(
                        {
                            "id": str(p.get("id") or ""),
                            "name": str(p.get("name") or ""),
                        }
                    )
            continue
        if rec.get("record") != "EVENT":
            continue
        if with_value_labels:
            evt_snap = rec.get("snapshot")
            if isinstance(evt_snap, dict):
                last_snapshot = evt_snap
        if not _is_decision(rec):
            continue
        snap = rec["snapshot"]
        players = snap.get("players") or []
        priority_id = snap.get("priorityPlayerId")
        if with_value_labels and not priority_id:
            continue
        perspective = 0
        for idx, p in enumerate(players):
            if p.get("id") == priority_id:
                perspective = idx
                break
        self_p = players[perspective] if perspective < len(players) else {}
        opp_p = players[1 - perspective] if len(players) > 1 else {}

        cards_by_zone, row_order, card_id_to_uuid_idx = _build_placement_index(
            snap, perspective, name_to_row
        )

        emitter = _DirectEmitter(tables, name_to_row)
        emitter.emit(
            turn=int(snap.get("turn", 0) or 0),
            step_id=_step_id(snap.get("step")),
            self_life=int(self_p.get("life", 0) or 0),
            opp_life=int(opp_p.get("life", 0) or 0),
            self_mana=_parse_mana_pool(self_p.get("manaPool")),
            opp_mana=_parse_mana_pool(opp_p.get("manaPool")),
            cards_by_zone=cards_by_zone,
            row_order=row_order,
            self_lib=int(self_p.get("librarySize", 0) or 0),
            opp_lib=int(opp_p.get("librarySize", 0) or 0),
            actions=list(snap.get("playableActions") or []),
            target_self_id=str(self_p.get("id") or ""),
            target_opp_id=str(opp_p.get("id") or ""),
            card_id_to_uuid_idx=card_id_to_uuid_idx,
        )

        ids = emitter.out
        if vocab_size and any(i >= vocab_size or i > 65535 for i in ids):
            raise ValueError(
                f"token id out of range (vocab_size={vocab_size}) for {in_path.name};"
                " refusing to truncate to uint16"
            )
        chunk = np.asarray(ids, dtype=np.uint16)
        if with_value_labels:
            span_records.append(
                {
                    "offset": int(cursor),
                    "length": int(chunk.shape[0]),
                    "perspective_id": str(priority_id),
                }
            )
            cursor += int(chunk.shape[0])
        token_chunks.append(chunk)
        n_decisions += 1

    if game_id is None:
        game_id = in_path.stem.split(".")[0]
    out_path = out_dir / f"{game_id}.bin"
    flat = np.concatenate(token_chunks) if token_chunks else np.empty((0,), dtype=np.uint16)
    flat.astype(np.uint16, copy=False).tofile(out_path)
    if with_value_labels:
        winner_id = meta_winner_id
        if winner_id is None and last_snapshot is not None:
            winner_id = _winner_from_snapshot(last_snapshot)
        n_dec = len(span_records)
        spans = [
            {
                **rec,
                "label": _resolve_label(winner_id, str(rec["perspective_id"])),
                "steps_to_end": int(n_dec - 1 - i),
            }
            for i, rec in enumerate(span_records)
        ]
        sidecar = {
            "game_id": game_id,
            "winner_id": winner_id,
            "is_draw": winner_id is None,
            "players": players_meta,
            "spans": spans,
        }
        out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return out_path, n_decisions, int(flat.shape[0])


# --- worker for multiprocessing ----------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(
    tokenizer_dir: str,
    cache_path: str,
    vocab_size: int,
    name_to_row: dict[str, int],
    with_value_labels: bool = False,
) -> None:
    tokenizer = load_tokenizer(tokenizer_dir)
    cache = _load_or_build_cache(cache_path, tokenizer, list(name_to_row.keys()))
    tables = build_token_tables(tokenizer, cache=cache)
    _WORKER["tables"] = tables
    _WORKER["name_to_row"] = name_to_row
    _WORKER["vocab_size"] = vocab_size
    _WORKER["with_value_labels"] = with_value_labels


def _worker_convert(args: tuple[str, str]) -> tuple[str, int, int]:
    in_path, out_dir = args
    out_path, n_dec, n_tok = convert_one(
        Path(in_path),
        Path(out_dir),
        _WORKER["tables"],
        _WORKER["name_to_row"],
        _WORKER["vocab_size"],
        with_value_labels=bool(_WORKER.get("with_value_labels", False)),
    )
    return str(out_path), n_dec, n_tok


# --- cache helpers -----------------------------------------------------------


def _load_or_build_cache(
    cache_path: str | None,
    tokenizer: Any,
    names: list[str],
) -> CardTokenCache:
    """Build a cache for ``names``. ``cache_path`` is reserved for future
    on-disk caching; for now we always rebuild in-process so a renamed corpus
    can't pick up a stale cache."""

    _ = cache_path
    oracle = load_oracle_db(names=names)
    return build_card_cache(
        registered_names=names,
        oracle=oracle,
        tokenizer=tokenizer,
        missing_policy="warn",
    )


# --- CLI ---------------------------------------------------------------------


def _iter_inputs(in_dir: Path) -> Iterable[Path]:
    return sorted(in_dir.glob("*.jsonl.gz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("data/games"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/games_bin"))
    parser.add_argument("--tokenizer-dir", type=str, default="data/text_encoder_tokenizer")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = all).")
    parser.add_argument(
        "--with-value-labels",
        action="store_true",
        help="also write a <gameId>.json sidecar with winner_id, players, and "
        "per-span perspective-signed labels for value-head pretraining.",
    )
    args = parser.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = list(_iter_inputs(in_dir))
    if args.limit:
        inputs = inputs[: args.limit]
    if not inputs:
        print(f"no *.jsonl.gz files found under {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"scanning {len(inputs)} files for card names...", flush=True)
    names_set = collect_card_names(inputs)
    names = sorted(names_set)
    print(f"  {len(names)} unique card names", flush=True)

    print("building tokenizer + card cache + token tables...", flush=True)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    cache = _load_or_build_cache(None, tokenizer, names)
    tables = build_token_tables(tokenizer, cache=cache)
    vocab_size = len(tokenizer)

    # ``cache.row_to_name[0]`` is the unknown sentinel; rows 1..N map to names.
    name_to_row: dict[str, int] = {}
    for row, n in enumerate(cache.row_to_name):
        if row == 0:
            continue
        name_to_row[n] = row

    print(f"converting {len(inputs)} files -> {out_dir} (workers={args.workers})", flush=True)

    if args.workers <= 1:
        total_dec = 0
        total_tok = 0
        for i, p in enumerate(inputs):
            out_path, n_dec, n_tok = convert_one(
                p,
                out_dir,
                tables,
                name_to_row,
                vocab_size,
                with_value_labels=bool(args.with_value_labels),
            )
            total_dec += n_dec
            total_tok += n_tok
            if (i + 1) % 50 == 0 or i == len(inputs) - 1:
                print(
                    f"[{i + 1}/{len(inputs)}] wrote {out_path} (decisions={n_dec}, tokens={n_tok})",
                    flush=True,
                )
        print(
            f"done. {len(inputs)} files, {total_dec} decisions, {total_tok} tokens",
            flush=True,
        )
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(
            args.tokenizer_dir,
            "",
            vocab_size,
            name_to_row,
            bool(args.with_value_labels),
        ),
    ) as pool:
        total_dec = 0
        total_tok = 0
        tasks = [(str(p), str(out_dir)) for p in inputs]
        for i, (out_path, n_dec, n_tok) in enumerate(
            pool.imap_unordered(_worker_convert, tasks, chunksize=4)
        ):
            total_dec += n_dec
            total_tok += n_tok
            if (i + 1) % 50 == 0 or i == len(inputs) - 1:
                print(
                    f"[{i + 1}/{len(inputs)}] wrote {out_path} (decisions={n_dec}, tokens={n_tok})",
                    flush=True,
                )
        print(
            f"done. {len(inputs)} files, {total_dec} decisions, {total_tok} tokens",
            flush=True,
        )


# Silence the unused-import lints for symbols only referenced from build paths.
_ = OWNER_NAMES


if __name__ == "__main__":
    main()
