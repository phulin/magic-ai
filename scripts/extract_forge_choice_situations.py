"""Extract reloadable choice-training situations from a Forge game archive.

The Forge JSONL logs do not currently carry `snapshot.playableActions`, so this
extractor stores a self-contained pre-choice snapshot, rendered token ids, the
observed post-choice event, and terminal outcome metadata. A future training
dataset can reconstruct legal inline-blank candidates at load time and map the
stored observed choice onto its policy target while choosing the value target
construction (terminal sign, discounted return, PPO/GAE-style target, etc.) at
runtime.

Output format defaults to a directory of sharded PyTorch ``part-*.pt`` files.
Gzip JSONL is still available for inspection/debugging by choosing a
``.jsonl.gz`` output path. By default, two situations are selected per game.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import orjson
import torch

# Allow direct invocation as ``uv run python scripts/extract_forge_choice_situations.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.game_state import MANA_COLORS, GameStateSnapshot
from magic_ai.text_encoder.render import RenderError, load_oracle_text, render_snapshot
from magic_ai.text_encoder.tokenizer import (
    MANA_TOKENS,
    load_tokenizer,
)

ChoiceKind = Literal["priority", "attack", "block", "may", "choose"]
OutputFormat = Literal["torch_shards", "jsonl.gz"]

FORMAT_VERSION = 2
DEFAULT_KIND_PRIORITY: tuple[ChoiceKind, ...] = ("may", "block", "attack", "choose", "priority")
CHOICE_KINDS: tuple[ChoiceKind, ...] = ("priority", "attack", "block", "may", "choose")

_ATTACKS_HEADER_RE = re.compile(
    r"^(?P<attacker_player>.+?) attacks (?P<defender_player>.+?) with (?P<count>\d+) creatures?$"
)
# Single attacker line emitted during DECLARE_BLOCKERS step:
#   "Attacker: <name> [<id3>] (<P>/<T>) unblocked"
#   "Attacker: <name> [<id3>] (<P>/<T>) blocked by <name> [<id3>] (<P>/<T>) ..."
_ATTACKER_LINE_RE = re.compile(
    r"^Attacker:\s+(?P<name>.+?)\s+\[(?P<id>[0-9a-f]+)\]\s+\((?P<pt>[^)]+)\)\s+(?P<rest>.+)$"
)
_BLOCKER_TOKEN_RE = re.compile(r"(?P<name>.+?)\s+\[(?P<id>[0-9a-f]+)\]\s+\((?P<pt>[^)]+)\)")
_PLAYER_PREFIX_RE = re.compile(r"^(?P<player>Player[A-Z0-9_ -]+)")


@dataclass(frozen=True)
class _ChoiceCandidate:
    kind: ChoiceKind
    game_id: str
    archive_member: str
    source_seq: int
    target_seq: int
    perspective_id: str
    perspective_name: str
    snapshot: dict[str, Any]
    token_ids: list[int]
    text: str
    observed: dict[str, Any]
    candidate_index: int = 0


@dataclass(frozen=True)
class _GameMeta:
    game_id: str
    winner_id: str | None
    winner_name: str | None
    players: list[dict[str, Any]]
    extras: dict[str, Any]


def _loads(line: bytes) -> dict[str, Any]:
    return cast(dict[str, Any], orjson.loads(line))


def _dumps_text(value: Any) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def _iter_zip_jsonl(zip_path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".jsonl.gz"):
                continue
            rows: list[dict[str, Any]] = []
            with gzip.open(zf.open(name), "rb") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = _loads(line)
                    record = row.get("record")
                    if record == "META" or (
                        record == "EVENT" and isinstance(row.get("snapshot"), dict)
                    ):
                        rows.append(row)
            yield name, rows


def _iter_dir_jsonl(dir_path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    for path in sorted(dir_path.glob("*.jsonl.gz")):
        rows: list[dict[str, Any]] = []
        with gzip.open(path, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = _loads(line)
                record = row.get("record")
                if record == "META" or (
                    record == "EVENT" and isinstance(row.get("snapshot"), dict)
                ):
                    rows.append(row)
        yield path.name, rows


def _iter_jsonl(input_path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    if input_path.is_dir():
        yield from _iter_dir_jsonl(input_path)
    else:
        yield from _iter_zip_jsonl(input_path)


def _meta_from_rows(rows: Sequence[dict[str, Any]]) -> _GameMeta | None:
    for row in rows:
        if row.get("record") != "META":
            continue
        return _GameMeta(
            game_id=str(row.get("gameId") or ""),
            winner_id=str(row["winnerId"]) if row.get("winnerId") else None,
            winner_name=str(row["winnerName"]) if row.get("winnerName") else None,
            players=list(row.get("players") or []),
            extras=dict(row.get("extras") or {}),
        )
    return None


def _mana_pool(raw: str | None) -> dict[str, int]:
    out = {color: 0 for color in MANA_COLORS}
    if not raw:
        return out
    glyph_to_color = {
        "W": "White",
        "U": "Blue",
        "B": "Black",
        "R": "Red",
        "G": "Green",
        "C": "Colorless",
    }
    for ch in raw:
        color = glyph_to_color.get(ch.upper())
        if color is not None:
            out[color] += 1
    return out


def _zone_card(raw: Any, owner_id: str, zone: str, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        return {
            "ID": f"{owner_id}:{zone}:{index}:{raw}",
            "Name": raw,
        }
    if isinstance(raw, dict):
        name = str(raw.get("name") or raw.get("Name") or "")
        cid = str(raw.get("id") or raw.get("ID") or f"{owner_id}:{zone}:{index}:{name}")
        out: dict[str, Any] = {
            "ID": cid,
            "Name": name,
        }
        if "tapped" in raw:
            out["Tapped"] = bool(raw.get("tapped"))
        elif "Tapped" in raw:
            out["Tapped"] = bool(raw.get("Tapped"))
        return out
    return {"ID": f"{owner_id}:{zone}:{index}:unknown", "Name": ""}


def _player(raw: dict[str, Any]) -> dict[str, Any]:
    pid = str(raw.get("id") or raw.get("ID") or "")
    hand = [_zone_card(card, pid, "hand", i) for i, card in enumerate(raw.get("hand") or [])]
    graveyard = [
        _zone_card(card, pid, "graveyard", i) for i, card in enumerate(raw.get("graveyard") or [])
    ]
    exile = [_zone_card(card, pid, "exile", i) for i, card in enumerate(raw.get("exile") or [])]
    return {
        "ID": pid,
        "Name": str(raw.get("name") or raw.get("Name") or ""),
        "Life": int(raw.get("life") or raw.get("Life") or 0),
        "HandCount": int(raw.get("handSize") or len(hand)),
        "GraveyardCount": len(graveyard),
        "LibraryCount": int(raw.get("librarySize") or raw.get("LibraryCount") or 0),
        "Hand": hand,
        "Graveyard": graveyard,
        "Exile": exile,
        "ManaPool": _mana_pool(raw.get("manaPool")),
    }


def _step(raw: str | None) -> str:
    if not raw:
        return "Unknown"
    mapping = {
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
        "POSTCOMBAT_MAIN": "Postcombat Main",
        "END": "End",
        "END_TURN": "End",
        "CLEANUP": "Cleanup",
    }
    return mapping.get(raw.upper(), raw)


_FORGE_KIND_TO_OPTION_KIND: dict[str, str] = {
    # Legacy zip schema (`snapshot.playableActions[].kind`).
    "PLAY_LAND": "play",
    "CAST_SPELL": "cast",
    "ACTIVATE": "activate",
    "ACTIVATE_MANA": "activate",
    # Current Forge schema (`snapshot.playableActions[].type`).
    "LAND": "play",
    "SPELL": "cast",
    "ACTIVATED": "activate",
}


def _priority_pending_from_playable_actions(
    raw_snapshot: dict[str, Any], perspective_id: str
) -> dict[str, Any] | None:
    """Build a ``PendingState``-shaped dict for priority decisions.

    Forge logs each event with ``snapshot.playableActions``: a list of
    {abilityId, controllerId, sourceId, sourceName, kind, cost, ...} dicts
    for the player whose priority it is. We map those to
    ``PendingOptionState`` entries so the V2 decoder-target translator can
    score the observed pick.
    """

    actions = raw_snapshot.get("playableActions") or []
    priority_id = str(raw_snapshot.get("priorityPlayerId") or "")
    perspective_idx = 0 if perspective_id and priority_id == perspective_id else 0
    options: list[dict[str, Any]] = []
    for i, act in enumerate(actions):
        if not isinstance(act, dict):
            continue
        forge_kind = str(act.get("type") or act.get("kind") or "")
        if forge_kind == "PASS":
            # Pass is appended unconditionally below; skip the duplicate entry.
            continue
        opt_kind = _FORGE_KIND_TO_OPTION_KIND.get(forge_kind)
        if opt_kind is None:
            continue
        ability_id = str(act.get("abilityId") or "")
        source_id = str(act.get("sourceId") or "")
        source_name = str(act.get("sourceName") or "")
        options.append(
            {
                "id": ability_id or f"opt:{i}",
                "kind": opt_kind,
                "card_id": source_id,
                "card_name": source_name,
                "permanent_id": source_id,
                "label": str(act.get("description") or source_name),
                "mana_cost": str(act.get("cost") or ""),
            }
        )
    # Always include a pass option — it's grammar-legal for every priority window.
    options.append({"id": "pass", "kind": "pass", "label": "Pass priority"})
    return {
        "kind": "priority",
        "player_idx": perspective_idx,
        "options": options,
    }


def _attackers_pending_from_state(
    raw_snapshot: dict[str, Any], perspective_id: str
) -> dict[str, Any] | None:
    """Build a ``pending`` dict for a DECLARE_ATTACKERS decision.

    Candidate attackers are battlefield permanents controlled by the
    perspective player whose ``power > 0`` (creature-ish). The renderer
    treats every option as a legal attacker; the BC translator just
    needs the chosen attackers to be present in the option list.
    """

    options: list[dict[str, Any]] = []
    for card in raw_snapshot.get("battlefield") or []:
        if not isinstance(card, dict):
            continue
        if str(card.get("controllerId") or "") != perspective_id:
            continue
        try:
            power = int(card.get("power") or 0)
        except TypeError, ValueError:
            power = 0
        if power <= 0:
            continue
        options.append(
            {
                "id": str(card.get("id") or ""),
                "kind": "attacker",
                "card_id": str(card.get("id") or ""),
                "card_name": str(card.get("name") or ""),
                "permanent_id": str(card.get("id") or ""),
                "label": str(card.get("name") or ""),
            }
        )
    if not options:
        return None
    return {"kind": "attackers", "player_idx": 0, "options": options}


def _blockers_pending_from_state(
    raw_snapshot: dict[str, Any],
    perspective_id: str,
    attacker_ids: Sequence[str],
) -> dict[str, Any] | None:
    """Build a ``pending`` dict for a DECLARE_BLOCKERS decision.

    Each blocker option is a perspective-controlled creature whose
    ``valid_targets`` list is the set of attacking permanents (full UUIDs
    so the translator can match by 3-char prefix). The renderer derives
    the LEGAL_ATTACKER anchor order from the union of ``valid_targets``
    in first-seen blocker-option order; we use the same attacker order
    by attaching the same list to every blocker.
    """

    bf = raw_snapshot.get("battlefield") or []

    # Resolve full attacker UUIDs and labels from the battlefield.
    attacker_targets: list[dict[str, str]] = []
    for prefix in attacker_ids:
        for card in bf:
            if not isinstance(card, dict):
                continue
            cid = str(card.get("id") or "")
            if cid.startswith(prefix):
                attacker_targets.append({"id": cid, "label": str(card.get("name") or "")})
                break
    if not attacker_targets:
        return None

    options: list[dict[str, Any]] = []
    for card in bf:
        if not isinstance(card, dict):
            continue
        if str(card.get("controllerId") or "") != perspective_id:
            continue
        try:
            power = int(card.get("power") or 0)
        except TypeError, ValueError:
            power = 0
        if power <= 0:
            continue
        if bool(card.get("tapped")):
            continue
        options.append(
            {
                "id": str(card.get("id") or ""),
                "kind": "block",
                "card_id": str(card.get("id") or ""),
                "card_name": str(card.get("name") or ""),
                "permanent_id": str(card.get("id") or ""),
                "label": str(card.get("name") or ""),
                "valid_targets": list(attacker_targets),
            }
        )
    if not options:
        return None
    return {"kind": "blockers", "player_idx": 0, "options": options}


def _normalize_snapshot(raw: dict[str, Any], perspective_id: str) -> dict[str, Any]:
    players = [_player(p) for p in raw.get("players") or []]
    battlefield_by_player: dict[str, list[dict[str, Any]]] = {str(p["ID"]): [] for p in players}
    for i, card in enumerate(raw.get("battlefield") or []):
        if not isinstance(card, dict):
            continue
        controller = str(card.get("controllerId") or "")
        battlefield_by_player.setdefault(controller, []).append(
            _zone_card(card, controller, "bf", i)
        )
    for player in players:
        player["Battlefield"] = battlefield_by_player.get(str(player["ID"]), [])

    ordered = players
    if perspective_id:
        for idx, player in enumerate(players):
            if player["ID"] == perspective_id:
                ordered = [players[idx], *players[:idx], *players[idx + 1 :]]
                break

    active_id = str(raw.get("activePlayerId") or "")
    stack = []
    for i, obj in enumerate(raw.get("stack") or []):
        if isinstance(obj, dict):
            name = str(obj.get("name") or obj.get("description") or "")
        else:
            name = str(obj)
        stack.append({"id": f"stack:{i}:{name}", "name": name})

    return {
        "turn": int(raw.get("turn") or 0),
        "active_player": active_id,
        "step": _step(raw.get("step")),
        "players": ordered,
        "stack": stack,
    }


def _player_id_by_name(snapshot: dict[str, Any], name: str) -> str:
    for player in snapshot.get("players") or []:
        if str(player.get("name") or "") == name:
            return str(player.get("id") or "")
    return ""


def _player_name_by_id(snapshot: dict[str, Any], pid: str) -> str:
    for player in snapshot.get("players") or []:
        if str(player.get("id") or "") == pid:
            return str(player.get("name") or "")
    return ""


def _opponent_id(snapshot: dict[str, Any], pid: str) -> str:
    for player in snapshot.get("players") or []:
        candidate = str(player.get("id") or "")
        if candidate and candidate != pid:
            return candidate
    return ""


def _event_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("record") == "EVENT" and isinstance(r.get("snapshot"), dict)]


def _parse_attacks_header(description: str) -> dict[str, str] | None:
    """Match ``"<player> attacks <defender> with N creatures"``.

    The new Forge log format announces an attack with a single header
    line during the ``DECLARE_ATTACKERS`` step; per-attacker detail rows
    follow during ``DECLARE_BLOCKERS``.
    """

    match = _ATTACKS_HEADER_RE.match(description)
    if match is None:
        return None
    return {
        "attacker_player": match.group("attacker_player"),
        "defender_player": match.group("defender_player"),
        "count": match.group("count"),
    }


def _parse_attacker_line(description: str) -> dict[str, Any] | None:
    """Match an ``Attacker: <name> [<id3>] (<P>/<T>) <rest>`` log line.

    ``rest`` is either ``"unblocked"`` or
    ``"blocked by <name> [<id3>] (<P>/<T>) ..."`` (one or more blocker
    triples concatenated).
    """

    match = _ATTACKER_LINE_RE.match(description)
    if match is None:
        return None
    rest = match.group("rest")
    blockers: list[dict[str, str]] = []
    if rest != "unblocked" and rest.startswith("blocked by "):
        for blk in _BLOCKER_TOKEN_RE.finditer(rest[len("blocked by ") :]):
            blockers.append({"name": blk.group("name"), "id_prefix": blk.group("id")})
    return {
        "name": match.group("name"),
        "id_prefix": match.group("id"),
        "blockers": blockers,
    }


def _attack_observed(
    rows: Sequence[dict[str, Any]],
    header_idx: int,
    header: dict[str, str],
) -> dict[str, Any]:
    """Walk forward from the attack-header to gather per-attacker rows."""

    attackers: list[dict[str, str]] = []
    for row in rows[header_idx + 1 :]:
        snap = row.get("snapshot") or {}
        step = str(snap.get("step") or "")
        if step not in ("DECLARE_ATTACKERS", "DECLARE_BLOCKERS"):
            break
        if str(row.get("type") or "") != "LOG":
            continue
        desc = str(row.get("description") or "")
        if desc.startswith("Attacker:"):
            parsed = _parse_attacker_line(desc)
            if parsed is not None:
                attackers.append({"name": parsed["name"], "id_prefix": parsed["id_prefix"]})
                if len(attackers) >= int(header["count"]):
                    break
    return {
        "raw": f"{header['attacker_player']} attacks {header['defender_player']} "
        f"with {header['count']} creatures",
        "actor_name": header["attacker_player"],
        "defender_name": header["defender_player"],
        "attackers": attackers,
    }


def _block_observed(
    rows: Sequence[dict[str, Any]],
    header_idx: int,
    header: dict[str, str],
) -> dict[str, Any] | None:
    """Walk forward to collect ``Attacker: ... blocked by ...`` rows."""

    assignments: list[dict[str, Any]] = []
    seen_attackers: list[dict[str, str]] = []
    for row in rows[header_idx + 1 :]:
        snap = row.get("snapshot") or {}
        step = str(snap.get("step") or "")
        if step not in ("DECLARE_ATTACKERS", "DECLARE_BLOCKERS"):
            break
        if str(row.get("type") or "") != "LOG":
            continue
        desc = str(row.get("description") or "")
        if not desc.startswith("Attacker:"):
            continue
        parsed = _parse_attacker_line(desc)
        if parsed is None:
            continue
        seen_attackers.append({"name": parsed["name"], "id_prefix": parsed["id_prefix"]})
        for blk in parsed["blockers"]:
            assignments.append(
                {
                    "attacker_name": parsed["name"],
                    "attacker_id_prefix": parsed["id_prefix"],
                    "blocker_name": blk["name"],
                    "blocker_id_prefix": blk["id_prefix"],
                }
            )
        if len(seen_attackers) >= int(header["count"]):
            break
    if not assignments:
        return None
    return {
        "raw": f"{header['attacker_player']} attacks {header['defender_player']} "
        f"with {header['count']} creatures (blockers declared)",
        "actor_name": header["defender_player"],
        "attacker_player": header["attacker_player"],
        "attackers": seen_attackers,
        "assignments": assignments,
    }


_STACK_PUSH_NAME_RE = re.compile(r"^STACK_PUSH\s+(?P<name>.+?)\s+by\s+(?P<player>.+?)$")
_LOG_PUTS_BF_RE = re.compile(
    r"^(?P<player>.+?) puts (?P<name>.+?) \[[^\]]+\] from hand onto the Battlefield"
)


def _priority_label(description: str, event_type: str) -> dict[str, Any] | None:
    lower = description.lower()
    # Legacy zip format: "PlayerA played Plains", "PlayerB cast Lightning Bolt"
    if event_type == "STACK_PUSH" and (" cast " in lower or " activated " in lower):
        return {"raw": description, "event_type": event_type}
    if " played " in lower:
        return {"raw": description, "event_type": event_type}
    # Newer Forge log format: STACK_PUSH "STACK_PUSH <CardName> by <Player>",
    # land plays: LOG "PlayerA puts Plains [c00] from hand onto the Battlefield".
    m = _STACK_PUSH_NAME_RE.match(description)
    if m is not None:
        return {
            "raw": description,
            "event_type": event_type,
            "card_name": m.group("name"),
            "player_name": m.group("player"),
        }
    m = _LOG_PUTS_BF_RE.match(description)
    if m is not None:
        return {
            "raw": description,
            "event_type": event_type,
            "card_name": m.group("name"),
            "player_name": m.group("player"),
            "is_land_play": True,
        }
    return None


def _choose_label(description: str) -> dict[str, Any] | None:
    if "choose" not in description.lower():
        return None
    match = _PLAYER_PREFIX_RE.match(description)
    return {
        "raw": description,
        "actor_name": match.group("player") if match is not None else "",
    }


def _may_label(rows: Sequence[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    row = rows[idx]
    description = str(row.get("description") or "")
    if row.get("type") != "STACK_RESOLVE" or "you may" not in description.lower():
        return None
    actor_name = ""
    match = _PLAYER_PREFIX_RE.match(description)
    if match is not None:
        actor_name = match.group("player")
    window = rows[max(0, idx - 4) : idx + 1]
    effect_logs = [
        str(r.get("description") or "")
        for r in window
        if r.get("type") == "LOG" and str(r.get("description") or "")
    ]
    # Conservative signal only: if an effect log immediately precedes the
    # resolve, the player accepted/paid. Otherwise leave the exact yes/no target
    # to a later loader with richer card-specific semantics.
    accepted = bool(effect_logs)
    return {
        "raw": description,
        "actor_name": actor_name,
        "accepted": accepted,
        "effect_logs": effect_logs,
    }


def _may_source_row(rows: Sequence[dict[str, Any]], idx: int) -> dict[str, Any]:
    description = str(rows[idx].get("description") or "")
    for prev in reversed(rows[:idx]):
        if prev.get("type") == "PRIORITY":
            return prev
        if prev.get("type") == "STACK_PUSH" and str(prev.get("description") or "") == description:
            return prev
    return rows[max(0, idx - 1)]


def _tokenize_state(
    snapshot: dict[str, Any],
    tokenizer: Any,
    oracle: dict[str, Any],
    token_ids_by_name: dict[str, int],
) -> tuple[list[int], str]:
    rendered = render_snapshot(
        cast(GameStateSnapshot, snapshot),
        oracle=oracle,
        self_token_id=token_ids_by_name["<self>"],
        opp_token_id=token_ids_by_name["<opp>"],
        mana_token_ids=[token_ids_by_name[t] for t in MANA_TOKENS[:6]],
        card_ref_token_ids=[token_ids_by_name[f"<card-ref:{i}>"] for i in range(64)],
    )
    encoding = tokenizer(rendered.text, add_special_tokens=False, return_attention_mask=False)
    return [int(x) for x in encoding["input_ids"]], rendered.text


def _make_candidate(
    *,
    kind: ChoiceKind,
    archive_member: str,
    game_id: str,
    source: dict[str, Any],
    target: dict[str, Any],
    perspective_id: str,
    perspective_name: str,
    observed: dict[str, Any],
) -> _ChoiceCandidate | None:
    source_snapshot = source.get("snapshot") or {}
    normalized = _normalize_snapshot(source_snapshot, perspective_id)
    if kind == "priority":
        pending = _priority_pending_from_playable_actions(source_snapshot, perspective_id)
        if pending is not None:
            normalized["pending"] = pending
    elif kind == "attack":
        attacker_ids = [a["id_prefix"] for a in observed.get("attackers") or []]
        if not attacker_ids:
            return None
        pending = _attackers_pending_from_state(source_snapshot, perspective_id)
        if pending is None:
            return None
        # Drop if any chosen attacker is not present in the option list.
        opt_ids = [str(o.get("permanent_id") or "") for o in pending["options"]]
        if not all(any(oid.startswith(p) for oid in opt_ids) for p in attacker_ids):
            return None
        normalized["pending"] = pending
    elif kind == "block":
        attacker_ids = [a["id_prefix"] for a in observed.get("attackers") or []]
        pending = _blockers_pending_from_state(source_snapshot, perspective_id, attacker_ids)
        if pending is None:
            return None
        # Verify every block-assignment's blocker UUID is in options.
        opt_ids = [str(o.get("permanent_id") or "") for o in pending["options"]]
        for assignment in observed.get("assignments") or []:
            blk_prefix = str(assignment.get("blocker_id_prefix") or "")
            if not any(oid.startswith(blk_prefix) for oid in opt_ids):
                return None
        normalized["pending"] = pending
    return _ChoiceCandidate(
        kind=kind,
        game_id=game_id,
        archive_member=archive_member,
        source_seq=int(source.get("seq") or 0),
        target_seq=int(target.get("seq") or 0),
        perspective_id=perspective_id,
        perspective_name=perspective_name,
        snapshot=normalized,
        token_ids=[],
        text="",
        observed=observed,
    )


def _materialize_tokens(
    candidate: _ChoiceCandidate,
    *,
    tokenizer: Any,
    oracle: dict[str, Any],
    token_ids_by_name: dict[str, int],
) -> _ChoiceCandidate | None:
    try:
        token_ids, text = _tokenize_state(
            candidate.snapshot,
            tokenizer,
            oracle,
            token_ids_by_name,
        )
    except RenderError, RuntimeError, KeyError, TypeError, ValueError:
        return None
    return _ChoiceCandidate(
        kind=candidate.kind,
        game_id=candidate.game_id,
        archive_member=candidate.archive_member,
        source_seq=candidate.source_seq,
        target_seq=candidate.target_seq,
        perspective_id=candidate.perspective_id,
        perspective_name=candidate.perspective_name,
        snapshot=candidate.snapshot,
        token_ids=token_ids,
        text=text,
        observed=candidate.observed,
        candidate_index=candidate.candidate_index,
    )


def _extract_candidates(
    *,
    archive_member: str,
    meta: _GameMeta,
    rows: Sequence[dict[str, Any]],
    enabled_kinds: set[ChoiceKind],
) -> list[_ChoiceCandidate]:
    events = _event_rows(rows)
    candidates: list[_ChoiceCandidate] = []
    last_priority_by_player: dict[str, tuple[int, dict[str, Any]]] = {}
    for idx, row in enumerate(events):
        snapshot = row.get("snapshot") or {}
        row_type = str(row.get("type") or "")
        desc = str(row.get("description") or "")
        priority_id = str(snapshot.get("priorityPlayerId") or "")
        if row_type == "PRIORITY" and priority_id:
            last_priority_by_player[priority_id] = (idx, row)

        if "priority" in enabled_kinds:
            label = _priority_label(desc, row_type)
            actor_id = priority_id
            if label is not None and actor_id in last_priority_by_player:
                _priority_idx, source = last_priority_by_player[actor_id]
                made = _make_candidate(
                    kind="priority",
                    archive_member=archive_member,
                    game_id=meta.game_id,
                    source=source,
                    target=row,
                    perspective_id=actor_id,
                    perspective_name=_player_name_by_id(snapshot, actor_id),
                    observed=label,
                )
                if made is not None:
                    candidates.append(made)

        if row_type == "LOG":
            header = _parse_attacks_header(desc)
            if header is not None:
                if "attack" in enabled_kinds:
                    observed = _attack_observed(events, idx, header)
                    if observed["attackers"]:
                        actor_id = _player_id_by_name(snapshot, header["attacker_player"])
                        made = _make_candidate(
                            kind="attack",
                            archive_member=archive_member,
                            game_id=meta.game_id,
                            source=row,
                            target=row,
                            perspective_id=actor_id or str(snapshot.get("activePlayerId") or ""),
                            perspective_name=header["attacker_player"],
                            observed=observed,
                        )
                        if made is not None:
                            candidates.append(made)
                if "block" in enabled_kinds:
                    block_observed = _block_observed(events, idx, header)
                    if block_observed is not None:
                        defender_name = header["defender_player"]
                        actor_id = _player_id_by_name(snapshot, defender_name)
                        if not actor_id:
                            actor_id = _opponent_id(
                                snapshot, str(snapshot.get("activePlayerId") or "")
                            )
                        # Find the snapshot at the first Attacker: row (step
                        # DECLARE_BLOCKERS) so candidate blocker filtering
                        # reflects the post-declare-attackers board.
                        block_source = row
                        for jrow in events[idx + 1 :]:
                            if str(jrow.get("type") or "") != "LOG":
                                continue
                            jdesc = str(jrow.get("description") or "")
                            if jdesc.startswith("Attacker:"):
                                block_source = jrow
                                break
                        made = _make_candidate(
                            kind="block",
                            archive_member=archive_member,
                            game_id=meta.game_id,
                            source=block_source,
                            target=block_source,
                            perspective_id=actor_id,
                            perspective_name=defender_name,
                            observed=block_observed,
                        )
                        if made is not None:
                            candidates.append(made)

        if "may" in enabled_kinds:
            label = _may_label(events, idx)
            if label is not None:
                actor_id = _player_id_by_name(snapshot, str(label.get("actor_name") or ""))
                source = _may_source_row(events, idx)
                made = _make_candidate(
                    kind="may",
                    archive_member=archive_member,
                    game_id=meta.game_id,
                    source=source,
                    target=row,
                    perspective_id=actor_id or priority_id,
                    perspective_name=str(label.get("actor_name") or ""),
                    observed=label,
                )
                if made is not None:
                    candidates.append(made)

        if row_type == "LOG" and "choose" in desc.lower() and "choose" in enabled_kinds:
            label = _choose_label(desc)
            if label is not None:
                actor_id = _player_id_by_name(snapshot, str(label.get("actor_name") or ""))
                made = _make_candidate(
                    kind="choose",
                    archive_member=archive_member,
                    game_id=meta.game_id,
                    source=row,
                    target=row,
                    perspective_id=actor_id or priority_id,
                    perspective_name=str(label.get("actor_name") or ""),
                    observed=label,
                )
                if made is not None:
                    candidates.append(made)

    return [
        _ChoiceCandidate(
            kind=c.kind,
            game_id=c.game_id,
            archive_member=c.archive_member,
            source_seq=c.source_seq,
            target_seq=c.target_seq,
            perspective_id=c.perspective_id,
            perspective_name=c.perspective_name,
            snapshot=c.snapshot,
            token_ids=c.token_ids,
            text=c.text,
            observed=c.observed,
            candidate_index=i,
        )
        for i, c in enumerate(candidates)
    ]


def _stable_choice(candidates: Sequence[_ChoiceCandidate]) -> _ChoiceCandidate:
    digest = hashlib.blake2b(candidates[0].game_id.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest, byteorder="big", signed=False) % len(candidates)
    return candidates[idx]


def _stable_choices(candidates: Sequence[_ChoiceCandidate], count: int) -> list[_ChoiceCandidate]:
    if count <= 0:
        return []
    if count >= len(candidates):
        return list(candidates)
    return sorted(
        candidates,
        key=lambda c: hashlib.blake2b(
            f"{c.game_id}:{c.candidate_index}".encode(),
            digest_size=8,
        ).digest(),
    )[:count]


def _priority_choice(
    candidates: Sequence[_ChoiceCandidate],
    kind_priority: Sequence[ChoiceKind],
) -> _ChoiceCandidate:
    by_kind: dict[ChoiceKind, list[_ChoiceCandidate]] = {kind: [] for kind in CHOICE_KINDS}
    for candidate in candidates:
        by_kind[candidate.kind].append(candidate)
    for kind in kind_priority:
        bucket = by_kind.get(kind) or []
        if bucket:
            return bucket[0]
    return candidates[0]


def _priority_choices(
    candidates: Sequence[_ChoiceCandidate],
    kind_priority: Sequence[ChoiceKind],
    count: int,
) -> list[_ChoiceCandidate]:
    if count <= 0:
        return []
    by_kind: dict[ChoiceKind, list[_ChoiceCandidate]] = {kind: [] for kind in CHOICE_KINDS}
    for candidate in candidates:
        by_kind[candidate.kind].append(candidate)
    selected: list[_ChoiceCandidate] = []
    seen: set[int] = set()
    for kind in kind_priority:
        for candidate in by_kind.get(kind) or []:
            if candidate.candidate_index in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.candidate_index)
            if len(selected) >= count:
                return selected
    for candidate in candidates:
        if candidate.candidate_index in seen:
            continue
        selected.append(candidate)
        if len(selected) >= count:
            break
    return selected


def _terminal_sign(winner_id: str | None, perspective_id: str) -> float:
    if not winner_id or not perspective_id:
        return 0.0
    return 1.0 if winner_id == perspective_id else -1.0


def _record(
    candidate: _ChoiceCandidate,
    meta: _GameMeta,
    *,
    total_candidates: int,
) -> dict[str, Any]:
    return {
        "format": "forge_choice_situation",
        "format_version": FORMAT_VERSION,
        "game_id": candidate.game_id,
        "archive_member": candidate.archive_member,
        "choice": {
            "kind": candidate.kind,
            "candidate_index": candidate.candidate_index,
            "candidate_count": total_candidates,
            "source_seq": candidate.source_seq,
            "target_seq": candidate.target_seq,
            "perspective_id": candidate.perspective_id,
            "perspective_name": candidate.perspective_name,
            "observed": candidate.observed,
        },
        "state": {
            "token_ids": candidate.token_ids,
            "text": candidate.text,
            "snapshot": candidate.snapshot,
        },
        "outcome": {
            "winner_id": meta.winner_id,
            "winner_name": meta.winner_name,
            "terminal_sign": _terminal_sign(meta.winner_id, candidate.perspective_id),
            "players": meta.players,
            "extras": meta.extras,
        },
    }


def _parse_kinds(raw: str) -> set[ChoiceKind]:
    if raw == "all":
        return set(CHOICE_KINDS)
    out: set[ChoiceKind] = set()
    for item in raw.split(","):
        kind = item.strip()
        if kind not in CHOICE_KINDS:
            raise ValueError(f"unknown choice kind {kind!r}; expected one of {CHOICE_KINDS}")
        out.add(cast(ChoiceKind, kind))
    return out


def _parse_kind_priority(raw: str) -> tuple[ChoiceKind, ...]:
    kinds = []
    for item in raw.split(","):
        kind = item.strip()
        if not kind:
            continue
        if kind not in CHOICE_KINDS:
            raise ValueError(f"unknown priority kind {kind!r}; expected one of {CHOICE_KINDS}")
        kinds.append(cast(ChoiceKind, kind))
    return tuple(kinds) or DEFAULT_KIND_PRIORITY


def _token_ids_by_name(tokenizer: Any) -> dict[str, int]:
    names = {
        "<self>",
        "<opp>",
        *MANA_TOKENS[:6],
        *(f"<card-ref:{i}>" for i in range(64)),
    }
    out: dict[str, int] = {}
    for name in names:
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, list):
            raise TypeError(f"token {name!r} resolved to multiple ids")
        out[name] = int(tid)
    return out


def _output_format(path: Path) -> OutputFormat:
    name = path.name
    if name.endswith(".jsonl.gz"):
        return "jsonl.gz"
    return "torch_shards"


class _TorchShardWriter:
    def __init__(self, out_dir: Path, *, shard_size: int, overwrite: bool) -> None:
        if shard_size <= 0:
            raise ValueError("--shard-size must be positive")
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.shard_index = 0
        self.records: list[dict[str, Any]] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = [*self.out_dir.glob("part-*.pt"), self.out_dir / "manifest.json"]
        existing = [path for path in existing if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"{self.out_dir} already contains extracted shards; pass --overwrite to replace"
            )
        if overwrite:
            for path in existing:
                path.unlink()

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if len(self.records) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.out_dir / f"part-{self.shard_index:05d}.pt"
        tmp_path = path.with_name(f"{path.name}.tmp")
        torch.save(
            {
                "format": "forge_choice_situations_torch_shard",
                "format_version": FORMAT_VERSION,
                "shard_index": self.shard_index,
                "records": self.records,
            },
            tmp_path,
        )
        tmp_path.replace(path)
        self.shard_index += 1
        self.records = []

    def close(self, stats: dict[str, int]) -> None:
        self.flush()
        manifest = {
            "format": "forge_choice_situations_manifest",
            "format_version": FORMAT_VERSION,
            "shards": self.shard_index,
            "shard_size": self.shard_size,
            "stats": stats,
        }
        manifest_path = self.out_dir / "manifest.json"
        tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
        tmp_path.write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )
        tmp_path.replace(manifest_path)


def extract(args: argparse.Namespace) -> dict[str, int]:
    tokenizer = load_tokenizer(args.tokenizer_dir)
    oracle = load_oracle_text(args.oracle_path)
    token_ids_by_name = _token_ids_by_name(tokenizer)
    enabled_kinds = _parse_kinds(args.kinds)
    kind_priority = _parse_kind_priority(args.kind_priority)
    out_format = _output_format(args.out)
    stats = {
        "games_seen": 0,
        "games_written": 0,
        "records_written": 0,
        "games_without_candidates": 0,
        "candidates_seen": 0,
    }
    kind_counts = {kind: 0 for kind in CHOICE_KINDS}

    if out_format == "jsonl.gz":
        args.out.parent.mkdir(parents=True, exist_ok=True)
    shard_writer = (
        _TorchShardWriter(args.out, shard_size=args.shard_size, overwrite=args.overwrite)
        if out_format == "torch_shards"
        else None
    )
    jsonl_out = (
        gzip.open(args.out, "wb", compresslevel=args.compresslevel)
        if out_format == "jsonl.gz"
        else None
    )
    try:
        for member, rows in _iter_jsonl(args.zip):
            if args.limit_games is not None and stats["games_seen"] >= args.limit_games:
                break
            meta = _meta_from_rows(rows)
            if meta is None or not meta.game_id:
                continue
            stats["games_seen"] += 1
            candidates = _extract_candidates(
                archive_member=member,
                meta=meta,
                rows=rows,
                enabled_kinds=enabled_kinds,
            )
            stats["candidates_seen"] += len(candidates)
            if not candidates:
                stats["games_without_candidates"] += 1
                continue
            selected_candidates = (
                _stable_choices(candidates, args.choices_per_game)
                if args.selection == "hash"
                else _priority_choices(candidates, kind_priority, args.choices_per_game)
            )
            materialized = 0
            for selected_candidate in selected_candidates:
                selected = _materialize_tokens(
                    selected_candidate,
                    tokenizer=tokenizer,
                    oracle=oracle,
                    token_ids_by_name=token_ids_by_name,
                )
                if selected is None:
                    continue
                kind_counts[selected.kind] += 1
                record = _record(selected, meta, total_candidates=len(candidates))
                if shard_writer is not None:
                    shard_writer.write(record)
                elif jsonl_out is not None:
                    jsonl_out.write(orjson.dumps(record))
                    jsonl_out.write(b"\n")
                stats["records_written"] += 1
                materialized += 1
            if materialized == 0:
                stats["games_without_candidates"] += 1
                continue
            stats["games_written"] += 1
            if args.progress_every > 0 and stats["games_seen"] % args.progress_every == 0:
                print(_dumps_text(stats), file=sys.stderr, flush=True)
    finally:
        if jsonl_out is not None:
            jsonl_out.close()

    stats.update({f"written_{kind}": count for kind, count in kind_counts.items()})
    if shard_writer is not None:
        shard_writer.close(stats)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/forge-games-20260507-110843.zip"),
        help="Forge game archive containing *.jsonl.gz members.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/forge_choice_situations"),
        help="Output directory for sharded part-*.pt files, or .jsonl.gz for debug JSONL.",
    )
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing part-*.pt shards and manifest.json under --out",
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/text_encoder_tokenizer"))
    parser.add_argument(
        "--oracle-path", type=Path, default=Path("data/card_oracle_embeddings.json")
    )
    parser.add_argument(
        "--kinds",
        default="all",
        help="Comma-separated choice kinds to consider, or 'all'.",
    )
    parser.add_argument(
        "--selection",
        choices=("hash", "priority"),
        default="hash",
        help="How to select situations when a game has multiple candidates.",
    )
    parser.add_argument(
        "--choices-per-game",
        type=int,
        default=2,
        help="maximum selected situations to write per game",
    )
    parser.add_argument(
        "--kind-priority",
        default=",".join(DEFAULT_KIND_PRIORITY),
        help="Kind order used when --selection priority.",
    )
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress to stderr every N games; set 0 to disable.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    stats = extract(args)
    print(orjson.dumps(stats, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8"))


if __name__ == "__main__":
    main()
