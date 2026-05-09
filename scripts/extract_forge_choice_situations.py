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
    NUM_TOKENS,
    load_tokenizer,
)

ChoiceKind = Literal["priority", "attack", "block", "may", "choose"]
OutputFormat = Literal["torch_shards", "jsonl.gz"]

FORMAT_VERSION = 1
DEFAULT_KIND_PRIORITY: tuple[ChoiceKind, ...] = ("may", "block", "attack", "choose", "priority")
CHOICE_KINDS: tuple[ChoiceKind, ...] = ("priority", "attack", "block", "may", "choose")

_ASSIGNED_ATTACK_RE = re.compile(
    r"^(?P<player>.+?) assigned (?P<objects>.+?) to attack (?P<target>.+?)\.$"
)
_DID_NOT_BLOCK_RE = re.compile(r"^(?P<player>.+?) didn't block (?P<attacker>.+?)\.$")
_BLOCKED_RE = re.compile(r"^(?P<player>.+?) blocked (?P<attacker>.+?) with (?P<blockers>.+?)\.$")
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


def _next_non_priority(rows: Sequence[dict[str, Any]], start: int) -> dict[str, Any] | None:
    for row in rows[start + 1 :]:
        if row.get("type") == "PRIORITY":
            continue
        desc = str(row.get("description") or "")
        if not desc:
            continue
        return row
    return None


def _attack_label(description: str) -> dict[str, Any] | None:
    match = _ASSIGNED_ATTACK_RE.match(description)
    if match is None:
        return None
    return {
        "raw": description,
        "actor_name": match.group("player"),
        "attackers_text": match.group("objects"),
        "defender_text": match.group("target"),
    }


def _block_label(description: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in description.splitlines() if part.strip()]
    parsed: list[dict[str, Any]] = []
    actor = ""
    for part in parts:
        no_block = _DID_NOT_BLOCK_RE.match(part)
        if no_block is not None:
            actor = actor or no_block.group("player")
            parsed.append(
                {
                    "kind": "no_block",
                    "actor_name": no_block.group("player"),
                    "attacker_text": no_block.group("attacker"),
                    "blockers_text": "",
                }
            )
            continue
        blocked = _BLOCKED_RE.match(part)
        if blocked is not None:
            actor = actor or blocked.group("player")
            parsed.append(
                {
                    "kind": "block",
                    "actor_name": blocked.group("player"),
                    "attacker_text": blocked.group("attacker"),
                    "blockers_text": blocked.group("blockers"),
                }
            )
    if not parsed:
        return None
    return {"raw": description, "actor_name": actor, "assignments": parsed}


def _priority_label(description: str, event_type: str) -> dict[str, Any] | None:
    lower = description.lower()
    if event_type == "STACK_PUSH" and (" cast " in lower or " activated " in lower):
        return {"raw": description, "event_type": event_type}
    if " played " in lower:
        return {"raw": description, "event_type": event_type}
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
        chosen_token_id=token_ids_by_name["<chosen>"],
        none_token_id=token_ids_by_name["<none>"],
        yes_token_id=token_ids_by_name["<yes>"],
        no_token_id=token_ids_by_name["<no>"],
        mulligan_token_id=token_ids_by_name["<mulligan>"],
        keep_token_id=token_ids_by_name["<keep>"],
        self_token_id=token_ids_by_name["<self>"],
        opp_token_id=token_ids_by_name["<opp>"],
        num_token_ids=[token_ids_by_name[t] for t in NUM_TOKENS],
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

        if row_type == "LOG" and "Declare Attackers Step" in desc:
            target = _next_non_priority(events, idx)
            if target is not None and "attack" in enabled_kinds:
                label = _attack_label(str(target.get("description") or ""))
                if label is not None:
                    actor_id = _player_id_by_name(snapshot, str(label.get("actor_name") or ""))
                    made = _make_candidate(
                        kind="attack",
                        archive_member=archive_member,
                        game_id=meta.game_id,
                        source=row,
                        target=target,
                        perspective_id=actor_id or str(snapshot.get("activePlayerId") or ""),
                        perspective_name=str(label.get("actor_name") or ""),
                        observed=label,
                    )
                    if made is not None:
                        candidates.append(made)

        if row_type == "LOG" and "Declare Blockers Step" in desc:
            target = _next_non_priority(events, idx)
            if target is not None and "block" in enabled_kinds:
                label = _block_label(str(target.get("description") or ""))
                if label is not None:
                    actor_id = _player_id_by_name(snapshot, str(label.get("actor_name") or ""))
                    if not actor_id:
                        actor_id = _opponent_id(snapshot, str(snapshot.get("activePlayerId") or ""))
                    made = _make_candidate(
                        kind="block",
                        archive_member=archive_member,
                        game_id=meta.game_id,
                        source=row,
                        target=target,
                        perspective_id=actor_id,
                        perspective_name=str(label.get("actor_name") or ""),
                        observed=label,
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
        "<chosen>",
        "<none>",
        "<yes>",
        "<no>",
        "<mulligan>",
        "<keep>",
        "<self>",
        "<opp>",
        *NUM_TOKENS,
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
        for member, rows in _iter_zip_jsonl(args.zip):
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
