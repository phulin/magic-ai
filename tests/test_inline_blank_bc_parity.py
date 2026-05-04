"""Tests for the inline-blank BC parity gate script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

from magic_ai.game_state import GameStateSnapshot
from magic_ai.text_encoder.render import render_snapshot

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inline_blank_bc_parity.py"
SPEC = importlib.util.spec_from_file_location("inline_blank_bc_parity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)


def test_synthetic_trace_rows_are_priority_snapshots() -> None:
    rows = parity.make_synthetic_priority_trace(4)

    assert len(rows) == 4
    assert all(row.snapshot["pending"]["kind"] == "priority" for row in rows)
    assert [row.selected_option_index for row in rows] == [0, 1, 2, 3]


def test_trace_loader_accepts_selected_option_id(tmp_path: Path) -> None:
    [row] = parity.make_synthetic_priority_trace(2)[:1]
    selected = row.snapshot["pending"]["options"][2]["id"]
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"snapshot": row.snapshot, "selected_option_id": selected}) + "\n",
        encoding="utf-8",
    )

    loaded = parity.load_priority_trace(path)

    assert len(loaded) == 1
    assert loaded[0].selected_option_index == 2


def test_trace_loader_accepts_priority_trace_indices(tmp_path: Path) -> None:
    [row] = parity.make_synthetic_priority_trace(2)[:1]
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"snapshot": row.snapshot, "trace": {"kind": "priority", "indices": [0]}})
        + "\n",
        encoding="utf-8",
    )

    loaded = parity.load_priority_trace(path)

    assert loaded[0].selected_option_index == 3


def test_trace_loader_accepts_transcript_action_shape(tmp_path: Path) -> None:
    [row] = parity.make_synthetic_priority_trace(2)[:1]
    state = dict(row.snapshot)
    pending = dict(state.pop("pending"))
    options = list(pending["options"])
    options[0] = {**options[0], "kind": "cast_spell"}
    pending["options"] = options
    action = {"kind": "cast_spell", "card_id": options[0]["card_id"], "targets": []}
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        json.dumps({"state": state, "pending": pending, "action": action}) + "\n",
        encoding="utf-8",
    )

    loaded = parity.load_priority_trace(path)

    assert loaded[0].snapshot["pending"]["kind"] == "priority"
    assert loaded[0].selected_option_index == 0


def test_target_mappings_use_render_order_not_payload_order() -> None:
    [row] = parity.make_synthetic_priority_trace(2)[:1]
    snapshot = row.snapshot
    options = list(snapshot["pending"]["options"])
    permuted = [options[3], options[1], options[0], options[2]]
    selected_payload_index = 3
    permuted_snapshot = cast(
        GameStateSnapshot,
        {
            **snapshot,
            "pending": {**snapshot["pending"], "options": permuted},
        },
    )

    legacy = render_snapshot(permuted_snapshot, permuted)
    inline = render_snapshot(
        permuted_snapshot,
        permuted,
        use_inline_blanks=True,
        chosen_token_id=12345,
    )

    assert parity._target_legacy_ordinal(legacy, selected_payload_index) == 3
    assert parity._target_inline_blank_index(inline, selected_payload_index) == 0


def test_trace_loader_rejects_non_priority(tmp_path: Path) -> None:
    row: dict[str, Any] = {
        "snapshot": {
            "turn": 1,
            "active_player": "p1",
            "step": "Precombat Main",
            "players": [],
            "pending": {"kind": "main", "player_idx": 0, "options": []},
        },
        "selected_option_index": 0,
    }
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    try:
        parity.load_priority_trace(path)
    except ValueError as exc:
        assert "not a priority snapshot" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected load_priority_trace to reject non-priority rows")
