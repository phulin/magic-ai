from pathlib import Path

from magic_ai.text_encoder.policy_value_pretrain import _iter_records


def test_iter_records_reads_rust_extracted_arrow_fixture() -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "forge_decisions_arrow"

    records = list(_iter_records(fixture_dir))

    assert len(records) == 1
    record = records[0]
    assert record["format"] == "forge_choice_situation"
    assert record["format_version"] == 2
    assert record["game_id"] == "game-fixture-1"
    assert record["archive_member"] == "fixture.jsonl.gz"
    assert record["choice"] == {
        "kind": "priority",
        "candidate_index": 0,
        "candidate_count": 1,
        "source_seq": 10,
        "target_seq": 11,
        "perspective_id": "p1",
        "perspective_name": "PlayerA",
        "observed": {
            "event_type": "LOG",
            "raw": "PlayerA played Forest (42)",
        },
    }
    assert record["outcome"] == {
        "winner_id": "p1",
        "winner_name": "PlayerA",
        "terminal_sign": 1.0,
        "players": [
            {"id": "p1", "name": "PlayerA", "hasWon": True},
            {"id": "p2", "name": "PlayerB", "hasLost": True},
        ],
        "extras": {"turns": 3},
    }

    snapshot = record["state"]["snapshot"]
    assert snapshot["turn"] == 3
    assert snapshot["step"] == "Precombat Main"
    assert snapshot["active_player"] == "p1"
    assert snapshot["pending"]["kind"] == "priority"
    assert snapshot["pending"]["options"] == [
        {
            "id": "opt:0",
            "kind": "play",
            "card_id": "hand:forest",
            "card_name": "Forest",
            "permanent_id": "hand:forest",
            "label": "Play Forest",
            "mana_cost": "",
        },
        {"id": "pass", "kind": "pass", "label": "Pass priority"},
    ]
    assert snapshot["players"][0]["ID"] == "p1"
    assert snapshot["players"][0]["Name"] == "PlayerA"
