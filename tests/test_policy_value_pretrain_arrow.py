import gzip
from pathlib import Path
from typing import Any, cast

import orjson
from magic_ai.text_encoder.decision_spec import DecisionType
from magic_ai.text_encoder.policy_value_pretrain import (
    ForgeChoiceDataset,
    ForgePolicyValueConfig,
    _iter_records,
)
from magic_ai.text_encoder.tokenizer import load_tokenizer


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


def _record_from_fixture() -> dict[str, object]:
    fixture_dir = Path(__file__).parent / "fixtures" / "forge_decisions_arrow"
    return next(_iter_records(fixture_dir))


def _with_choice(
    base: dict[str, Any],
    *,
    game_id: str,
    seq: int,
    snapshot: dict[str, object],
    observed: dict[str, object],
) -> dict[str, Any]:
    record = dict(base)
    choice = cast(dict[str, Any], base["choice"])
    record["game_id"] = game_id
    record["choice"] = {
        **choice,
        "source_seq": seq,
        "target_seq": seq + 1,
        "observed": observed,
    }
    record["state"] = {"snapshot": snapshot}
    return record


def test_dataset_batches_pass_no_attack_and_no_block_targets(tmp_path: Path) -> None:
    base = _record_from_fixture()
    state = cast(dict[str, Any], base["state"])
    base_snapshot = dict(cast(dict[str, Any], state["snapshot"]))

    priority_snapshot = dict(base_snapshot)
    priority_record = _with_choice(
        base,
        game_id="game-pass",
        seq=10,
        snapshot=priority_snapshot,
        observed={"raw": "Pass priority", "event_type": "PASS", "is_inferred": True},
    )

    attack_snapshot = dict(base_snapshot)
    attack_snapshot["pending"] = {
        "kind": "attackers",
        "player_idx": 0,
        "options": [
            {
                "id": "bear-1",
                "kind": "attacker",
                "card_id": "bear-1",
                "card_name": "Grizzly Bears",
                "permanent_id": "bear-1",
                "label": "Grizzly Bears",
            }
        ],
    }
    attack_record = _with_choice(
        base,
        game_id="game-no-attack",
        seq=20,
        snapshot=attack_snapshot,
        observed={
            "raw": "PlayerA declares no attackers",
            "actor_name": "PlayerA",
            "attackers": [],
            "no_attack": True,
        },
    )

    block_snapshot = dict(base_snapshot)
    block_snapshot["pending"] = {
        "kind": "blockers",
        "player_idx": 0,
        "options": [
            {
                "id": "wall-1",
                "kind": "block",
                "card_id": "wall-1",
                "card_name": "Wall of Wood",
                "permanent_id": "wall-1",
                "label": "Wall of Wood",
                "valid_targets": [{"id": "bear-1", "label": "Grizzly Bears"}],
            }
        ],
    }
    block_record = _with_choice(
        base,
        game_id="game-no-block",
        seq=30,
        snapshot=block_snapshot,
        observed={
            "raw": "no blocks declared",
            "actor_name": "PlayerA",
            "attackers": [{"name": "Grizzly Bears", "id_prefix": "bear"}],
            "assignments": [],
            "no_block": True,
        },
    )

    path = tmp_path / "records.jsonl.gz"
    with gzip.open(path, "wb") as fh:
        for record in (priority_record, attack_record, block_record):
            fh.write(orjson.dumps(record))
            fh.write(b"\n")

    tokenizer = load_tokenizer()
    cfg = ForgePolicyValueConfig(
        data_path=path,
        batch_size=3,
        eval_fraction=0.0,
        pad_token_id=int(tokenizer.pad_token_id or 0),
        sequence_mode="none",
    )
    dataset = ForgeChoiceDataset(cfg, tokenizer=tokenizer, oracle={}, split="all")
    batch = dataset._batch_from_indices([0, 1, 2])

    assert dataset.n_examples == 3
    assert batch.decision_type_per_row.tolist() == [
        int(DecisionType.PRIORITY),
        int(DecisionType.DECLARE_ATTACKERS),
        int(DecisionType.DECLARE_BLOCKERS),
    ]
    assert batch.output_pad_mask.sum(dim=1).tolist() == [3, 2, 2]
