import gzip
from array import array
from pathlib import Path
from typing import Any, cast

import orjson
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch, pack_batch
from magic_ai.text_encoder.decision_spec import AnchorKind, DecisionType
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE
from magic_ai.text_encoder.policy import EncodedSnapshots
from magic_ai.text_encoder.policy_value_pretrain import (
    ForgeChoiceDataset,
    ForgeDecoderBatch,
    ForgePolicyValueConfig,
    ForgePolicyValueTrainer,
    ForgeSequencedBatch,
    _ArrowGameSpanCache,
    _iter_records,
    _load_arrow_game_index,
    _write_arrow_game_index,
)
from magic_ai.text_encoder.tokenizer import load_tokenizer
from torch import nn


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


def test_arrow_game_index_sidecar_round_trips(tmp_path: Path) -> None:
    spans = _ArrowGameSpanCache(
        shard_paths=(tmp_path / "part-000000.arrow",),
        game_ids=("game-a", "game-b"),
        game_shards=array("I", [0, 0]),
        game_batches=array("I", [0, 0]),
        game_starts=array("I", [0, 3]),
        game_lengths=array("I", [3, 2]),
    )

    _write_arrow_game_index(tmp_path, spans)
    loaded = _load_arrow_game_index(tmp_path, spans.shard_paths)

    assert loaded is not None
    assert loaded.game_ids == spans.game_ids
    assert loaded.game_shards.tolist() == spans.game_shards.tolist()
    assert loaded.game_batches.tolist() == spans.game_batches.tolist()
    assert loaded.game_starts.tolist() == spans.game_starts.tolist()
    assert loaded.game_lengths.tolist() == spans.game_lengths.tolist()


def _write_pretokenized_arrow_fixture(
    root: Path,
    records: list[dict[str, Any]],
    token_rows: list[list[int]],
    card_ref_rows: list[list[int]],
    action_token_rows: list[list[int]] | None = None,
) -> None:
    root.mkdir(parents=True)
    schema = pa.schema(
        [
            pa.field("format_version", pa.uint16(), nullable=False),
            pa.field("game_id", pa.utf8(), nullable=False),
            pa.field("archive_member", pa.utf8(), nullable=False),
            pa.field("kind_id", pa.uint8(), nullable=False),
            pa.field("candidate_index", pa.uint32(), nullable=False),
            pa.field("candidate_count", pa.uint32(), nullable=False),
            pa.field("source_seq", pa.int64(), nullable=False),
            pa.field("target_seq", pa.int64(), nullable=False),
            pa.field("perspective_id", pa.utf8(), nullable=False),
            pa.field("perspective_name", pa.utf8(), nullable=False),
            pa.field("winner_id", pa.utf8(), nullable=True),
            pa.field("winner_name", pa.utf8(), nullable=True),
            pa.field("terminal_sign", pa.float32(), nullable=False),
            pa.field("snapshot_json", pa.large_utf8(), nullable=False),
            pa.field("observed_json", pa.large_utf8(), nullable=False),
            pa.field("outcome_players_json", pa.large_utf8(), nullable=False),
            pa.field("outcome_extras_json", pa.large_utf8(), nullable=False),
            pa.field("token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("seq_length", pa.uint32(), nullable=False),
            pa.field("card_ref_positions", pa.list_(pa.int32()), nullable=False),
            pa.field("token_overflow", pa.bool_(), nullable=False),
            pa.field("card_ref_overflow", pa.bool_(), nullable=False),
            pa.field("action_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("action_seq_length", pa.uint32(), nullable=False),
            pa.field("decision_type", pa.int32(), nullable=False),
            pa.field("pointer_anchor_positions", pa.list_(pa.int32()), nullable=False),
            pa.field("pointer_anchor_kinds", pa.list_(pa.int32()), nullable=False),
            pa.field("pointer_anchor_subjects", pa.list_(pa.int32()), nullable=False),
            pa.field("pointer_anchor_handles", pa.list_(pa.int32()), nullable=False),
            pa.field("legal_edge_bitmap", pa.list_(pa.uint8()), nullable=False),
            pa.field("legal_edge_n_blockers", pa.uint32(), nullable=False),
            pa.field("legal_edge_n_attackers", pa.uint32(), nullable=False),
        ],
        metadata={
            "format": "forge_pretrain_decision_arrow",
            "format_version": "2",
            "stage": "pretokenized",
        },
    )
    rows = []
    if action_token_rows is None:
        action_token_rows = [[201, 202, 203, 204, 204, 205] for _ in records]
    for record, tokens, card_refs, action_tokens in zip(
        records, token_rows, card_ref_rows, action_token_rows, strict=True
    ):
        choice = cast(dict[str, Any], record["choice"])
        outcome = cast(dict[str, Any], record["outcome"])
        rows.append(
            [
                2,
                record["game_id"],
                record["archive_member"],
                0,
                choice["candidate_index"],
                choice["candidate_count"],
                choice["source_seq"],
                choice["target_seq"],
                choice["perspective_id"],
                choice["perspective_name"],
                outcome["winner_id"],
                outcome["winner_name"],
                outcome["terminal_sign"],
                orjson.dumps(cast(dict[str, Any], record["state"])["snapshot"]).decode(),
                orjson.dumps(choice["observed"]).decode(),
                orjson.dumps(outcome["players"]).decode(),
                orjson.dumps(outcome["extras"]).decode(),
                tokens,
                len(tokens),
                card_refs,
                False,
                False,
                action_tokens,
                len(action_tokens),
                int(DecisionType.PRIORITY),
                [3, 4],
                [int(AnchorKind.LEGAL_ACTION), int(AnchorKind.LEGAL_ACTION)],
                [0, 1],
                [0, 1],
                [],
                0,
                0,
            ]
        )
    columns = list(zip(*rows, strict=True))
    batch = pa.record_batch(
        [pa.array(col, type=field.type) for col, field in zip(columns, schema)],
        schema=schema,
    )
    with pa.OSFile(str(root / "part-000000.arrow"), "wb") as sink:
        with pa_ipc.new_file(sink, schema, options=pa_ipc.IpcWriteOptions(compression="zstd")) as w:
            w.write_batch(batch)
    (root / "manifest.json").write_bytes(
        orjson.dumps(
            {
                "format": "forge_pretrain_decision_arrow",
                "format_version": 2,
                "stage": "pretokenized",
                "compression": "zstd",
                "game_index": "game_index.arrow",
                "shards": 1,
                "records_written": len(records),
            }
        )
    )


def test_dataset_loads_pretokenized_arrow_state_tokens(tmp_path: Path) -> None:
    base = _record_from_fixture()
    record = _with_choice(
        base,
        game_id="game-pretokenized",
        seq=10,
        snapshot=cast(dict[str, Any], cast(dict[str, Any], base["state"])["snapshot"]),
        observed={"raw": "Pass priority", "event_type": "PASS", "is_inferred": True},
    )
    arrow_dir = tmp_path / "pretokenized"
    card_refs = [-1] * 256
    card_refs[7] = 1
    _write_pretokenized_arrow_fixture(arrow_dir, [record], [[101, 102, 103]], [card_refs])

    tokenizer = load_tokenizer()
    cfg = ForgePolicyValueConfig(
        data_path=arrow_dir,
        batch_size=1,
        eval_fraction=0.0,
        pad_token_id=int(tokenizer.pad_token_id or 0),
        sequence_mode="none",
    )
    dataset = ForgeChoiceDataset(cfg, tokenizer=tokenizer, oracle={}, split="all")
    batch = dataset._batch_from_indices([0])

    assert dataset.n_examples == 1
    assert batch.encoded.seq_lengths.tolist()[0] > 3
    assert batch.encoded.token_ids[0, :3].tolist() == [101, 102, 103]
    assert batch.encoded.token_ids[0, 3:9].tolist() == [201, 202, 203, 204, 204, 205]
    assert batch.encoded.card_ref_positions[0, 7].item() == 1


class _FakeGrammarDecoder(nn.Module):
    def forward_teacher_forced(
        self,
        target_tokens: torch.Tensor,
        encoded: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del encoded, attn_mask
        b, target_len = target_tokens.shape
        vocab = torch.zeros((b, target_len, GRAMMAR_VOCAB_SIZE), dtype=torch.float32)
        pointer = torch.zeros((b, target_len, 2), dtype=torch.float32)
        return vocab, pointer


class _FakeTextPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.grammar_decoder = _FakeGrammarDecoder()

    def run_heads(
        self, encoded: EncodedSnapshots, state_vec: torch.Tensor | None = None
    ) -> torch.Tensor:
        del encoded
        sv = state_vec if state_vec is not None else torch.empty(0)
        return sv.squeeze(-1) + self.param


class _FakeRecurrentPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_policy = _FakeTextPolicy()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
        self.lstm_layers = 1
        self.lstm_hidden = 1
        self.calls: list[torch.Tensor] = []

    def encode_with_history(
        self,
        batch: PackedTextBatch | TextEncodedBatch,
        h_in: torch.Tensor | None = None,
        c_in: torch.Tensor | None = None,
    ) -> tuple[EncodedSnapshots, torch.Tensor, torch.Tensor]:
        assert h_in is not None
        assert c_in is not None
        self.calls.append(h_in.detach().clone())
        if isinstance(batch, PackedTextBatch):
            state_vec = batch.token_ids[batch.state_positions.to(torch.long)].view(-1, 1).float()
            packed = batch
        else:
            state_vec = batch.token_ids[:, :1].float()
            packed = pack_batch(batch)
        state_vec = state_vec + 10.0 * h_in[-1]
        encoded = torch.zeros((int(packed.token_ids.shape[0]), 1), dtype=torch.float32)
        snaps = EncodedSnapshots(
            card_vectors=torch.zeros((int(batch.token_ids.shape[0]), 0, 1)),
            card_mask=torch.zeros((int(batch.token_ids.shape[0]), 0), dtype=torch.bool),
            state_vector=state_vec,
            encoded=encoded,
            packed=packed,
        )
        return snaps, state_vec.unsqueeze(0), c_in


def _tiny_encoded_batch() -> TextEncodedBatch:
    return TextEncodedBatch(
        token_ids=torch.tensor([[1, 0], [2, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 0], [1, 0]], dtype=torch.long),
        card_ref_positions=torch.full((2, 256), -1, dtype=torch.long),
        seq_lengths=torch.tensor([1, 1], dtype=torch.long),
        spec_tokens=torch.zeros((2, 0), dtype=torch.int32),
        spec_lens=torch.zeros((2,), dtype=torch.int32),
        decision_type=torch.full((2,), int(DecisionType.PRIORITY), dtype=torch.int32),
        pointer_anchor_positions=torch.full((2, 0), -1, dtype=torch.int32),
        pointer_anchor_kinds=torch.full((2, 0), -1, dtype=torch.int32),
        pointer_anchor_subjects=torch.full((2, 0), -1, dtype=torch.int32),
        pointer_anchor_handles=torch.full((2, 0), -1, dtype=torch.int32),
        total_tokens=2,
        seq_lengths_host=(1, 1),
    )


def test_sequenced_forward_uses_rl_style_lstm_history() -> None:
    decoder = ForgeDecoderBatch(
        encoded=_tiny_encoded_batch(),
        output_token_ids=torch.zeros((2, 1), dtype=torch.long),
        output_pointer_pos=torch.full((2, 1), -1, dtype=torch.long),
        output_is_pointer=torch.zeros((2, 1), dtype=torch.bool),
        output_pad_mask=torch.ones((2, 1), dtype=torch.bool),
        vocab_mask=torch.zeros((2, 1, GRAMMAR_VOCAB_SIZE), dtype=torch.bool),
        pointer_mask=torch.zeros((2, 1, 2), dtype=torch.bool),
        decision_type_per_row=torch.full((2,), int(DecisionType.PRIORITY), dtype=torch.long),
        value_targets=torch.tensor([0.0, 12.0], dtype=torch.float32),
    )
    decoder.vocab_mask[:, :, 0] = True
    batch = ForgeSequencedBatch(
        decoder=decoder,
        cell_game_idx=torch.tensor([0, 0], dtype=torch.long),
        cell_pos_idx=torch.tensor([0, 1], dtype=torch.long),
        loss_mask=torch.tensor([False, True], dtype=torch.bool),
        game_lengths=torch.tensor([2], dtype=torch.long),
        n_games=1,
        l_max=2,
        cell_indices_by_timestep=(
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        ),
        cell_indices_by_timestep_host=((0,), (1,)),
        game_indices_by_timestep=(
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ),
        loss_cell_indices_by_timestep=(
            torch.empty((0,), dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        ),
        loss_active_pos_by_timestep=(
            torch.empty((0,), dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ),
    )
    trainer = object.__new__(ForgePolicyValueTrainer)
    trainer.policy = _FakeRecurrentPolicy()
    trainer.cfg = ForgePolicyValueConfig(data_path=Path("."), batch_size=1, eval_fraction=0.0)

    out = trainer._sequenced_forward(batch)

    assert torch.allclose(trainer.policy.calls[0].squeeze(), torch.tensor(0.0))
    assert torch.allclose(trainer.policy.calls[1].squeeze(), torch.tensor(1.0))
    assert torch.allclose(out.value_predictions, torch.tensor([12.0]))
    assert torch.allclose(out.value_loss, torch.tensor(0.0))
