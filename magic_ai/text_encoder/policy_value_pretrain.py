"""Policy + value pretraining on extracted Forge choice situations.

Consumes the sharded torch, gzip JSONL, or Arrow artifact produced by
``scripts/extract_forge_choice_situations.py`` / ``rust/forge_extract``.
Each record stores a pre-choice snapshot, the action text Forge actually
took, and the terminal outcome.

The pipeline trains the autoregressive grammar decoder: it renders the
decision spec via :class:`DecisionSpecRenderer`, translates the observed
event into a flat decoder target via
:mod:`magic_ai.text_encoder.forge_target_encoding`, and runs the
:func:`decoder_cross_entropy_loss` loss together with the value head.

The on-disk record schema is the V2 extractor format which persists
``DecoderTarget`` and ``PendingState`` directly; older V1 records are no
longer supported (the cutover from inline-blank training dropped the
synthetic-pending path).
"""

from __future__ import annotations

import gzip
import hashlib
import math
from array import array
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.game_state import GameStateSnapshot, PendingState
from magic_ai.text_encoder.batch import (
    TextEncodedBatch,
    collate,
    tokenize_snapshot,
)
from magic_ai.text_encoder.decision_spec import AnchorKind, DecisionSpec, DecisionType
from magic_ai.text_encoder.forge_target_encoding import (
    DecoderTarget,
    pending_decision_type,
)
from magic_ai.text_encoder.forge_target_encoding import (
    translate as translate_observed_to_target,
)
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, batch_next_mask
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy
from magic_ai.text_encoder.render import OracleEntry, RenderError, render_snapshot
from magic_ai.text_encoder.render_spec import DecisionSpecRenderer
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS
from magic_ai.text_encoder.training import (
    decoder_cross_entropy_loss,
    decoder_per_step_accuracy,
    value_loss,
)

ValueTargetMode = Literal["terminal", "gae", "vtrace"]

ARROW_CORPUS_FORMAT = "forge_pretrain_decision_arrow"
ARROW_CORPUS_FORMAT_VERSION = 1
ARROW_EXTRACTED_STAGE = "extracted"
_ARROW_KIND_NAMES = ("priority", "attack", "block", "may", "choose")
_ARROW_EXTRACTED_COLUMNS = (
    "format_version",
    "game_id",
    "archive_member",
    "kind_id",
    "candidate_index",
    "candidate_count",
    "source_seq",
    "target_seq",
    "perspective_id",
    "perspective_name",
    "winner_id",
    "winner_name",
    "terminal_sign",
    "snapshot_json",
    "observed_json",
    "outcome_players_json",
    "outcome_extras_json",
)


@dataclass(frozen=True)
class ForgePolicyValueConfig:
    data_path: Path
    batch_size: int
    max_tokens: int | None = None
    eval_fraction: float = 0.05
    gamma: float = 1.0
    value_target_mode: ValueTargetMode = "terminal"
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    pad_token_id: int = 0
    # Sequenced pretraining: when ``sequence_mode == "full"``, each batch
    # element is a whole game replayed through the LSTM in temporal order;
    # losses fire on a random sample of ``loss_positions_per_game`` cells.
    # In ``"none"`` mode the trainer falls back to per-record IID batches
    # (the legacy path that bypasses the LSTM entirely).
    sequence_mode: Literal["none", "full"] = "full"
    games_per_batch: int = 2
    loss_positions_per_game: int = 16
    max_decisions_per_game: int = 64


@dataclass(frozen=True)
class ForgeDecoderBatch:
    """Decoder pipeline batch.

    ``encoded`` carries the combined ``[state, spec]`` token stream and
    pointer-anchor metadata produced by :func:`magic_ai.text_encoder.batch.collate`.
    """

    encoded: TextEncodedBatch
    output_token_ids: Tensor
    output_pointer_pos: Tensor
    output_is_pointer: Tensor
    output_pad_mask: Tensor
    vocab_mask: Tensor
    pointer_mask: Tensor
    decision_type_per_row: Tensor
    value_targets: Tensor


@dataclass(frozen=True)
class ForgeSequencedBatch:
    """Sequenced pretrain batch: ``B`` games × ``L_max`` decisions, flattened.

    The flat dimension ``N = sum_b L_b`` carries every valid (game, position)
    cell. ``cell_game_idx`` and ``cell_pos_idx`` say which (b, l) each flat
    row belongs to, so the trainer can scatter per-cell encoder outputs back
    into ``[B, L_max, D]`` for the LSTM scan over the L axis.

    ``loss_mask`` (shape ``[N]``) marks which cells contribute to the
    decoder + value losses — typically ``loss_positions_per_game`` per
    game, sampled uniformly without replacement.
    """

    decoder: ForgeDecoderBatch
    cell_game_idx: Tensor  # [N]
    cell_pos_idx: Tensor  # [N]
    loss_mask: Tensor  # [N] bool
    game_lengths: Tensor  # [B] int — number of valid cells in each game
    n_games: int
    l_max: int


@dataclass(frozen=True)
class _PreparedDecoderExample:
    """Per-row payload for the decoder pipeline."""

    encoded: Any  # TextEncodedExample
    spec: DecisionSpec
    target: DecoderTarget
    value_target: float
    # subject_index → encoder_position, by anchor kind, for fast pointer lookup.
    anchor_pos_by_kind: dict[int, list[int]] = field(default_factory=dict)


def _stable_eval_bucket(game_id: str, bucket: int) -> int:
    digest = hashlib.blake2b(game_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % bucket


def _arrow_manifest_path(path: Path) -> Path | None:
    manifest_path = path / "manifest.json" if path.is_dir() else path.with_name("manifest.json")
    if not manifest_path.exists():
        return None
    manifest = orjson.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError(f"Arrow manifest {manifest_path} must be a JSON object")
    return manifest_path if manifest.get("format") == ARROW_CORPUS_FORMAT else None


def _arrow_shard_paths(path: Path) -> list[Path]:
    manifest_path = _arrow_manifest_path(path)
    if manifest_path is None:
        raise ValueError(f"{path} is not a {ARROW_CORPUS_FORMAT} corpus")
    manifest = orjson.loads(manifest_path.read_bytes())
    version = int(manifest.get("format_version") or -1)
    if version != ARROW_CORPUS_FORMAT_VERSION:
        raise ValueError(
            f"unsupported Arrow corpus format_version={version}; "
            f"expected {ARROW_CORPUS_FORMAT_VERSION}"
        )
    stage = str(manifest.get("stage") or "")
    if stage != ARROW_EXTRACTED_STAGE:
        raise ValueError(
            f"unsupported Arrow corpus stage={stage!r}; "
            "ForgeChoiceDataset currently consumes the extracted schema"
        )

    root = manifest_path.parent
    shard_count = int(manifest.get("shards") or 0)
    if shard_count > 0:
        shard_paths = [root / f"part-{i:06d}.arrow" for i in range(shard_count)]
        missing = [p for p in shard_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Arrow corpus is missing shard {missing[0]}")
    else:
        shard_paths = sorted(root.glob("part-*.arrow"))
    if not shard_paths:
        raise ValueError(f"Arrow corpus {root} contains no part-*.arrow shards")
    return shard_paths


def _iter_arrow_records(path: Path) -> Iterator[dict[str, Any]]:
    shard_paths = _arrow_shard_paths(path)
    for shard_path in shard_paths:
        yield from _iter_arrow_shard_records(shard_path)


def _validate_arrow_schema(schema: pa.Schema, path: Path) -> None:
    metadata = schema.metadata or {}
    schema_format = metadata.get(b"format")
    if schema_format is not None and schema_format.decode() != ARROW_CORPUS_FORMAT:
        raise ValueError(f"unsupported Arrow shard format metadata in {path}")
    schema_stage = metadata.get(b"stage")
    if schema_stage is not None and schema_stage.decode() != ARROW_EXTRACTED_STAGE:
        raise ValueError(f"unsupported Arrow shard stage metadata in {path}")
    missing = [name for name in _ARROW_EXTRACTED_COLUMNS if schema.get_field_index(name) < 0]
    if missing:
        raise ValueError(f"Arrow shard {path} missing required columns: {', '.join(missing)}")


def _iter_arrow_shard_records(path: Path) -> Iterator[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        reader = pa_ipc.RecordBatchFileReader(source)
        _validate_arrow_schema(reader.schema, path)
        for batch_index in range(reader.num_record_batches):
            yield from _iter_arrow_batch_records(reader.get_batch(batch_index))


def _iter_arrow_batch_records(batch: pa.RecordBatch) -> Iterator[dict[str, Any]]:
    for i in range(batch.num_rows):
        yield _arrow_record_from_batch(batch, i)


def _arrow_scalar(batch: pa.RecordBatch, column: str, row: int) -> Any:
    return batch.column(column)[row].as_py()


def _arrow_json(batch: pa.RecordBatch, column: str, row: int) -> Any:
    return orjson.loads(cast(str, _arrow_scalar(batch, column, row)))


def _arrow_record_from_batch(batch: pa.RecordBatch, row: int) -> dict[str, Any]:
    kind_id = int(_arrow_scalar(batch, "kind_id", row))
    try:
        kind = _ARROW_KIND_NAMES[kind_id]
    except IndexError as exc:
        raise ValueError(f"unknown Arrow choice kind_id={kind_id}") from exc
    return {
        "format": "forge_choice_situation",
        "format_version": int(_arrow_scalar(batch, "format_version", row)),
        "game_id": str(_arrow_scalar(batch, "game_id", row)),
        "archive_member": str(_arrow_scalar(batch, "archive_member", row)),
        "choice": {
            "kind": kind,
            "candidate_index": int(_arrow_scalar(batch, "candidate_index", row)),
            "candidate_count": int(_arrow_scalar(batch, "candidate_count", row)),
            "source_seq": int(_arrow_scalar(batch, "source_seq", row)),
            "target_seq": int(_arrow_scalar(batch, "target_seq", row)),
            "perspective_id": str(_arrow_scalar(batch, "perspective_id", row)),
            "perspective_name": str(_arrow_scalar(batch, "perspective_name", row)),
            "observed": _arrow_json(batch, "observed_json", row),
        },
        "state": {
            "snapshot": _arrow_json(batch, "snapshot_json", row),
        },
        "outcome": {
            "winner_id": _arrow_scalar(batch, "winner_id", row),
            "winner_name": _arrow_scalar(batch, "winner_name", row),
            "terminal_sign": float(_arrow_scalar(batch, "terminal_sign", row)),
            "players": _arrow_json(batch, "outcome_players_json", row),
            "extras": _arrow_json(batch, "outcome_extras_json", row),
        },
    }


class _ArrowBatchCache:
    def __init__(self, shard_paths: Sequence[Path], *, max_batches: int = 8) -> None:
        self._shard_paths = list(shard_paths)
        self._max_batches = max_batches
        self._batches: OrderedDict[tuple[int, int], pa.RecordBatch] = OrderedDict()

    def get(self, shard_idx: int, batch_idx: int) -> pa.RecordBatch:
        key = (shard_idx, batch_idx)
        batch = self._batches.get(key)
        if batch is not None:
            self._batches.move_to_end(key)
            return batch

        path = self._shard_paths[shard_idx]
        with pa.memory_map(str(path), "r") as source:
            reader = pa_ipc.RecordBatchFileReader(source)
            _validate_arrow_schema(reader.schema, path)
            batch = reader.get_batch(batch_idx)
        self._batches[key] = batch
        if len(self._batches) > self._max_batches:
            self._batches.popitem(last=False)
        return batch


class _ArrowCorpusIndex:
    """Compact lazy index for large Rust-emitted Arrow corpora.

    The index stores only primitive row references plus game spans. Full JSON
    payloads are decoded from Arrow on demand when a batch needs them.
    """

    def __init__(
        self,
        *,
        shard_paths: Sequence[Path],
        record_shards: array[int],
        record_batches: array[int],
        record_rows: array[int],
        game_shards: array[int],
        game_batches: array[int],
        game_starts: array[int],
        game_lengths: array[int],
        kind_counts: dict[str, int],
    ) -> None:
        self.shard_paths = list(shard_paths)
        self.record_shards = np.frombuffer(record_shards, dtype=np.uint32)
        self.record_batches = np.frombuffer(record_batches, dtype=np.uint32)
        self.record_rows = np.frombuffer(record_rows, dtype=np.uint32)
        self.game_shards = np.frombuffer(game_shards, dtype=np.uint32)
        self.game_batches = np.frombuffer(game_batches, dtype=np.uint32)
        self.game_starts = np.frombuffer(game_starts, dtype=np.uint32)
        self.game_lengths = np.frombuffer(game_lengths, dtype=np.uint32)
        self.kind_counts = kind_counts
        self._cache = _ArrowBatchCache(self.shard_paths)

    @classmethod
    def build(
        cls,
        path: Path,
        *,
        split: str,
        eval_fraction: float,
    ) -> _ArrowCorpusIndex:
        shard_paths = _arrow_shard_paths(path)
        bucket = 0 if eval_fraction <= 0 else max(2, int(round(1.0 / eval_fraction)))

        record_shards: array[int] = array("I")
        record_batches: array[int] = array("I")
        record_rows: array[int] = array("I")
        game_shards: array[int] = array("I")
        game_batches: array[int] = array("I")
        game_starts: array[int] = array("I")
        game_lengths: array[int] = array("I")
        kind_counts = dict.fromkeys(_ARROW_KIND_NAMES, 0)

        for shard_idx, shard_path in enumerate(shard_paths):
            with pa.memory_map(str(shard_path), "r") as source:
                reader = pa_ipc.RecordBatchFileReader(source)
                _validate_arrow_schema(reader.schema, shard_path)
                for batch_idx in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_idx)
                    game_ids = batch.column("game_id").to_pylist()
                    kind_ids = batch.column("kind_id").to_numpy(zero_copy_only=False)
                    row = 0
                    while row < batch.num_rows:
                        game_id = str(game_ids[row])
                        end = row + 1
                        while end < batch.num_rows and game_ids[end] == game_id:
                            end += 1
                        include = True
                        if bucket > 0 and split != "all":
                            in_eval = _stable_eval_bucket(game_id, bucket) == 0
                            include = (split == "eval" and in_eval) or (
                                split == "train" and not in_eval
                            )
                        if include:
                            game_shards.append(shard_idx)
                            game_batches.append(batch_idx)
                            game_starts.append(row)
                            game_lengths.append(end - row)
                            for record_row in range(row, end):
                                record_shards.append(shard_idx)
                                record_batches.append(batch_idx)
                                record_rows.append(record_row)
                                kind_id = int(kind_ids[record_row])
                                if 0 <= kind_id < len(_ARROW_KIND_NAMES):
                                    kind = _ARROW_KIND_NAMES[kind_id]
                                    kind_counts[kind] = kind_counts.get(kind, 0) + 1
                        row = end

        return cls(
            shard_paths=shard_paths,
            record_shards=record_shards,
            record_batches=record_batches,
            record_rows=record_rows,
            game_shards=game_shards,
            game_batches=game_batches,
            game_starts=game_starts,
            game_lengths=game_lengths,
            kind_counts={k: v for k, v in kind_counts.items() if v},
        )

    @property
    def n_records(self) -> int:
        return int(self.record_rows.shape[0])

    @property
    def n_games(self) -> int:
        return int(self.game_lengths.shape[0])

    def record(self, index: int) -> dict[str, Any]:
        batch = self._cache.get(int(self.record_shards[index]), int(self.record_batches[index]))
        return _arrow_record_from_batch(batch, int(self.record_rows[index]))

    def game_records(self, index: int, *, cap: int) -> list[dict[str, Any]]:
        shard_idx = int(self.game_shards[index])
        batch_idx = int(self.game_batches[index])
        start = int(self.game_starts[index])
        length = min(int(self.game_lengths[index]), cap)
        batch = self._cache.get(shard_idx, batch_idx)
        records = [_arrow_record_from_batch(batch, row) for row in range(start, start + length)]
        records.sort(key=lambda r: int((r.get("choice") or {}).get("source_seq") or 0))
        return records


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if _arrow_manifest_path(path) is not None:
        yield from _iter_arrow_records(path)
        return
    paths = (
        sorted(
            [
                *path.rglob("*.pt"),
                *path.rglob("*.pth"),
                *path.rglob("*.pt.gz"),
                *path.rglob("*.pth.gz"),
                *path.rglob("*.jsonl.gz"),
                *path.rglob("*.arrow"),
            ]
        )
        if path.is_dir()
        else [path]
    )
    for item in paths:
        if item.suffix == ".arrow":
            yield from _iter_arrow_records(item)
            continue
        if item.name.endswith((".pt.gz", ".pth.gz")):
            with gzip.open(item, "rb") as fh:
                payload = torch.load(cast(Any, fh), map_location="cpu", weights_only=False)
            records = payload.get("records") if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise ValueError(f"torch choice artifact {item} does not contain a records list")
            yield from cast(list[dict[str, Any]], records)
            continue
        if item.suffix in (".pt", ".pth"):
            payload = torch.load(item, map_location="cpu", weights_only=False)
            records = payload.get("records") if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise ValueError(f"torch choice artifact {item} does not contain a records list")
            yield from cast(list[dict[str, Any]], records)
            continue
        opener = gzip.open if item.suffix == ".gz" else open
        with opener(item, "rt", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    yield cast(dict[str, Any], orjson.loads(stripped))


def _value_target(record: dict[str, Any], cfg: ForgePolicyValueConfig) -> float:
    sign = float((record.get("outcome") or {}).get("terminal_sign") or 0.0)
    if cfg.value_target_mode == "terminal":
        return sign
    choice = record.get("choice") or {}
    remaining = max(
        0, int(choice.get("candidate_count") or 1) - int(choice.get("candidate_index") or 0) - 1
    )
    return float(sign * (cfg.gamma**remaining))


class ForgeChoiceDataset:
    def __init__(
        self,
        cfg: ForgePolicyValueConfig,
        *,
        tokenizer: PreTrainedTokenizerFast,
        oracle: dict[str, OracleEntry],
        split: str = "train",
    ) -> None:
        if split not in ("train", "eval", "all"):
            raise ValueError(f"split must be train/eval/all, got {split!r}")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.oracle = oracle
        self._spec_renderer = DecisionSpecRenderer(tokenizer)
        self._arrow_index: _ArrowCorpusIndex | None = None
        self.records: list[dict[str, Any]] = []
        # Group records by game_id while preserving input order (the sharder
        # is game-atomic so records arrive grouped).
        self.games: list[list[dict[str, Any]]] = []
        if _arrow_manifest_path(cfg.data_path) is not None:
            self._arrow_index = _ArrowCorpusIndex.build(
                cfg.data_path,
                split=split,
                eval_fraction=cfg.eval_fraction,
            )
        else:
            bucket = 0 if cfg.eval_fraction <= 0 else max(2, int(round(1.0 / cfg.eval_fraction)))
            current_game: list[dict[str, Any]] | None = None
            current_game_id: str | None = None
            for record in _iter_records(cfg.data_path):
                game_id = str(record.get("game_id") or "")
                if bucket > 0 and split != "all":
                    in_eval = _stable_eval_bucket(game_id, bucket) == 0
                    if split == "train" and in_eval:
                        continue
                    if split == "eval" and not in_eval:
                        continue
                self.records.append(record)
                if current_game_id is None or game_id != current_game_id:
                    current_game = []
                    self.games.append(current_game)
                    current_game_id = game_id
                assert current_game is not None
                current_game.append(record)
            # Within each game, sort by source_seq to guarantee temporal order
            # even if upstream shuffled (defense in depth — the rust extractor
            # already emits sorted, and game-atomic shards preserve that).
            for game in self.games:
                game.sort(key=lambda r: int((r.get("choice") or {}).get("source_seq") or 0))
        if self.n_examples == 0:
            raise ValueError(f"no Forge choice records loaded from {cfg.data_path} split={split}")

    @property
    def n_examples(self) -> int:
        if self._arrow_index is not None:
            return self._arrow_index.n_records
        return len(self.records)

    @property
    def n_games(self) -> int:
        if self._arrow_index is not None:
            return self._arrow_index.n_games
        return len(self.games)

    def kind_counts(self) -> dict[str, int]:
        if self._arrow_index is not None:
            return dict(self._arrow_index.kind_counts)
        out: dict[str, int] = {}
        for record in self.records:
            kind = str((record.get("choice") or {}).get("kind") or "unknown")
            out[kind] = out.get(kind, 0) + 1
        return out

    def _record_at(self, index: int) -> dict[str, Any]:
        if self._arrow_index is not None:
            return self._arrow_index.record(index)
        return self.records[index]

    def _game_records(self, index: int) -> list[dict[str, Any]]:
        if self._arrow_index is not None:
            return self._arrow_index.game_records(index, cap=self.cfg.max_decisions_per_game)
        return self.games[index][: self.cfg.max_decisions_per_game]

    def _prepare_decoder(self, record: dict[str, Any]) -> _PreparedDecoderExample | None:
        choice = record.get("choice") or {}
        observed = cast(dict[str, Any], choice.get("observed") or {})
        snapshot = cast(dict[str, Any], (record.get("state") or {}).get("snapshot") or {})
        pending_raw = snapshot.get("pending")
        if pending_raw is None:
            return None
        pending = cast(PendingState, pending_raw)
        decision_type = pending_decision_type(pending)
        if decision_type is None:
            return None
        target = translate_observed_to_target(pending, observed)
        if target is None or not target.output_token_ids:
            return None

        try:
            rendered = render_snapshot(
                cast(GameStateSnapshot, snapshot),
                oracle=self.oracle,
                max_card_refs=MAX_CARD_REFS,
            )
            encoded = tokenize_snapshot(rendered, self.tokenizer)
            spec = self._spec_renderer.render(
                cast(GameStateSnapshot, snapshot), card_refs=rendered.card_refs
            )
        except RenderError, RuntimeError, KeyError, TypeError, ValueError, NotImplementedError:
            return None

        # Build per-kind subject_index → encoder_position lookup; the
        # spec anchors are positioned relative to the spec section start,
        # so add the row's state-token length here (the same offset the
        # collator applies).
        state_len = len(encoded.token_ids)
        anchor_pos_by_kind: dict[int, list[int]] = {}
        for anchor in spec.anchors:
            arr = anchor_pos_by_kind.setdefault(int(anchor.kind), [])
            while len(arr) <= anchor.subject_index:
                arr.append(-1)
            arr[anchor.subject_index] = int(anchor.token_position) + state_len

        return _PreparedDecoderExample(
            encoded=encoded,
            spec=spec,
            target=target,
            value_target=_value_target(record, self.cfg),
            anchor_pos_by_kind=anchor_pos_by_kind,
        )

    def _batch_from_indices(self, indices: Sequence[int]) -> ForgeDecoderBatch:
        prepared: list[_PreparedDecoderExample] = []
        cursor = 0
        while len(prepared) < len(indices) and cursor < len(indices) * 4:
            item = self._prepare_decoder(self._record_at(int(indices[cursor % len(indices)])))
            cursor += 1
            if item is not None:
                prepared.append(item)
        if not prepared:
            raise ValueError("no renderable Forge decoder examples in selected batch")
        return self._batch_from_prepared(prepared)

    def _batch_from_prepared(self, prepared: list[_PreparedDecoderExample]) -> ForgeDecoderBatch:
        if not prepared:
            raise ValueError("_batch_from_prepared got empty list")

        encoded = collate(
            [p.encoded for p in prepared],
            [p.spec for p in prepared],
            pad_id=self.cfg.pad_token_id,
        )

        batch_size = len(prepared)
        L = max(len(p.target.output_token_ids) for p in prepared)
        T_enc = int(encoded.token_ids.shape[1])

        out_tokens = torch.zeros((batch_size, L), dtype=torch.long)
        # Sentinel -1 for unfilled pointer targets: CE/gather will fail loud
        # if pad_mask has a hole that lets a padded slot reach the loss.
        out_pointer_pos = torch.full((batch_size, L), -1, dtype=torch.long)
        out_is_pointer = torch.zeros((batch_size, L), dtype=torch.bool)
        out_pad_mask = torch.zeros((batch_size, L), dtype=torch.bool)
        decision_type_per_row = torch.empty((batch_size,), dtype=torch.long)
        values = torch.empty((batch_size,), dtype=torch.float32)

        vocab_mask = torch.zeros((batch_size, L, GRAMMAR_VOCAB_SIZE), dtype=torch.bool)
        pointer_mask = torch.zeros((batch_size, L, T_enc), dtype=torch.bool)

        for b, item in enumerate(prepared):
            decision_type_per_row[b] = int(item.target.decision_type)
            values[b] = float(item.value_target)
            tokens = item.target.output_token_ids
            subjects = item.target.output_pointer_subjects
            is_ptrs = item.target.output_is_pointer
            n = len(tokens)
            out_pad_mask[b, :n] = True
            for i, (tok, subj, is_ptr) in enumerate(zip(tokens, subjects, is_ptrs, strict=True)):
                out_tokens[b, i] = int(tok)
                out_is_pointer[b, i] = bool(is_ptr)
                if is_ptr:
                    kind = _expected_pointer_kind(item.spec, item.target, i)
                    positions = item.anchor_pos_by_kind.get(int(kind), [])
                    if 0 <= subj < len(positions) and positions[subj] >= 0:
                        out_pointer_pos[b, i] = positions[subj]
                    else:
                        out_pad_mask[b, i] = False

        prefix_tokens_np = np.zeros((batch_size, L), dtype=np.int64)
        prefix_subjects_np = np.full((batch_size, L), -1, dtype=np.int64)
        prefix_lens_np = np.zeros((batch_size,), dtype=np.int64)
        for b, item in enumerate(prepared):
            for i, (tok, subj) in enumerate(
                zip(item.target.output_token_ids, item.target.output_pointer_subjects, strict=True)
            ):
                prefix_tokens_np[b, i] = int(tok)
                prefix_subjects_np[b, i] = int(subj)

        specs = [p.spec for p in prepared]
        per_row_max_prefix = np.array(
            [max(0, len(p.target.output_token_ids) - 1) for p in prepared], dtype=np.int64
        )
        for step in range(L):
            safe_prefix_lens = np.minimum(prefix_lens_np, per_row_max_prefix)
            v_mask, _ptr_mask_subj = batch_next_mask(
                specs,
                prefix_tokens_np,
                prefix_subjects_np,
                safe_prefix_lens,
            )
            vocab_mask[:, step, :] = torch.from_numpy(v_mask)
            for b, item in enumerate(prepared):
                if step >= len(item.target.output_token_ids):
                    continue
                if not item.target.output_is_pointer[step]:
                    continue
                kind = _expected_pointer_kind(item.spec, item.target, step)
                positions = item.anchor_pos_by_kind.get(int(kind), [])
                for pos in positions:
                    if 0 <= pos < T_enc:
                        pointer_mask[b, step, pos] = True
            for b, item in enumerate(prepared):
                if step < len(item.target.output_token_ids):
                    prefix_lens_np[b] = step + 1

        if (
            self.cfg.max_tokens is not None
            and int(encoded.token_ids.shape[1]) > self.cfg.max_tokens
        ):
            cap = int(self.cfg.max_tokens)
            encoded = _truncate_encoded_batch(encoded, max_tokens=cap)
            beyond = out_pointer_pos >= cap
            out_pad_mask = out_pad_mask & ~(out_is_pointer & beyond)
            pointer_mask = pointer_mask[:, :, :cap]

        return ForgeDecoderBatch(
            encoded=encoded,
            output_token_ids=out_tokens,
            output_pointer_pos=out_pointer_pos,
            output_is_pointer=out_is_pointer,
            output_pad_mask=out_pad_mask,
            vocab_mask=vocab_mask,
            pointer_mask=pointer_mask,
            decision_type_per_row=decision_type_per_row,
            value_targets=values,
        )

    def iter_epoch(self, batch_size: int, rng: np.random.Generator) -> Iterator[ForgeDecoderBatch]:
        if self._arrow_index is not None:
            for _ in range(self.n_examples // batch_size):
                yield self.sample_batch(batch_size, rng)
            return
        order = rng.permutation(len(self.records))
        end = (len(order) // batch_size) * batch_size
        for off in range(0, end, batch_size):
            yield self._batch_from_indices([int(i) for i in order[off : off + batch_size]])

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> ForgeDecoderBatch:
        indices = rng.integers(0, self.n_examples, size=batch_size)
        return self._batch_from_indices([int(i) for i in indices])

    def iter_epoch_games(
        self,
        games_per_batch: int,
        loss_positions_per_game: int,
        rng: np.random.Generator,
    ) -> Iterator[ForgeSequencedBatch]:
        if self._arrow_index is not None:
            for _ in range(self.n_games // games_per_batch):
                yield self.sample_sequenced_batch(
                    games_per_batch,
                    loss_positions_per_game,
                    rng,
                )
            return
        order = rng.permutation(self.n_games)
        end = (len(order) // games_per_batch) * games_per_batch
        for off in range(0, end, games_per_batch):
            batch = self._sequenced_batch_from_games(
                [int(i) for i in order[off : off + games_per_batch]],
                loss_positions_per_game,
                rng,
            )
            if batch is not None:
                yield batch

    def sample_sequenced_batch(
        self,
        games_per_batch: int,
        loss_positions_per_game: int,
        rng: np.random.Generator,
    ) -> ForgeSequencedBatch:
        # Eval-side sampler. Skip degenerate batches (e.g. a draw of games
        # that all fail to render any cell) by retrying.
        for _ in range(8):
            indices = rng.integers(0, self.n_games, size=games_per_batch).tolist()
            batch = self._sequenced_batch_from_games(
                [int(i) for i in indices], loss_positions_per_game, rng
            )
            if batch is not None:
                return batch
        raise RuntimeError("could not sample a non-empty sequenced batch in 8 tries")

    def _sequenced_batch_from_games(
        self,
        game_indices: Sequence[int],
        loss_positions_per_game: int,
        rng: np.random.Generator,
    ) -> ForgeSequencedBatch | None:
        """Prepare a batch where each row is a complete game's decision sequence.

        Records that fail to render or to translate to a decoder target are
        dropped from their game's sequence (silently — same behavior as the
        IID path), which can shrink ``L_b`` below the original game length.
        Each game then has ``min(L_b, loss_positions_per_game)`` cells
        sampled uniformly to count toward decoder + value losses.
        """

        prepared_per_game: list[list[_PreparedDecoderExample]] = []
        for gi in game_indices:
            game_records = self._game_records(gi)
            prepared_game: list[_PreparedDecoderExample] = []
            for record in game_records:
                item = self._prepare_decoder(record)
                if item is not None:
                    prepared_game.append(item)
            if prepared_game:
                prepared_per_game.append(prepared_game)
        if not prepared_per_game:
            return None

        l_per_game = [len(g) for g in prepared_per_game]
        l_max = max(l_per_game)
        n_games = len(prepared_per_game)

        # Flatten cells in (game-major, position-major) order so the trainer
        # can scatter encoder outputs back into [B, L_max] via the indices.
        flat: list[_PreparedDecoderExample] = []
        cell_game_idx: list[int] = []
        cell_pos_idx: list[int] = []
        loss_mask_list: list[bool] = []
        for b, game in enumerate(prepared_per_game):
            k = min(loss_positions_per_game, len(game))
            chosen = set(rng.choice(len(game), size=k, replace=False).tolist())
            for pos, item in enumerate(game):
                flat.append(item)
                cell_game_idx.append(b)
                cell_pos_idx.append(pos)
                loss_mask_list.append(pos in chosen)

        # Build the underlying ForgeDecoderBatch from the flat cell list.
        decoder_batch = self._batch_from_prepared(flat)

        return ForgeSequencedBatch(
            decoder=decoder_batch,
            cell_game_idx=torch.tensor(cell_game_idx, dtype=torch.long),
            cell_pos_idx=torch.tensor(cell_pos_idx, dtype=torch.long),
            loss_mask=torch.tensor(loss_mask_list, dtype=torch.bool),
            game_lengths=torch.tensor(l_per_game, dtype=torch.long),
            n_games=n_games,
            l_max=l_max,
        )


def _expected_pointer_kind(
    spec: DecisionSpec, target: DecoderTarget, step_index: int
) -> AnchorKind:
    """Anchor kind expected at ``target.output[step_index]`` (a pointer step)."""

    dt = DecisionType(target.decision_type)
    if dt is DecisionType.PRIORITY:
        return AnchorKind.LEGAL_ACTION
    if dt is DecisionType.CHOOSE_TARGETS:
        return AnchorKind.LEGAL_TARGET
    if dt is DecisionType.DECLARE_ATTACKERS:
        body_off = step_index - 1
        return AnchorKind.LEGAL_ATTACKER if body_off % 4 == 1 else AnchorKind.DEFENDER
    if dt is DecisionType.DECLARE_BLOCKERS:
        body_off = step_index - 1
        return AnchorKind.LEGAL_BLOCKER if body_off % 4 == 1 else AnchorKind.LEGAL_ATTACKER
    raise ValueError(f"decision type {dt} has no pointer steps")


class ForgePolicyValueTrainer:
    def __init__(
        self,
        policy: RecurrentTextPolicy,
        cfg: ForgePolicyValueConfig,
        *,
        lr: float,
        grad_clip: float | None,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)

    def step(self, batch: ForgeDecoderBatch, *, compute_stats: bool = True) -> dict[str, float]:
        return self._decoder_step(batch, compute_stats=compute_stats)

    def sequenced_step(
        self,
        batch: ForgeSequencedBatch,
        *,
        compute_stats: bool = True,
        accum_index: int = 0,
        accum_total: int = 1,
    ) -> dict[str, float]:
        return self._sequenced_step(
            batch,
            compute_stats=compute_stats,
            accum_index=accum_index,
            accum_total=accum_total,
        )

    def _sequenced_step(
        self,
        batch: ForgeSequencedBatch,
        *,
        compute_stats: bool,
        accum_index: int = 0,
        accum_total: int = 1,
    ) -> dict[str, float]:
        """Sequenced pretrain: replay each game through the LSTM in temporal
        order; decoder + value losses fire on a sampled subset of cells.

        The LSTM scan over [B, L_max] gives the value head proper history-
        aggregated state. Per-cell padded LSTM updates beyond ``game_lengths``
        are unused (loss is masked off there), so we don't pay the cost of
        ``pack_padded_sequence``.
        """
        self.policy.train()
        if accum_index == 0:
            self.optimizer.zero_grad(set_to_none=True)
        text_policy = self.policy.text_policy
        if text_policy.grammar_decoder is None:
            raise RuntimeError("decoder pipeline requires TextPolicy with a grammar_decoder")
        device = next(text_policy.parameters()).device

        decoder_batch = batch.decoder
        cell_game_idx = batch.cell_game_idx.to(device)
        cell_pos_idx = batch.cell_pos_idx.to(device)
        loss_mask_flat = batch.loss_mask.to(device)
        n_games = batch.n_games
        l_max = batch.l_max

        # One encoder forward feeds both decoder cross-attn and the value
        # head. Use the packed varlen path (`encode_only` packs internally)
        # to avoid `flex_attention` recompiles, then scatter back to
        # [N, T_enc, D] for the decoder cross-attn — same convention as the
        # rationalized `forward_decoder_teacher_forced`, but inlined here so
        # the encoder isn't re-run for the value head.
        from magic_ai.text_encoder.batch import scatter_packed_to_padded

        encoded_snaps = text_policy.encode_only(decoder_batch.encoded)
        encoded_hidden, attn_mask = scatter_packed_to_padded(
            encoded_snaps.encoded, encoded_snaps.packed
        )
        state_vec = encoded_snaps.state_vector  # [N, D]
        target_tokens = decoder_batch.output_token_ids.to(device)
        vocab_logits, pointer_logits = text_policy.grammar_decoder.forward_teacher_forced(
            target_tokens, encoded_hidden, attn_mask
        )
        # Apply loss mask by zeroing the per-row pad mask on non-loss cells —
        # the existing decoder_cross_entropy_loss reduces over True entries
        # in output_pad_mask.
        out_pad_mask = decoder_batch.output_pad_mask.to(device) & loss_mask_flat.unsqueeze(-1)
        decoder_loss = decoder_cross_entropy_loss(
            vocab_logits,
            pointer_logits,
            target_tokens,
            decoder_batch.output_pointer_pos.to(device),
            decoder_batch.output_is_pointer.to(device),
            decoder_batch.vocab_mask.to(device),
            decoder_batch.pointer_mask.to(device),
            out_pad_mask,
        )

        # Value path: scatter per-cell state_vec into [B, L_max, D], run the
        # LSTM scan with zero init state, decode every position, mask losses.
        d_model = state_vec.shape[-1]
        state_vec_seq = torch.zeros((n_games, l_max, d_model), dtype=state_vec.dtype, device=device)
        state_vec_seq[cell_game_idx, cell_pos_idx] = state_vec

        lstm_input_seq = self.policy.in_proj(state_vec_seq)  # [B, L, lstm_hidden]
        h0 = torch.zeros(
            self.policy.lstm_layers,
            n_games,
            self.policy.lstm_hidden,
            dtype=lstm_input_seq.dtype,
            device=device,
        )
        c0 = torch.zeros_like(h0)
        lstm_out_seq, _ = self.policy.lstm(lstm_input_seq, (h0, c0))
        # out_proj projects back to d_model (the value head's input space).
        state_for_heads_seq = self.policy.out_proj(lstm_out_seq)  # [B, L, D]
        values_seq = text_policy.value_head(state_for_heads_seq)  # [B, L]

        # Gather per-cell predictions back into flat-cell space and apply
        # loss mask for MSE.
        values_flat = values_seq[cell_game_idx, cell_pos_idx]  # [N]
        value_targets_flat = decoder_batch.value_targets.to(device)
        v_loss_per_cell = (values_flat.float() - value_targets_flat) ** 2
        # Reduce only over loss_mask cells; clamp denominator to avoid /0.
        loss_count = loss_mask_flat.float().sum().clamp(min=1.0)
        v_loss = (v_loss_per_cell * loss_mask_flat.float()).sum() / loss_count

        loss = self.cfg.policy_loss_weight * decoder_loss + self.cfg.value_loss_weight * v_loss
        # Scale loss by accum_total so the accumulated gradient matches the
        # average loss across micro-batches (rather than the sum).
        (loss / accum_total).backward()
        is_last_micro = accum_index + 1 == accum_total
        if is_last_micro:
            if self.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            else:
                grad_norm = torch.tensor(0.0, device=loss.device)
            self.optimizer.step()
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)
        if not compute_stats:
            return {}

        with torch.no_grad():
            v_pred_loss = values_flat.float()[loss_mask_flat]
            v_targ_loss = value_targets_flat[loss_mask_flat]
            v_pred_mean = float(v_pred_loss.mean()) if v_pred_loss.numel() > 0 else 0.0
            v_pred_std = float(v_pred_loss.std()) if v_pred_loss.numel() > 1 else 0.0
            v_targ_mean = float(v_targ_loss.mean()) if v_targ_loss.numel() > 0 else 0.0
            v_targ_std = float(v_targ_loss.std()) if v_targ_loss.numel() > 1 else 0.0
            denom = float(v_pred_loss.std() * v_targ_loss.std()) if v_pred_loss.numel() > 1 else 0.0
            if denom > 1e-8:
                centered = (v_pred_loss - v_pred_loss.mean()) * (v_targ_loss - v_targ_loss.mean())
                v_corr = float(centered.mean() / denom)
            else:
                v_corr = 0.0
            non_draw = v_targ_loss.abs() > 1e-6
            sign_acc = (
                float(
                    (torch.sign(v_pred_loss[non_draw]) == torch.sign(v_targ_loss[non_draw]))
                    .float()
                    .mean()
                )
                if non_draw.any()
                else 0.0
            )
            value_head = self.policy.text_policy.value_head
            v_grad_norm = float(
                torch.norm(
                    torch.stack(
                        [
                            p.grad.detach().norm()
                            for p in value_head.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                if any(p.grad is not None for p in value_head.parameters())
                else 0.0
            )
            lstm_grad_norm = float(
                torch.norm(
                    torch.stack(
                        [
                            p.grad.detach().norm()
                            for p in self.policy.lstm.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                if any(p.grad is not None for p in self.policy.lstm.parameters())
                else 0.0
            )

        return {
            "loss": float(loss.detach()),
            "policy_loss": float(decoder_loss.detach()),
            "value_loss": float(v_loss.detach()),
            "v_pred_mean": v_pred_mean,
            "v_pred_std": v_pred_std,
            "v_targ_mean": v_targ_mean,
            "v_targ_std": v_targ_std,
            "v_corr": v_corr,
            "value_sign_accuracy": sign_acc,
            "v_head_grad_norm": v_grad_norm,
            "lstm_grad_norm": lstm_grad_norm,
            "grad_norm": float(grad_norm.detach()),
            "n_loss_cells": int(loss_count.detach()),
        }

    def _decoder_step(self, batch: ForgeDecoderBatch, *, compute_stats: bool) -> dict[str, float]:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        text_policy = self.policy.text_policy
        if text_policy.grammar_decoder is None:
            raise RuntimeError("decoder pipeline requires TextPolicy with a grammar_decoder")
        device = next(text_policy.parameters()).device
        encoded = batch.encoded
        target_tokens = batch.output_token_ids.to(device)
        # One packed encoder forward, used for both the decoder cross-attn
        # (after re-padding) and the value head's per-cell state vector.
        # Pretrain runs without rollout history, so feeding the value head
        # through the LSTM (always with h=c=0) collapses every batch to a
        # near-constant prediction (the LSTM's recurrent weights never see
        # gradient when h_prev is always 0). Read the value head directly
        # off the encoder's state_vector during pretrain — RL rollouts can
        # still use the LSTM path because they carry real history.
        from magic_ai.text_encoder.batch import scatter_packed_to_padded

        encoded_snaps = text_policy.encode_only(encoded)
        encoded_padded, attn_mask = scatter_packed_to_padded(
            encoded_snaps.encoded, encoded_snaps.packed
        )
        vocab_logits, pointer_logits = text_policy.grammar_decoder.forward_teacher_forced(
            target_tokens, encoded_padded, attn_mask
        )
        values = text_policy.value_head(encoded_snaps.state_vector)

        decoder_loss = decoder_cross_entropy_loss(
            vocab_logits,
            pointer_logits,
            target_tokens,
            batch.output_pointer_pos.to(device),
            batch.output_is_pointer.to(device),
            batch.vocab_mask.to(device),
            batch.pointer_mask.to(device),
            batch.output_pad_mask.to(device),
        )
        v_loss = value_loss(values.float(), batch.value_targets.to(values.device))
        loss = self.cfg.policy_loss_weight * decoder_loss + self.cfg.value_loss_weight * v_loss
        loss.backward()
        if self.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)
        self.optimizer.step()
        if not compute_stats:
            return {}
        with torch.no_grad():
            per_step = decoder_per_step_accuracy(
                vocab_logits,
                pointer_logits,
                target_tokens,
                batch.output_pointer_pos.to(device),
                batch.output_is_pointer.to(device),
                batch.vocab_mask.to(device),
                batch.pointer_mask.to(device),
                batch.output_pad_mask.to(device),
            )
            neg_inf = torch.finfo(vocab_logits.dtype).min
            v_pred = vocab_logits.masked_fill(~batch.vocab_mask.to(device), neg_inf).argmax(-1)
            p_pred = pointer_logits.masked_fill(~batch.pointer_mask.to(device), neg_inf).argmax(-1)
            correct_per_step = torch.where(
                batch.output_is_pointer.to(device),
                p_pred == batch.output_pointer_pos.to(device),
                v_pred == target_tokens,
            )
            row_correct = ((~batch.output_pad_mask.to(device)) | correct_per_step).all(dim=-1)
            combat_kinds = (
                (batch.decision_type_per_row == int(DecisionType.DECLARE_ATTACKERS))
                | (batch.decision_type_per_row == int(DecisionType.DECLARE_BLOCKERS))
            ).to(device)
            combat_total = int(combat_kinds.sum().item())
            combat_exact = (
                float((row_correct & combat_kinds).sum().item()) / combat_total
                if combat_total > 0
                else 0.0
            )
            pred_sign = torch.sign(values.float())
            target = batch.value_targets.to(values.device)
            non_draw = target.abs() > 1e-6
            sign_acc = (
                (pred_sign[non_draw] == torch.sign(target[non_draw])).float().mean()
                if non_draw.any()
                else torch.tensor(0.0, device=loss.device)
            )
        with torch.no_grad():
            v_pred_f = values.float()
            v_targ_f = batch.value_targets.to(v_pred_f.device)
            v_pred_mean = float(v_pred_f.mean())
            v_pred_std = float(v_pred_f.std())
            v_targ_mean = float(v_targ_f.mean())
            v_targ_std = float(v_targ_f.std())
            # Pearson correlation between predictions and targets — best
            # diagnostic for "is the head learning anything useful at all?"
            denom = float(v_pred_f.std() * v_targ_f.std())
            v_corr = (
                float(((v_pred_f - v_pred_f.mean()) * (v_targ_f - v_targ_f.mean())).mean() / denom)
                if denom > 1e-8
                else 0.0
            )
            # Gradient norm of just the value head, to confirm gradients are
            # actually reaching it.
            value_head = self.policy.text_policy.value_head
            v_grad_norm = float(
                torch.norm(
                    torch.stack(
                        [
                            p.grad.detach().norm()
                            for p in value_head.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                if any(p.grad is not None for p in value_head.parameters())
                else 0.0
            )
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(decoder_loss.detach()),
            "value_loss": float(v_loss.detach()),
            "decoder_step_accuracy": float(per_step["accuracy"]),
            "decoder_combat_exact_match": float(combat_exact),
            "value_sign_accuracy": float(sign_acc.detach()),
            "grad_norm": float(grad_norm.detach()),
            "v_pred_mean": v_pred_mean,
            "v_pred_std": v_pred_std,
            "v_targ_mean": v_targ_mean,
            "v_targ_std": v_targ_std,
            "v_corr": v_corr,
            "v_head_grad_norm": v_grad_norm,
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataset: ForgeChoiceDataset,
        rng: np.random.Generator,
        *,
        batches: int = 8,
        device: torch.device,
    ) -> dict[str, float]:
        self.policy.eval()
        totals: dict[str, float] = {}
        count = 0
        for _ in range(batches):
            batch = dataset.sample_batch(self.cfg.batch_size, rng)
            batch = _batch_to_device(batch, device)
            text_policy = self.policy.text_policy
            if text_policy.grammar_decoder is None:
                continue
            from magic_ai.text_encoder.batch import scatter_packed_to_padded

            encoded_snaps = text_policy.encode_only(batch.encoded)
            encoded_padded, attn_mask = scatter_packed_to_padded(
                encoded_snaps.encoded, encoded_snaps.packed
            )
            vocab_logits, pointer_logits = text_policy.grammar_decoder.forward_teacher_forced(
                batch.output_token_ids, encoded_padded, attn_mask
            )
            values = text_policy.value_head(encoded_snaps.state_vector)
            policy_l = decoder_cross_entropy_loss(
                vocab_logits,
                pointer_logits,
                batch.output_token_ids,
                batch.output_pointer_pos,
                batch.output_is_pointer,
                batch.vocab_mask,
                batch.pointer_mask,
                batch.output_pad_mask,
            )
            v_loss = value_loss(values.float(), batch.value_targets)
            stats = {
                "eval_policy_loss": float(policy_l.detach()),
                "eval_value_loss": float(v_loss.detach()),
            }
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        denom = max(1, count)
        return {key: value / denom for key, value in totals.items()} | {
            "eval_batches": float(count)
        }

    @torch.no_grad()
    def evaluate_sequenced(
        self,
        dataset: ForgeChoiceDataset,
        rng: np.random.Generator,
        *,
        batches: int = 8,
        device: torch.device,
    ) -> dict[str, float]:
        self.policy.eval()
        totals: dict[str, float] = {}
        count = 0
        for _ in range(batches):
            batch = dataset.sample_sequenced_batch(
                self.cfg.games_per_batch, self.cfg.loss_positions_per_game, rng
            )
            batch = _sequenced_batch_to_device(batch, device)
            text_policy = self.policy.text_policy
            if text_policy.grammar_decoder is None:
                continue
            decoder_batch = batch.decoder
            from magic_ai.text_encoder.batch import scatter_packed_to_padded

            encoded_snaps = text_policy.encode_only(decoder_batch.encoded)
            encoded_hidden, attn_mask = scatter_packed_to_padded(
                encoded_snaps.encoded, encoded_snaps.packed
            )
            state_vec = encoded_snaps.state_vector
            target_tokens = decoder_batch.output_token_ids
            vocab_logits, pointer_logits = text_policy.grammar_decoder.forward_teacher_forced(
                target_tokens, encoded_hidden, attn_mask
            )
            out_pad_mask = decoder_batch.output_pad_mask & batch.loss_mask.unsqueeze(-1)
            policy_l = decoder_cross_entropy_loss(
                vocab_logits,
                pointer_logits,
                target_tokens,
                decoder_batch.output_pointer_pos,
                decoder_batch.output_is_pointer,
                decoder_batch.vocab_mask,
                decoder_batch.pointer_mask,
                out_pad_mask,
            )
            d_model = state_vec.shape[-1]
            state_vec_seq = torch.zeros(
                (batch.n_games, batch.l_max, d_model),
                dtype=state_vec.dtype,
                device=device,
            )
            state_vec_seq[batch.cell_game_idx, batch.cell_pos_idx] = state_vec
            lstm_input_seq = self.policy.in_proj(state_vec_seq)
            h0 = torch.zeros(
                self.policy.lstm_layers,
                batch.n_games,
                self.policy.lstm_hidden,
                dtype=lstm_input_seq.dtype,
                device=device,
            )
            c0 = torch.zeros_like(h0)
            lstm_out_seq, _ = self.policy.lstm(lstm_input_seq, (h0, c0))
            state_for_heads_seq = self.policy.out_proj(lstm_out_seq)
            values_seq = text_policy.value_head(state_for_heads_seq)
            values_flat = values_seq[batch.cell_game_idx, batch.cell_pos_idx]
            v_pred = values_flat.float()[batch.loss_mask]
            v_targ = decoder_batch.value_targets[batch.loss_mask]
            v_loss = ((v_pred - v_targ) ** 2).mean() if v_pred.numel() > 0 else torch.tensor(0.0)
            non_draw = v_targ.abs() > 1e-6
            sign_acc = (
                float((torch.sign(v_pred[non_draw]) == torch.sign(v_targ[non_draw])).float().mean())
                if non_draw.any()
                else 0.0
            )
            stats = {
                "eval_policy_loss": float(policy_l.detach()),
                "eval_value_loss": float(v_loss.detach()),
                "eval_value_sign_accuracy": sign_acc,
                "eval_v_pred_std": float(v_pred.std()) if v_pred.numel() > 1 else 0.0,
            }
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        denom = max(1, count)
        return {key: value / denom for key, value in totals.items()} | {
            "eval_batches": float(count)
        }


def _encoded_to_device(batch: TextEncodedBatch, device: torch.device) -> TextEncodedBatch:
    return TextEncodedBatch(
        token_ids=batch.token_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        card_ref_positions=batch.card_ref_positions.to(device),
        seq_lengths=batch.seq_lengths.to(device),
        total_tokens=batch.total_tokens,
        seq_lengths_host=batch.seq_lengths_host,
        spec_tokens=batch.spec_tokens.to(device),
        spec_lens=batch.spec_lens.to(device),
        decision_type=batch.decision_type.to(device),
        pointer_anchor_positions=batch.pointer_anchor_positions.to(device),
        pointer_anchor_kinds=batch.pointer_anchor_kinds.to(device),
        pointer_anchor_subjects=batch.pointer_anchor_subjects.to(device),
        pointer_anchor_handles=batch.pointer_anchor_handles.to(device),
        legal_edge_bitmap=(
            batch.legal_edge_bitmap.to(device) if batch.legal_edge_bitmap is not None else None
        ),
    )


def _truncate_encoded_batch(batch: TextEncodedBatch, *, max_tokens: int) -> TextEncodedBatch:
    seq_lengths = batch.seq_lengths.clamp(max=max_tokens)
    attention_mask = batch.attention_mask[:, :max_tokens].clone()
    for row, seq_len in enumerate(seq_lengths.tolist()):
        attention_mask[row, int(seq_len) :] = 0
    card_ref_positions = batch.card_ref_positions.clone()
    card_ref_positions[card_ref_positions >= max_tokens] = -1
    return TextEncodedBatch(
        token_ids=batch.token_ids[:, :max_tokens].contiguous(),
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lengths,
        total_tokens=int(seq_lengths.sum().item()),
        seq_lengths_host=tuple(int(v) for v in seq_lengths.tolist()),
        spec_tokens=batch.spec_tokens,
        spec_lens=batch.spec_lens,
        decision_type=batch.decision_type,
        pointer_anchor_positions=batch.pointer_anchor_positions,
        pointer_anchor_kinds=batch.pointer_anchor_kinds,
        pointer_anchor_subjects=batch.pointer_anchor_subjects,
        pointer_anchor_handles=batch.pointer_anchor_handles,
        legal_edge_bitmap=batch.legal_edge_bitmap,
    )


def _batch_to_device(batch: ForgeDecoderBatch, device: torch.device) -> ForgeDecoderBatch:
    return ForgeDecoderBatch(
        encoded=_encoded_to_device(batch.encoded, device),
        output_token_ids=batch.output_token_ids.to(device),
        output_pointer_pos=batch.output_pointer_pos.to(device),
        output_is_pointer=batch.output_is_pointer.to(device),
        output_pad_mask=batch.output_pad_mask.to(device),
        vocab_mask=batch.vocab_mask.to(device),
        pointer_mask=batch.pointer_mask.to(device),
        decision_type_per_row=batch.decision_type_per_row.to(device),
        value_targets=batch.value_targets.to(device),
    )


def _sequenced_batch_to_device(
    batch: ForgeSequencedBatch, device: torch.device
) -> ForgeSequencedBatch:
    return ForgeSequencedBatch(
        decoder=_batch_to_device(batch.decoder, device),
        cell_game_idx=batch.cell_game_idx.to(device),
        cell_pos_idx=batch.cell_pos_idx.to(device),
        loss_mask=batch.loss_mask.to(device),
        game_lengths=batch.game_lengths.to(device),
        n_games=batch.n_games,
        l_max=batch.l_max,
    )


def batches_per_epoch(n_examples: int, batch_size: int) -> int:
    return int(math.floor(n_examples / batch_size))


__all__ = [
    "ForgeChoiceDataset",
    "ForgeDecoderBatch",
    "ForgeSequencedBatch",
    "ForgePolicyValueConfig",
    "ForgePolicyValueTrainer",
    "ValueTargetMode",
    "batches_per_epoch",
    "_batch_to_device",
    "_sequenced_batch_to_device",
]
