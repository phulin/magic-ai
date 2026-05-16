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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as pa_ipc
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.game_state import GameStateSnapshot, PendingState
from magic_ai.text_encoder.batch import (
    PackedTextBatch,
    TextEncodedBatch,
    TokenizationContext,
    collate,
    packed_sequence_layout,
    tokenize_snapshots,
)
from magic_ai.text_encoder.decision_spec import AnchorKind, DecisionSpec, DecisionType
from magic_ai.text_encoder.forge_target_encoding import (
    DecoderTarget,
    pending_decision_type,
)
from magic_ai.text_encoder.forge_target_encoding import (
    translate as translate_observed_to_target,
)
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, batch_next_mask_torch
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy
from magic_ai.text_encoder.render import OracleEntry, RenderError, SnapshotRenderer
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
ARROW_GAME_INDEX_FILENAME = "game_index.arrow"
ARROW_GAME_INDEX_FORMAT = "forge_pretrain_game_index_arrow"
ARROW_GAME_INDEX_FORMAT_VERSION = 1
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
class _ArrowGameSpanCache:
    shard_paths: tuple[Path, ...]
    game_ids: tuple[str, ...]
    game_shards: array[int]
    game_batches: array[int]
    game_starts: array[int]
    game_lengths: array[int]


_ARROW_GAME_SPAN_CACHE: dict[str, _ArrowGameSpanCache] = {}


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
    cell_indices_by_timestep: tuple[Tensor, ...]  # len L_max; flat cell rows active at t
    cell_indices_by_timestep_host: tuple[tuple[int, ...], ...]  # CPU mirror for shape metadata
    game_indices_by_timestep: tuple[Tensor, ...]  # len L_max; game row for each active cell
    loss_cell_indices_by_timestep: tuple[Tensor, ...]  # len L_max; flat loss-cell rows at t
    loss_active_pos_by_timestep: tuple[Tensor, ...]  # len L_max; positions inside active rows


@dataclass(frozen=True)
class _PreparedDecoderExample:
    """Per-row payload for the decoder pipeline."""

    encoded: Any  # TextEncodedExample
    spec: DecisionSpec
    target: DecoderTarget
    value_target: float


@dataclass(frozen=True)
class _DecoderCandidate:
    """Record fields that survive target translation and are worth rendering."""

    snapshot: GameStateSnapshot
    target: DecoderTarget
    value_target: float


@dataclass(frozen=True)
class _SequencedForwardOutput:
    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    value_predictions: Tensor
    value_targets: Tensor
    n_loss_cells: Tensor


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


def _arrow_game_index_path(path: Path) -> Path:
    root = path if path.is_dir() else path.parent
    return root / ARROW_GAME_INDEX_FILENAME


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


def _arrow_game_starts(game_ids: pa.Array) -> list[int]:
    if len(game_ids) == 0:
        return []
    if len(game_ids) == 1:
        return [0, 1]
    changed = pc.call_function(
        "not_equal", [game_ids.slice(1), game_ids.slice(0, len(game_ids) - 1)]
    )
    start_offsets = cast(pa.Array, pc.call_function("indices_nonzero", [changed]))
    starts_arr = start_offsets.to_numpy(zero_copy_only=False)
    starts = [0, *[int(i) + 1 for i in starts_arr]]
    starts.append(len(game_ids))
    return starts


def _arrow_game_index_schema() -> pa.Schema:
    metadata = {
        "format": ARROW_GAME_INDEX_FORMAT,
        "format_version": str(ARROW_GAME_INDEX_FORMAT_VERSION),
    }
    return pa.schema(
        [
            pa.field("game_id", pa.utf8(), nullable=False),
            pa.field("shard_idx", pa.uint32(), nullable=False),
            pa.field("batch_idx", pa.uint32(), nullable=False),
            pa.field("row_start", pa.uint32(), nullable=False),
            pa.field("row_count", pa.uint32(), nullable=False),
        ],
        metadata=metadata,
    )


def _load_arrow_game_index(path: Path, shard_paths: Sequence[Path]) -> _ArrowGameSpanCache | None:
    index_path = _arrow_game_index_path(path)
    if not index_path.exists():
        return None
    with pa.memory_map(str(index_path), "r") as source:
        reader = pa_ipc.RecordBatchFileReader(source)
        metadata = reader.schema.metadata or {}
        fmt = metadata.get(b"format", b"").decode("utf-8")
        version = int(metadata.get(b"format_version", b"-1").decode("utf-8"))
        if fmt != ARROW_GAME_INDEX_FORMAT or version != ARROW_GAME_INDEX_FORMAT_VERSION:
            raise ValueError(
                f"unsupported Arrow game index {index_path}: format={fmt!r} version={version}"
            )
        game_ids: list[str] = []
        game_shards: array[int] = array("I")
        game_batches: array[int] = array("I")
        game_starts: array[int] = array("I")
        game_lengths: array[int] = array("I")
        for batch_idx in range(reader.num_record_batches):
            batch = reader.get_batch(batch_idx)
            game_ids.extend(str(x) for x in batch.column("game_id").to_pylist())
            game_shards.extend(
                int(x) for x in batch.column("shard_idx").to_numpy(zero_copy_only=False)
            )
            game_batches.extend(
                int(x) for x in batch.column("batch_idx").to_numpy(zero_copy_only=False)
            )
            game_starts.extend(
                int(x) for x in batch.column("row_start").to_numpy(zero_copy_only=False)
            )
            game_lengths.extend(
                int(x) for x in batch.column("row_count").to_numpy(zero_copy_only=False)
            )
    out = _ArrowGameSpanCache(
        shard_paths=tuple(shard_paths),
        game_ids=tuple(game_ids),
        game_shards=game_shards,
        game_batches=game_batches,
        game_starts=game_starts,
        game_lengths=game_lengths,
    )
    print(f"[policy-value] loaded Arrow game index games={len(game_ids)}", flush=True)
    return out


def _write_arrow_game_index(path: Path, spans: _ArrowGameSpanCache) -> None:
    index_path = _arrow_game_index_path(path)
    batch = pa.record_batch(
        [
            pa.array(spans.game_ids, type=pa.utf8()),
            pa.array(np.frombuffer(spans.game_shards, dtype=np.uint32), type=pa.uint32()),
            pa.array(np.frombuffer(spans.game_batches, dtype=np.uint32), type=pa.uint32()),
            pa.array(np.frombuffer(spans.game_starts, dtype=np.uint32), type=pa.uint32()),
            pa.array(np.frombuffer(spans.game_lengths, dtype=np.uint32), type=pa.uint32()),
        ],
        schema=_arrow_game_index_schema(),
    )
    tmp = index_path.with_suffix(".arrow.tmp")
    options = pa_ipc.IpcWriteOptions(compression="zstd")
    with pa.OSFile(str(tmp), "wb") as sink:
        with pa_ipc.new_file(sink, batch.schema, options=options) as writer:
            writer.write_batch(batch)
    tmp.replace(index_path)
    print(f"[policy-value] wrote Arrow game index -> {index_path}", flush=True)


def _cached_arrow_game_spans(path: Path, shard_paths: Sequence[Path]) -> _ArrowGameSpanCache:
    cache_key = str(path.resolve())
    cached = _ARROW_GAME_SPAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cached_disk = _load_arrow_game_index(path, shard_paths)
    if cached_disk is not None:
        _ARROW_GAME_SPAN_CACHE[cache_key] = cached_disk
        return cached_disk

    game_ids_out: list[str] = []
    game_shards: array[int] = array("I")
    game_batches: array[int] = array("I")
    game_starts: array[int] = array("I")
    game_lengths: array[int] = array("I")
    progress_stride = max(1, len(shard_paths) // 10)
    for shard_idx, shard_path in enumerate(shard_paths):
        if shard_idx % progress_stride == 0:
            print(
                f"[policy-value] indexing Arrow games shard={shard_idx + 1}/{len(shard_paths)}",
                flush=True,
            )
        with pa.memory_map(str(shard_path), "r") as source:
            reader = pa_ipc.RecordBatchFileReader(source)
            _validate_arrow_schema(reader.schema, shard_path)
            for batch_idx in range(reader.num_record_batches):
                batch = reader.get_batch(batch_idx)
                game_ids = batch.column("game_id")
                starts = _arrow_game_starts(game_ids)
                for start, end in zip(starts[:-1], starts[1:], strict=True):
                    game_ids_out.append(str(game_ids[start].as_py()))
                    game_shards.append(shard_idx)
                    game_batches.append(batch_idx)
                    game_starts.append(start)
                    game_lengths.append(end - start)

    out = _ArrowGameSpanCache(
        shard_paths=tuple(shard_paths),
        game_ids=tuple(game_ids_out),
        game_shards=game_shards,
        game_batches=game_batches,
        game_starts=game_starts,
        game_lengths=game_lengths,
    )
    _ARROW_GAME_SPAN_CACHE[cache_key] = out
    print(f"[policy-value] indexed Arrow games games={len(game_ids_out)}", flush=True)
    _write_arrow_game_index(path, out)
    return out


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
        include_records: bool = True,
    ) -> _ArrowCorpusIndex:
        shard_paths = _arrow_shard_paths(path)
        bucket = 0 if eval_fraction <= 0 else max(2, int(round(1.0 / eval_fraction)))

        if not include_records:
            span_cache = _cached_arrow_game_spans(path, shard_paths)
            game_shards: array[int] = array("I")
            game_batches: array[int] = array("I")
            game_starts: array[int] = array("I")
            game_lengths: array[int] = array("I")
            for i, game_id in enumerate(span_cache.game_ids):
                include = True
                if bucket > 0 and split != "all":
                    in_eval = _stable_eval_bucket(game_id, bucket) == 0
                    include = (split == "eval" and in_eval) or (split == "train" and not in_eval)
                if include:
                    game_shards.append(span_cache.game_shards[i])
                    game_batches.append(span_cache.game_batches[i])
                    game_starts.append(span_cache.game_starts[i])
                    game_lengths.append(span_cache.game_lengths[i])
            return cls(
                shard_paths=span_cache.shard_paths,
                record_shards=array("I"),
                record_batches=array("I"),
                record_rows=array("I"),
                game_shards=game_shards,
                game_batches=game_batches,
                game_starts=game_starts,
                game_lengths=game_lengths,
                kind_counts={},
            )

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
                    if batch.num_rows == 0:
                        continue
                    game_ids = batch.column("game_id")
                    starts = _arrow_game_starts(game_ids)
                    kind_ids = (
                        batch.column("kind_id").to_numpy(zero_copy_only=False)
                        if include_records
                        else None
                    )
                    for start, end in zip(starts[:-1], starts[1:], strict=True):
                        game_id = str(game_ids[start].as_py())
                        include = True
                        if bucket > 0 and split != "all":
                            in_eval = _stable_eval_bucket(game_id, bucket) == 0
                            include = (split == "eval" and in_eval) or (
                                split == "train" and not in_eval
                            )
                        if include:
                            game_shards.append(shard_idx)
                            game_batches.append(batch_idx)
                            game_starts.append(start)
                            game_lengths.append(end - start)
                            if include_records:
                                if kind_ids is None:
                                    raise RuntimeError(
                                        "kind_ids must be loaded when include_records=True"
                                    )
                                for record_row in range(start, end):
                                    record_shards.append(shard_idx)
                                    record_batches.append(batch_idx)
                                    record_rows.append(record_row)
                                    kind_id = int(kind_ids[record_row])
                                    if 0 <= kind_id < len(_ARROW_KIND_NAMES):
                                        kind = _ARROW_KIND_NAMES[kind_id]
                                        kind_counts[kind] = kind_counts.get(kind, 0) + 1

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
        if self.record_rows.shape[0] > 0:
            return int(self.record_rows.shape[0])
        return int(self.game_lengths.sum())

    @property
    def n_games(self) -> int:
        return int(self.game_lengths.shape[0])

    def record(self, index: int) -> dict[str, Any]:
        if self.record_rows.shape[0] == 0:
            raise RuntimeError("Arrow index was built without per-record rows")
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
        self._tokenization_context = TokenizationContext.from_tokenizer(tokenizer)
        self._snapshot_renderer = SnapshotRenderer(
            oracle,
            max_card_refs=MAX_CARD_REFS,
            record_char_anchors=False,
        )
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
                include_records=cfg.sequence_mode != "full",
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

    def _decoder_candidate(self, record: dict[str, Any]) -> _DecoderCandidate | None:
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
        return _DecoderCandidate(
            snapshot=cast(GameStateSnapshot, snapshot),
            target=target,
            value_target=_value_target(record, self.cfg),
        )

    def _prepare_decoder(self, record: dict[str, Any]) -> _PreparedDecoderExample | None:
        prepared = self._prepare_decoder_many([record])
        return prepared[0] if prepared else None

    def _prepare_decoder_many(
        self, records: Sequence[dict[str, Any]]
    ) -> list[_PreparedDecoderExample]:
        candidates: list[_DecoderCandidate] = []
        rendered_rows = []
        specs: list[DecisionSpec] = []
        for record in records:
            candidate = self._decoder_candidate(record)
            if candidate is None:
                continue
            try:
                rendered = self._snapshot_renderer.render(candidate.snapshot)
                spec = self._spec_renderer.render(candidate.snapshot, card_refs=rendered.card_refs)
            except RenderError, RuntimeError, KeyError, TypeError, ValueError, NotImplementedError:
                continue
            candidates.append(candidate)
            rendered_rows.append(rendered)
            specs.append(spec)

        if not candidates:
            return []
        try:
            encoded_rows = tokenize_snapshots(
                rendered_rows,
                self.tokenizer,
                context=self._tokenization_context,
            )
        except RenderError, RuntimeError, KeyError, TypeError, ValueError, NotImplementedError:
            return []

        return [
            _PreparedDecoderExample(
                encoded=encoded,
                spec=spec,
                target=candidate.target,
                value_target=candidate.value_target,
            )
            for candidate, encoded, spec in zip(candidates, encoded_rows, specs, strict=True)
        ]

    def _batch_from_indices(self, indices: Sequence[int]) -> ForgeDecoderBatch:
        prepared: list[_PreparedDecoderExample] = []
        cursor = 0
        while len(prepared) < len(indices) and cursor < len(indices) * 4:
            remaining = len(indices) - len(prepared)
            record_indices = [
                int(indices[(cursor + off) % len(indices)]) for off in range(remaining)
            ]
            cursor += remaining
            prepared.extend(
                self._prepare_decoder_many([self._record_at(i) for i in record_indices])
            )
        if not prepared:
            raise ValueError("no renderable Forge decoder examples in selected batch")
        return self._batch_from_prepared(prepared[: len(indices)])

    def _batch_from_prepared(self, prepared: list[_PreparedDecoderExample]) -> ForgeDecoderBatch:
        if not prepared:
            raise ValueError("_batch_from_prepared got empty list")

        encoded = collate(
            [p.encoded for p in prepared],
            [p.spec for p in prepared],
            pad_id=self.cfg.pad_token_id,
        )

        batch_size = len(prepared)
        target_lens = torch.as_tensor(
            [len(p.target.output_token_ids) for p in prepared], dtype=torch.long
        )
        L = int(target_lens.max().item())
        T_enc = int(encoded.token_ids.shape[1])

        out_tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(p.target.output_token_ids, dtype=torch.long) for p in prepared],
            batch_first=True,
            padding_value=0,
        )
        output_pointer_subjects = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(p.target.output_pointer_subjects, dtype=torch.long) for p in prepared],
            batch_first=True,
            padding_value=-1,
        )
        out_is_pointer = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(p.target.output_is_pointer, dtype=torch.bool) for p in prepared],
            batch_first=True,
            padding_value=False,
        )
        decision_type_per_row = torch.as_tensor(
            [int(p.target.decision_type) for p in prepared], dtype=torch.long
        )
        values = torch.as_tensor([float(p.value_target) for p in prepared], dtype=torch.float32)

        steps = torch.arange(L, dtype=torch.long)
        out_pad_mask = steps.unsqueeze(0) < target_lens.unsqueeze(1)

        anchor_positions = encoded.pointer_anchor_positions.to(torch.long)
        anchor_kinds = encoded.pointer_anchor_kinds.to(torch.long)
        anchor_subjects = encoded.pointer_anchor_subjects.to(torch.long)
        expected_kinds = _expected_pointer_kind_matrix(decision_type_per_row, L)
        valid_anchor_match = (
            out_is_pointer.unsqueeze(2)
            & (anchor_positions[:, None, :] >= 0)
            & (anchor_positions[:, None, :] < T_enc)
            & (anchor_kinds[:, None, :] == expected_kinds[:, :, None])
            & (anchor_subjects[:, None, :] == output_pointer_subjects[:, :, None])
        )
        candidate_pointer_pos = torch.where(
            valid_anchor_match,
            anchor_positions[:, None, :],
            torch.full_like(anchor_positions[:, None, :], T_enc),
        )
        min_pointer_pos = candidate_pointer_pos.min(dim=2).values
        has_pointer_target = min_pointer_pos < T_enc
        out_pointer_pos = torch.where(
            out_is_pointer & has_pointer_target,
            min_pointer_pos,
            torch.full_like(min_pointer_pos, -1),
        )
        out_pad_mask = out_pad_mask & (~out_is_pointer | has_pointer_target)

        anchor_counts = _anchor_counts_from_encoded(encoded.pointer_anchor_kinds)
        max_values = torch.as_tensor([int(p.spec.max_value) for p in prepared], dtype=torch.long)
        per_row_max_prefix = (target_lens - 1).clamp_min(0)
        prefix_lens = torch.minimum(steps[:, None], per_row_max_prefix[None, :])
        flat_prefix_tokens = out_tokens.unsqueeze(0).expand(L, -1, -1).reshape(L * batch_size, L)
        flat_prefix_subjects = (
            output_pointer_subjects.unsqueeze(0).expand(L, -1, -1).reshape(L * batch_size, L)
        )
        flat_prefix_lens = prefix_lens.reshape(L * batch_size)
        flat_decision_type = decision_type_per_row.unsqueeze(0).expand(L, -1).reshape(-1)
        flat_anchor_counts = (
            anchor_counts.unsqueeze(0).expand(L, -1, -1).reshape(L * batch_size, -1)
        )
        flat_max_values = max_values.unsqueeze(0).expand(L, -1).reshape(-1)
        flat_legal_edges = (
            None
            if encoded.legal_edge_bitmap is None
            else encoded.legal_edge_bitmap.unsqueeze(0)
            .expand(L, -1, -1, -1)
            .reshape(L * batch_size, *encoded.legal_edge_bitmap.shape[1:])
        )
        flat_vocab_mask, flat_pointer_subject_mask = batch_next_mask_torch(
            flat_decision_type,
            flat_anchor_counts,
            flat_max_values,
            flat_legal_edges,
            flat_prefix_tokens,
            flat_prefix_subjects,
            flat_prefix_lens,
        )
        vocab_mask = flat_vocab_mask.view(L, batch_size, GRAMMAR_VOCAB_SIZE).permute(1, 0, 2)
        pointer_subject_mask = flat_pointer_subject_mask.view(L, batch_size, -1).permute(1, 0, 2)
        pointer_mask = torch.zeros((batch_size, L, T_enc), dtype=torch.bool)
        if pointer_subject_mask.shape[2] > 0:
            safe_subjects = anchor_subjects.clamp(0, pointer_subject_mask.shape[2] - 1)
            subject_allowed = pointer_subject_mask.gather(
                2, safe_subjects[:, None, :].expand(-1, L, -1)
            )
            valid_pointer_anchor = (
                (anchor_positions[:, None, :] >= 0)
                & (anchor_positions[:, None, :] < T_enc)
                & (anchor_kinds[:, None, :] == expected_kinds[:, :, None])
                & subject_allowed
            )
            valid_rows, valid_steps, valid_anchor_cols = valid_pointer_anchor.nonzero(as_tuple=True)
            if int(valid_rows.numel()) > 0:
                pointer_mask[
                    valid_rows,
                    valid_steps,
                    anchor_positions[valid_rows, valid_anchor_cols],
                ] = True

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
            prepared_game = self._prepare_decoder_many(game_records)
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
        cell_indices_by_timestep: list[list[int]] = [[] for _ in range(l_max)]
        game_indices_by_timestep: list[list[int]] = [[] for _ in range(l_max)]
        loss_cell_indices_by_timestep: list[list[int]] = [[] for _ in range(l_max)]
        loss_active_pos_by_timestep: list[list[int]] = [[] for _ in range(l_max)]
        for b, game in enumerate(prepared_per_game):
            k = min(loss_positions_per_game, len(game))
            chosen = set(rng.choice(len(game), size=k, replace=False).tolist())
            for pos, item in enumerate(game):
                flat_idx = len(flat)
                flat.append(item)
                cell_game_idx.append(b)
                cell_pos_idx.append(pos)
                contributes_loss = pos in chosen
                loss_mask_list.append(contributes_loss)
                active_pos = len(cell_indices_by_timestep[pos])
                cell_indices_by_timestep[pos].append(flat_idx)
                game_indices_by_timestep[pos].append(b)
                if contributes_loss:
                    loss_cell_indices_by_timestep[pos].append(flat_idx)
                    loss_active_pos_by_timestep[pos].append(active_pos)

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
            cell_indices_by_timestep=tuple(
                torch.tensor(xs, dtype=torch.long) for xs in cell_indices_by_timestep
            ),
            cell_indices_by_timestep_host=tuple(tuple(xs) for xs in cell_indices_by_timestep),
            game_indices_by_timestep=tuple(
                torch.tensor(xs, dtype=torch.long) for xs in game_indices_by_timestep
            ),
            loss_cell_indices_by_timestep=tuple(
                torch.tensor(xs, dtype=torch.long) for xs in loss_cell_indices_by_timestep
            ),
            loss_active_pos_by_timestep=tuple(
                torch.tensor(xs, dtype=torch.long) for xs in loss_active_pos_by_timestep
            ),
        )


def _anchor_counts_from_encoded(pointer_anchor_kinds: Tensor) -> Tensor:
    valid = pointer_anchor_kinds >= 0
    safe_kinds = pointer_anchor_kinds.clamp(min=0).to(torch.long)
    counts = torch.zeros(
        (int(pointer_anchor_kinds.shape[0]), len(AnchorKind)),
        dtype=torch.long,
        device=pointer_anchor_kinds.device,
    )
    return counts.scatter_add(1, safe_kinds, valid.to(torch.long))


def _expected_pointer_kind_matrix(decision_type: Tensor, max_len: int) -> Tensor:
    steps = torch.arange(max_len, dtype=torch.long, device=decision_type.device)
    body_off = steps - 1
    out = torch.full(
        (int(decision_type.shape[0]), max_len),
        -1,
        dtype=torch.long,
        device=decision_type.device,
    )
    out[decision_type == int(DecisionType.PRIORITY), :] = int(AnchorKind.LEGAL_ACTION)
    out[decision_type == int(DecisionType.CHOOSE_TARGETS), :] = int(AnchorKind.LEGAL_TARGET)

    attackers = decision_type == int(DecisionType.DECLARE_ATTACKERS)
    if bool(attackers.any()):
        attacker_kind = torch.where(
            body_off % 4 == 1,
            int(AnchorKind.LEGAL_ATTACKER),
            int(AnchorKind.DEFENDER),
        )
        out[attackers] = attacker_kind

    blockers = decision_type == int(DecisionType.DECLARE_BLOCKERS)
    if bool(blockers.any()):
        blocker_kind = torch.where(
            body_off % 4 == 1,
            int(AnchorKind.LEGAL_BLOCKER),
            int(AnchorKind.LEGAL_ATTACKER),
        )
        out[blockers] = blocker_kind
    return out


def _index_text_encoded_batch(
    batch: TextEncodedBatch, rows: Tensor, rows_host: Sequence[int]
) -> TextEncodedBatch:
    seq_lengths = batch.seq_lengths.index_select(0, rows)
    source_lengths = batch.seq_lengths_host
    if source_lengths is None:
        seq_lengths_host = tuple(int(x) for x in seq_lengths.detach().cpu().tolist())
    else:
        seq_lengths_host = tuple(int(source_lengths[i]) for i in rows_host)
    max_tokens = max(seq_lengths_host, default=0)
    legal_edge_bitmap = (
        batch.legal_edge_bitmap.index_select(0, rows)
        if batch.legal_edge_bitmap is not None
        else None
    )
    return TextEncodedBatch(
        token_ids=batch.token_ids.index_select(0, rows)[:, :max_tokens].contiguous(),
        attention_mask=batch.attention_mask.index_select(0, rows)[:, :max_tokens].contiguous(),
        card_ref_positions=batch.card_ref_positions.index_select(0, rows).contiguous(),
        seq_lengths=seq_lengths,
        spec_tokens=batch.spec_tokens.index_select(0, rows).contiguous(),
        spec_lens=batch.spec_lens.index_select(0, rows).contiguous(),
        decision_type=batch.decision_type.index_select(0, rows).contiguous(),
        pointer_anchor_positions=batch.pointer_anchor_positions.index_select(0, rows).contiguous(),
        pointer_anchor_kinds=batch.pointer_anchor_kinds.index_select(0, rows).contiguous(),
        pointer_anchor_subjects=batch.pointer_anchor_subjects.index_select(0, rows).contiguous(),
        pointer_anchor_handles=batch.pointer_anchor_handles.index_select(0, rows).contiguous(),
        legal_edge_bitmap=legal_edge_bitmap,
        total_tokens=sum(seq_lengths_host),
        seq_lengths_host=seq_lengths_host,
    )


def _index_text_encoded_batch_packed(
    batch: TextEncodedBatch, rows: Tensor, rows_host: Sequence[int]
) -> PackedTextBatch:
    seq_lengths = batch.seq_lengths.index_select(0, rows).to(torch.int32)
    source_lengths = batch.seq_lengths_host
    if source_lengths is None:
        seq_lengths_host = tuple(int(x) for x in seq_lengths.detach().cpu().tolist())
    else:
        seq_lengths_host = tuple(int(source_lengths[i]) for i in rows_host)
    total_tokens = sum(seq_lengths_host)
    max_seqlen = max(seq_lengths_host, default=0)
    cu, state_positions, seq_id, pos_in_seq = packed_sequence_layout(
        seq_lengths,
        total_tokens=total_tokens,
    )
    row_token_ids = batch.token_ids.index_select(0, rows)
    token_ids = row_token_ids[seq_id.to(torch.long), pos_in_seq.to(torch.long)].to(torch.int32)
    legal_edge_bitmap = (
        batch.legal_edge_bitmap.index_select(0, rows)
        if batch.legal_edge_bitmap is not None
        else None
    )
    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu,
        seq_lengths=seq_lengths,
        state_positions=state_positions.clone(),
        card_ref_positions=batch.card_ref_positions.index_select(0, rows).to(torch.int32),
        spec_lens=batch.spec_lens.index_select(0, rows).contiguous(),
        decision_type=batch.decision_type.index_select(0, rows).contiguous(),
        pointer_anchor_positions=batch.pointer_anchor_positions.index_select(0, rows).to(
            torch.int32
        ),
        pointer_anchor_kinds=batch.pointer_anchor_kinds.index_select(0, rows).contiguous(),
        pointer_anchor_subjects=batch.pointer_anchor_subjects.index_select(0, rows).contiguous(),
        pointer_anchor_handles=batch.pointer_anchor_handles.index_select(0, rows).contiguous(),
        legal_edge_bitmap=legal_edge_bitmap,
        total_tokens=total_tokens,
        seq_lengths_host=seq_lengths_host,
        max_seqlen=max_seqlen,
    )


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

    def _sequenced_forward(self, batch: ForgeSequencedBatch) -> _SequencedForwardOutput:
        """Run the RL-style recurrent forward over a sequenced pretrain batch.

        This mirrors rollout/replay semantics: each timestep encodes the
        current snapshot with ``hist_proj(h_in[-1])`` injected into the encoder,
        scores decoder/value heads from that history-conditioned encoding, and
        only then advances ``h/c`` for the next timestep. Losses are evaluated
        only on the sampled cells, but every active cell updates recurrence.
        """

        text_policy = self.policy.text_policy
        if text_policy.grammar_decoder is None:
            raise RuntimeError("decoder pipeline requires TextPolicy with a grammar_decoder")
        device = next(text_policy.parameters()).device
        dtype = next(self.policy.lstm.parameters()).dtype
        h = torch.zeros(
            self.policy.lstm_layers,
            batch.n_games,
            self.policy.lstm_hidden,
            dtype=dtype,
            device=device,
        )
        c = torch.zeros_like(h)
        policy_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        policy_count = torch.zeros((), dtype=torch.float32, device=device)
        value_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        value_count = torch.zeros((), dtype=torch.float32, device=device)
        value_predictions: list[Tensor] = []
        value_targets: list[Tensor] = []

        from magic_ai.text_encoder.batch import scatter_packed_to_padded

        for t in range(batch.l_max):
            active_cells = batch.cell_indices_by_timestep[t]
            if int(active_cells.numel()) == 0:
                continue
            active_games = batch.game_indices_by_timestep[t]
            timestep_encoded = _index_text_encoded_batch_packed(
                batch.decoder.encoded, active_cells, batch.cell_indices_by_timestep_host[t]
            )
            h_in = h.index_select(1, active_games)
            c_in = c.index_select(1, active_games)
            encoded_snaps, h_out, c_out = self.policy.encode_with_history(
                timestep_encoded,
                h_in=h_in,
                c_in=c_in,
            )
            h_out = h_out.to(dtype=h.dtype)
            c_out = c_out.to(dtype=c.dtype)
            h = h.index_copy(1, active_games, h_out)
            c = c.index_copy(1, active_games, c_out)

            loss_active_pos = batch.loss_active_pos_by_timestep[t]
            if int(loss_active_pos.numel()) == 0:
                continue
            loss_cells = batch.loss_cell_indices_by_timestep[t]
            encoded_hidden, attn_mask = scatter_packed_to_padded(
                encoded_snaps.encoded, encoded_snaps.packed
            )
            enc_width = int(encoded_hidden.shape[1])
            encoded_loss = encoded_hidden.index_select(0, loss_active_pos)
            attn_loss = attn_mask.index_select(0, loss_active_pos)
            target_tokens = batch.decoder.output_token_ids.index_select(0, loss_cells).to(
                device, non_blocking=True
            )
            vocab_logits, pointer_logits = text_policy.grammar_decoder.forward_teacher_forced(
                target_tokens,
                encoded_loss,
                attn_loss,
            )
            out_pad_mask = batch.decoder.output_pad_mask.index_select(0, loss_cells).to(
                device, non_blocking=True
            )
            step_count = out_pad_mask.sum().to(dtype=torch.float32)
            policy_l = decoder_cross_entropy_loss(
                vocab_logits,
                pointer_logits,
                target_tokens,
                batch.decoder.output_pointer_pos.index_select(0, loss_cells).to(
                    device, non_blocking=True
                ),
                batch.decoder.output_is_pointer.index_select(0, loss_cells).to(
                    device, non_blocking=True
                ),
                batch.decoder.vocab_mask.index_select(0, loss_cells).to(device, non_blocking=True),
                batch.decoder.pointer_mask.index_select(0, loss_cells)[:, :, :enc_width].to(
                    device, non_blocking=True
                ),
                out_pad_mask,
            )
            policy_loss_sum = policy_loss_sum + policy_l.float() * step_count
            policy_count = policy_count + step_count

            state_vec_loss = encoded_snaps.state_vector.index_select(0, loss_active_pos)
            values = text_policy.run_heads(encoded_snaps, state_vec=state_vec_loss)
            targets = batch.decoder.value_targets.index_select(0, loss_cells).to(
                device, non_blocking=True
            )
            value_loss_sum = value_loss_sum + (values.float() - targets.float()).pow(2).sum()
            value_count = value_count + float(targets.numel())
            value_predictions.append(values)
            value_targets.append(targets)

        policy_loss = policy_loss_sum / policy_count.clamp(min=1.0)
        value_loss_out = value_loss_sum / value_count.clamp(min=1.0)
        loss = (
            self.cfg.policy_loss_weight * policy_loss + self.cfg.value_loss_weight * value_loss_out
        )
        if value_predictions:
            pred = torch.cat(value_predictions, dim=0)
            targ = torch.cat(value_targets, dim=0)
        else:
            pred = torch.empty((0,), dtype=torch.float32, device=device)
            targ = torch.empty((0,), dtype=torch.float32, device=device)
        return _SequencedForwardOutput(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss_out,
            value_predictions=pred,
            value_targets=targ,
            n_loss_cells=value_count,
        )

    def _sequenced_step(
        self,
        batch: ForgeSequencedBatch,
        *,
        compute_stats: bool,
        accum_index: int = 0,
        accum_total: int = 1,
    ) -> dict[str, float]:
        """Sequenced pretrain: replay each game through the LSTM in temporal order."""
        self.policy.train()
        if accum_index == 0:
            self.optimizer.zero_grad(set_to_none=True)
        out = self._sequenced_forward(batch)
        loss = out.loss
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
            v_pred_loss = out.value_predictions.float()
            v_targ_loss = out.value_targets.float()
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
            "policy_loss": float(out.policy_loss.detach()),
            "value_loss": float(out.value_loss.detach()),
            "v_pred_mean": v_pred_mean,
            "v_pred_std": v_pred_std,
            "v_targ_mean": v_targ_mean,
            "v_targ_std": v_targ_std,
            "v_corr": v_corr,
            "value_sign_accuracy": sign_acc,
            "v_head_grad_norm": v_grad_norm,
            "lstm_grad_norm": lstm_grad_norm,
            "grad_norm": float(grad_norm.detach()),
            "n_loss_cells": int(out.n_loss_cells.detach()),
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
            if self.policy.text_policy.grammar_decoder is None:
                continue
            out = self._sequenced_forward(batch)
            v_pred = out.value_predictions.float()
            v_targ = out.value_targets.float()
            non_draw = v_targ.abs() > 1e-6
            sign_acc = (
                float((torch.sign(v_pred[non_draw]) == torch.sign(v_targ[non_draw])).float().mean())
                if non_draw.any()
                else 0.0
            )
            stats = {
                "eval_policy_loss": float(out.policy_loss.detach()),
                "eval_value_loss": float(out.value_loss.detach()),
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
        token_ids=batch.token_ids.to(device, non_blocking=True),
        attention_mask=batch.attention_mask.to(device, non_blocking=True),
        card_ref_positions=batch.card_ref_positions.to(device, non_blocking=True),
        seq_lengths=batch.seq_lengths.to(device, non_blocking=True),
        total_tokens=batch.total_tokens,
        seq_lengths_host=batch.seq_lengths_host,
        spec_tokens=batch.spec_tokens.to(device, non_blocking=True),
        spec_lens=batch.spec_lens.to(device, non_blocking=True),
        decision_type=batch.decision_type.to(device, non_blocking=True),
        pointer_anchor_positions=batch.pointer_anchor_positions.to(device, non_blocking=True),
        pointer_anchor_kinds=batch.pointer_anchor_kinds.to(device, non_blocking=True),
        pointer_anchor_subjects=batch.pointer_anchor_subjects.to(device, non_blocking=True),
        pointer_anchor_handles=batch.pointer_anchor_handles.to(device, non_blocking=True),
        legal_edge_bitmap=(
            batch.legal_edge_bitmap.to(device, non_blocking=True)
            if batch.legal_edge_bitmap is not None
            else None
        ),
    )


def _truncate_encoded_batch(batch: TextEncodedBatch, *, max_tokens: int) -> TextEncodedBatch:
    seq_lengths = batch.seq_lengths.clamp(max=max_tokens)
    positions = torch.arange(max_tokens, device=batch.attention_mask.device)
    attention_mask = (positions.unsqueeze(0) < seq_lengths.unsqueeze(1)).to(
        dtype=batch.attention_mask.dtype
    )
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
        output_token_ids=batch.output_token_ids.to(device, non_blocking=True),
        output_pointer_pos=batch.output_pointer_pos.to(device, non_blocking=True),
        output_is_pointer=batch.output_is_pointer.to(device, non_blocking=True),
        output_pad_mask=batch.output_pad_mask.to(device, non_blocking=True),
        vocab_mask=batch.vocab_mask.to(device, non_blocking=True),
        pointer_mask=batch.pointer_mask.to(device, non_blocking=True),
        decision_type_per_row=batch.decision_type_per_row.to(device, non_blocking=True),
        value_targets=batch.value_targets.to(device, non_blocking=True),
    )


def _sequenced_batch_to_device(
    batch: ForgeSequencedBatch, device: torch.device
) -> ForgeSequencedBatch:
    return ForgeSequencedBatch(
        decoder=_batch_to_device(batch.decoder, device),
        cell_game_idx=batch.cell_game_idx.to(device, non_blocking=True),
        cell_pos_idx=batch.cell_pos_idx.to(device, non_blocking=True),
        loss_mask=batch.loss_mask.to(device, non_blocking=True),
        game_lengths=batch.game_lengths.to(device, non_blocking=True),
        n_games=batch.n_games,
        l_max=batch.l_max,
        cell_indices_by_timestep=tuple(
            x.to(device, non_blocking=True) for x in batch.cell_indices_by_timestep
        ),
        cell_indices_by_timestep_host=batch.cell_indices_by_timestep_host,
        game_indices_by_timestep=tuple(
            x.to(device, non_blocking=True) for x in batch.game_indices_by_timestep
        ),
        loss_cell_indices_by_timestep=tuple(
            x.to(device, non_blocking=True) for x in batch.loss_cell_indices_by_timestep
        ),
        loss_active_pos_by_timestep=tuple(
            x.to(device, non_blocking=True) for x in batch.loss_active_pos_by_timestep
        ),
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
