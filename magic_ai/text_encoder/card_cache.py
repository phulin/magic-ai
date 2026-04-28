"""Pre-tokenized card-body cache for the text encoder hot path.

PR 13-A from ``docs/text_encoder_plan.md``. At startup we render the
``<card> Name <sep> Type <sep> P/T <sep> oracle </card>`` fragment for every
card the Go engine knows about, BPE-tokenize it once, and pack the results
into a flat ``torch.int32`` buffer keyed by the engine's 1-indexed card-row IDs.
The hot-path assembler (PR 13-C) memcpys slices out of this cache; per-step
BPE is avoided entirely.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import orjson
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.render import OracleEntry, OracleFace, render_card_body

DEFAULT_ORACLE_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "oracle-cards.json"

logger = logging.getLogger(__name__)

UNKNOWN_NAME = "<unknown>"


class MissingOracleTextError(RuntimeError):
    """Raised when one or more registered cards lack oracle text."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        preview = ", ".join(self.missing[:20])
        suffix = "" if len(self.missing) <= 20 else f" (+{len(self.missing) - 20} more)"
        super().__init__(
            f"{len(self.missing)} registered card(s) have no oracle entry: {preview}{suffix}"
        )


@dataclass(frozen=True)
class CardTokenCache:
    """Flat int32 token buffer for every registered card body.

    Row K's body lives at ``token_buffer[offsets[K]:offsets[K+1]]``. Row 0 is
    the unknown sentinel — empty body — to match the Go engine's 1-indexed
    card-row IDs.
    """

    token_buffer: Tensor  # shape [total_tokens] dtype torch.int32
    offsets: Tensor  # shape [num_card_rows + 1] dtype torch.int64
    row_to_name: list[str]  # length num_card_rows + 1; index 0 = UNKNOWN_NAME
    engine_card_set_hash: str

    @property
    def num_rows(self) -> int:
        return len(self.row_to_name)

    def body_tokens(self, row: int) -> Tensor:
        return self.token_buffer[int(self.offsets[row]) : int(self.offsets[row + 1])]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def compute_card_set_hash(registered_names: Sequence[str]) -> str:
    """Hash of the sorted registered-card-name set (16-hex-char prefix)."""

    payload = json.dumps(sorted(registered_names)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _scryfall_record_to_entry(record: Mapping[str, Any]) -> OracleEntry:
    """Convert a Scryfall bulk-data oracle-cards record to an OracleEntry."""

    name = str(record.get("name") or "")
    pt: str | None = None
    power = record.get("power")
    toughness = record.get("toughness")
    if power is not None and toughness is not None:
        pt = f"{power}/{toughness}"
    entry: OracleEntry = {
        "name": name,
        "type_line": str(record.get("type_line") or ""),
        "mana_cost": str(record.get("mana_cost") or ""),
        "oracle_text": str(record.get("oracle_text") or ""),
        "power_toughness": pt,
        "colors": list(record.get("colors") or []),
    }
    layout = record.get("layout")
    if layout:
        entry["layout"] = str(layout)
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
                    "name": str(face.get("name") or ""),
                    "type_line": str(face.get("type_line") or ""),
                    "mana_cost": str(face.get("mana_cost") or ""),
                    "oracle_text": str(face.get("oracle_text") or ""),
                    "power_toughness": face_pt,
                }
            )
        entry["card_faces"] = faces
    return entry


def load_oracle_db(
    path: Path | str = DEFAULT_ORACLE_DB_PATH,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, OracleEntry]:
    """Load Scryfall ``oracle-cards.json`` bulk dump keyed by canonical name.

    Uses orjson because the bulk dump is ~170 MB and ``json.loads`` is the
    dominant cost on a cold cache build. When ``names`` is provided the
    return value is restricted to that set; the full dump still has to be
    parsed once but only the matching subset is converted to ``OracleEntry``.
    """

    p = Path(path)
    with p.open("rb") as fh:
        records = orjson.loads(fh.read())
    if not isinstance(records, list):
        raise RuntimeError(f"{p} did not contain a top-level JSON array")
    wanted: set[str] | None = set(names) if names is not None else None
    out: dict[str, OracleEntry] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        if wanted is not None and name not in wanted:
            continue
        out[name] = _scryfall_record_to_entry(record)
    return out


def build_card_cache(
    registered_names: Sequence[str],
    oracle: Mapping[str, OracleEntry] | None = None,
    tokenizer: PreTrainedTokenizerFast | None = None,
    *,
    oracle_db_path: Path | str | None = DEFAULT_ORACLE_DB_PATH,
    missing_policy: Literal["raise", "warn", "skip"] = "raise",
) -> CardTokenCache:
    """Render + tokenize every registered card body and pack into a flat cache.

    Engine row IDs are 1-indexed (row 0 = unknown sentinel). Names are kept in
    the caller-supplied order; the dedupe / canonical sort happens only inside
    :func:`compute_card_set_hash`.

    Oracle lookup precedence: callers may pass a pre-built ``oracle`` mapping
    (test fixtures, custom-card overrides). Names not found there fall back
    to ``oracle_db_path`` (default: the Scryfall bulk dump
    ``data/oracle-cards.json``), which is loaded lazily on first miss and
    only kept in memory for names actually requested.

    Coverage: by default raises :class:`MissingOracleTextError` listing every
    registered name still without an oracle entry. ``warn`` logs the same
    message; ``skip`` writes empty bodies for missing names.
    """

    if tokenizer is None:
        raise TypeError("tokenizer is required")

    names = list(registered_names)
    base_oracle: Mapping[str, OracleEntry] = oracle if oracle is not None else {}
    missing_in_base = [n for n in names if n not in base_oracle]
    db_oracle: dict[str, OracleEntry] = {}
    if missing_in_base and oracle_db_path is not None and Path(oracle_db_path).exists():
        db_oracle = load_oracle_db(oracle_db_path, names=missing_in_base)
    merged: dict[str, OracleEntry] = {**db_oracle, **dict(base_oracle)}
    missing = [n for n in names if n not in merged]
    if missing:
        if missing_policy == "raise":
            raise MissingOracleTextError(missing)
        msg = (
            f"{len(missing)} registered card(s) have no oracle entry; "
            f"first 20: {', '.join(missing[:20])}"
        )
        if missing_policy == "warn":
            warnings.warn(msg, stacklevel=2)
        else:
            logger.warning("card_cache: %s", msg)

    # Row 0 = unknown sentinel: empty body.
    row_to_name: list[str] = [UNKNOWN_NAME]
    pieces: list[Tensor] = []
    offsets: list[int] = [0, 0]  # offsets[0] = offsets[1] = 0
    cursor = 0
    for name in names:
        entry = merged.get(name)
        if entry is None and missing_policy != "skip":
            # Will already have been raised above when policy == "raise".
            # For "warn" we still emit the body using just the name.
            body_text = render_card_body(name, None)
        elif entry is None:
            body_text = ""  # "skip" -> empty body
        else:
            body_text = render_card_body(name, entry)
        if body_text:
            ids = tokenizer.encode(body_text, add_special_tokens=False)
            arr = torch.tensor(ids, dtype=torch.int32)
        else:
            arr = torch.empty((0,), dtype=torch.int32)
        pieces.append(arr)
        cursor += int(arr.shape[0])
        offsets.append(cursor)
        row_to_name.append(name)

    token_buffer = torch.cat(pieces) if pieces else torch.empty((0,), dtype=torch.int32)
    offsets_arr = torch.tensor(offsets, dtype=torch.int64)
    return CardTokenCache(
        token_buffer=token_buffer,
        offsets=offsets_arr,
        row_to_name=row_to_name,
        engine_card_set_hash=compute_card_set_hash(names),
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_card_cache(cache: CardTokenCache, path: Path | str) -> None:
    """Write the cache to ``path`` as a PyTorch checkpoint."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "token_buffer": cache.token_buffer.detach().cpu().to(dtype=torch.int32),
            "offsets": cache.offsets.detach().cpu().to(dtype=torch.int64),
            "row_to_name": cache.row_to_name,
            "engine_card_set_hash": cache.engine_card_set_hash,
        },
        p,
    )


def load_card_cache(path: Path | str) -> CardTokenCache:
    p = Path(path)
    payload = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"card cache {p} did not contain a dict payload")
    token_buffer = payload.get("token_buffer")
    offsets = payload.get("offsets")
    row_to_name = payload.get("row_to_name")
    engine_card_set_hash = payload.get("engine_card_set_hash")
    if not isinstance(token_buffer, Tensor) or not isinstance(offsets, Tensor):
        raise RuntimeError(f"card cache {p} is missing tensor buffers")
    if not isinstance(row_to_name, list) or not isinstance(engine_card_set_hash, str):
        raise RuntimeError(f"card cache {p} is missing metadata")
    return CardTokenCache(
        token_buffer=token_buffer.to(dtype=torch.int32).contiguous(),
        offsets=offsets.to(dtype=torch.int64).contiguous(),
        row_to_name=[str(name) for name in row_to_name],
        engine_card_set_hash=engine_card_set_hash,
    )


# ---------------------------------------------------------------------------
# FFI: pull the registered-card-name list from libmage.
# ---------------------------------------------------------------------------


def _load_libmage() -> ctypes.CDLL:
    """Open the mage shared library via ctypes, mirroring native_encoder.py."""

    try:
        mage = importlib.import_module("mage")
    except ImportError as exc:  # pragma: no cover - depends on local build
        raise RuntimeError(
            "libmage is not importable. Build mage-go (`make pylib` in ../mage-go) "
            "or run with --names-from oracle to skip the engine."
        ) from exc
    path = getattr(mage, "_lib_path_used", None)
    if not path:
        try:
            cast(Any, mage).load()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"mage.load() failed: {exc}") from exc
        path = getattr(mage, "_lib_path_used", None)
    if not path:
        raise RuntimeError("Could not determine libmage shared-library path.")
    return ctypes.CDLL(path)


def fetch_registered_card_names_from_engine() -> list[str]:
    """Call ``MageRegisteredCards()`` and parse the JSON name list.

    Raises a clear ``RuntimeError`` if libmage isn't available.
    """

    lib = _load_libmage()
    try:
        lib.MageRegisteredCards.argtypes = []
        lib.MageRegisteredCards.restype = ctypes.c_void_p
        lib.MageFreeString.argtypes = [ctypes.c_void_p]
        lib.MageFreeString.restype = None
    except AttributeError as exc:
        raise RuntimeError(f"libmage missing MageRegisteredCards / MageFreeString: {exc}") from exc

    raw = lib.MageRegisteredCards()
    if not raw:
        raise RuntimeError("MageRegisteredCards() returned NULL")
    try:
        payload = ctypes.string_at(raw).decode("utf-8")
    finally:
        lib.MageFreeString(raw)
    names = json.loads(payload)
    if not isinstance(names, list):
        raise RuntimeError(f"MageRegisteredCards() returned non-list JSON: {type(names).__name__}")
    return sorted(str(n) for n in names)


# ---------------------------------------------------------------------------
# Convenience: stat helpers used by the build script.
# ---------------------------------------------------------------------------


def cache_length_stats(cache: CardTokenCache) -> dict[str, float | int]:
    """Mean / p50 / p90 / max body-token count, excluding row 0."""

    if cache.num_rows <= 1:
        return {"count": 0, "mean": 0.0, "p50": 0, "p90": 0, "max": 0}
    lens = cache.offsets[2:] - cache.offsets[1:-1]  # one entry per real card row
    lens_f = lens.to(dtype=torch.float32)
    return {
        "count": int(lens.numel()),
        "mean": float(lens_f.mean()),
        "p50": int(torch.quantile(lens_f, 0.5)),
        "p90": int(torch.quantile(lens_f, 0.9)),
        "max": int(lens.max()),
    }
