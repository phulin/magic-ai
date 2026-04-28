"""Pre-tokenized card-body cache for the text encoder hot path.

PR 13-A from ``docs/text_encoder_plan.md``. At startup we render the
``<card> Name <sep> Type <sep> P/T <sep> oracle </card>`` fragment for every
card the Go engine knows about, BPE-tokenize it once, and pack the results
into a flat ``int32`` buffer keyed by the engine's 1-indexed card-row IDs.
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

import numpy as np
from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.render import OracleEntry, render_card_body

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

    token_buffer: np.ndarray  # shape [total_tokens] dtype int32
    offsets: np.ndarray  # shape [num_card_rows + 1] dtype int64
    row_to_name: list[str]  # length num_card_rows + 1; index 0 = UNKNOWN_NAME
    engine_card_set_hash: str

    @property
    def num_rows(self) -> int:
        return len(self.row_to_name)

    def body_tokens(self, row: int) -> np.ndarray:
        return self.token_buffer[int(self.offsets[row]) : int(self.offsets[row + 1])]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def compute_card_set_hash(registered_names: Sequence[str]) -> str:
    """Hash of the sorted registered-card-name set (16-hex-char prefix)."""

    payload = json.dumps(sorted(registered_names)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_card_cache(
    registered_names: Sequence[str],
    oracle: Mapping[str, OracleEntry],
    tokenizer: PreTrainedTokenizerFast,
    *,
    missing_policy: Literal["raise", "warn", "skip"] = "raise",
) -> CardTokenCache:
    """Render + tokenize every registered card body and pack into a flat cache.

    Engine row IDs are 1-indexed (row 0 = unknown sentinel). Names are kept in
    the caller-supplied order; the dedupe / canonical sort happens only inside
    :func:`compute_card_set_hash`.

    Coverage: by default raises :class:`MissingOracleTextError` listing every
    registered name without an oracle entry. ``warn`` logs the same message;
    ``skip`` writes empty bodies for missing names.
    """

    names = list(registered_names)
    missing = [n for n in names if n not in oracle]
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
    pieces: list[np.ndarray] = []
    offsets: list[int] = [0, 0]  # offsets[0] = offsets[1] = 0
    cursor = 0
    for name in names:
        entry = oracle.get(name)
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
            arr = np.asarray(ids, dtype=np.int32)
        else:
            arr = np.empty((0,), dtype=np.int32)
        pieces.append(arr)
        cursor += int(arr.shape[0])
        offsets.append(cursor)
        row_to_name.append(name)

    token_buffer = (np.concatenate(pieces) if pieces else np.empty((0,), dtype=np.int32)).astype(
        np.int32, copy=False
    )
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    return CardTokenCache(
        token_buffer=token_buffer,
        offsets=offsets_arr,
        row_to_name=row_to_name,
        engine_card_set_hash=compute_card_set_hash(names),
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


_SIDECAR_KEY = "metadata_json"


def save_card_cache(cache: CardTokenCache, path: Path | str) -> None:
    """Write the cache to ``path`` as an ``np.savez`` archive (no pickle)."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "row_to_name": cache.row_to_name,
            "engine_card_set_hash": cache.engine_card_set_hash,
        }
    )
    # Persist metadata as bytes via a 1-D uint8 array to avoid pickle.
    metadata_arr = np.frombuffer(metadata.encode("utf-8"), dtype=np.uint8)
    np.savez(
        p,
        token_buffer=cache.token_buffer,
        offsets=cache.offsets,
        metadata_json=metadata_arr,
    )


def load_card_cache(path: Path | str) -> CardTokenCache:
    p = Path(path)
    with np.load(p, allow_pickle=False) as archive:
        token_buffer = np.asarray(archive["token_buffer"], dtype=np.int32)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        metadata_bytes = bytes(archive[_SIDECAR_KEY].tobytes())
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    return CardTokenCache(
        token_buffer=token_buffer,
        offsets=offsets,
        row_to_name=list(metadata["row_to_name"]),
        engine_card_set_hash=str(metadata["engine_card_set_hash"]),
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
    lens = np.diff(cache.offsets[1:])  # one entry per real card row
    return {
        "count": int(lens.size),
        "mean": float(lens.mean()),
        "p50": int(np.percentile(lens, 50)),
        "p90": int(np.percentile(lens, 90)),
        "max": int(lens.max()),
    }
