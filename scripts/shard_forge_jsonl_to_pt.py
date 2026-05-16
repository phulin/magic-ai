"""Convert a Forge choice JSONL.GZ artifact (e.g. produced by the Rust
``forge_extract`` binary) into the sharded ``part-*.pt`` directory format
that ``policy_value_pretrain.py`` consumes.

The records are written through verbatim — no rendering or tokenization
happens here, since ``ForgeChoiceDataset`` re-renders snapshots at
training time anyway.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Any, cast

import orjson
import torch

FORMAT_VERSION = 2


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield orjson.loads(stripped)


def _write_shard(
    out_dir: Path, shard_index: int, records: list[dict[str, Any]], *, compress: bool
) -> None:
    suffix = ".pt.gz" if compress else ".pt"
    path = out_dir / f"part-{shard_index:05d}{suffix}"
    tmp = path.with_name(f"{path.name}.tmp")
    payload = {
        "format": "forge_choice_situations_torch_shard",
        "format_version": FORMAT_VERSION,
        "shard_index": shard_index,
        "records": records,
    }
    if compress:
        with gzip.open(tmp, "wb", compresslevel=6) as fh:
            torch.save(payload, cast(Any, fh))
    else:
        torch.save(payload, tmp)
    tmp.replace(path)


def _intern_game_atoms(
    record: dict[str, Any],
    *,
    current_game_id: str | None,
    current_outcome: Any,
    current_archive_member: str | None,
) -> tuple[str, Any, str | None]:
    """Reuse identical per-game objects so torch.save pickles them once per shard."""

    game_id = str(record.get("game_id") or "")
    if game_id != current_game_id:
        archive_member = str(record.get("archive_member") or "")
        return game_id, record.get("outcome"), archive_member

    assert current_game_id is not None
    record["game_id"] = current_game_id
    if current_archive_member is not None:
        record["archive_member"] = current_archive_member
    if current_outcome is not None:
        record["outcome"] = current_outcome
    return current_game_id, current_outcome, current_archive_member


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument(
        "--compress",
        action="store_true",
        help="write gzip-compressed part-*.pt.gz shards instead of uncompressed .pt files",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    existing = [
        *args.out.glob("part-*.pt"),
        *args.out.glob("part-*.pt.gz"),
        args.out / "manifest.json",
    ]
    existing = [p for p in existing if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{args.out} already populated; pass --overwrite")
    for p in existing:
        p.unlink()

    shard: list[dict[str, Any]] = []
    shard_index = 0
    total = 0
    kind_counts: dict[str, int] = {}
    last_game_id: str | None = None
    current_outcome: Any = None
    current_archive_member: str | None = None
    # Game-atomic shards: flush only at a game boundary, so the dataset can
    # group records by game without crossing shard files. Sequenced
    # pretraining (`--pretrain-mlm-sequence-mode full`) relies on this.
    for record in _iter_jsonl(args.input):
        kind = str((record.get("choice") or {}).get("kind") or "unknown")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        game_id = str(record.get("game_id") or "")
        if last_game_id is not None and game_id != last_game_id and len(shard) >= args.shard_size:
            _write_shard(args.out, shard_index, shard, compress=args.compress)
            shard_index += 1
            shard = []
            current_outcome = None
            current_archive_member = None
        game_id, current_outcome, current_archive_member = _intern_game_atoms(
            record,
            current_game_id=last_game_id,
            current_outcome=current_outcome,
            current_archive_member=current_archive_member,
        )
        shard.append(record)
        total += 1
        last_game_id = game_id
    if shard:
        _write_shard(args.out, shard_index, shard, compress=args.compress)
        shard_index += 1

    manifest = {
        "format": "forge_choice_situations_manifest",
        "format_version": FORMAT_VERSION,
        "compressed": bool(args.compress),
        "shards": shard_index,
        "shard_size": args.shard_size,
        "stats": {
            "records_written": total,
            **{f"written_{k}": v for k, v in kind_counts.items()},
        },
    }
    (args.out / "manifest.json").write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    print(orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode())


if __name__ == "__main__":
    main()
