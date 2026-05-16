"""Benchmark the Rust Forge extractor on a prefix of a Forge game ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CRATE_DIR = ROOT / "rust" / "forge_extract"
DEFAULT_ZIP = ROOT / "data" / "forge-games-20260509-115204.zip"


def _run(cmd: list[str], *, cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _extract_summary(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"extractor stdout did not contain a JSON summary: {stdout!r}")
    return json.loads(stdout[start : end + 1])


def _output_path(root: Path, output_format: str, run_idx: int) -> Path:
    if output_format == "arrow":
        return root / f"arrow-{run_idx:02d}"
    return root / f"choices-{run_idx:02d}.jsonl.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--limit-games", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=0, help="0 uses rayon default")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/forge_extract_bench"))
    parser.add_argument("--output-format", choices=("arrow", "jsonl-gz"), default="arrow")
    parser.add_argument("--kinds", default="all")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--arrow-shard-rows", type=int, default=262_144)
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()

    zip_path = args.zip.resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    _run(["cargo", "build", "--release"], cwd=CRATE_DIR)
    binary = CRATE_DIR / "target" / "release" / "forge_extract"

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_idx in range(args.runs):
        out = _output_path(args.out_root, args.output_format, run_idx)
        if out.exists():
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()
        cmd = [
            str(binary),
            "--zip",
            str(zip_path),
            "--out",
            str(out),
            "--output-format",
            args.output_format,
            "--limit-games",
            str(args.limit_games),
            "--kinds",
            args.kinds,
            "--progress-every",
            "0",
        ]
        if args.threads > 0:
            cmd += ["--threads", str(args.threads)]
        if args.trajectory:
            cmd.append("--trajectory")
        if args.output_format == "arrow":
            cmd += ["--arrow-shard-rows", str(args.arrow_shard_rows)]

        start = time.perf_counter()
        proc = _run(cmd, cwd=ROOT, capture=True)
        elapsed = time.perf_counter() - start
        summary = _extract_summary(proc.stdout)
        games_seen = int(summary["games_seen"])
        records_written = int(summary["records_written"])
        candidates_seen = int(summary["candidates_seen"])
        rows.append(
            {
                "run": run_idx,
                "seconds": elapsed,
                "games_per_second": games_seen / elapsed,
                "records_per_second": records_written / elapsed if records_written else 0.0,
                "candidates_per_second": candidates_seen / elapsed if candidates_seen else 0.0,
                **summary,
            }
        )
        if not args.keep_output and out.exists():
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()

    print(json.dumps({"zip": str(zip_path), "results": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
