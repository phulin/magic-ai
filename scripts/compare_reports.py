#!/usr/bin/env python3
"""
Compare two or more eval_card_embeddings reports as a side-by-side table.

Usage:
    uv run scripts/compare_reports.py reports/v0_json.json reports/v1_*.json

Each positional argument is a JSON report produced by `eval_card_embeddings.py`.
The first report is used as the baseline column for delta highlighting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    reports = [(p, json.loads(p.read_text())) for p in args.reports]
    if not reports:
        sys.exit("no reports provided")

    rows = [
        ("synonyms.mrr", lambda r: r["synonyms"].get("mrr")),
        ("synonyms.recall@1", lambda r: r["synonyms"].get("recall@1")),
        ("synonyms.recall@5", lambda r: r["synonyms"].get("recall@5")),
        ("near_neighbors.mrr", lambda r: r["near_neighbors"].get("mrr")),
        ("nn_strict.mrr", lambda r: r.get("near_neighbors_strict", {}).get("mrr")),
        ("nn_strict.recall@1", lambda r: r.get("near_neighbors_strict", {}).get("recall@1")),
        ("anti_pairs.violations", lambda r: r["anti_pairs"].get("violations")),
        ("clusters.intra_mean", lambda r: r["clusters"].get("intra_mean")),
        ("clusters.inter_mean", lambda r: r["clusters"].get("inter_mean")),
        ("clusters.gap", lambda r: r["clusters"].get("gap")),
        ("clusters.precision@5", lambda r: r["clusters"].get("precision@5")),
        ("clusters.precision_normalized", lambda r: r["clusters"].get("precision_normalized")),
        ("clusters.triplet_accuracy", lambda r: r["clusters"].get("triplet_accuracy")),
        ("held_out.synonym_recall@3", lambda r: r["held_out"].get("synonym_recall@3")),
        ("HEADLINE composite", lambda r: r.get("headline", {}).get("composite")),
    ]

    name_w = max(len(label) for label, _ in rows)
    col_w = 10
    label_keys = [Path(p).stem for p, _ in reports]
    header = f"{'metric':<{name_w}}  " + "  ".join(f"{k:>{col_w}}" for k in label_keys)
    print(header)
    print("-" * len(header))

    for label, getter in rows:
        cells: list[str] = []
        for _, report in reports:
            cells.append(format_cell(getter(report)))
        print(f"{label:<{name_w}}  " + "  ".join(f"{c:>{col_w}}" for c in cells))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    return parser.parse_args()


def format_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
