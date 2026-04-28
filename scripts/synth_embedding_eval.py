#!/usr/bin/env python3
"""
Embed synthetic cards and check invariances + analogies that should hold under
a faithful rules-text encoder.

The test cards are constructed in-process — same `build_embedding_text` and
`add_transformers_embeddings` code paths as `build_card_embeddings.py`, so
results are directly comparable to that pipeline's outputs.

Usage:
    uv run scripts/synth_embedding_eval.py \
        --embedding-text-format json \
        [--embedding-model Qwen/Qwen3-Embedding-0.6B] \
        [--embedding-dimensions 1024] \
        [--report reports/synth_v0_json.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from scripts.build_card_embeddings import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_LENGTH,
    EMBEDDING_TEXT_FORMATS,
    add_transformers_embeddings,
    card_to_record,
)


def main() -> None:
    args = parse_args()
    cards = synthetic_cards()
    records = [
        card_to_record(card, embedding_text_format=args.embedding_text_format) for card in cards
    ]
    add_transformers_embeddings(
        records,
        model_name=args.embedding_model,
        dimensions=args.embedding_dimensions,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
    )

    embeddings = torch.tensor([r["embedding"] for r in records], dtype=torch.float32)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    name_to_idx = {r["name"]: i for i, r in enumerate(records)}

    print(
        f"format={args.embedding_text_format} "
        f"model={args.embedding_model} dims={embeddings.shape[1]} "
        f"cards={len(records)}"
    )

    report: dict[str, Any] = {
        "format": args.embedding_text_format,
        "model": args.embedding_model,
        "dimensions": embeddings.shape[1],
        "card_count": len(records),
    }
    report["permutation_invariance"] = run_permutation_tests(embeddings, name_to_idx)
    report["type_reorder"] = run_type_reorder_tests(embeddings, name_to_idx)
    report["differentiation"] = run_differentiation_tests(embeddings, name_to_idx)
    report["keyword_analogies"] = run_analogy_tests(embeddings, name_to_idx)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote report -> {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-text-format", choices=EMBEDDING_TEXT_FORMATS, default="json")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dimensions", type=int, default=DEFAULT_EMBEDDING_DIMENSIONS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Synthetic card construction
# ---------------------------------------------------------------------------


def make_card(
    *,
    name: str,
    type_line: str,
    oracle_text: str,
    mana_cost: str = "{1}{G}",
    power: str | None = "2",
    toughness: str | None = "2",
) -> dict[str, Any]:
    """Build a Scryfall-card-shaped dict consumable by `card_to_record`."""
    card: dict[str, Any] = {
        "name": name,
        "type_line": type_line,
        "oracle_text": oracle_text,
        "mana_cost": mana_cost,
        "colors": [],
        "color_identity": [],
    }
    if power is not None:
        card["power"] = power
    if toughness is not None:
        card["toughness"] = toughness
    return card


def keyword_card(label: str, keywords: list[str]) -> dict[str, Any]:
    """A 2/2 french-vanilla 'Bear' for {1}{G} with the given keywords as oracle text."""
    oracle = ", ".join(keywords) if keywords else ""
    return make_card(
        name=f"Synth {label}",
        type_line="Creature — Bear",
        oracle_text=oracle,
    )


def synthetic_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    # Singletons (used as analogy anchors)
    singletons = {
        "vanilla": [],
        "T": ["Trample"],
        "L": ["Lifelink"],
        "F": ["First strike"],
        "Y": ["Flying"],
        "V": ["Vigilance"],
        "H": ["Haste"],
        "D": ["Deathtouch"],
    }
    for label, kws in singletons.items():
        cards.append(keyword_card(label, kws))

    # Two-keyword permutation pairs (test ordering invariance)
    two_keyword_groups: dict[str, list[list[str]]] = {
        "TL": [["Trample", "Lifelink"], ["Lifelink", "Trample"]],
        "FL": [["First strike", "Lifelink"], ["Lifelink", "First strike"]],
        "YT": [["Flying", "Trample"], ["Trample", "Flying"]],
        "DT": [["Deathtouch", "Trample"], ["Trample", "Deathtouch"]],
        "VH": [["Vigilance", "Haste"], ["Haste", "Vigilance"]],
    }
    for group, perms in two_keyword_groups.items():
        for i, kws in enumerate(perms):
            cards.append(keyword_card(f"{group}_p{i}", kws))

    # Three-keyword full-permutation group (Flying/Trample/Lifelink) — 6 cards
    base = ["Flying", "Trample", "Lifelink"]
    for i, perm in enumerate(itertools.permutations(base)):
        cards.append(keyword_card(f"YTL_p{i}", list(perm)))

    # Creature-type reorder pair
    cards.append(
        make_card(
            name="Synth GoblinSoldier",
            type_line="Creature — Goblin Soldier",
            oracle_text="",
            mana_cost="{1}{R}",
        )
    )
    cards.append(
        make_card(
            name="Synth SoldierGoblin",
            type_line="Creature — Soldier Goblin",
            oracle_text="",
            mana_cost="{1}{R}",
        )
    )

    # Triggered-ability reorder pair (the controller chooses order on resolution
    # under MTG rules, so the embedding should be invariant).
    triggers_a = (
        "When CARDNAME enters the battlefield, draw a card.\nWhen CARDNAME dies, gain 2 life."
    )
    triggers_b = (
        "When CARDNAME dies, gain 2 life.\nWhen CARDNAME enters the battlefield, draw a card."
    )
    cards.append(
        make_card(
            name="Synth TrigA",
            type_line="Creature — Spirit",
            oracle_text=triggers_a,
            mana_cost="{2}{W}",
            power="2",
            toughness="2",
        )
    )
    cards.append(
        make_card(
            name="Synth TrigB",
            type_line="Creature — Spirit",
            oracle_text=triggers_b,
            mana_cost="{2}{W}",
            power="2",
            toughness="2",
        )
    )

    return cards


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


def cos(embeddings: torch.Tensor, i: int, j: int) -> float:
    return float(torch.dot(embeddings[i], embeddings[j]).item())


def diff_cos(embeddings: torch.Tensor, ai: int, bi: int, ci: int, di: int) -> float:
    """cos(embed[a]-embed[b], embed[c]-embed[d])."""
    u = embeddings[ai] - embeddings[bi]
    v = embeddings[ci] - embeddings[di]
    nu, nv = torch.linalg.norm(u), torch.linalg.norm(v)
    if nu.item() == 0.0 or nv.item() == 0.0:
        return float("nan")
    return float((u @ v / (nu * nv)).item())


def run_permutation_tests(embeddings: torch.Tensor, name_to_idx: dict[str, int]) -> dict[str, Any]:
    """For each keyword-set group, average pairwise cos sim across permutations.
    Should be ~1.0 for an order-invariant encoder."""
    print("\n# permutation invariance (target: cos ≈ 1.0)")
    groups = {
        "TL (2 perms)": ["Synth TL_p0", "Synth TL_p1"],
        "FL (2 perms)": ["Synth FL_p0", "Synth FL_p1"],
        "YT (2 perms)": ["Synth YT_p0", "Synth YT_p1"],
        "DT (2 perms)": ["Synth DT_p0", "Synth DT_p1"],
        "VH (2 perms)": ["Synth VH_p0", "Synth VH_p1"],
        "YTL (6 perms)": [f"Synth YTL_p{i}" for i in range(6)],
        "Triggered abilities order": ["Synth TrigA", "Synth TrigB"],
    }
    out: dict[str, Any] = {}
    for label, names in groups.items():
        idx = [name_to_idx[n] for n in names]
        sims: list[float] = []
        for a, b in itertools.combinations(idx, 2):
            sims.append(cos(embeddings, a, b))
        mean_sim = sum(sims) / len(sims)
        min_sim = min(sims)
        out[label] = {"mean_cos": round(mean_sim, 4), "min_cos": round(min_sim, 4)}
        print(f"  {label:<30s} mean={mean_sim:.4f}  min={min_sim:.4f}")
    return out


def run_type_reorder_tests(embeddings: torch.Tensor, name_to_idx: dict[str, int]) -> dict[str, Any]:
    print("\n# creature-type reorder (target: cos ≈ 1.0)")
    a = name_to_idx["Synth GoblinSoldier"]
    b = name_to_idx["Synth SoldierGoblin"]
    c = cos(embeddings, a, b)
    print(f"  Goblin Soldier vs Soldier Goblin     cos={c:.4f}")
    return {"goblin_soldier_vs_soldier_goblin": round(c, 4)}


def run_differentiation_tests(
    embeddings: torch.Tensor, name_to_idx: dict[str, int]
) -> dict[str, Any]:
    """Cards with different keyword sets should be measurably less similar than
    same-set permutations. We compare these to the permutation invariance numbers."""
    print("\n# differentiation (different keywords → lower cos than permutations)")
    pairs = [
        ("Synth T", "Synth L", "Trample vs Lifelink"),
        ("Synth T", "Synth F", "Trample vs First strike"),
        ("Synth Y", "Synth T", "Flying vs Trample"),
        ("Synth TL_p0", "Synth FL_p0", "{T,L} vs {F,L}"),
        ("Synth vanilla", "Synth T", "vanilla vs Trample"),
        ("Synth vanilla", "Synth Y", "vanilla vs Flying"),
    ]
    out: dict[str, Any] = {}
    for a_name, b_name, label in pairs:
        c = cos(embeddings, name_to_idx[a_name], name_to_idx[b_name])
        out[label] = round(c, 4)
        print(f"  {label:<30s} cos={c:.4f}")
    return out


def run_analogy_tests(embeddings: torch.Tensor, name_to_idx: dict[str, int]) -> dict[str, Any]:
    """Direction-of-keyword analogies. e.g. (TL - T) and (FL - F) should both
    encode '+ Lifelink' and so have high cosine similarity. Equivalently for
    '+ Trample', '+ Flying'."""
    print("\n# keyword direction analogies (target: cos ≈ 1.0)")
    n = name_to_idx
    analogies = [
        # +Lifelink direction
        ("(TL−T) vs (FL−F)  [+Lifelink]", "Synth TL_p0", "Synth T", "Synth FL_p0", "Synth F"),
        ("(L−vanilla) vs (TL−T) [+Lifelink]", "Synth L", "Synth vanilla", "Synth TL_p0", "Synth T"),
        # +Trample direction
        ("(T−vanilla) vs (TL−L)  [+Trample]", "Synth T", "Synth vanilla", "Synth TL_p0", "Synth L"),
        ("(YT−Y) vs (T−vanilla)  [+Trample]", "Synth YT_p0", "Synth Y", "Synth T", "Synth vanilla"),
        # +Flying direction
        ("(Y−vanilla) vs (YT−T)  [+Flying]", "Synth Y", "Synth vanilla", "Synth YT_p0", "Synth T"),
        # Cross-keyword orthogonality (target: lower than +keyword analogies)
        (
            "(T−vanilla) vs (L−vanilla)  [orthogonal]",
            "Synth T",
            "Synth vanilla",
            "Synth L",
            "Synth vanilla",
        ),
        (
            "(F−vanilla) vs (Y−vanilla)  [orthogonal]",
            "Synth F",
            "Synth vanilla",
            "Synth Y",
            "Synth vanilla",
        ),
    ]
    out: dict[str, Any] = {}
    for label, a, b, c, d in analogies:
        v = diff_cos(embeddings, n[a], n[b], n[c], n[d])
        out[label] = round(v, 4)
        print(f"  {label:<45s} cos={v:.4f}")
    return out


if __name__ == "__main__":
    main()
