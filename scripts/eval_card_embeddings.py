#!/usr/bin/env python3
"""
Evaluate a card-embeddings JSON against a labeled pairs/clusters file.

Usage:
    uv run scripts/eval_card_embeddings.py \
        --embeddings data/card_oracle_embeddings.json \
        --labels data/embedding_eval_pairs.json \
        [--dimensions 512] [--top-k 5] [--report report.json]
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def main() -> None:
    args = parse_args()
    embeddings, names, name_to_index = load_embeddings(args.embeddings, dimensions=args.dimensions)
    labels = json.loads(args.labels.read_text())

    report: dict[str, Any] = {
        "embeddings_path": str(args.embeddings),
        "labels_path": str(args.labels),
        "card_count": len(names),
        "dimensions": embeddings.shape[1],
    }

    sims = embeddings @ embeddings.T

    print(f"embeddings: {args.embeddings} (cards={len(names)}, dims={embeddings.shape[1]})")

    report["synonyms"] = score_pairs(
        labels.get("synonyms", []),
        sims=sims,
        name_to_index=name_to_index,
        names=names,
        top_k=args.top_k,
        label="synonyms",
    )
    report["near_neighbors"] = score_pairs(
        labels.get("near_neighbors", []),
        sims=sims,
        name_to_index=name_to_index,
        names=names,
        top_k=args.top_k,
        label="near_neighbors",
    )
    report["anti_pairs"] = score_anti_pairs(
        labels.get("anti_pairs", []),
        sims=sims,
        name_to_index=name_to_index,
    )
    synonym_set = build_synonym_set(labels.get("synonyms", []))
    report["near_neighbors_strict"] = score_near_neighbors_strict(
        labels.get("near_neighbors", []),
        sims=sims,
        name_to_index=name_to_index,
        synonym_set=synonym_set,
        top_k=args.top_k,
    )
    report["clusters"] = score_clusters(
        labels.get("clusters", {}),
        embeddings=embeddings,
        sims=sims,
        name_to_index=name_to_index,
        names=names,
        top_k=args.top_k,
    )
    report["held_out"] = score_held_out(
        labels.get("held_out", []),
        labels=labels,
        sims=sims,
        name_to_index=name_to_index,
        names=names,
    )
    report["qualitative"] = qualitative_dump(
        labels.get("qualitative_queries", []),
        sims=sims,
        name_to_index=name_to_index,
        names=names,
        top_k=args.top_k,
    )
    report["headline"] = compute_headline(report)
    print(
        f"\nheadline       composite={fmt(report['headline']['composite'])}  "
        f"({', '.join(f'{k}={fmt(v)}' for k, v in report['headline']['components'].items())})"
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote report -> {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="truncate embeddings to this many dimensions and re-normalize",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_embeddings(
    path: Path, *, dimensions: int | None
) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    payload = json.loads(path.read_text())
    cards = payload["cards"]
    names: list[str] = []
    vectors: list[list[float]] = []
    for card in cards:
        emb = card.get("embedding")
        if emb is None:
            continue
        names.append(card["name"])
        vectors.append(emb)
    if not vectors:
        raise ValueError(f"no embeddings found in {path}")

    embeddings = torch.tensor(vectors, dtype=torch.float32)
    if dimensions is not None:
        if dimensions > embeddings.shape[1]:
            raise ValueError(
                f"--dimensions {dimensions} exceeds embedding width {embeddings.shape[1]}"
            )
        embeddings = embeddings[:, :dimensions]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    name_to_index = {card_key(name): i for i, name in enumerate(names)}
    return embeddings, names, name_to_index


def card_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def lookup(name: str, name_to_index: dict[str, int]) -> int | None:
    return name_to_index.get(card_key(name))


def neighbor_rank(sims: torch.Tensor, query: int, target: int) -> int:
    """Rank of `target` among neighbors of `query` (1-indexed; 1 == best),
    excluding `query` itself."""
    row = sims[query].clone()
    row[query] = float("-inf")
    order = torch.argsort(row, descending=True)
    (positions,) = torch.where(order == target)
    return int(positions.item()) + 1


def score_pairs(
    pairs: list[dict[str, Any]],
    *,
    sims: torch.Tensor,
    name_to_index: dict[str, int],
    names: list[str],
    top_k: int,
    label: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cos_values: list[float] = []
    rrs: list[float] = []
    hits1 = 0
    hits_k = 0
    missing: list[tuple[str, str]] = []

    for pair in pairs:
        a_name, b_name = pair["a"], pair["b"]
        i = lookup(a_name, name_to_index)
        j = lookup(b_name, name_to_index)
        if i is None or j is None:
            missing.append((a_name, b_name))
            continue
        cos = float(sims[i, j].item())
        rank_ab = neighbor_rank(sims, i, j)
        rank_ba = neighbor_rank(sims, j, i)
        rank = min(rank_ab, rank_ba)
        cos_values.append(cos)
        rrs.append(1.0 / rank)
        if rank == 1:
            hits1 += 1
        if rank <= top_k:
            hits_k += 1
        rows.append(
            {
                "a": a_name,
                "b": b_name,
                "cos": round(cos, 4),
                "rank_a_to_b": rank_ab,
                "rank_b_to_a": rank_ba,
            }
        )

    n = len(rows)
    mean_cos = mean(cos_values)
    mrr = mean(rrs)
    recall_1: float | None = hits1 / n if n else None
    recall_k: float | None = hits_k / n if n else None
    summary: dict[str, Any] = {
        "n": n,
        "missing": [list(pair) for pair in missing],
        "mean_cos": mean_cos,
        "mrr": mrr,
        "recall@1": recall_1,
        f"recall@{top_k}": recall_k,
        "rows": rows,
    }
    print(
        f"{label:<14} n={n:<3d} mean_cos={fmt(mean_cos)} "
        f"MRR={fmt(mrr)} "
        f"r@1={fmt(recall_1)} r@{top_k}={fmt(recall_k)}"
        + (f"  missing={len(missing)}" if missing else "")
    )
    return summary


def score_anti_pairs(
    anti: list[dict[str, Any]],
    *,
    sims: torch.Tensor,
    name_to_index: dict[str, int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    violations = 0
    for pair in anti:
        a_name, b_name = pair["a"], pair["b"]
        i = lookup(a_name, name_to_index)
        j = lookup(b_name, name_to_index)
        if i is None or j is None:
            rows.append({"a": a_name, "b": b_name, "missing": True})
            continue
        cos_ab = float(sims[i, j].item())
        row: dict[str, Any] = {"a": a_name, "b": b_name, "cos": round(cos_ab, 4)}
        violated = False

        if "max_cos" in pair:
            row["max_cos"] = pair["max_cos"]
            if cos_ab > pair["max_cos"]:
                violated = True
                row["violation"] = "exceeds_max_cos"

        decoy_name = pair.get("decoy")
        if decoy_name is not None:
            k = lookup(decoy_name, name_to_index)
            if k is None:
                row["decoy_missing"] = decoy_name
            else:
                cos_ak = float(sims[i, k].item())
                row["decoy"] = decoy_name
                row["cos_a_to_decoy"] = round(cos_ak, 4)
                if cos_ak <= cos_ab:
                    violated = True
                    row["violation"] = "decoy_not_closer"

        if violated:
            violations += 1
        rows.append(row)

    print(
        f"anti_pairs     n={len(anti):<3d} violations={violations}"
        + (
            "  ["
            + ", ".join(f"{r['a']}↔{r['b']} cos={r['cos']}" for r in rows if r.get("violation"))
            + "]"
            if violations
            else ""
        )
    )
    return {"violations": violations, "rows": rows}


def build_synonym_set(synonyms: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Map card_key -> set of card_keys it shouldn't compete with in near-neighbor
    retrieval (its declared synonyms)."""
    out: dict[str, set[str]] = {}
    for pair in synonyms:
        a, b = card_key(pair["a"]), card_key(pair["b"])
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def score_near_neighbors_strict(
    pairs: list[dict[str, Any]],
    *,
    sims: torch.Tensor,
    name_to_index: dict[str, int],
    synonym_set: dict[str, set[str]],
    top_k: int,
) -> dict[str, Any]:
    """Like score_pairs but for each anchor, exclude the anchor's *declared synonyms*
    from the candidate pool before ranking. Tests near-neighbor recall after removing
    the trivial 'a card has a stricter synonym' case."""
    rows: list[dict[str, Any]] = []
    rrs: list[float] = []
    hits1 = 0
    hits_k = 0
    n_total = 0
    for pair in pairs:
        a_name, b_name = pair["a"], pair["b"]
        i = lookup(a_name, name_to_index)
        j = lookup(b_name, name_to_index)
        if i is None or j is None:
            continue
        n_total += 1

        def filtered_rank(query: int, target: int, anchor_name: str) -> int:
            row = sims[query].clone()
            row[query] = float("-inf")
            anchor_synonyms = synonym_set.get(card_key(anchor_name), set())
            for syn_name in anchor_synonyms:
                idx = name_to_index.get(syn_name)
                if idx is not None and idx != target:
                    row[idx] = float("-inf")
            order = torch.argsort(row, descending=True)
            (positions,) = torch.where(order == target)
            return int(positions.item()) + 1

        rank_ab = filtered_rank(i, j, a_name)
        rank_ba = filtered_rank(j, i, b_name)
        rank = min(rank_ab, rank_ba)
        rrs.append(1.0 / rank)
        if rank == 1:
            hits1 += 1
        if rank <= top_k:
            hits_k += 1
        rows.append({"a": a_name, "b": b_name, "rank_a_to_b": rank_ab, "rank_b_to_a": rank_ba})

    summary: dict[str, Any] = {
        "n": n_total,
        "mrr": mean(rrs),
        "recall@1": hits1 / n_total if n_total else None,
        f"recall@{top_k}": hits_k / n_total if n_total else None,
        "rows": rows,
    }
    print(
        f"nn_strict      n={n_total:<3d} MRR={fmt(summary['mrr'])} "
        f"r@1={fmt(summary['recall@1'])} r@{top_k}={fmt(summary[f'recall@{top_k}'])}"
        " (synonyms excluded from candidate pool)"
    )
    return summary


def score_clusters(
    clusters: dict[str, list[str]],
    *,
    embeddings: torch.Tensor,
    sims: torch.Tensor,
    name_to_index: dict[str, int],
    names: list[str],
    top_k: int,
) -> dict[str, Any]:
    resolved: dict[str, list[int]] = {}
    missing: dict[str, list[str]] = {}
    for cname, members in clusters.items():
        idx: list[int] = []
        miss: list[str] = []
        for m in members:
            k = lookup(m, name_to_index)
            if k is None:
                miss.append(m)
            else:
                idx.append(k)
        if idx:
            resolved[cname] = idx
        if miss:
            missing[cname] = miss

    intra_per: dict[str, float] = {}
    for cname, idx in resolved.items():
        if len(idx) < 2:
            continue
        sub = sims[idx][:, idx]
        n = len(idx)
        # mean over off-diagonal upper triangle
        triu = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        intra_per[cname] = float(sub[triu].mean().item())

    inter_per: dict[str, dict[str, float]] = {}
    cluster_names = list(resolved.keys())
    for c1, c2 in combinations(cluster_names, 2):
        sub = sims[resolved[c1]][:, resolved[c2]]
        v = float(sub.mean().item())
        inter_per.setdefault(c1, {})[c2] = v
        inter_per.setdefault(c2, {})[c1] = v

    intra_mean = mean(list(intra_per.values()))
    inter_values = [v for inner in inter_per.values() for v in inner.values()]
    inter_mean = mean(inter_values) if inter_values else None

    # Per-card precision@k inside its own cluster.
    name_to_cluster: dict[int, str] = {}
    for cname, idx in resolved.items():
        for k in idx:
            name_to_cluster[k] = cname

    # Normalized precision: cap denominator at min(top_k, available_same_cluster).
    precisions_norm: list[float] = []
    precisions_at_k: list[float] = []
    per_card_gap: list[tuple[str, float, str]] = []  # (name, gap, nearest_other_cluster_card)
    for query_idx, cname in name_to_cluster.items():
        row = sims[query_idx].clone()
        row[query_idx] = float("-inf")
        order = torch.argsort(row, descending=True).tolist()
        topk_order = order[:top_k]
        same_in_topk = sum(1 for n in topk_order if name_to_cluster.get(n) == cname)
        precisions_at_k.append(same_in_topk / top_k)
        denom = min(top_k, len(resolved[cname]) - 1)
        if denom > 0:
            precisions_norm.append(same_in_topk / denom)

        # Per-card placement: gap = best-in-cluster cos minus best-out-of-cluster cos.
        best_in = float("-inf")
        best_out = float("-inf")
        best_out_idx: int | None = None
        for cand in order:
            cos = float(sims[query_idx, cand].item())
            other_cluster = name_to_cluster.get(cand)
            if other_cluster == cname:
                if cos > best_in:
                    best_in = cos
            else:
                if cos > best_out:
                    best_out = cos
                    best_out_idx = cand
        if best_in > float("-inf") and best_out > float("-inf"):
            per_card_gap.append(
                (
                    names[query_idx],
                    best_in - best_out,
                    names[best_out_idx] if best_out_idx is not None else "",
                )
            )

    # Triplet accuracy: for every (anchor, positive same-cluster, negative other-cluster)
    # triplet, fraction where cos(a,p) > cos(a,n). Computed efficiently in tensor form.
    triplet_correct = 0
    triplet_total = 0
    for query_idx, cname in name_to_cluster.items():
        same_idx = [k for k in resolved[cname] if k != query_idx]
        other_idx = [k for k, c in name_to_cluster.items() if c != cname and k != query_idx]
        if not same_idx or not other_idx:
            continue
        pos = sims[query_idx, same_idx]  # shape (P,)
        neg = sims[query_idx, other_idx]  # shape (N,)
        # broadcast: each positive vs each negative
        diffs = pos.unsqueeze(1) - neg.unsqueeze(0)  # (P, N), positive iff a,p > a,n
        triplet_correct += int((diffs > 0).sum().item())
        triplet_total += diffs.numel()
    triplet_acc = triplet_correct / triplet_total if triplet_total else None

    # Worst cluster: lowest (intra - max_other_inter).
    worst = None
    worst_gap = None
    for cname, intra in intra_per.items():
        others = inter_per.get(cname, {})
        max_other = max(others.values()) if others else 0.0
        gap = intra - max_other
        if worst_gap is None or gap < worst_gap:
            worst_gap = gap
            worst = cname

    # Worst-placed individual cards (lowest in/out gap).
    per_card_gap.sort(key=lambda row: row[1])
    worst_cards = [
        {"name": n, "in_minus_out_cos": round(g, 4), "nearest_outside_cluster": o}
        for n, g, o in per_card_gap[:5]
    ]

    gap = (intra_mean - inter_mean) if (intra_mean is not None and inter_mean is not None) else None
    print(
        f"clusters       intra={fmt(intra_mean)} inter={fmt(inter_mean)} "
        f"gap={fmt(gap)}  prec@{top_k}={fmt(mean(precisions_at_k))} "
        f"prec_norm={fmt(mean(precisions_norm))}  triplet_acc={fmt(triplet_acc)}"
    )
    if worst is not None:
        max_other_name = max(inter_per[worst].items(), key=lambda kv: kv[1])
        print(
            f"  worst cluster: {worst} (intra={fmt(intra_per[worst])}, "
            f"nearest_other={max_other_name[0]} @ {fmt(max_other_name[1])})"
        )
    if worst_cards:
        print("  worst-placed cards (best-in − best-out cosine):")
        for entry in worst_cards:
            gap_value = float(entry["in_minus_out_cos"])
            print(
                f"    {entry['name']:<25s} {gap_value:+.4f} "
                f"(nearest outsider: {entry['nearest_outside_cluster']})"
            )
    if missing:
        for cname, miss in missing.items():
            print(f"  missing in {cname}: {miss}")

    return {
        "intra_per_cluster": {k: round(v, 4) for k, v in intra_per.items()},
        "inter_per_pair": {
            k: {k2: round(v2, 4) for k2, v2 in v.items()} for k, v in inter_per.items()
        },
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "gap": gap,
        f"precision@{top_k}": mean(precisions_at_k),
        "precision_normalized": mean(precisions_norm),
        "triplet_accuracy": triplet_acc,
        "worst_cluster": worst,
        "worst_placed_cards": worst_cards,
        "missing": missing,
    }


def compute_headline(report: dict[str, Any]) -> dict[str, Any]:
    """Composite single-number score combining the main quality signals.

    All components are normalized to [0, 1]; higher is better. The composite is
    a simple weighted mean — weights reflect the project goal of generalizing
    to unseen cards via faithful rules-text encoding."""
    syn = report.get("synonyms", {})
    nn = report.get("near_neighbors_strict", {})
    clusters = report.get("clusters", {})
    anti = report.get("anti_pairs", {})
    held = report.get("held_out", {})

    syn_mrr = syn.get("mrr") or 0.0
    nn_mrr = nn.get("mrr") or 0.0
    triplet = clusters.get("triplet_accuracy") or 0.0
    cluster_gap = clusters.get("gap") or 0.0
    held_recall = held.get("synonym_recall@3") or 0.0
    n_anti_rows = len(anti.get("rows", []))
    anti_pass = 1.0 - (anti.get("violations", 0) / n_anti_rows) if n_anti_rows else 1.0

    components = {
        "synonym_mrr": syn_mrr,
        "near_neighbor_mrr_strict": nn_mrr,
        "triplet_accuracy": triplet,
        "cluster_gap": cluster_gap,  # not bounded to [0,1] but typically 0-0.3
        "anti_pair_pass_rate": anti_pass,
        "held_out_recall_at_3": held_recall,
    }
    weights = {
        "synonym_mrr": 0.25,
        "near_neighbor_mrr_strict": 0.15,
        "triplet_accuracy": 0.25,
        "cluster_gap": 0.10,
        "anti_pair_pass_rate": 0.15,
        "held_out_recall_at_3": 0.10,
    }
    composite = sum(weights[k] * components[k] for k in weights)
    return {"composite": composite, "components": components, "weights": weights}


def score_held_out(
    held_out: list[str],
    *,
    labels: dict[str, Any],
    sims: torch.Tensor,
    name_to_index: dict[str, int],
    names: list[str],
) -> dict[str, Any]:
    synonym_map: dict[str, list[str]] = {}
    for pair in labels.get("synonyms", []):
        synonym_map.setdefault(card_key(pair["a"]), []).append(pair["b"])
        synonym_map.setdefault(card_key(pair["b"]), []).append(pair["a"])

    rows: list[dict[str, Any]] = []
    hits = 0
    n_with_synonym = 0
    for query in held_out:
        i = lookup(query, name_to_index)
        if i is None:
            rows.append({"query": query, "missing": True})
            continue
        row = sims[i].clone()
        row[i] = float("-inf")
        top3 = torch.argsort(row, descending=True)[:3].tolist()
        top_neighbors = [names[k] for k in top3]
        synonyms = synonym_map.get(card_key(query), [])
        entry: dict[str, Any] = {
            "query": query,
            "top3": top_neighbors,
            "synonyms": synonyms,
        }
        if synonyms:
            n_with_synonym += 1
            top3_keys = {card_key(n) for n in top_neighbors}
            if any(card_key(s) in top3_keys for s in synonyms):
                hits += 1
                entry["synonym_in_top3"] = True
            else:
                entry["synonym_in_top3"] = False
        rows.append(entry)

    recall = hits / n_with_synonym if n_with_synonym else None
    print(
        f"held_out       n={len(held_out):<3d} synonym_recall@3={fmt(recall)} "
        f"({hits}/{n_with_synonym})"
    )
    return {"synonym_recall@3": recall, "rows": rows}


def qualitative_dump(
    queries: list[str],
    *,
    sims: torch.Tensor,
    name_to_index: dict[str, int],
    names: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not queries:
        return out
    print(f"\nqualitative top-{top_k} neighbors:")
    for query in queries:
        i = lookup(query, name_to_index)
        if i is None:
            print(f"  {query}: <not in embeddings>")
            continue
        row = sims[i].clone()
        row[i] = float("-inf")
        topk = torch.argsort(row, descending=True)[:top_k].tolist()
        neighbors = [(names[k], round(float(sims[i, k].item()), 4)) for k in topk]
        pretty = ", ".join(f"{n} ({c})" for n, c in neighbors)
        print(f"  {query}: {pretty}")
        out.append({"query": query, "neighbors": neighbors})
    return out


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
