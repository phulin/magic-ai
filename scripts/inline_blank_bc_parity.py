"""Priority-only BC parity gate for inline blanks.

Runs the Step 5 gate from ``docs/text_encoder_inline_blanks_plan.md``: train
legacy option-head BC and inline cross-blank BC on the same fixed priority trace
set, then compare held-out accuracy.

Trace JSONL rows may use either of these shapes:

    {"snapshot": {...}, "selected_option_index": 2}
    {"snapshot": {...}, "selected_option_id": "engine-option-id"}

Only ``pending.kind == "priority"`` rows are accepted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

# Allow direct invocation as ``uv run python scripts/inline_blank_bc_parity.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.actions import (  # noqa: E402
    build_priority_candidates,
    selected_priority_candidate_index,
)
from magic_ai.game_state import (  # noqa: E402
    GameCardState,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
    PlayerState,
)
from magic_ai.text_encoder.batch import TextEncodedBatch, collate, tokenize_snapshot  # noqa: E402
from magic_ai.text_encoder.model import TextEncoderConfig  # noqa: E402
from magic_ai.text_encoder.policy import TextPolicy  # noqa: E402
from magic_ai.text_encoder.render import (  # noqa: E402
    OracleEntry,
    RenderedSnapshot,
    load_oracle_text,
    render_snapshot,
)
from magic_ai.text_encoder.tokenizer import TOKENIZER_DIR, load_tokenizer  # noqa: E402
from magic_ai.text_encoder.training import (  # noqa: E402
    inline_blank_priority_accuracy,
    inline_blank_priority_loss,
)


@dataclass(frozen=True)
class PriorityTraceRow:
    snapshot: GameStateSnapshot
    selected_option_index: int


@dataclass(frozen=True)
class EncodedParityRows:
    legacy: TextEncodedBatch
    inline: TextEncodedBatch
    legacy_target: Tensor
    inline_target: Tensor


@dataclass(frozen=True)
class EvalStats:
    loss: float
    accuracy: float
    total: int


def _hash_path(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    else:
        with path.open("rt", encoding="utf-8") as fh:
            yield from fh


def _snapshot_from_payload(payload: dict[str, Any]) -> GameStateSnapshot:
    if "snapshot" in payload:
        return cast(GameStateSnapshot, payload["snapshot"])
    if "state" in payload:
        snapshot = dict(cast(dict[str, Any], payload["state"]))
        if "pending" in payload:
            snapshot["pending"] = payload["pending"]
        return cast(GameStateSnapshot, snapshot)
    return cast(GameStateSnapshot, payload)


def _priority_option_index_from_trace(pending: PendingState, trace: dict[str, Any]) -> int:
    if trace.get("kind") != "priority":
        raise ValueError("trace row is not a priority trace")
    indices = trace.get("indices", [])
    if not indices:
        return 0
    candidate_index = int(indices[0])
    candidates = build_priority_candidates(pending)
    if candidates:
        if not 0 <= candidate_index < len(candidates):
            raise ValueError(
                f"priority trace index {candidate_index} out of range for "
                f"{len(candidates)} candidates"
            )
        return int(candidates[candidate_index].option_index)
    return candidate_index


def _selected_option_index(snapshot: GameStateSnapshot, payload: dict[str, Any]) -> int:
    pending = snapshot.get("pending")
    if pending is None or pending.get("kind") != "priority":
        raise ValueError("trace row is not a priority snapshot")
    options = pending.get("options", [])
    if "selected_option_index" in payload:
        index = int(payload["selected_option_index"])
        if not 0 <= index < len(options):
            raise ValueError(
                f"selected_option_index {index} out of range for {len(options)} options"
            )
        return index
    if "selected_option_id" in payload:
        selected_id = str(payload["selected_option_id"])
        for idx, option in enumerate(options):
            if str(option.get("id", "")) == selected_id:
                return idx
        raise ValueError(f"selected_option_id {selected_id!r} not present in priority options")
    trace = payload.get("trace")
    if isinstance(trace, dict):
        index = _priority_option_index_from_trace(pending, trace)
        if not 0 <= index < len(options):
            raise ValueError(
                f"trace-selected option {index} out of range for {len(options)} options"
            )
        return index
    action = payload.get("action")
    if isinstance(action, dict):
        candidate_index = selected_priority_candidate_index(pending, action)
        if candidate_index < 0:
            raise ValueError("action does not match any priority candidate")
        candidates = build_priority_candidates(pending)
        if not 0 <= candidate_index < len(candidates):
            raise ValueError(
                f"action-selected candidate {candidate_index} out of range for "
                f"{len(candidates)} candidates"
            )
        return int(candidates[candidate_index].option_index)
    raise ValueError(
        "trace row needs selected_option_index, selected_option_id, priority trace, "
        "or priority action"
    )


def load_priority_trace(path: Path, *, limit: int | None = None) -> list[PriorityTraceRow]:
    rows: list[PriorityTraceRow] = []
    for line_no, line in enumerate(_open_text(path), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        snapshot = _snapshot_from_payload(payload)
        try:
            selected = _selected_option_index(snapshot, payload)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        rows.append(PriorityTraceRow(snapshot=snapshot, selected_option_index=selected))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"no priority trace rows loaded from {path}")
    return rows


def _card(cid: str, name: str, *, tapped: bool | None = None) -> GameCardState:
    out: dict[str, object] = {"ID": cid, "Name": name}
    if tapped is not None:
        out["Tapped"] = tapped
    return cast(GameCardState, out)


def _player(
    pid: str,
    name: str,
    *,
    hand: list[GameCardState] | None = None,
    battlefield: list[GameCardState] | None = None,
) -> PlayerState:
    return cast(
        PlayerState,
        {
            "ID": pid,
            "Name": name,
            "Life": 20,
            "HandCount": len(hand or []),
            "GraveyardCount": 0,
            "LibraryCount": 53,
            "Hand": hand or [],
            "Battlefield": battlefield or [],
            "Graveyard": [],
            "ManaPool": {
                "White": 0,
                "Blue": 0,
                "Black": 0,
                "Red": 0,
                "Green": 0,
                "Colorless": 0,
            },
        },
    )


def make_synthetic_priority_trace(size: int) -> list[PriorityTraceRow]:
    if size < 2:
        raise ValueError("synthetic fixture needs at least two rows")
    names = ("Lightning Bolt", "Counterspell", "Llanowar Elves", "Serra Angel")
    rows: list[PriorityTraceRow] = []
    for i in range(size):
        hand = [
            _card(f"h{i}-0", names[i % len(names)]),
            _card(f"h{i}-1", names[(i + 1) % len(names)]),
        ]
        battlefield = [_card(f"b{i}-0", "Llanowar Elves", tapped=bool(i % 2))]
        options = [
            cast(
                PendingOptionState,
                {
                    "id": f"cast-{i}-0",
                    "kind": "cast",
                    "card_id": hand[0]["ID"],
                    "card_name": hand[0]["Name"],
                },
            ),
            cast(
                PendingOptionState,
                {
                    "id": f"cast-{i}-1",
                    "kind": "cast",
                    "card_id": hand[1]["ID"],
                    "card_name": hand[1]["Name"],
                },
            ),
            cast(
                PendingOptionState,
                {
                    "id": f"act-{i}",
                    "kind": "activate",
                    "permanent_id": battlefield[0]["ID"],
                    "ability_index": 0,
                },
            ),
            cast(PendingOptionState, {"id": f"pass-{i}", "kind": "pass"}),
        ]
        pending = cast(PendingState, {"kind": "priority", "player_idx": 0, "options": options})
        snapshot = cast(
            GameStateSnapshot,
            {
                "turn": 1 + (i % 6),
                "active_player": "p1",
                "step": "Precombat Main",
                "players": [
                    _player("p1", "Self", hand=hand, battlefield=battlefield),
                    _player("p2", "Opp"),
                ],
                "pending": pending,
            },
        )
        rows.append(PriorityTraceRow(snapshot=snapshot, selected_option_index=i % len(options)))
    return rows


def _target_legacy_ordinal(rendered: RenderedSnapshot, selected_option_index: int) -> int:
    for ordinal, anchor in enumerate(rendered.option_anchors):
        if anchor.option_index == selected_option_index:
            return ordinal
    raise ValueError(f"selected option {selected_option_index} has no legacy option anchor")


def _target_inline_blank_index(rendered: RenderedSnapshot, selected_option_index: int) -> int:
    for anchor in rendered.blank_anchors:
        if anchor.option_index == selected_option_index:
            return int(anchor.blank_index)
    raise ValueError(f"selected option {selected_option_index} has no inline blank anchor")


def encode_parity_rows(
    rows: Sequence[PriorityTraceRow],
    *,
    tokenizer: Any,
    oracle: dict[str, OracleEntry] | None,
    chosen_token_id: int,
) -> EncodedParityRows:
    legacy_rendered: list[RenderedSnapshot] = []
    inline_rendered: list[RenderedSnapshot] = []
    legacy_targets: list[int] = []
    inline_targets: list[int] = []
    for row in rows:
        actions = row.snapshot.get("pending", {}).get("options", [])
        legacy = render_snapshot(row.snapshot, actions, oracle=oracle)
        inline = render_snapshot(
            row.snapshot,
            actions,
            oracle=oracle,
            use_inline_blanks=True,
            chosen_token_id=chosen_token_id,
        )
        legacy_rendered.append(legacy)
        inline_rendered.append(inline)
        legacy_targets.append(_target_legacy_ordinal(legacy, row.selected_option_index))
        inline_targets.append(_target_inline_blank_index(inline, row.selected_option_index))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad token")
    return EncodedParityRows(
        legacy=collate([tokenize_snapshot(r, tokenizer) for r in legacy_rendered], int(pad_id)),
        inline=collate([tokenize_snapshot(r, tokenizer) for r in inline_rendered], int(pad_id)),
        legacy_target=torch.tensor(legacy_targets, dtype=torch.long),
        inline_target=torch.tensor(inline_targets, dtype=torch.long),
    )


def _legacy_priority_loss(logits: Tensor, mask: Tensor, target: Tensor) -> Tensor:
    safe_target = target.clamp(min=0, max=max(0, logits.shape[1] - 1))
    target_is_supported = mask.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    valid_rows = (target >= 0) & (target < logits.shape[1]) & target_is_supported
    if not valid_rows.any():
        return logits.sum() * 0.0
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    return F.cross_entropy(masked_logits[valid_rows], target[valid_rows])


@torch.no_grad()
def _legacy_priority_accuracy(
    logits: Tensor, mask: Tensor, target: Tensor
) -> dict[str, float | int]:
    safe_target = target.clamp(min=0, max=max(0, logits.shape[1] - 1))
    target_is_supported = mask.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    valid_rows = (target >= 0) & (target < logits.shape[1]) & target_is_supported
    total = int(valid_rows.sum().item())
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    pred = logits.masked_fill(~mask, float("-inf"))[valid_rows].argmax(dim=-1)
    correct = int((pred == target[valid_rows]).sum().item())
    return {"accuracy": correct / total, "correct": correct, "total": total}


def _batch_to_device(batch: TextEncodedBatch, device: torch.device) -> TextEncodedBatch:
    return TextEncodedBatch(
        token_ids=batch.token_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        card_ref_positions=batch.card_ref_positions.to(device),
        option_positions=batch.option_positions.to(device),
        option_mask=batch.option_mask.to(device),
        target_positions=batch.target_positions.to(device),
        target_mask=batch.target_mask.to(device),
        seq_lengths=batch.seq_lengths.to(device),
        blank_positions=batch.blank_positions.to(device),
        blank_kind=batch.blank_kind.to(device),
        blank_group=batch.blank_group.to(device),
        blank_group_kind=batch.blank_group_kind.to(device),
        blank_legal_ids=batch.blank_legal_ids.to(device),
        blank_legal_mask=batch.blank_legal_mask.to(device),
    )


def _slice_batch(batch: TextEncodedBatch, idx: Tensor) -> TextEncodedBatch:
    return TextEncodedBatch(
        token_ids=batch.token_ids[idx],
        attention_mask=batch.attention_mask[idx],
        card_ref_positions=batch.card_ref_positions[idx],
        option_positions=batch.option_positions[idx],
        option_mask=batch.option_mask[idx],
        target_positions=batch.target_positions[idx],
        target_mask=batch.target_mask[idx],
        seq_lengths=batch.seq_lengths[idx],
        blank_positions=batch.blank_positions[idx],
        blank_kind=batch.blank_kind[idx],
        blank_group=batch.blank_group[idx],
        blank_group_kind=batch.blank_group_kind[idx],
        blank_legal_ids=batch.blank_legal_ids[idx],
        blank_legal_mask=batch.blank_legal_mask[idx],
    )


def _iter_minibatches(
    n: int, batch_size: int, *, shuffle: bool, device: torch.device
) -> Iterable[Tensor]:
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def train_legacy(
    model: TextPolicy,
    batch: TextEncodedBatch,
    target: Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = int(target.shape[0])
    for _epoch in range(epochs):
        for idx in _iter_minibatches(n, batch_size, shuffle=True, device=target.device):
            out = model(_slice_batch(batch, idx))
            loss = _legacy_priority_loss(out.policy_logits, out.option_mask, target[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def train_inline(
    model: TextPolicy,
    batch: TextEncodedBatch,
    target: Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = int(target.shape[0])
    for _epoch in range(epochs):
        for idx in _iter_minibatches(n, batch_size, shuffle=True, device=target.device):
            out = model(_slice_batch(batch, idx))
            if out.blank_logits is None or out.blank_group is None or out.blank_group_kind is None:
                raise RuntimeError("inline model did not produce blank outputs")
            loss = inline_blank_priority_loss(
                out.blank_logits,
                out.blank_group,
                out.blank_group_kind,
                cast(Tensor, out.blank_legal_mask),
                target[idx],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


@torch.no_grad()
def eval_legacy(model: TextPolicy, batch: TextEncodedBatch, target: Tensor) -> EvalStats:
    model.eval()
    out = model(batch)
    loss = _legacy_priority_loss(out.policy_logits, out.option_mask, target)
    stats = _legacy_priority_accuracy(out.policy_logits, out.option_mask, target)
    return EvalStats(
        loss=float(loss.item()),
        accuracy=float(stats["accuracy"]),
        total=int(stats["total"]),
    )


@torch.no_grad()
def eval_inline(model: TextPolicy, batch: TextEncodedBatch, target: Tensor) -> EvalStats:
    model.eval()
    out = model(batch)
    if out.blank_logits is None or out.blank_group is None or out.blank_group_kind is None:
        raise RuntimeError("inline model did not produce blank outputs")
    loss = inline_blank_priority_loss(
        out.blank_logits,
        out.blank_group,
        out.blank_group_kind,
        cast(Tensor, out.blank_legal_mask),
        target,
    )
    stats = inline_blank_priority_accuracy(
        out.blank_logits,
        out.blank_group,
        out.blank_group_kind,
        cast(Tensor, out.blank_legal_mask),
        target,
    )
    return EvalStats(
        loss=float(loss.item()),
        accuracy=float(stats["accuracy"]),
        total=int(stats["total"]),
    )


def _split_indices(n: int, train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if n < 2:
        raise ValueError("need at least two rows for a train/eval split")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_n = min(n - 1, max(1, int(round(n * train_frac))))
    return indices[:train_n], indices[train_n:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace-jsonl", type=Path)
    source.add_argument("--synthetic-fixture", type=int)
    p.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--oracle", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=192)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--regression-pp", type=float, default=0.5)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("--train-frac must be in (0, 1)")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    chosen_id = tokenizer.convert_tokens_to_ids("<chosen>")
    if isinstance(chosen_id, list):
        raise TypeError("convert_tokens_to_ids('<chosen>') returned a list")
    oracle = load_oracle_text(args.oracle) if args.oracle is not None else None

    if args.trace_jsonl is not None:
        source_kind = "trace_jsonl"
        source_path = str(args.trace_jsonl)
        source_sha256 = _hash_path(args.trace_jsonl)
        rows = load_priority_trace(args.trace_jsonl, limit=args.limit)
    else:
        source_kind = "synthetic_fixture"
        source_path = None
        source_sha256 = None
        rows = make_synthetic_priority_trace(int(args.synthetic_fixture))
        if args.limit is not None:
            rows = rows[: args.limit]

    train_idx, eval_idx = _split_indices(len(rows), float(args.train_frac), int(args.seed))
    train_rows = [rows[i] for i in train_idx]
    eval_rows = [rows[i] for i in eval_idx]

    train = encode_parity_rows(
        train_rows,
        tokenizer=tokenizer,
        oracle=oracle,
        chosen_token_id=int(chosen_id),
    )
    eval_rows_encoded = encode_parity_rows(
        eval_rows,
        tokenizer=tokenizer,
        oracle=oracle,
        chosen_token_id=int(chosen_id),
    )

    legacy_train = _batch_to_device(train.legacy, device)
    inline_train = _batch_to_device(train.inline, device)
    legacy_eval = _batch_to_device(eval_rows_encoded.legacy, device)
    inline_eval = _batch_to_device(eval_rows_encoded.inline, device)
    legacy_train_target = train.legacy_target.to(device)
    inline_train_target = train.inline_target.to(device)
    legacy_eval_target = eval_rows_encoded.legacy_target.to(device)
    inline_eval_target = eval_rows_encoded.inline_target.to(device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad token")
    legacy_model = TextPolicy(
        TextEncoderConfig(
            vocab_size=len(tokenizer),
            d_model=int(args.d_model),
            n_layers=int(args.n_layers),
            n_heads=int(args.n_heads),
            d_ff=int(args.d_ff),
            max_seq_len=int(args.max_seq_len),
            pad_id=int(pad_id),
        )
    ).to(device)
    torch.manual_seed(args.seed)
    inline_model = TextPolicy(
        TextEncoderConfig(
            vocab_size=len(tokenizer),
            d_model=int(args.d_model),
            n_layers=int(args.n_layers),
            n_heads=int(args.n_heads),
            d_ff=int(args.d_ff),
            max_seq_len=int(args.max_seq_len),
            pad_id=int(pad_id),
        )
    ).to(device)

    train_legacy(
        legacy_model,
        legacy_train,
        legacy_train_target,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
    )
    train_inline(
        inline_model,
        inline_train,
        inline_train_target,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
    )

    legacy_stats = eval_legacy(legacy_model, legacy_eval, legacy_eval_target)
    inline_stats = eval_inline(inline_model, inline_eval, inline_eval_target)
    regression_pp = (legacy_stats.accuracy - inline_stats.accuracy) * 100.0
    passed = regression_pp <= float(args.regression_pp)
    result = {
        "source": {
            "kind": source_kind,
            "path": source_path,
            "sha256": source_sha256,
            "limit": args.limit,
        },
        "seed": int(args.seed),
        "model": {
            "d_model": int(args.d_model),
            "n_layers": int(args.n_layers),
            "n_heads": int(args.n_heads),
            "d_ff": int(args.d_ff),
            "max_seq_len": int(args.max_seq_len),
        },
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "train_frac": float(args.train_frac),
        },
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "legacy": legacy_stats.__dict__,
        "inline": inline_stats.__dict__,
        "accuracy_regression_pp": regression_pp,
        "threshold_pp": float(args.regression_pp),
        "passed": passed,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
