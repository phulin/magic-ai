"""Decoder convergence smoke (post-inline-blank cutover).

Trains the autoregressive grammar decoder for N steps on the Forge
choice-situation corpus and reports per-decision-type accuracy on a
held-out eval split.

Usage::

    uv run scripts/decoder_bc_parity.py \
        --data data/forge_choice_situations \
        --steps 1500 --batch-size 32 --eval-batches 32

The default hyperparameters target a fast smoke check on a single A100
(~10 minutes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from magic_ai.text_encoder.card_cache import load_oracle_db  # noqa: E402
from magic_ai.text_encoder.decoder import GrammarDecoderConfig  # noqa: E402
from magic_ai.text_encoder.model import TextEncoderConfig  # noqa: E402
from magic_ai.text_encoder.policy_value_pretrain import (  # noqa: E402
    ForgeChoiceDataset,
    ForgePolicyValueConfig,
    ForgePolicyValueTrainer,
    _batch_to_device,
)
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
)
from magic_ai.text_encoder.tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/forge_choice_situations"))
    p.add_argument("--oracle-db", type=Path, default=Path("data/oracle-cards.json"))
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument(
        "--filter-kinds",
        type=str,
        default="",
        help="Comma-separated pending-kinds to include (priority, attackers, blockers, ...).",
    )
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def _build_policy(
    args: argparse.Namespace,
    *,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
) -> RecurrentTextPolicy:
    enc_cfg = TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_tokens,
        pad_id=pad_id,
    )
    dec_cfg = GrammarDecoderConfig(d_model=args.d_model, n_heads=args.heads, n_layers=2)
    rec_cfg = RecurrentTextPolicyConfig(
        encoder=enc_cfg,
        lstm_hidden=args.d_model,
        compile_forward=False,
        grammar_decoder_cfg=dec_cfg,
    )
    return RecurrentTextPolicy(rec_cfg).to(device)


def _filter_dataset(ds: ForgeChoiceDataset, kinds: set[str]) -> None:
    if not kinds:
        return
    ds.records = [r for r in ds.records if str((r.get("choice") or {}).get("kind") or "") in kinds]
    if not ds.records:
        raise RuntimeError(f"after filtering, no records remain (kinds={kinds})")


_DECISION_TYPE_NAMES = {
    0: "priority",
    1: "declare_attackers",
    2: "declare_blockers",
    3: "choose_targets",
    4: "may",
    5: "choose_mode",
    6: "choose_x",
}


def _decision_type_name(dt_val: int) -> str:
    return _DECISION_TYPE_NAMES.get(dt_val, f"unknown({dt_val})")


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"data path {args.data} does not exist", file=sys.stderr)
        return 1
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")

    print(f"[decoder-smoke] device={device} steps={args.steps} bs={args.batch_size}")
    tokenizer = load_tokenizer()
    pad_id = int(tokenizer.pad_token_id or 0)
    vocab_size = int(tokenizer.vocab_size + len(tokenizer.get_added_vocab()))
    print(f"[decoder-smoke] tokenizer vocab={vocab_size} pad_id={pad_id}")

    print(f"[decoder-smoke] loading oracle from {args.oracle_db}")
    oracle = load_oracle_db(args.oracle_db, names=None)
    print(f"[decoder-smoke] oracle entries={len(oracle):,}")

    cfg = ForgePolicyValueConfig(
        data_path=args.data,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        eval_fraction=0.1,
        pad_token_id=pad_id,
    )
    print("[decoder-smoke] loading splits")
    train_ds = ForgeChoiceDataset(cfg, tokenizer=tokenizer, oracle=oracle, split="train")
    eval_ds = ForgeChoiceDataset(cfg, tokenizer=tokenizer, oracle=oracle, split="eval")
    kind_filter = {k.strip() for k in args.filter_kinds.split(",") if k.strip()}
    if kind_filter:
        print(f"[decoder-smoke] filtering to kinds={kind_filter}")
        _filter_dataset(train_ds, kind_filter)
        _filter_dataset(eval_ds, kind_filter)
    print(f"[decoder-smoke] train={train_ds.n_examples} eval={eval_ds.n_examples}")
    print(f"[decoder-smoke] kind_counts={train_ds.kind_counts()}")

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    policy = _build_policy(args, vocab_size=vocab_size, pad_id=pad_id, device=device)
    trainer = ForgePolicyValueTrainer(policy, cfg, lr=args.lr, grad_clip=1.0)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[decoder-smoke] params={n_params:,}")

    use_amp = device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()

    t0 = time.time()
    step = 0
    while step < args.steps:
        for host_batch in train_ds.iter_epoch(args.batch_size, np_rng):
            if step >= args.steps:
                break
            batch = _batch_to_device(host_batch, device)
            log_now = (step % args.log_every == 0) or (step == args.steps - 1)
            with amp_ctx:
                stats = trainer.step(batch, compute_stats=log_now)
            if log_now:
                print(
                    f"[decoder-smoke] step={step:5d} loss={stats['loss']:.4f} "
                    f"policy={stats['policy_loss']:.4f} "
                    f"value={stats['value_loss']:.4f} "
                    f"acc={stats.get('decoder_step_accuracy', 0.0):.3f} "
                    f"grad={stats['grad_norm']:.2f} "
                    f"elapsed={time.time() - t0:.1f}s"
                )
            step += 1
    print(f"[decoder-smoke] train time {time.time() - t0:.1f}s")

    # Per-decision-type pointer-step accuracy on the eval split.
    policy.eval()
    eval_rng = np.random.default_rng(args.seed + 999)
    bucket: dict[str, list[int]] = {}

    def _bump(kind: str, c: int, t: int) -> None:
        if t == 0:
            return
        cur = bucket.setdefault(kind, [0, 0])
        cur[0] += c
        cur[1] += t

    with torch.no_grad():
        for _ in range(args.eval_batches):
            host_batch = eval_ds.sample_batch(args.batch_size, eval_rng)
            batch = _batch_to_device(host_batch, device)
            with amp_ctx:
                text_policy = policy.text_policy
                vocab_logits, pointer_logits = text_policy.forward_decoder_teacher_forced(
                    batch.encoded, batch.output_token_ids
                )
                neg_inf = torch.finfo(pointer_logits.dtype).min
                p_pred = pointer_logits.masked_fill(~batch.pointer_mask, neg_inf).argmax(-1)
                pointer_correct = (p_pred == batch.output_pointer_pos) & (
                    batch.output_is_pointer & batch.output_pad_mask
                )
                pointer_valid = batch.output_is_pointer & batch.output_pad_mask
                dt = batch.decision_type_per_row
                for dt_val in torch.unique(dt).tolist():
                    rows = dt == int(dt_val)
                    c = int(pointer_correct[rows].sum().item())
                    t = int(pointer_valid[rows].sum().item())
                    _bump(_decision_type_name(int(dt_val)), c, t)

    overall_correct = sum(c for c, _ in bucket.values())
    overall_total = sum(t for _, t in bucket.values())
    overall_acc = overall_correct / max(overall_total, 1)
    print("\n=========================== DECODER SMOKE ===========================")
    print("(Per-decision-type accuracy on the supervised pointer step.)")
    for kind in sorted(bucket):
        c, t = bucket[kind]
        print(f"  {kind:20s} {c / t:.4f}  ({c}/{t})")
    print(f"  {'OVERALL':20s} {overall_acc:.4f}  ({overall_correct}/{overall_total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
