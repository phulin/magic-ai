"""BC parity gate (steps 4–7 of docs/decoder_grammar_plan.md).

Trains the inline-blank pipeline and the autoregressive grammar decoder
back-to-back on the *same* Forge corpus with the *same* seed and
hyperparameters, then reports per-pipeline policy accuracy on a held-out
eval split. Used as the gate for proceeding to the native cgo path / RL
training under the new pipeline.

Usage::

    uv run scripts/decoder_bc_parity.py \
        --data data/forge_choice_situations \
        --steps 1500 --batch-size 32 --eval-batches 32 \
        --filter-kinds priority

The default hyperparameters target a fast parity check on a single A100
(~10 minutes per pipeline at d_model=128). Bump --d-model / --layers /
--steps for tighter convergence.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from magic_ai.text_encoder.card_cache import load_oracle_db  # noqa: E402
from magic_ai.text_encoder.decoder import GrammarDecoderConfig  # noqa: E402
from magic_ai.text_encoder.model import TextEncoderConfig  # noqa: E402
from magic_ai.text_encoder.policy_value_pretrain import (  # noqa: E402
    ForgeChoiceBatch,
    ForgeChoiceDataset,
    ForgeDecoderBatch,
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
    p.add_argument(
        "--pipelines",
        type=str,
        default="inline,decoder",
        help="Comma-separated subset of {inline,decoder}.",
    )
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def _build_policy(
    args: argparse.Namespace,
    *,
    use_grammar_decoder: bool,
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
    dec_cfg = (
        GrammarDecoderConfig(d_model=args.d_model, n_heads=args.heads, n_layers=2)
        if use_grammar_decoder
        else None
    )
    rec_cfg = RecurrentTextPolicyConfig(
        encoder=enc_cfg,
        lstm_hidden=args.d_model,
        compile_forward=False,
        use_grammar_decoder=use_grammar_decoder,
        grammar_decoder_cfg=dec_cfg,
    )
    return RecurrentTextPolicy(rec_cfg).to(device)


def _filter_dataset(ds: ForgeChoiceDataset, kinds: set[str]) -> None:
    if not kinds:
        return
    ds.records = [r for r in ds.records if str((r.get("choice") or {}).get("kind") or "") in kinds]
    if not ds.records:
        raise RuntimeError(f"after filtering, no records remain (kinds={kinds})")


def _train_one_pipeline(
    *,
    label: str,
    use_grammar_decoder: bool,
    args: argparse.Namespace,
    train_ds: ForgeChoiceDataset,
    eval_ds: ForgeChoiceDataset,
    tokenizer_vocab_size: int,
    pad_id: int,
    device: torch.device,
) -> dict[str, float]:
    print(f"\n[{label}] building policy use_grammar_decoder={use_grammar_decoder}")
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    policy = _build_policy(
        args,
        use_grammar_decoder=use_grammar_decoder,
        vocab_size=tokenizer_vocab_size,
        pad_id=pad_id,
        device=device,
    )
    cfg = ForgePolicyValueConfig(
        data_path=args.data,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        eval_fraction=0.1,
        decoder=use_grammar_decoder,
        pad_token_id=pad_id,
    )
    trainer = ForgePolicyValueTrainer(policy, cfg, lr=args.lr, grad_clip=1.0)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[{label}] params={n_params:,} train={train_ds.n_examples} eval={eval_ds.n_examples}")
    print(f"[{label}] kind_counts={train_ds.kind_counts()}")

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
                acc_key = "decoder_step_accuracy" if use_grammar_decoder else "priority_accuracy"
                print(
                    f"[{label}] step={step:5d} loss={stats['loss']:.4f} "
                    f"policy={stats['policy_loss']:.4f} "
                    f"value={stats['value_loss']:.4f} "
                    f"acc={stats.get(acc_key, 0.0):.3f} "
                    f"grad={stats['grad_norm']:.2f} "
                    f"elapsed={time.time() - t0:.1f}s"
                )
            step += 1
    print(f"[{label}] train time {time.time() - t0:.1f}s")

    # Eval
    policy.eval()
    eval_rng = np.random.default_rng(args.seed + 999)
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            host_batch = eval_ds.sample_batch(args.batch_size, eval_rng)
            batch = _batch_to_device(host_batch, device)
            with amp_ctx:
                if use_grammar_decoder:
                    dec_batch = cast(ForgeDecoderBatch, batch)
                    text_policy = policy.text_policy
                    vocab_logits, pointer_logits = text_policy.forward_decoder_teacher_forced(
                        dec_batch.encoded, dec_batch.output_token_ids
                    )
                    neg_inf = torch.finfo(vocab_logits.dtype).min
                    v_pred = vocab_logits.masked_fill(~dec_batch.vocab_mask, neg_inf).argmax(-1)
                    p_pred = pointer_logits.masked_fill(~dec_batch.pointer_mask, neg_inf).argmax(-1)
                    correct_per_step = torch.where(
                        dec_batch.output_is_pointer,
                        p_pred == dec_batch.output_pointer_pos,
                        v_pred == dec_batch.output_token_ids,
                    )
                    valid = dec_batch.output_pad_mask
                    correct += int((correct_per_step & valid).sum().item())
                    total += int(valid.sum().item())
                else:
                    inl_batch = cast(ForgeChoiceBatch, batch)
                    out, _ = policy(inl_batch.encoded, h_in=None, c_in=None)
                    blank_logits = out.blank_logits
                    if blank_logits is None:
                        continue
                    masked_scores = blank_logits[..., 0].masked_fill(
                        ~(inl_batch.encoded.blank_legal_mask[..., 0]), float("-inf")
                    )
                    pri_pred = masked_scores.argmax(dim=-1)
                    pri_target = inl_batch.priority_target_blank.to(pri_pred.device)
                    pri_correct = (pri_pred == pri_target).float()
                    valid_pri = pri_target >= 0
                    correct += int((pri_correct[valid_pri]).sum().item())
                    total += int(valid_pri.sum().item())

    final_acc = correct / max(total, 1)
    print(f"[{label}] eval per-step accuracy = {final_acc:.4f} (correct {correct}/{total})")
    return {"accuracy": final_acc, "correct": correct, "total": total}


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

    print(f"[parity] device={device} steps={args.steps} bs={args.batch_size}")
    tokenizer = load_tokenizer()
    pad_id = int(tokenizer.pad_token_id or 0)
    vocab_size = int(tokenizer.vocab_size + len(tokenizer.get_added_vocab()))
    print(f"[parity] tokenizer vocab={vocab_size} pad_id={pad_id}")

    print(f"[parity] loading oracle from {args.oracle_db}")
    oracle = load_oracle_db(args.oracle_db, names=None)
    print(f"[parity] oracle entries={len(oracle):,}")

    # Build datasets ONCE (one cfg per pipeline; the dataset's pre-loaded
    # records are shared, only cfg.decoder differs).
    base_cfg = ForgePolicyValueConfig(
        data_path=args.data,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        eval_fraction=0.1,
        decoder=False,
        pad_token_id=pad_id,
    )
    print("[parity] loading train split")
    train_ds = ForgeChoiceDataset(base_cfg, tokenizer=tokenizer, oracle=oracle, split="train")
    eval_ds = ForgeChoiceDataset(base_cfg, tokenizer=tokenizer, oracle=oracle, split="eval")
    kind_filter = {k.strip() for k in args.filter_kinds.split(",") if k.strip()}
    if kind_filter:
        print(f"[parity] filtering to kinds={kind_filter}")
        _filter_dataset(train_ds, kind_filter)
        _filter_dataset(eval_ds, kind_filter)
    print(f"[parity] train={train_ds.n_examples} eval={eval_ds.n_examples}")

    pipelines = [s.strip() for s in args.pipelines.split(",") if s.strip()]
    results: dict[str, dict[str, float]] = {}
    for pipeline in pipelines:
        if pipeline not in ("inline", "decoder"):
            print(f"unknown pipeline {pipeline!r}; skipping", file=sys.stderr)
            continue
        # Rebuild datasets with the right cfg.decoder so the DecisionSpecRenderer
        # gets initialized — it's set in ForgeChoiceDataset.__init__ based on cfg.decoder.
        run_cfg = replace(base_cfg, decoder=(pipeline == "decoder"))
        print(f"[parity] rebuilding datasets for pipeline={pipeline}")
        train_run = ForgeChoiceDataset(run_cfg, tokenizer=tokenizer, oracle=oracle, split="train")
        eval_run = ForgeChoiceDataset(run_cfg, tokenizer=tokenizer, oracle=oracle, split="eval")
        if kind_filter:
            _filter_dataset(train_run, kind_filter)
            _filter_dataset(eval_run, kind_filter)
        results[pipeline] = _train_one_pipeline(
            label=pipeline,
            use_grammar_decoder=(pipeline == "decoder"),
            args=args,
            train_ds=train_run,
            eval_ds=eval_run,
            tokenizer_vocab_size=vocab_size,
            pad_id=pad_id,
            device=device,
        )

    print("\n=========================== PARITY SUMMARY ===========================")
    for pipeline, r in results.items():
        print(f"  {pipeline:8s} accuracy={r['accuracy']:.4f}  ({r['correct']}/{r['total']})")
    if "inline" in results and "decoder" in results:
        delta = results["decoder"]["accuracy"] - results["inline"]["accuracy"]
        print(f"  delta(decoder - inline) = {delta:+.4f} ({delta * 100:+.2f}pp)")
        gate_pp = 0.005
        if abs(delta) <= gate_pp:
            print(f"  PARITY GATE PASSED (|delta| ≤ {gate_pp * 100:.1f}pp)")
        elif delta > 0:
            print(f"  decoder OUTPERFORMS inline by {delta * 100:+.2f}pp — better than parity")
        else:
            print(f"  decoder UNDER inline by {-delta * 100:.2f}pp — investigate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
