"""Tests for ``magic_ai.text_encoder.value_pretrain``.

Covers paired bin/json loading, label parsing, and a smoke that the
encoder + value head actually train under the MSE objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from magic_ai.text_encoder.model import TextEncoderConfig, TextStateEncoder, ValueHead
from magic_ai.text_encoder.value_pretrain import (
    ValueLabeledBinDataset,
    ValuePretrainConfig,
    ValuePretrainTrainer,
)


def _write_pair(
    dir_: Path,
    game_id: str,
    tokens: np.ndarray,
    spans: list[dict],
    winner_id: str | None,
) -> None:
    tokens.astype(np.uint16, copy=False).tofile(dir_ / f"{game_id}.bin")
    sidecar = {
        "game_id": game_id,
        "winner_id": winner_id,
        "is_draw": winner_id is None,
        "players": [{"id": "p0", "name": "A"}, {"id": "p1", "name": "B"}],
        "spans": spans,
    }
    (dir_ / f"{game_id}.json").write_text(json.dumps(sidecar))


def test_value_dataset_loads_paired_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, 200, size=512, dtype=np.int64)
    spans = [
        {"offset": 0, "length": 128, "perspective_id": "p0", "label": 1.0},
        {"offset": 128, "length": 128, "perspective_id": "p1", "label": -1.0},
        {"offset": 256, "length": 64, "perspective_id": "p0", "label": 0.0},
    ]
    _write_pair(tmp_path, "game-a", tokens, spans, winner_id="p0")
    ds = ValueLabeledBinDataset(tmp_path, seq_len=128, eval_fraction=0.0)
    assert ds.n_spans == 3
    counts = ds.label_counts()
    assert counts == {"wins": 1, "losses": 1, "draws": 1}

    np_rng = np.random.default_rng(0)
    out_tokens, out_labels = ds.sample_batch(2, np_rng)
    assert out_tokens.shape == (2, 128)
    assert out_labels.shape == (2,)
    assert set(out_labels.tolist()).issubset({-1.0, 0.0, 1.0})


def test_value_dataset_pads_short_spans(tmp_path: Path) -> None:
    tokens = np.arange(64, dtype=np.int64)
    spans = [{"offset": 0, "length": 32, "perspective_id": "p0", "label": 1.0}]
    _write_pair(tmp_path, "g", tokens, spans, winner_id="p0")
    ds = ValueLabeledBinDataset(tmp_path, seq_len=64, pad_token_id=0, eval_fraction=0.0)
    np_rng = np.random.default_rng(0)
    t, _ = ds.sample_batch(1, np_rng)
    # First 32 are real tokens, rest are pad.
    assert (t[0, :32] == np.arange(32)).all()
    assert (t[0, 32:] == 0).all()


def test_value_dataset_eval_split_is_deterministic(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for i in range(40):
        tokens = rng.integers(1, 200, size=64, dtype=np.int64)
        spans = [{"offset": 0, "length": 64, "perspective_id": "p0", "label": 1.0}]
        _write_pair(tmp_path, f"g{i:02d}", tokens, spans, winner_id="p0")
    train = ValueLabeledBinDataset(tmp_path, seq_len=64, split="train", eval_fraction=0.25)
    eval_ = ValueLabeledBinDataset(tmp_path, seq_len=64, split="eval", eval_fraction=0.25)
    assert train.n_games + eval_.n_games == 40
    assert eval_.n_games > 0


def test_value_pretrain_trainer_smoke_trains_loss_down(tmp_path: Path) -> None:
    """A learnable signal: label = sign of mean(token_id - midpoint)."""

    torch.manual_seed(0)
    np_rng = np.random.default_rng(0)
    vocab_size = 64
    seq_len = 32

    # Build a corpus where the label is +1 when most tokens are in [40, 64)
    # and -1 when most are in [1, 24). Highly learnable.
    n_games = 16
    spans_per_game = 8
    for g in range(n_games):
        all_tokens = []
        spans = []
        cursor = 0
        for s in range(spans_per_game):
            label = 1.0 if (g + s) % 2 == 0 else -1.0
            if label > 0:
                tok = np_rng.integers(40, 64, size=seq_len)
            else:
                tok = np_rng.integers(1, 24, size=seq_len)
            all_tokens.append(tok)
            spans.append(
                {
                    "offset": int(cursor),
                    "length": int(seq_len),
                    "perspective_id": "p0",
                    "label": float(label),
                }
            )
            cursor += seq_len
        flat = np.concatenate(all_tokens).astype(np.int64)
        _write_pair(tmp_path, f"g{g:02d}", flat, spans, winner_id="p0")

    cfg_enc = TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_seq_len=seq_len,
        pad_id=0,
    )
    encoder = TextStateEncoder(cfg_enc)
    value_head = ValueHead(cfg_enc.d_model)
    cfg = ValuePretrainConfig(
        data_dir=tmp_path, seq_len=seq_len, batch_size=8, pad_token_id=0, eval_fraction=0.0
    )
    ds = ValueLabeledBinDataset(tmp_path, seq_len=seq_len, eval_fraction=0.0)
    trainer = ValuePretrainTrainer(encoder, value_head, cfg, lr=3e-3)

    losses: list[float] = []
    for _ in range(120):
        tokens_np, labels_np = ds.sample_batch(cfg.batch_size, np_rng)
        token_ids = torch.from_numpy(tokens_np).to(dtype=torch.long)
        labels = torch.from_numpy(labels_np).to(dtype=torch.float32)
        stats = trainer.step(token_ids, labels)
        losses.append(stats["loss"])

    first = sum(losses[:20]) / 20.0
    last = sum(losses[-20:]) / 20.0
    assert last < 0.75 * first, (
        f"value pretrain failed to converge: first={first:.4f} last={last:.4f}"
    )
