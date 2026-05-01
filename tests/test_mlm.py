"""Tests for ``magic_ai.text_encoder.mlm``.

Covers binary-file loading, masking distribution shape, and a smoke that the
encoder trunk actually trains under the masked-LM objective.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from magic_ai.text_encoder.mlm import (
    BinTokenDataset,
    MLMConfig,
    MLMTrainer,
    apply_mlm_mask,
)
from magic_ai.text_encoder.model import TextEncoderConfig, TextStateEncoder


def _write_bin(path: Path, tokens: np.ndarray) -> None:
    arr = tokens.astype(np.uint16, copy=False)
    arr.tofile(path)


def test_bin_token_dataset_samples_in_bounds(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, 200, size=4096, dtype=np.int64)
    _write_bin(tmp_path / "shard.bin", tokens)
    ds = BinTokenDataset(tmp_path, seq_len=128)
    sample = ds.sample(rng)
    assert sample.shape == (128,)
    assert sample.min() >= 0 and sample.max() < 200


def test_bin_token_dataset_skips_short_files(tmp_path: Path) -> None:
    _write_bin(tmp_path / "tiny.bin", np.arange(8, dtype=np.int64))
    _write_bin(tmp_path / "ok.bin", np.arange(2048, dtype=np.int64))
    ds = BinTokenDataset(tmp_path, seq_len=128)
    # Only the second file qualifies; sampling must always succeed.
    rng = np.random.default_rng(1)
    for _ in range(20):
        ds.sample(rng)


def test_apply_mlm_mask_respects_special_tokens() -> None:
    cfg = MLMConfig(
        data_dir=Path("/tmp"),
        seq_len=8,
        batch_size=1,
        mask_prob=1.0,  # try to mask everything
        mask_token_id=99,
        pad_token_id=0,
        vocab_size=200,
        special_token_ids=(0, 1, 2),
    )
    rng = torch.Generator()
    rng.manual_seed(0)
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    inputs, labels = apply_mlm_mask(tokens, cfg, rng)
    # Specials never become labels.
    for sid in cfg.special_token_ids:
        assert ((tokens == sid) & (labels != -100)).sum().item() == 0
    # Non-specials all selected (mask_prob=1.0).
    non_special = ~torch.isin(tokens, torch.tensor(cfg.special_token_ids))
    assert (labels[non_special] != -100).all()


def test_mlm_trainer_smoke_trains_loss_down() -> None:
    """One short MLM run on a small encoder; loss should drop noticeably."""

    torch.manual_seed(0)
    vocab_size = 64
    cfg_enc = TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_seq_len=64,
        pad_id=0,
    )
    encoder = TextStateEncoder(cfg_enc)
    mlm_cfg = MLMConfig(
        data_dir=Path("/tmp"),
        seq_len=32,
        batch_size=4,
        mask_prob=0.15,
        mask_token_id=63,
        pad_token_id=0,
        vocab_size=vocab_size,
        special_token_ids=(0, 63),
    )
    trainer = MLMTrainer(encoder, mlm_cfg, lr=3e-3)

    # Highly learnable signal: every position is a fixed token + small noise so
    # the MLM head trivially predicts the majority class. With only 200 steps
    # we should at least see a notable drop.
    np_rng = np.random.default_rng(0)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(0)

    losses: list[float] = []
    for _ in range(150):
        # Mostly token id 7, with random low-id noise.
        t = np.full((mlm_cfg.batch_size, mlm_cfg.seq_len), 7, dtype=np.int64)
        flip = np_rng.random(t.shape) < 0.1
        t[flip] = np_rng.integers(1, 60, size=int(flip.sum()))
        token_ids = torch.from_numpy(t).to(torch.long)
        stats = trainer.step(token_ids, torch_rng)
        losses.append(stats["loss"])

    first = sum(losses[:20]) / 20.0
    last = sum(losses[-20:]) / 20.0
    assert last < 0.75 * first, f"MLM loss did not decrease: first={first} last={last}"
