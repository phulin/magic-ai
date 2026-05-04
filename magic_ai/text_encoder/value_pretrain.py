"""Supervised value-head pretraining for the text encoder.

Data is produced by ``scripts/jsonl_games_to_bin.py --with-value-labels``.



Stage 3a of ``docs/text_encoder_value_pretrain_plan.md``: after MLM pretrain
warms the encoder, this module trains the encoder + value head to predict
the perspective player's terminal win/loss/draw outcome from a single
decision-point snapshot. It exists as a separate phase (not folded into
MLM or RL) so the artifact can be checkpointed and resumed independently.

Data layout (per game, in ``data/games_value_bin/``):

* ``<gameId>.bin`` — flat ``uint16`` token stream, byte-identical to
  ``data/games_bin/`` (each decision-point span wrapped in
  ``<bos>...<eos>``, concatenated back-to-back).
* ``<gameId>.json`` — small sidecar carrying ``winner_id``, ``players``,
  and a ``spans`` list of ``{offset, length, perspective_id, label}`` per
  decision point. The label is precomputed (perspective-signed scalar in
  ``{-1.0, 0.0, +1.0}``) so the loader does not have to reason about
  player ids; the raw fields are kept for forward-compat.

The trainer runs the encoder forward over a padded ``[B, T]`` token batch,
gathers ``hidden[:, 0, :]`` (the ``<bos>``/``<state>`` opener — the same
position the production value head reads), and minimises MSE against the
scalar label.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.mlm import make_mlm_batch
from magic_ai.text_encoder.model import (
    TextStateEncoder,
    ValueHead,
    gather_state_vector,
)


@dataclass
class ValuePretrainConfig:
    data_dir: Path
    seq_len: int
    batch_size: int
    pad_token_id: int = 0
    eval_fraction: float = 0.05  # held-out games for eval; 0 disables
    gamma: float = 1.0  # MC return-to-go discount; 1.0 = undiscounted terminal sign


@dataclass
class _Span:
    file_idx: int
    offset: int
    length: int
    sign: float  # +1 / -1 / 0 — terminal outcome sign for this span's perspective
    steps_to_end: int  # decision-point steps from this span to terminal


class ValueLabeledBinDataset:
    """Memory-mapped ``*.bin`` + parsed ``*.json`` corpus of labeled snapshots.

    Each span is a single decision point wrapped in ``<bos>...<eos>``. The
    label is the Monte-Carlo return-to-go from that decision: each game is
    treated as a rollout, so the perspective-signed terminal reward
    propagates back as ``γ^steps_to_end * sign``. With ``gamma=1.0`` this
    is just the terminal sign for every span in the trajectory; with
    ``gamma<1`` early-game spans get a softer label than late-game ones,
    matching ``magic_ai.ppo.gae_returns`` under no-bootstrap λ=1
    conventions. At sample time, the span is right-truncated to
    ``seq_len`` (preserving the ``<bos>`` prefix); spans shorter than
    ``seq_len`` are right-padded with ``pad_token_id``.

    Held-out eval shard is selected deterministically via
    ``hash(game_id) % bucket == 0`` so the same game always lands in the
    same split.
    """

    def __init__(
        self,
        data_dir: Path,
        seq_len: int,
        *,
        pad_token_id: int = 0,
        split: str = "train",
        eval_fraction: float = 0.05,
        gamma: float = 1.0,
    ) -> None:
        if split not in ("train", "eval", "all"):
            raise ValueError(f"split must be train/eval/all, got {split!r}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma!r}")
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.gamma = gamma
        bin_files = sorted(Path(data_dir).rglob("*.bin"))
        if not bin_files:
            raise FileNotFoundError(f"no *.bin files under {data_dir}")
        self._arrays: list[np.ndarray] = []
        self._spans: list[_Span] = []
        self._game_ids: list[str] = []
        n_skipped_no_json = 0
        n_skipped_split = 0
        bucket = 0 if eval_fraction <= 0 else max(2, int(round(1.0 / eval_fraction)))
        for path in bin_files:
            json_path = path.with_suffix(".json")
            if not json_path.exists():
                n_skipped_no_json += 1
                continue
            meta = json.loads(json_path.read_text())
            game_id = str(meta.get("game_id") or path.stem)
            if bucket > 0 and split != "all":
                in_eval = (hash(game_id) % bucket) == 0
                if split == "train" and in_eval:
                    n_skipped_split += 1
                    continue
                if split == "eval" and not in_eval:
                    n_skipped_split += 1
                    continue
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            file_idx = len(self._arrays)
            self._arrays.append(arr)
            self._game_ids.append(game_id)
            game_spans = meta.get("spans") or []
            n_dec = len(game_spans)
            for i, span in enumerate(game_spans):
                offset = int(span["offset"])
                length = int(span["length"])
                if length <= 0 or offset + length > arr.shape[0]:
                    continue
                sign = float(span["label"])  # game-level sign, undiscounted
                # Older sidecars (pre-gamma) lack steps_to_end; fall back to
                # span-index from end so γ=1 keeps producing identical labels
                # and γ<1 still gets a reasonable discount profile.
                steps_to_end = int(span.get("steps_to_end", n_dec - 1 - i))
                self._spans.append(_Span(file_idx, offset, length, sign, steps_to_end))
        if not self._spans:
            raise ValueError(
                f"no usable spans under {data_dir} (split={split}, "
                f"skipped_no_json={n_skipped_no_json}, skipped_split={n_skipped_split})"
            )
        signs = np.array([s.sign for s in self._spans], dtype=np.float32)
        steps = np.array([s.steps_to_end for s in self._spans], dtype=np.int64)
        self._labels = (signs * (self.gamma ** steps.astype(np.float64))).astype(np.float32)

    @property
    def n_spans(self) -> int:
        return len(self._spans)

    @property
    def n_games(self) -> int:
        return len(self._arrays)

    def label_counts(self) -> dict[str, int]:
        # Classify by the undiscounted sign so the W/L/D split stays
        # meaningful under γ<1 (where mid-game labels can shrink toward 0).
        signs = np.array([s.sign for s in self._spans], dtype=np.float32)
        wins = int((signs > 0.5).sum())
        losses = int((signs < -0.5).sum())
        draws = int(((signs >= -0.5) & (signs <= 0.5)).sum())
        return {"wins": wins, "losses": losses, "draws": draws}

    def _read_span(self, span: _Span) -> np.ndarray:
        arr = self._arrays[span.file_idx]
        n = min(span.length, self.seq_len)
        out = np.full((self.seq_len,), self.pad_token_id, dtype=np.int64)
        out[:n] = np.asarray(arr[span.offset : span.offset + n], dtype=np.int64)
        return out

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self._spans), size=batch_size)
        tokens = np.empty((batch_size, self.seq_len), dtype=np.int64)
        labels = np.empty((batch_size,), dtype=np.float32)
        for j, i in enumerate(idx):
            i_int = int(i)
            tokens[j] = self._read_span(self._spans[i_int])
            labels[j] = self._labels[i_int]
        return tokens, labels

    def iter_epoch(self, batch_size: int, rng: np.random.Generator, drop_last: bool = True):
        order = rng.permutation(len(self._spans))
        n = order.shape[0]
        end = (n // batch_size) * batch_size if drop_last else n
        for off in range(0, end, batch_size):
            sel = order[off : off + batch_size]
            tokens = np.empty((sel.shape[0], self.seq_len), dtype=np.int64)
            labels = np.empty((sel.shape[0],), dtype=np.float32)
            for j, i in enumerate(sel):
                i_int = int(i)
                tokens[j] = self._read_span(self._spans[i_int])
                labels[j] = self._labels[i_int]
            yield tokens, labels


class ValuePretrainTrainer:
    """Encoder + ``ValueHead`` (taken by reference) + AdamW + MSE step.

    ``value_head`` is the same ``nn.Module`` instance used by the live
    ``TextPolicy``; we train its parameters directly so a head refactor in
    ``model.py`` automatically applies to both phases.
    """

    def __init__(
        self,
        encoder: TextStateEncoder,
        value_head: ValueHead,
        cfg: ValuePretrainConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        grad_clip: float | None = 1.0,
    ) -> None:
        self.encoder = encoder
        self.value_head = value_head
        self.cfg = cfg
        self.grad_clip = grad_clip
        params = list(encoder.parameters()) + list(value_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def _forward(self, batch: TextEncodedBatch) -> Tensor:
        hidden = self.encoder(batch)
        state_vec = gather_state_vector(hidden, batch)
        return self.value_head(state_vec)

    def step(
        self,
        token_ids: Tensor,
        labels: Tensor,
        *,
        compute_stats: bool = True,
    ) -> dict[str, float]:
        """Run a single optimizer step.

        ``compute_stats=False`` skips all `.item()` conversions on the
        return path so the caller can avoid ~7 D2H syncs per step on
        iterations where the stats won't be logged.
        """
        self.encoder.train()
        self.value_head.train()
        device = next(self.encoder.parameters()).device
        batch = make_mlm_batch(token_ids.to(device), self.cfg.pad_token_id, device)
        preds = self._forward(batch)
        loss = F.mse_loss(preds, labels.to(device=device, dtype=preds.dtype))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                max_norm=self.grad_clip,
            )
        else:
            grad_norm = torch.tensor(0.0)
        self.optimizer.step()

        if not compute_stats:
            return {}

        labels_dev = labels.detach().to(device=device, dtype=preds.dtype)
        with torch.no_grad():
            decisive = labels_dev.abs() > 0.5
            decisive_count = decisive.sum().clamp_min(1)
            sign_match = (torch.sign(preds.detach()) == torch.sign(labels_dev)) & decisive
            sign_acc_t = sign_match.float().sum() / decisive_count.float()
        loss_v = float(loss.detach().item())
        return {
            "loss": loss_v,
            "rmse": math.sqrt(max(0.0, loss_v)),
            "grad_norm": float(grad_norm),
            "pred_mean": float(preds.detach().mean().item()),
            "pred_abs_mean": float(preds.detach().abs().mean().item()),
            "label_mean": float(labels_dev.mean().item()),
            "sign_accuracy": float(sign_acc_t.item()),
        }

    @torch.no_grad()
    def evaluate(
        self, dataset: ValueLabeledBinDataset, np_rng: np.random.Generator, max_batches: int = 32
    ) -> dict[str, float]:
        self.encoder.eval()
        self.value_head.eval()
        device = next(self.encoder.parameters()).device
        total_se = 0.0
        n = 0
        n_decisive = 0
        n_correct = 0
        for i, (tokens_np, labels_np) in enumerate(
            dataset.iter_epoch(self.cfg.batch_size, np_rng, drop_last=True)
        ):
            if i >= max_batches:
                break
            tokens = torch.from_numpy(tokens_np).to(device=device, dtype=torch.long)
            labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)
            batch = make_mlm_batch(tokens, self.cfg.pad_token_id, device)
            preds = self._forward(batch)
            total_se += float(((preds - labels) ** 2).sum().item())
            n += int(preds.shape[0])
            decisive = labels.abs() > 0.5
            if decisive.any():
                n_decisive += int(decisive.sum().item())
                n_correct += int(
                    (torch.sign(preds[decisive]) == torch.sign(labels[decisive])).sum().item()
                )
        if n == 0:
            return {"eval_loss": 0.0, "eval_sign_accuracy": 0.0, "eval_n": 0}
        return {
            "eval_loss": total_se / n,
            "eval_sign_accuracy": (n_correct / n_decisive) if n_decisive > 0 else 0.0,
            "eval_n": float(n),
        }


__all__ = [
    "ValueLabeledBinDataset",
    "ValuePretrainConfig",
    "ValuePretrainTrainer",
]
