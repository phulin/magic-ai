"""Masked language model pretraining for the text encoder.

Consumes a directory of ``*.bin`` files containing raw uint16 token streams
delimited by ``<bos>`` / ``<eos>`` tokens, samples fixed-length spans, applies
BERT-style masking (15% of non-special positions; of those, 80% replaced with
``[MASK]``, 10% replaced with a random token, 10% left unchanged), and trains
the :class:`TextStateEncoder` trunk via a tied-embedding LM head.

Used as a warm-start before RL training: pretrain on a large rendered-text
corpus, save the encoder weights into a checkpoint that
``scripts/train.py --checkpoint`` can resume from for the PPO/R-NaD phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.model import TextEncoderConfig, TextStateEncoder
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


@dataclass
class MLMConfig:
    data_dir: Path
    seq_len: int
    batch_size: int
    mask_prob: float = 0.15
    mask_token_id: int = 0
    pad_token_id: int = 0
    vocab_size: int = 0
    # Token ids excluded from being chosen as masking targets (special tokens
    # like <bos>/<eos>/<pad>/[MASK]). Loss is still computed only on positions
    # selected by the masking distribution, which never selects these.
    special_token_ids: tuple[int, ...] = ()


class BinTokenDataset:
    """Memory-mapped uint16 token streams over a directory of ``*.bin`` files.

    Spans may straddle ``<bos>`` / ``<eos>`` boundaries — the encoder learns to
    treat them as in-stream separators, matching how BERT-style training packs
    documents back-to-back.

    Two iteration modes:

    * :meth:`sample` / :meth:`sample_batch` — random span sampling with
      replacement. Useful when you want a fixed step budget regardless of
      corpus size.
    * :meth:`iter_epoch` — yields each non-overlapping fixed-length span in
      the corpus exactly once (in shuffled order). The number of batches per
      epoch is :meth:`spans_per_epoch` // ``batch_size``.
    """

    def __init__(self, data_dir: Path, seq_len: int) -> None:
        self.seq_len = seq_len
        files = sorted(Path(data_dir).rglob("*.bin"))
        if not files:
            raise FileNotFoundError(f"no *.bin files under {data_dir}")
        self._arrays: list[np.ndarray] = []
        for path in files:
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            if arr.shape[0] < seq_len:
                continue
            self._arrays.append(arr)
        if not self._arrays:
            raise ValueError(f"no *.bin file under {data_dir} has >= seq_len={seq_len} tokens")
        self._lengths = np.array([a.shape[0] for a in self._arrays], dtype=np.int64)
        self._weights = self._lengths / self._lengths.sum()
        # (file_idx, start) pairs covering every non-overlapping span in the
        # corpus. Tail tokens beyond the last full span in a file are dropped.
        spans: list[tuple[int, int]] = []
        for i, arr in enumerate(self._arrays):
            n = arr.shape[0] // seq_len
            for k in range(n):
                spans.append((i, k * seq_len))
        self._spans = np.array(spans, dtype=np.int64)  # [N, 2]

    @property
    def total_tokens(self) -> int:
        return int(self._lengths.sum())

    def spans_per_epoch(self) -> int:
        return int(self._spans.shape[0])

    def _read_span(self, file_idx: int, start: int) -> np.ndarray:
        arr = self._arrays[file_idx]
        return np.asarray(arr[start : start + self.seq_len], dtype=np.int64)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        idx = int(rng.choice(len(self._arrays), p=self._weights))
        arr = self._arrays[idx]
        start = int(rng.integers(0, arr.shape[0] - self.seq_len + 1))
        return np.asarray(arr[start : start + self.seq_len], dtype=np.int64)

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for b in range(batch_size):
            out[b] = self.sample(rng)
        return out

    def iter_epoch(self, batch_size: int, rng: np.random.Generator, drop_last: bool = True):
        """Yield ``[batch_size, seq_len]`` arrays covering each span once.

        Spans are shuffled per epoch. If ``drop_last`` is True (default), the
        trailing partial batch is dropped to keep batch shape constant.
        """

        order = rng.permutation(self._spans.shape[0])
        n = order.shape[0]
        end = (n // batch_size) * batch_size if drop_last else n
        for off in range(0, end, batch_size):
            sel = order[off : off + batch_size]
            out = np.empty((sel.shape[0], self.seq_len), dtype=np.int64)
            for j, span_idx in enumerate(sel):
                fi, st = self._spans[int(span_idx)]
                out[j] = self._read_span(int(fi), int(st))
            yield out


def apply_mlm_mask(
    token_ids: Tensor,
    cfg: MLMConfig,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """BERT-style masking. Returns ``(masked_inputs, labels)``.

    ``labels`` is ``token_ids`` at masked positions and ``-100`` elsewhere
    (matching :func:`torch.nn.functional.cross_entropy`'s ignore-index
    convention).
    """

    device = token_ids.device
    labels = token_ids.clone()

    # Don't mask special tokens.
    not_special = torch.ones_like(token_ids, dtype=torch.bool)
    for sid in cfg.special_token_ids:
        not_special &= token_ids != sid

    probs = torch.rand(token_ids.shape, generator=rng, device=device)
    masked_positions = (probs < cfg.mask_prob) & not_special

    labels = torch.where(masked_positions, labels, torch.full_like(labels, -100))

    # Of masked positions: 80% -> [MASK], 10% -> random, 10% -> unchanged.
    sub = torch.rand(token_ids.shape, generator=rng, device=device)
    replace_with_mask = masked_positions & (sub < 0.8)
    replace_with_random = masked_positions & (sub >= 0.8) & (sub < 0.9)

    inputs = token_ids.clone()
    inputs = torch.where(replace_with_mask, torch.full_like(inputs, cfg.mask_token_id), inputs)
    if replace_with_random.any():
        random_tokens = torch.randint(
            low=0,
            high=cfg.vocab_size,
            size=token_ids.shape,
            generator=rng,
            device=device,
            dtype=token_ids.dtype,
        )
        inputs = torch.where(replace_with_random, random_tokens, inputs)
    return inputs, labels


def make_mlm_batch(
    inputs: Tensor,
    pad_id: int,
    device: torch.device,
) -> TextEncodedBatch:
    """Wrap ``[B, T]`` token ids in a :class:`TextEncodedBatch` for the encoder.

    Card-ref and blank anchors are unused under the MLM objective. The
    ``attention_mask`` is set from ``pad_id`` so the encoder ignores padded
    positions if any are present (random-span sampling never produces padded
    positions, but the field is required by :class:`TextEncodedBatch`).
    """

    b, t = inputs.shape
    attention_mask = (inputs != pad_id).to(torch.int64)
    seq_lengths = attention_mask.sum(dim=-1).to(torch.int64)
    return TextEncodedBatch(
        token_ids=inputs.to(torch.long),
        attention_mask=attention_mask,
        card_ref_positions=torch.full((b, MAX_CARD_REFS), -1, dtype=torch.int64, device=device),
        seq_lengths=seq_lengths,
    )


class MLMHead(nn.Module):
    """Tied-weight LM head: ``hidden -> vocab`` via the embedding transpose.

    Mirrors BERT's MLM head shape (dense + GELU + LayerNorm + tied embedding
    projection + per-vocab bias). The dense block lets the encoder devote its
    final hidden states to features rather than vocabulary projections.
    """

    def __init__(self, encoder: TextStateEncoder) -> None:
        super().__init__()
        d = encoder.cfg.d_model
        v = encoder.cfg.vocab_size
        self.dense = nn.Linear(d, d)
        self.layer_norm = nn.LayerNorm(d, eps=1e-12)
        self.bias = nn.Parameter(torch.zeros(v))
        self._embedding_weight_ref = encoder.tok_emb  # tied, not stored

    def forward(self, hidden: Tensor) -> Tensor:
        h = F.gelu(self.dense(hidden))
        h = self.layer_norm(h)
        return F.linear(h, self._embedding_weight_ref.weight, self.bias)


def initialize_mlm_head_from_hf(head: MLMHead, encoder_cfg: TextEncoderConfig) -> bool:
    """Warm-init ``head`` from the HF MLM head/decoder weights.

    Returns True if HF weights were copied, False if init was skipped (no HF
    name, or layer truncation requested — the trunk no longer matches the
    pretraining task, so the user opted to start the head fresh in that case).

    Copies ``head.dense.weight`` (+ bias if present in HF), ``head.norm.weight``
    (+ bias if present), and ``decoder.bias`` (truncated/padded to local vocab
    size). The decoder weight is tied to the embedding and is therefore covered
    by :func:`initialize_text_state_encoder_from_hf`.
    """

    from transformers import AutoModelForMaskedLM

    if encoder_cfg.hf_model_name is None:
        return False
    if encoder_cfg.hf_truncate_layers is not None:
        return False

    hf = AutoModelForMaskedLM.from_pretrained(
        encoder_cfg.hf_model_name,
        revision=encoder_cfg.hf_revision,
        trust_remote_code=encoder_cfg.hf_trust_remote_code,
    )
    hf.resize_token_embeddings(encoder_cfg.vocab_size, pad_to_multiple_of=None)
    sd = hf.state_dict()

    def _copy(dst: Tensor, src: Tensor) -> None:
        with torch.no_grad():
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    if "head.dense.weight" in sd:
        _copy(head.dense.weight, sd["head.dense.weight"])
    if "head.dense.bias" in sd:
        _copy(head.dense.bias, sd["head.dense.bias"])
    if "head.norm.weight" in sd:
        _copy(head.layer_norm.weight, sd["head.norm.weight"])
    if "head.norm.bias" in sd:
        _copy(head.layer_norm.bias, sd["head.norm.bias"])
    if "decoder.bias" in sd:
        n = min(int(head.bias.shape[0]), int(sd["decoder.bias"].shape[0]))
        with torch.no_grad():
            head.bias[:n].copy_(sd["decoder.bias"][:n].to(device=head.bias.device))
    return True


class MLMTrainer:
    """Encoder + tied LM head + AdamW; one step computes masked-token CE loss."""

    def __init__(
        self,
        encoder: TextStateEncoder,
        cfg: MLMConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        grad_clip: float | None = 1.0,
        encoder_cfg: TextEncoderConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.cfg = cfg
        self.head = MLMHead(encoder).to(next(encoder.parameters()).device)
        # Warm-init the LM head from HF when the encoder cfg is supplied (and
        # has an hf_model_name). Skipped when the encoder was truncated, since
        # the head was pretrained against the full-depth trunk.
        self.hf_head_initialized = False
        if encoder_cfg is not None:
            self.hf_head_initialized = initialize_mlm_head_from_hf(self.head, encoder_cfg)
        self.grad_clip = grad_clip
        params = list(encoder.parameters()) + [
            p for n, p in self.head.named_parameters() if "_embedding_weight_ref" not in n
        ]
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def step(
        self,
        token_ids: Tensor,
        torch_rng: torch.Generator,
    ) -> dict[str, float]:
        self.encoder.train()
        self.head.train()

        inputs, labels = apply_mlm_mask(token_ids, self.cfg, torch_rng)
        device = next(self.encoder.parameters()).device
        batch = make_mlm_batch(inputs, self.cfg.pad_token_id, device)
        hidden = self.encoder(batch)  # [B, T, D]
        logits = self.head(hidden)  # [B, T, V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

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

        with torch.no_grad():
            mask = labels != -100
            n_masked = int(mask.sum().item())
            if n_masked > 0:
                pred = logits.argmax(dim=-1)
                acc = float(((pred == labels) & mask).sum().item()) / n_masked
            else:
                acc = 0.0
        return {
            "loss": float(loss.detach().item()),
            "perplexity": float(math.exp(min(20.0, loss.detach().item()))),
            "grad_norm": float(grad_norm),
            "n_masked": float(n_masked),
            "accuracy": acc,
        }


__all__ = [
    "BinTokenDataset",
    "MLMConfig",
    "MLMHead",
    "MLMTrainer",
    "apply_mlm_mask",
    "initialize_mlm_head_from_hf",
    "make_mlm_batch",
]
