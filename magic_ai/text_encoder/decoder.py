"""Autoregressive grammar decoder.

Causal transformer that emits a typed action program one token at a time,
cross-attending to the bidirectional encoder context. Two heads:

- ``vocab_head``: distribution over the small fixed grammar vocabulary
  (``<ATTACK>``, ``<END>``, …). Decoder vocab is private — not tied to
  the encoder vocab.
- ``pointer_head``: bilinear dot-product attention back over encoder
  positions, used for ``OBJ_REF`` / ``ACTION_REF`` slots.

Self-attn uses RoPE on Q/K. Cross-attn keys/values are projected once
from the encoder context and reused across decoding steps (the cache
holds the projected K/V — the encoder hidden states themselves are
untransformed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE
except ImportError:
    # Peer subagent is creating grammar.py in parallel; fall back so the
    # decoder module + tests can run independently.
    GRAMMAR_VOCAB_SIZE = 26


@dataclass
class GrammarDecoderConfig:
    d_model: int = 384
    n_layers: int = 4
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.0
    max_decode_len: int = 64
    grammar_vocab_size: int = GRAMMAR_VOCAB_SIZE
    pointer_temperature: float = 1.0


def _build_rope_cache(
    seq_len: int, head_dim: int, *, base: float = 10000.0
) -> tuple[Tensor, Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D/2].
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos_b - x2 * sin_b
    out[..., half:] = x2 * cos_b + x1 * sin_b
    return out


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class _GeGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.gelu(gate) * up)


class _CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GrammarDecoderConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """Teacher-forced (uncached) self-attention over a full sequence."""
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos_t = cos[:t]
        sin_t = sin[:t]
        q = _apply_rope(q, cos_t, sin_t)
        k = _apply_rope(k, cos_t, sin_t)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=t > 1)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.proj(out)

    def step(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache_k: Tensor,
        cache_v: Tensor,
        step_idx: Tensor,
    ) -> Tensor:
        """Single-token cached step.

        ``cache_k`` / ``cache_v`` are pre-allocated ``[B, H, L, Dh]`` buffers
        owned by the caller; the slot at ``step_idx`` is filled in place each
        call. An attention mask masks out positions ``> step_idx``. Shapes
        stay constant across steps so dynamo doesn't recompile on cache
        growth.
        """
        b, _, _ = x.shape
        L = cache_k.shape[-2]
        qkv = self.qkv(x).view(b, 1, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, 1, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        idx_1d = step_idx.view(1)
        cos_t = cos.index_select(0, idx_1d)
        sin_t = sin.index_select(0, idx_1d)
        q = _apply_rope(q, cos_t, sin_t)
        k = _apply_rope(k, cos_t, sin_t)
        cache_k.index_copy_(2, idx_1d, k)
        cache_v.index_copy_(2, idx_1d, v)
        pos = torch.arange(L, device=q.device)
        attn_mask = (pos <= step_idx).view(1, 1, 1, L)
        out = F.scaled_dot_product_attention(q, cache_k, cache_v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(b, 1, self.n_heads * self.head_dim)
        return self.proj(out)


class _CrossAttention(nn.Module):
    def __init__(self, cfg: GrammarDecoderConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def project_kv(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        # encoded: [B, T_enc, D] → ([B, H, T_enc, Dh], [B, H, T_enc, Dh])
        b, t, _ = encoded.shape
        kv = self.kv_proj(encoded).view(b, t, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        # mask: [B, T_enc] bool, True at valid keys.
        attn_mask = encoder_attention_mask[:, None, None, :].to(dtype=torch.bool)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.proj(out)


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: GrammarDecoderConfig) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(cfg.d_model)
        self.self_attn = _CausalSelfAttention(cfg)
        self.norm2 = _RMSNorm(cfg.d_model)
        self.cross_attn = _CrossAttention(cfg)
        self.norm3 = _RMSNorm(cfg.d_model)
        self.ffn = _GeGLU(cfg.d_model, cfg.d_ff)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        x = x + self.self_attn(self.norm1(x), cos, sin)
        x = x + self.cross_attn(self.norm2(x), cross_k, cross_v, encoder_attention_mask)
        x = x + self.ffn(self.norm3(x))
        return x

    def step(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        encoder_attention_mask: Tensor,
        cache_k: Tensor,
        cache_v: Tensor,
        step_idx: Tensor,
    ) -> Tensor:
        x = x + self.self_attn.step(self.norm1(x), cos, sin, cache_k, cache_v, step_idx)
        x = x + self.cross_attn(self.norm2(x), cross_k, cross_v, encoder_attention_mask)
        x = x + self.ffn(self.norm3(x))
        return x


@dataclass
class DecoderState:
    """Per-layer KV cache plus once-projected cross-attn K/V.

    ``self_k`` / ``self_v`` are pre-allocated to the max decode length
    ``[B, H, L, Dh]``; each step writes one slot in place. ``step_idx``
    is a 0-D long tensor pointing at the next slot to fill. Fixed shapes
    keep the compiled step body from recompiling on cache growth.
    """

    self_k: list[Tensor] = field(default_factory=list)
    self_v: list[Tensor] = field(default_factory=list)
    cross_k: list[Tensor] = field(default_factory=list)
    cross_v: list[Tensor] = field(default_factory=list)
    step_idx: Tensor | None = None


class GrammarDecoder(nn.Module):
    def __init__(self, cfg: GrammarDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.grammar_vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([_DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = _RMSNorm(cfg.d_model)
        self.vocab_head = nn.Linear(cfg.d_model, cfg.grammar_vocab_size, bias=False)
        self.pointer_head = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = _build_rope_cache(cfg.max_decode_len, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _heads(self, h: Tensor, encoded: Tensor) -> tuple[Tensor, Tensor]:
        vocab_logits = self.vocab_head(h)
        q = self.pointer_head(h)
        pointer_logits = torch.matmul(q, encoded.transpose(-1, -2)) / self.cfg.pointer_temperature
        return vocab_logits, pointer_logits

    def _decoder_body(
        self, target_tokens: Tensor, encoded: Tensor, encoder_attention_mask: Tensor
    ) -> Tensor:
        """Shared transformer body for the teacher-forced forwards.

        Returns the post-norm hidden state ``[B, L, D]``. Pulled out so
        the dense-head and per-cell-pointer-head variants can share it
        without duplicating the layer loop.
        """
        x = self.tok_emb(target_tokens)
        _, t, _ = x.shape
        cos = cast(Tensor, self.rope_cos)[:t].to(dtype=x.dtype, device=x.device)
        sin = cast(Tensor, self.rope_sin)[:t].to(dtype=x.dtype, device=x.device)
        for raw_layer in self.layers:
            layer = cast(_DecoderLayer, raw_layer)
            cross_k, cross_v = layer.cross_attn.project_kv(encoded)
            x = layer(x, cos, sin, cross_k, cross_v, encoder_attention_mask)
        return self.final_norm(x)

    def forward_teacher_forced(
        self,
        target_tokens: Tensor,
        encoded: Tensor,
        encoder_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns ``(vocab_logits [B, L, V], pointer_logits [B, L, T_enc])``.

        Pointer logits cover ALL encoder positions; the caller is
        responsible for masking down to legal anchor positions.
        """
        h = self._decoder_body(target_tokens, encoded, encoder_attention_mask)
        return self._heads(h, encoded)

    def forward_teacher_forced_with_cells(
        self,
        target_tokens: Tensor,
        encoded: Tensor,
        encoder_attention_mask: Tensor,
        *,
        p_cell_b: Tensor,
        p_cell_t: Tensor,
        p_legal_cell_id: Tensor,
        p_legal_choice: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Same body as ``forward_teacher_forced``, but the pointer head
        runs per-legal-cell instead of dense `[B, L, T_enc]`.

        For pointer cells (``p_cell_b[i], p_cell_t[i]``) and their legal
        encoder positions (``p_legal_choice``), returns

        ``(vocab_logits [B, L, V], p_legal_logits [N_p_legal])``

        where ``p_legal_logits[i] = q[cell_b, cell_t] · encoded[cell_b,
        legal_choice] / pointer_temperature``. Avoids materializing the
        dense ``[B, L, T_enc]`` pointer-logits tensor, the main compute
        win of the packed-cell layout.

        ``vocab_logits`` stays dense (``V`` is small in the grammar
        decoder, so the per-cell variant wouldn't recoup the indexing
        overhead).
        """
        h = self._decoder_body(target_tokens, encoded, encoder_attention_mask)
        vocab_logits = self.vocab_head(h)
        if p_cell_b.numel() == 0:
            p_legal_logits = h.new_empty((0,))
            return vocab_logits, p_legal_logits
        h_at_p_cells = h[p_cell_b, p_cell_t]  # [N_p_cells, D]
        q_at_p_cells = self.pointer_head(h_at_p_cells)  # [N_p_cells, D]
        q_per_p_legal = q_at_p_cells[p_legal_cell_id]  # [N_p_legal, D]
        p_legal_b = p_cell_b[p_legal_cell_id]  # [N_p_legal]
        enc_at_p_legal = encoded[p_legal_b, p_legal_choice]  # [N_p_legal, D]
        p_legal_logits = (q_per_p_legal * enc_at_p_legal).sum(-1) / self.cfg.pointer_temperature
        return vocab_logits, p_legal_logits

    def init_state(self, encoded: Tensor, max_decode_len: int | None = None) -> DecoderState:
        """Pre-project cross-attn K/V and pre-allocate self-attn KV cache.

        Self-attn cache is shaped ``[B, H, L, Dh]`` once up front so the
        compiled step body sees constant tensor shapes — no recompile on
        cache growth. Each step writes one slot in place via
        ``index_copy_`` and masks out unfilled positions.
        """
        L = int(max_decode_len if max_decode_len is not None else self.cfg.max_decode_len)
        b = int(encoded.shape[0])
        head_dim = self.cfg.d_model // self.cfg.n_heads
        state = DecoderState(
            step_idx=torch.zeros((), dtype=torch.long, device=encoded.device),
        )
        for raw_layer in self.layers:
            layer = cast(_DecoderLayer, raw_layer)
            k, v = layer.cross_attn.project_kv(encoded)
            state.cross_k.append(k)
            state.cross_v.append(v)
            state.self_k.append(
                torch.zeros(
                    (b, self.cfg.n_heads, L, head_dim),
                    dtype=encoded.dtype,
                    device=encoded.device,
                )
            )
            state.self_v.append(
                torch.zeros(
                    (b, self.cfg.n_heads, L, head_dim),
                    dtype=encoded.dtype,
                    device=encoded.device,
                )
            )
        return state

    def step(
        self,
        prev_token: Tensor,
        prev_pointer_pos: Tensor,
        encoded: Tensor,
        encoder_attention_mask: Tensor,
        state: DecoderState,
    ) -> tuple[Tensor, Tensor, DecoderState]:
        """One autoregressive step. Caller is responsible for the host-sync
        that turns logits into a chosen token before the engine mask
        callback can consume the prefix.

        ``state`` must be created via :meth:`init_state` before the first
        step — this method does not handle a ``None`` state.

        ``prev_pointer_pos`` is currently unused — the decoder embeds the
        previously chosen *grammar token* (pointer steps embed the same
        token, with the actual choice carried by the cross-attn cache).
        Plumbed through so a later revision can fold pointer-position
        info into the input embedding without an API change.
        """
        del prev_pointer_pos
        x = self.tok_emb(prev_token).unsqueeze(1)  # [B, 1, D]
        cos = cast(Tensor, self.rope_cos).to(dtype=x.dtype, device=x.device)
        sin = cast(Tensor, self.rope_sin).to(dtype=x.dtype, device=x.device)
        step_idx = cast(Tensor, state.step_idx)
        for i, raw_layer in enumerate(self.layers):
            layer = cast(_DecoderLayer, raw_layer)
            x = layer.step(
                x,
                cos,
                sin,
                state.cross_k[i],
                state.cross_v[i],
                encoder_attention_mask,
                state.self_k[i],
                state.self_v[i],
                step_idx,
            )
        x = self.final_norm(x)
        vocab_logits, pointer_logits = self._heads(x, encoded)
        # Self-attn caches are mutated in place by `_CausalSelfAttention.step`;
        # only step_idx is replaced (a new 0-D tensor with the next position).
        new_state = DecoderState(
            self_k=state.self_k,
            self_v=state.self_v,
            cross_k=state.cross_k,
            cross_v=state.cross_v,
            step_idx=step_idx + 1,
        )
        return vocab_logits.squeeze(1), pointer_logits.squeeze(1), new_state


def combined_sample(
    vocab_logits: Tensor,
    pointer_logits: Tensor,
    vocab_mask: Tensor,
    pointer_mask: Tensor,
    is_pointer_step: Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
) -> tuple[Tensor, Tensor]:
    """Sample a (vocab, pointer) pair per batch row.

    Mixed batches are supported: pointer rows return ``-1`` in the vocab
    slot and vice versa. All work is on-device; no Python loops over the
    batch.
    """
    neg_inf = float("-inf")
    vocab_l = vocab_logits.masked_fill(~vocab_mask, neg_inf) / temperature
    pointer_l = pointer_logits.masked_fill(~pointer_mask, neg_inf) / temperature
    # Rows where no head is legal (e.g. vocab on a pointer step) get a uniform
    # zero-logit dummy distribution so Categorical.sample doesn't blow up; the
    # caller selects the right column via ``is_pointer_step``.
    vocab_dummy = (~vocab_mask.any(dim=-1, keepdim=True)).expand_as(vocab_l)
    pointer_dummy = (~pointer_mask.any(dim=-1, keepdim=True)).expand_as(pointer_l)
    vocab_l = torch.where(vocab_dummy, torch.zeros_like(vocab_l), vocab_l)
    pointer_l = torch.where(pointer_dummy, torch.zeros_like(pointer_l), pointer_l)
    if greedy:
        sampled_vocab = vocab_l.argmax(dim=-1)
        sampled_pointer = pointer_l.argmax(dim=-1)
    else:
        sampled_vocab = torch.distributions.Categorical(logits=vocab_l).sample()
        sampled_pointer = torch.distributions.Categorical(logits=pointer_l).sample()
    neg_one = torch.full_like(sampled_vocab, -1)
    out_vocab = torch.where(is_pointer_step, neg_one, sampled_vocab)
    out_pointer = torch.where(is_pointer_step, sampled_pointer, neg_one)
    return out_vocab, out_pointer


__all__ = [
    "GrammarDecoderConfig",
    "GrammarDecoder",
    "DecoderState",
    "combined_sample",
    "GRAMMAR_VOCAB_SIZE",
]
