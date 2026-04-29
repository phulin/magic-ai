"""Text-encoder model: ModernBERT-shaped scratch trunk + pooling + heads.

Implements PR #3 from ``docs/text_encoder_plan.md``: a bidirectional
transformer encoder that consumes pre-tokenized state/action text and
produces (a) per-card / per-option / per-target / global state vectors via
position-anchored gather pools, and (b) policy / target / value head logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch

# flex_attention only generates the fused FA-style kernel under torch.compile;
# called eagerly it materializes the full [B, H, T, T] score tensor and emits
# a UserWarning. Compiling the *whole* encoder trunk turned out to have a
# bigger per-call dispatch cost than compiling these ops individually
# (332ms/call vs 281ms/call in profiling), so we keep the per-op compile and
# accept the multiple dispatch boundaries.
_flex_attention = torch.compile(flex_attention, dynamic=True)


def _create_block_mask(*args: Any, **kwargs: Any) -> BlockMask:
    return create_block_mask(*args, _compile=True, **kwargs)


@dataclass
class TextEncoderConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 2048
    dropout: float = 0.0
    pad_id: int = 0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _build_rope_cache(
    seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [T, half]
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def _key_pad_mask_mod(key_pad: Tensor):
    """Build a flex_attention mask_mod for a per-batch key-padding mask.

    ``key_pad`` is shape ``[B, T]`` and is True at masked-out (padded) key
    positions. Returns a closure that captures it; flex_attention compiles
    the closure into the FA-style kernel.
    """

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        del h, q_idx
        return ~key_pad[b, kv_idx]

    return mask_mod


def _document_mask_mod(seq_id: Tensor):
    """flex_attention mask_mod for sequence packing.

    ``seq_id`` is [T_packed] int. The kernel allows attention only when
    ``seq_id[q] == seq_id[kv]``, giving a block-diagonal allowed region.
    """

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        del b, h
        return seq_id[q_idx] == seq_id[kv_idx]

    return mask_mod


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    rx1 = x1 * cos_b - x2 * sin_b
    rx2 = x2 * cos_b + x1 * sin_b
    return torch.cat((rx1, rx2), dim=-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: Tensor,
        block_mask: BlockMask,
        attn_bias: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, D]
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # Three attention paths:
        #   (1) packed varlen on CUDA via NJT + SDPA → FlashAttention varlen
        #       kernel, no flex_attention compile.
        #   (2) padded CUDA via flex_attention's BlockMask (compiled).
        #   (3) CPU additive-bias SDPA (flex_attention has no CPU backward
        #       in torch 2.11).
        if cu_seqlens is not None:
            # q/k/v are [1, H, T_packed, Dh]; strip the B=1, wrap as NJT
            # over the doc-length offsets, run SDPA on [B_docs, H, j, Dh],
            # then unwrap back to packed [T_packed, H*Dh].
            q_p = q.squeeze(0).transpose(0, 1).contiguous()  # [T, H, Dh]
            k_p = k.squeeze(0).transpose(0, 1).contiguous()
            v_p = v.squeeze(0).transpose(0, 1).contiguous()
            offsets = cu_seqlens.to(torch.int64)
            q_nt = torch.nested.nested_tensor_from_jagged(q_p, offsets).transpose(1, 2)
            k_nt = torch.nested.nested_tensor_from_jagged(k_p, offsets).transpose(1, 2)
            v_nt = torch.nested.nested_tensor_from_jagged(v_p, offsets).transpose(1, 2)
            out_nt = F.scaled_dot_product_attention(
                q_nt, k_nt, v_nt, dropout_p=self.dropout if self.training else 0.0
            )
            out_p = out_nt.transpose(1, 2).values()  # [T, H, Dh]
            out = out_p.reshape(1, t, self.n_heads * self.head_dim)
            return self.proj(out)
        if x.device.type == "cuda":
            result = _flex_attention(q, k, v, block_mask=block_mask)
            out = result if isinstance(result, Tensor) else result[0]
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=self.dropout if self.training else 0.0
            )
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.proj(out)


class GeGLUFFN(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.gelu(gate) * up)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = GeGLUFFN(cfg)

    def forward(
        self,
        x: Tensor,
        block_mask: BlockMask,
        attn_bias: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), block_mask, attn_bias, cos, sin, cu_seqlens)
        x = x + self.ffn(self.norm2(x))
        return x


class TextStateEncoder(nn.Module):
    # v1: all-global; alternating local/global is a follow-up.
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, batch: TextEncodedBatch) -> Tensor:
        token_ids = batch.token_ids
        attention_mask = batch.attention_mask
        b, t = token_ids.shape
        # nn.Embedding requires Long; the replay buffer stores int32 to halve
        # token-storage memory and the live assembler may emit int32 too.
        # ``.to(torch.long)`` is a no-op when the tensor is already Long.
        x = self.tok_emb(token_ids.to(torch.long))
        key_pad = ~attention_mask.bool()
        # Guard rows whose entire key axis is masked: softmax over an empty
        # set is NaN, which would poison the whole batch. Such rows produce
        # meaningless hidden states either way; un-mask them here so
        # attention reduces to a finite (if useless) average. Downstream
        # code only consumes hidden states at valid positions anyway.
        all_masked = key_pad.all(dim=-1, keepdim=True)
        key_pad = key_pad & ~all_masked
        # Two mask formulations — flex_attention's BlockMask for CUDA (FA
        # kernel) and the additive-bias SDPA fallback for CPU. Building both
        # is cheap; only the one for the active device is consumed.
        if x.device.type == "cuda":
            block_mask = _create_block_mask(
                _key_pad_mask_mod(key_pad),
                B=b,
                H=None,
                Q_LEN=t,
                KV_LEN=t,
                device=x.device,
            )
            attn_bias = x.new_zeros(())
        else:
            block_mask = cast(BlockMask, None)
            attn_bias = torch.zeros(b, 1, 1, t, device=x.device, dtype=x.dtype)
            attn_bias = attn_bias.masked_fill(key_pad[:, None, None, :], float("-inf"))
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _build_rope_cache(t, head_dim, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, block_mask, attn_bias, cos, sin)
        x = self.final_norm(x)
        return x

    def forward_packed(self, batch: PackedTextBatch) -> Tensor:
        """Run the encoder over a packed (varlen) batch.

        Returns hidden states of shape ``[T_packed, D]``. Documents are
        kept independent via a flex_attention mask_mod that compares
        ``seq_id[q]`` to ``seq_id[kv]``; RoPE positions are reset per
        document by gathering the precomputed sin/cos table at
        ``pos_in_seq``. No Python loop over documents.
        """

        token_ids = batch.token_ids.to(torch.long)  # [T]
        t = int(token_ids.shape[0])
        x = self.tok_emb(token_ids).unsqueeze(0)  # [1, T, D]
        seq_id = batch.seq_id.to(device=x.device, dtype=torch.long)
        pos = batch.pos_in_seq.to(device=x.device, dtype=torch.long)

        head_dim = self.cfg.d_model // self.cfg.n_heads
        # Cache size has to cover the largest in-doc position; the
        # configured ``max_seq_len`` is the model's hard upper bound,
        # and any individual document is ``<= max_seq_len`` by
        # construction.
        cache_len = self.cfg.max_seq_len
        cos_full, sin_full = _build_rope_cache(cache_len, head_dim, x.device, x.dtype)
        # [T, D/2] — same shape the dense path feeds into ``_apply_rope``,
        # so the existing rope kernel works unchanged.
        cos = cos_full.index_select(0, pos)
        sin = sin_full.index_select(0, pos)

        # Packed varlen attention via NJT + SDPA on CUDA (FlashAttention
        # varlen) — avoids the flex_attention create_block_mask compile.
        # CPU keeps the additive-mask SDPA fallback used by the tests.
        if x.device.type == "cuda":
            block_mask = cast(BlockMask, None)
            attn_bias = x.new_zeros(())
            cu_seqlens: Tensor | None = batch.cu_seqlens.to(device=x.device, dtype=torch.int64)
        else:
            block_mask = cast(BlockMask, None)
            same = seq_id[:, None] == seq_id[None, :]  # [T, T] bool
            attn_bias = torch.zeros(1, 1, t, t, device=x.device, dtype=x.dtype)
            attn_bias = attn_bias.masked_fill(~same[None, None, :, :], float("-inf"))
            cu_seqlens = None

        for block in self.blocks:
            x = block(x, block_mask, attn_bias, cos, sin, cu_seqlens)
        x = self.final_norm(x)
        return x.squeeze(0)  # [T, D]


def _gather_at(hidden: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Gather hidden states at integer positions; -1 entries become zeros.

    hidden: [B, T, D]; positions: [B, ...] int64. Returns (gathered, mask)
    where ``gathered`` has shape ``[B, ..., D]`` and ``mask`` has shape
    ``[B, ...]`` (bool, True where present).
    """

    b, t, d = hidden.shape
    mask = positions >= 0
    safe = positions.clamp(min=0)
    flat = safe.reshape(b, -1)  # [B, K]
    # ``torch.gather`` requires Long indices; the replay buffer stores
    # positions as int32 to halve their footprint.
    idx = flat.unsqueeze(-1).expand(-1, -1, d).to(torch.long)  # [B, K, D]
    gathered = torch.gather(hidden, 1, idx)  # [B, K, D]
    gathered = gathered.reshape(*positions.shape, d)
    gathered = gathered * mask.unsqueeze(-1).to(gathered.dtype)
    return gathered, mask


def gather_card_vectors(hidden: Tensor, batch: TextEncodedBatch) -> tuple[Tensor, Tensor]:
    return _gather_at(hidden, batch.card_ref_positions)


def gather_option_vectors(hidden: Tensor, batch: TextEncodedBatch) -> tuple[Tensor, Tensor]:
    vecs, _ = _gather_at(hidden, batch.option_positions)
    return vecs, batch.option_mask


def gather_target_vectors(hidden: Tensor, batch: TextEncodedBatch) -> tuple[Tensor, Tensor]:
    vecs, _ = _gather_at(hidden, batch.target_positions)
    return vecs, batch.target_mask


def gather_state_vector(hidden: Tensor, batch: TextEncodedBatch) -> Tensor:
    del batch  # position 0 is always the <bos>/<state> opener.
    return hidden[:, 0, :]


def _gather_packed(hidden: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Gather from a packed ``[T_packed, D]`` hidden at integer offsets.

    ``positions`` is any int tensor whose values are absolute offsets
    into the packed row, with ``-1`` for absent slots.
    """

    mask = positions >= 0
    safe = positions.clamp(min=0).to(torch.long)
    flat = safe.reshape(-1)
    gathered = hidden.index_select(0, flat).reshape(*positions.shape, hidden.shape[-1])
    gathered = gathered * mask.unsqueeze(-1).to(gathered.dtype)
    return gathered, mask


def gather_card_vectors_packed(hidden: Tensor, batch: PackedTextBatch) -> tuple[Tensor, Tensor]:
    return _gather_packed(hidden, batch.card_ref_positions)


def gather_option_vectors_packed(hidden: Tensor, batch: PackedTextBatch) -> tuple[Tensor, Tensor]:
    vecs, _ = _gather_packed(hidden, batch.option_positions)
    return vecs, batch.option_mask


def gather_target_vectors_packed(hidden: Tensor, batch: PackedTextBatch) -> tuple[Tensor, Tensor]:
    vecs, _ = _gather_packed(hidden, batch.target_positions)
    return vecs, batch.target_mask


def gather_state_vector_packed(hidden: Tensor, batch: PackedTextBatch) -> Tensor:
    idx = batch.state_positions.to(torch.long)
    return hidden.index_select(0, idx)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class PolicyHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = _MLP(2 * d_model, d_model, 1)

    def forward(
        self,
        option_vecs: Tensor,  # [B, O, D]
        state_vec: Tensor,  # [B, D]
        option_mask: Tensor,  # [B, O] bool
    ) -> Tensor:
        b, o, _ = option_vecs.shape
        state_b = state_vec.unsqueeze(1).expand(b, o, -1)
        x = torch.cat([option_vecs, state_b], dim=-1)
        logits = self.mlp(x).squeeze(-1)  # [B, O]
        neg_inf = torch.full_like(logits, float("-inf"))
        return torch.where(option_mask, logits, neg_inf)


class TargetHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = _MLP(3 * d_model, d_model, 1)

    def forward(
        self,
        target_vecs: Tensor,  # [B, O, M, D]
        option_vecs: Tensor,  # [B, O, D]
        state_vec: Tensor,  # [B, D]
        target_mask: Tensor,  # [B, O, M] bool
    ) -> Tensor:
        b, o, m, _ = target_vecs.shape
        opt_b = option_vecs.unsqueeze(2).expand(b, o, m, -1)
        state_b = state_vec[:, None, None, :].expand(b, o, m, -1)
        x = torch.cat([target_vecs, opt_b, state_b], dim=-1)
        logits = self.mlp(x).squeeze(-1)  # [B, O, M]
        neg_inf = torch.full_like(logits, float("-inf"))
        return torch.where(target_mask, logits, neg_inf)


class ValueHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = _MLP(d_model, d_model, 1)

    def forward(self, state_vec: Tensor) -> Tensor:
        return self.mlp(state_vec).squeeze(-1)


__all__ = [
    "TextEncoderConfig",
    "TextStateEncoder",
    "PolicyHead",
    "TargetHead",
    "ValueHead",
    "gather_card_vectors",
    "gather_option_vectors",
    "gather_target_vectors",
    "gather_state_vector",
    "gather_card_vectors_packed",
    "gather_option_vectors_packed",
    "gather_target_vectors_packed",
    "gather_state_vector_packed",
]
