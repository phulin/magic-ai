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
from transformers import AutoConfig, AutoModelForMaskedLM

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
    hf_model_name: str | None = None
    hf_revision: str | None = None
    hf_truncate_layers: int | None = None
    hf_trust_remote_code: bool = False


DEFAULT_HF_ENCODER_MODEL = "jhu-clsp/ettin-encoder-17m"


def text_encoder_config_from_hf(
    *,
    model_name: str = DEFAULT_HF_ENCODER_MODEL,
    vocab_size: int,
    pad_id: int,
    revision: str | None = None,
    truncate_layers: int | None = None,
    max_seq_len: int | None = None,
    dropout: float = 0.0,
    trust_remote_code: bool = False,
) -> TextEncoderConfig:
    """Build a :class:`TextEncoderConfig` from a Hugging Face encoder config.

    The loaded HF checkpoint remains the source of truth for hidden size,
    number of heads, feed-forward width, and layer count. ``truncate_layers``
    optionally keeps only the first N transformer layers at model construction
    time; when omitted, the full checkpoint depth is used.
    """

    hf_cfg = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    hidden_size = int(getattr(hf_cfg, "hidden_size"))
    num_heads = int(getattr(hf_cfg, "num_attention_heads"))
    full_layers = int(getattr(hf_cfg, "num_hidden_layers"))
    n_layers = full_layers if truncate_layers is None else int(truncate_layers)
    if n_layers < 1:
        raise ValueError("truncate_layers must be at least 1")
    if n_layers > full_layers:
        raise ValueError(
            f"truncate_layers ({n_layers}) exceeds checkpoint layer count ({full_layers})"
        )
    intermediate = int(getattr(hf_cfg, "intermediate_size", hidden_size * 4))
    if max_seq_len is None:
        max_seq_len = int(getattr(hf_cfg, "max_position_embeddings", 2048))
    return TextEncoderConfig(
        vocab_size=vocab_size,
        d_model=hidden_size,
        n_layers=n_layers,
        n_heads=num_heads,
        d_ff=intermediate,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pad_id=pad_id,
        hf_model_name=model_name,
        hf_revision=revision,
        hf_truncate_layers=truncate_layers,
        hf_trust_remote_code=trust_remote_code,
    )


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
    # Write the two rotated halves directly into a single output buffer to
    # skip the [B, H, T, D] cat allocation that runs on every block × forward.
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


def _apply_rope_packed(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [T, H, D]; cos/sin: [T, D/2]
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos_b - x2 * sin_b
    out[..., half:] = x2 * cos_b + x1 * sin_b
    return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.max_seq_len = cfg.max_seq_len
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
        max_seqlen: int | None = None,
    ) -> Tensor:
        # Two layouts share this module:
        #   (1) packed varlen [T, D] when cu_seqlens is not None — flash_attn varlen
        #       kernel; q/k/v stay as packed [T, H, Dh], no NJT overhead.
        #   (2) padded [B, T, D] otherwise — flex_attention on CUDA,
        #       additive-bias SDPA on CPU.
        if cu_seqlens is not None:
            return self._forward_packed(x, cu_seqlens, cos, sin, max_seqlen=max_seqlen)
        return self._forward_dense(x, block_mask, attn_bias, cos, sin)

    def _forward_packed(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        max_seqlen: int | None = None,
    ) -> Tensor:
        # x: [T, D]; cos/sin: [T, Dh/2]
        # flash_attn_varlen_func takes packed [T, H, Dh] q/k/v directly — no NJT
        # construction, no Python subclass dispatch, one kernel launch per block.
        from flash_attn import flash_attn_varlen_func

        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each [T, D]
        q = _apply_rope_packed(q.unflatten(-1, [self.n_heads, self.head_dim]), cos, sin)
        k = _apply_rope_packed(k.unflatten(-1, [self.n_heads, self.head_dim]), cos, sin)
        v = v.unflatten(-1, [self.n_heads, self.head_dim])  # [T, H, Dh]
        # flash_attn requires fp16/bf16; training runs under autocast (already bf16),
        # but direct calls (tests, eval) may arrive as fp32.
        src_dtype = q.dtype
        if src_dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        cu = cu_seqlens.to(torch.int32)
        eff_max_seqlen = self.max_seq_len if max_seqlen is None else int(max_seqlen)
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu,
            cu,
            eff_max_seqlen,
            eff_max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [T, H, Dh]
        return self.proj(out.flatten(1).to(src_dtype))

    def _forward_dense(
        self,
        x: Tensor,
        block_mask: BlockMask,
        attn_bias: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, D]
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
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
        max_seqlen: int | None = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), block_mask, attn_bias, cos, sin, cu_seqlens, max_seqlen)
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
        # RoPE table is fixed for a given (max_seq_len, head_dim); rebuilding
        # it every forward showed up in profiling. Stored float32 and cast to
        # the activation dtype at use, so .to(dtype=...) on the module body
        # doesn't desync this from the activations.
        head_dim = cfg.d_model // cfg.n_heads
        cos_full, sin_full = _build_rope_cache(
            cfg.max_seq_len, head_dim, torch.device("cpu"), torch.float32
        )
        self.register_buffer("rope_cos", cos_full, persistent=False)
        self.register_buffer("rope_sin", sin_full, persistent=False)
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
        x = self.tok_emb(token_ids)
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
        rope_cos = cast(Tensor, self.rope_cos)
        rope_sin = cast(Tensor, self.rope_sin)
        cos = rope_cos[:t].to(x.dtype)
        sin = rope_sin[:t].to(x.dtype)
        for block in self.blocks:
            x = block(x, block_mask, attn_bias, cos, sin)
        x = self.final_norm(x)
        return x

    def forward_packed(self, batch: PackedTextBatch) -> Tensor:
        """Run the encoder over a packed (varlen) batch.

        Returns hidden states of shape ``[T_packed, D]``. Documents are
        kept independent via the cu_seqlens-aware NJT path on CUDA; the
        CPU fallback uses an additive block-diagonal mask. RoPE positions
        are reset per document by gathering the cached sin/cos table at
        ``pos_in_seq``.
        """

        token_ids = batch.token_ids  # [T]
        t = token_ids.shape[0]
        x = self.tok_emb(token_ids)  # [T, D] — packed path stays rank-2
        pos = batch.pos_in_seq.to(device=x.device)
        rope_cos = cast(Tensor, self.rope_cos)
        rope_sin = cast(Tensor, self.rope_sin)
        cos = rope_cos.index_select(0, pos).to(x.dtype)  # [T, Dh/2]
        sin = rope_sin.index_select(0, pos).to(x.dtype)

        if x.device.type == "cuda":
            block_mask = cast(BlockMask, None)
            attn_bias = x.new_zeros(())
            cu_seqlens: Tensor | None = batch.cu_seqlens.to(device=x.device, dtype=torch.int32)
            max_seqlen = batch.max_seqlen
            for block in self.blocks:
                x = block(x, block_mask, attn_bias, cos, sin, cu_seqlens, max_seqlen)
            x = self.final_norm(x)
            return x

        # CPU fallback: flex_attention has no CPU backward in current torch,
        # so run dense SDPA with a block-diagonal additive mask. We add a
        # leading B=1 here just for the dense attention path's expectations.
        seq_id = batch.seq_id.to(device=x.device)
        same = seq_id[:, None] == seq_id[None, :]  # [T, T] bool
        attn_bias = torch.zeros(1, 1, t, t, device=x.device, dtype=x.dtype)
        attn_bias = attn_bias.masked_fill(~same[None, None, :, :], float("-inf"))
        block_mask = cast(BlockMask, None)
        x_dense = x.unsqueeze(0)  # [1, T, D]
        for block in self.blocks:
            x_dense = block(x_dense, block_mask, attn_bias, cos, sin, None)
        x_dense = self.final_norm(x_dense)
        return x_dense.squeeze(0)


def _copy_matching_param(dst: Tensor, src: Tensor, *, name: str) -> None:
    if dst.shape != src.shape:
        raise ValueError(
            f"{name} shape mismatch: local {tuple(dst.shape)} vs HF {tuple(src.shape)}"
        )
    with torch.no_grad():
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _copy_embedding_param(dst: Tensor, src: Tensor) -> None:
    rows = min(dst.shape[0], src.shape[0])
    if dst.shape[1:] != src.shape[1:]:
        raise ValueError(
            f"embedding shape mismatch: local {tuple(dst.shape)} vs HF {tuple(src.shape)}"
        )
    with torch.no_grad():
        dst[:rows].copy_(src[:rows].to(device=dst.device, dtype=dst.dtype))


def initialize_text_state_encoder_from_hf(
    encoder: TextStateEncoder, cfg: TextEncoderConfig
) -> None:
    """Warm-initialize ``encoder`` from an HF ModernBERT/Ettin-style checkpoint.

    The HF model is only used as a checkpoint reader. Compatible tensors are
    copied into the local PyTorch implementation, then the temporary HF module
    goes out of scope; inference continues through :class:`TextStateEncoder`.
    """

    if cfg.hf_model_name is None:
        raise ValueError("hf_model_name is required for HF initialization")
    # Load the ForMaskedLM wrapper (rather than AutoModel) so the head/decoder
    # weights count as expected when the checkpoint includes them — otherwise
    # transformers prints a LOAD REPORT flagging head.dense, head.norm, and
    # decoder.* as unexpected, which is misleading. We only copy trunk weights
    # here; MLM head init is handled separately in
    # :func:`magic_ai.text_encoder.mlm.initialize_mlm_head_from_hf`.
    hf_wrapper = AutoModelForMaskedLM.from_pretrained(
        cfg.hf_model_name,
        revision=cfg.hf_revision,
        trust_remote_code=cfg.hf_trust_remote_code,
    )
    hf_wrapper.resize_token_embeddings(cfg.vocab_size, pad_to_multiple_of=None)
    hf_model = hf_wrapper.model
    hidden_size = int(getattr(hf_model.config, "hidden_size"))
    if hidden_size != cfg.d_model:
        raise ValueError(
            f"HF model hidden_size ({hidden_size}) does not match cfg.d_model ({cfg.d_model})"
        )
    sd = hf_model.state_dict()
    _copy_embedding_param(encoder.tok_emb.weight, sd["embeddings.tok_embeddings.weight"])
    first_block = cast(EncoderBlock, encoder.blocks[0])
    if "embeddings.norm.weight" in sd:
        _copy_matching_param(
            first_block.norm1.weight,
            sd["embeddings.norm.weight"],
            name="blocks.0.norm1.weight",
        )
    for i, raw_block in enumerate(encoder.blocks):
        block = cast(EncoderBlock, raw_block)
        prefix = f"layers.{i}"
        attn_norm = sd.get(f"{prefix}.attn_norm.weight")
        if attn_norm is not None:
            _copy_matching_param(block.norm1.weight, attn_norm, name=f"blocks.{i}.norm1.weight")
        _copy_matching_param(
            block.attn.qkv.weight,
            sd[f"{prefix}.attn.Wqkv.weight"],
            name=f"blocks.{i}.attn.qkv.weight",
        )
        _copy_matching_param(
            block.attn.proj.weight,
            sd[f"{prefix}.attn.Wo.weight"],
            name=f"blocks.{i}.attn.proj.weight",
        )
        _copy_matching_param(
            block.norm2.weight,
            sd[f"{prefix}.mlp_norm.weight"],
            name=f"blocks.{i}.norm2.weight",
        )
        _copy_matching_param(
            block.ffn.gate_up.weight,
            sd[f"{prefix}.mlp.Wi.weight"],
            name=f"blocks.{i}.ffn.gate_up.weight",
        )
        _copy_matching_param(
            block.ffn.down.weight,
            sd[f"{prefix}.mlp.Wo.weight"],
            name=f"blocks.{i}.ffn.down.weight",
        )
    _copy_matching_param(
        encoder.final_norm.weight, sd["final_norm.weight"], name="final_norm.weight"
    )


def _gather_at(hidden: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Gather hidden states at integer positions; -1 entries become zeros.

    hidden: [B, T, D]; positions: [B, ...] int. Returns (gathered, mask)
    where ``gathered`` has shape ``[B, ..., D]`` and ``mask`` has shape
    ``[B, ...]`` (bool, True where present).
    """

    b, t, d = hidden.shape
    mask = positions >= 0
    safe = positions.clamp(min=0)
    flat = safe.reshape(b, -1)  # [B, K]
    idx = flat.unsqueeze(-1).expand(-1, -1, d)  # [B, K, D]
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
    safe = positions.clamp(min=0)
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
    return hidden.index_select(0, batch.state_positions)


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
        self.d_model = d_model
        self.mlp = _MLP(2 * d_model, d_model, 1)

    def forward(
        self,
        option_vecs: Tensor,  # [B, O, D]
        state_vec: Tensor,  # [B, D]
        option_mask: Tensor,  # [B, O] bool
    ) -> Tensor:
        # Math-equivalent to ``fc1(cat([option_vecs, state_b], -1))`` where
        # state_b is state_vec broadcast over O. Splitting fc1 across the two
        # input chunks avoids the [B, O, 2D] cat allocation and the per-call
        # broadcast of state through the GEMM.
        d = self.d_model
        w = self.mlp.fc1.weight  # [D, 2D]
        b1 = self.mlp.fc1.bias  # [D]
        h = F.linear(option_vecs, w[:, :d]) + F.linear(state_vec, w[:, d:], b1).unsqueeze(1)
        logits = self.mlp.fc2(F.gelu(h)).squeeze(-1)  # [B, O]
        neg_inf = torch.full_like(logits, float("-inf"))
        return torch.where(option_mask, logits, neg_inf)


class TargetHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = _MLP(3 * d_model, d_model, 1)

    def forward(
        self,
        target_vecs: Tensor,  # [B, O, M, D]
        option_vecs: Tensor,  # [B, O, D]
        state_vec: Tensor,  # [B, D]
        target_mask: Tensor,  # [B, O, M] bool
    ) -> Tensor:
        # Same trick as PolicyHead, three-way split. Avoids the [B, O, M, 3D]
        # cat allocation and the broadcasts of option/state through the GEMM.
        d = self.d_model
        w = self.mlp.fc1.weight  # [D, 3D]
        b1 = self.mlp.fc1.bias  # [D]
        h_tgt = F.linear(target_vecs, w[:, :d])  # [B, O, M, D]
        h_opt = F.linear(option_vecs, w[:, d : 2 * d]).unsqueeze(2)  # [B, O, 1, D]
        h_state = F.linear(state_vec, w[:, 2 * d :], b1)[:, None, None, :]  # [B, 1, 1, D]
        logits = self.mlp.fc2(F.gelu(h_tgt + h_opt + h_state)).squeeze(-1)
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
    "DEFAULT_HF_ENCODER_MODEL",
    "gather_card_vectors",
    "gather_option_vectors",
    "gather_target_vectors",
    "gather_state_vector",
    "gather_card_vectors_packed",
    "gather_option_vectors_packed",
    "gather_target_vectors_packed",
    "gather_state_vector_packed",
    "initialize_text_state_encoder_from_hf",
    "text_encoder_config_from_hf",
]
