"""Self-contained text-encoder policy facade.

Owns one ``TextStateEncoder``, one ``GrammarDecoder``, and the value head.
Exposes a ``forward`` that takes a :class:`TextEncodedBatch` (combined
``state || spec`` token stream) and returns the encoded context plus value;
plus ``forward_decoder_teacher_forced`` for teacher-forced decoder CE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.game_state import GameStateSnapshot
from magic_ai.text_encoder.batch import (
    PackedTextBatch,
    TextEncodedBatch,
    TokenizationContext,
    collate,
    pack_batch,
    tokenize_snapshots,
)
from magic_ai.text_encoder.decision_spec import DecisionSpec
from magic_ai.text_encoder.decoder import GrammarDecoder, GrammarDecoderConfig
from magic_ai.text_encoder.mlm import MLMHead
from magic_ai.text_encoder.model import (
    TextEncoderConfig,
    TextStateEncoder,
    ValueHead,
    gather_card_vectors_packed,
    gather_state_vector_packed,
    initialize_text_state_encoder_from_hf,
)
from magic_ai.text_encoder.render import OracleEntry, RenderedSnapshot, render_snapshot
from magic_ai.text_encoder.render_spec import DecisionSpecRenderer
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS


@dataclass
class EncodedSnapshots:
    """Pooled encoder outputs for a :class:`TextEncodedBatch`."""

    card_vectors: Tensor
    card_mask: Tensor
    state_vector: Tensor
    encoded: Tensor  # [T_packed, D] full hidden states (for decoder cross-attn)
    packed: PackedTextBatch  # the packed batch the encoder ran on


@dataclass
class TextPolicyOutput:
    """Bundle of tensors produced by :meth:`TextPolicy.forward`."""

    values: Tensor
    card_vectors: Tensor
    card_mask: Tensor
    state_vector: Tensor


class TextPolicy(nn.Module):
    """Encoder + grammar decoder + value head."""

    def __init__(
        self,
        cfg: TextEncoderConfig,
        *,
        decoder_cfg: GrammarDecoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TextStateEncoder(cfg)
        if cfg.hf_model_name is not None:
            initialize_text_state_encoder_from_hf(self.encoder, cfg)
        self.value_head = ValueHead(cfg.d_model)
        self.mlm_head = MLMHead(self.encoder)
        if decoder_cfg is None:
            # Mirror the encoder width/depth by default, with a compatible
            # head count for tiny test configs.
            default_heads = min(cfg.n_heads, GrammarDecoderConfig.n_heads)
            n_heads = next(
                (h for h in range(default_heads, 0, -1) if cfg.d_model % h == 0),
                1,
            )
            decoder_cfg = GrammarDecoderConfig(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=n_heads,
                d_ff=cfg.d_ff,
            )
        self.grammar_decoder = GrammarDecoder(decoder_cfg)

    def encode_only(
        self, batch: TextEncodedBatch, *, hist_emb: Tensor | None = None
    ) -> EncodedSnapshots:
        packed = pack_batch(batch)
        return self.encode_packed_only(packed, hist_emb=hist_emb)

    def encode_packed_only(
        self, batch: PackedTextBatch, *, hist_emb: Tensor | None = None
    ) -> EncodedSnapshots:
        hidden = self.encoder.forward_packed(batch, hist_emb=hist_emb)
        card_vecs, card_mask = gather_card_vectors_packed(hidden, batch)
        state_vec = gather_state_vector_packed(hidden, batch)
        return EncodedSnapshots(
            card_vectors=card_vecs,
            card_mask=card_mask,
            state_vector=state_vec,
            encoded=hidden,
            packed=batch,
        )

    def run_heads(self, encoded: EncodedSnapshots, state_vec: Tensor | None = None) -> Tensor:
        sv = encoded.state_vector if state_vec is None else state_vec
        return self.value_head(sv)

    def forward(
        self, batch: TextEncodedBatch, *, hist_emb: Tensor | None = None
    ) -> TextPolicyOutput:
        encoded = self.encode_only(batch, hist_emb=hist_emb)
        values = self.run_heads(encoded)
        return TextPolicyOutput(
            values=values,
            card_vectors=encoded.card_vectors,
            card_mask=encoded.card_mask,
            state_vector=encoded.state_vector,
        )

    def forward_decoder_teacher_forced(
        self,
        batch: TextEncodedBatch,
        target_tokens: Tensor,
        *,
        hist_emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run encoder + grammar decoder in teacher-forced mode.

        ``batch.token_ids`` is the combined ``[state, spec]`` stream produced
        by :func:`collate`. Returns
        ``(vocab_logits [B, L, V_grammar], pointer_logits [B, L, T_enc])``.

        The encoder runs through the packed ``flash_attn_varlen`` path to
        avoid per-shape ``flex_attention`` recompiles; we then scatter the
        packed hidden back to ``[B, T_enc, D]`` for the decoder cross-attn.
        Inference, replay scoring, and pretrain all funnel through this
        same convention so there's a single padded↔packed seam.
        """

        from magic_ai.text_encoder.batch import scatter_packed_to_padded

        encoded_snaps = self.encode_only(batch, hist_emb=hist_emb)
        encoded, attn_mask = scatter_packed_to_padded(encoded_snaps.encoded, encoded_snaps.packed)
        return self.grammar_decoder.forward_teacher_forced(target_tokens, encoded, attn_mask)

    @staticmethod
    def encode_snapshots(
        snapshots: Sequence[GameStateSnapshot],
        oracle: dict[str, OracleEntry] | None,
        tokenizer: PreTrainedTokenizerFast,
    ) -> TextEncodedBatch:
        """Render -> tokenize -> render-spec -> collate.

        Snapshots without a pending decision (or with a kind that is deferred
        post-v1) get a ``None`` spec and ``decision_type = -1``.
        """

        if len(snapshots) == 0:
            raise ValueError("encode_snapshots() requires at least one snapshot")

        rendered_list = _render_snapshots_with_token_ids(
            snapshots,
            oracle=oracle,
            tokenizer=tokenizer,
        )
        tokenize_ctx = TokenizationContext.from_tokenizer(tokenizer)
        examples = tokenize_snapshots(rendered_list, tokenizer, context=tokenize_ctx)
        spec_renderer = DecisionSpecRenderer(tokenizer)
        specs: list[DecisionSpec | None] = []
        for snap, rendered in zip(snapshots, rendered_list, strict=True):
            pending = snap.get("pending")
            if pending is None:
                specs.append(None)
                continue
            try:
                spec = spec_renderer.render(snap, card_refs=rendered.card_refs)
            except NotImplementedError, ValueError:
                spec = None
            specs.append(spec)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no pad_token_id; cannot collate")
        return collate(examples, specs, pad_id=int(pad_id))


def _resolve_single_token(tokenizer: PreTrainedTokenizerFast, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if isinstance(tid, list):
        raise TypeError(f"convert_tokens_to_ids({token!r}) returned a list")
    return int(tid)


def _render_snapshots_with_token_ids(
    snapshots: Sequence[GameStateSnapshot],
    *,
    oracle: dict[str, OracleEntry] | None,
    tokenizer: PreTrainedTokenizerFast,
) -> list[RenderedSnapshot]:
    """Render every snapshot with all special-token ids plumbed through."""

    if len(snapshots) == 0:
        raise ValueError("encode_snapshots() requires at least one snapshot")

    self_token_id = _resolve_single_token(tokenizer, "<self>")
    opp_token_id = _resolve_single_token(tokenizer, "<opp>")
    mana_token_ids = [
        _resolve_single_token(tokenizer, f"<mana:{s}>") for s in ("W", "U", "B", "R", "G", "C")
    ]
    card_ref_token_ids = [
        _resolve_single_token(tokenizer, f"<card-ref:{k}>") for k in range(MAX_CARD_REFS)
    ]

    rendered_list: list[RenderedSnapshot] = []
    for snap in snapshots:
        rendered_list.append(
            render_snapshot(
                snap,
                oracle=oracle,
                self_token_id=self_token_id,
                opp_token_id=opp_token_id,
                mana_token_ids=mana_token_ids,
                card_ref_token_ids=card_ref_token_ids,
            )
        )
    return rendered_list


def build_text_policy(
    tokenizer: PreTrainedTokenizerFast,
    cfg: TextEncoderConfig | None = None,
) -> TextPolicy:
    """Construct a :class:`TextPolicy` whose config matches ``tokenizer``."""

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad_token_id")
    vocab_size = len(tokenizer)
    if cfg is None:
        cfg = TextEncoderConfig(vocab_size=vocab_size, pad_id=int(pad_id))
    else:
        if cfg.vocab_size != vocab_size:
            raise ValueError(
                f"cfg.vocab_size ({cfg.vocab_size}) does not match len(tokenizer) ({vocab_size})"
            )
        if cfg.pad_id != int(pad_id):
            raise ValueError(
                f"cfg.pad_id ({cfg.pad_id}) does not match tokenizer pad id ({int(pad_id)})"
            )
    return TextPolicy(cfg)


def parameter_count(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


__all__ = [
    "EncodedSnapshots",
    "TextPolicy",
    "TextPolicyOutput",
    "build_text_policy",
    "parameter_count",
]
