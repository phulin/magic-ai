"""Self-contained text-encoder policy facade.

This module is the integration target for the eventual ``policy.encoder =
"text"`` branch in ``magic_ai/model.py``. It owns one ``TextStateEncoder`` and
the value / inline-blank heads, exposes a single ``forward`` that
takes a :class:`TextEncodedBatch` and returns every tensor downstream code is
likely to need, and provides a ``encode_snapshots`` convenience that runs the
render -> tokenize -> collate pipeline so callers do not have to import four
modules just to score a list of snapshots.

Out of scope: ``PPOPolicy``, ``RolloutBuffer``, native rollouts, RnaD. See
``docs/text_encoder_plan.md`` §11 for the planned integration path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.game_state import GameStateSnapshot, PendingOptionState
from magic_ai.text_encoder.batch import (
    PackedTextBatch,
    TextEncodedBatch,
    collate,
    pack_batch,
    tokenize_snapshot,
)
from magic_ai.text_encoder.decoder import GrammarDecoder, GrammarDecoderConfig
from magic_ai.text_encoder.mlm import MLMHead
from magic_ai.text_encoder.model import (
    InlineBlankPolicy,
    TextEncoderConfig,
    TextStateEncoder,
    ValueHead,
    gather_card_vectors_packed,
    gather_state_vector_packed,
    initialize_text_state_encoder_from_hf,
)
from magic_ai.text_encoder.render import OracleEntry, RenderedSnapshot, render_snapshot
from magic_ai.text_encoder.render_spec import DecisionSpecRenderer
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS, MAX_NUM


@dataclass
class EncodedSnapshots:
    """Pooled encoder outputs for a :class:`TextEncodedBatch`.

    Produced by :meth:`TextPolicy.encode_only` so that callers (notably the
    recurrent wrapper in ``magic_ai.text_encoder.recurrent``) can reuse the
    encoder + pools and drive the heads with a different state vector (e.g. an
    LSTM hidden state) without re-running the trunk.
    """

    card_vectors: Tensor
    card_mask: Tensor
    state_vector: Tensor
    blank_logits: Tensor | None = None


@dataclass
class TextPolicyOutput:
    """Bundle of tensors produced by :meth:`TextPolicy.forward`.

    Shapes (B = batch, K = ``MAX_CARD_REFS``, D = ``cfg.d_model``):

    * ``values``: ``[B]``.
    * ``card_vectors``: ``[B, K, D]``, zero rows where ``card_mask`` is False.
    * ``card_mask``: ``[B, K]`` bool.
    * ``state_vector``: ``[B, D]`` — pooled at position 0 (``<bos>``).
    * ``blank_logits``: ``[B, K_blank, V_max]``.
    """

    values: Tensor
    card_vectors: Tensor
    card_mask: Tensor
    state_vector: Tensor
    blank_logits: Tensor | None = None
    blank_positions: Tensor | None = None
    blank_kind: Tensor | None = None
    blank_group: Tensor | None = None
    blank_group_kind: Tensor | None = None
    blank_option_index: Tensor | None = None
    blank_legal_ids: Tensor | None = None
    blank_legal_mask: Tensor | None = None


class TextPolicy(nn.Module):
    """Encoder + three heads, callable on a :class:`TextEncodedBatch`."""

    def __init__(
        self,
        cfg: TextEncoderConfig,
        *,
        use_grammar_decoder: bool = False,
        decoder_cfg: GrammarDecoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TextStateEncoder(cfg)
        if cfg.hf_model_name is not None:
            initialize_text_state_encoder_from_hf(self.encoder, cfg)
        self.value_head = ValueHead(cfg.d_model)
        self.mlm_head = MLMHead(self.encoder)
        self.inline_blank_policy = InlineBlankPolicy(
            self.encoder.tok_emb,
            self.mlm_head.dense,
            self.mlm_head.layer_norm,
            num_kinds=cfg.vocab_size,
        )
        self.use_grammar_decoder = use_grammar_decoder
        self.grammar_decoder: GrammarDecoder | None = None
        if use_grammar_decoder:
            self.grammar_decoder = GrammarDecoder(
                decoder_cfg or GrammarDecoderConfig(d_model=cfg.d_model)
            )

    def encode_only(self, batch: TextEncodedBatch) -> EncodedSnapshots:
        """Run the encoder + pool ops; return raw vectors without head logits.

        Used by :class:`TextPolicy.forward` and by the recurrent wrapper so the
        latter can substitute an LSTM hidden state for ``state_vector`` before
        running the heads.
        """

        # Internally we run the packed (varlen) forward path: pack the
        # padded batch, run forward_packed, gather at the rebased anchor
        # positions. Output shapes match the dense path 1:1, so callers
        # don't change. Wins scale with how skewed the per-row sequence
        # lengths are; equal-length batches are no slower.
        packed = pack_batch(batch)
        return self.encode_packed_only(packed)

    def encode_packed_only(self, batch: PackedTextBatch) -> EncodedSnapshots:
        """Run the encoder + pool ops on an already-packed batch."""

        hidden = self.encoder.forward_packed(batch)
        card_vecs, card_mask = gather_card_vectors_packed(hidden, batch)
        state_vec = gather_state_vector_packed(hidden, batch)
        blank_logits = self.inline_blank_policy(
            hidden,
            batch.blank_positions,
            batch.blank_kind,
            batch.blank_legal_ids,
            batch.blank_legal_mask,
        )
        return EncodedSnapshots(
            card_vectors=card_vecs,
            card_mask=card_mask,
            state_vector=state_vec,
            blank_logits=blank_logits,
        )

    def encode_packed_replay_only(
        self,
        batch: PackedTextBatch,
        *,
        blank_row_mask: Tensor | None = None,
    ) -> EncodedSnapshots:
        """Run replay scoring encoder outputs without retaining card vectors."""

        with torch.profiler.record_function("encode_packed_replay_only/encoder_forward_packed"):
            hidden = self.encoder.forward_packed(batch)
        with torch.profiler.record_function("encode_packed_replay_only/gather_state"):
            state_vec = gather_state_vector_packed(hidden, batch)
        if blank_row_mask is None:
            with torch.profiler.record_function("encode_packed_replay_only/inline_blank_policy"):
                blank_logits = self.inline_blank_policy(
                    hidden,
                    batch.blank_positions,
                    batch.blank_kind,
                    batch.blank_legal_ids,
                    batch.blank_legal_mask,
                )
        else:
            with torch.profiler.record_function("encode_packed_replay_only/blank_mask_to_device"):
                blank_row_mask = blank_row_mask.to(device=hidden.device, dtype=torch.bool)
            with torch.profiler.record_function("encode_packed_replay_only/inline_blank_policy"):
                blank_logits = self.inline_blank_policy(
                    hidden,
                    batch.blank_positions,
                    batch.blank_kind,
                    batch.blank_legal_ids,
                    batch.blank_legal_mask,
                )
            with torch.profiler.record_function("encode_packed_replay_only/blank_row_mask"):
                blank_logits = blank_logits.masked_fill(
                    ~blank_row_mask[:, None, None],
                    float("-inf"),
                )
        return EncodedSnapshots(
            card_vectors=hidden.new_empty((int(batch.seq_lengths.shape[0]), 0, hidden.shape[-1])),
            card_mask=torch.empty(
                (int(batch.seq_lengths.shape[0]), 0),
                dtype=torch.bool,
                device=hidden.device,
            ),
            state_vector=state_vec,
            blank_logits=blank_logits,
        )

    def run_heads(self, encoded: EncodedSnapshots, state_vec: Tensor | None = None) -> Tensor:
        """Run the value head against ``encoded`` using ``state_vec``.

        If ``state_vec`` is ``None``, ``encoded.state_vector`` is used.
        """

        sv = encoded.state_vector if state_vec is None else state_vec
        return self.value_head(sv)

    def forward(self, batch: TextEncodedBatch) -> TextPolicyOutput:
        encoded = self.encode_only(batch)
        values = self.run_heads(encoded)
        return TextPolicyOutput(
            values=values,
            card_vectors=encoded.card_vectors,
            card_mask=encoded.card_mask,
            state_vector=encoded.state_vector,
            blank_logits=encoded.blank_logits,
            blank_positions=batch.blank_positions,
            blank_kind=batch.blank_kind,
            blank_group=batch.blank_group,
            blank_group_kind=batch.blank_group_kind,
            blank_option_index=batch.blank_option_index,
            blank_legal_ids=batch.blank_legal_ids,
            blank_legal_mask=batch.blank_legal_mask,
        )

    def forward_decoder_teacher_forced(
        self,
        batch: TextEncodedBatch,
        target_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run encoder + grammar decoder in teacher-forced mode.

        ``batch.token_ids`` should already be the combined ``[state, spec]``
        stream produced by :func:`collate_with_specs`. Returns
        ``(vocab_logits [B, L, V_grammar], pointer_logits [B, L, T_enc])``.
        """

        if self.grammar_decoder is None:
            raise RuntimeError("TextPolicy was constructed with use_grammar_decoder=False")
        encoded = self.encoder(batch)
        return self.grammar_decoder.forward_teacher_forced(
            target_tokens,
            encoded,
            batch.attention_mask.to(dtype=torch.bool),
        )

    @staticmethod
    def encode_snapshots_with_specs(
        snapshots: Sequence[GameStateSnapshot],
        actions_per_snapshot: Sequence[Sequence[PendingOptionState] | None] | None,
        oracle: dict[str, OracleEntry] | None,
        tokenizer: PreTrainedTokenizerFast,
    ) -> TextEncodedBatch:
        """Render -> tokenize -> render-spec -> collate with combined streams.

        Snapshots without a pending decision (or with a kind that is deferred
        post-v1) get a ``None`` spec and ``decision_type = -1``.
        """

        if len(snapshots) == 0:
            raise ValueError("encode_snapshots_with_specs() requires at least one snapshot")
        if actions_per_snapshot is not None and len(actions_per_snapshot) != len(snapshots):
            raise ValueError(
                "actions_per_snapshot length must match snapshots length "
                f"({len(actions_per_snapshot)} vs {len(snapshots)})"
            )
        from magic_ai.text_encoder.batch import collate_with_specs as _collate
        from magic_ai.text_encoder.batch import tokenize_snapshot as _tokenize
        from magic_ai.text_encoder.decision_spec import DecisionSpec

        rendered_list = _render_snapshots_with_token_ids(
            snapshots,
            actions_per_snapshot,
            oracle=oracle,
            tokenizer=tokenizer,
        )
        spec_renderer = DecisionSpecRenderer(tokenizer)
        examples = []
        specs: list[DecisionSpec | None] = []
        for snap, rendered in zip(snapshots, rendered_list, strict=True):
            examples.append(_tokenize(rendered, tokenizer))
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
        return _collate(examples, specs, pad_id=int(pad_id))

    @staticmethod
    def encode_snapshots(
        snapshots: Sequence[GameStateSnapshot],
        actions_per_snapshot: Sequence[Sequence[PendingOptionState] | None] | None,
        oracle: dict[str, OracleEntry] | None,
        tokenizer: PreTrainedTokenizerFast,
        *,
        chosen_token_id: int | None = None,
    ) -> TextEncodedBatch:
        """Render -> tokenize -> collate convenience (inline-blank path)."""

        rendered_list = _render_snapshots_with_token_ids(
            snapshots,
            actions_per_snapshot,
            oracle=oracle,
            tokenizer=tokenizer,
            chosen_token_id=chosen_token_id,
        )
        examples = [tokenize_snapshot(r, tokenizer) for r in rendered_list]
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no pad_token_id; cannot collate")
        return collate(examples, pad_id=int(pad_id))


def _resolve_single_token(tokenizer: PreTrainedTokenizerFast, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if isinstance(tid, list):
        raise TypeError(f"convert_tokens_to_ids({token!r}) returned a list")
    return int(tid)


def _render_snapshots_with_token_ids(
    snapshots: Sequence[GameStateSnapshot],
    actions_per_snapshot: Sequence[Sequence[PendingOptionState] | None] | None,
    *,
    oracle: dict[str, OracleEntry] | None,
    tokenizer: PreTrainedTokenizerFast,
    chosen_token_id: int | None = None,
) -> list[RenderedSnapshot]:
    """Render every snapshot with all special-token ids plumbed through.

    Shared by :meth:`TextPolicy.encode_snapshots` (inline-blank path) and
    :meth:`TextPolicy.encode_snapshots_with_specs` (decoder path) so both
    pipelines see identical state-text rendering. Resolves the token-id
    table once and reuses it across snapshots.
    """

    if len(snapshots) == 0:
        raise ValueError("encode_snapshots() requires at least one snapshot")
    if actions_per_snapshot is None:
        action_lists: list[Sequence[PendingOptionState] | None] = [None] * len(snapshots)
    else:
        if len(actions_per_snapshot) != len(snapshots):
            raise ValueError(
                "actions_per_snapshot length must match snapshots length "
                f"({len(actions_per_snapshot)} vs {len(snapshots)})"
            )
        action_lists = list(actions_per_snapshot)

    if chosen_token_id is None:
        chosen_token_id = _resolve_single_token(tokenizer, "<chosen>")
    none_token_id = _resolve_single_token(tokenizer, "<none>")
    yes_token_id = _resolve_single_token(tokenizer, "<yes>")
    no_token_id = _resolve_single_token(tokenizer, "<no>")
    mulligan_token_id = _resolve_single_token(tokenizer, "<mulligan>")
    keep_token_id = _resolve_single_token(tokenizer, "<keep>")
    self_token_id = _resolve_single_token(tokenizer, "<self>")
    opp_token_id = _resolve_single_token(tokenizer, "<opp>")
    num_token_ids = [_resolve_single_token(tokenizer, f"<num:{k}>") for k in range(MAX_NUM)]
    mana_token_ids = [
        _resolve_single_token(tokenizer, f"<mana:{s}>") for s in ("W", "U", "B", "R", "G", "C")
    ]
    card_ref_token_ids = [
        _resolve_single_token(tokenizer, f"<card-ref:{k}>") for k in range(MAX_CARD_REFS)
    ]

    rendered_list: list[RenderedSnapshot] = []
    for snap, actions in zip(snapshots, action_lists, strict=True):
        rendered_list.append(
            render_snapshot(
                snap,
                actions,
                oracle=oracle,
                chosen_token_id=chosen_token_id,
                none_token_id=none_token_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                mulligan_token_id=mulligan_token_id,
                keep_token_id=keep_token_id,
                self_token_id=self_token_id,
                opp_token_id=opp_token_id,
                num_token_ids=num_token_ids,
                mana_token_ids=mana_token_ids,
                card_ref_token_ids=card_ref_token_ids,
            )
        )
    return rendered_list


def build_text_policy(
    tokenizer: PreTrainedTokenizerFast,
    cfg: TextEncoderConfig | None = None,
) -> TextPolicy:
    """Construct a :class:`TextPolicy` whose config matches ``tokenizer``.

    If ``cfg`` is omitted, a default :class:`TextEncoderConfig` is built with
    ``vocab_size = len(tokenizer)`` and ``pad_id = tokenizer.pad_token_id``.
    If ``cfg`` is provided, ``vocab_size`` and ``pad_id`` are validated against
    the tokenizer to catch drift between the saved tokenizer artifact and the
    config used to instantiate the policy.
    """

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
    """Total number of parameters in ``module`` (trainable or not)."""

    return sum(int(p.numel()) for p in module.parameters())


__all__ = [
    "EncodedSnapshots",
    "TextPolicy",
    "TextPolicyOutput",
    "build_text_policy",
    "parameter_count",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke
    cfg = TextEncoderConfig(vocab_size=50569)
    policy = TextPolicy(cfg)
    n = parameter_count(policy)
    # Use plain string formatting; the harness disallows print elsewhere but
    # this is a deliberate user-facing CLI smoke for sizing the model.
    print(f"TextPolicy(vocab_size={cfg.vocab_size}) parameters: {n:,}")
