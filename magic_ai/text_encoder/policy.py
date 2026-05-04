"""Self-contained text-encoder policy facade.

This module is the integration target for the eventual ``policy.encoder =
"text"`` branch in ``magic_ai/model.py``. It owns one ``TextStateEncoder`` and
the three heads (policy / target / value), exposes a single ``forward`` that
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
from magic_ai.text_encoder.mlm import MLMHead
from magic_ai.text_encoder.model import (
    InlineBlankPolicy,
    TextEncoderConfig,
    TextStateEncoder,
    ValueHead,
    gather_card_vectors_packed,
    gather_option_vectors_packed,
    gather_state_vector_packed,
    gather_target_vectors_packed,
    initialize_text_state_encoder_from_hf,
)
from magic_ai.text_encoder.render import OracleEntry, render_snapshot
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
    option_vectors: Tensor
    option_mask: Tensor
    target_vectors: Tensor
    target_mask: Tensor
    state_vector: Tensor
    blank_logits: Tensor | None = None


@dataclass
class TextPolicyOutput:
    """Bundle of tensors produced by :meth:`TextPolicy.forward`.

    Shapes (B = batch, O = max options, M = max targets, K = ``MAX_CARD_REFS``,
    D = ``cfg.d_model``):

    * ``values``: ``[B]``.
    * ``card_vectors``: ``[B, K, D]``, zero rows where ``card_mask`` is False.
    * ``card_mask``: ``[B, K]`` bool.
    * ``option_vectors``: ``[B, O, D]``.
    * ``option_mask``: ``[B, O]`` bool.
    * ``target_vectors``: ``[B, O, M, D]``.
    * ``target_mask``: ``[B, O, M]`` bool.
    * ``state_vector``: ``[B, D]`` — pooled at position 0 (``<bos>``).
    * ``blank_logits``: ``[B, K, V_max]`` when inline blanks are enabled,
      otherwise ``None``.
    """

    values: Tensor
    card_vectors: Tensor
    card_mask: Tensor
    option_vectors: Tensor
    option_mask: Tensor
    target_vectors: Tensor
    target_mask: Tensor
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

    def __init__(self, cfg: TextEncoderConfig) -> None:
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
        option_vecs, option_mask = gather_option_vectors_packed(hidden, batch)
        target_vecs, target_mask = gather_target_vectors_packed(hidden, batch)
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
            option_vectors=option_vecs,
            option_mask=option_mask,
            target_vectors=target_vecs,
            target_mask=target_mask,
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
            option_vectors=encoded.option_vectors,
            option_mask=encoded.option_mask,
            target_vectors=encoded.target_vectors,
            target_mask=encoded.target_mask,
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

    @staticmethod
    def encode_snapshots(
        snapshots: Sequence[GameStateSnapshot],
        actions_per_snapshot: Sequence[Sequence[PendingOptionState] | None] | None,
        oracle: dict[str, OracleEntry] | None,
        tokenizer: PreTrainedTokenizerFast,
        *,
        chosen_token_id: int | None = None,
    ) -> TextEncodedBatch:
        """Render -> tokenize -> collate convenience.

        ``actions_per_snapshot`` may be ``None`` (in which case each snapshot's
        own ``pending.options`` is used) or a sequence aligned with
        ``snapshots`` whose entries can each be ``None`` to defer to the
        snapshot's pending options.
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

        examples = []
        none_token_id: int | None = None
        yes_token_id: int | None = None
        no_token_id: int | None = None
        num_token_ids: list[int] | None = None
        mana_token_ids: list[int] | None = None
        card_ref_token_ids: list[int] | None = None
        if chosen_token_id is None:
            tid = tokenizer.convert_tokens_to_ids("<chosen>")
            if isinstance(tid, list):
                raise TypeError("convert_tokens_to_ids('<chosen>') returned a list")
            chosen_token_id = int(tid)
        none_tid = tokenizer.convert_tokens_to_ids("<none>")
        if isinstance(none_tid, list):
            raise TypeError("convert_tokens_to_ids('<none>') returned a list")
        none_token_id = int(none_tid)
        yes_tid = tokenizer.convert_tokens_to_ids("<yes>")
        if isinstance(yes_tid, list):
            raise TypeError("convert_tokens_to_ids('<yes>') returned a list")
        yes_token_id = int(yes_tid)
        no_tid = tokenizer.convert_tokens_to_ids("<no>")
        if isinstance(no_tid, list):
            raise TypeError("convert_tokens_to_ids('<no>') returned a list")
        no_token_id = int(no_tid)
        num_token_ids = []
        for k in range(MAX_NUM):
            tid = tokenizer.convert_tokens_to_ids(f"<num:{k}>")
            if isinstance(tid, list):
                raise TypeError(f"convert_tokens_to_ids('<num:{k}>') returned a list")
            num_token_ids.append(int(tid))
        mana_token_ids = []
        for symbol in ("W", "U", "B", "R", "G", "C"):
            tid = tokenizer.convert_tokens_to_ids(f"<mana:{symbol}>")
            if isinstance(tid, list):
                raise TypeError(f"convert_tokens_to_ids('<mana:{symbol}>') returned a list")
            mana_token_ids.append(int(tid))
        card_ref_token_ids = []
        for k in range(MAX_CARD_REFS):
            tid = tokenizer.convert_tokens_to_ids(f"<card-ref:{k}>")
            if isinstance(tid, list):
                raise TypeError(f"convert_tokens_to_ids('<card-ref:{k}>') returned a list")
            card_ref_token_ids.append(int(tid))
        for snap, actions in zip(snapshots, action_lists, strict=True):
            rendered = render_snapshot(
                snap,
                actions,
                oracle=oracle,
                use_inline_blanks=True,
                chosen_token_id=chosen_token_id,
                none_token_id=none_token_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                num_token_ids=num_token_ids,
                mana_token_ids=mana_token_ids,
                card_ref_token_ids=card_ref_token_ids,
            )
            examples.append(tokenize_snapshot(rendered, tokenizer))

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no pad_token_id; cannot collate")
        return collate(examples, pad_id=int(pad_id))


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
