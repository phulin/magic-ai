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
    TextEncodedBatch,
    collate,
    tokenize_snapshot,
)
from magic_ai.text_encoder.model import (
    PolicyHead,
    TargetHead,
    TextEncoderConfig,
    TextStateEncoder,
    ValueHead,
    gather_card_vectors,
    gather_option_vectors,
    gather_state_vector,
    gather_target_vectors,
)
from magic_ai.text_encoder.render import OracleEntry, render_snapshot


@dataclass
class TextPolicyOutput:
    """Bundle of tensors produced by :meth:`TextPolicy.forward`.

    Shapes (B = batch, O = max options, M = max targets, K = ``MAX_CARD_REFS``,
    D = ``cfg.d_model``):

    * ``policy_logits``: ``[B, O]``, ``-inf`` at masked-out option slots.
    * ``target_logits``: ``[B, O, M]``, ``-inf`` at masked-out target slots.
    * ``values``: ``[B]``.
    * ``card_vectors``: ``[B, K, D]``, zero rows where ``card_mask`` is False.
    * ``card_mask``: ``[B, K]`` bool.
    * ``option_vectors``: ``[B, O, D]``.
    * ``option_mask``: ``[B, O]`` bool.
    * ``target_vectors``: ``[B, O, M, D]``.
    * ``target_mask``: ``[B, O, M]`` bool.
    * ``state_vector``: ``[B, D]`` — pooled at position 0 (``<bos>``).
    """

    policy_logits: Tensor
    target_logits: Tensor
    values: Tensor
    card_vectors: Tensor
    card_mask: Tensor
    option_vectors: Tensor
    option_mask: Tensor
    target_vectors: Tensor
    target_mask: Tensor
    state_vector: Tensor


class TextPolicy(nn.Module):
    """Encoder + three heads, callable on a :class:`TextEncodedBatch`."""

    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TextStateEncoder(cfg)
        self.policy_head = PolicyHead(cfg.d_model)
        self.target_head = TargetHead(cfg.d_model)
        self.value_head = ValueHead(cfg.d_model)

    def forward(self, batch: TextEncodedBatch) -> TextPolicyOutput:
        hidden = self.encoder(batch)

        card_vecs, card_mask = gather_card_vectors(hidden, batch)
        option_vecs, option_mask = gather_option_vectors(hidden, batch)
        target_vecs, target_mask = gather_target_vectors(hidden, batch)
        state_vec = gather_state_vector(hidden, batch)

        policy_logits = self.policy_head(option_vecs, state_vec, option_mask)
        target_logits = self.target_head(target_vecs, option_vecs, state_vec, target_mask)
        values = self.value_head(state_vec)

        return TextPolicyOutput(
            policy_logits=policy_logits,
            target_logits=target_logits,
            values=values,
            card_vectors=card_vecs,
            card_mask=card_mask,
            option_vectors=option_vecs,
            option_mask=option_mask,
            target_vectors=target_vecs,
            target_mask=target_mask,
            state_vector=state_vec,
        )

    @staticmethod
    def encode_snapshots(
        snapshots: Sequence[GameStateSnapshot],
        actions_per_snapshot: Sequence[Sequence[PendingOptionState] | None] | None,
        oracle: dict[str, OracleEntry] | None,
        tokenizer: PreTrainedTokenizerFast,
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
        for snap, actions in zip(snapshots, action_lists, strict=True):
            rendered = render_snapshot(snap, actions, oracle=oracle)
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
