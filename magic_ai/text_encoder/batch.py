"""Bridge from rendered snapshots to padded token-id batches.

PR #3/#4 of the text-encoder plan in ``docs/text_encoder_plan.md`` (§5). The
renderer (``magic_ai.text_encoder.render``) emits a string plus anchor
metadata at the *string* level. The model wants per-example token-id tensors
plus *token-position* anchors so it can gather hidden states at every
``<card-ref:K>``, ``<option>``, and ``<target>`` location.

This module owns the tokenize-and-collate path:

* :func:`tokenize_snapshot` turns a :class:`RenderedSnapshot` into a per-example
  :class:`TextEncodedExample`. Anchor positions are recovered by scanning the
  output ``token_ids`` for the relevant special-token ids — robust against
  any whitespace/offset-mapping quirks the renderer may introduce.
* :func:`collate` right-pads a sequence of examples into a
  :class:`TextEncodedBatch` of int64 tensors with ``-1`` sentinels for absent
  positions and matching boolean masks.

Nothing here touches the model, the rollout buffer, or RnaD — those are
separate PRs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.render import RenderedSnapshot
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS

_CARD_REF_RE = re.compile(r"^<card-ref:(\d+)>$")


@dataclass
class TextEncodedExample:
    """Single rendered snapshot tokenized into ids + anchor positions."""

    token_ids: list[int]
    attention_mask: list[int]
    card_ref_positions: dict[int, int]
    option_positions: list[int]
    target_positions: list[list[int]]
    card_ref_engine_ids: dict[int, str] = field(default_factory=dict)


@dataclass
class TextEncodedBatch:
    """Padded batch of :class:`TextEncodedExample` ready for the encoder."""

    token_ids: Tensor  # [B, T] int64, padded with pad_id
    attention_mask: Tensor  # [B, T] int64
    card_ref_positions: Tensor  # [B, MAX_CARD_REFS] int64, -1 = absent
    option_positions: Tensor  # [B, max_opts] int64, -1 = absent
    option_mask: Tensor  # [B, max_opts] bool
    target_positions: Tensor  # [B, max_opts, max_targets] int64, -1 = absent
    target_mask: Tensor  # [B, max_opts, max_targets] bool
    seq_lengths: Tensor  # [B] int64


# ---------------------------------------------------------------------------
# Tokenize a single rendered snapshot
# ---------------------------------------------------------------------------


def tokenize_snapshot(
    rendered: RenderedSnapshot,
    tokenizer: PreTrainedTokenizerFast,
) -> TextEncodedExample:
    """Tokenize ``rendered.text`` and recover anchor token positions.

    The renderer already emits ``<bos>`` / ``<eos>`` (and every other custom
    token) as single added tokens, so we pass ``add_special_tokens=False`` to
    avoid the tokenizer adding any of its own.

    Anchor positions are recovered by id-equality on the produced ``input_ids``
    rather than via offset arithmetic. This is robust against any whitespace
    quirks in the rendered string and matches the recommendation in §5 of the
    plan.
    """

    encoding = tokenizer(
        rendered.text,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    token_ids: list[int] = list(encoding["input_ids"])
    attention_mask: list[int] = list(encoding["attention_mask"])

    def _single_id(token: str) -> int:
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, list):
            raise TypeError(f"convert_tokens_to_ids({token!r}) returned a list")
        return int(tid)

    option_token_id = _single_id("<option>")
    target_token_id = _single_id("<target>")

    # ----- card-ref positions -------------------------------------------------
    # Map every <card-ref:K> token id to its K. We look up via the tokenizer's
    # added-tokens map so we don't have to allocate 64 individual lookups.
    card_ref_id_to_k: dict[int, int] = {}
    unk = tokenizer.unk_token_id
    for k in range(MAX_CARD_REFS):
        tid = _single_id(f"<card-ref:{k}>")
        if tid == unk:
            continue
        card_ref_id_to_k[tid] = k

    card_ref_positions: dict[int, int] = {}
    for pos, tid in enumerate(token_ids):
        k = card_ref_id_to_k.get(tid)
        if k is None:
            continue
        # First occurrence wins; the renderer emits each <card-ref:K> exactly
        # once at the card binding site, but a target referencing the same K
        # would also re-emit the token. The binding site comes first in the
        # rendered order, so first-occurrence is the right pick.
        if k not in card_ref_positions:
            card_ref_positions[k] = pos

    # Cross-check against parsing the token strings so a typo in a future
    # rename of the token format surfaces as a test failure rather than silent
    # mismatch.
    for pos, tid in enumerate(token_ids):
        token_str = tokenizer.convert_ids_to_tokens(int(tid))
        if not isinstance(token_str, str):
            continue
        match = _CARD_REF_RE.match(token_str)
        if match is None:
            continue
        k_parsed = int(match.group(1))
        # Make sure the lookup-table path agrees with the string-parse path.
        assert card_ref_id_to_k.get(int(tid)) == k_parsed, (
            f"card-ref id mismatch at pos={pos}: id={tid} parsed={k_parsed}"
        )

    # ----- option / target positions -----------------------------------------
    option_positions: list[int] = [
        pos for pos, tid in enumerate(token_ids) if tid == option_token_id
    ]
    target_positions: list[list[int]] = [[] for _ in option_positions]
    if option_positions:
        # Walk left-to-right and bucket each <target> into the most recent
        # <option> seen so far.
        opt_idx = -1
        for pos, tid in enumerate(token_ids):
            if tid == option_token_id:
                opt_idx += 1
            elif tid == target_token_id and opt_idx >= 0:
                target_positions[opt_idx].append(pos)

    # ----- engine-id mapping (K -> engine card id) ---------------------------
    card_ref_engine_ids: dict[int, str] = {}
    for engine_id, k in rendered.card_refs.items():
        card_ref_engine_ids[int(k)] = engine_id

    return TextEncodedExample(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        option_positions=option_positions,
        target_positions=target_positions,
        card_ref_engine_ids=card_ref_engine_ids,
    )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate(examples: Sequence[TextEncodedExample], pad_id: int) -> TextEncodedBatch:
    """Right-pad a list of :class:`TextEncodedExample` into a tensor batch.

    * ``token_ids`` / ``attention_mask`` are right-padded to ``max(T_i)``
      with ``pad_id`` and ``0`` respectively.
    * ``card_ref_positions`` is shaped ``[B, MAX_CARD_REFS]`` with ``-1`` for
      absent ref indices (no separate mask: ``-1`` is the sentinel and downstream
      gathers should clamp).
    * ``option_positions`` / ``target_positions`` are padded to the per-batch
      maxima with ``-1`` and accompanied by boolean masks.
    """

    if len(examples) == 0:
        raise ValueError("collate() requires at least one example")

    batch_size = len(examples)
    seq_lengths = [len(ex.token_ids) for ex in examples]
    max_t = max(seq_lengths)
    max_opts = max((len(ex.option_positions) for ex in examples), default=0)
    max_targets = max(
        (len(targets) for ex in examples for targets in ex.target_positions),
        default=0,
    )

    token_ids = torch.full((batch_size, max_t), pad_id, dtype=torch.int64)
    attention_mask = torch.zeros((batch_size, max_t), dtype=torch.int64)
    card_ref_positions = torch.full((batch_size, MAX_CARD_REFS), -1, dtype=torch.int64)
    option_positions = torch.full((batch_size, max_opts), -1, dtype=torch.int64)
    option_mask = torch.zeros((batch_size, max_opts), dtype=torch.bool)
    target_positions = torch.full((batch_size, max_opts, max_targets), -1, dtype=torch.int64)
    target_mask = torch.zeros((batch_size, max_opts, max_targets), dtype=torch.bool)

    for b, ex in enumerate(examples):
        t_i = seq_lengths[b]
        token_ids[b, :t_i] = torch.as_tensor(ex.token_ids, dtype=torch.int64)
        attention_mask[b, :t_i] = torch.as_tensor(ex.attention_mask, dtype=torch.int64)

        for k, pos in ex.card_ref_positions.items():
            if 0 <= k < MAX_CARD_REFS:
                card_ref_positions[b, k] = int(pos)

        for o, pos in enumerate(ex.option_positions):
            option_positions[b, o] = int(pos)
            option_mask[b, o] = True
            for t, tpos in enumerate(ex.target_positions[o]):
                target_positions[b, o, t] = int(tpos)
                target_mask[b, o, t] = True

    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        option_positions=option_positions,
        option_mask=option_mask,
        target_positions=target_positions,
        target_mask=target_mask,
        seq_lengths=torch.as_tensor(seq_lengths, dtype=torch.int64),
    )
