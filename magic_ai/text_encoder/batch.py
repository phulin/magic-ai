"""Bridge from rendered snapshots to padded token-id batches.

PR #3/#4 of the text-encoder plan in ``docs/text_encoder_plan.md`` (§5). The
renderer (``magic_ai.text_encoder.render``) emits a string plus anchor
metadata at the *string* level. The model wants per-example token-id tensors
plus *token-position* anchors so it can gather hidden states at every
``<card-ref:K>`` and inline decision blank location.

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
    """Single rendered snapshot tokenized into ids + anchor positions.

    ``blank_*`` fields hold inline-blank metadata recovered from the rendered
    snapshot's :attr:`~magic_ai.text_encoder.render.RenderedSnapshot.blank_anchors`
    by id-equality on the kind tokens. Each list has length equal to the
    blank count for this example (variable across the batch); collation
    pads to the per-batch maxima.
    """

    token_ids: list[int]
    attention_mask: list[int]
    card_ref_positions: dict[int, int]
    card_ref_engine_ids: dict[int, str] = field(default_factory=dict)
    blank_positions: list[int] = field(default_factory=list)
    blank_kind_ids: list[int] = field(default_factory=list)  # token id of the blank kind
    blank_group_ids: list[int] = field(default_factory=list)
    blank_group_kinds: list[int] = field(default_factory=list)  # int enum
    blank_legal_ids: list[list[int]] = field(default_factory=list)
    blank_option_indices: list[int] = field(default_factory=list)


@dataclass
class TextEncodedBatch:
    """Padded batch of :class:`TextEncodedExample` ready for the encoder.

    ``blank_*`` fields carry the inline-blank tensors introduced in Step 3 of
    ``docs/text_encoder_inline_blanks_plan.md``. Rows without decisions use
    zero-shape ``[B, 0]`` / ``[B, 0, 0]`` tensors.
    """

    token_ids: Tensor  # [B, T] int64, padded with pad_id
    attention_mask: Tensor  # [B, T] int64
    card_ref_positions: Tensor  # [B, MAX_CARD_REFS] int64, -1 = absent
    seq_lengths: Tensor  # [B] int64
    # Inline-blank tensors. ``K`` is the per-batch maximum blank count and
    # ``V_max`` the per-batch maximum legal-id count across blanks.
    blank_positions: Tensor = field(  # [B, K] int32, -1 = absent
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32)
    )
    blank_kind: Tensor = field(  # [B, K] int32 (kind token id), 0 = absent
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32)
    )
    blank_group: Tensor = field(  # [B, K] int32, -1 = absent
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32)
    )
    blank_group_kind: Tensor = field(  # [B, K] int32, see render_plan.BLANK_GROUP_*
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32)
    )
    blank_option_index: Tensor = field(  # [B, K] int32, -1 = absent / not engine-option-backed
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32)
    )
    blank_legal_ids: Tensor = field(  # [B, K, V_max] int32, 0 = pad
        default_factory=lambda: torch.zeros((0, 0, 0), dtype=torch.int32)
    )
    blank_legal_mask: Tensor = field(  # [B, K, V_max] bool
        default_factory=lambda: torch.zeros((0, 0, 0), dtype=torch.bool)
    )


@dataclass
class PackedTextBatch:
    """Batch with all rows concatenated along one sequence axis.

    Anchor positions are absolute offsets into the packed row (i.e. they
    have already been shifted by ``cu_seqlens[:-1]``). ``-1`` is preserved
    as the absent-slot sentinel for inline blanks.
    """

    token_ids: Tensor  # [T_packed] int32
    seq_id: Tensor  # [T_packed] int32, document index for each token
    pos_in_seq: Tensor  # [T_packed] int32, RoPE position (resets per doc)
    cu_seqlens: Tensor  # [B + 1] int32, cumulative per-doc lengths
    seq_lengths: Tensor  # [B] int32

    state_positions: Tensor  # [B] int32, packed-offset of each row's first token
    card_ref_positions: Tensor  # [B, MAX_CARD_REFS] int32, -1 = absent
    # Inline-blank tensors (Step 3 of inline-blanks plan). Same shapes /
    # sentinels as :class:`TextEncodedBatch`, but ``blank_positions`` are
    # absolute packed-row offsets (rebased by ``state_positions``).
    blank_positions: Tensor = field(default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32))
    blank_kind: Tensor = field(default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32))
    blank_group: Tensor = field(default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32))
    blank_group_kind: Tensor = field(default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32))
    blank_option_index: Tensor = field(
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.int32)
    )
    blank_legal_ids: Tensor = field(
        default_factory=lambda: torch.zeros((0, 0, 0), dtype=torch.int32)
    )
    blank_legal_mask: Tensor = field(
        default_factory=lambda: torch.zeros((0, 0, 0), dtype=torch.bool)
    )
    # Optional host-side upper bound on per-row sequence length, used to size
    # flash_attn_varlen tiling. None falls back to the encoder's static config
    # max. Producers that know the batch's true max (the native assembler does)
    # should set this to give the kernel a tighter bound.
    max_seqlen: int | None = None


def packed_sequence_layout(seq_lengths: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build packed sequence offsets and token coordinates from row lengths."""

    seq_lens = seq_lengths.to(torch.int32)
    if seq_lens.dim() != 1:
        raise ValueError("seq_lengths must be 1-D")
    batch_size = int(seq_lens.numel())
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=seq_lens.device)
    cu_seqlens[1:] = seq_lens.cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())
    state_positions = cu_seqlens[:-1]
    seq_id = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=seq_lens.device),
        seq_lens,
    )
    pos_in_seq = torch.arange(total_tokens, dtype=torch.int32, device=seq_lens.device) - (
        state_positions.repeat_interleave(seq_lens)
    )
    return cu_seqlens, state_positions, seq_id, pos_in_seq


def add_packed_offsets(pos: Tensor, state_positions: Tensor) -> Tensor:
    """Shift per-row positions into packed coordinates, preserving -1 sentinels."""

    pos_i32 = pos.to(torch.int32)
    valid = pos_i32 >= 0
    view_shape = (int(state_positions.shape[0]),) + (1,) * (pos_i32.dim() - 1)
    shifted = pos_i32 + state_positions.to(torch.int32).view(view_shape)
    return torch.where(valid, shifted, pos_i32)


def subtract_packed_offsets(pos: Tensor, state_positions: Tensor) -> Tensor:
    """Shift packed positions back to row-local coordinates, preserving -1 sentinels."""

    pos_i32 = pos.to(torch.int32)
    valid = pos_i32 >= 0
    view_shape = (int(state_positions.shape[0]),) + (1,) * (pos_i32.dim() - 1)
    shifted = pos_i32 - state_positions.to(torch.int32).view(view_shape)
    return torch.where(valid, shifted, pos_i32)


def pack_batch(padded: TextEncodedBatch) -> PackedTextBatch:
    """Pack a padded :class:`TextEncodedBatch` into a :class:`PackedTextBatch`.

    Fully vectorized: no Python-level per-row loop. Live tokens are
    selected via ``attention_mask`` and concatenated; ``seq_id`` and
    ``pos_in_seq`` are derived from the per-row sequence lengths with
    :func:`torch.repeat_interleave` and an :func:`arange` subtraction.
    Anchors are rebased by adding the per-row base offset, with ``-1``
    sentinels preserved through a ``where``.
    """

    seq_lens = padded.seq_lengths.to(torch.int32)
    cu, state_positions, seq_id, pos_in_seq = packed_sequence_layout(seq_lens)
    t_packed = int(cu[-1].item())

    live = padded.attention_mask.to(torch.bool)
    token_ids = padded.token_ids[live].to(torch.int32)
    if int(token_ids.numel()) != t_packed:
        raise ValueError(
            "attention_mask live-count mismatch with seq_lengths "
            f"({int(token_ids.numel())} vs {t_packed})"
        )

    if padded.blank_positions.numel():
        blank_positions = add_packed_offsets(padded.blank_positions, state_positions)
    else:
        blank_positions = padded.blank_positions
    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu,
        seq_lengths=seq_lens,
        state_positions=state_positions.clone(),
        card_ref_positions=add_packed_offsets(padded.card_ref_positions, state_positions),
        blank_positions=blank_positions,
        blank_kind=padded.blank_kind,
        blank_group=padded.blank_group,
        blank_group_kind=padded.blank_group_kind,
        blank_option_index=padded.blank_option_index,
        blank_legal_ids=padded.blank_legal_ids,
        blank_legal_mask=padded.blank_legal_mask,
    )


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

    # ----- engine-id mapping (K -> engine card id) ---------------------------
    card_ref_engine_ids: dict[int, str] = {}
    for engine_id, k in rendered.card_refs.items():
        card_ref_engine_ids[int(k)] = engine_id

    # ----- inline-blank anchors ---------------------------------------------
    # For every BlankAnchor on the rendered snapshot, locate its kind token in
    # the produced ``token_ids`` stream. Render order is deterministic, so we
    # scan left-to-right and pair the n-th occurrence of any blank-kind token
    # with the n-th anchor (across all kinds, in render order).
    from magic_ai.text_encoder.render_plan import blank_group_kind_id

    blank_positions: list[int] = []
    blank_kind_ids: list[int] = []
    blank_group_ids: list[int] = []
    blank_group_kinds: list[int] = []
    blank_legal_ids: list[list[int]] = []
    blank_option_indices: list[int] = []
    if rendered.blank_anchors:
        # Pre-resolve each anchor's kind-token id and bucket the anchors by
        # kind so we don't repeatedly scan the full anchor list per token.
        anchors = list(rendered.blank_anchors)
        anchor_kind_tids: list[int] = [_single_id(anchor.kind) for anchor in anchors]
        kind_tid_set = set(anchor_kind_tids)
        # Walk tokens and consume anchors in order, requiring the n-th match
        # to use a kind token id that equals the n-th anchor's kind tid.
        cursor = 0
        for pos, tid in enumerate(token_ids):
            if cursor >= len(anchors):
                break
            if tid not in kind_tid_set:
                continue
            if tid != anchor_kind_tids[cursor]:
                # Render order requires anchor[cursor] to come next; if the
                # tokenizer emitted an unrelated blank-kind token first the
                # render fixture is broken — surface that as a hard error.
                raise RuntimeError(
                    f"blank-anchor render-order mismatch at token pos={pos}: "
                    f"got tid={tid}, expected {anchor_kind_tids[cursor]} "
                    f"(blank_index={anchors[cursor].blank_index})"
                )
            anchor = anchors[cursor]
            blank_positions.append(pos)
            blank_kind_ids.append(tid)
            blank_group_ids.append(int(anchor.group_id))
            blank_group_kinds.append(blank_group_kind_id(anchor.group_kind))
            blank_legal_ids.append([int(t) for t in anchor.legal_token_ids])
            blank_option_indices.append(int(anchor.option_index))
            cursor += 1
        if cursor != len(anchors):
            raise RuntimeError(
                f"only {cursor}/{len(anchors)} blank anchors located in token stream"
            )

    return TextEncodedExample(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        card_ref_engine_ids=card_ref_engine_ids,
        blank_positions=blank_positions,
        blank_kind_ids=blank_kind_ids,
        blank_group_ids=blank_group_ids,
        blank_group_kinds=blank_group_kinds,
        blank_legal_ids=blank_legal_ids,
        blank_option_indices=blank_option_indices,
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
    * ``blank_*`` fields are padded to the per-batch maxima with ``-1`` / ``0``
      sentinels and accompanied by boolean legal masks.
    """

    if len(examples) == 0:
        raise ValueError("collate() requires at least one example")

    batch_size = len(examples)
    seq_lengths = [len(ex.token_ids) for ex in examples]
    max_t = max(seq_lengths)
    max_blanks = max((len(ex.blank_positions) for ex in examples), default=0)
    max_legal = max(
        (len(legal) for ex in examples for legal in ex.blank_legal_ids),
        default=0,
    )

    token_ids = torch.full((batch_size, max_t), pad_id, dtype=torch.int64)
    attention_mask = torch.zeros((batch_size, max_t), dtype=torch.int64)
    card_ref_positions = torch.full((batch_size, MAX_CARD_REFS), -1, dtype=torch.int64)

    blank_positions = torch.full((batch_size, max_blanks), -1, dtype=torch.int32)
    blank_kind = torch.zeros((batch_size, max_blanks), dtype=torch.int32)
    blank_group = torch.full((batch_size, max_blanks), -1, dtype=torch.int32)
    blank_group_kind = torch.zeros((batch_size, max_blanks), dtype=torch.int32)
    blank_option_index = torch.full((batch_size, max_blanks), -1, dtype=torch.int32)
    blank_legal_ids = torch.zeros((batch_size, max_blanks, max_legal), dtype=torch.int32)
    blank_legal_mask = torch.zeros((batch_size, max_blanks, max_legal), dtype=torch.bool)

    for b, ex in enumerate(examples):
        t_i = seq_lengths[b]
        token_ids[b, :t_i] = torch.as_tensor(ex.token_ids, dtype=torch.int64)
        attention_mask[b, :t_i] = torch.as_tensor(ex.attention_mask, dtype=torch.int64)

        for k, pos in ex.card_ref_positions.items():
            if 0 <= k < MAX_CARD_REFS:
                card_ref_positions[b, k] = int(pos)

        for k_idx, pos in enumerate(ex.blank_positions):
            blank_positions[b, k_idx] = int(pos)
            blank_kind[b, k_idx] = int(ex.blank_kind_ids[k_idx])
            blank_group[b, k_idx] = int(ex.blank_group_ids[k_idx])
            blank_group_kind[b, k_idx] = int(ex.blank_group_kinds[k_idx])
            blank_option_index[b, k_idx] = int(ex.blank_option_indices[k_idx])
            for v_idx, tid in enumerate(ex.blank_legal_ids[k_idx]):
                blank_legal_ids[b, k_idx, v_idx] = int(tid)
                blank_legal_mask[b, k_idx, v_idx] = True

    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=torch.as_tensor(seq_lengths, dtype=torch.int64),
        blank_positions=blank_positions,
        blank_kind=blank_kind,
        blank_group=blank_group,
        blank_group_kind=blank_group_kind,
        blank_option_index=blank_option_index,
        blank_legal_ids=blank_legal_ids,
        blank_legal_mask=blank_legal_mask,
    )
