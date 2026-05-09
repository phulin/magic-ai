"""Bridge from rendered snapshots to padded token-id batches.

The renderer (``magic_ai.text_encoder.render``) emits state text plus
card-ref anchors at the *string* level. The decision-spec renderer
(``magic_ai.text_encoder.render_spec``) emits the per-decision spec
section. The model wants per-example token-id tensors plus *token-position*
anchors so it can gather hidden states at every ``<card-ref:K>`` and at
each pointer-anchor position.

This module owns the tokenize-and-collate path:

* :func:`tokenize_snapshot` turns a :class:`RenderedSnapshot` into a
  per-example :class:`TextEncodedExample`.
* :func:`collate` right-pads a sequence of examples into a
  :class:`TextEncodedBatch` of int64 tensors. The combined token stream
  is ``state_tokens || spec_tokens``; pointer anchor positions are
  shifted by the row's state length so they index into the combined
  stream.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.decision_spec import AnchorKind, DecisionSpec, DecisionType
from magic_ai.text_encoder.render import RenderedSnapshot
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS

# Sentinel default for dataclass fields that __post_init__ always populates.
# Typing as ``Tensor`` (rather than ``Tensor | None``) keeps callers and
# downstream consumers free of None-guards; the runtime None is replaced
# before the first user-visible read.
_UNSET_TENSOR: Tensor = cast(Tensor, None)

_CARD_REF_RE = re.compile(r"^<card-ref:(\d+)>$")


@dataclass
class TextEncodedExample:
    """Single rendered snapshot tokenized into ids + anchor positions."""

    token_ids: list[int]
    attention_mask: list[int]
    card_ref_positions: dict[int, int]
    card_ref_engine_ids: dict[int, str] = field(default_factory=dict)


@dataclass
class TextEncodedBatch:
    """Padded batch of :class:`TextEncodedExample` ready for the encoder.

    Combined token stream is ``state || spec`` per row. Pointer-anchor
    positions are absolute offsets into that combined stream.
    """

    token_ids: Tensor  # [B, T] int64, padded with pad_id
    attention_mask: Tensor  # [B, T] int64
    card_ref_positions: Tensor  # [B, MAX_CARD_REFS] int64, -1 = absent
    seq_lengths: Tensor  # [B] int64
    # Decision-spec fields are optional inputs (state-only batches like MLM
    # pretraining leave them unset); ``__post_init__`` always materializes
    # them, so consumers see ``Tensor`` not ``Tensor | None``.
    spec_tokens: Tensor = _UNSET_TENSOR  # [B, T_spec_max] int32, 0 = pad
    spec_lens: Tensor = _UNSET_TENSOR  # [B] int32
    decision_type: Tensor = _UNSET_TENSOR  # [B] int32 (DecisionType enum, -1 = no pending)
    pointer_anchor_positions: Tensor = _UNSET_TENSOR  # [B, N_anchors_max] int32, -1 = pad
    pointer_anchor_kinds: Tensor = _UNSET_TENSOR  # [B, N_anchors_max] int32 (AnchorKind), -1 = pad
    pointer_anchor_subjects: Tensor = _UNSET_TENSOR  # [B, N_anchors_max] int32, -1 = pad
    pointer_anchor_handles: Tensor = _UNSET_TENSOR  # [B, N_anchors_max] int32, -1 = pad
    # ``[B, N_blockers_max, N_attackers_max]`` bool. ``None`` when no row in
    # the batch has DECLARE_BLOCKERS legal-edge data.
    legal_edge_bitmap: Tensor | None = None
    total_tokens: int | None = None
    seq_lengths_host: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.seq_lengths_host is None and self.seq_lengths.device.type == "cpu":
            self.seq_lengths_host = tuple(int(x) for x in self.seq_lengths.tolist())
        if self.total_tokens is None and self.seq_lengths_host is not None:
            self.total_tokens = sum(self.seq_lengths_host)
        b = int(self.seq_lengths.shape[0])
        device = self.seq_lengths.device
        if self.spec_tokens is None:
            self.spec_tokens = torch.zeros((b, 0), dtype=torch.int32, device=device)
        if self.spec_lens is None:
            self.spec_lens = torch.zeros((b,), dtype=torch.int32, device=device)
        if self.decision_type is None:
            self.decision_type = torch.full((b,), -1, dtype=torch.int32, device=device)
        if self.pointer_anchor_positions is None:
            self.pointer_anchor_positions = torch.full((b, 0), -1, dtype=torch.int32, device=device)
        if self.pointer_anchor_kinds is None:
            self.pointer_anchor_kinds = torch.full((b, 0), -1, dtype=torch.int32, device=device)
        if self.pointer_anchor_subjects is None:
            self.pointer_anchor_subjects = torch.full((b, 0), -1, dtype=torch.int32, device=device)
        if self.pointer_anchor_handles is None:
            self.pointer_anchor_handles = torch.full((b, 0), -1, dtype=torch.int32, device=device)


@dataclass
class PackedTextBatch:
    """Batch with all rows concatenated along one sequence axis."""

    token_ids: Tensor  # [T_packed] int32
    seq_id: Tensor  # [T_packed] int32, document index for each token
    pos_in_seq: Tensor  # [T_packed] int32, RoPE position (resets per doc)
    cu_seqlens: Tensor  # [B + 1] int32, cumulative per-doc lengths
    seq_lengths: Tensor  # [B] int32

    state_positions: Tensor  # [B] int32, packed-offset of each row's first token
    card_ref_positions: Tensor  # [B, MAX_CARD_REFS] int32, -1 = absent
    spec_lens: Tensor = _UNSET_TENSOR  # [B] int32
    decision_type: Tensor = _UNSET_TENSOR  # [B] int32
    pointer_anchor_positions: Tensor = _UNSET_TENSOR  # [B, N_anchors_max] int32, -1 = pad
    pointer_anchor_kinds: Tensor = _UNSET_TENSOR  # [B, N_anchors_max] int32, -1 = pad
    pointer_anchor_subjects: Tensor = _UNSET_TENSOR  # [B, N_anchors_max] int32, -1 = pad
    pointer_anchor_handles: Tensor = _UNSET_TENSOR  # [B, N_anchors_max] int32, -1 = pad
    legal_edge_bitmap: Tensor | None = None
    total_tokens: int | None = None
    seq_lengths_host: tuple[int, ...] | None = None
    max_seqlen: int | None = None

    def __post_init__(self) -> None:
        b = int(self.seq_lengths.shape[0])
        device = self.seq_lengths.device
        if self.spec_lens is None:
            self.spec_lens = torch.zeros((b,), dtype=torch.int32, device=device)
        if self.decision_type is None:
            self.decision_type = torch.full((b,), -1, dtype=torch.int32, device=device)
        if self.pointer_anchor_positions is None:
            self.pointer_anchor_positions = torch.full((b, 0), -1, dtype=torch.int32, device=device)
        if self.pointer_anchor_kinds is None:
            self.pointer_anchor_kinds = torch.full((b, 0), -1, dtype=torch.int32, device=device)
        if self.pointer_anchor_subjects is None:
            self.pointer_anchor_subjects = torch.full((b, 0), -1, dtype=torch.int32, device=device)
        if self.pointer_anchor_handles is None:
            self.pointer_anchor_handles = torch.full((b, 0), -1, dtype=torch.int32, device=device)


def packed_sequence_layout(
    seq_lengths: Tensor,
    *,
    total_tokens: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build packed sequence offsets and token coordinates from row lengths."""

    seq_lens = seq_lengths.to(torch.int32)
    if seq_lens.dim() != 1:
        raise ValueError("seq_lengths must be 1-D")
    batch_size = int(seq_lens.numel())
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=seq_lens.device)
    cu_seqlens[1:] = seq_lens.cumsum(0)
    if total_tokens is None:
        total_tokens = int(cu_seqlens[-1].item())
    state_positions = cu_seqlens[:-1]
    seq_id = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=seq_lens.device),
        seq_lens,
        output_size=total_tokens,
    )
    pos_in_seq = torch.arange(total_tokens, dtype=torch.int32, device=seq_lens.device) - (
        state_positions.repeat_interleave(seq_lens, output_size=total_tokens)
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
    """Pack a padded :class:`TextEncodedBatch` into a :class:`PackedTextBatch`."""

    if padded.seq_lengths_host is None:
        raise ValueError("pack_batch requires seq_lengths_host")

    seq_lens = padded.seq_lengths.to(torch.int32)
    total_tokens = padded.total_tokens
    if total_tokens is None:
        total_tokens = int(seq_lens.sum().item())
    cu, state_positions, seq_id, pos_in_seq = packed_sequence_layout(
        seq_lens,
        total_tokens=total_tokens,
    )
    token_ids = padded.token_ids[seq_id, pos_in_seq].to(torch.int32)

    max_seqlen = max(padded.seq_lengths_host, default=0)
    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu,
        seq_lengths=seq_lens,
        state_positions=state_positions.clone(),
        card_ref_positions=add_packed_offsets(padded.card_ref_positions, state_positions),
        spec_lens=padded.spec_lens,
        decision_type=padded.decision_type,
        pointer_anchor_positions=add_packed_offsets(
            padded.pointer_anchor_positions, state_positions
        ),
        pointer_anchor_kinds=padded.pointer_anchor_kinds,
        pointer_anchor_subjects=padded.pointer_anchor_subjects,
        pointer_anchor_handles=padded.pointer_anchor_handles,
        legal_edge_bitmap=padded.legal_edge_bitmap,
        total_tokens=total_tokens,
        seq_lengths_host=padded.seq_lengths_host,
        max_seqlen=max_seqlen,
    )


# ---------------------------------------------------------------------------
# Tokenize a single rendered snapshot
# ---------------------------------------------------------------------------


def tokenize_snapshot(
    rendered: RenderedSnapshot,
    tokenizer: PreTrainedTokenizerFast,
) -> TextEncodedExample:
    """Tokenize ``rendered.text`` and recover anchor token positions."""

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
        if k not in card_ref_positions:
            card_ref_positions[k] = pos

    for pos, tid in enumerate(token_ids):
        token_str = tokenizer.convert_ids_to_tokens(int(tid))
        if not isinstance(token_str, str):
            continue
        match = _CARD_REF_RE.match(token_str)
        if match is None:
            continue
        k_parsed = int(match.group(1))
        assert card_ref_id_to_k.get(int(tid)) == k_parsed, (
            f"card-ref id mismatch at pos={pos}: id={tid} parsed={k_parsed}"
        )

    card_ref_engine_ids: dict[int, str] = {}
    for engine_id, k in rendered.card_refs.items():
        card_ref_engine_ids[int(k)] = engine_id

    return TextEncodedExample(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        card_ref_engine_ids=card_ref_engine_ids,
    )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate(
    examples: Sequence[TextEncodedExample],
    specs: Sequence[DecisionSpec | None],
    pad_id: int,
) -> TextEncodedBatch:
    """Collate ``examples`` and per-row ``specs`` into a tensor batch.

    Each row's combined token stream is ``state_tokens || spec_tokens``; the
    encoder consumes the combined stream so the trunk sees both the rendered
    state and the spec section in one forward pass. Pointer-anchor positions
    (which the renderer records relative to the spec section's start) are
    shifted by the row's state-token length so they index into the combined
    stream.

    Rows with ``specs[i] is None`` contribute no spec tokens and decision_type
    ``-1`` — useful when batching state-only snapshots alongside decision
    snapshots (e.g. for MLM pretraining batches).
    """

    if len(examples) != len(specs):
        raise ValueError(f"examples / specs length mismatch: {len(examples)} vs {len(specs)}")
    if len(examples) == 0:
        raise ValueError("collate() requires at least one example")

    batch_size = len(examples)
    state_lens = [len(ex.token_ids) for ex in examples]
    spec_lens = [len(s.spec_tokens) if s is not None else 0 for s in specs]
    seq_lengths = [state_lens[i] + spec_lens[i] for i in range(batch_size)]
    max_t = max(seq_lengths)
    n_anchors = [len(s.anchors) if s is not None else 0 for s in specs]
    max_n_anchors = max(n_anchors) if n_anchors else 0

    token_ids = torch.full((batch_size, max_t), pad_id, dtype=torch.int64)
    attention_mask = torch.zeros((batch_size, max_t), dtype=torch.int64)
    card_ref_positions = torch.full((batch_size, MAX_CARD_REFS), -1, dtype=torch.int64)

    max_t_spec = max(spec_lens) if spec_lens else 0
    spec_tokens = torch.zeros((batch_size, max_t_spec), dtype=torch.int32)
    decision_type_t = torch.full((batch_size,), -1, dtype=torch.int32)
    pointer_positions = torch.full((batch_size, max_n_anchors), -1, dtype=torch.int32)
    pointer_kinds = torch.full((batch_size, max_n_anchors), -1, dtype=torch.int32)
    pointer_subjects = torch.full((batch_size, max_n_anchors), -1, dtype=torch.int32)
    pointer_handles = torch.full((batch_size, max_n_anchors), -1, dtype=torch.int32)

    max_blockers = 0
    max_attackers = 0
    has_edges = False
    for s in specs:
        if s is None or s.legal_edge_bitmap is None:
            continue
        has_edges = True
        rows, cols = s.legal_edge_bitmap.shape
        if rows > max_blockers:
            max_blockers = rows
        if cols > max_attackers:
            max_attackers = cols
    legal_edge_bitmap: Tensor | None = (
        torch.zeros((batch_size, max_blockers, max_attackers), dtype=torch.bool)
        if has_edges
        else None
    )

    for b, ex in enumerate(examples):
        t_state = state_lens[b]
        t_spec = spec_lens[b]
        t_total = t_state + t_spec
        token_ids[b, :t_state] = torch.as_tensor(ex.token_ids, dtype=torch.int64)
        attention_mask[b, :t_total] = 1
        spec = specs[b]
        if spec is not None and t_spec > 0:
            spec_int32 = torch.as_tensor(spec.spec_tokens, dtype=torch.int32)
            token_ids[b, t_state:t_total] = spec_int32.to(torch.int64)
            spec_tokens[b, :t_spec] = spec_int32

        for k, pos in ex.card_ref_positions.items():
            if 0 <= k < MAX_CARD_REFS:
                card_ref_positions[b, k] = int(pos)

        if spec is None:
            continue
        decision_type_t[b] = int(spec.decision_type)
        for a_idx, anchor in enumerate(spec.anchors):
            pointer_positions[b, a_idx] = int(anchor.token_position) + t_state
            pointer_kinds[b, a_idx] = int(anchor.kind)
            pointer_subjects[b, a_idx] = int(anchor.subject_index)
            pointer_handles[b, a_idx] = int(anchor.handle)
        if legal_edge_bitmap is not None and spec.legal_edge_bitmap is not None:
            rows, cols = spec.legal_edge_bitmap.shape
            legal_edge_bitmap[b, :rows, :cols] = torch.from_numpy(spec.legal_edge_bitmap)

    seq_lens_t = torch.as_tensor(seq_lengths, dtype=torch.int64)
    spec_lens_t = torch.as_tensor(spec_lens, dtype=torch.int32)

    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lens_t,
        total_tokens=sum(seq_lengths),
        seq_lengths_host=tuple(seq_lengths),
        spec_tokens=spec_tokens,
        spec_lens=spec_lens_t,
        decision_type=decision_type_t,
        pointer_anchor_positions=pointer_positions,
        pointer_anchor_kinds=pointer_kinds,
        pointer_anchor_subjects=pointer_subjects,
        pointer_anchor_handles=pointer_handles,
        legal_edge_bitmap=legal_edge_bitmap,
    )


# Re-export AnchorKind / DecisionType for callers.
_ = (AnchorKind, DecisionType)


__all__ = [
    "TextEncodedExample",
    "TextEncodedBatch",
    "PackedTextBatch",
    "tokenize_snapshot",
    "collate",
    "pack_batch",
    "packed_sequence_layout",
    "add_packed_offsets",
    "subtract_packed_offsets",
]
