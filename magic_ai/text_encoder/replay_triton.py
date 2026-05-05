"""Triton helpers for text replay-buffer batch appends."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from magic_ai.replay_buffer import DecisionLayoutBatch

_triton: Any = None
_tl: Any = None
try:
    _triton = importlib.import_module("triton")
    _tl = importlib.import_module("triton.language")
except ImportError:  # pragma: no cover - exercised on CPU-only/dev installs.
    pass

triton: Any = _triton
tl: Any = _tl

TRITON_AVAILABLE = triton is not None and tl is not None

_clear_decision_rows_kernel: Any = None
_write_decision_rows_kernel: Any = None
_gather_packed_tokens_kernel: Any = None
_gather_decisions_compact_kernel: Any = None


def _launch(kernel: Any, grid: tuple[int, ...], *args: Any) -> None:
    kernel[grid](*args)


def _cdiv(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


if TRITON_AVAILABLE and not TYPE_CHECKING:

    @triton.jit(
        do_not_specialize=[
            "batch_size",
            "total_choice_slots",
            "total_group_slots",
            "max_total",
        ]
    )
    def _clear_decision_rows_kernel(
        rows,
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
        batch_size,
        max_decision_groups: tl.constexpr,
        max_cached_choices: tl.constexpr,
        total_choice_slots,
        total_group_slots,
        max_total,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)

        choice_mask = offsets < total_choice_slots
        slots_per_row = max_decision_groups * max_cached_choices
        choice_b = offsets // slots_per_row
        choice_in_row = offsets - choice_b * slots_per_row
        choice_row = tl.load(rows + choice_b, mask=choice_mask, other=0)
        choice_dst = choice_row * slots_per_row + choice_in_row
        tl.store(decision_option_idx + choice_dst, -1, mask=choice_mask)
        tl.store(decision_target_idx + choice_dst, -1, mask=choice_mask)
        tl.store(decision_mask + choice_dst, 0, mask=choice_mask)

        group_mask = offsets < total_group_slots
        group_b = offsets // max_decision_groups
        group_in_row = offsets - group_b * max_decision_groups
        group_row = tl.load(rows + group_b, mask=group_mask, other=0)
        group_dst = group_row * max_decision_groups + group_in_row
        tl.store(uses_none_head + group_dst, 0, mask=group_mask)
        tl.store(selected_indices + group_dst, -1, mask=group_mask)

    @triton.jit
    def _write_decision_rows_kernel(
        rows,
        group_starts,
        decision_count_in,
        decision_option_idx_in,
        decision_target_idx_in,
        decision_mask_in,
        uses_none_head_in,
        selected_indices_in,
        decision_option_idx_out,
        decision_target_idx_out,
        decision_mask_out,
        uses_none_head_out,
        selected_indices_out,
        decision_count_out,
        trace_kind_id_in,
        may_selected_in,
        old_log_prob_in,
        value_in,
        perspective_player_idx_in,
        trace_kind_id_out,
        may_selected_out,
        old_log_prob_out,
        value_out,
        perspective_player_idx_out,
        max_decision_groups: tl.constexpr,
        max_cached_choices: tl.constexpr,
        slots_per_row: tl.constexpr,
        choice_block: tl.constexpr,
    ):
        b = tl.program_id(0)
        row = tl.load(rows + b)
        group_start = tl.load(group_starts + b)
        raw_count = tl.load(decision_count_in + b)
        stored_count = tl.minimum(raw_count, max_decision_groups)
        tl.store(decision_count_out + row, stored_count)
        tl.store(trace_kind_id_out + row, tl.load(trace_kind_id_in + b))
        tl.store(may_selected_out + row, tl.load(may_selected_in + b))
        tl.store(old_log_prob_out + row, tl.load(old_log_prob_in + b))
        tl.store(value_out + row, tl.load(value_in + b))
        tl.store(perspective_player_idx_out + row, tl.load(perspective_player_idx_in + b))

        choice_offsets = tl.arange(0, choice_block)
        choice_mask = choice_offsets < slots_per_row
        group = choice_offsets // max_cached_choices
        choice = choice_offsets - group * max_cached_choices
        valid_group = group < stored_count
        flat_group = group_start + group
        in_choice_offsets = flat_group * max_cached_choices + choice
        out_choice_offsets = row * slots_per_row + choice_offsets

        option_value = tl.load(
            decision_option_idx_in + in_choice_offsets,
            mask=choice_mask & valid_group,
            other=-1,
        )
        target_value = tl.load(
            decision_target_idx_in + in_choice_offsets,
            mask=choice_mask & valid_group,
            other=-1,
        )
        mask_value = tl.load(
            decision_mask_in + in_choice_offsets,
            mask=choice_mask & valid_group,
            other=0,
        )
        tl.store(
            decision_option_idx_out + out_choice_offsets,
            tl.where(valid_group, option_value, -1),
            mask=choice_mask,
        )
        tl.store(
            decision_target_idx_out + out_choice_offsets,
            tl.where(valid_group, target_value, -1),
            mask=choice_mask,
        )
        tl.store(
            decision_mask_out + out_choice_offsets,
            tl.where(valid_group, mask_value, 0),
            mask=choice_mask,
        )

        group_mask = choice_offsets < max_decision_groups
        group_valid = choice_offsets < stored_count
        flat_group_offsets = group_start + choice_offsets
        out_group_offsets = row * max_decision_groups + choice_offsets
        none_value = tl.load(
            uses_none_head_in + flat_group_offsets,
            mask=group_mask & group_valid,
            other=0,
        )
        selected_value = tl.load(
            selected_indices_in + flat_group_offsets,
            mask=group_mask & group_valid,
            other=-1,
        )
        tl.store(
            uses_none_head_out + out_group_offsets,
            tl.where(group_valid, none_value, 0),
            mask=group_mask,
        )
        tl.store(
            selected_indices_out + out_group_offsets,
            tl.where(group_valid, selected_value, -1),
            mask=group_mask,
        )

    @triton.jit
    def _gather_packed_tokens_kernel(
        rows,
        row_token_start,
        seq_lengths,
        cu_seqlens,
        src_token_ids,
        dst_token_ids,
        dst_seq_id,
        dst_pos_in_seq,
        chunk: tl.constexpr,
        store_seq_id: tl.constexpr,
    ):
        b = tl.program_id(0)
        row = tl.load(rows + b)
        row_start = tl.load(row_token_start + row)
        length = tl.load(seq_lengths + b)
        out_start = tl.load(cu_seqlens + b)
        for start in range(0, length, chunk):
            offsets = start + tl.arange(0, chunk)
            mask = offsets < length
            values = tl.load(src_token_ids + row_start + offsets, mask=mask, other=0)
            out_offsets = out_start + offsets
            tl.store(dst_token_ids + out_offsets, values, mask=mask)
            if store_seq_id:
                tl.store(dst_seq_id + out_offsets, b, mask=mask)
            tl.store(dst_pos_in_seq + out_offsets, offsets, mask=mask)

    @triton.jit(
        do_not_specialize=[
            "batch_size",
            "total_choice",
            "total_group",
            "max_total",
        ]
    )
    def _gather_decisions_compact_kernel(
        rows,
        gathered_starts,
        decision_start,
        decision_count,
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
        decision_option_idx_out,
        decision_target_idx_out,
        decision_mask_out,
        uses_none_head_out,
        selected_indices_out,
        step_for_group_out,
        batch_size,
        max_decision_groups: tl.constexpr,
        max_cached_choices: tl.constexpr,
        total_choice,
        total_group,
        max_total,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)

        group_m = offsets < total_group
        b = offsets // max_decision_groups
        group_in_row = offsets - b * max_decision_groups
        row = tl.load(rows + b, mask=group_m, other=0)
        count = tl.load(decision_count + row, mask=group_m, other=0)
        valid_group = group_m & (group_in_row < count)
        src_start = tl.load(decision_start + row, mask=group_m, other=0)
        dst_start = tl.load(gathered_starts + b, mask=group_m, other=0)
        src_group = src_start + group_in_row
        dst_group = dst_start + group_in_row
        none_val = tl.load(uses_none_head + src_group, mask=valid_group, other=0)
        selected_val = tl.load(selected_indices + src_group, mask=valid_group, other=-1)
        tl.store(uses_none_head_out + dst_group, none_val, mask=valid_group)
        tl.store(selected_indices_out + dst_group, selected_val, mask=valid_group)
        tl.store(step_for_group_out + dst_group, b, mask=valid_group)

        choice_m = offsets < total_choice
        slots_per_row = max_decision_groups * max_cached_choices
        choice_b = offsets // slots_per_row
        choice_in_row = offsets - choice_b * slots_per_row
        choice_group = choice_in_row // max_cached_choices
        choice_col = choice_in_row - choice_group * max_cached_choices
        choice_row = tl.load(rows + choice_b, mask=choice_m, other=0)
        choice_count = tl.load(decision_count + choice_row, mask=choice_m, other=0)
        valid_choice = choice_m & (choice_group < choice_count)
        choice_src_start = tl.load(decision_start + choice_row, mask=choice_m, other=0)
        choice_dst_start = tl.load(gathered_starts + choice_b, mask=choice_m, other=0)
        choice_src = (choice_src_start + choice_group) * max_cached_choices + choice_col
        choice_dst = (choice_dst_start + choice_group) * max_cached_choices + choice_col
        tl.store(
            decision_option_idx_out + choice_dst,
            tl.load(decision_option_idx + choice_src, mask=valid_choice, other=-1),
            mask=valid_choice,
        )
        tl.store(
            decision_target_idx_out + choice_dst,
            tl.load(decision_target_idx + choice_src, mask=valid_choice, other=-1),
            mask=valid_choice,
        )
        tl.store(
            decision_mask_out + choice_dst,
            tl.load(decision_mask + choice_src, mask=valid_choice, other=0),
            mask=valid_choice,
        )


def clear_append_decisions_triton(
    *,
    rows: Tensor,
    decision_option_idx: Tensor,
    decision_target_idx: Tensor,
    decision_mask: Tensor,
    uses_none_head: Tensor,
    selected_indices: Tensor,
) -> bool:
    """Clear fixed-width per-row decision slots with one Triton launch."""

    if not TRITON_AVAILABLE or not rows.is_cuda:
        return False
    tensors = (
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return False

    batch_size = int(rows.shape[0])
    max_decision_groups = int(uses_none_head.shape[1])
    max_cached_choices = int(decision_option_idx.shape[2])
    total_choice_slots = batch_size * max_decision_groups * max_cached_choices
    total_group_slots = batch_size * max_decision_groups
    max_total = max(total_choice_slots, total_group_slots)
    if max_total == 0:
        return True
    block = 512
    _launch(
        _clear_decision_rows_kernel,
        (_cdiv(max_total, block),),
        rows,
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
        batch_size,
        max_decision_groups,
        max_cached_choices,
        total_choice_slots,
        total_group_slots,
        max_total,
        block,
    )
    return True


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def write_append_decisions_triton(
    *,
    rows: Tensor,
    decision_count: Tensor,
    decision_option_idx: Tensor,
    decision_target_idx: Tensor,
    decision_mask: Tensor,
    uses_none_head: Tensor,
    selected_indices: Tensor,
    trace_kind_id: Tensor,
    may_selected: Tensor,
    old_log_prob: Tensor,
    value: Tensor,
    perspective_player_idx: Tensor,
    dst_decision_option_idx: Tensor,
    dst_decision_target_idx: Tensor,
    dst_decision_mask: Tensor,
    dst_uses_none_head: Tensor,
    dst_selected_indices: Tensor,
    dst_decision_count: Tensor,
    dst_trace_kind_id: Tensor,
    dst_may_selected: Tensor,
    dst_old_log_prob: Tensor,
    dst_value: Tensor,
    dst_perspective_player_idx: Tensor,
) -> bool:
    """Write and clear per-row decision slots directly from flattened groups."""

    if not TRITON_AVAILABLE or not rows.is_cuda:
        return False
    tensors = (
        decision_count,
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
        trace_kind_id,
        may_selected,
        old_log_prob,
        value,
        perspective_player_idx,
        dst_decision_option_idx,
        dst_decision_target_idx,
        dst_decision_mask,
        dst_uses_none_head,
        dst_selected_indices,
        dst_decision_count,
        dst_trace_kind_id,
        dst_may_selected,
        dst_old_log_prob,
        dst_value,
        dst_perspective_player_idx,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return False

    batch_size = int(rows.shape[0])
    max_decision_groups = int(dst_uses_none_head.shape[1])
    max_cached_choices = int(dst_decision_option_idx.shape[2])
    slots_per_row = max_decision_groups * max_cached_choices
    choice_block = _next_power_of_2(max(slots_per_row, max_decision_groups))
    if choice_block > 1024:
        return False
    stored_counts = decision_count.clamp(max=max_decision_groups)
    group_starts_excl = torch.zeros(batch_size, dtype=torch.int64, device=rows.device)
    if batch_size > 1:
        torch.cumsum(stored_counts[:-1], dim=0, out=group_starts_excl[1:])
    _launch(
        _write_decision_rows_kernel,
        (batch_size,),
        rows,
        group_starts_excl,
        decision_count,
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
        dst_decision_option_idx,
        dst_decision_target_idx,
        dst_decision_mask,
        dst_uses_none_head,
        dst_selected_indices,
        dst_decision_count,
        trace_kind_id,
        may_selected,
        old_log_prob,
        value,
        perspective_player_idx,
        dst_trace_kind_id,
        dst_may_selected,
        dst_old_log_prob,
        dst_value,
        dst_perspective_player_idx,
        max_decision_groups,
        max_cached_choices,
        slots_per_row,
        choice_block,
    )
    return True


def gather_decisions_triton(
    *,
    rows: Tensor,
    decision_start: Tensor,
    decision_count: Tensor,
    decision_option_idx: Tensor,
    decision_target_idx: Tensor,
    decision_mask: Tensor,
    uses_none_head: Tensor,
    selected_indices: Tensor,
    max_decision_groups: int,
) -> DecisionLayoutBatch | None:
    """Gather compact per-step decision groups with Triton."""

    if not TRITON_AVAILABLE or not rows.is_cuda:
        return None
    tensors = (
        rows,
        decision_start,
        decision_count,
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return None

    batch_size = int(rows.shape[0])
    if batch_size == 0:
        return None

    counts = decision_count[rows]
    gathered_starts = torch.zeros(batch_size, dtype=counts.dtype, device=rows.device)
    torch.cumsum(counts[:-1], dim=0, out=gathered_starts[1:])
    total_groups = int(counts.sum().item())
    max_cached_choices = int(decision_option_idx.shape[1])
    index_dtype = (
        decision_option_idx.dtype
        if decision_option_idx.dtype in (torch.int32, torch.long)
        else torch.int32
    )
    if total_groups == 0:
        return DecisionLayoutBatch(
            decision_start=gathered_starts,
            decision_count=counts,
            decision_option_idx=torch.empty(
                (0, max_cached_choices), dtype=index_dtype, device=rows.device
            ),
            decision_target_idx=torch.empty(
                (0, max_cached_choices), dtype=index_dtype, device=rows.device
            ),
            decision_mask=torch.empty(
                (0, max_cached_choices), dtype=torch.bool, device=rows.device
            ),
            uses_none_head=torch.empty(0, dtype=torch.bool, device=rows.device),
            selected_indices=torch.empty(0, dtype=index_dtype, device=rows.device),
            step_for_group=torch.empty(0, dtype=torch.long, device=rows.device),
        )

    decision_option_idx_out = torch.empty(
        (total_groups, max_cached_choices), dtype=index_dtype, device=rows.device
    )
    decision_target_idx_out = torch.empty_like(decision_option_idx_out)
    decision_mask_out = torch.empty(
        (total_groups, max_cached_choices), dtype=torch.bool, device=rows.device
    )
    uses_none_head_out = torch.empty(total_groups, dtype=torch.bool, device=rows.device)
    selected_indices_out = torch.empty(total_groups, dtype=index_dtype, device=rows.device)
    step_for_group_out = torch.empty(total_groups, dtype=torch.long, device=rows.device)

    total_group_slots = batch_size * max_decision_groups
    total_choice_slots = total_group_slots * max_cached_choices
    block = 256
    max_total = max(total_group_slots, total_choice_slots)
    _launch(
        _gather_decisions_compact_kernel,
        (_cdiv(max_total, block),),
        rows,
        gathered_starts,
        decision_start,
        decision_count,
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
        decision_option_idx_out,
        decision_target_idx_out,
        decision_mask_out,
        uses_none_head_out,
        selected_indices_out,
        step_for_group_out,
        batch_size,
        max_decision_groups,
        max_cached_choices,
        total_choice_slots,
        total_group_slots,
        max_total,
        block,
    )
    return DecisionLayoutBatch(
        decision_start=gathered_starts,
        decision_count=counts,
        decision_option_idx=decision_option_idx_out,
        decision_target_idx=decision_target_idx_out,
        decision_mask=decision_mask_out,
        uses_none_head=uses_none_head_out,
        selected_indices=selected_indices_out,
        step_for_group=step_for_group_out,
    )


def gather_sequence_layout_triton(
    *,
    rows: Tensor,
    row_token_length: Tensor,
) -> tuple[Tensor, Tensor] | None:
    """Gather sequence lengths and build compact packed offsets on GPU."""

    if not rows.is_cuda or not row_token_length.is_cuda:
        return None

    batch_size = int(rows.shape[0])
    if batch_size == 0:
        return None

    seq_lengths = row_token_length[rows]
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=rows.device)
    torch.cumsum(seq_lengths, dim=0, out=cu_seqlens[1:])
    return seq_lengths, cu_seqlens


def gather_encoded_triton(
    *,
    rows: Tensor,
    seq_lengths: Tensor,
    cu_seqlens: Tensor,
    packed_token_ids: Tensor,
    row_token_start: Tensor,
    include_seq_id: bool = True,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Gather packed token sequences with Triton (token copy only)."""

    if not TRITON_AVAILABLE or not rows.is_cuda:
        return None
    tensors = (rows, seq_lengths, cu_seqlens, packed_token_ids, row_token_start)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return None

    batch_size = int(seq_lengths.shape[0])
    if batch_size == 0:
        return None
    total_tokens = cu_seqlens[-1]

    token_count = cast(Any, total_tokens)
    token_ids = torch.empty(token_count, dtype=packed_token_ids.dtype, device=rows.device)
    seq_id = (
        torch.empty(token_count, dtype=torch.int32, device=rows.device)
        if include_seq_id
        else torch.empty(0, dtype=torch.int32, device=rows.device)
    )
    pos_in_seq = torch.empty(token_count, dtype=torch.int32, device=rows.device)
    seq_id_out = seq_id if include_seq_id else torch.empty(1, dtype=torch.int32, device=rows.device)
    _launch(
        _gather_packed_tokens_kernel,
        (batch_size,),
        rows,
        row_token_start,
        seq_lengths,
        cu_seqlens,
        packed_token_ids,
        token_ids,
        seq_id_out,
        pos_in_seq,
        512,
        include_seq_id,
    )
    return token_ids, seq_id, pos_in_seq


__all__ = [
    "TRITON_AVAILABLE",
    "clear_append_decisions_triton",
    "gather_decisions_triton",
    "gather_encoded_triton",
    "gather_sequence_layout_triton",
    "write_append_decisions_triton",
]
