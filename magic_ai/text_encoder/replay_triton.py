"""Triton helpers for text replay-buffer batch appends."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from torch import Tensor

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

_copy_packed_tokens_kernel: Any = None
_write_rebased_fields_kernel: Any = None
_clear_decision_rows_kernel: Any = None
_write_decision_rows_kernel: Any = None


def _launch(kernel: Any, grid: tuple[int, ...], *args: Any) -> None:
    kernel[grid](*args)


def _cdiv(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


if TRITON_AVAILABLE and not TYPE_CHECKING:

    @triton.jit
    def _copy_packed_tokens_kernel(
        src,
        dst,
        token_start,
        total_tokens: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < total_tokens
        values = tl.load(src + offsets, mask=mask, other=0)
        tl.store(dst + token_start + offsets, values, mask=mask)

    @triton.jit
    def _write_rebased_fields_kernel(
        rows,
        cu_seqlens,
        seq_lengths,
        state_positions,
        card_pos,
        option_pos,
        option_mask,
        target_pos,
        target_mask,
        row_token_start,
        row_token_length,
        seq_lengths_out,
        dst_card_pos,
        dst_option_pos,
        dst_option_mask,
        dst_target_pos,
        dst_target_mask,
        token_start,
        batch_size: tl.constexpr,
        card_width: tl.constexpr,
        dst_card_width: tl.constexpr,
        src_options: tl.constexpr,
        src_targets: tl.constexpr,
        dst_options: tl.constexpr,
        dst_targets: tl.constexpr,
        total_card: tl.constexpr,
        total_option: tl.constexpr,
        total_target: tl.constexpr,
        max_total: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)

        row_mask = offsets < batch_size
        row_b = offsets
        row_values = tl.load(rows + row_b, mask=row_mask, other=0)
        starts = tl.load(cu_seqlens + row_b, mask=row_mask, other=0) + token_start
        lengths = tl.load(seq_lengths + row_b, mask=row_mask, other=0)
        tl.store(row_token_start + row_values, starts, mask=row_mask)
        tl.store(row_token_length + row_values, lengths, mask=row_mask)
        tl.store(seq_lengths_out + row_values, lengths, mask=row_mask)

        card_mask = offsets < total_card
        card_b = offsets // card_width
        card_col = offsets - card_b * card_width
        card_row = tl.load(rows + card_b, mask=card_mask, other=0)
        card_base = tl.load(state_positions + card_b, mask=card_mask, other=0)
        card_value = tl.load(card_pos + offsets, mask=card_mask, other=-1)
        card_shifted = tl.where(card_value >= 0, card_value - card_base, card_value)
        tl.store(
            dst_card_pos + card_row * dst_card_width + card_col,
            card_shifted,
            mask=card_mask,
        )

        option_mask_offsets = offsets < total_option
        option_b = offsets // dst_options
        option_col = offsets - option_b * dst_options
        option_row = tl.load(rows + option_b, mask=option_mask_offsets, other=0)
        option_in_src = option_col < src_options
        option_src_offsets = option_b * src_options + option_col
        option_base = tl.load(state_positions + option_b, mask=option_mask_offsets, other=0)
        option_value = tl.load(
            option_pos + option_src_offsets,
            mask=option_mask_offsets & option_in_src,
            other=-1,
        )
        option_valid = tl.load(
            option_mask + option_src_offsets,
            mask=option_mask_offsets & option_in_src,
            other=0,
        )
        option_shifted = tl.where(option_value >= 0, option_value - option_base, option_value)
        option_dst_offsets = option_row * dst_options + option_col
        tl.store(
            dst_option_pos + option_dst_offsets,
            tl.where(option_in_src, option_shifted, -1),
            mask=option_mask_offsets,
        )
        tl.store(
            dst_option_mask + option_dst_offsets,
            tl.where(option_in_src, option_valid, 0),
            mask=option_mask_offsets,
        )

        target_mask_offsets = offsets < total_target
        target_row_area = dst_options * dst_targets
        target_b = offsets // target_row_area
        target_in_row = offsets - target_b * target_row_area
        target_opt = target_in_row // dst_targets
        target_col = target_in_row - target_opt * dst_targets
        target_row = tl.load(rows + target_b, mask=target_mask_offsets, other=0)
        target_in_src = (target_opt < src_options) & (target_col < src_targets)
        target_src_offsets = (
            target_b * src_options * src_targets + target_opt * src_targets + target_col
        )
        target_base = tl.load(state_positions + target_b, mask=target_mask_offsets, other=0)
        target_value = tl.load(
            target_pos + target_src_offsets,
            mask=target_mask_offsets & target_in_src,
            other=-1,
        )
        target_valid = tl.load(
            target_mask + target_src_offsets,
            mask=target_mask_offsets & target_in_src,
            other=0,
        )
        target_shifted = tl.where(target_value >= 0, target_value - target_base, target_value)
        target_dst_offsets = target_row * target_row_area + target_opt * dst_targets + target_col
        tl.store(
            dst_target_pos + target_dst_offsets,
            tl.where(target_in_src, target_shifted, -1),
            mask=target_mask_offsets,
        )
        tl.store(
            dst_target_mask + target_dst_offsets,
            tl.where(target_in_src, target_valid, 0),
            mask=target_mask_offsets,
        )

    @triton.jit
    def _clear_decision_rows_kernel(
        rows,
        decision_option_idx,
        decision_target_idx,
        decision_mask,
        uses_none_head,
        selected_indices,
        batch_size: tl.constexpr,
        max_decision_groups: tl.constexpr,
        max_cached_choices: tl.constexpr,
        total_choice_slots: tl.constexpr,
        total_group_slots: tl.constexpr,
        max_total: tl.constexpr,
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
        batch_size: tl.constexpr,
        max_decision_groups: tl.constexpr,
        max_cached_choices: tl.constexpr,
        slots_per_row: tl.constexpr,
        choice_block: tl.constexpr,
        prefix_block: tl.constexpr,
    ):
        b = tl.program_id(0)
        row = tl.load(rows + b)
        prefix_offsets = tl.arange(0, prefix_block)
        prefix_mask = prefix_offsets < b
        prefix_counts = tl.load(decision_count_in + prefix_offsets, mask=prefix_mask, other=0)
        group_start = tl.sum(prefix_counts, axis=0)
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


def append_batch_encoded_triton(
    *,
    token_ids: Tensor,
    cu_seqlens: Tensor,
    seq_lengths: Tensor,
    state_positions: Tensor,
    card_ref_positions: Tensor,
    option_positions: Tensor,
    option_mask: Tensor,
    target_positions: Tensor,
    target_mask: Tensor,
    rows: Tensor,
    packed_token_ids: Tensor,
    row_token_start: Tensor,
    row_token_length: Tensor,
    dst_card_ref_positions: Tensor,
    dst_option_positions: Tensor,
    dst_option_mask: Tensor,
    dst_target_positions: Tensor,
    dst_target_mask: Tensor,
    dst_seq_lengths: Tensor,
    token_start: int,
) -> bool:
    """Write packed replay encoded fields with Triton when CUDA is available."""

    if not TRITON_AVAILABLE or not rows.is_cuda:
        return False

    batch_size = int(seq_lengths.shape[0])
    total_tokens = int(token_ids.numel())
    if batch_size == 0:
        return True

    inputs = (
        token_ids,
        cu_seqlens,
        seq_lengths,
        state_positions,
        card_ref_positions,
        option_positions,
        option_mask,
        target_positions,
        target_mask,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in inputs):
        return False

    block = 512
    if total_tokens > 0:
        _launch(
            _copy_packed_tokens_kernel,
            (_cdiv(total_tokens, block),),
            token_ids,
            packed_token_ids,
            token_start,
            total_tokens,
            block,
        )
    card_width = int(card_ref_positions.shape[1])
    src_options = int(option_positions.shape[1])
    dst_options = int(dst_option_positions.shape[1])
    src_targets = int(target_positions.shape[2])
    dst_targets = int(dst_target_positions.shape[2])
    total_card = batch_size * card_width
    total_option = batch_size * dst_options
    total_target = batch_size * dst_options * dst_targets
    max_total = max(batch_size, total_card, total_option, total_target)
    _launch(
        _write_rebased_fields_kernel,
        (_cdiv(max_total, block),),
        rows,
        cu_seqlens,
        seq_lengths,
        state_positions,
        card_ref_positions,
        option_positions,
        option_mask,
        target_positions,
        target_mask,
        row_token_start,
        row_token_length,
        dst_seq_lengths,
        dst_card_ref_positions,
        dst_option_positions,
        dst_option_mask,
        dst_target_positions,
        dst_target_mask,
        token_start,
        batch_size,
        card_width,
        int(dst_card_ref_positions.shape[1]),
        src_options,
        src_targets,
        dst_options,
        dst_targets,
        total_card,
        total_option,
        total_target,
        max_total,
        block,
    )
    return True


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
    prefix_block = _next_power_of_2(batch_size)
    if choice_block > 1024 or prefix_block > 1024:
        return False
    _launch(
        _write_decision_rows_kernel,
        (batch_size,),
        rows,
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
        batch_size,
        max_decision_groups,
        max_cached_choices,
        slots_per_row,
        choice_block,
        prefix_block,
    )
    return True


__all__ = [
    "TRITON_AVAILABLE",
    "append_batch_encoded_triton",
    "clear_append_decisions_triton",
    "write_append_decisions_triton",
]
