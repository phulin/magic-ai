"""Python wrapper for ``MageEncodeTokens``: native text-encoder assembler.

Phase 4 of the assembler-port. Allocates the token-output tensors,
binds them into the cffi struct, and calls into Go. The native side
walks the render-plan opcode stream emitted in the same call and writes
a dense ``(B, max_tokens)`` int64 token tensor + anchor arrays directly,
matching the Python ``_assemble_one`` walker byte-for-byte.

Requires :func:`magic_ai.text_encoder.native_token_tables.register_native_token_tables`
to have been called once at startup so the Go side has the closed-vocabulary
token tables to dispatch through.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, cast

import mage
import torch

from magic_ai.text_encoder.batch import TextEncodedBatch


@dataclass
class NativeAssemblerOutputs:
    """Allocated output tensors for one ``MageEncodeTokens`` call."""

    token_ids: torch.Tensor  # (B, max_tokens) int64
    attention_mask: torch.Tensor  # (B, max_tokens) int64
    seq_lengths: torch.Tensor  # (B,) int64
    option_positions: torch.Tensor  # (B, max_options) int64 (-1 absent)
    option_mask: torch.Tensor  # (B, max_options) uint8
    target_positions: torch.Tensor  # (B, max_options, max_targets) int64
    target_mask: torch.Tensor  # (B, max_options, max_targets) uint8
    card_ref_positions: torch.Tensor  # (B, max_card_refs) int64
    token_overflow: torch.Tensor  # (B,) int32

    def to_text_encoded_batch(self) -> TextEncodedBatch:
        """Slice/widen the dense outputs into a ``TextEncodedBatch`` matching
        what the Python assembler produces."""

        max_seq = int(self.seq_lengths.max().item()) if self.seq_lengths.numel() else 0
        opt_mask_b = self.option_mask.to(dtype=torch.bool)
        tgt_mask_b = self.target_mask.to(dtype=torch.bool)
        max_opts = int(opt_mask_b.any(dim=0).sum().item()) if opt_mask_b.numel() else 0
        # Compress max_targets to actual extent, like the Python assembler.
        max_tgts = int(tgt_mask_b.any(dim=0).any(dim=0).sum().item()) if tgt_mask_b.numel() else 0

        token_ids = self.token_ids[:, :max_seq] if max_seq > 0 else self.token_ids[:, :0]
        attention_mask = (
            self.attention_mask[:, :max_seq] if max_seq > 0 else self.attention_mask[:, :0]
        )
        option_positions = (
            self.option_positions[:, :max_opts] if max_opts > 0 else self.option_positions[:, :0]
        )
        option_mask = opt_mask_b[:, :max_opts] if max_opts > 0 else opt_mask_b[:, :0]
        target_positions = self.target_positions[:, :max_opts, :max_tgts]
        target_mask = tgt_mask_b[:, :max_opts, :max_tgts]
        return TextEncodedBatch(
            token_ids=token_ids,
            attention_mask=attention_mask,
            card_ref_positions=self.card_ref_positions,
            option_positions=option_positions,
            option_mask=option_mask,
            target_positions=target_positions,
            target_mask=target_mask,
            seq_lengths=self.seq_lengths,
        )


_OUTPUT_FIELDS_INT64: dict[str, ctypes.c_void_p] = {}


def allocate_outputs(
    batch_size: int,
    *,
    max_tokens: int,
    max_options: int,
    max_targets: int,
    max_card_refs: int,
) -> NativeAssemblerOutputs:
    """Pre-allocate output tensors for one MageEncodeTokens call."""

    return NativeAssemblerOutputs(
        token_ids=torch.empty((batch_size, max_tokens), dtype=torch.int64),
        attention_mask=torch.empty((batch_size, max_tokens), dtype=torch.int64),
        seq_lengths=torch.empty((batch_size,), dtype=torch.int64),
        option_positions=torch.full((batch_size, max_options), -1, dtype=torch.int64),
        option_mask=torch.zeros((batch_size, max_options), dtype=torch.uint8),
        target_positions=torch.full((batch_size, max_options, max_targets), -1, dtype=torch.int64),
        target_mask=torch.zeros((batch_size, max_options, max_targets), dtype=torch.uint8),
        card_ref_positions=torch.full((batch_size, max_card_refs), -1, dtype=torch.int64),
        token_overflow=torch.zeros((batch_size,), dtype=torch.int32),
    )


def encode_tokens(
    encoder: Any,
    games: list[Any],
    *,
    perspective_player_indices: list[int],
    max_tokens: int,
    max_options: int,
    max_targets: int,
    max_card_refs: int,
    outputs: NativeAssemblerOutputs | None = None,
) -> tuple[Any, NativeAssemblerOutputs]:
    """Run ``MageEncodeTokens`` for a batch of games.

    ``encoder`` must be a :class:`magic_ai.native_encoder.NativeBatchEncoder`
    constructed with ``emit_render_plan=True``. The native side fills both
    the existing encoder outputs (slot/option/target tensors etc.) and the
    new token outputs in a single FFI call.

    Returns the existing :class:`NativeEncodedBatch` *and* the freshly-
    populated :class:`NativeAssemblerOutputs`. Both reference Python-owned
    storage; the encoder's scratch buffers are reused across calls.
    """

    mage._ensure_loaded()
    ffi = mage._ffi
    lib = mage._lib

    batch_size = len(games)
    if batch_size == 0:
        raise ValueError("encode_tokens requires at least one game")
    if len(perspective_player_indices) != batch_size:
        raise ValueError("perspective_player_indices length mismatch")

    if outputs is None:
        outputs = allocate_outputs(
            batch_size,
            max_tokens=max_tokens,
            max_options=max_options,
            max_targets=max_targets,
            max_card_refs=max_card_refs,
        )

    # Fill encoder scratch + emit render plan as usual; that gives us the
    # populated MageEncodeOutputs struct. We bypass encoder.encode_handles
    # and call MageEncodeTokens with both struct pointers in one shot, so
    # the encoder doesn't redo work the native side will do anyway.
    decision_capacity = max(1, batch_size * encoder.max_options)
    buffers = encoder._scratch_buffers(batch_size, decision_capacity)
    scratch = encoder._scratch
    handles_t = scratch.handles_t[:batch_size]
    perspectives_t = scratch.perspectives_t[:batch_size]
    handles_np = scratch.handles_np
    for i, g in enumerate(games):
        handles_np[i] = getattr(g, "_id", None) or g.handle
    scratch.perspectives_np[:batch_size] = perspective_player_indices

    req = ffi.new(
        "MageBatchRequest *",
        {
            "n": batch_size,
            "handles": ffi.cast("int64_t *", handles_t.data_ptr()),
            "perspective_player_idx": ffi.cast("int64_t *", perspectives_t.data_ptr()),
        },
    )
    cfg = ffi.new(
        "MageEncodeConfig *",
        {
            "max_options": encoder.max_options,
            "max_targets_per_option": encoder.max_targets_per_option,
            "max_cached_choices": encoder.max_cached_choices,
            "zone_slot_count": cast(int, encoder.zone_slot_count),
            "game_info_dim": cast(int, encoder.game_info_dim),
            "option_scalar_dim": cast(int, encoder.option_scalar_dim),
            "target_scalar_dim": cast(int, encoder.target_scalar_dim),
            "decision_capacity": decision_capacity,
            "emit_render_plan": 1,
            "render_plan_capacity": encoder.render_plan_capacity,
        },
    )

    # Existing-encoder outputs.
    enc_out_fields: dict[str, Any] = {
        "trace_kind_id": ffi.cast("int64_t *", buffers.trace_kind_id.data_ptr()),
        "slot_card_rows": ffi.cast("int64_t *", buffers.slot_card_rows.data_ptr()),
        "slot_occupied": ffi.cast("float *", buffers.slot_occupied.data_ptr()),
        "slot_tapped": ffi.cast("float *", buffers.slot_tapped.data_ptr()),
        "game_info": ffi.cast("float *", buffers.game_info.data_ptr()),
        "pending_kind_id": ffi.cast("int64_t *", buffers.pending_kind_id.data_ptr()),
        "num_present_options": ffi.cast("int64_t *", buffers.num_present_options.data_ptr()),
        "option_kind_ids": ffi.cast("int64_t *", buffers.option_kind_ids.data_ptr()),
        "option_scalars": ffi.cast("float *", buffers.option_scalars.data_ptr()),
        "option_mask": ffi.cast("float *", buffers.option_mask.data_ptr()),
        "option_ref_slot_idx": ffi.cast("int64_t *", buffers.option_ref_slot_idx.data_ptr()),
        "option_ref_card_row": ffi.cast("int64_t *", buffers.option_ref_card_row.data_ptr()),
        "target_mask": ffi.cast("float *", buffers.target_mask.data_ptr()),
        "target_type_ids": ffi.cast("int64_t *", buffers.target_type_ids.data_ptr()),
        "target_scalars": ffi.cast("float *", buffers.target_scalars.data_ptr()),
        "target_overflow": ffi.cast("float *", buffers.target_overflow.data_ptr()),
        "target_ref_slot_idx": ffi.cast("int64_t *", buffers.target_ref_slot_idx.data_ptr()),
        "target_ref_is_player": ffi.cast("uint8_t *", buffers.target_ref_is_player_u8.data_ptr()),
        "target_ref_is_self": ffi.cast("uint8_t *", buffers.target_ref_is_self_u8.data_ptr()),
        "may_mask": ffi.cast("uint8_t *", buffers.may_mask_u8.data_ptr()),
        "decision_start": ffi.cast("int64_t *", buffers.decision_start.data_ptr()),
        "decision_count": ffi.cast("int64_t *", buffers.decision_count.data_ptr()),
        "decision_option_idx": ffi.cast("int64_t *", buffers.decision_option_idx.data_ptr()),
        "decision_target_idx": ffi.cast("int64_t *", buffers.decision_target_idx.data_ptr()),
        "decision_mask": ffi.cast("uint8_t *", buffers.decision_mask_u8.data_ptr()),
        "uses_none_head": ffi.cast("uint8_t *", buffers.uses_none_head_u8.data_ptr()),
        "render_plan": ffi.cast("int32_t *", buffers.render_plan.data_ptr()),
        "render_plan_lengths": ffi.cast("int64_t *", buffers.render_plan_lengths.data_ptr()),
        "render_plan_overflow": ffi.cast("int64_t *", buffers.render_plan_overflow.data_ptr()),
    }
    enc_out = ffi.new("MageEncodeOutputs *", enc_out_fields)

    tok_cfg = ffi.new(
        "MageTokenAssemblerConfig *",
        {
            "max_tokens": max_tokens,
            "max_options": max_options,
            "max_targets": max_targets,
            "max_card_refs": max_card_refs,
        },
    )
    tok_out = ffi.new(
        "MageTokenAssemblerOutputs *",
        {
            "token_ids": ffi.cast("int64_t *", outputs.token_ids.data_ptr()),
            "attention_mask": ffi.cast("int64_t *", outputs.attention_mask.data_ptr()),
            "seq_lengths": ffi.cast("int64_t *", outputs.seq_lengths.data_ptr()),
            "option_positions": ffi.cast("int64_t *", outputs.option_positions.data_ptr()),
            "option_mask": ffi.cast("uint8_t *", outputs.option_mask.data_ptr()),
            "target_positions": ffi.cast("int64_t *", outputs.target_positions.data_ptr()),
            "target_mask": ffi.cast("uint8_t *", outputs.target_mask.data_ptr()),
            "card_ref_positions": ffi.cast("int64_t *", outputs.card_ref_positions.data_ptr()),
            "token_overflow": ffi.cast("int32_t *", outputs.token_overflow.data_ptr()),
        },
    )

    result = lib.MageEncodeTokens(req, cfg, enc_out, tok_cfg, tok_out)
    if result.error_code != 0:
        message = "MageEncodeTokens failed"
        if result.error_message != ffi.NULL:
            try:
                message = ffi.string(result.error_message).decode("utf-8")
            finally:
                lib.MageFreeString(result.error_message)
        from magic_ai.native_encoder import NativeEncodingError

        raise NativeEncodingError(message)

    # Build the existing NativeEncodedBatch the same way encode_handles
    # would. We rely on the encoder's slice helper rather than duplicating
    # field assembly.
    decision_rows_written = int(result.decision_rows_written)
    batch = encoder._slice_batch_buffers(buffers, batch_size)
    from magic_ai.native_encoder import (
        TRACE_KIND_VALUES,
        NativeEncodedBatch,
    )

    trace_kind_id = batch["trace_kind_id"]
    trace_kinds = [TRACE_KIND_VALUES[int(idx)] for idx in trace_kind_id.tolist()]
    decision_option_idx = buffers.decision_option_idx[:decision_rows_written]
    decision_target_idx = buffers.decision_target_idx[:decision_rows_written]
    decision_mask = buffers.decision_mask_u8[:decision_rows_written]
    uses_none_head = buffers.uses_none_head_u8[:decision_rows_written]
    native_batch = NativeEncodedBatch(
        trace_kind_id=trace_kind_id,
        slot_card_rows=batch["slot_card_rows"],
        slot_occupied=batch["slot_occupied"],
        slot_tapped=batch["slot_tapped"],
        game_info=batch["game_info"],
        pending_kind_id=batch["pending_kind_id"],
        num_present_options=batch["num_present_options"],
        option_kind_ids=batch["option_kind_ids"],
        option_scalars=batch["option_scalars"],
        option_mask=batch["option_mask"],
        option_ref_slot_idx=batch["option_ref_slot_idx"],
        option_ref_card_row=batch["option_ref_card_row"],
        target_mask=batch["target_mask"],
        target_type_ids=batch["target_type_ids"],
        target_scalars=batch["target_scalars"],
        target_overflow=batch["target_overflow"],
        target_ref_slot_idx=batch["target_ref_slot_idx"],
        target_ref_is_player=batch["target_ref_is_player"],
        target_ref_is_self=batch["target_ref_is_self"],
        may_mask=batch["may_mask"],
        decision_start=batch["decision_start"],
        decision_count=batch["decision_count"],
        decision_option_idx=decision_option_idx,
        decision_target_idx=decision_target_idx,
        decision_mask=decision_mask,
        uses_none_head=uses_none_head,
        decision_rows_written=decision_rows_written,
        pendings=[],
        trace_kinds=trace_kinds,
        render_plan=batch.get("render_plan"),
        render_plan_lengths=batch.get("render_plan_lengths"),
        render_plan_overflow=batch.get("render_plan_overflow"),
    )
    return native_batch, outputs


__all__ = [
    "NativeAssemblerOutputs",
    "allocate_outputs",
    "encode_tokens",
]
