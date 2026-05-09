"""Python wrapper for ``MageEncodeTokensPacked`` + ``MageEncodeDecisionSpec``.

Allocates packed-token and decision-spec output tensors, binds them into
ctypes/cffi structs, and calls into Go. The native side writes a flat live-
token region, absolute anchor offsets, plus per-row decision-spec tokens and
pointer-anchor side tensors used by the decoder grammar mask.

Requires :func:`magic_ai.text_encoder.native_token_tables.register_native_token_tables`
to have been called once at startup so the Go side has the closed-vocabulary
token tables to dispatch through; the spec-tag id table is registered as part
of :func:`register_native_token_tables` here (it now takes a tokenizer).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import mage
import torch


class _MageTokenAssemblerConfigC(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_int32),
        ("max_options", ctypes.c_int32),
        ("max_targets", ctypes.c_int32),
        ("max_card_refs", ctypes.c_int32),
    ]


class _MagePackedTokenAssemblerOutputsC(ctypes.Structure):
    _fields_ = [
        ("token_ids", ctypes.POINTER(ctypes.c_int32)),
        ("cu_seqlens", ctypes.POINTER(ctypes.c_int32)),
        ("seq_lengths", ctypes.POINTER(ctypes.c_int32)),
        ("state_positions", ctypes.POINTER(ctypes.c_int32)),
        ("card_ref_positions", ctypes.POINTER(ctypes.c_int32)),
        ("token_overflow", ctypes.POINTER(ctypes.c_int32)),
    ]


class _MagePackedSpecOutputsC(ctypes.Structure):
    """Mirror of ``MagePackedSpecOutputs`` in mage-go/cmd/pylib/abi.h.

    Field order must match the C struct exactly.
    """

    _fields_ = [
        ("spec_tokens", ctypes.POINTER(ctypes.c_int32)),
        ("spec_lens", ctypes.POINTER(ctypes.c_int32)),
        ("decision_type", ctypes.POINTER(ctypes.c_int32)),
        ("pointer_anchor_positions", ctypes.POINTER(ctypes.c_int32)),
        ("pointer_anchor_kinds", ctypes.POINTER(ctypes.c_int32)),
        ("pointer_anchor_subjects", ctypes.POINTER(ctypes.c_int32)),
        ("pointer_anchor_handles", ctypes.POINTER(ctypes.c_int32)),
        ("pointer_anchor_counts", ctypes.POINTER(ctypes.c_int32)),
        ("legal_edge_bitmap", ctypes.POINTER(ctypes.c_uint8)),
        ("legal_edge_n_blockers", ctypes.POINTER(ctypes.c_int32)),
        ("legal_edge_n_attackers", ctypes.POINTER(ctypes.c_int32)),
        ("T_spec_max", ctypes.c_int32),
        ("N_anchors_max", ctypes.c_int32),
        ("N_blockers_max", ctypes.c_int32),
        ("N_attackers_max", ctypes.c_int32),
        ("spec_overflow", ctypes.c_int32),
    ]


# Default per-row caps. Tuned for the worst-case decision the decoder needs
# to emit — large enough that ``spec_overflow`` should stay zero on real
# states. Small enough that the per-batch allocation is a few MB.
DEFAULT_T_SPEC_MAX: int = 128
DEFAULT_N_ANCHORS_MAX: int = 64
DEFAULT_N_BLOCKERS_MAX: int = 16
DEFAULT_N_ATTACKERS_MAX: int = 16


_packed_ctypes_lib: ctypes.CDLL | None = None


def _tensor_ptr(tensor: torch.Tensor, ctype: Any) -> Any:
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctype))


def _load_packed_ctypes_lib() -> ctypes.CDLL | None:
    """Load MageEncodeTokensPacked + MageEncodeDecisionSpec via ctypes so
    the native work runs without holding the GIL."""

    global _packed_ctypes_lib
    if _packed_ctypes_lib is not None:
        return _packed_ctypes_lib

    from magic_ai.slot_encoder.native_encoder import (
        _load_encode_ctypes_lib,
        _MageBatchRequest,
        _MageEncodeConfig,
        _MageEncodeOutputs,
        _MageEncodeResult,
    )

    lib = _load_encode_ctypes_lib()
    if lib is None:
        return None
    try:
        lib.MageEncodeTokensPacked.argtypes = [
            ctypes.POINTER(_MageBatchRequest),
            ctypes.POINTER(_MageEncodeConfig),
            ctypes.POINTER(_MageEncodeOutputs),
            ctypes.POINTER(_MageTokenAssemblerConfigC),
            ctypes.POINTER(_MagePackedTokenAssemblerOutputsC),
        ]
        lib.MageEncodeTokensPacked.restype = _MageEncodeResult
        lib.MageEncodeDecisionSpec.argtypes = [
            ctypes.POINTER(_MageBatchRequest),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(_MagePackedSpecOutputsC),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.MageEncodeDecisionSpec.restype = _MageEncodeResult
        lib.MageReleaseBatchHandle.argtypes = [ctypes.c_int64]
        lib.MageReleaseBatchHandle.restype = None
        lib.MageFreeString.argtypes = [ctypes.c_void_p]
        lib.MageFreeString.restype = None
    except AttributeError:
        return None
    _packed_ctypes_lib = lib
    return lib


@dataclass
class NativePackedAssemblerOutputs:
    """Pre-allocated buffers for one ``encode_tokens_packed`` call.

    Holds both the packed-token outputs (``token_ids`` + anchors) and the
    decision-spec outputs (per-row spec tokens + pointer anchors + legal-edge
    bitmap), plus the ``batch_handle`` for the most recent
    ``MageEncodeDecisionSpec`` call. Token-shaped buffers are sized at the
    worst case ``B * max_tokens``; spec buffers are sized at per-row caps.
    """

    # Packed-token outputs (one per row, varlen via cu_seqlens).
    token_ids: torch.Tensor  # (B*max_tokens,) int32
    cu_seqlens: torch.Tensor  # (B+1,) int32
    seq_lengths: torch.Tensor  # (B,) int32
    state_positions: torch.Tensor  # (B,) int32
    card_ref_positions: torch.Tensor  # (B, max_card_refs) int32
    token_overflow: torch.Tensor  # (B,) int32

    # Decision-spec outputs.
    spec_tokens: torch.Tensor  # (B, T_spec_max) int32, 0 = pad
    spec_lens: torch.Tensor  # (B,) int32
    decision_type: torch.Tensor  # (B,) int32, -1 = no pending
    pointer_anchor_positions: torch.Tensor  # (B, N_anchors_max) int32
    pointer_anchor_kinds: torch.Tensor  # (B, N_anchors_max) int32
    pointer_anchor_subjects: torch.Tensor  # (B, N_anchors_max) int32
    pointer_anchor_handles: torch.Tensor  # (B, N_anchors_max) int32
    pointer_anchor_counts: torch.Tensor  # (B,) int32
    legal_edge_bitmap: torch.Tensor  # (B, N_blockers_max, N_attackers_max) uint8
    legal_edge_n_blockers: torch.Tensor  # (B,) int32
    legal_edge_n_attackers: torch.Tensor  # (B,) int32
    spec_overflow: torch.Tensor  # (B,) int32 — always length 1 in current ABI

    # Batch handle from the most recent MageEncodeDecisionSpec call. Released
    # via :func:`release_batch_handle` once the decoder is done with this batch.
    batch_handle: int = 0
    active_batch_size: int = 0

    # Cached cffi/ctypes structs so we don't re-allocate on every call.
    _packed_out_cffi: Any = None
    _tok_cfg_cffi: Any = None
    _spec_out_cffi: Any = None
    _packed_out_ctypes: _MagePackedTokenAssemblerOutputsC | None = None
    _tok_cfg_ctypes: _MageTokenAssemblerConfigC | None = None
    _spec_out_ctypes: _MagePackedSpecOutputsC | None = None


def allocate_packed_outputs(
    batch_size: int,
    *,
    max_tokens: int,
    max_options: int,
    max_targets: int,
    max_card_refs: int,
    t_spec_max: int = DEFAULT_T_SPEC_MAX,
    n_anchors_max: int = DEFAULT_N_ANCHORS_MAX,
    n_blockers_max: int = DEFAULT_N_BLOCKERS_MAX,
    n_attackers_max: int = DEFAULT_N_ATTACKERS_MAX,
) -> NativePackedAssemblerOutputs:
    """Allocate all output buffers for one batch.

    The ``max_options`` / ``max_targets`` parameters are accepted for caller
    compatibility but only ``max_tokens`` and ``max_card_refs`` size the
    packed-token buffers; option/target anchors live inside the encoder's
    own scratch and aren't surfaced here.
    """

    _ = max_options, max_targets

    pin = torch.cuda.is_available()
    cap = batch_size * max_tokens
    return NativePackedAssemblerOutputs(
        token_ids=torch.empty((cap,), dtype=torch.int32, pin_memory=pin),
        cu_seqlens=torch.zeros((batch_size + 1,), dtype=torch.int32, pin_memory=pin),
        seq_lengths=torch.zeros((batch_size,), dtype=torch.int32, pin_memory=pin),
        state_positions=torch.zeros((batch_size,), dtype=torch.int32, pin_memory=pin),
        card_ref_positions=torch.full(
            (batch_size, max_card_refs), -1, dtype=torch.int32, pin_memory=pin
        ),
        token_overflow=torch.zeros((batch_size,), dtype=torch.int32),
        spec_tokens=torch.zeros((batch_size, t_spec_max), dtype=torch.int32, pin_memory=pin),
        spec_lens=torch.zeros((batch_size,), dtype=torch.int32, pin_memory=pin),
        decision_type=torch.full((batch_size,), -1, dtype=torch.int32, pin_memory=pin),
        pointer_anchor_positions=torch.full(
            (batch_size, n_anchors_max), -1, dtype=torch.int32, pin_memory=pin
        ),
        pointer_anchor_kinds=torch.full(
            (batch_size, n_anchors_max), -1, dtype=torch.int32, pin_memory=pin
        ),
        pointer_anchor_subjects=torch.zeros(
            (batch_size, n_anchors_max), dtype=torch.int32, pin_memory=pin
        ),
        pointer_anchor_handles=torch.full(
            (batch_size, n_anchors_max), -1, dtype=torch.int32, pin_memory=pin
        ),
        pointer_anchor_counts=torch.zeros((batch_size,), dtype=torch.int32, pin_memory=pin),
        legal_edge_bitmap=torch.zeros(
            (batch_size, n_blockers_max, n_attackers_max), dtype=torch.uint8, pin_memory=pin
        ),
        legal_edge_n_blockers=torch.zeros((batch_size,), dtype=torch.int32, pin_memory=pin),
        legal_edge_n_attackers=torch.zeros((batch_size,), dtype=torch.int32, pin_memory=pin),
        spec_overflow=torch.zeros((1,), dtype=torch.int32),
    )


def release_batch_handle(batch_handle: int) -> None:
    """Drop a batch handle previously stored on a ``NativePackedAssemblerOutputs``.

    Thin wrapper over ``mage.release_batch_handle``. Calling with handle 0
    (the sentinel for "no batch") is a no-op.
    """

    if batch_handle == 0:
        return
    mage.release_batch_handle(batch_handle)


def encode_tokens_packed(
    encoder: Any,
    games: list[Any],
    *,
    perspective_player_indices: list[int],
    max_tokens: int,
    max_options: int,
    max_targets: int,
    max_card_refs: int,
    t_spec_max: int = DEFAULT_T_SPEC_MAX,
    n_anchors_max: int = DEFAULT_N_ANCHORS_MAX,
    n_blockers_max: int = DEFAULT_N_BLOCKERS_MAX,
    n_attackers_max: int = DEFAULT_N_ATTACKERS_MAX,
    outputs: NativePackedAssemblerOutputs | None = None,
    include_trace_kinds: bool = True,
) -> tuple[Any, NativePackedAssemblerOutputs]:
    """Run ``MageEncodeTokensPacked`` followed by ``MageEncodeDecisionSpec``.

    Two native passes feed the same output object:

      1. ``MageEncodeTokensPacked`` writes the per-row state-text token
         stream (``token_ids`` + ``cu_seqlens`` + anchor positions) plus the
         standard slot/option encoder outputs.
      2. ``MageEncodeDecisionSpec`` reads each row's pending decision and
         emits the per-row spec-token stream + pointer-anchor table + the
         legal-edge bitmap (for DECLARE_BLOCKERS), shifting all anchor
         positions by the row's state-text length so they reference the
         combined ``[state_tokens] + [spec_tokens]`` stream.

    Returns ``(native_batch, outputs)``. ``outputs.batch_handle`` references
    the spec-side state on the Go side and must be released via
    :func:`release_batch_handle` once the decoder finishes this batch.
    """

    mage._ensure_loaded()
    ffi = mage._ffi
    lib = mage._lib

    batch_size = len(games)
    if batch_size == 0:
        raise ValueError("encode_tokens_packed requires at least one game")
    if len(perspective_player_indices) != batch_size:
        raise ValueError("perspective_player_indices length mismatch")

    if outputs is None:
        outputs = allocate_packed_outputs(
            batch_size,
            max_tokens=max_tokens,
            max_options=max_options,
            max_targets=max_targets,
            max_card_refs=max_card_refs,
            t_spec_max=t_spec_max,
            n_anchors_max=n_anchors_max,
            n_blockers_max=n_blockers_max,
            n_attackers_max=n_attackers_max,
        )
    outputs.active_batch_size = batch_size

    decision_capacity = max(1, batch_size * encoder.max_options)
    encoder._scratch_buffers(batch_size, decision_capacity)
    scratch = encoder._scratch
    if scratch.req_cffi is None or scratch.cfg_cffi is None or scratch.enc_out_cffi is None:
        encoder.rebuild_cffi_structs(ffi, decision_capacity)
    req = scratch.req_cffi
    cfg = scratch.cfg_cffi
    enc_out = scratch.enc_out_cffi
    req.n = batch_size
    cfg.decision_capacity = decision_capacity
    cfg.emit_render_plan = 0
    cfg.render_plan_capacity = int(encoder.render_plan_capacity)

    handles_np = scratch.handles_np
    for i, g in enumerate(games):
        handles_np[i] = getattr(g, "_id", None) or g.handle
    scratch.perspectives_np[:batch_size] = perspective_player_indices

    if outputs._tok_cfg_cffi is None:
        outputs._tok_cfg_cffi = ffi.new(
            "MageTokenAssemblerConfig *",
            {
                "max_tokens": max_tokens,
                "max_options": max_options,
                "max_targets": max_targets,
                "max_card_refs": max_card_refs,
            },
        )
    if outputs._packed_out_cffi is None:
        outputs._packed_out_cffi = ffi.new(
            "MagePackedTokenAssemblerOutputs *",
            {
                "token_ids": ffi.cast("int32_t *", outputs.token_ids.data_ptr()),
                "cu_seqlens": ffi.cast("int32_t *", outputs.cu_seqlens.data_ptr()),
                "seq_lengths": ffi.cast("int32_t *", outputs.seq_lengths.data_ptr()),
                "state_positions": ffi.cast("int32_t *", outputs.state_positions.data_ptr()),
                "card_ref_positions": ffi.cast("int32_t *", outputs.card_ref_positions.data_ptr()),
                "token_overflow": ffi.cast("int32_t *", outputs.token_overflow.data_ptr()),
            },
        )
    tok_cfg = outputs._tok_cfg_cffi
    packed_out = outputs._packed_out_cffi

    if outputs._spec_out_cffi is None:
        outputs._spec_out_cffi = ffi.new(
            "MagePackedSpecOutputs *",
            {
                "spec_tokens": ffi.cast("int32_t *", outputs.spec_tokens.data_ptr()),
                "spec_lens": ffi.cast("int32_t *", outputs.spec_lens.data_ptr()),
                "decision_type": ffi.cast("int32_t *", outputs.decision_type.data_ptr()),
                "pointer_anchor_positions": ffi.cast(
                    "int32_t *", outputs.pointer_anchor_positions.data_ptr()
                ),
                "pointer_anchor_kinds": ffi.cast(
                    "int32_t *", outputs.pointer_anchor_kinds.data_ptr()
                ),
                "pointer_anchor_subjects": ffi.cast(
                    "int32_t *", outputs.pointer_anchor_subjects.data_ptr()
                ),
                "pointer_anchor_handles": ffi.cast(
                    "int32_t *", outputs.pointer_anchor_handles.data_ptr()
                ),
                "pointer_anchor_counts": ffi.cast(
                    "int32_t *", outputs.pointer_anchor_counts.data_ptr()
                ),
                "legal_edge_bitmap": ffi.cast("uint8_t *", outputs.legal_edge_bitmap.data_ptr()),
                "legal_edge_n_blockers": ffi.cast(
                    "int32_t *", outputs.legal_edge_n_blockers.data_ptr()
                ),
                "legal_edge_n_attackers": ffi.cast(
                    "int32_t *", outputs.legal_edge_n_attackers.data_ptr()
                ),
                "T_spec_max": int(outputs.spec_tokens.shape[1]),
                "N_anchors_max": int(outputs.pointer_anchor_positions.shape[1]),
                "N_blockers_max": int(outputs.legal_edge_bitmap.shape[1]),
                "N_attackers_max": int(outputs.legal_edge_bitmap.shape[2]),
                "spec_overflow": 0,
            },
        )
    spec_out = outputs._spec_out_cffi

    buffers = scratch.buffers
    assert buffers is not None

    # First pass: state-text tokens.
    result = lib.MageEncodeTokensPacked(req, cfg, enc_out, tok_cfg, packed_out)
    if result.error_code != 0:
        message = "MageEncodeTokensPacked failed"
        if result.error_message != ffi.NULL:
            try:
                message = ffi.string(result.error_message).decode("utf-8")
            finally:
                lib.MageFreeString(result.error_message)
        from magic_ai.slot_encoder.native_encoder import NativeEncodingError

        raise NativeEncodingError(message)

    # Second pass: decision-spec tokens. Pointer anchors are shifted by the
    # per-row state-text length so they reference the combined stream.
    state_lens_buf = ffi.cast("int32_t *", outputs.seq_lengths.data_ptr())
    handles_buf = ffi.cast("int64_t *", handles_np.ctypes.data)
    spec_req = ffi.new("MageBatchRequest *")
    spec_req.n = batch_size
    spec_req.handles = handles_buf
    spec_req.perspective_player_idx = ffi.NULL
    handle_out = ffi.new("int64_t *")
    spec_result = lib.MageEncodeDecisionSpec(spec_req, state_lens_buf, spec_out, handle_out)
    if spec_result.error_code != 0:
        message = "MageEncodeDecisionSpec failed"
        if spec_result.error_message != ffi.NULL:
            try:
                message = ffi.string(spec_result.error_message).decode("utf-8")
            finally:
                lib.MageFreeString(spec_result.error_message)
        from magic_ai.slot_encoder.native_encoder import NativeEncodingError

        raise NativeEncodingError(message)
    outputs.batch_handle = int(handle_out[0])
    outputs.spec_overflow[0] = int(spec_out.spec_overflow)
    if int(spec_out.spec_overflow) != 0:
        from magic_ai.slot_encoder.native_encoder import NativeEncodingError

        raise NativeEncodingError(
            f"MageEncodeDecisionSpec spec_overflow={int(spec_out.spec_overflow)}: "
            f"per-row spec exceeded T_spec_max={int(outputs.spec_tokens.shape[1])} "
            f"or N_anchors_max={int(outputs.pointer_anchor_positions.shape[1])}"
        )

    decision_rows_written = int(result.decision_rows_written)
    batch = encoder._slice_batch_buffers(buffers, batch_size)
    from magic_ai.actions import TRACE_KIND_VALUES
    from magic_ai.slot_encoder.native_encoder import NativeEncodedBatch

    trace_kind_id = batch["trace_kind_id"]
    trace_kinds: list[str] = (
        [TRACE_KIND_VALUES[int(idx)] for idx in trace_kind_id.tolist()]
        if include_trace_kinds
        else []
    )
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
        render_plan=None,
        render_plan_lengths=None,
        render_plan_overflow=None,
    )
    return native_batch, outputs


__all__ = [
    "DEFAULT_N_ANCHORS_MAX",
    "DEFAULT_N_ATTACKERS_MAX",
    "DEFAULT_N_BLOCKERS_MAX",
    "DEFAULT_T_SPEC_MAX",
    "NativePackedAssemblerOutputs",
    "allocate_packed_outputs",
    "encode_tokens_packed",
    "release_batch_handle",
]
