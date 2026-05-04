"""Python wrapper for ``MageEncodeTokensPacked``: native text-encoder assembler.

Allocates packed token-output tensors, binds them into cffi/ctypes structs,
and calls into Go. The native side writes a flat live-token region plus
absolute anchor offsets for the packed text encoder path.

Requires :func:`magic_ai.text_encoder.native_token_tables.register_native_token_tables`
to have been called once at startup so the Go side has the closed-vocabulary
token tables to dispatch through.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import mage
import torch

from magic_ai.text_encoder.batch import PackedTextBatch


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
        ("option_positions", ctypes.POINTER(ctypes.c_int32)),
        ("option_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("target_positions", ctypes.POINTER(ctypes.c_int32)),
        ("target_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("card_ref_positions", ctypes.POINTER(ctypes.c_int32)),
        ("token_overflow", ctypes.POINTER(ctypes.c_int32)),
    ]


class _MageBlankAssemblerConfigC(ctypes.Structure):
    _fields_ = [
        ("max_blanks", ctypes.c_int32),
        ("max_legal_per_blank", ctypes.c_int32),
    ]


class _MagePackedBlankOutputsC(ctypes.Structure):
    _fields_ = [
        ("k_max", ctypes.c_int32),
        ("v_max", ctypes.c_int32),
        ("blank_positions", ctypes.POINTER(ctypes.c_int32)),
        ("blank_kind", ctypes.POINTER(ctypes.c_int32)),
        ("blank_group", ctypes.POINTER(ctypes.c_int32)),
        ("blank_group_kind", ctypes.POINTER(ctypes.c_int32)),
        ("blank_legal_ids", ctypes.POINTER(ctypes.c_int32)),
        ("blank_legal_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("blank_overflow", ctypes.POINTER(ctypes.c_int32)),
    ]


_packed_ctypes_lib: ctypes.CDLL | None = None


def _tensor_ptr(tensor: torch.Tensor, ctype: Any) -> Any:
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctype))


def _load_packed_ctypes_lib() -> ctypes.CDLL | None:
    """Load MageEncodeTokensPacked through ctypes so native work releases the GIL."""

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
            ctypes.POINTER(_MageBlankAssemblerConfigC),
            ctypes.POINTER(_MagePackedBlankOutputsC),
        ]
        lib.MageEncodeTokensPacked.restype = _MageEncodeResult
        lib.MageFreeString.argtypes = [ctypes.c_void_p]
        lib.MageFreeString.restype = None
    except AttributeError:
        return None
    _packed_ctypes_lib = lib
    return lib


@dataclass
class NativePackedAssemblerOutputs:
    """Pre-allocated buffers for one ``MageEncodeTokensPacked`` call.

    Token-shaped buffers are sized at the worst case ``B * max_tokens``
    so a single allocation can accept any live-token total. The Go side
    writes ``cu_seqlens[B]`` to indicate the live region; trailing
    storage is unspecified.
    """

    token_ids: torch.Tensor  # (B*max_tokens,) int32
    seq_id: torch.Tensor  # (B*max_tokens,) int32
    pos_in_seq: torch.Tensor  # (B*max_tokens,) int32
    cu_seqlens: torch.Tensor  # (B+1,) int32
    seq_lengths: torch.Tensor  # (B,) int32
    state_positions: torch.Tensor  # (B,) int32
    option_positions: torch.Tensor  # (B, max_options) int32 (-1 absent)
    option_mask: torch.Tensor  # (B, max_options) uint8
    target_positions: torch.Tensor  # (B, max_options, max_targets) int32
    target_mask: torch.Tensor  # (B, max_options, max_targets) uint8
    card_ref_positions: torch.Tensor  # (B, max_card_refs) int32
    token_overflow: torch.Tensor  # (B,) int32
    blank_positions: torch.Tensor  # (B, max_blanks) int32
    blank_kind: torch.Tensor  # (B, max_blanks) int32
    blank_group: torch.Tensor  # (B, max_blanks) int32
    blank_group_kind: torch.Tensor  # (B, max_blanks) int32
    blank_legal_ids: torch.Tensor  # (B, max_blanks, max_legal_per_blank) int32
    blank_legal_mask: torch.Tensor  # (B, max_blanks, max_legal_per_blank) uint8
    blank_overflow: torch.Tensor  # (B,) int32
    active_batch_size: int = 0
    _packed_out_cffi: Any = None
    _tok_cfg_cffi: Any = None
    _blank_cfg_cffi: Any = None
    _blank_out_cffi: Any = None
    _packed_out_ctypes: _MagePackedTokenAssemblerOutputsC | None = None
    _tok_cfg_ctypes: _MageTokenAssemblerConfigC | None = None
    _blank_cfg_ctypes: _MageBlankAssemblerConfigC | None = None
    _blank_out_ctypes: _MagePackedBlankOutputsC | None = None

    def to_packed_text_batch(
        self, *, trim: bool = True, derive_token_metadata: bool = True
    ) -> PackedTextBatch:
        """Slice the live region into a :class:`PackedTextBatch`.

        Trims anchor tensors to per-batch maxima so the result matches
        ``pack_batch(assemble_batch(...))`` shape-for-shape, rather than the
        full pre-allocated capacity.
        """

        active_n = self.active_batch_size or int(self.seq_lengths.shape[0])
        cu_seqlens = self.cu_seqlens[: active_n + 1]
        seq_lengths = self.seq_lengths[:active_n]
        state_positions = self.state_positions[:active_n]
        card_ref_positions = self.card_ref_positions[:active_n]
        option_positions_full = self.option_positions[:active_n]
        option_mask_full = self.option_mask[:active_n]
        target_positions_full = self.target_positions[:active_n]
        target_mask_full = self.target_mask[:active_n]
        blank_positions_full = self.blank_positions[:active_n]
        blank_kind_full = self.blank_kind[:active_n]
        blank_group_full = self.blank_group[:active_n]
        blank_group_kind_full = self.blank_group_kind[:active_n]
        blank_legal_ids_full = self.blank_legal_ids[:active_n]
        blank_legal_mask_full = self.blank_legal_mask[:active_n]

        total = int(cu_seqlens[-1].item()) if cu_seqlens.numel() else 0
        token_ids = self.token_ids[:total]
        if derive_token_metadata:
            seq_id = torch.repeat_interleave(
                torch.arange(active_n, dtype=torch.int32, device=seq_lengths.device),
                seq_lengths,
            )
            pos_in_seq = torch.arange(total, dtype=torch.int32, device=seq_lengths.device) - (
                cu_seqlens[:-1].repeat_interleave(seq_lengths)
            )
        else:
            seq_id = self.token_ids[:0]
            pos_in_seq = self.token_ids[:0]

        if trim:
            opt_any = option_mask_full.any(dim=0)
            max_opts = int(opt_any.sum().item()) if opt_any.numel() else 0
            if max_opts > 0:
                tgt_any = target_mask_full[:, :max_opts].any(dim=0).any(dim=0)
                max_tgts = int(tgt_any.sum().item()) if tgt_any.numel() else 0
            else:
                max_tgts = 0
            option_positions = (
                option_positions_full[:, :max_opts]
                if max_opts > 0
                else option_positions_full[:, :0]
            )
            option_mask = (
                option_mask_full[:, :max_opts] if max_opts > 0 else option_mask_full[:, :0]
            )
            target_positions = target_positions_full[:, :max_opts, :max_tgts]
            target_mask = target_mask_full[:, :max_opts, :max_tgts]
            blank_any = blank_positions_full >= 0
            blank_cols = blank_any.any(dim=0)
            max_blanks = int(blank_cols.sum().item()) if blank_cols.numel() else 0
            if max_blanks > 0:
                legal_cols = blank_legal_mask_full[:, :max_blanks].bool().any(dim=0).any(dim=0)
                max_legal = int(legal_cols.sum().item()) if legal_cols.numel() else 0
            else:
                max_legal = 0
            blank_positions = (
                blank_positions_full[:, :max_blanks]
                if max_blanks > 0
                else blank_positions_full[:, :0]
            )
            blank_kind = (
                blank_kind_full[:, :max_blanks] if max_blanks > 0 else blank_kind_full[:, :0]
            )
            blank_group = (
                blank_group_full[:, :max_blanks] if max_blanks > 0 else blank_group_full[:, :0]
            )
            blank_group_kind = (
                blank_group_kind_full[:, :max_blanks]
                if max_blanks > 0
                else blank_group_kind_full[:, :0]
            )
            blank_legal_ids = blank_legal_ids_full[:, :max_blanks, :max_legal]
            blank_legal_mask = blank_legal_mask_full[:, :max_blanks, :max_legal].bool()
        else:
            option_positions = option_positions_full
            option_mask = option_mask_full
            target_positions = target_positions_full
            target_mask = target_mask_full
            blank_positions = blank_positions_full
            blank_kind = blank_kind_full
            blank_group = blank_group_full
            blank_group_kind = blank_group_kind_full
            blank_legal_ids = blank_legal_ids_full
            blank_legal_mask = blank_legal_mask_full.bool()

        # seq_lengths is on pinned CPU memory; .max() here is a host-only op
        # (no GPU sync) and gives flash_attn_varlen a tight per-batch tile bound.
        max_seqlen = int(seq_lengths.max().item()) if seq_lengths.numel() else 0
        return PackedTextBatch(
            token_ids=token_ids,
            seq_id=seq_id,
            pos_in_seq=pos_in_seq,
            cu_seqlens=cu_seqlens,
            seq_lengths=seq_lengths,
            state_positions=state_positions,
            card_ref_positions=card_ref_positions,
            option_positions=option_positions,
            option_mask=option_mask.bool(),
            target_positions=target_positions,
            target_mask=target_mask.bool(),
            blank_positions=blank_positions,
            blank_kind=blank_kind,
            blank_group=blank_group,
            blank_group_kind=blank_group_kind,
            blank_legal_ids=blank_legal_ids,
            blank_legal_mask=blank_legal_mask,
            max_seqlen=max_seqlen,
        )


def allocate_packed_outputs(
    batch_size: int,
    *,
    max_tokens: int,
    max_options: int,
    max_targets: int,
    max_card_refs: int,
    max_blanks: int = 0,
    max_legal_per_blank: int = 0,
) -> NativePackedAssemblerOutputs:
    pin = torch.cuda.is_available()
    cap = batch_size * max_tokens
    return NativePackedAssemblerOutputs(
        token_ids=torch.empty((cap,), dtype=torch.int32, pin_memory=pin),
        seq_id=torch.empty((cap,), dtype=torch.int32, pin_memory=pin),
        pos_in_seq=torch.empty((cap,), dtype=torch.int32, pin_memory=pin),
        cu_seqlens=torch.zeros((batch_size + 1,), dtype=torch.int32, pin_memory=pin),
        seq_lengths=torch.zeros((batch_size,), dtype=torch.int32, pin_memory=pin),
        state_positions=torch.zeros((batch_size,), dtype=torch.int32, pin_memory=pin),
        option_positions=torch.full(
            (batch_size, max_options), -1, dtype=torch.int32, pin_memory=pin
        ),
        option_mask=torch.zeros((batch_size, max_options), dtype=torch.uint8, pin_memory=pin),
        target_positions=torch.full(
            (batch_size, max_options, max_targets), -1, dtype=torch.int32, pin_memory=pin
        ),
        target_mask=torch.zeros(
            (batch_size, max_options, max_targets), dtype=torch.uint8, pin_memory=pin
        ),
        card_ref_positions=torch.full(
            (batch_size, max_card_refs), -1, dtype=torch.int32, pin_memory=pin
        ),
        token_overflow=torch.zeros((batch_size,), dtype=torch.int32),
        blank_positions=torch.full((batch_size, max_blanks), -1, dtype=torch.int32, pin_memory=pin),
        blank_kind=torch.zeros((batch_size, max_blanks), dtype=torch.int32, pin_memory=pin),
        blank_group=torch.full((batch_size, max_blanks), -1, dtype=torch.int32, pin_memory=pin),
        blank_group_kind=torch.zeros((batch_size, max_blanks), dtype=torch.int32, pin_memory=pin),
        blank_legal_ids=torch.zeros(
            (batch_size, max_blanks, max_legal_per_blank), dtype=torch.int32, pin_memory=pin
        ),
        blank_legal_mask=torch.zeros(
            (batch_size, max_blanks, max_legal_per_blank), dtype=torch.uint8, pin_memory=pin
        ),
        blank_overflow=torch.zeros((batch_size,), dtype=torch.int32),
    )


def encode_tokens_packed(
    encoder: Any,
    games: list[Any],
    *,
    perspective_player_indices: list[int],
    max_tokens: int,
    max_options: int,
    max_targets: int,
    max_card_refs: int,
    max_blanks: int = 64,
    max_legal_per_blank: int = 64,
    outputs: NativePackedAssemblerOutputs | None = None,
    include_trace_kinds: bool = True,
    use_inline_blanks: bool = False,
) -> tuple[Any, NativePackedAssemblerOutputs]:
    """Run ``MageEncodeTokensPacked`` for a batch of games.

    Writes into a packed variable-length output buffer. Anchors come back as
    absolute offsets into ``token_ids``; ``cu_seqlens[-1]`` is the live token
    count.
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
            max_blanks=max_blanks if use_inline_blanks else 0,
            max_legal_per_blank=max_legal_per_blank if use_inline_blanks else 0,
        )
    elif use_inline_blanks and (
        outputs.blank_positions.shape[1] < max_blanks
        or outputs.blank_legal_ids.shape[2] < max_legal_per_blank
    ):
        raise ValueError(
            "provided outputs do not have enough blank capacity: "
            f"K={outputs.blank_positions.shape[1]} V={outputs.blank_legal_ids.shape[2]}, "
            f"requested K={max_blanks} V={max_legal_per_blank}"
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
    original_emit_render_plan = int(cfg.emit_render_plan)
    original_render_plan_capacity = int(cfg.render_plan_capacity)
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
                "option_positions": ffi.cast("int32_t *", outputs.option_positions.data_ptr()),
                "option_mask": ffi.cast("uint8_t *", outputs.option_mask.data_ptr()),
                "target_positions": ffi.cast("int32_t *", outputs.target_positions.data_ptr()),
                "target_mask": ffi.cast("uint8_t *", outputs.target_mask.data_ptr()),
                "card_ref_positions": ffi.cast("int32_t *", outputs.card_ref_positions.data_ptr()),
                "token_overflow": ffi.cast("int32_t *", outputs.token_overflow.data_ptr()),
            },
        )
    tok_cfg = outputs._tok_cfg_cffi
    packed_out = outputs._packed_out_cffi
    blank_cfg: Any = ffi.NULL
    blank_out: Any = ffi.NULL
    if use_inline_blanks:
        if outputs._blank_cfg_cffi is None:
            outputs._blank_cfg_cffi = ffi.new(
                "MageBlankAssemblerConfig *",
                {
                    "max_blanks": max_blanks,
                    "max_legal_per_blank": max_legal_per_blank,
                },
            )
        if outputs._blank_out_cffi is None:
            outputs._blank_out_cffi = ffi.new(
                "MagePackedBlankOutputs *",
                {
                    "k_max": max_blanks,
                    "v_max": max_legal_per_blank,
                    "blank_positions": ffi.cast("int32_t *", outputs.blank_positions.data_ptr()),
                    "blank_kind": ffi.cast("int32_t *", outputs.blank_kind.data_ptr()),
                    "blank_group": ffi.cast("int32_t *", outputs.blank_group.data_ptr()),
                    "blank_group_kind": ffi.cast("int32_t *", outputs.blank_group_kind.data_ptr()),
                    "blank_legal_ids": ffi.cast("int32_t *", outputs.blank_legal_ids.data_ptr()),
                    "blank_legal_mask": ffi.cast("uint8_t *", outputs.blank_legal_mask.data_ptr()),
                    "blank_overflow": ffi.cast("int32_t *", outputs.blank_overflow.data_ptr()),
                },
            )
        blank_cfg = outputs._blank_cfg_cffi
        blank_out = outputs._blank_out_cffi

    buffers = scratch.buffers
    assert buffers is not None
    ctypes_lib = _load_packed_ctypes_lib()
    try:
        if ctypes_lib is not None and scratch.req_c is not None:
            req_c = scratch.req_c
            cfg_c = scratch.cfg_c
            enc_out_c = scratch.out_c
            assert cfg_c is not None and enc_out_c is not None
            req_c.n = batch_size
            cfg_c.decision_capacity = decision_capacity
            original_emit_render_plan_c = int(cfg_c.emit_render_plan)
            original_render_plan_capacity_c = int(cfg_c.render_plan_capacity)
            cfg_c.emit_render_plan = 0
            cfg_c.render_plan_capacity = int(encoder.render_plan_capacity)
            try:
                if outputs._tok_cfg_ctypes is None:
                    outputs._tok_cfg_ctypes = _MageTokenAssemblerConfigC(
                        max_tokens=max_tokens,
                        max_options=max_options,
                        max_targets=max_targets,
                        max_card_refs=max_card_refs,
                    )
                if outputs._packed_out_ctypes is None:
                    outputs._packed_out_ctypes = _MagePackedTokenAssemblerOutputsC(
                        token_ids=_tensor_ptr(outputs.token_ids, ctypes.c_int32),
                        cu_seqlens=_tensor_ptr(outputs.cu_seqlens, ctypes.c_int32),
                        seq_lengths=_tensor_ptr(outputs.seq_lengths, ctypes.c_int32),
                        state_positions=_tensor_ptr(outputs.state_positions, ctypes.c_int32),
                        option_positions=_tensor_ptr(outputs.option_positions, ctypes.c_int32),
                        option_mask=_tensor_ptr(outputs.option_mask, ctypes.c_uint8),
                        target_positions=_tensor_ptr(outputs.target_positions, ctypes.c_int32),
                        target_mask=_tensor_ptr(outputs.target_mask, ctypes.c_uint8),
                        card_ref_positions=_tensor_ptr(outputs.card_ref_positions, ctypes.c_int32),
                        token_overflow=_tensor_ptr(outputs.token_overflow, ctypes.c_int32),
                    )
                tok_cfg_c = outputs._tok_cfg_ctypes
                packed_out_c = outputs._packed_out_ctypes
                blank_cfg_c: _MageBlankAssemblerConfigC | None = None
                blank_out_c: _MagePackedBlankOutputsC | None = None
                if use_inline_blanks:
                    if outputs._blank_cfg_ctypes is None:
                        outputs._blank_cfg_ctypes = _MageBlankAssemblerConfigC(
                            max_blanks=max_blanks,
                            max_legal_per_blank=max_legal_per_blank,
                        )
                    if outputs._blank_out_ctypes is None:
                        outputs._blank_out_ctypes = _MagePackedBlankOutputsC(
                            k_max=max_blanks,
                            v_max=max_legal_per_blank,
                            blank_positions=_tensor_ptr(outputs.blank_positions, ctypes.c_int32),
                            blank_kind=_tensor_ptr(outputs.blank_kind, ctypes.c_int32),
                            blank_group=_tensor_ptr(outputs.blank_group, ctypes.c_int32),
                            blank_group_kind=_tensor_ptr(outputs.blank_group_kind, ctypes.c_int32),
                            blank_legal_ids=_tensor_ptr(outputs.blank_legal_ids, ctypes.c_int32),
                            blank_legal_mask=_tensor_ptr(outputs.blank_legal_mask, ctypes.c_uint8),
                            blank_overflow=_tensor_ptr(outputs.blank_overflow, ctypes.c_int32),
                        )
                    blank_cfg_c = outputs._blank_cfg_ctypes
                    blank_out_c = outputs._blank_out_ctypes
                assert tok_cfg_c is not None and packed_out_c is not None
                result = ctypes_lib.MageEncodeTokensPacked(
                    ctypes.byref(req_c),
                    ctypes.byref(cfg_c),
                    ctypes.byref(enc_out_c),
                    ctypes.byref(tok_cfg_c),
                    ctypes.byref(packed_out_c),
                    ctypes.byref(blank_cfg_c) if blank_cfg_c is not None else None,
                    ctypes.byref(blank_out_c) if blank_out_c is not None else None,
                )
                if result.error_code != 0:
                    message = "MageEncodeTokensPacked failed"
                    if result.error_message:
                        try:
                            raw = ctypes.cast(result.error_message, ctypes.c_char_p).value
                            if raw is not None:
                                message = raw.decode("utf-8")
                        finally:
                            ctypes_lib.MageFreeString(result.error_message)
                    if "requires cfg.emit_render_plan=1" in message:
                        cfg_c.emit_render_plan = original_emit_render_plan_c
                        cfg_c.render_plan_capacity = original_render_plan_capacity_c
                        result = ctypes_lib.MageEncodeTokensPacked(
                            ctypes.byref(req_c),
                            ctypes.byref(cfg_c),
                            ctypes.byref(enc_out_c),
                            ctypes.byref(tok_cfg_c),
                            ctypes.byref(packed_out_c),
                            ctypes.byref(blank_cfg_c) if blank_cfg_c is not None else None,
                            ctypes.byref(blank_out_c) if blank_out_c is not None else None,
                        )
                        if result.error_code == 0:
                            message = ""
                    from magic_ai.slot_encoder.native_encoder import NativeEncodingError

                    if result.error_code != 0:
                        raise NativeEncodingError(message)
            finally:
                cfg_c.emit_render_plan = original_emit_render_plan_c
                cfg_c.render_plan_capacity = original_render_plan_capacity_c
        else:
            result = lib.MageEncodeTokensPacked(
                req, cfg, enc_out, tok_cfg, packed_out, blank_cfg, blank_out
            )
            if result.error_code != 0:
                message = "MageEncodeTokensPacked failed"
                if result.error_message != ffi.NULL:
                    try:
                        message = ffi.string(result.error_message).decode("utf-8")
                    finally:
                        lib.MageFreeString(result.error_message)
                if "requires cfg.emit_render_plan=1" in message:
                    cfg.emit_render_plan = original_emit_render_plan
                    cfg.render_plan_capacity = original_render_plan_capacity
                    result = lib.MageEncodeTokensPacked(
                        req, cfg, enc_out, tok_cfg, packed_out, blank_cfg, blank_out
                    )
                    if result.error_code == 0:
                        message = ""
                from magic_ai.slot_encoder.native_encoder import NativeEncodingError

                if result.error_code != 0:
                    raise NativeEncodingError(message)
    finally:
        cfg.emit_render_plan = original_emit_render_plan
        cfg.render_plan_capacity = original_render_plan_capacity

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
    "NativePackedAssemblerOutputs",
    "allocate_packed_outputs",
    "encode_tokens_packed",
]
