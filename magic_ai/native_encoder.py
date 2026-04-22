from __future__ import annotations

import ctypes
import importlib
import json
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from magic_ai.game_state import PendingState


class NativeEncodingError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeEncodedBatch:
    trace_kind_id: Tensor
    slot_card_rows: Tensor
    slot_occupied: Tensor
    slot_tapped: Tensor
    game_info: Tensor
    pending_kind_id: Tensor
    num_present_options: Tensor
    option_kind_ids: Tensor
    option_scalars: Tensor
    option_mask: Tensor
    option_ref_slot_idx: Tensor
    option_ref_card_row: Tensor
    target_mask: Tensor
    target_type_ids: Tensor
    target_scalars: Tensor
    target_overflow: Tensor
    target_ref_slot_idx: Tensor
    target_ref_is_player: Tensor
    target_ref_is_self: Tensor
    may_mask: Tensor
    decision_start: Tensor
    decision_count: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    decision_rows_written: int
    pendings: list[PendingState]
    trace_kinds: list[str]


class _MageBatchRequest(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int64),
        ("handles", ctypes.POINTER(ctypes.c_int64)),
        ("perspective_player_idx", ctypes.POINTER(ctypes.c_int64)),
    ]


class _MageEncodeConfig(ctypes.Structure):
    _fields_ = [
        ("max_options", ctypes.c_int64),
        ("max_targets_per_option", ctypes.c_int64),
        ("max_cached_choices", ctypes.c_int64),
        ("zone_slot_count", ctypes.c_int64),
        ("game_info_dim", ctypes.c_int64),
        ("option_scalar_dim", ctypes.c_int64),
        ("target_scalar_dim", ctypes.c_int64),
        ("decision_capacity", ctypes.c_int64),
    ]


class _MageEncodeOutputs(ctypes.Structure):
    _fields_ = [
        ("trace_kind_id", ctypes.POINTER(ctypes.c_int64)),
        ("slot_card_rows", ctypes.POINTER(ctypes.c_int64)),
        ("slot_occupied", ctypes.POINTER(ctypes.c_float)),
        ("slot_tapped", ctypes.POINTER(ctypes.c_float)),
        ("game_info", ctypes.POINTER(ctypes.c_float)),
        ("pending_kind_id", ctypes.POINTER(ctypes.c_int64)),
        ("num_present_options", ctypes.POINTER(ctypes.c_int64)),
        ("option_kind_ids", ctypes.POINTER(ctypes.c_int64)),
        ("option_scalars", ctypes.POINTER(ctypes.c_float)),
        ("option_mask", ctypes.POINTER(ctypes.c_float)),
        ("option_ref_slot_idx", ctypes.POINTER(ctypes.c_int64)),
        ("option_ref_card_row", ctypes.POINTER(ctypes.c_int64)),
        ("target_mask", ctypes.POINTER(ctypes.c_float)),
        ("target_type_ids", ctypes.POINTER(ctypes.c_int64)),
        ("target_scalars", ctypes.POINTER(ctypes.c_float)),
        ("target_overflow", ctypes.POINTER(ctypes.c_float)),
        ("target_ref_slot_idx", ctypes.POINTER(ctypes.c_int64)),
        ("target_ref_is_player", ctypes.POINTER(ctypes.c_uint8)),
        ("target_ref_is_self", ctypes.POINTER(ctypes.c_uint8)),
        ("may_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("decision_start", ctypes.POINTER(ctypes.c_int64)),
        ("decision_count", ctypes.POINTER(ctypes.c_int64)),
        ("decision_option_idx", ctypes.POINTER(ctypes.c_int64)),
        ("decision_target_idx", ctypes.POINTER(ctypes.c_int64)),
        ("decision_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("uses_none_head", ctypes.POINTER(ctypes.c_uint8)),
    ]


class _MageEncodeResult(ctypes.Structure):
    _fields_ = [
        ("decision_rows_written", ctypes.c_int64),
        ("error_code", ctypes.c_int64),
        ("error_message", ctypes.c_void_p),
    ]


TRACE_KIND_VALUES = (
    "priority",
    "attackers",
    "blockers",
    "choice_index",
    "choice_ids",
    "choice_color",
    "may",
)


@dataclass
class _BufferSet:
    trace_kind_id: Tensor
    slot_card_rows: Tensor
    slot_occupied: Tensor
    slot_tapped: Tensor
    game_info: Tensor
    pending_kind_id: Tensor
    num_present_options: Tensor
    option_kind_ids: Tensor
    option_scalars: Tensor
    option_mask: Tensor
    option_ref_slot_idx: Tensor
    option_ref_card_row: Tensor
    target_mask: Tensor
    target_type_ids: Tensor
    target_scalars: Tensor
    target_overflow: Tensor
    target_ref_slot_idx: Tensor
    target_ref_is_player_u8: Tensor
    target_ref_is_self_u8: Tensor
    may_mask_u8: Tensor
    decision_start: Tensor
    decision_count: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask_u8: Tensor
    uses_none_head_u8: Tensor


def _ptr(tensor: Tensor, ctype: Any) -> Any:
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctype))


def _validate_decision_layout(
    *,
    decision_option_idx: Tensor,
    decision_target_idx: Tensor,
    decision_mask: Tensor,
    uses_none_head: Tensor,
    max_options: int,
    max_targets_per_option: int,
) -> None:
    valid = decision_mask.nonzero(as_tuple=False)
    if valid.numel() == 0:
        return

    rows = valid[:, 0]
    cols = valid[:, 1]
    none_choices = uses_none_head[rows] & cols.eq(0)
    scored = ~none_choices
    if not scored.any():
        return

    scored_options = decision_option_idx[rows[scored], cols[scored]]
    bad_options = (scored_options < 0) | (scored_options >= max_options)
    if bad_options.any():
        bad = scored_options[bad_options][0].item()
        raise NativeEncodingError(
            f"native encoder produced decision option index {bad}, outside [0, {max_options})"
        )

    scored_targets = decision_target_idx[rows[scored], cols[scored]]
    bad_targets = scored_targets >= max_targets_per_option
    if bad_targets.any():
        bad = scored_targets[bad_targets][0].item()
        raise NativeEncodingError(
            f"native encoder produced decision target index {bad}, "
            f"outside [0, {max_targets_per_option})"
        )


class NativeBatchEncoder:
    def __init__(
        self,
        *,
        max_options: int,
        max_targets_per_option: int,
        max_cached_choices: int,
        zone_slot_count: int | None = None,
        game_info_dim: int | None = None,
        option_scalar_dim: int | None = None,
        target_scalar_dim: int | None = None,
        lib: Any | None = None,
        ffi: Any | None = None,
        card_name_to_row: dict[str, int] | None = None,
    ) -> None:
        self.max_options = max_options
        self.max_targets_per_option = max_targets_per_option
        self.max_cached_choices = max_cached_choices
        self.zone_slot_count = zone_slot_count
        self.game_info_dim = game_info_dim
        self.option_scalar_dim = option_scalar_dim
        self.target_scalar_dim = target_scalar_dim
        self.lib = lib
        self.ffi = ffi
        self.is_available = False
        if self.lib is None:
            return
        if not self._has_required_symbols():
            return
        if self.ffi is None:
            self._configure_ctypes()
        self.is_available = True
        if card_name_to_row is not None:
            self.register_card_rows(card_name_to_row)

    def _has_required_symbols(self) -> bool:
        return all(
            self._has_symbol(name)
            for name in ("MageEncodeBatch", "MageSetCardNameRows", "MageFreeString")
        )

    def _has_symbol(self, name: str) -> bool:
        try:
            getattr(self.lib, name)
        except AttributeError:
            return False
        return True

    @classmethod
    def for_policy(cls, policy: Any) -> NativeBatchEncoder:
        try:
            mage = importlib.import_module("mage")
            mage_any = cast(Any, mage)
            if mage_any._lib is None or mage_any._ffi is None:
                mage_any.load()
            lib = mage_any._lib
            ffi = mage_any._ffi
        except Exception:
            return cls(
                max_options=policy.max_options,
                max_targets_per_option=policy.max_targets_per_option,
                max_cached_choices=policy.max_cached_choices,
            )
        return cls(
            max_options=policy.max_options,
            max_targets_per_option=policy.max_targets_per_option,
            max_cached_choices=policy.max_cached_choices,
            zone_slot_count=int(policy.rollout_buffer.slot_card_rows.shape[1]),
            game_info_dim=int(policy.rollout_buffer.game_info.shape[1]),
            option_scalar_dim=int(policy.action_encoder.option_scalar_projection.in_features),
            target_scalar_dim=int(policy.action_encoder.target_scalar_projection.in_features),
            lib=lib,
            ffi=ffi,
            card_name_to_row=policy.game_state_encoder._card_name_to_row,
        )

    def _configure_ctypes(self) -> None:
        lib = cast(Any, self.lib)
        lib.MageSetCardNameRows.argtypes = [ctypes.c_char_p]
        lib.MageSetCardNameRows.restype = ctypes.c_void_p
        lib.MageEncodeBatch.argtypes = [
            ctypes.POINTER(_MageBatchRequest),
            ctypes.POINTER(_MageEncodeConfig),
            ctypes.POINTER(_MageEncodeOutputs),
        ]
        lib.MageEncodeBatch.restype = _MageEncodeResult
        lib.MageFreeString.argtypes = [ctypes.c_void_p]
        lib.MageFreeString.restype = None

    def register_card_rows(self, card_name_to_row: dict[str, int]) -> None:
        if not self.is_available:
            raise NativeEncodingError("MageEncodeBatch is unavailable")
        payload = json.dumps(card_name_to_row).encode("utf-8")
        if self.ffi is not None:
            lib = cast(Any, self.lib)
            raw = lib.MageSetCardNameRows(self.ffi.new("char[]", payload))
            if raw == self.ffi.NULL:
                raise NativeEncodingError("MageSetCardNameRows returned null")
            try:
                response = json.loads(self.ffi.string(raw).decode("utf-8"))
            finally:
                lib.MageFreeString(raw)
            if not response.get("ok", False):
                raise NativeEncodingError(response.get("error", "failed to register card rows"))
            return
        lib = cast(Any, self.lib)
        raw = lib.MageSetCardNameRows(ctypes.c_char_p(payload))
        if not raw:
            raise NativeEncodingError("MageSetCardNameRows returned null")
        try:
            raw_value = ctypes.cast(raw, ctypes.c_char_p).value
            if raw_value is None:
                raise NativeEncodingError("MageSetCardNameRows returned null payload")
            response = json.loads(raw_value.decode("utf-8"))
        finally:
            lib.MageFreeString(raw)
        if not response.get("ok", False):
            raise NativeEncodingError(response.get("error", "failed to register card rows"))

    def _alloc(self, batch_size: int, decision_capacity: int) -> _BufferSet:
        if self.zone_slot_count is None or self.game_info_dim is None:
            raise NativeEncodingError("NativeBatchEncoder is missing shape metadata")
        if self.option_scalar_dim is None or self.target_scalar_dim is None:
            raise NativeEncodingError("NativeBatchEncoder is missing scalar metadata")
        return _BufferSet(
            trace_kind_id=torch.empty((batch_size,), dtype=torch.int64),
            slot_card_rows=torch.empty((batch_size, self.zone_slot_count), dtype=torch.int64),
            slot_occupied=torch.empty((batch_size, self.zone_slot_count), dtype=torch.float32),
            slot_tapped=torch.empty((batch_size, self.zone_slot_count), dtype=torch.float32),
            game_info=torch.empty((batch_size, self.game_info_dim), dtype=torch.float32),
            pending_kind_id=torch.empty((batch_size,), dtype=torch.int64),
            num_present_options=torch.empty((batch_size,), dtype=torch.int64),
            option_kind_ids=torch.empty((batch_size, self.max_options), dtype=torch.int64),
            option_scalars=torch.empty(
                (batch_size, self.max_options, self.option_scalar_dim), dtype=torch.float32
            ),
            option_mask=torch.empty((batch_size, self.max_options), dtype=torch.float32),
            option_ref_slot_idx=torch.empty((batch_size, self.max_options), dtype=torch.int64),
            option_ref_card_row=torch.empty((batch_size, self.max_options), dtype=torch.int64),
            target_mask=torch.empty(
                (batch_size, self.max_options, self.max_targets_per_option), dtype=torch.float32
            ),
            target_type_ids=torch.empty(
                (batch_size, self.max_options, self.max_targets_per_option), dtype=torch.int64
            ),
            target_scalars=torch.empty(
                (
                    batch_size,
                    self.max_options,
                    self.max_targets_per_option,
                    self.target_scalar_dim,
                ),
                dtype=torch.float32,
            ),
            target_overflow=torch.empty((batch_size, self.max_options), dtype=torch.float32),
            target_ref_slot_idx=torch.empty(
                (batch_size, self.max_options, self.max_targets_per_option), dtype=torch.int64
            ),
            target_ref_is_player_u8=torch.empty(
                (batch_size, self.max_options, self.max_targets_per_option), dtype=torch.uint8
            ),
            target_ref_is_self_u8=torch.empty(
                (batch_size, self.max_options, self.max_targets_per_option), dtype=torch.uint8
            ),
            may_mask_u8=torch.empty((batch_size,), dtype=torch.uint8),
            decision_start=torch.empty((batch_size,), dtype=torch.int64),
            decision_count=torch.empty((batch_size,), dtype=torch.int64),
            decision_option_idx=torch.empty(
                (decision_capacity, self.max_cached_choices), dtype=torch.int64
            ),
            decision_target_idx=torch.empty(
                (decision_capacity, self.max_cached_choices), dtype=torch.int64
            ),
            decision_mask_u8=torch.empty(
                (decision_capacity, self.max_cached_choices), dtype=torch.uint8
            ),
            uses_none_head_u8=torch.empty((decision_capacity,), dtype=torch.uint8),
        )

    def encode_batch(
        self,
        games: list[Any],
        pendings: list[PendingState],
        *,
        perspective_player_indices: list[int],
    ) -> NativeEncodedBatch:
        if not self.is_available:
            raise NativeEncodingError("MageEncodeBatch is unavailable")
        batch_size = len(games)
        if batch_size == 0:
            raise NativeEncodingError("empty batch")
        if batch_size != len(pendings) or batch_size != len(perspective_player_indices):
            raise NativeEncodingError("games, pendings, and perspective_player_indices must match")
        return self._encode_games(
            games,
            pendings,
            perspective_player_indices=perspective_player_indices,
        )

    def encode_handles(
        self,
        games: list[Any],
        *,
        perspective_player_indices: list[int],
    ) -> NativeEncodedBatch:
        if not self.is_available:
            raise NativeEncodingError("MageEncodeBatch is unavailable")
        if len(games) != len(perspective_player_indices):
            raise NativeEncodingError("games and perspective_player_indices must match")
        return self._encode_games(
            games,
            [],
            perspective_player_indices=perspective_player_indices,
        )

    def _encode_games(
        self,
        games: list[Any],
        pendings: list[PendingState],
        *,
        perspective_player_indices: list[int],
    ) -> NativeEncodedBatch:
        batch_size = len(games)
        if batch_size == 0:
            raise NativeEncodingError("empty batch")
        decision_capacity = max(1, batch_size * self.max_options)
        buffers = self._alloc(batch_size, decision_capacity)
        if self.ffi is not None:
            return self._encode_batch_cffi(
                buffers,
                games,
                pendings,
                perspective_player_indices=perspective_player_indices,
                decision_capacity=decision_capacity,
            )
        handles_t = torch.tensor([int(game.handle) for game in games], dtype=torch.int64)
        perspectives_t = torch.tensor(perspective_player_indices, dtype=torch.int64)
        request = _MageBatchRequest(
            n=batch_size,
            handles=_ptr(handles_t, ctypes.c_int64),
            perspective_player_idx=_ptr(perspectives_t, ctypes.c_int64),
        )
        config = _MageEncodeConfig(
            max_options=self.max_options,
            max_targets_per_option=self.max_targets_per_option,
            max_cached_choices=self.max_cached_choices,
            zone_slot_count=cast(int, self.zone_slot_count),
            game_info_dim=cast(int, self.game_info_dim),
            option_scalar_dim=cast(int, self.option_scalar_dim),
            target_scalar_dim=cast(int, self.target_scalar_dim),
            decision_capacity=decision_capacity,
        )
        outputs = _MageEncodeOutputs(
            trace_kind_id=_ptr(buffers.trace_kind_id, ctypes.c_int64),
            slot_card_rows=_ptr(buffers.slot_card_rows, ctypes.c_int64),
            slot_occupied=_ptr(buffers.slot_occupied, ctypes.c_float),
            slot_tapped=_ptr(buffers.slot_tapped, ctypes.c_float),
            game_info=_ptr(buffers.game_info, ctypes.c_float),
            pending_kind_id=_ptr(buffers.pending_kind_id, ctypes.c_int64),
            num_present_options=_ptr(buffers.num_present_options, ctypes.c_int64),
            option_kind_ids=_ptr(buffers.option_kind_ids, ctypes.c_int64),
            option_scalars=_ptr(buffers.option_scalars, ctypes.c_float),
            option_mask=_ptr(buffers.option_mask, ctypes.c_float),
            option_ref_slot_idx=_ptr(buffers.option_ref_slot_idx, ctypes.c_int64),
            option_ref_card_row=_ptr(buffers.option_ref_card_row, ctypes.c_int64),
            target_mask=_ptr(buffers.target_mask, ctypes.c_float),
            target_type_ids=_ptr(buffers.target_type_ids, ctypes.c_int64),
            target_scalars=_ptr(buffers.target_scalars, ctypes.c_float),
            target_overflow=_ptr(buffers.target_overflow, ctypes.c_float),
            target_ref_slot_idx=_ptr(buffers.target_ref_slot_idx, ctypes.c_int64),
            target_ref_is_player=_ptr(buffers.target_ref_is_player_u8, ctypes.c_uint8),
            target_ref_is_self=_ptr(buffers.target_ref_is_self_u8, ctypes.c_uint8),
            may_mask=_ptr(buffers.may_mask_u8, ctypes.c_uint8),
            decision_start=_ptr(buffers.decision_start, ctypes.c_int64),
            decision_count=_ptr(buffers.decision_count, ctypes.c_int64),
            decision_option_idx=_ptr(buffers.decision_option_idx, ctypes.c_int64),
            decision_target_idx=_ptr(buffers.decision_target_idx, ctypes.c_int64),
            decision_mask=_ptr(buffers.decision_mask_u8, ctypes.c_uint8),
            uses_none_head=_ptr(buffers.uses_none_head_u8, ctypes.c_uint8),
        )
        lib = cast(Any, self.lib)
        result = lib.MageEncodeBatch(
            ctypes.byref(request),
            ctypes.byref(config),
            ctypes.byref(outputs),
        )
        if result.error_code != 0:
            message = "native encoder failed"
            if result.error_message:
                try:
                    raw_value = ctypes.cast(result.error_message, ctypes.c_char_p).value
                    if raw_value is not None:
                        message = raw_value.decode("utf-8")
                finally:
                    lib.MageFreeString(result.error_message)
            raise NativeEncodingError(message)
        decision_rows_written = int(result.decision_rows_written)
        trace_kind_id = buffers.trace_kind_id
        trace_kinds = [TRACE_KIND_VALUES[int(idx)] for idx in trace_kind_id.tolist()]
        decision_option_idx = buffers.decision_option_idx[:decision_rows_written]
        decision_target_idx = buffers.decision_target_idx[:decision_rows_written]
        decision_mask = buffers.decision_mask_u8[:decision_rows_written].ne(0)
        uses_none_head = buffers.uses_none_head_u8[:decision_rows_written].ne(0)
        _validate_decision_layout(
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            max_options=self.max_options,
            max_targets_per_option=self.max_targets_per_option,
        )
        return NativeEncodedBatch(
            trace_kind_id=trace_kind_id,
            slot_card_rows=buffers.slot_card_rows,
            slot_occupied=buffers.slot_occupied,
            slot_tapped=buffers.slot_tapped,
            game_info=buffers.game_info,
            pending_kind_id=buffers.pending_kind_id,
            num_present_options=buffers.num_present_options,
            option_kind_ids=buffers.option_kind_ids,
            option_scalars=buffers.option_scalars,
            option_mask=buffers.option_mask,
            option_ref_slot_idx=buffers.option_ref_slot_idx,
            option_ref_card_row=buffers.option_ref_card_row,
            target_mask=buffers.target_mask,
            target_type_ids=buffers.target_type_ids,
            target_scalars=buffers.target_scalars,
            target_overflow=buffers.target_overflow,
            target_ref_slot_idx=buffers.target_ref_slot_idx,
            target_ref_is_player=buffers.target_ref_is_player_u8.ne(0),
            target_ref_is_self=buffers.target_ref_is_self_u8.ne(0),
            may_mask=buffers.may_mask_u8.ne(0),
            decision_start=buffers.decision_start,
            decision_count=buffers.decision_count,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            decision_rows_written=decision_rows_written,
            pendings=pendings,
            trace_kinds=trace_kinds,
        )

    def _encode_batch_cffi(
        self,
        buffers: _BufferSet,
        games: list[Any],
        pendings: list[PendingState],
        *,
        perspective_player_indices: list[int],
        decision_capacity: int,
    ) -> NativeEncodedBatch:
        ffi = self.ffi
        assert ffi is not None
        lib = cast(Any, self.lib)
        handles_t = torch.tensor(
            [int(getattr(game, "handle", getattr(game, "_id"))) for game in games],
            dtype=torch.int64,
        )
        perspectives_t = torch.tensor(perspective_player_indices, dtype=torch.int64)
        req = ffi.new(
            "MageBatchRequest *",
            {
                "n": len(games),
                "handles": ffi.cast("int64_t *", handles_t.data_ptr()),
                "perspective_player_idx": ffi.cast("int64_t *", perspectives_t.data_ptr()),
            },
        )
        cfg = ffi.new(
            "MageEncodeConfig *",
            {
                "max_options": self.max_options,
                "max_targets_per_option": self.max_targets_per_option,
                "max_cached_choices": self.max_cached_choices,
                "zone_slot_count": cast(int, self.zone_slot_count),
                "game_info_dim": cast(int, self.game_info_dim),
                "option_scalar_dim": cast(int, self.option_scalar_dim),
                "target_scalar_dim": cast(int, self.target_scalar_dim),
                "decision_capacity": decision_capacity,
            },
        )
        out = ffi.new(
            "MageEncodeOutputs *",
            {
                "trace_kind_id": ffi.cast("int64_t *", buffers.trace_kind_id.data_ptr()),
                "slot_card_rows": ffi.cast("int64_t *", buffers.slot_card_rows.data_ptr()),
                "slot_occupied": ffi.cast("float *", buffers.slot_occupied.data_ptr()),
                "slot_tapped": ffi.cast("float *", buffers.slot_tapped.data_ptr()),
                "game_info": ffi.cast("float *", buffers.game_info.data_ptr()),
                "pending_kind_id": ffi.cast("int64_t *", buffers.pending_kind_id.data_ptr()),
                "num_present_options": ffi.cast(
                    "int64_t *", buffers.num_present_options.data_ptr()
                ),
                "option_kind_ids": ffi.cast("int64_t *", buffers.option_kind_ids.data_ptr()),
                "option_scalars": ffi.cast("float *", buffers.option_scalars.data_ptr()),
                "option_mask": ffi.cast("float *", buffers.option_mask.data_ptr()),
                "option_ref_slot_idx": ffi.cast(
                    "int64_t *", buffers.option_ref_slot_idx.data_ptr()
                ),
                "option_ref_card_row": ffi.cast(
                    "int64_t *", buffers.option_ref_card_row.data_ptr()
                ),
                "target_mask": ffi.cast("float *", buffers.target_mask.data_ptr()),
                "target_type_ids": ffi.cast("int64_t *", buffers.target_type_ids.data_ptr()),
                "target_scalars": ffi.cast("float *", buffers.target_scalars.data_ptr()),
                "target_overflow": ffi.cast("float *", buffers.target_overflow.data_ptr()),
                "target_ref_slot_idx": ffi.cast(
                    "int64_t *", buffers.target_ref_slot_idx.data_ptr()
                ),
                "target_ref_is_player": ffi.cast(
                    "uint8_t *", buffers.target_ref_is_player_u8.data_ptr()
                ),
                "target_ref_is_self": ffi.cast(
                    "uint8_t *", buffers.target_ref_is_self_u8.data_ptr()
                ),
                "may_mask": ffi.cast("uint8_t *", buffers.may_mask_u8.data_ptr()),
                "decision_start": ffi.cast("int64_t *", buffers.decision_start.data_ptr()),
                "decision_count": ffi.cast("int64_t *", buffers.decision_count.data_ptr()),
                "decision_option_idx": ffi.cast(
                    "int64_t *", buffers.decision_option_idx.data_ptr()
                ),
                "decision_target_idx": ffi.cast(
                    "int64_t *", buffers.decision_target_idx.data_ptr()
                ),
                "decision_mask": ffi.cast("uint8_t *", buffers.decision_mask_u8.data_ptr()),
                "uses_none_head": ffi.cast("uint8_t *", buffers.uses_none_head_u8.data_ptr()),
            },
        )
        result = lib.MageEncodeBatch(req, cfg, out)
        if result.error_code != 0:
            message = "native encoder failed"
            if result.error_message != ffi.NULL:
                try:
                    message = ffi.string(result.error_message).decode("utf-8")
                finally:
                    lib.MageFreeString(result.error_message)
            raise NativeEncodingError(message)
        decision_rows_written = int(result.decision_rows_written)
        trace_kind_id = buffers.trace_kind_id
        trace_kinds = [TRACE_KIND_VALUES[int(idx)] for idx in trace_kind_id.tolist()]
        decision_option_idx = buffers.decision_option_idx[:decision_rows_written]
        decision_target_idx = buffers.decision_target_idx[:decision_rows_written]
        decision_mask = buffers.decision_mask_u8[:decision_rows_written].ne(0)
        uses_none_head = buffers.uses_none_head_u8[:decision_rows_written].ne(0)
        _validate_decision_layout(
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            max_options=self.max_options,
            max_targets_per_option=self.max_targets_per_option,
        )
        return NativeEncodedBatch(
            trace_kind_id=trace_kind_id,
            slot_card_rows=buffers.slot_card_rows,
            slot_occupied=buffers.slot_occupied,
            slot_tapped=buffers.slot_tapped,
            game_info=buffers.game_info,
            pending_kind_id=buffers.pending_kind_id,
            num_present_options=buffers.num_present_options,
            option_kind_ids=buffers.option_kind_ids,
            option_scalars=buffers.option_scalars,
            option_mask=buffers.option_mask,
            option_ref_slot_idx=buffers.option_ref_slot_idx,
            option_ref_card_row=buffers.option_ref_card_row,
            target_mask=buffers.target_mask,
            target_type_ids=buffers.target_type_ids,
            target_scalars=buffers.target_scalars,
            target_overflow=buffers.target_overflow,
            target_ref_slot_idx=buffers.target_ref_slot_idx,
            target_ref_is_player=buffers.target_ref_is_player_u8.ne(0),
            target_ref_is_self=buffers.target_ref_is_self_u8.ne(0),
            may_mask=buffers.may_mask_u8.ne(0),
            decision_start=buffers.decision_start,
            decision_count=buffers.decision_count,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            decision_rows_written=decision_rows_written,
            pendings=pendings,
            trace_kinds=trace_kinds,
        )
