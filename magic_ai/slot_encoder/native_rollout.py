from __future__ import annotations

import ctypes
import threading
from ctypes import POINTER, Structure, byref, c_char_p, c_int64
from dataclasses import dataclass
from typing import Any, cast

import torch


class NativeRolloutUnavailable(RuntimeError):
    pass


# ctypes binding for MageBatchStepByChoice. The mage package loads the
# library via cffi in ABI mode, which holds the GIL across foreign calls
# and serialises step_by_choice across worker threads. ctypes' default is
# to release the GIL around foreign calls, so opening the same .so via
# ctypes lets ShardedNativeRolloutDriver actually run shards in parallel.
class _MageStepChoiceRequestC(Structure):
    _fields_ = [
        ("n", c_int64),
        ("max_options", c_int64),
        ("max_targets_per_option", c_int64),
        ("handles", POINTER(c_int64)),
        ("decision_start", POINTER(c_int64)),
        ("decision_count", POINTER(c_int64)),
        ("selected_choice_cols", POINTER(c_int64)),
        ("may_selected", POINTER(c_int64)),
    ]


class _MageEncodeResultC(Structure):
    _fields_ = [
        ("decision_rows_written", c_int64),
        ("error_code", c_int64),
        ("error_message", c_char_p),
    ]


class _MageTextRolloutStartRequestC(Structure):
    _fields_ = [
        ("n", c_int64),
        ("handles", POINTER(c_int64)),
        ("slot_ids", POINTER(c_int64)),
        ("episode_ids", POINTER(c_int64)),
        ("max_steps_per_game", c_int64),
        ("max_options", c_int64),
        ("max_targets_per_option", c_int64),
        ("max_cached_choices", c_int64),
        ("zone_slot_count", c_int64),
        ("game_info_dim", c_int64),
        ("option_scalar_dim", c_int64),
        ("target_scalar_dim", c_int64),
        ("render_plan_capacity", c_int64),
        ("dedup_card_bodies", c_int64),
        ("max_tokens", c_int64),
        ("max_card_refs", c_int64),
        ("max_blanks", c_int64),
        ("max_legal_per_blank", c_int64),
        ("ready_queue_capacity", c_int64),
        ("terminal_queue_capacity", c_int64),
    ]


class _MageTextReadyBatchResultC(Structure):
    _fields_ = [
        ("rows_written", c_int64),
        ("terminal_events_written", c_int64),
        ("decision_rows_written", c_int64),
        ("error_code", c_int64),
        ("error_message", c_char_p),
    ]


class _MageTextChoiceSubmitRequestC(Structure):
    _fields_ = [
        ("n", c_int64),
        ("request_ids", POINTER(c_int64)),
        ("decision_count", POINTER(c_int64)),
        ("selected_choice_cols", POINTER(c_int64)),
        ("may_selected", POINTER(c_int64)),
    ]


_step_lib_lock = threading.Lock()
_step_lib: ctypes.CDLL | None = None


def _load_step_ctypes_lib(mage: Any) -> ctypes.CDLL | None:
    """Open the mage shared library via ctypes for GIL-releasing step calls."""

    global _step_lib
    with _step_lib_lock:
        if _step_lib is not None:
            return _step_lib
        path = getattr(mage, "_lib_path_used", None)
        if not path:
            return None
        try:
            lib = ctypes.CDLL(path)
            lib.MageBatchStepByChoice.argtypes = [POINTER(_MageStepChoiceRequestC)]
            lib.MageBatchStepByChoice.restype = _MageEncodeResultC
            lib.MageFreeString.argtypes = [c_char_p]
            lib.MageFreeString.restype = None
        except OSError, AttributeError:
            return None
        _step_lib = lib
        return lib


def _load_text_rollout_ctypes_lib(mage: Any) -> ctypes.CDLL | None:
    lib = _load_step_ctypes_lib(mage)
    if lib is None:
        return None
    try:
        from magic_ai.slot_encoder.native_encoder import _MageEncodeOutputs
        from magic_ai.text_encoder.native_assembler import (
            _MagePackedBlankOutputsC,
            _MagePackedTokenAssemblerOutputsC,
        )

        class _MageTextReadyBatchOutputsC(Structure):
            _fields_ = [
                ("request_ids", POINTER(c_int64)),
                ("slot_ids", POINTER(c_int64)),
                ("episode_ids", POINTER(c_int64)),
                ("step_indices", POINTER(c_int64)),
                ("perspective_player_idx", POINTER(c_int64)),
                ("terminal_slot_ids", POINTER(c_int64)),
                ("terminal_episode_ids", POINTER(c_int64)),
                ("terminal_winner_idx", POINTER(c_int64)),
                ("terminal_is_timeout", POINTER(c_int64)),
                ("terminal_life_p0", POINTER(c_int64)),
                ("terminal_life_p1", POINTER(c_int64)),
                ("encode", _MageEncodeOutputs),
                ("packed_tokens", _MagePackedTokenAssemblerOutputsC),
                ("blanks", _MagePackedBlankOutputsC),
            ]

        lib.MageStartTextRollout.argtypes = [POINTER(_MageTextRolloutStartRequestC)]
        lib.MageStartTextRollout.restype = _MageEncodeResultC
        lib.MageNextTextInferenceBatch.argtypes = [
            c_int64,
            c_int64,
            POINTER(_MageTextReadyBatchOutputsC),
        ]
        lib.MageNextTextInferenceBatch.restype = _MageTextReadyBatchResultC
        lib.MageSubmitTextChoices.argtypes = [POINTER(_MageTextChoiceSubmitRequestC)]
        lib.MageSubmitTextChoices.restype = _MageEncodeResultC
        lib.MageStopTextRollout.argtypes = [ctypes.c_int32]
        lib.MageStopTextRollout.restype = ctypes.c_int32
        setattr(lib, "_magic_ai_text_ready_outputs_c", _MageTextReadyBatchOutputsC)
    except AttributeError:
        return None
    return lib


REQUIRED_NATIVE_ROLLOUT_SYMBOLS = (
    "MageBatchPoll",
    "MageBatchStepByChoice",
    "MageEncodeBatch",
    "MageFree",
    "MageFreeString",
)

OPTIONAL_STATUS_SYMBOLS = (
    "MageIsOver",
    "MagePendingPlayer",
    "MageWinner",
    "MageFreeString",
)

REQUIRED_TEXT_ROLLOUT_SYMBOLS = (
    "MageStartTextRollout",
    "MageNextTextInferenceBatch",
    "MageSubmitTextChoices",
    "MageStopTextRollout",
    "MageEncodeTokensPacked",
)


def _missing_symbols(lib: Any, names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            getattr(lib, name)
        except AttributeError:
            missing.append(name)
    return missing


@dataclass(frozen=True)
class TextTerminalEvent:
    slot_id: int
    episode_id: int
    winner_idx: int
    is_timeout: bool
    life_p0: int
    life_p1: int


@dataclass(frozen=True)
class TextReadyBatch:
    request_ids: torch.Tensor
    slot_ids: torch.Tensor
    episode_ids: torch.Tensor
    step_indices: torch.Tensor
    perspective_player_indices: torch.Tensor
    native_batch: Any
    packed_outputs: Any
    terminal_events: tuple[TextTerminalEvent, ...] = ()

    @property
    def rows(self) -> int:
        return int(self.request_ids.numel())


@dataclass(frozen=True)
class NativeMageStatus:
    """Small wrapper for non-JSON scalar status APIs exposed by mage-go."""

    lib: Any
    ffi: Any

    @classmethod
    def for_mage(cls, mage: Any) -> NativeMageStatus | None:
        try:
            if mage._lib is None or mage._ffi is None:
                mage.load()
            lib = mage._lib
            ffi = mage._ffi
        except Exception:
            return None
        if lib is None or ffi is None or _missing_symbols(lib, OPTIONAL_STATUS_SYMBOLS):
            return None
        return cls(lib=lib, ffi=ffi)

    def is_over(self, game: Any) -> bool:
        return bool(self.lib.MageIsOver(int(game.handle)))

    def pending_player(self, game: Any) -> int:
        return int(self.lib.MagePendingPlayer(int(game.handle)))

    def winner(self, game: Any) -> str:
        raw = self.lib.MageWinner(int(game.handle))
        if raw == self.ffi.NULL:
            return ""
        try:
            return self.ffi.string(raw).decode("utf-8")
        finally:
            self.lib.MageFreeString(raw)


@dataclass
class NativeRolloutDriver:
    """Capability guard for the planned no-JSON rollout API.

    The current mage Python package still requires JSON for legal requests and
    stepping. This class names the native symbols the hot path needs before we
    can remove JSON entirely from PPO rollout.
    """

    lib: Any
    ffi: Any
    # Populated by __post_init__ / for_mage via object.__setattr__ since the
    # dataclass is not actually frozen but the assignment-by-init pattern is
    # only used for `lib` and `ffi`.
    _step_ctypes_lib: ctypes.CDLL | None = None
    _step_handles_np: Any = None
    _step_starts_np: Any = None
    _step_counts_np: Any = None
    _step_may_np: Any = None
    _step_selected_np: Any = None
    _step_request_c: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_poll_capacity", 0)
        object.__setattr__(self, "_poll_handles", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_poll_ready", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_poll_game_over", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_poll_pending_player_idx", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_poll_winner_player_idx", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_step_capacity", 0)
        object.__setattr__(self, "_step_handles", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_step_starts", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_step_counts", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_step_selected_capacity", 0)
        object.__setattr__(self, "_step_selected", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_step_may", torch.empty((0,), dtype=torch.int64))
        object.__setattr__(self, "_step_ctypes_lib", None)
        object.__setattr__(self, "_step_handles_np", self._step_handles.numpy())
        object.__setattr__(self, "_step_starts_np", self._step_starts.numpy())
        object.__setattr__(self, "_step_counts_np", self._step_counts.numpy())
        object.__setattr__(self, "_step_may_np", self._step_may.numpy())
        object.__setattr__(self, "_step_selected_np", self._step_selected.numpy())
        object.__setattr__(self, "_step_request_c", None)

    @classmethod
    def for_mage(cls, mage: Any) -> NativeRolloutDriver:
        try:
            if mage._lib is None or mage._ffi is None:
                mage.load()
            lib = mage._lib
            ffi = mage._ffi
        except Exception as exc:
            raise NativeRolloutUnavailable("failed to load mage native library") from exc
        if lib is None or ffi is None:
            raise NativeRolloutUnavailable("mage native library is not loaded")
        missing = _missing_symbols(lib, REQUIRED_NATIVE_ROLLOUT_SYMBOLS)
        if missing:
            raise NativeRolloutUnavailable(
                "native no-JSON rollout requires missing mage symbols: " + ", ".join(missing)
            )
        driver = cls(lib=lib, ffi=ffi)
        object.__setattr__(driver, "_step_ctypes_lib", _load_step_ctypes_lib(mage))
        return driver

    @staticmethod
    def require_available(mage: Any) -> None:
        NativeRolloutDriver.for_mage(mage)

    def poll(
        self, games: list[Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(games)
        if n > self._poll_capacity:
            self._poll_capacity = n
            self._poll_handles = torch.empty((n,), dtype=torch.int64)
            self._poll_ready = torch.empty((n,), dtype=torch.int64)
            self._poll_game_over = torch.empty((n,), dtype=torch.int64)
            self._poll_pending_player_idx = torch.empty((n,), dtype=torch.int64)
            self._poll_winner_player_idx = torch.empty((n,), dtype=torch.int64)
        handles = self._poll_handles[:n]
        ready = self._poll_ready[:n]
        game_over = self._poll_game_over[:n]
        pending_player_idx = self._poll_pending_player_idx[:n]
        winner_player_idx = self._poll_winner_player_idx[:n]
        handles.copy_(torch.tensor([int(game.handle) for game in games], dtype=torch.int64))
        req = self.ffi.new(
            "MageBatchRequest *",
            {
                "n": n,
                "handles": self.ffi.cast("int64_t *", handles.data_ptr()),
                "perspective_player_idx": self.ffi.NULL,
            },
        )
        out = self.ffi.new(
            "MageBatchPollOutputs *",
            {
                "ready": self.ffi.cast("int64_t *", ready.data_ptr()),
                "game_over": self.ffi.cast("int64_t *", game_over.data_ptr()),
                "pending_player_idx": self.ffi.cast("int64_t *", pending_player_idx.data_ptr()),
                "winner_player_idx": self.ffi.cast("int64_t *", winner_player_idx.data_ptr()),
            },
        )
        result = self.lib.MageBatchPoll(req, out)
        self._raise_for_result(result, "MageBatchPoll")
        return ready, game_over, pending_player_idx, winner_player_idx

    def _ensure_step_capacity(self, n: int, selected_n: int) -> None:
        # Allocate scratch + rebuild cached ctypes/cffi request structs only
        # when capacity grows. Pointers into scratch are stable while the
        # underlying tensors don't get reallocated, so subsequent calls can
        # reuse the cached request objects and just bump the scalar fields.
        grew = False
        if n > self._step_capacity:
            self._step_capacity = max(n, self._step_capacity * 2 or 64)
            self._step_handles = torch.empty((self._step_capacity,), dtype=torch.int64)
            self._step_starts = torch.empty((self._step_capacity,), dtype=torch.int64)
            self._step_counts = torch.empty((self._step_capacity,), dtype=torch.int64)
            self._step_may = torch.empty((self._step_capacity,), dtype=torch.int64)
            object.__setattr__(self, "_step_handles_np", self._step_handles.numpy())
            object.__setattr__(self, "_step_starts_np", self._step_starts.numpy())
            object.__setattr__(self, "_step_counts_np", self._step_counts.numpy())
            object.__setattr__(self, "_step_may_np", self._step_may.numpy())
            grew = True
        if selected_n > self._step_selected_capacity:
            self._step_selected_capacity = max(selected_n, self._step_selected_capacity * 2 or 256)
            self._step_selected = torch.empty((self._step_selected_capacity,), dtype=torch.int64)
            object.__setattr__(self, "_step_selected_np", self._step_selected.numpy())
            grew = True
        if grew and self._step_ctypes_lib is not None:
            object.__setattr__(
                self,
                "_step_request_c",
                _MageStepChoiceRequestC(
                    n=0,
                    max_options=0,
                    max_targets_per_option=0,
                    handles=ctypes.cast(self._step_handles.data_ptr(), POINTER(c_int64)),
                    decision_start=ctypes.cast(self._step_starts.data_ptr(), POINTER(c_int64)),
                    decision_count=ctypes.cast(self._step_counts.data_ptr(), POINTER(c_int64)),
                    selected_choice_cols=ctypes.cast(
                        self._step_selected.data_ptr(), POINTER(c_int64)
                    ),
                    may_selected=ctypes.cast(self._step_may.data_ptr(), POINTER(c_int64)),
                ),
            )

    def step_by_choice(
        self,
        games: list[Any],
        *,
        decision_starts: list[int],
        decision_counts: list[int],
        selected_choice_cols: list[int],
        may_selected: list[int],
        max_options: int,
        max_targets_per_option: int,
    ) -> None:
        n = len(games)
        selected_n = len(selected_choice_cols)
        self._ensure_step_capacity(n, selected_n)

        # Fill scratch in place via numpy views — avoids 5x torch.tensor(...)
        # allocations and the extra .copy_ per call. handles uses Game._id
        # directly to skip the property lookup overhead.
        handles_np = self._step_handles_np
        for i, g in enumerate(games):
            handles_np[i] = g._id
        self._step_starts_np[:n] = decision_starts
        self._step_counts_np[:n] = decision_counts
        self._step_may_np[:n] = may_selected
        if selected_n:
            self._step_selected_np[:selected_n] = selected_choice_cols

        ctypes_lib = self._step_ctypes_lib
        if ctypes_lib is not None:
            # ctypes path: releases the GIL across the C call so that worker
            # threads in ShardedNativeRolloutDriver can run the engine in
            # parallel. cffi (ABI mode) holds the GIL, which serialises them.
            req = self._step_request_c
            req.n = n
            req.max_options = max_options
            req.max_targets_per_option = max_targets_per_option
            result_c = ctypes_lib.MageBatchStepByChoice(byref(req))
            if result_c.error_code != 0:
                msg = "MageBatchStepByChoice failed"
                if result_c.error_message:
                    try:
                        msg = result_c.error_message.decode("utf-8")
                    finally:
                        ctypes_lib.MageFreeString(result_c.error_message)
                raise NativeRolloutUnavailable(msg)
            return

        handles = self._step_handles[:n]
        starts = self._step_starts[:n]
        counts = self._step_counts[:n]
        may = self._step_may[:n]
        selected = self._step_selected[:selected_n]

        req = self.ffi.new(
            "MageStepChoiceRequest *",
            {
                "n": n,
                "max_options": max_options,
                "max_targets_per_option": max_targets_per_option,
                "handles": self.ffi.cast("int64_t *", handles.data_ptr()),
                "decision_start": self.ffi.cast("int64_t *", starts.data_ptr()),
                "decision_count": self.ffi.cast("int64_t *", counts.data_ptr()),
                "selected_choice_cols": self.ffi.cast("int64_t *", selected.data_ptr()),
                "may_selected": self.ffi.cast("int64_t *", may.data_ptr()),
            },
        )
        result = self.lib.MageBatchStepByChoice(req)
        self._raise_for_result(result, "MageBatchStepByChoice")

    def _raise_for_result(self, result: Any, fn_name: str) -> None:
        if int(result.error_code) == 0:
            return
        message = f"{fn_name} failed"
        if result.error_message != self.ffi.NULL:
            try:
                message = self.ffi.string(result.error_message).decode("utf-8")
            finally:
                self.lib.MageFreeString(result.error_message)
        raise NativeRolloutUnavailable(message)


@dataclass
class NativeTextRolloutDriver:
    """ctypes wrapper for the Go-owned native text rollout scheduler."""

    lib: ctypes.CDLL

    @classmethod
    def for_mage(cls, mage: Any) -> NativeTextRolloutDriver:
        try:
            if mage._lib is None or mage._ffi is None:
                mage.load()
            cffi_lib = mage._lib
        except Exception as exc:
            raise NativeRolloutUnavailable("failed to load mage native library") from exc
        if cffi_lib is None:
            raise NativeRolloutUnavailable("mage native library is not loaded")
        lib = _load_text_rollout_ctypes_lib(mage)
        if lib is None:
            raise NativeRolloutUnavailable("failed to load native text rollout ctypes library")
        missing = _missing_symbols(lib, REQUIRED_TEXT_ROLLOUT_SYMBOLS)
        if missing:
            raise NativeRolloutUnavailable(
                "native text rollout scheduler requires missing mage symbols: " + ", ".join(missing)
            )
        return cls(lib=lib)

    def start(
        self,
        games: list[Any],
        *,
        slot_ids: list[int],
        episode_ids: list[int],
        encoder: Any,
        max_steps_per_game: int,
        max_tokens: int,
        max_card_refs: int = 256,
        max_blanks: int = 64,
        max_legal_per_blank: int = 64,
        ready_queue_capacity: int | None = None,
        terminal_queue_capacity: int | None = None,
    ) -> None:
        if len(slot_ids) != len(games) or len(episode_ids) != len(games):
            raise ValueError("games, slot_ids, and episode_ids length mismatch")
        handles = torch.tensor(
            [int(getattr(g, "_id", None) or g.handle) for g in games],
            dtype=torch.int64,
        )
        slots = torch.tensor(slot_ids, dtype=torch.int64)
        episodes = torch.tensor(episode_ids, dtype=torch.int64)
        req = _MageTextRolloutStartRequestC(
            n=len(games),
            handles=ctypes.cast(handles.data_ptr(), POINTER(c_int64)),
            slot_ids=ctypes.cast(slots.data_ptr(), POINTER(c_int64)),
            episode_ids=ctypes.cast(episodes.data_ptr(), POINTER(c_int64)),
            max_steps_per_game=int(max_steps_per_game),
            max_options=int(encoder.max_options),
            max_targets_per_option=int(encoder.max_targets_per_option),
            max_cached_choices=int(encoder.max_cached_choices),
            zone_slot_count=int(encoder.zone_slot_count),
            game_info_dim=int(encoder.game_info_dim),
            option_scalar_dim=int(encoder.option_scalar_dim),
            target_scalar_dim=int(encoder.target_scalar_dim),
            render_plan_capacity=int(encoder.render_plan_capacity),
            dedup_card_bodies=1 if bool(getattr(encoder, "dedup_card_bodies", False)) else 0,
            max_tokens=int(max_tokens),
            max_card_refs=int(max_card_refs),
            max_blanks=int(max_blanks),
            max_legal_per_blank=int(max_legal_per_blank),
            ready_queue_capacity=(
                len(games) if ready_queue_capacity is None else int(ready_queue_capacity)
            ),
            terminal_queue_capacity=(
                len(games) if terminal_queue_capacity is None else int(terminal_queue_capacity)
            ),
        )
        result = self.lib.MageStartTextRollout(ctypes.byref(req))
        self._raise_for_ctypes_result(result, "MageStartTextRollout")

    def next_text_inference_batch(
        self,
        encoder: Any,
        *,
        max_rows: int,
        timeout_ms: int,
        max_tokens: int,
        max_card_refs: int = 256,
        max_blanks: int = 64,
        max_legal_per_blank: int = 64,
    ) -> TextReadyBatch:
        from magic_ai.actions import TRACE_KIND_VALUES
        from magic_ai.slot_encoder.native_encoder import NativeEncodedBatch
        from magic_ai.text_encoder.native_assembler import (
            _MageBlankAssemblerConfigC,
            _MagePackedBlankOutputsC,
            _MagePackedTokenAssemblerOutputsC,
            _MageTokenAssemblerConfigC,
            _tensor_ptr,
            allocate_packed_outputs,
        )

        n = int(max_rows)
        if n < 0:
            raise ValueError("max_rows must be non-negative")
        if n == 0:
            raise ValueError("max_rows must be positive")

        decision_capacity = max(1, n * int(encoder.max_options))
        buffers = encoder._scratch_buffers(n, decision_capacity)
        if encoder._scratch.out_c is None:
            encoder._rebuild_encode_structs()
        scratch = encoder._scratch
        assert scratch.out_c is not None

        packed_outputs = allocate_packed_outputs(
            n,
            max_tokens=max_tokens,
            max_options=encoder.max_options,
            max_targets=encoder.max_targets_per_option,
            max_card_refs=max_card_refs,
            max_blanks=max_blanks,
            max_legal_per_blank=max_legal_per_blank,
        )
        packed_outputs._tok_cfg_ctypes = _MageTokenAssemblerConfigC(
            max_tokens=max_tokens,
            max_options=encoder.max_options,
            max_targets=encoder.max_targets_per_option,
            max_card_refs=max_card_refs,
        )
        packed_outputs._packed_out_ctypes = _MagePackedTokenAssemblerOutputsC(
            token_ids=_tensor_ptr(packed_outputs.token_ids, ctypes.c_int32),
            cu_seqlens=_tensor_ptr(packed_outputs.cu_seqlens, ctypes.c_int32),
            seq_lengths=_tensor_ptr(packed_outputs.seq_lengths, ctypes.c_int32),
            state_positions=_tensor_ptr(packed_outputs.state_positions, ctypes.c_int32),
            card_ref_positions=_tensor_ptr(packed_outputs.card_ref_positions, ctypes.c_int32),
            token_overflow=_tensor_ptr(packed_outputs.token_overflow, ctypes.c_int32),
        )
        packed_outputs._blank_cfg_ctypes = _MageBlankAssemblerConfigC(
            max_blanks=max_blanks,
            max_legal_per_blank=max_legal_per_blank,
        )
        packed_outputs._blank_out_ctypes = _MagePackedBlankOutputsC(
            k_max=max_blanks,
            v_max=max_legal_per_blank,
            blank_positions=_tensor_ptr(packed_outputs.blank_positions, ctypes.c_int32),
            blank_kind=_tensor_ptr(packed_outputs.blank_kind, ctypes.c_int32),
            blank_group=_tensor_ptr(packed_outputs.blank_group, ctypes.c_int32),
            blank_group_kind=_tensor_ptr(packed_outputs.blank_group_kind, ctypes.c_int32),
            blank_option_index=_tensor_ptr(packed_outputs.blank_option_index, ctypes.c_int32),
            blank_legal_ids=_tensor_ptr(packed_outputs.blank_legal_ids, ctypes.c_int32),
            blank_legal_mask=_tensor_ptr(packed_outputs.blank_legal_mask, ctypes.c_uint8),
            blank_overflow=_tensor_ptr(packed_outputs.blank_overflow, ctypes.c_int32),
            blank_count=_tensor_ptr(packed_outputs.blank_count, ctypes.c_int32),
            blank_legal_count=_tensor_ptr(packed_outputs.blank_legal_count, ctypes.c_int32),
        )

        request_ids = torch.empty(n, dtype=torch.int64)
        slot_ids = torch.empty(n, dtype=torch.int64)
        episode_ids = torch.empty(n, dtype=torch.int64)
        step_indices = torch.empty(n, dtype=torch.int64)
        perspectives = torch.empty(n, dtype=torch.int64)
        terminal_slot_ids = torch.empty(n, dtype=torch.int64)
        terminal_episode_ids = torch.empty(n, dtype=torch.int64)
        terminal_winner = torch.empty(n, dtype=torch.int64)
        terminal_timeout = torch.empty(n, dtype=torch.int64)
        terminal_life_p0 = torch.empty(n, dtype=torch.int64)
        terminal_life_p1 = torch.empty(n, dtype=torch.int64)

        outputs_c_type = getattr(self.lib, "_magic_ai_text_ready_outputs_c")
        out = outputs_c_type(
            request_ids=ctypes.cast(request_ids.data_ptr(), POINTER(c_int64)),
            slot_ids=ctypes.cast(slot_ids.data_ptr(), POINTER(c_int64)),
            episode_ids=ctypes.cast(episode_ids.data_ptr(), POINTER(c_int64)),
            step_indices=ctypes.cast(step_indices.data_ptr(), POINTER(c_int64)),
            perspective_player_idx=ctypes.cast(perspectives.data_ptr(), POINTER(c_int64)),
            terminal_slot_ids=ctypes.cast(terminal_slot_ids.data_ptr(), POINTER(c_int64)),
            terminal_episode_ids=ctypes.cast(terminal_episode_ids.data_ptr(), POINTER(c_int64)),
            terminal_winner_idx=ctypes.cast(terminal_winner.data_ptr(), POINTER(c_int64)),
            terminal_is_timeout=ctypes.cast(terminal_timeout.data_ptr(), POINTER(c_int64)),
            terminal_life_p0=ctypes.cast(terminal_life_p0.data_ptr(), POINTER(c_int64)),
            terminal_life_p1=ctypes.cast(terminal_life_p1.data_ptr(), POINTER(c_int64)),
            encode=scratch.out_c,
            packed_tokens=packed_outputs._packed_out_ctypes,
            blanks=packed_outputs._blank_out_ctypes,
        )
        result = self.lib.MageNextTextInferenceBatch(n, int(timeout_ms), ctypes.byref(out))
        self._raise_for_ctypes_result(result, "MageNextTextInferenceBatch")

        rows = int(result.rows_written)
        terminals_n = int(result.terminal_events_written)
        packed_outputs.active_batch_size = rows
        batch = encoder._slice_batch_buffers(buffers, rows)
        trace_kind_id = batch["trace_kind_id"]
        trace_kinds = cast(
            list[str], [TRACE_KIND_VALUES[int(idx)] for idx in trace_kind_id.tolist()]
        )
        decision_rows = int(result.decision_rows_written)
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
            decision_option_idx=buffers.decision_option_idx[:decision_rows],
            decision_target_idx=buffers.decision_target_idx[:decision_rows],
            decision_mask=buffers.decision_mask_u8[:decision_rows],
            uses_none_head=buffers.uses_none_head_u8[:decision_rows],
            decision_rows_written=decision_rows,
            pendings=[],
            trace_kinds=trace_kinds,
            render_plan=None,
            render_plan_lengths=None,
            render_plan_overflow=None,
        )
        terminals = tuple(
            TextTerminalEvent(
                slot_id=int(terminal_slot_ids[i]),
                episode_id=int(terminal_episode_ids[i]),
                winner_idx=int(terminal_winner[i]),
                is_timeout=bool(int(terminal_timeout[i])),
                life_p0=int(terminal_life_p0[i]),
                life_p1=int(terminal_life_p1[i]),
            )
            for i in range(terminals_n)
        )
        return TextReadyBatch(
            request_ids=request_ids[:rows],
            slot_ids=slot_ids[:rows],
            episode_ids=episode_ids[:rows],
            step_indices=step_indices[:rows],
            perspective_player_indices=perspectives[:rows],
            native_batch=native_batch,
            packed_outputs=packed_outputs,
            terminal_events=terminals,
        )

    def submit_text_choices(
        self,
        *,
        request_ids: list[int],
        decision_counts: list[int],
        selected_choice_cols: list[int],
        may_selected: list[int],
    ) -> None:
        n = len(request_ids)
        if len(decision_counts) != n or len(may_selected) != n:
            raise ValueError("request_ids, decision_counts, and may_selected length mismatch")
        if sum(decision_counts) != len(selected_choice_cols):
            raise ValueError("selected_choice_cols length must equal sum(decision_counts)")
        request_t = torch.tensor(request_ids, dtype=torch.int64)
        count_t = torch.tensor(decision_counts, dtype=torch.int64)
        selected_t = torch.tensor(selected_choice_cols or [0], dtype=torch.int64)
        may_t = torch.tensor(may_selected, dtype=torch.int64)
        req = _MageTextChoiceSubmitRequestC(
            n=n,
            request_ids=ctypes.cast(request_t.data_ptr(), POINTER(c_int64)),
            decision_count=ctypes.cast(count_t.data_ptr(), POINTER(c_int64)),
            selected_choice_cols=ctypes.cast(selected_t.data_ptr(), POINTER(c_int64)),
            may_selected=ctypes.cast(may_t.data_ptr(), POINTER(c_int64)),
        )
        result = self.lib.MageSubmitTextChoices(ctypes.byref(req))
        self._raise_for_ctypes_result(result, "MageSubmitTextChoices")

    def stop_text_rollout(self, *, wait_for_active: bool = True) -> None:
        self.lib.MageStopTextRollout(1 if wait_for_active else 0)

    def _raise_for_ctypes_result(self, result: Any, fn_name: str) -> None:
        if int(result.error_code) == 0:
            return
        message = f"{fn_name} failed"
        if result.error_message:
            try:
                message = result.error_message.decode("utf-8")
            finally:
                self.lib.MageFreeString(result.error_message)
        raise NativeRolloutUnavailable(message)
