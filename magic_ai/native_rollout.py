from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class NativeRolloutUnavailable(RuntimeError):
    pass


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


def _missing_symbols(lib: Any, names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            getattr(lib, name)
        except AttributeError:
            missing.append(name)
    return missing


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
        return cls(lib=lib, ffi=ffi)

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
        if n > self._step_capacity:
            self._step_capacity = n
            self._step_handles = torch.empty((n,), dtype=torch.int64)
            self._step_starts = torch.empty((n,), dtype=torch.int64)
            self._step_counts = torch.empty((n,), dtype=torch.int64)
            self._step_may = torch.empty((n,), dtype=torch.int64)
        if selected_n > self._step_selected_capacity:
            self._step_selected_capacity = selected_n
            self._step_selected = torch.empty((selected_n,), dtype=torch.int64)
        handles = self._step_handles[:n]
        starts = self._step_starts[:n]
        counts = self._step_counts[:n]
        may = self._step_may[:n]
        selected = self._step_selected[:selected_n]
        handles.copy_(torch.tensor([int(game.handle) for game in games], dtype=torch.int64))
        starts.copy_(torch.tensor(decision_starts, dtype=torch.int64))
        counts.copy_(torch.tensor(decision_counts, dtype=torch.int64))
        may.copy_(torch.tensor(may_selected, dtype=torch.int64))
        if selected_n:
            selected.copy_(torch.tensor(selected_choice_cols, dtype=torch.int64))
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
