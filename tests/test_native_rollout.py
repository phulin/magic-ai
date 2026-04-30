from __future__ import annotations

import unittest

from magic_ai.slot_encoder.native_rollout import (
    NativeMageStatus,
    NativeRolloutDriver,
    NativeRolloutUnavailable,
)


class _PartialLib:
    MageEncodeBatch = object()
    MageFree = object()
    MageFreeString = object()


class _FullLib:
    MageBatchPoll = object()
    MageBatchStepByChoice = object()
    MageEncodeBatch = object()
    MageFree = object()
    MageFreeString = object()


class _StatusLib:
    MageFreeString = object()
    MageIsOver = object()
    MagePendingPlayer = object()
    MageWinner = object()


class _Mage:
    def __init__(self, lib: object, ffi: object = object()) -> None:
        self._lib = lib
        self._ffi = ffi

    def load(self) -> None:
        return None


class NativeRolloutTests(unittest.TestCase):
    def test_native_rollout_reports_missing_symbols(self) -> None:
        with self.assertRaisesRegex(
            NativeRolloutUnavailable,
            "MageBatchPoll, MageBatchStepByChoice",
        ):
            NativeRolloutDriver.for_mage(_Mage(_PartialLib()))

    def test_native_rollout_accepts_required_symbols(self) -> None:
        driver = NativeRolloutDriver.for_mage(_Mage(_FullLib()))
        self.assertIsInstance(driver, NativeRolloutDriver)

    def test_native_status_is_optional_when_symbols_are_missing(self) -> None:
        self.assertIsNone(NativeMageStatus.for_mage(_Mage(_PartialLib())))

    def test_native_status_is_available_with_status_symbols(self) -> None:
        status = NativeMageStatus.for_mage(_Mage(_StatusLib()))
        self.assertIsInstance(status, NativeMageStatus)


if __name__ == "__main__":
    unittest.main()
