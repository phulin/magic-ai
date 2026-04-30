"""Re-export shim — moved to ``magic_ai.slot_encoder.native_rollout``.

External code that imports from ``magic_ai.native_rollout`` continues to work;
only the definition has moved.
"""

from magic_ai.slot_encoder.native_rollout import (  # noqa: F401
    NativeMageStatus,
    NativeRolloutDriver,
    NativeRolloutUnavailable,
)

__all__ = [
    "NativeMageStatus",
    "NativeRolloutDriver",
    "NativeRolloutUnavailable",
]
