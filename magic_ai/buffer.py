"""Re-export shim — buffers moved to ``magic_ai.slot_encoder.buffer``.

External code that imports from ``magic_ai.buffer`` continues to work; only the
definition has moved.
"""

from magic_ai.slot_encoder.buffer import (  # noqa: F401
    BufferWrite,
    NativeTrajectoryBuffer,
    RolloutBuffer,
)

__all__ = [
    "BufferWrite",
    "NativeTrajectoryBuffer",
    "RolloutBuffer",
]
