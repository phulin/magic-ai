"""Re-export shim — moved to ``magic_ai.slot_encoder.native_encoder``.

External code that imports from ``magic_ai.native_encoder`` continues to work;
only the definition has moved.
"""

from magic_ai.slot_encoder.native_encoder import (  # noqa: F401
    NativeBatchEncoder,
    NativeEncodedBatch,
    NativeEncodingError,
    _validate_decision_layout,
)

__all__ = [
    "NativeBatchEncoder",
    "NativeEncodedBatch",
    "NativeEncodingError",
    "_validate_decision_layout",
]
