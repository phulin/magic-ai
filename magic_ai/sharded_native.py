"""Re-export shim — moved to ``magic_ai.slot_encoder.sharded_native``.

External code that imports from ``magic_ai.sharded_native`` continues to work;
only the definition has moved.
"""

from magic_ai.slot_encoder.sharded_native import (  # noqa: F401
    ShardedNativeBatchEncoder,
    ShardedNativeRolloutDriver,
    _concat_encoded_batches,
    _shard_ranges,
)

__all__ = [
    "ShardedNativeBatchEncoder",
    "ShardedNativeRolloutDriver",
    "_concat_encoded_batches",
    "_shard_ranges",
]
