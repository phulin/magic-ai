"""Compatibility imports for shared native sharding helpers.

The implementation lives in :mod:`magic_ai.native.sharded` because the
batched native encoder and rollout sharding are shared by slot and text
encoder training paths.
"""

from magic_ai.native.sharded import (
    ShardedNativeBatchEncoder,
    ShardedNativeRolloutDriver,
    _concat_encoded_batches,
    _merge_packed_outputs,
    _shard_ranges,
)

__all__ = [
    "ShardedNativeBatchEncoder",
    "ShardedNativeRolloutDriver",
    "_concat_encoded_batches",
    "_merge_packed_outputs",
    "_shard_ranges",
]
