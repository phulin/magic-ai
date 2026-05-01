# magic_ai/native/

Shared native-runtime orchestration used by multiple policy encoders.

## Files
- `__init__.py` — package marker for shared native helpers.
- `sharded.py` — thread-pool sharding for parallel native encoder/rollout calls (ShardedNativeBatchEncoder, ShardedNativeRolloutDriver).

