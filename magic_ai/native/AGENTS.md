# magic_ai/native/

Shared native-runtime orchestration used by multiple policy encoders.

## Files
- `__init__.py` — package marker for shared native helpers.
- `sharded.py` — thread-pool sharding for parallel native encoder/rollout calls (ShardedNativeBatchEncoder, ShardedNativeRolloutDriver).
- `inference_server.py` — IMPALA-style GPU inference server: accepts actor-produced encoded text rollout chunks into contiguous arena storage, dynamic-batches descriptors into one forward pass, and scatters per-request host scalars/replay payloads back via futures. Pause/resume API for PPO update windows.
- `rollout_actor.py` — TextRolloutActor: per-thread CPU rollout driver owning a disjoint env slice, its own NativeBatchEncoder+NativeRolloutDriver, and a handle to the inference server. Polls envs, encodes via packed-token assembler, submits to server, applies returned actions via step_by_choice. Coordinates env lifecycle with the learner via finished/refill queues.
