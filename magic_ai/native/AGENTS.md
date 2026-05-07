# magic_ai/native/

Shared native-runtime orchestration used by multiple policy encoders.

## Files
- `__init__.py` — package marker for shared native helpers.
- `sharded.py` — thread-pool sharding for parallel native encoder/rollout calls (ShardedNativeBatchEncoder, ShardedNativeRolloutDriver).
- `inference_server.py` — IMPALA-style GPU inference server: accepts actor-produced encoded text rollout chunks into contiguous arena storage, adaptive-batches descriptors into one forward pass, appends replay payloads directly into the global ring when configured, and scatters per-request host scalars/row ids back via futures.
- `policy_version.py` — double-buffered inference-policy handoff with atomic version publication for learner/update overlap.
- `rollout_actor.py` — TextRolloutActor: per-thread CPU rollout driver owning a disjoint env slice, its own NativeBatchEncoder+NativeRolloutDriver, and a handle to the inference server. Polls envs, keeps a row-targeted fraction in flight for GPU inference, records returned replay row ids, applies actions via step_by_choice, and coordinates env lifecycle with the learner via finished/refill queues.
