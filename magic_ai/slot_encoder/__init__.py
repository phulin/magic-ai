"""Slot-encoder backend for the Magic AI package.

Submodules:
- ``game_state`` — ``GameStateEncoder`` and game-state parsing helpers.
- ``model`` — ``PPOPolicy`` actor-critic for the slot-encoder backend.
- ``buffer`` — ``RolloutBuffer``, ``NativeTrajectoryBuffer`` GPU rollout buffers.
- ``native_encoder`` — ``NativeBatchEncoder``, ``NativeEncodedBatch`` (Cgo-backed).
- ``native_rollout`` — ``NativeRolloutDriver``, ``NativeRolloutUnavailable``.
- ``sharded_native`` — ``ShardedNativeBatchEncoder``, ``ShardedNativeRolloutDriver``.
- ``encoder_parity`` — parity helpers for validating batch encoding.
"""
