"""Slot-encoder backend for the Magic AI package.

Submodules:
- ``game_state`` — ``GameStateEncoder`` and game-state parsing helpers.
- ``model`` — ``PPOPolicy`` actor-critic for the slot-encoder backend.
- ``buffer`` — ``RolloutBuffer``, ``NativeTrajectoryBuffer`` GPU rollout buffers.
- ``native_encoder`` — ``NativeBatchEncoder``, ``NativeEncodedBatch`` (Cgo-backed).
- ``native_rollout`` — ``NativeRolloutDriver``, ``NativeRolloutUnavailable``.
- ``sharded_native`` — compatibility re-export for shared native sharding helpers.
- ``encoder_parity`` — parity helpers for validating batch encoding.
"""
