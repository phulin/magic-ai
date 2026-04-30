# magic_ai/slot_encoder/

Slot-based game-state encoding, PPO actor-critic, rollout buffers, and native Cgo bindings for the slot-encoder backend.

## Files
- `__init__.py` — package docstring; submodule list (no re-exports to avoid circular imports)
- `buffer.py` — preallocated GPU buffers for rollout trajectory data (RolloutBuffer, NativeTrajectoryBuffer)
- `encoder_parity.py` — parity validators comparing native batch encoding against Python reference
- `game_state.py` — GameStateEncoder and private helpers (_build_card_embedding_table, _fill_game_info, etc.); shared types imported from magic_ai.game_state
- `model.py` — PPOPolicy actor-critic with policy/value heads and action decoding for legal action spaces
- `native_encoder.py` — ctypes bindings to libmage's parallel batch encoder (game→features)
- `native_rollout.py` — ctypes bindings to libmage's step-by-choice game simulator
- `sharded_native.py` — thread-pool sharding for parallel encoder/rollout calls (ShardedNativeBatchEncoder, ShardedNativeRolloutDriver)
