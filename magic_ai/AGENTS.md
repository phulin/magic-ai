# magic_ai/

Core training and inference pipeline for a Magic: The Gathering-playing RL agent using PPO/R-NaD algorithms with native game encoding and rollout simulation.

## Subpackages
- `slot_encoder/` — slot-based game-state encoder, PPO actor-critic, rollout buffers, and native Cgo bindings
- `text_encoder/` — alternative policy using card-text embeddings instead of token slots

## Files
- `__init__.py` — public API exports for actions, buffers, and game state
- `actions.py` — encode/decode legal action options; also defines shared policy types (TraceKind, ActionTrace, PolicyStep, ParsedStep, ParsedBatch, build_decision_layout_rows)
- `buffer.py` — re-export shim for `slot_encoder.buffer` (RolloutBuffer, NativeTrajectoryBuffer)
- `encoder_parity.py` — re-export shim for `slot_encoder.encoder_parity` (parity validators)
- `game_state.py` — shared TypedDicts, constants, and ParsedGameState/ParsedGameStateBatch; re-exports GameStateEncoder from slot_encoder.game_state
- `lstm_recompute.py` — four LSTM input-state recompute strategies for R-NaD trajectory loss
- `model.py` — re-export shim for `slot_encoder.model` (PPOPolicy) and `actions` (policy types)
- `native_encoder.py` — re-export shim for `slot_encoder.native_encoder` (NativeBatchEncoder, NativeEncodedBatch)
- `native_rollout.py` — re-export shim for `slot_encoder.native_rollout` (NativeRolloutDriver)
- `opponent_pool.py` — opponent pool with TrueSkill ratings and periodic evaluation
- `ppo.py` — PPO update helpers (advantages, gradient steps, loss computation)
- `replay_decisions.py` — backend-neutral decision-group scoring for replay policy evaluation
- `rnad.py` — R-NaD primitives (reward transform, v-trace, NeuRD loss, discretization)
- `rnad_trainer.py` — R-NaD trainer state and batch-level update dispatch
- `sharded_native.py` — re-export shim for `slot_encoder.sharded_native` (thread-pool sharding)
- `training_interfaces.py` — protocol definitions for policy implementations (PPO/R-NaD trainers)
