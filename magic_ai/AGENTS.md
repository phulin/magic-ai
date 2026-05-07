# magic_ai/

Core training and inference pipeline for a Magic: The Gathering-playing RL agent using PPO/R-NaD algorithms with native game encoding and rollout simulation.

## Subpackages
- `slot_encoder/` — slot-based game-state encoder, PPO actor-critic, rollout buffers, and native Cgo bindings
- `native/` — shared native-runtime orchestration for sharded encoder/rollout calls
- `text_encoder/` — alternative policy using card-text embeddings instead of token slots

## Files
- `__init__.py` — public API exports for actions, buffers, and game state
- `actions.py` — encode/decode legal action options; shared policy types (TraceKind, ActionTrace, PolicyStep, ParsedStep, ParsedBatch) and build_decision_layout_rows
- `game_state.py` — shared TypedDicts (GameStateSnapshot, PlayerState, PendingState, etc.), zone constants, and ParsedGameState/ParsedGameStateBatch
- `lstm_recompute.py` — four LSTM input-state recompute strategies for R-NaD trajectory loss
- `model_state.py` — shared state-dict key classification for runtime buffers and snapshot filtering
- `opponent_pool.py` — opponent pool with TrueSkill ratings and periodic evaluation
- `ppo.py` — PPO update helpers (advantages, gradient steps, loss computation)
- `replay_decisions.py` — backend-neutral decision-group scoring for replay policy evaluation
- `rollout.py` — shared RolloutStep, trainer stats, and terminal reward resolution helpers
- `rnad.py` — R-NaD primitives (reward transform, v-trace, NeuRD loss, discretization)
- `rnad_trainer.py` — R-NaD trainer state and batch-level update dispatch
- `replay_buffer.py` — backend-neutral ReplayCore for append-only replay rows, rollout metadata, PPO targets, recurrent inputs, and flat decisions
- `training_interfaces.py` — protocol definitions for policy implementations (PPO/R-NaD trainers)
