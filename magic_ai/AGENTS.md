# magic_ai/

Core training and inference pipeline for a Magic: The Gathering-playing RL agent using PPO/R-NaD algorithms with native game encoding and rollout simulation.

## Subpackages
- `text_encoder/` — alternative policy implementation using card text embeddings instead of token slots

## Files
- `__init__.py` — public API exports for actions, buffers, and game state
- `actions.py` — encode legal action options and decode selected actions from policy logits
- `buffer.py` — preallocated GPU buffers for rollout trajectory data (PPO replay)
- `encoder_parity.py` — parity validators comparing native batch encoding against Python reference
- `game_state.py` — typed game state format and vector encoder for frozen card embeddings
- `lstm_recompute.py` — four LSTM input-state recompute strategies for R-NaD trajectory loss
- `model.py` — actor-critic model with policy/value heads and action decoding for legal action spaces
- `native_encoder.py` — ctypes bindings to libmage's parallel batch encoder (game->features)
- `native_rollout.py` — ctypes bindings to libmage's step-by-choice game simulator
- `opponent_pool.py` — opponent pool with TrueSkill ratings and periodic evaluation
- `ppo.py` — PPO update helpers (advantages, gradient steps, loss computation)
- `replay_decisions.py` — backend-neutral decision-group scoring for replay policy evaluation
- `rnad.py` — R-NaD primitives (reward transform, v-trace, NeuRD loss, discretization)
- `rnad_trainer.py` — R-NaD trainer state and batch-level update dispatch
- `sharded_native.py` — thread-pool sharding for parallel encoder/rollout calls
- `training_interfaces.py` — protocol definitions for policy implementations (PPO/R-NaD trainers)
