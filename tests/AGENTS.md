# tests/

Comprehensive test suite covering text encoding (rendering, tokenization, assembler), neural network components (LSTM, encoders, policy), reinforcement learning (R-NaD primitives and NFG convergence), and end-to-end training integration.

## Files

- `conftest.py` — Pytest configuration and path setup.
- `fixtures/` — Static input/output artifacts used by tests.
- `test_card_cache.py` — Card cache build, roundtrip, slice extraction, hash stability, and parity with slow renderer.
- `test_actions.py` — Tests for shared action decoding helpers and trace/action payload construction.
- `test_lstm_recompute_strategies.py` — Correctness and performance comparison for LSTM recompute strategies (pad/gather/packed/legacy).
- `test_native_encoder.py` — Unit tests for native batch encoder and decision-layout validation.
- `test_native_rollout.py` — Tests for native rollout driver library availability checks.
- `test_native_token_tables.py` — Phase 3 round-trip parity test: TokenTables via Go FFI and echo back.
- `test_opponent_pool.py` — Tests for opponent snapshot state filtering and runtime-buffer-safe weight loading.
- `test_policy_value_pretrain_arrow.py` — Tests Forge policy/value pretraining input loading for spec-shaped Arrow IPC corpora.
- `test_ppo.py` — Unit tests for PPO replay minibatch construction and token-capped batching.
- `test_recurrent_text_policy.py` — Smoke tests for v1 history adapter with LSTM around state vector.
- `test_replay_buffer.py` — Unit tests for backend-neutral ReplayCore row, decision, PPO target, and common metadata storage.
- `test_replay_decisions.py` — Unit tests for replay scoring, decision logits, and Bernoulli scoring helpers.
- `test_rnad.py` — Unit tests for R-NaD primitives (v-trace, NeuRD loss, critic loss, reward transform).
- `test_rnad_coverage.py` — Coverage tests for four open R-NaD claims (v-trace end-to-end, Polyak stabilization, beta-gate guard, dense NeuRD).
- `test_rnad_nfg.py` — Toy normal-form-game convergence test (matching pennies) validating R-NaD outer-loop algorithm.
- `test_rnad_trainer.py` — Integration tests for R-NaD trainer-state glue and checkpoint management.
- `test_sharded_native.py` — Tests for sharded native rollout driver batch concatenation and shard range splitting.
- `test_forge_target_encoding.py` — Tests for Forge observed-event → decoder-target translators; asserts each translator's output is grammar-legal under `next_mask` for its `DecisionType`.
- `test_text_actor_critic.py` — Smoke tests for text-encoder actor-critic layout and decision trace inference.
- `test_text_encoder_batch.py` — Tests for tokenization anchor positions and collation padding/masking.
- `test_text_encoder_model.py` — Forward/pooling/heads smoke test for text-encoder state and policy models.
- `test_text_encoder_packed.py` — Parity tests between padded and packed forward paths over text-state encoder.
- `test_text_encoder_tokenizer.py` — Round-trip tests for augmented ModernBERT tokenizer and custom tokens.
- `test_text_encoder_training.py` — Tests for distillation loss correctness, masking, and backprop through recurrent policy.
- `test_text_policy.py` — Smoke tests for `TextPolicy` end-to-end pipeline.
- `test_text_render.py` — Tests for game state renderer: determinism, card refs, zones, actions, busy mid-game scenarios.
- `test_text_replay_buffer.py` — Tests for replay buffer storage and batch packing.
- `test_text_rollout.py` — Smoke tests for cache/emitter/assembler/policy pipeline against live mage engine.
- `test_text_encoder_parity.py` — Parity between Python batch encoder and single-item reference implementations.
- `test_text_token_tables.py` — Parity test: precomputed token lists match live tokenizer output for each table entry.
- `test_train.py` — Integration tests for training script: slot/text backend setup, checkpoint loading, opponent pool, transcript snapshots.
