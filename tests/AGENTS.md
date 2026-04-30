# tests/

Comprehensive test suite covering text encoding (rendering, tokenization, assembler), neural network components (LSTM, encoders, policy), reinforcement learning (R-NaD primitives and NFG convergence), and end-to-end training integration.

## Files

- `conftest.py` — Pytest configuration and path setup.
- `test_assembler.py` — Hand-written opcode-decoder tests for `assemble_batch` without coupling to render-plan emission.
- `test_assembler_dedup.py` — Tests for card-body deduplication opcodes; validates token efficiency gains in v2 plan.
- `test_card_cache.py` — Card cache build, roundtrip, slice extraction, hash stability, and parity with slow renderer.
- `test_encoder_parity.py` — Parity between Python batch encoder and single-item reference implementations.
- `test_lstm_recompute_strategies.py` — Correctness and performance comparison for LSTM recompute strategies (pad/gather/packed/legacy).
- `test_native_assembler_parity.py` — Infrastructure for parity gate between Python and native (Go) assembler paths.
- `test_native_encoder.py` — Unit tests for native batch encoder and decision-layout validation.
- `test_native_rollout.py` — Tests for native rollout driver library availability checks.
- `test_native_token_tables.py` — Phase 3 round-trip parity test: TokenTables via Go FFI and echo back.
- `test_recurrent_text_policy.py` — Smoke tests for v1 history adapter with LSTM around state vector.
- `test_render_plan_parity.py` — Byte-for-byte parity between slow renderer and fast assembler paths.
- `test_render_plan_writer.py` — Round-trip test for int32 render-plan wire format independence.
- `test_replay_decisions.py` — Unit tests for replay scoring, decision logits, and Bernoulli scoring helpers.
- `test_rnad.py` — Unit tests for R-NaD primitives (v-trace, NeuRD loss, critic loss, reward transform).
- `test_rnad_coverage.py` — Coverage tests for four open R-NaD claims (v-trace end-to-end, Polyak stabilization, beta-gate guard, dense NeuRD).
- `test_rnad_nfg.py` — Toy normal-form-game convergence test (matching pennies) validating R-NaD outer-loop algorithm.
- `test_rnad_trainer.py` — Integration tests for R-NaD trainer-state glue and checkpoint management.
- `test_sharded_native.py` — Tests for sharded native rollout driver batch concatenation and shard range splitting.
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
- `test_token_tables.py` — Parity test: precomputed token lists match live tokenizer output for each table entry.
- `test_train.py` — Integration tests for training script: slot/text backend setup, checkpoint loading, opponent pool, transcript snapshots.
