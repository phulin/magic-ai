# magic_ai/text_encoder/

Bidirectional text-encoder for game state and actions. Core pipeline: game snapshot → rendered text → tokenized batch → ModernBERT trunk → policy/target/value heads. Includes fast hot-path assembler with native Go acceleration, replay buffer, training scaffolding, and Python rollout worker.

## Files
- `__init__.py` — Package exports for the tokenizer API.
- `actor_critic.py` — Training-facing actor-critic wrapper bridging policy outputs to action supervision.
- `assembler.py` — Render-plan → TextEncodedBatch converter; walks opcodes and memcpys cached card tokens.
- `batch.py` — Tokenize snapshots and collate into padded batches with position-anchored gather masks.
- `card_cache.py` — Pre-tokenized card-body cache (Name/Type/P/T/oracle) keyed by engine card-row IDs.
- `model.py` — ModernBERT encoder trunk with position-anchored gather pools and policy/target/value heads.
- `native_assembler.py` — Python wrapper for native Go MageEncodeTokens assembler; manages tensor I/O.
- `native_token_tables.py` — Serialize TokenTables into flat buffers and register with Go-side mage lib.
- `policy.py` — Self-contained text-encoder policy facade; exports encode_snapshots and forward.
- `recurrent.py` — LSTM history adapter wrapping TextPolicy; carries recurrence through encoder state vector.
- `render.py` — Deterministic GameStateSnapshot → text renderer; produces custom-token-laced strings and anchor metadata.
- `render_plan.py` — Opcode definitions, writer, and Python emitter for render-plan ABI.
- `replay_buffer.py` — Frozen dataclass storing encoded snapshots plus policy/value/trace metadata for training.
- `replay_triton.py` — Optional CUDA/Triton kernels for packed replay-buffer batch append writes and position rebasing.
- `rollout.py` — End-to-end Python Magic game player using text encoder and RecurrentTextPolicy.
- `token_tables.py` — Closed-vocabulary token-id lookup tables for every assembler emission; source of truth for native side.
- `tokenizer.py` — ModernBERT BPE tokenizer augmented with custom mana/card-ref/status/structural tokens.
- `training.py` — Value pretrain and BC distillation scaffolding; pure PyTorch losses and TextEncoderTrainer.
