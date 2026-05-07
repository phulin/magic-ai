# magic_ai/text_encoder/

Bidirectional text-encoder for game state and actions. Core pipeline: game snapshot → rendered text with inline decision blanks → tokenized batch → ModernBERT trunk → tied inline-blank scorer and value head. Includes fast hot-path assembler with native Go acceleration, replay buffer, training scaffolding, and Python rollout worker.

## Files
- `__init__.py` — Package exports for the tokenizer API.
- `actor_critic.py` — Training-facing actor-critic wrapper bridging policy outputs to action supervision.
- `assembler.py` — Render-plan → TextEncodedBatch converter; walks opcodes and memcpys cached card tokens.
- `batch.py` — Tokenize snapshots and collate into padded batches with card-ref positions and inline-blank metadata.
- `card_cache.py` — Pre-tokenized card-body cache (Name/Type/P/T/oracle) keyed by engine card-row IDs.
- `mlm.py` — Masked-LM pretraining: uint16 .bin token-stream dataset, BERT-style masking, tied LM head, MLMTrainer.
- `model.py` — ModernBERT encoder trunk with card/state gather pools, value head, and inline-blank legal-token scoring.
- `native_assembler.py` — Python wrapper for native Go MageEncodeTokensPacked assembler; manages packed token and inline-blank tensor I/O.
- `native_token_tables.py` — Serialize TokenTables, including inline-blank singletons, into flat buffers and register with Go-side mage lib.
- `policy.py` — Self-contained text-encoder policy facade; renders inline blanks and returns value plus blank logits.
- `recurrent.py` — LSTM history adapter wrapping TextPolicy; carries recurrence through encoder state vector.
- `render.py` — Deterministic GameStateSnapshot → text renderer; produces custom-token-laced strings plus inline-blank anchor metadata.
- `render_plan.py` — Opcode definitions, writer, and Python emitter for render-plan ABI.
- `replay_buffer.py` — Single global concurrent ring for encoded text snapshots, inline blank metadata, episode/policy-version metadata, completed-window claims, and GPU-side PPO return building.
- `replay_triton.py` — Optional CUDA/Triton kernels for packed replay-buffer batch append writes, gather packing, and position rebasing.
- `rollout.py` — End-to-end Python Magic game player using text encoder and RecurrentTextPolicy.
- `token_tables.py` — Closed-vocabulary token-id lookup tables for every assembler emission; source of truth for native side.
- `tokenizer.py` — ModernBERT BPE tokenizer augmented with custom mana/card-ref/status/structural tokens.
- `training.py` — Value pretrain and inline-blank BC loss scaffolding plus TextEncoderTrainer.
- `value_pretrain.py` — Stage-3a value-head pretrain: ValueLabeledBinDataset (paired *.bin/*.json) + ValuePretrainTrainer (encoder + ValueHead, MSE on perspective-signed terminal outcomes).
