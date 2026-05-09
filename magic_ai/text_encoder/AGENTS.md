# magic_ai/text_encoder/

Bidirectional text-encoder for game state and actions. Core pipeline: game snapshot → rendered text with inline decision blanks → tokenized batch → ModernBERT trunk → tied inline-blank scorer and value head. Includes fast hot-path assembler with native Go acceleration, replay buffer, training scaffolding, and Python rollout worker.

## Files
- `__init__.py` — Package exports for the tokenizer API.
- `actor_critic.py` — Training-facing actor-critic wrapper bridging policy outputs to action supervision.
- `batch.py` — Tokenize snapshots and collate into padded batches with card-ref positions, inline-blank metadata, and (additive) decision-spec tokens, pointer anchors, and legal-edge bitmaps for the grammar decoder via `collate_with_specs`.
- `card_cache.py` — Pre-tokenized card-body cache (Name/Type/P/T/oracle) keyed by engine card-row IDs.
- `mlm.py` — Masked-LM pretraining: uint16 .bin token-stream dataset, BERT-style masking, tied LM head, MLMTrainer.
- `model.py` — ModernBERT encoder trunk with card/state gather pools, value head, and inline-blank legal-token scoring.
- `native_assembler.py` — Python wrapper for native Go MageEncodeTokensPacked assembler; manages packed token and inline-blank tensor I/O.
- `native_token_tables.py` — Serialize TokenTables, including inline-blank singletons, into flat buffers and register with Go-side mage lib.
- `policy.py` — Self-contained text-encoder policy facade; renders inline blanks and returns value plus blank logits. Optional `use_grammar_decoder=True` wires a `GrammarDecoder` and exposes `forward_decoder_teacher_forced` + `encode_snapshots_with_specs` (additive, leaves the inline-blank path unchanged when the flag is off).
- `policy_value_pretrain.py` — Forge choice-situation dataset and trainer for joint inline-policy/value pretraining from extracted torch or JSONL artifacts.
- `recurrent.py` — LSTM history adapter wrapping TextPolicy; carries recurrence through encoder state vector.
- `render.py` — Deterministic GameStateSnapshot → text renderer; produces custom-token-laced strings plus inline-blank anchor metadata.
- `render_spec.py` — GameStateSnapshot pending → DecisionSpec renderer for the autoregressive grammar decoder (additive, does not touch inline-blank path).
- `replay_buffer.py` — Single global concurrent ring for encoded text snapshots, inline blank metadata, episode/policy-version metadata, completed-window claims, and GPU-side PPO return building.
- `replay_triton.py` — Optional CUDA/Triton kernels for packed replay-buffer batch append writes, gather packing, and position rebasing.
- `rollout.py` — End-to-end Python Magic game player using text encoder and RecurrentTextPolicy.
- `token_tables.py` — Closed-vocabulary token-id lookup tables for every assembler emission; source of truth for native side.
- `tokenizer.py` — ModernBERT BPE tokenizer augmented with custom mana/card-ref/status/structural tokens.
- `training.py` — Value/inline-blank loss helpers plus TextEncoderTrainer.
- `inline_blanks.py` — Stable inline-blank group-kind enums shared by renderer, batches, training losses, and policy sampling.
- `decision_spec.py` — Decoder-pipeline types: `DecisionType`/`AnchorKind` enums, `DecisionSpec` dataclass, and pointer-anchor helpers.
- `grammar.py` — Per-decision-type grammar state machines (`next_mask`, `batch_next_mask`) and the decoder's small `GrammarVocab` enum.
- `decoder.py` — Autoregressive `GrammarDecoder`: causal transformer with cross-attention to encoder context, vocab + pointer heads, KV-cached `step` API, and `combined_sample` helper.
