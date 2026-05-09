# scripts/

Utility scripts for building data pipelines, evaluating card embeddings, pretraining the text encoder, and training RL policies. Run these to prepare datasets, benchmark components, and execute training/evaluation workflows.

## Files

- `bench_packed_encoder.py` — Benchmark padded vs packed forward+backward performance on varying sequence-length distributions.
- `bench_text_replay_append.py` — Benchmark text replay-buffer append_batch/gather with and without the optional Triton kernels.
- `build_card_embeddings.py` — Download Magic card oracle text from Scryfall and compute embeddings using a transformer model.
- `build_text_card_embeddings.py` — Extract per-card embeddings by pooling card-ref hidden states from the text encoder.
- `build_text_encoder_card_cache.py` — Pre-tokenize and cache all card bodies for efficient rollout assembly.
- `build_text_encoder_vocab.py` — Register custom tokens in ModernBERT tokenizer and save augmented vocab.
- `compare_reports.py` — Display side-by-side comparison of eval_card_embeddings JSON reports with delta highlighting.
- `eval_card_embeddings.py` — Evaluate embeddings against triplet/synonym/cluster labels and compute composite scores.
- `extract_forge_choice_situations.py` — Stream Forge game ZIP logs and write reloadable choice situations as sharded torch snapshots or debug JSONL with token ids, observed choice metadata, and terminal outcome labels.
- `jsonl_games_to_bin.py` — Convert recorded `*.jsonl.gz` game logs into per-game uint16 `.bin` token streams (one rendered snapshot per decision point) for MLM pretraining; pass `--with-value-labels` to also emit a `<gameId>.json` sidecar (winner_id, players, per-span perspective-signed labels) for value-head pretraining.
- `inline_blank_bc_parity.py` — Priority-only inline-blank BC smoke gate on fixed JSONL traces or a synthetic fixture.
- `play_text_rollout.py` — Run smoke-test episodes with RecurrentTextPolicy against the mage engine; can emit priority JSONL traces for the inline-blank BC gate.
- `pretrain_text_encoder.py` — Smoke-test the text encoder value training loop with synthetic data.
- `synth_embedding_eval.py` — Evaluate synthetic card embeddings for invariance and analogies under rules-text encoding.
- `test_forge_target_encoding_smoke.py` — Synthesize one fake decision per pending kind, run them through the V2 decoder-target encoder, and assert grammar legality (no Forge JVM / corpus required).
- `train.py` — Train PPO/R-NaD self-play policies against mage-go, including the native text IMPALA actor/server/ring/learner path and steady-state benchmark logging.
