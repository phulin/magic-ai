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
- `play_text_rollout.py` — Run smoke-test episodes with RecurrentTextPolicy against the mage engine.
- `pretrain_text_encoder.py` — Smoke-test the text encoder training loop with synthetic data (value or distill mode).
- `synth_embedding_eval.py` — Evaluate synthetic card embeddings for invariance and analogies under rules-text encoding.
- `train.py` — Train a PPO self-play policy against the mage-go Python engine.
