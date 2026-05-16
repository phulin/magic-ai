# Decoder packed grammar layout (tight scope)

## Scope

Compute cells at gather time from existing dense masks (sync-free, CPU).
Use cells in the per-choice consumer. Rewrite the **pointer head** of
the decoder training forward to produce per-cell logits instead of
materializing `[B, L, T_enc]`. Keep dense storage and the dense vocab
head (V=26, negligible).

## Why this scope

- `GRAMMAR_VOCAB_SIZE = 26` (decoder.py:33) → vocab side of the
  storage and forward is already tiny. No win from refactoring it.
- `pointer_logits = q @ encoded.T` is `O(B·L·T_enc·D)` and scales with
  encoder length. With `text_max_tokens` in the hundreds-to-thousands,
  this is where the dense decoder forward actually spends time.
- The replay-storage refactor (rip out dense masks, add packed cell
  rings) is high effort for ~zero win at V=26 + bool pointer mask.
  Defer indefinitely.

## End state

- `TextReplayBatch` carries packed-cell tensors (`cell_b`, `cell_t`,
  `cell_is_ptr`, `cell_chosen_local`, `cell_legal_start`,
  `cell_legal_count`, `legal_idx`) populated in `gather` on CPU from
  the dense masks. Sync-free.
- `decoder_score_replay` computes pointer logits per-cell instead of
  dense `[B, L, T_enc]`. Vocab logits stay dense at `[B, L, V=26]`.
- `evaluate_replay_batch_per_choice` consumes cells directly: vocab
  gathers from the dense vocab_log_softmax, pointer reads the
  per-cell pointer log-softmax. No `.nonzero` / `.nonzero_static` /
  cumsum dance left.

## Impl checklist

Phase 2 (cells layout + per-choice consumer) — landed in this commit:

- [x] Define `DecoderCells` dataclass (host + device variants). Two
      arenas internally: vocab cells and pointer cells, separated so
      the consumer never needs `.where`-style splitting.
- [x] Compute cells at gather in `TextReplayBuffer.gather`:
      iterate active cells on CPU, build the packed arrays, attach to
      `TextReplayBatch`. Use the dense `output_pad_mask`,
      `output_is_pointer`, `vocab_mask`, `pointer_mask` already
      gathered. CPU-only, no GPU sync.
- [x] Plumb cells through `TextReplayBatch.to(device)` (move int32
      packed arrays to GPU non-blocking).
- [x] Rewrite `evaluate_replay_batch_per_choice` to consume cells —
      direct gather, no nonzero/cumsum/group_ids construction.
- [x] Parity test: cell builder produces (b, t, choice) tuples that
      match the dense-mask `.nonzero()` enumeration on a fixed seed
      batch; `is_chosen` matches per-cell target lookup; behavior
      log-prob matches per-cell `output_log_prob` lookup; all-zero
      edge case returns empty arenas.
- [x] Run full test suite. `ruff format / check / ty`.

Phase 3 (per-cell pointer head in decoder forward) — TODO:

- [ ] Add `GrammarDecoder.forward_teacher_forced_pointer_cells` that
      runs the transformer body, returns dense vocab_logits + per-cell
      pointer logits. Cells consumed: pointer cell `(b, t)` indices
      and per-cell `legal_idx` (encoder positions).
- [ ] Update `decoder_score_replay` to call the new forward and
      compute `per_step_log_pi` / `per_row_log_pi` / `per_row_entropy`
      from the mixed dense-vocab + per-cell-pointer representation.
      `DecoderReplayScores` gains `p_legal_logits` / `p_legal_log_probs`
      and drops dense `pointer_logits` / `pointer_log_softmax`.
- [ ] Update `evaluate_replay_batch_per_choice` to consume the new
      per-cell pointer scores directly — no `[B, L, T_enc]` indexing.
- [ ] Parity test: `decoder_score_replay` output (per_row_log_pi,
      per_row_entropy, per_step_log_pi) matches the old dense impl on
      a fixed seed batch.
- [ ] Benchmark: end-to-end R-NaD update time pre/post. Expect the
      pointer-head compute drop to dominate (everything else
      unchanged).

## Non-goals

- Don't touch `decoder_sample` (rollout autoregressive sampling stays
  dense — single-step, fast already).
- Don't touch `policy_value_pretrain.py` (off the perf-critical loop).
- Don't change replay storage schema.
