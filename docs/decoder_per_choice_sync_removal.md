# Decoder per-choice: eliminate CPU↔GPU syncs

## Problem

`LSTMStatefulTextPolicy.evaluate_replay_batch_per_choice` is ~48% of training
wall time per py-spy. The three `.nonzero(as_tuple=False)` calls
(lstm_stateful_text_policy.py:457, 469, 480) each force a sync because the
output shape is data-dependent. Called 4× per R-NaD update (online / target /
reg_cur / reg_prev) ⇒ 12 syncs/update.

## Approach (Phase 1 — this PR)

`torch.nonzero_static(mask, size=N)` is sync-free when `N` is known
host-side. All three masks are derived from data the replay buffer already
holds on CPU (storage device is `cpu`), so we can compute the True-counts on
CPU at gather time and pass them through.

This is the minimum-surgery path: no schema change, no decoder refactor,
no R-NaD API change.

## Why this is correct

- `pad_mask`, `is_pointer_step`, `vocab_mask`, `pointer_mask` are gathered
  from CPU replay storage in `TextReplayBuffer.gather` (replay_buffer.py:1695+).
- Their bool sums fully determine the output sizes of the three nonzeros:
  - `n_active_steps = pad_mask.sum()`
  - `n_vocab_cells = ((pad_mask & ~is_pointer_step).unsqueeze(-1) & vocab_mask).sum()`
  - `n_ptr_cells   = ((pad_mask &  is_pointer_step).unsqueeze(-1) & pointer_mask).sum()`
- Computing these on CPU bool tensors before `.to(device)` is a vectorized
  C sum — no GPU involvement.
- `torch.nonzero_static(mask, size=N)` with `N` equal to the true count
  returns the exact same tensor as `mask.nonzero(as_tuple=False)`, with no
  sync (verified against `torch.__version__ == 2.11.0+cu128`).

## Impl checklist

- [x] Add `n_active_steps`, `n_vocab_cells`, `n_ptr_cells` int fields to
      `TextReplayBatch`. Computed on CPU inside `TextReplayBuffer.gather`
      before the `.to(device)` move.
- [x] Plumb the three counts from `_score_replay_rows` back to
      `evaluate_replay_batch_per_choice` (`batch` is already in the extras
      tuple, fields readable as `batch.n_*`; no return-tuple change needed).
- [x] Replace `group_active.nonzero(as_tuple=False)` with
      `torch.nonzero_static(group_active, size=n_active_steps)`.
- [x] Replace `vocab_cell_active.nonzero(as_tuple=False)` with
      `torch.nonzero_static(vocab_cell_active, size=n_vocab_cells)`.
- [x] Replace `ptr_cell_active.nonzero(as_tuple=False)` with
      `torch.nonzero_static(ptr_cell_active, size=n_ptr_cells)`.
- [x] Drop `step_for_decision_group = g_b` indirection — `group_idx` =
      `torch.cat([v_b, p_b])` directly (the cell's b-coord == the b-coord
      of its group). Saves one `[]`-index op per call.
- [x] Compute counts via the lighter formulation: sum the choice axis
      first (`vocab_mask.sum(dim=-1)` → `[B, L]`) then mask-and-sum, so
      the `[B, L, V]` / `[B, L, T_enc]` AND tensor never materializes.
- [x] Unit test: parity between the old `.nonzero` impl and the new
      `nonzero_static` impl on a fixed-seed batch (3 cases:
      random-mask count parity, `nonzero_static`-vs-`nonzero` index parity,
      all-zero edge case). See
      `tests/test_per_choice_sync_free.py`.
- [x] Run `tests/test_rnad*` and `tests/test_text_rollout.py` to confirm
      no regressions. (63 passed across replay_buffer + text_rollout +
      rnad + decoder_inference + decoder_loss; no behavior change.)
- [x] `uv run ruff format`, `uv run ruff check --fix`, `uv run ty check`.

## Followups (not this PR)

- **Phase 2** — Stop storing `vocab_mask: [capacity, L, V]` and
  `pointer_mask: [capacity, L, T_enc]` dense in replay. Replace with a
  packed cells ring (per-row `(t, is_ptr, chosen_local_idx,
  chosen_log_prob)` + per-cell ring of `legal_choice_idx`). Saves replay
  memory ~1000× on these fields and eliminates the per-gather mask
  reconstruction. Touches replay schema; gather changes; per-choice
  consumer becomes a straight gather.
- **Phase 3** — End-to-end packed grammar layout: decoder forward emits
  per-legal-cell logits instead of dense `[B,L,V]` / `[B,L,T_enc]`. Drops
  decoder compute from `O(B·L·V)` to `O(Σ n_legal)`. The compiled-decoder
  hot path (commit `82971d9`) is where most of the care goes.
