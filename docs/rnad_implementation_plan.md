# R-NaD Implementation Checklist

Companion to `docs/rnad_design.md`. Work top-to-bottom; each phase ends at a runnable/testable milestone.

## Phase 0 — Scaffolding
- [ ] Create `magic_ai/rnad.py` with empty module docstring and imports.
- [ ] Create `tests/test_rnad.py` skeleton.
- [ ] Add `--trainer {ppo,rnad}` argparse flag in `scripts/train_ppo.py`; default `ppo`; raise `NotImplementedError` for `rnad` path.
- [ ] Add R-NaD CLI flags: `--rnad-eta`, `--rnad-delta-m`, `--rnad-m`, `--rnad-neurd-beta`, `--rnad-neurd-clip`, `--rnad-target-ema`, `--rnad-finetune-eps`, `--rnad-finetune-ndisc`.
- [ ] Add `mkdir -p checkpoints/rnad` logic in the script (first run).
- [ ] `uv run ruff format && uv run ruff check --fix && uv run ty check` clean.

## Phase 1 — Config & primitives
- [ ] Define `RNaDConfig` dataclass with defaults from the design doc.
- [ ] Implement `transform_rewards(rewards, logp_theta, logp_reg_cur, logp_reg_prev, alpha, eta, perspective_idx)`.
- [ ] Implement `threshold_discretize(probs, eps, n_disc)`.
- [ ] Unit tests: symbolic check of `transform_rewards` sign flip; `threshold_discretize` sum-to-1, ε-drop, quantization grid.

## Phase 2 — Two-player v-trace
- [ ] Implement `two_player_vtrace(...)` returning `(v_hat, q_hat_per_head)` via backward recursion over a full episode.
- [ ] Handle own-turn vs opponent-turn branches per §172 vs §174–181.
- [ ] Ensure `q_hat` is produced for all legal actions (per head), not just sampled ones.
- [ ] Unit test: hand-crafted 3-step trajectory matches a Python reference implementation.
- [ ] Unit test: importance-weight clipping at `rho_bar=c_bar=1.0` matches expected values.

## Phase 3 — NeuRD loss + critic loss
- [ ] Implement `neurd_loss(logits, q_hat, legal_mask, beta, clip)` with `Clip(Q, c_neurd)` and the $[-\beta, \beta]$ logit gate.
- [ ] Implement `critic_loss(v_theta, v_hat, perspective_mask)` (L1 over own-turn steps).
- [ ] Unit test: in unclipped regime, NeuRD gradient matches autograd of $-\sum \pi Q$.
- [ ] Unit test: outside $[-\beta, \beta]$, NeuRD gradient is zero.

## Phase 4 — Target network + reg snapshots
- [ ] Add a target-policy copy of `PPOPolicy`; wire Polyak EMA update helper (mirror SPR EMA code in `model.py`).
- [ ] Extend `evaluate_replay_batch` (or add a sibling `evaluate_replay_batch_multi`) to forward the same batch through N frozen policies in one pass and return per-step logits for each.
- [ ] Add `save_reg_snapshot(policy, m)` / `load_reg_snapshot(m)` under `checkpoints/rnad/reg_m{N}.pt`.
- [ ] Smoke test: save → load → forward reproduces logits bit-exactly.

## Phase 5 — `rnad_update` orchestration
- [ ] Implement `rnad_update(online, target, reg_cur, reg_prev, batch, config, step_in_m)`.
  - [ ] Compute α_n from `step_in_m` and `delta_m`.
  - [ ] Forward online + target + both reg policies on the batch.
  - [ ] Call `transform_rewards`, `two_player_vtrace`.
  - [ ] Sum `critic_loss` + per-head `neurd_loss`.
  - [ ] Adam step with `grad_clip`; Polyak update target.
  - [ ] Return a stats dataclass (loss, critic_loss, neurd_loss_per_head, v_hat_mean, grad_norm).
- [ ] Wire into `scripts/train_ppo.py`: when `--trainer rnad`, replace the PPO update call with `rnad_update`.
- [ ] When `--trainer rnad` and `--draw-penalty` unset, force it to 0 and log a warning.

## Phase 6 — Outer loop
- [ ] Add outer loop over `m in range(M)`:
  - [ ] Initialize `reg_cur`, `reg_prev` (both = initial policy snapshot).
  - [ ] Inner loop over `Δ_m` gradient steps, each interleaved with rollouts as in PPO.
  - [ ] After inner loop: `reg_prev ← reg_cur`; `reg_cur ← target`; persist snapshot.
  - [ ] Log outer-iteration marker to wandb.
- [ ] Checkpoint / resume: persist `(m, step_in_m, online, target, reg_cur, reg_prev, optimizer)` to `checkpoints/rnad/state.pt`.
- [ ] Resume test: kill after M=1; restart; confirm seamless continuation.

## Phase 7 — Rollout integration
- [ ] Confirm rollouts use the **target** policy (not online) for action sampling. Add a flag-free code path that selects target when `--trainer rnad`.
- [ ] Confirm opponent-pool sampling is bypassed during R-NaD rollouts (self-play vs current target only).
- [ ] Keep opponent-pool **evaluation** active — periodic TrueSkill eval against PPO snapshots.

## Phase 8 — Toy-game sanity check
- [ ] Add an NFG harness (matching pennies / RPS) in `tests/test_rnad.py` that runs the full R-NaD loop for a small M, Δ_m.
- [ ] Assert final policy is within tolerance of the known Nash equilibrium.
- [ ] Mark as slow test; exclude from default test run.

## Phase 9 — First full-MTG run
- [ ] Launch `--trainer rnad` with defaults (M=20, Δ_m=25k, η=0.2).
- [ ] Monitor: critic loss trend, v_hat distribution, per-head NeuRD loss, fraction of logits inside $[-\beta, \beta]$, TrueSkill vs PPO snapshot pool.
- [ ] Abort criteria: loss diverges; v_hat collapses to ±1; policy entropy collapses to 0.
- [ ] Success: TrueSkill within 1 σ of PPO snapshot after matched wall-clock.

## Phase 10 — Fine-tuning phase
- [ ] After final outer iteration, run one more inner loop with `threshold_discretize` applied inside `sample_native_batch` at rollout time.
- [ ] Compare pre- vs post-finetune TrueSkill.
- [ ] Expose fine-tune phase behind `--rnad-finetune-steps N`.

## Phase 11 — Evaluation tooling
- [ ] Add a CLI: `scripts/eval_rnad.py --snapshot X --baseline Y --games N` that runs head-to-head with threshold/discretize at eval time.
- [ ] Record per-outer-iteration TrueSkill curve for the design-doc appendix.

## Phase 12 — Docs & cleanup
- [ ] Update `README.md` with R-NaD quickstart: `uv run scripts/train_ppo.py --trainer rnad ...`.
- [ ] Update `docs/rnad_design.md` with any deviations discovered during implementation.
- [ ] Final lint pass: `uv run ruff format && uv run ruff check --fix && uv run ty check`.
- [ ] Open PR with: design doc, implementation plan, code, tests, first-run TrueSkill plot.

## Out of scope for this PR
- Joint-action NeuRD enumeration (revisit only if per-head underperforms).
- MTG analogues of Stratego's memory / value-bounds / eagerness heuristics.
- JAX port, pmap parallelism, Sebulba redesign.
- Deployment-phase head (no MTG analogue).
