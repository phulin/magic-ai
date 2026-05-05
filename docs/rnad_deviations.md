# R-NaD deviations for Magic: The Gathering

The DeepNash R-NaD paper (arXiv:2206.15378) targets Stratego, a single-decision
imperfect-information game with a single-categorical action per step and a
full-information observation graph for an actor sampling from a 768-TPU pool.
This codebase adapts R-NaD to a much smaller-budget Magic: The Gathering
trainer with several deliberate deviations. This document records them so that
future debugging does not treat intentional approximations as accidental bugs.

The numbered "issues" referenced here are the items from the original
implementation review.

## 1. Per-actor LSTM cache is runtime state, not model state

`PPOPolicy.live_lstm_h` / `live_lstm_c` are env-indexed sampling caches owned
by whichever module is currently driving rollouts. They are NOT trainable model
state and are NOT averaged or copied across policies:

- `polyak_update_` skips any buffer whose name starts with `live_lstm_h`,
  `live_lstm_c`, or `rollout_buffer.` (see `_ACTOR_RUNTIME_BUFFER_PREFIXES`).
- `_clone_policy_sharing_buffer` resets the clone's `live_lstm_*` to empty
  buffers (the rollout buffer is intentionally shared so target/reg can
  evaluate replay rows without duplicating gigabytes of trajectory tensors).
- Recurrent state for replay evaluation is recomputed per policy from h=c=0
  (issue 2 below).

## 2. Per-policy recurrent-state recomputation in R-NaD updates

The rollout buffer stores the LSTM input hidden state at each step under the
*behavior* policy. Evaluating target or reg from those stored states would
silently mix policies. Each policy now re-runs the LSTM scan from zero through
the replayed episode under its own parameters via
`PPOPolicy.recompute_lstm_states_for_episode` and passes the resulting per-
step `(h_in, c_in)` to `evaluate_replay_batch_per_choice` as
`lstm_state_override`.

The scan does features-batched + LSTM-cell-sequential (T iterations of a
single-step LSTM cell), so the cost is one batched embedding pass plus T
small recurrent-cell ops per policy per episode.

## 3. Production NeuRD is always full per-action

There is no sampled-action NeuRD path in the trainer. `rnad_trajectory_loss`
uses `neurd_loss_per_choice` over all legal per-choice logits with the paper's
β-magnitude logit gate (paper §188-189), and a true two-action Bernoulli
NeuRD on the may head (issue 5 below).

## 4. Factored MTG actions: autoregressive decomposition

The paper assumes a single-categorical action per step. Magic steps are
factored into one or more decision groups (each its own conditional softmax)
plus an optional ``may`` Bernoulli. We treat each decision group as its own
paper-faithful step for NeuRD purposes:

- Per-group sampled correction with per-group `1/mu_k`, not joint `1/∏_k mu_k`.
- Per-group β-gated logit update.
- Per-group `-eta · log(pi/pi_reg)(a)` regularization on every legal choice.

The trajectory-level pieces (v-trace, reward transform, critic) stay at step
granularity since the reward is delivered at the joint step, not per group.

**Resolved behavior-policy storage.** New replay rows store the rollout-time
sampled per-group log-prob alongside each decision group. R-NaD's per-action Q
correction now uses that stored value as `mu_k`, so Polyak lag between online
and target no longer biases the per-group sampled correction. The joint
v-trace IS ratio still uses the stored joint `logp_mu` from rollout time.

`ReplayPerChoice` requires this per-group behavior field. Direct test or
utility callers must provide it explicitly; there is no target-policy
substitution fallback.

## 5. Bernoulli may head: true two-action NeuRD

The may head outputs a single logit. We treat it as a 2-action softmax with
logits `(l, 0)` so the full NeuRD gradient covers BOTH branches:

```
d(loss)/dl = -[(1 - p) · Clip(Q_accept, c) - p · Clip(Q_decline, c)]
              · 1[|l_post| <= beta]
```

The trainer constructs `Q_accept` and `Q_decline` separately, each carrying
its own `-eta · log(pi/pi_reg)` regularization term. The previous 1-logit
form regularized only the sampled branch.

## 6. 1/mu clip default is loose

`q_corr_rho_bar = 100` (default) bounds the per-group sampled-correction
magnitude without aggressively biasing the estimator. Under the per-group
decomposition (issue 4), `1/mu_k` is bounded by one decision group's sampled
behavior probability, not by the multiplicative `1/∏_k mu_k` of the joint
formulation.

`run_rnad_update` logs `q_clip_fraction` (the fraction of per-action Q values
clamped by `[-neurd_clip, +neurd_clip]` in the per-choice NeuRD loss). A high
clip fraction is the signal that the per-action Q estimator has drifted; a
clip frequency above a few percent should prompt a closer look before raising
the bound further.

## 7. Loss normalization is by total step / action count

`rnad_trajectory_loss` returns sum-form losses paired with counts:

- `cl_sum` / `cl_count` over own-turn step count.
- `pl_sum` / `pl_count` over total active per-choice actions across all
  decision groups + may branches.

`run_rnad_update` aggregates across the rollout-batch's episodes and divides
by the aggregate counts. This is the paper-faithful 1/t_effective weighting
and avoids over-weighting short games or players with fewer own-turn
decisions.

## 8. Other deliberate deviations carried over from earlier passes

- **The opponent pool is not used for R-NaD self-play.** Opponents are the
  current target. Snapshots still land in the pool for TrueSkill evaluation
  against PPO baselines.
- **Stratego-specific heuristics absent**: e.g. piece-revealing belief tracker,
  perfect-information critic warmup. None of these are MTG-portable.
- **Hyperparameters scaled ~100x down** from the paper to fit a single-GPU
  rollout-batch cadence (paper: 768 TPU nodes, 7.21M steps;
  `RNaDConfig.delta_m=1_000`, `num_outer_iterations=50`). Increase
  proportionally with rollout-batch size.
- **Draw handling**: `winner_idx = -1` skips the terminal reward (zero-sum
  rewards remain zero everywhere) — a permissible interpretation but it
  loses the "pressure to break draws" signal that an explicit small negative
  draw reward would inject. Configurable in future work.
