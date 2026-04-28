# Train Encoder Backend Refactor Checklist

Goal: wire the text/render-plan encoder into `scripts/train.py` as an option without forcing the existing slot encoder, rollout buffer, and `PPOPolicy` internals to support two incompatible state representations.

## Target Shape

- [x] Add `--encoder {slots,text}` to `scripts/train.py`.
- [x] Keep `slots` as the default path and preserve current behavior.
- [x] Treat the current `PPOPolicy` + `RolloutBuffer` + native encoder path as the slot backend.
- [ ] Add a separate text backend built from `RecurrentTextPolicy`, `CardTokenCache`, render-plan assembly, and a text replay buffer.
- [ ] Refactor PPO/RNaD trainer code to depend on a replay-policy interface instead of concrete `PPOPolicy`.

## Phase 1: Name The Existing Boundaries

- [x] Document that `PPOPolicy` is currently a slot-encoder actor-critic stack, not a generic PPO policy.
  - `magic_ai/training_interfaces.py` now names trainer-facing interfaces separately from the concrete slot model.
- [x] Identify all trainer-facing methods used by `ppo_update`:
  - [x] `parameters()`
  - [x] `evaluate_replay_batch(...)`
  - [x] `compute_spr_loss(...)` when SPR is enabled
  - [x] `update_spr_target()` when SPR is enabled
  - [x] `spr_enabled`
- [x] Identify all trainer-facing methods used by `run_rnad_update`:
  - [x] `evaluate_replay_batch_per_choice(...)`
  - [x] `recompute_lstm_outputs_for_episodes(...)`
  - [x] `recompute_lstm_states_for_episodes(...)`
  - [x] access to replay-buffer decision metadata used for counting active steps
  - [x] clone/target/reg policy construction requirements
- [x] Add lightweight protocol types for the PPO and RNaD policy surfaces.
  - Added `PPOReplayPolicy`, `RNaDReplayPolicy`, and `RNaDTrainablePolicy`.
- [x] Change type hints in `magic_ai/ppo.py` from concrete `PPOPolicy` to the PPO protocol.
- [x] Change type hints in `magic_ai/rnad_trainer.py` from concrete `PPOPolicy` to the RNaD protocol where practical.
  - `run_rnad_update` now accepts `RNaDTrainablePolicy`; clone/build state remains slot-specific because it deep-copies `PPOPolicy` and shares `RolloutBuffer`.

## Phase 2: Extract Shared Decision Replay Logic

- [x] Extract the decision-group distribution code out of `PPOPolicy` into a backend-neutral helper module.
  - Added `magic_ai/replay_decisions.py`; `PPOPolicy` now delegates decision logits, flat distribution, and validation helpers to it.
- [ ] Preserve support for:
  - [x] priority choices
  - [x] targets
  - [x] attackers/blockers multi-decision groups
  - [x] blocker/pass "none" choice
  - [ ] may Bernoulli head
    - May remains in `PPOPolicy.evaluate_replay_batch*`; only decision-group scoring was extracted in this pass.
- [ ] Define a common forward-output shape for replay scoring:
  - [ ] `values`
  - [ ] `option_vectors`
  - [ ] `target_vectors`
  - [ ] `query` or direct option/target logits
  - [ ] `none_logits`
  - [ ] `may_logits`
  - [ ] recurrent hidden vector for auxiliary losses
- [x] Keep slot behavior byte-for-byte or test-equivalent after extraction.
- [x] Add tests that compare old slot replay log-probs against the extracted helper.
  - Added `tests/test_replay_decisions.py` for manual logits/entropy expectations and the `PPOPolicy` adapter path.

## Phase 3: Slot Backend Wrapper

- [x] Add a small `SlotTrainingBackend` wrapper around current setup.
- [x] Move slot-specific construction out of `main()`:
  - [x] `GameStateEncoder.from_embedding_json(...)`
  - [x] `PPOPolicy(...)`
  - [x] `ShardedNativeBatchEncoder.for_policy(...)`
  - [x] `NativeTrajectoryBuffer(...)`
- [x] Keep current rollout collection logic working through the slot backend.
  - `main()` now builds a `SlotTrainingBackend` and passes its existing policy, native encoder, and staging buffer into the unchanged loop.
- [x] Keep checkpoint save/load metadata compatible with existing slot checkpoints.
  - Checkpoints now save `metadata.encoder`; legacy checkpoints without it are treated as `slots`.
- [ ] Verify `--encoder slots` produces the same smoke behavior as before.

## Phase 4: Text Replay Buffer

- [x] Add `TextReplayBuffer` for text-encoded replay rows.
- [ ] Store per step:
  - [x] `token_ids`
  - [x] `attention_mask`
  - [x] `card_ref_positions`
  - [x] `option_positions`
  - [x] `option_mask`
  - [x] `target_positions`
  - [x] `target_mask`
  - [x] `seq_lengths`
  - [x] `trace_kind_id`
  - [x] decision layout tensors
  - [x] selected decision columns
  - [x] `may_selected`
  - [x] old log-prob
  - [x] value
  - [x] perspective player index
  - [x] recurrent `h_in` / `c_in`
- [x] Decide padding strategy for replay rows:
  - [x] fixed `text_max_tokens` buffer for speed, or
  - [ ] compact variable-length storage with minibatch collation.
- [x] Add replay-row append/release/reset methods matching the slot buffer lifecycle.
- [x] Add focused buffer tests for append, replay gather, and episode grouping.
  - Added `magic_ai/text_encoder/replay_buffer.py` and `tests/test_text_replay_buffer.py`.

## Phase 5: Text Actor-Critic Training Surface

- [x] Add `TextActorCritic` wrapper around `RecurrentTextPolicy`.
- [ ] Implement live env recurrent state management:
  - [x] initialize per-env player states
  - [x] reset states for completed games
  - [x] gather/scatter state for active env batches
    - Added `magic_ai/text_encoder/actor_critic.py`; sampling/replay evaluation are still pending.
- [x] Implement `sample_text_batch(...)` using:
  - [ ] native render-plan emission when enabled
  - [ ] Python render-plan emission as fallback
  - [ ] `assemble_batch(...)`
  - [x] shared decision replay logic
  - `TextActorCritic.sample_text_batch(...)` now consumes assembled text batches, samples decision groups/may decisions, updates live LSTM state, appends `TextReplayBuffer` rows, and returns `PolicyStep` entries with `replay_idx`. Render-plan emission and assembly are still owned by the pending collector.
- [x] Implement `evaluate_replay_batch(...)` for PPO.
  - `TextActorCritic.evaluate_replay_batch(...)` now replays `TextReplayBuffer` rows through `RecurrentTextPolicy`, scores direct option/target logits plus none/may heads, and supports PPO loss/backward.
- [x] Implement `evaluate_replay_batch_per_choice(...)` for RNaD.
  - Text replay now returns the same `ReplayPerChoice` shape used by R-NaD's slot path; recurrent recompute is still pending.
- [x] Implement recurrent recompute methods needed by RNaD.
  - Added text LSTM state/output recompute methods for episode replay overrides.
- [x] Initially disable SPR for text unless/until a text-specific SPR target path is implemented.
  - `TextActorCritic.spr_enabled` is fixed false and rejects SPR replay extras.

## Phase 6: Text Backend Collector

- [x] Add `TextTrainingBackend`.
- [ ] Load text artifacts:
  - [x] tokenizer
  - [x] oracle text
  - [x] `data/text_encoder_card_tokens.pt`
- [x] Build in-memory card-token cache when the `.pt` cache is missing.
- [ ] Add CLI args:
  - [x] `--card-token-cache`
  - [x] `--text-max-tokens`
  - [x] `--text-d-model`
  - [x] `--text-layers`
  - [x] `--text-heads`
  - [x] `--text-d-ff`
  - [x] `--native-render-plan`
  - [x] `--render-plan-capacity`
- [ ] Reuse existing game/deck scheduling where possible.
- [ ] Collect finished episodes into the trainer's existing `RolloutStep` / `EpisodeBatch` structures.
- [x] Ensure sampled text replay rows carry `replay_idx` exactly like slot replay rows.

## Phase 7: `train.py` Integration

- [x] Split current setup into `build_slot_backend(...)`.
- [x] Add `build_text_backend(...)`.
- [ ] Branch once after argument validation:
  - [x] `args.encoder == "slots"`
  - [ ] `args.encoder == "text"`
- [ ] Keep the main training loop backend-oriented:
  - [ ] collect rollout
  - [ ] compute returns or RNaD episode batch
  - [ ] run selected trainer update
  - [ ] save checkpoint
  - [ ] log metrics
- [ ] Save checkpoint metadata including:
  - [x] encoder kind
  - [ ] text config when `encoder=text`
  - [ ] tokenizer path/hash if available
  - [ ] card-token cache hash
- [x] Reject incompatible checkpoint/CLI combinations clearly.

## Phase 8: Validation

- [ ] Verify existing slot tests pass unchanged.
- [ ] Add `--encoder slots` train smoke test.
- [ ] Add `--encoder text` single-game rollout smoke test.
- [ ] Add text PPO replay test:
  - [ ] sample action
  - [ ] store replay row
  - [ ] reevaluate log-prob from replay
  - [ ] confirm finite loss/backward
- [ ] Add text RNaD replay test after per-choice replay is implemented.
- [ ] Run:
  - [ ] `uv run ruff format`
  - [ ] `uv run ruff check --fix`
  - [ ] `uv run ty check`
  - [ ] focused pytest for slot replay
  - [ ] focused pytest for text replay

## Non-Goals For The First Pass

- [ ] Do not make one universal rollout buffer that stores both slot and text tensors.
- [ ] Do not force `PPOPolicy` to internally switch between slot and text encoders.
- [ ] Do not enable text SPR until the text replay surface is stable.
- [ ] Do not delete the current slot path.
- [ ] Do not require native render-plan emission for the first text training smoke; Python render-plan emission is acceptable for correctness.
