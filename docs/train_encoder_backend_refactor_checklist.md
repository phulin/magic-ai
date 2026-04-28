# Train Encoder Backend Refactor Checklist

Goal: wire the text/render-plan encoder into `scripts/train.py` as an option without forcing the existing slot encoder, rollout buffer, and `PPOPolicy` internals to support two incompatible state representations.

## Target Shape

- [ ] Add `--encoder {slots,text}` to `scripts/train.py`.
- [ ] Keep `slots` as the default path and preserve current behavior.
- [ ] Treat the current `PPOPolicy` + `RolloutBuffer` + native encoder path as the slot backend.
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

- [ ] Extract the decision-group distribution code out of `PPOPolicy` into a backend-neutral helper module.
- [ ] Preserve support for:
  - [ ] priority choices
  - [ ] targets
  - [ ] attackers/blockers multi-decision groups
  - [ ] blocker/pass "none" choice
  - [ ] may Bernoulli head
- [ ] Define a common forward-output shape for replay scoring:
  - [ ] `values`
  - [ ] `option_vectors`
  - [ ] `target_vectors`
  - [ ] `query` or direct option/target logits
  - [ ] `none_logits`
  - [ ] `may_logits`
  - [ ] recurrent hidden vector for auxiliary losses
- [ ] Keep slot behavior byte-for-byte or test-equivalent after extraction.
- [ ] Add tests that compare old slot replay log-probs against the extracted helper.

## Phase 3: Slot Backend Wrapper

- [ ] Add a small `SlotTrainingBackend` wrapper around current setup.
- [ ] Move slot-specific construction out of `main()`:
  - [ ] `GameStateEncoder.from_embedding_json(...)`
  - [ ] `PPOPolicy(...)`
  - [ ] `ShardedNativeBatchEncoder.for_policy(...)`
  - [ ] `NativeTrajectoryBuffer(...)`
- [ ] Keep current rollout collection logic working through the slot backend.
- [ ] Keep checkpoint save/load metadata compatible with existing slot checkpoints.
- [ ] Verify `--encoder slots` produces the same smoke behavior as before.

## Phase 4: Text Replay Buffer

- [ ] Add `TextReplayBuffer` for text-encoded replay rows.
- [ ] Store per step:
  - [ ] `token_ids`
  - [ ] `attention_mask`
  - [ ] `card_ref_positions`
  - [ ] `option_positions`
  - [ ] `option_mask`
  - [ ] `target_positions`
  - [ ] `target_mask`
  - [ ] `seq_lengths`
  - [ ] `trace_kind_id`
  - [ ] decision layout tensors
  - [ ] selected decision columns
  - [ ] `may_selected`
  - [ ] old log-prob
  - [ ] value
  - [ ] perspective player index
  - [ ] recurrent `h_in` / `c_in`
- [ ] Decide padding strategy for replay rows:
  - [ ] fixed `text_max_tokens` buffer for speed, or
  - [ ] compact variable-length storage with minibatch collation.
- [ ] Add replay-row append/release/reset methods matching the slot buffer lifecycle.
- [ ] Add focused buffer tests for append, replay gather, and episode grouping.

## Phase 5: Text Actor-Critic Training Surface

- [ ] Add `TextActorCritic` wrapper around `RecurrentTextPolicy`.
- [ ] Implement live env recurrent state management:
  - [ ] initialize per-env player states
  - [ ] reset states for completed games
  - [ ] gather/scatter state for active env batches
- [ ] Implement `sample_text_batch(...)` using:
  - [ ] native render-plan emission when enabled
  - [ ] Python render-plan emission as fallback
  - [ ] `assemble_batch(...)`
  - [ ] shared decision replay logic
- [ ] Implement `evaluate_replay_batch(...)` for PPO.
- [ ] Implement `evaluate_replay_batch_per_choice(...)` for RNaD.
- [ ] Implement recurrent recompute methods needed by RNaD.
- [ ] Initially disable SPR for text unless/until a text-specific SPR target path is implemented.

## Phase 6: Text Backend Collector

- [ ] Add `TextTrainingBackend`.
- [ ] Load text artifacts:
  - [ ] tokenizer
  - [ ] oracle text
  - [ ] `data/text_encoder_card_tokens.pt`
- [ ] Build in-memory card-token cache when the `.pt` cache is missing.
- [ ] Add CLI args:
  - [ ] `--card-token-cache`
  - [ ] `--text-max-tokens`
  - [ ] `--text-d-model`
  - [ ] `--text-layers`
  - [ ] `--text-heads`
  - [ ] `--text-d-ff`
  - [ ] `--native-render-plan`
  - [ ] `--render-plan-capacity`
- [ ] Reuse existing game/deck scheduling where possible.
- [ ] Collect finished episodes into the trainer's existing `RolloutStep` / `EpisodeBatch` structures.
- [ ] Ensure sampled text replay rows carry `replay_idx` exactly like slot replay rows.

## Phase 7: `train.py` Integration

- [ ] Split current setup into `build_slot_backend(...)`.
- [ ] Add `build_text_backend(...)`.
- [ ] Branch once after argument validation:
  - [ ] `args.encoder == "slots"`
  - [ ] `args.encoder == "text"`
- [ ] Keep the main training loop backend-oriented:
  - [ ] collect rollout
  - [ ] compute returns or RNaD episode batch
  - [ ] run selected trainer update
  - [ ] save checkpoint
  - [ ] log metrics
- [ ] Save checkpoint metadata including:
  - [ ] encoder kind
  - [ ] text config when `encoder=text`
  - [ ] tokenizer path/hash if available
  - [ ] card-token cache hash
- [ ] Reject incompatible checkpoint/CLI combinations clearly.

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
