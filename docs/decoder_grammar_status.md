# Decoder grammar migration — status

Tracking implementation of `docs/decoder_grammar_plan.md`.

## Cutover status (2026-05-09)

Phases 1–7 of the inline-blank → grammar-decoder cutover are landed on
`decoder-cutover`. The Forge BC pretraining path, the Python rollout
(`magic_ai/text_encoder/rollout.py`), the in-process train.py text-encoder
path (`sample_text_policy_batch`), the actor-critic sampler/scorer, and the
replay buffer all run on the decoder pipeline. Verification:

- `pytest -x -q` → 313 passed, 1 skipped, 2 xfailed
- `scripts/smoke_decoder_train.py` → smoke OK (sample + teacher-forced loss + grad)
- `scripts/decoder_bc_parity.py` (V2 corpus, 30 steps) → priority accuracy 1.000
- `mage-go go test ./cmd/pylib/...` → ok

### What still uses inline blanks

The native batched IMPALA actor pipeline
(`train_text_native_batched_envs` → `run_text_rollouts_actor_loop`,
`magic_ai/native/inference_server.py`, `magic_ai/native/rollout_actor.py`,
the Go-side selected_choice_cols protocol, `NativeTextTrajectoryBuffer`)
still expects inline-blank shapes. The four legacy stubs in
`scripts/train.py` (`build_text_decision_layout`, `infer_text_trace_kind`,
`_decode_text_action`, `NativeTextTrajectoryBuffer.__init__`) raise loudly
on construction with a pointer to the decoder-shaped replacements.

To re-enable that high-throughput path the following needs an end-to-end
protocol redesign — none of the work is done:

- `TextInferenceServer` request/response contract: replace
  `selected_choice_cols`/`may_selected` with per-row decoder layouts
  (output_token_ids, output_pointer_pos, output_is_pointer, output_pad_mask,
  decision_type, pointer_anchor_handles).
- `TextRolloutActor`: receive the decoder layout, call
  `decode_decoder_action(pending, layout)`, drive the engine via
  `step_by_choice` accordingly (Go-side `step_by_choice` may need a
  per-decision-type dispatch since it currently expects flat
  selected_choice_cols).
- `NativeTextTrajectoryBuffer`: ring-buffer of `DecoderDecisionPayload`
  records (one per env-step) instead of the inline-blank rectangles;
  `commit_decoder_decision` already exists on the replay buffer.

The single-policy in-process loop (`train_text_envs`) covers all current
tested paths and is wired correctly; the IMPALA path is parked.

### Forge BC corpus V2

`scripts/extract_forge_choice_situations.py` previously wrote V1 records
without `pending` state; the V2 decoder-target translator therefore had
nothing to translate. The extractor now:

- Accepts a `--zip` argument that may be either a zipfile or a directory
  of `*.jsonl.gz` shards (Forge writes loose dirs in newer batches).
- Builds a `pending` dict from `snapshot.playableActions` for priority
  candidates, mapping Forge `kind` (`PLAY_LAND`/`CAST_SPELL`/`ACTIVATE`/
  `ACTIVATE_MANA`) to the engine's `play`/`cast`/`activate` option kinds.
- Recognizes the new Forge log description formats
  (`STACK_PUSH <name> by <player>`, `<player> puts <name> [id] from hand
  onto the Battlefield`) in addition to the legacy
  `played`/`cast`/`activated` strings.

`forge_target_encoding.translate_priority` now matches by structured
`(card_name, is_land_play)` from the observed-event dict when present.

The V2 corpus extracted from `data/games/` (1000 games) yields 4000 records
with ~83% priority translation rate. Attack and block extraction is not yet
ported to the new Forge log format — those translators expect
`PlayerA assigned X to attack PlayerB` / `PlayerA blocked X with Y` log
lines, but the current logs use `PlayerA attacks PlayerB with N creatures`
followed by per-creature `Attacker:` / `Blocker:` lines. Re-porting
`_attack_label` / `_block_label` is the next bounded follow-up.

## Done

| Step | Description | Where |
|---|---|---|
| 1 | `DecisionType`, `AnchorKind`, `PointerAnchor`, `DecisionSpec` types | `magic_ai/text_encoder/decision_spec.py` |
| 1 | Tokenizer additions (spec-open/close, decision-type-name, stack-ref:k); persisted tokenizer rebuilt | `magic_ai/text_encoder/tokenizer.py` |
| 1 | `render_decision_spec` for PRIORITY, DECLARE_ATTACKERS, DECLARE_BLOCKERS, CHOOSE_TARGETS, MAY, CHOOSE_MODE, CHOOSE_X | `magic_ai/text_encoder/render_spec.py` |
| 2 | Per-decision-type grammar state machines; `GrammarVocab` (26 ids); `next_mask`, `batch_next_mask` | `magic_ai/text_encoder/grammar.py` |
| 3 | `GrammarDecoder` causal transformer (RoPE self-attn + cross-attn + GeGLU FFN); `DecoderState` KV cache; `combined_sample` | `magic_ai/text_encoder/decoder.py` |
| 3 | `TextPolicy.use_grammar_decoder` flag + `forward_decoder_teacher_forced`; `RecurrentTextPolicyConfig.use_grammar_decoder` | `magic_ai/text_encoder/policy.py`, `recurrent.py` |
| 3 | `TextEncodedBatch` spec/anchor/legal-edge fields; `collate_with_specs` | `magic_ai/text_encoder/batch.py` |
| 8 | Native `decisionSpec_emitter.go` (mirrors `render_spec.py`); `MagePackedSpecOutputs` ABI struct | `mage-go/cmd/pylib/decision_spec_emitter.go` |
| 9 | Native `decision_mask.go` (mirrors `grammar.next_mask`); `MageDecisionMaskNext` cgo export with batched goroutine fan-out; `BatchHandle` lifecycle | `mage-go/cmd/pylib/decision_mask.go`, `decision_spec_ffi.go` |
| 8/9 | Magic-ai-side spec-tag id registration (tokenizer → `mage.register_decision_spec_tokens` + digit lookup) | `magic_ai/text_encoder/native_decision_spec.py` |
| 10 | Forge target translators (per pending kind) → grammar token sequences | `magic_ai/text_encoder/forge_target_encoding.py` |
| 10 | `ForgeDecoderBatch`, `_decoder_step` in trainer using `decoder_cross_entropy_loss` + combat exact-match | `magic_ai/text_encoder/policy_value_pretrain.py` |
| 10 | `--pretrain-mlm-decoder` flag in `scripts/train.py` | `scripts/train.py` |
| - | `decoder_cross_entropy_loss` + `decoder_per_step_accuracy` | `magic_ai/text_encoder/training.py` |
| - | BC parity harness (`scripts/decoder_bc_parity.py`) — first GPU run confirmed both pipelines train end-to-end on real Forge data, no NaN, smooth loss decrease | `scripts/decoder_bc_parity.py` |

**Tests:** 35 new Python + 16 new Go, all passing. Lint + ty clean across all touched files.

## Empirically validated

The decoder pipeline trains end-to-end on the Forge corpus on an A100 with no
NaN, smooth loss decrease, and reaches ~0.93 per-step accuracy after 1500
steps (d_model=192, 3 layers, bs=32). Wall-time is ~1.7× inline (autoregressive
teacher-forced cross-attention overhead, expected and manageable). The
inline pipeline has a pre-existing intermittent NaN at ~1 step in 1500 on
real Forge data — independent of this migration.

## Not yet done — production live-path migration

The Forge BC pretraining path is fully migrated. The **live R-NaD/PPO
sampling and replay path** is **not** — it still uses inline blanks
exclusively. Bringing the decoder pipeline online for actual RL training
requires the following work, none of which has been done:

### `magic_ai/text_encoder/actor_critic.py` (~3.5 kLoC, 112 inline-blank refs)

The `TextActorCritic` wraps the recurrent policy and exposes the live-step
APIs (`act`, the various score functions used by R-NaD, etc.). Today this
samples per-blank from `blank_logits`; the decoder migration requires:

- Replace per-blank sampling with an **autoregressive loop** that, at each
  decoder step, calls `GrammarDecoder.step` (with KV cache), then
  `mage.decision_mask_next` to get the legality mask, then `combined_sample`,
  then advances by one token.
- Translate the decoder's pointer picks back to engine option indices via
  the `pointer_anchor_handles` recorded by the renderer.
- Honor the LSTM history adapter — the encoder runs once per env-step;
  the decoder loop runs many times within that one encoder forward.
- Re-implement R-NaD/PPO log-prob and entropy computations: they currently
  sum over per-blank softmaxes; the decoder analog is a sum of per-step
  log-softmaxes over the masked vocab+pointer combined distribution.

This is the single largest remaining piece. Risk: breaking the live R-NaD
training that's running on `main`. Recommended: build it on a worktree under
the `use_grammar_decoder` flag, run an apples-to-apples R-NaD comparison vs
the inline path on a small experiment, then switch.

### `magic_ai/text_encoder/replay_buffer.py` (~1.85 kLoC)

The replay ring stores per-blank fields (`positions`, `kind`, `group`,
`group_kind`, `option_index`, `legal_ids`, `legal_mask`). The decoder
migration requires:

- New `decoder` AggregateTensor: `output_token_ids [capacity, L]`,
  `output_pointer_pos [capacity, L]`, `output_is_pointer [capacity, L]`,
  `output_pad_mask [capacity, L]`, `decision_type [capacity]`,
  `pointer_anchor_*` (positions, kinds, subjects, handles), `legal_edge_bitmap`
  (when block decisions present).
- Bump on-disk format version; add the V1→V2 migration path or
  document that V1 replays must be discarded.
- `replay_triton.py` may need new gather kernels for the new packed shapes.

Either keep the per-blank fields alongside the new decoder fields (gated by
flag) for the migration period, or do a hard cutover with version bump.

### `magic_ai/text_encoder/native_assembler.py` (~630 LoC)

Currently calls `MageEncodeTokensPacked` to produce token + per-blank
outputs. The decoder migration requires *additionally* calling
`mage.encode_decision_spec` per env, gathering the resulting per-row
spec tensors (`spec_tokens`, `pointer_anchor_*`, `legal_edge_bitmap`,
`batch_handle`), and surfacing them through the assembler's outputs to the
Python side. The `BatchHandle` ids must flow through to the actor_critic
so subsequent `decision_mask_next` calls can find the per-env spec state.

Add `register_decision_spec_token_table(tokenizer)` to the assembler's
init path so the native side has the spec-tag ids.

### `magic_ai/text_encoder/rollout.py` (~455 LoC)

The Python-only rollout worker (used as a fallback / for tests) needs
the same autoregressive-sampling loop changes as `actor_critic.py`.

### Tests / experiment

- An R-NaD smoke run with `use_grammar_decoder=True` against a fixed deck
  pool, comparing terminal win-rate trajectory to the inline baseline.
- An apples-to-apples per-decision-type accuracy report from
  `decoder_bc_parity.py` on a longer training run (3-10k steps), per the
  step 4-7 plan gates.

## Steps 11, 12

- **Step 11** (decoder pretraining on R-NaD trajectories) — optional
  follow-up. Requires step 10 + replay_buffer migration first.
- **Step 12** (delete inline-blank surfaces) — gated on the live-path
  migration above being validated and switched to default.

## Files touched in this migration

```
magic_ai/text_encoder/decision_spec.py        (new)
magic_ai/text_encoder/render_spec.py          (new)
magic_ai/text_encoder/grammar.py              (new)
magic_ai/text_encoder/decoder.py              (new)
magic_ai/text_encoder/forge_target_encoding.py (new)
magic_ai/text_encoder/native_decision_spec.py (new)
magic_ai/text_encoder/tokenizer.py            (DECISION_SPEC_TOKENS)
magic_ai/text_encoder/policy.py               (use_grammar_decoder flag)
magic_ai/text_encoder/recurrent.py            (config field)
magic_ai/text_encoder/batch.py                (spec / anchor / legal-edge fields)
magic_ai/text_encoder/policy_value_pretrain.py (ForgeDecoderBatch + decoder step)
magic_ai/text_encoder/training.py             (decoder_cross_entropy_loss)
scripts/train.py                              (--pretrain-mlm-decoder flag)
scripts/decoder_bc_parity.py                  (new harness)
data/text_encoder_tokenizer/*                 (rebuilt with new tokens)

mage-go/cmd/pylib/decision_spec_emitter.go    (new — emit spec stream)
mage-go/cmd/pylib/decision_mask.go            (new — grammar state machine)
mage-go/cmd/pylib/decision_spec_ffi.go        (new — cgo exports)
mage-go/cmd/pylib/abi.h                       (MagePackedSpecOutputs, MageDecisionSpecTokens)
mage-go/cmd/pylib/mage/__init__.py            (Python wrapper)
```

Plus 35 Python + 16 Go test files.

## Phase 7 (final cutover)

Branch `decoder-cutover` at HEAD now passes:

- `cd /home/user/mage-go && go test ./cmd/pylib/...` → ok (cached).
- `uv run pytest -x -q` → **313 passed, 1 skipped, 2 xfailed, 20 subtests passed**.
- `uv run scripts/smoke_decoder_train.py` → exits 0, emits `smoke OK`
  after sampling a legal action and confirming gradients flow through
  the grammar decoder on CUDA (`teacher-forced loss=8.1436`).
- `uv run scripts/decoder_bc_parity.py …` → does **not** run on the
  available `data/forge_choice_situations` corpus: the on-disk records
  are V1 (`format_version=1`), but Phase 6 dropped the V1 synthesizer
  in `policy_value_pretrain.py` and now requires V2 (PendingState +
  DecoderTarget persisted). The dataset must be re-extracted with
  `scripts/extract_forge_choice_situations.py` before this verification
  command will produce numbers. Pre-existing gap from Phase 6, not
  introduced by Phase 7.

### Phase 7 deltas

- `magic_ai/text_encoder/actor_critic.py::TextActorCritic` gained the
  polymorphic R-NaD / PPO surface so the trainers can drive the text
  pipeline without branching on backend kind:
  `precompute_replay_forward`, `count_active_replay_steps`,
  `evaluate_replay_batch`, `evaluate_replay_batch_per_choice`,
  `write_ppo_targets`, `gather_ppo_targets`,
  `gather_replay_old_log_prob_value`,
  `recompute_lstm_states_for_episode`,
  `recompute_lstm_outputs_for_episodes`. The decoder collapses each
  step to a single decision row, so the slot's per-decision-group
  per-choice axis collapses too: every replay row contributes one
  log-π / one entropy / one value, packed into the existing
  `ReplayPerChoice` container so R-NaD's NeuRD assembly runs unchanged.
- `magic_ai/opponent_pool.py::_disable_text_replay_capture` now
  actually swaps the policy's `rollout_buffer` for the duration of the
  context (was a no-op stub).
- `tests/test_train.py` — five inline-blank-shaped skipped tests
  deleted (each constructed obsolete blank-group / decision-group /
  payload tensors that no longer exist on the buffer); the rest of the
  file (47 tests, including `test_train_text_envs_single_game_rollout_smoke`)
  passes.
- `tests/test_inference_server.py` — `InferenceWorkRingTest` and
  `InferenceServerBatchingTest` (and their `_FakePolicy` helper)
  deleted; both classes posted `NativeTextSampleBatch(decision_counts=,
  selected_choice_cols=, may_selected=, …)` against the now-removed
  inline-blank shape. `ConcatPackedBatchTest` (which is shape-neutral)
  remains and passes.
- `scripts/smoke_decoder_train.py` (new) — end-to-end CUDA smoke
  exercising sampling → action decoding → teacher-forced replay →
  `loss.backward()` → optimizer step.

### Known follow-ups (not in scope of this branch)

- `scripts/train.py` text-encoder rollout (`NativeTextTrajectoryBuffer`,
  `_decode_text_action`, `infer_text_trace_kind`,
  `build_text_decision_layout`) still has Phase 6 `NotImplementedError`
  placeholders — they fail loudly only when the live text-encoder
  rollout is invoked (every code path through them is currently
  mocked out by the test suite or driven by the slot encoder). The
  decoder-shaped reimplementation will delete `MAX_NUM`, the
  staging buffer's blank-* tensors, and the legacy
  `_decode_text_action` helper, and replace them with one call into
  `actor_critic.sample_batch` + `decode_decoder_action` per env step.
- Re-extract `data/forge_choice_situations` with V2 records so
  `scripts/decoder_bc_parity.py` can run again as the live regression.
- Wire the native `mage.decision_mask_next` callback into
  `decoder_sample` to remove the per-step CPU mask + host sync
  (currently the Python `batch_next_mask` is the bottleneck on small
  batches).

### Cumulative cutover stats (phases 1-7, vs `main`)

```
43 files changed, 2272 insertions(+), 12314 deletions(-)
```

Net **-10 042 LoC**.
