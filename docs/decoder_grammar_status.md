# Decoder grammar migration — status

Tracking implementation of `docs/decoder_grammar_plan.md`.

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
