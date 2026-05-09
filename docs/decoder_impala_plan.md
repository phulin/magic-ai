# Decoder IMPALA pipeline migration — plan

After the inline-blank → grammar-decoder cutover landed (`19542b2`), the
single-policy in-process text loop (`train_text_envs`) and all training
verifications run on the decoder pipeline. The **batched IMPALA actor pipeline**
(`train_text_native_batched_envs` and friends) still expects inline-blank
shapes and is gated behind four `NotImplementedError` stubs in `scripts/train.py`
(`build_text_decision_layout`, `infer_text_trace_kind`, `_decode_text_action`,
`NativeTextTrajectoryBuffer.__init__`).

This doc plans the protocol redesign needed to bring that high-throughput path
back online with the decoder.

## Why it's a redesign, not a port

The IMPALA path uses an inference-server / actor-loop pattern: many envs run
on CPU actors (`TextRolloutActor`); a central `TextInferenceServer` batches
their snapshots, runs the policy on GPU, and ships back a per-env action
selection that gets fed into `MageBatchStepByChoice` (the batched native
engine-step). The wire shape between actor → server → engine is currently:

```
class NativeTextSampleBatch:
    decision_counts: list[int]          # per env: how many independent inline blanks
    selected_choice_cols: list[int]     # flat: chosen legal-id-column per blank
    may_selected: list[int]             # per env, MAY answer (0/1)

# C ABI for the engine step:
typedef struct {
    int64_t n;
    int64_t max_options;
    int64_t max_targets_per_option;
    const int64_t* handles;             # per-env engine handle
    const int64_t* decision_start;      # offset into selected_choice_cols
    const int64_t* decision_count;      # length per env
    const int64_t* selected_choice_cols;# flat per-blank choices
    const int64_t* may_selected;        # per env
} MageStepChoiceRequest;
```

That schema embeds the inline-blank assumption that an action is N independent
column picks plus one MAY bit. Decoder actions don't fit this:

- `PRIORITY` → one engine option index (single pointer step)
- `DECLARE_ATTACKERS` → variable-length list of `(attacker_id, defender_id)` tuples
- `DECLARE_BLOCKERS` → variable-length list of `(blocker_id, attacker_id)` tuples
- `CHOOSE_TARGETS` → 1 or more target ids (variable per spell)
- `MAY` → boolean
- `CHOOSE_MODE` / `CHOOSE_X` → integer

The batched engine step needs to accept all of these shapes.

## Two viable shapes for the new wire protocol

### Option A: thin wire, fat translator (recommended)

Keep the wire compact: per-env send the *raw decoder output* plus the
`pointer_anchor_handles`. The Go side runs a per-env decision-type-dispatched
translator (mirroring Python's `decode_decoder_action`) and applies the
resulting engine action.

```c
typedef struct {
    int64_t n;
    int64_t max_decode_len;
    int64_t max_anchors;
    const int64_t* handles;                    // [n]
    const int64_t* decision_type;              // [n] DecisionType enum
    const int32_t* output_token_ids;           // [n, max_decode_len] padded
    const int32_t* output_pointer_subjects;    // [n, max_decode_len] subject_index per pointer step (-1 elsewhere)
    const uint8_t*  output_is_pointer;         // [n, max_decode_len] bool
    const int32_t* output_lens;                // [n] valid length per env
    const int32_t* pointer_anchor_handles;     // [n, max_anchors] engine handles per anchor subject
    const int32_t* pointer_anchor_count;       // [n] anchors per env
} MageDecoderStepRequest;
```

Pros:
- Wire is small: ~64 ints per env per step at typical lengths.
- The translator lives in one place (Go), reused across actors.
- Maps cleanly to `mage-go/cmd/pylib/decision_spec_emitter.go`'s subject-index
  → handle bookkeeping.

Cons:
- Need to write a Go-side per-decision-type dispatch
  (`decodeDecoderAction(decisionType, tokens, pointers, handles)` →
  per-decision-type engine call). This is genuinely new code, ~300 LoC.

### Option B: per-decision-type wire structs

Send the already-translated engine action over the wire — one struct per
DecisionType. PRIORITY: int. DECLARE_ATTACKERS: list of pairs. Etc. Python
calls `decode_decoder_action` actor-side, ships the engine action, Go just
applies it.

Pros:
- No new Go translator; reuse the existing per-decision-type engine APIs.
- Easier to validate: one struct per decision matches the engine's API directly.

Cons:
- Wire is variable-shape, requires a tagged-union encoding.
- More Python work: actor-side translation, more allocs per step.

**Recommendation: Option A**. Centralizing the translator in Go matches the
existing pattern (`MageBatchStepByChoice` already does the per-env dispatch
inside the cgo entrypoint), and the wire stays a flat int rectangle which
plays nicely with `unsafe.Slice` zero-copy.

## File-by-file work plan (Option A)

Estimated: 2–3 focused days. Items in order, each independently shippable.

### 1. Go ABI: new request struct + cgo entrypoint

**`mage-go/cmd/pylib/abi.h`** — add `MageDecoderStepRequest` (above).

**`mage-go/cmd/pylib/main.go`** — add `MageBatchStepByDecoderAction(req
*C.MageDecoderStepRequest) (res C.MageEncodeResult)`. Body parallels
`MageBatchStepByChoice`: validate, slice the input arrays, parallel-fan-out
per env via existing handle-pool, call `applyDecoderAction(handle, ...)`.

**`mage-go/cmd/pylib/decision_action_apply.go`** (new) — implements
`applyDecoderAction(decisionType int32, tokens []int32, pointerSubjects []int32,
isPointer []uint8, length int32, anchorHandles []int32, anchorCount int32,
handle int64) error`. Dispatch on `decisionType`:

- `PRIORITY` (0): single pointer step. `engineOptionIdx := anchorHandles[ptr_subject]`.
  Call `engine.PlayPriority(handle, engineOptionIdx)`.
- `DECLARE_ATTACKERS` (1): walk tokens. For each `ATTACK ptr_attacker DEFENDER ptr_defender`
  4-tuple, build `(attacker_handle, defender_handle)`. Call
  `engine.DeclareAttackers(handle, pairs)`.
- `DECLARE_BLOCKERS` (2): same shape, `BLOCK ptr_blocker ATTACKER ptr_attacker`.
- `CHOOSE_TARGETS` (3): walk pointer steps; collect `target_handle` per slot.
  `engine.ChooseTargets(handle, targets)`.
- `MAY` (4): single vocab step (YES/NO). `engine.AnswerMay(handle, isYes)`.
- `CHOOSE_MODE` (5) / `CHOOSE_X` (6): walk digit tokens, parse to int.
  `engine.ChooseMode(handle, n)` / `engine.ChooseX(handle, x)`.

The engine-side per-decision-type APIs already exist (the inline-blank path
called them via `MageBatchStepByChoice`'s dispatcher — find the analog code
in `main.go` around line 1380+ and reuse).

Tests: `decision_action_apply_test.go` — one fixture per decision kind asserts
the engine state advances correctly after a synthetic decoder action.

### 2. Python wrapper

**`mage-go/cmd/pylib/mage/__init__.py`** — add the cffi `_CDEF` for
`MageDecoderStepRequest` + `MageBatchStepByDecoderAction`. Add a Python
wrapper `batch_step_by_decoder_action(handles, decision_type, output_tokens,
output_pointer_subjects, output_is_pointer, output_lens, anchor_handles,
anchor_count)` that allocates the cffi struct, casts the torch / numpy
buffers, calls the cgo function, returns the encode result.

Reinstall: `cd /home/user/magic-ai && uv sync --reinstall-package mage-go`.

### 3. Replace `NativeTextSampleBatch` with `NativeTextDecoderBatch`

**`scripts/train.py`** — new dataclass mirroring the Go ABI fields:

```python
@dataclass
class NativeTextDecoderBatch:
    decision_type: Tensor              # [B] int32
    output_token_ids: Tensor           # [B, L_max] int32
    output_pointer_subjects: Tensor    # [B, L_max] int32
    output_is_pointer: Tensor          # [B, L_max] bool
    output_lens: Tensor                # [B] int32
    pointer_anchor_handles: Tensor     # [B, N_max] int32
    pointer_anchor_count: Tensor       # [B] int32
    # Plus the IMPALA-side bookkeeping that today lives on NativeTextSampleBatch:
    log_probs: Tensor                  # [B, L_max] for IMPALA importance ratios
    value: Tensor                      # [B] from value head
    pad_mask: Tensor                   # [B, L_max]
```

Replace every `NativeTextSampleBatch` use site (~9 in train.py per the grep
above) with the new shape. Drop `selected_choice_cols` and `may_selected`
references entirely.

### 4. `TextInferenceServer` request/response contract

**`magic_ai/native/inference_server.py`** — the server batches snapshots from
multiple actors and ships back a per-env action selection. Today it returns
the inline-blank shape. New flow:

- Actors submit `(env_id, encoded_batch_slice, batch_handle)` requests.
- Server gathers, runs `text_policy.encode_only` on the concatenated batch,
  calls `actor_critic.sample_batch(...)`.
- Server returns per-env `NativeTextDecoderBatch` slices (one row per env).
- Actors call `mage.batch_step_by_decoder_action(...)` with their assigned slice.

Most of the gather/scatter plumbing stays — only the per-env payload shape
changes. Find the existing `NativeTextSampleBatch`-shaped queue + response
types and rewrite.

### 5. `TextRolloutActor`

**`magic_ai/native/rollout_actor.py`** — receives the decoder batch slice from
the server, calls `mage.batch_step_by_decoder_action` for its envs, captures
the decoder targets into the per-actor `NativeTextTrajectoryBuffer` for
replay. Simpler than today since the actor no longer translates per-blank
columns — the engine does the translation server-side.

### 6. `NativeTextTrajectoryBuffer`

**`scripts/train.py`** — implement the stub. Per-env ring of
`DecoderDecisionPayload` (already defined in `replay_buffer.py`). On each
env-step, append the per-env slice of the latest `NativeTextDecoderBatch`
(plus the pointer_mask + vocab_mask snapshots needed by replay-time scoring).
On episode end, call `replay_buffer.commit_decoder_decision(reservation,
payload)` once per captured decision. The episode-boundary detection logic
already exists — find it (`flush_episode` or similar).

Storage shape per env-step: same as `DecoderDecisionPayload` defined in
`replay_buffer.py:DecoderDecisionPayload` (reuse, don't redefine).

### 7. `_decode_text_action` / `infer_text_trace_kind` / `build_text_decision_layout`

These three stubs become thin wrappers (or vanish entirely):

- `_decode_text_action`: redundant; the Go side now applies the action.
  Delete.
- `infer_text_trace_kind`: small switch from `pending["kind"]` to TraceKind.
  Already implemented in `scripts/train.py::_decision_type_to_trace_kind`
  (Phase 7) — promote it.
- `build_text_decision_layout`: returns `DecoderDecisionLayout` from the
  decoder sample output. Already exists in `actor_critic.py`. Promote.

### 8. Re-enable skipped tests

`tests/test_inference_server.py::InferenceWorkRingTest`,
`InferenceServerBatchingTest`, `tests/test_train.py::test_native_text_staging_commits_*`
were deleted in Phase 7 because their fixtures used inline-blank shapes.
Rewrite them against `NativeTextDecoderBatch` with synthetic decoder
fixtures (mirror `tests/test_actor_critic_decoder.py`'s setup).

### 9. End-to-end smoke

Extend `scripts/smoke_decoder_train.py` (or add `scripts/smoke_impala_train.py`)
to:

- Build a `TextInferenceServer` + `TextRolloutActor` with N=4 envs.
- Run 16 native env-steps via the IMPALA loop.
- Assert: every step produced a legal action, replay buffer has the expected
  decoder targets, no NaN in log_probs.

Then a real (small) RL training run:
```
uv run scripts/train.py --encoder text --trainer rnad --num-envs 32 \
    --rollout-steps 64 --total-steps 1024 --device cuda \
    --pretrain-mlm-dir data/forge_choice_situations_v2
```
Should run 1024 steps without crashing and produce sensible loss curves.

## Forge attack/block translator follow-up (parallel, smaller)

Independent of the IMPALA work but blocking full BC parity coverage:

**`magic_ai/text_encoder/forge_target_encoding.py`** —
`_attack_label`/`_block_label` were written for legacy Forge log format
(`"PlayerA assigned X to attack PlayerB"` /
`"PlayerA blocked X with Y"`). Current Forge logs use:

```
PlayerA attacks PlayerB with N creatures
  Attacker: <name> [<id>]
  Attacker: <name> [<id>]
PlayerB blocks with M creatures
  Blocker: <name> [<id>] blocks <attacker_name> [<attacker_id>]
```

Update the translators to match the new format. Re-extract Forge corpus
afterwards. Should yield ~17k attack records + ~17k block records (vs the
current 0).

Estimated: half a day. Tests at `tests/test_forge_target_encoding.py` cover
the attacker / blocker translators — extend with fixtures from real Forge
log lines.

## Sequencing recommendation

Three sessions:

1. **Session A: Go-side engine apply + cgo + Python wrapper** (items 1–2). Independently testable in isolation: the new Go function + cgo glue + `tests/test_native_decoder_action.py`. End: `mage.batch_step_by_decoder_action(...)` works against a real game in a unit test.
2. **Session B: Inference server + actor + trajectory buffer + train.py rewiring** (items 3–7). Depends on Session A.
3. **Session C: Tests + smoke + Forge translator follow-up** (items 8–9 + Forge).

Each session ends with a green pytest + a working smoke. The IMPALA pipeline
is non-trivial but not fundamentally hard — the key insight is moving the
per-env action-translation responsibility from per-actor Python to per-env
Go inside the batched cgo step.
