# Inference pipeline refactor

## Motivation

Post-profile state (2026-05): the inference forward is eager, runs ~500
`cudaLaunchKernel` per merged batch, and accounts for ~75% of CPU time in the
profile. Two compile wrappers exist in the codebase but neither covers the
inference forward. The policy stack is a four-class onion
(`LSTMStatefulTextPolicy` → `RecurrentTextPolicy` → `TextPolicy` →
`TextStateEncoder`) with three different entry points (`forward_packed`,
`encode_with_history`, `encode_packed_only`). The inference server pokes
`policy.policy.text_policy` to reach through these.

Host-side bookkeeping is fine; the model forward is not. The plan below
collapses the wrapper stack, extracts a reusable inference pipeline, drops
dead reply/request fields, replaces the host concat with a CPU arena, and
finally captures the encoder + decoder as CUDA graphs.

## Commit order

1. **Phase D** — Drop unused reply/request fields (warm-up).
2. **Phase A** — Collapse policy + compile encoder.
3. **Phase B** — Extract `TextInferencePipeline`.
4. **Phase E** — Host (CPU) arena replaces `_concat_packed_text_batches`.
5. **Phase C** — Bucketing + `mode="reduce-overhead"` (CUDA Graphs).
6. **Phase F** — Compile decoder loop.

D unblocks A by shrinking the surface. A + B produce the single forward C/F
target. E precedes C because CUDA Graphs want fixed input addresses, which
the arena provides. C delivers the asymptotic launch-overhead collapse;
F compounds.

---

## Phase D — Drop unused reply/request fields ✅ (commit 0e3fcae)

**Goal:** remove dead code that the prior cleanups left around.

- [x] Drop `decoder_batch` from `TextInferenceReply`.
- [x] Move any transcript reads to `host_decoder`.
- [x] Drop `native_batch` from `TextInferenceRequest` and `_SubmitBatch`.
- [x] Drop `NativeEncodedBatch` import where no longer needed.
- [x] Drop the dead `ready_event` field too.
- [x] Tests pass.

## Phase A — Policy stack collapse + compile ⚠️ (scope reduced; commit e041cb1)

**Goal:** one `forward` on a single policy class, compile-wrapped, used by
both inference and training.

Scope reduced to the perf-critical subset: compile-wrap
`RecurrentTextPolicy.encode_with_history` (the inference forward) instead
of fully collapsing the wrapper stack. Tests bind directly to each class
in the onion (LSTMStateful, Recurrent, Text, Encoder); a full collapse is
high blast radius and the perf gain (compiling the forward) is the same.

- [x] `encode_with_history` lazily compile-wrapped on first CUDA call.
- [x] Nested compile from `forward_packed → _run` bypassed via direct
      `_encode_with_history_impl` call.
- [x] Tests pass.
- [ ] (Deferred) merge `RecurrentTextPolicy` + `TextPolicy` into one
      class. Likely needs its own scoped refactor PR.

## Phase B — Extract `TextInferencePipeline` ✅ (commit 9c93da5)

**Goal:** the inference forward is its own object, reusable outside the
server.

- [x] Create `magic_ai/text_encoder/inference_pipeline.py` with
      `TextInferencePipeline`.
- [x] Move `_sample_decoder` body into `pipeline.encode_and_sample`.
- [x] Server holds a pipeline; calls `pipeline.encode_and_sample(policy,
      …)` (pipeline is stateless w.r.t. policy so it composes with the
      version manager).
- [x] Server's `_sample_decoder` shrinks to env-state plumbing only.
- [x] Tests pass.
- [ ] (Future) opponent-pool inline forward → pipeline. Low priority.

## Phase E — Host arena replaces `_concat_packed_text_batches` ⚠️ (server-side variant)

**Goal:** eliminate the per-merge alloc/concat pattern; give CUDA Graphs
fixed-address inputs.

Scope: server-side arena (single-threaded write inside `_process`, not
the actor-direct-write ring originally sketched). The actor-side
direct-write variant — actors reserve a slot and write their packed
batch into pinned host memory before submitting — is a follow-up. The
server-side arena delivers the same fixed-address property for Phase C
and removes the per-merge allocations; the only thing it doesn't
amortize is the per-actor copy from encoder scratch to merged buffers,
which is fast anyway.

- [x] `_HostPackedArena`: long-lived (lazily grown, pinned-host where
      CUDA is available) buffers for the merged batch.
- [x] `_process` calls `arena.merge(items)` instead of
      `_concat_packed_text_batches(...)`.
- [x] Actors keep their `batch.packed` reference for per-row staging.
- [x] Arena correctness covered by tests (matches `_concat_*` output
      on single + multi-merge with shrinking widths).
- [ ] (Future) Actor-side direct-write arena, slot reservation.
- [ ] (Future) Drop `_concat_packed_text_batches` entirely once nothing
      uses it outside tests.

## Phase C — Bucketing + CUDA Graphs ✅ (encoder only)

**Goal:** the inference encoder forward is captured per shape bucket and
replayed; near-all per-launch overhead on the encoder side disappears.

- [x] `DEFAULT_BUCKETS = [(16,2048), (64,8192), (192,24576), (384,49152),
      (768,98304)]` — sized so `T_b - R_b` ≥ typical merged `(T - R)`.
- [x] `_pad_packed_to_bucket(host_packed, R_b, T_b)`: pad with dummy
      rows that absorb the slack PAD tokens; each dummy row has ≥ 1
      token so flash_attn varlen rejects nothing.
- [x] `_slice_padded_packed_to_live` + `_slice_encoded_to_live`: cut
      the encoder outputs back to live rows/tokens for downstream
      decoder + heads.
- [x] Per-bucket lazy `torch.compile(..., mode="reduce-overhead",
      dynamic=False)` on `_encode_with_history_impl`.
- [x] Server passes the host (unpadded) batch into the pipeline; pad +
      H→D happens inside the bucketed path.
- [x] Smoke test on tiny model: first call 7.9 s (compile), steady 3.6
      ms/call. Decoder + value head still eager (Phase F).
- [ ] (Future) Tune bucket boundaries against real workload distribution.
- [ ] (Future) Verify correctness vs. eager on the full pipeline (the
      smoke checks shape only).

## Phase F — Decoder loop compile

**Goal:** capture the autoregressive decoder inside the graph too.

- [ ] Try `torch.compile(fullgraph=True)` on `encode_and_sample`.
- [ ] If it fails, capture the decoder loop as a separate graph with fixed
      L_max per bucket.
- [ ] Verify outputs match the non-compiled path.
- [ ] Profile: `cudaLaunchKernel` ~= 1 launch per bucket replay.
- [ ] Document bucket set + tuning knobs in
      `magic_ai/text_encoder/AGENTS.md`.

---

## What we are explicitly NOT doing

- Not moving replay back to GPU.
- Not bringing back a GPU arena.
- Not fusing actor + server threads.
