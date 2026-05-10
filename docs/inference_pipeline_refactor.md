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

## Phase D — Drop unused reply/request fields

**Goal:** remove dead code that the prior cleanups left around.

- [ ] Drop `decoder_batch` from `TextInferenceReply`.
- [ ] Move any transcript reads to `host_decoder`.
- [ ] Drop `native_batch` from `TextInferenceRequest` and `_SubmitBatch`.
- [ ] Drop `NativeEncodedBatch` import where no longer needed.
- [ ] Tests + smoke training pass.

## Phase A — Policy stack collapse + compile

**Goal:** one `forward` on a single policy class, compile-wrapped, used by
both inference and training.

- [ ] Identify callers of `encode_with_history`, `encode_only`,
      `encode_packed_only`, `forward_packed`, `TextPolicy.forward`.
- [ ] Merge `RecurrentTextPolicy` + `TextPolicy` into one class.
- [ ] Add `TextPolicy.forward(merged_packed, h_in, c_in) → ForwardOutput`.
- [ ] `torch.compile(..., dynamic=True)` around it on first CUDA call.
- [ ] `_sample_decoder` calls the unified forward (no `hasattr(policy,
      "policy")` branching).
- [ ] `_score_replay_rows` and opponent-pool eval use the same forward.
- [ ] Delete dead entry points.
- [ ] Tests + training; verify loss/throughput.

## Phase B — Extract `TextInferencePipeline`

**Goal:** the inference forward is its own object, reusable outside the
server.

- [ ] Create `magic_ai/text_encoder/inference_pipeline.py` with
      `TextInferencePipeline`.
- [ ] Move `_sample_decoder` body into `pipeline.encode_and_sample`.
- [ ] Server holds a pipeline rather than a raw policy.
- [ ] Replace opponent-pool's inline forward with the pipeline.
- [ ] Move `lstm_env_state_inputs` / `scatter_lstm_env_states` calls into a
      clear "env state" layer wrapping the pipeline so the pipeline is
      stateless.
- [ ] Tests + training.

## Phase E — Host arena replaces `_concat_packed_text_batches`

**Goal:** eliminate the 137 ms/5 s `aten::index_fill_` from host concat
padding; give CUDA Graphs fixed-address inputs.

- [ ] Define `HostPackedArena` (pinned-host ring buffers mirroring the old
      GPU arena's fields).
- [ ] `_PendingItem` regains `row_start/row_end/token_start/token_end`.
- [ ] `_InferenceWorkRing.put` writes into the arena with offset shifts.
- [ ] Replace `_concat_packed_text_batches` in `_process` with
      `arena.slice_for(items)` returning a host `PackedTextBatch` view.
- [ ] Actors keep their `batch.packed` reference for per-row staging
      slices (simpler than plumbing the arena through).
- [ ] Drop `_concat_packed_text_batches` (or move to tests/).
- [ ] Profile: `index_fill_` gone.

## Phase C — Bucketing + CUDA Graphs

**Goal:** the inference forward is captured per shape bucket and replayed;
near-all per-launch overhead disappears.

- [ ] Define `Bucket(rows, max_seq)` and a small `BucketTable`.
- [ ] Pipeline pre-allocates static GPU input/output buffers per bucket.
- [ ] `pad_merged_to_bucket(host_packed, bucket) → host_padded`.
- [ ] Wrap `encode_and_sample` with `torch.compile(mode="reduce-overhead",
      dynamic=False)`, one compiled callable per bucket.
- [ ] If reduce-overhead can't fold the decoder loop, fall back to explicit
      `torch.cuda.graph()` capture (warmup → capture → replay).
- [ ] Correctness check vs. non-bucketed path within fp tolerance.
- [ ] Profile: `cudaLaunchKernel` drops dramatically.
- [ ] Training end-to-end: loss curve unchanged.

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
