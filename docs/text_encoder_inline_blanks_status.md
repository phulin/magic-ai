# Inline-Blank Migration — Status

Companion to `docs/text_encoder_inline_blanks_plan.md`. Tracks live state of
the eight-step migration. Update at every step boundary.

## Worktrees

- **magic-ai**: `/home/user/magic-ai-inline-blanks` on branch
  `text-encoder-inline-blanks` (off main `379cd7d`). All Python-side work
  lives here. Main checkout at `/home/user/magic-ai` is reserved for
  unrelated work.
- **mage-go**: `/home/user/mage-go/.claude/worktrees/inline-blanks` on
  branch `text-encoder-inline-blanks` (commit `a773863`). All native ABI
  work lives here.

## Step status

| # | Step                                       | Status        | Notes |
|---|--------------------------------------------|---------------|-------|
| 1 | Token table additions                      | ✅ done       | 30 new tokens; tokenizer rebuilt; Python `_Packed` holds singletons + `num_ids`. |
| 2 | Render priority blanks (flag-gated)        | ✅ done       | `BlankAnchor`, `RenderError`, `<choices>…</choices>` block. Legacy path byte-identical when flag off. |
| 3 | Batch + native assembler plumbing          | ✅ done       | Python + native paths tested; mage-go exposes `MagePackedBlankOutputs` and regenerated cffi. |
| 4 | `InlineBlankPolicy` + value-head wiring    | ✅ done       | Python path wired behind `TextEncoderConfig.use_inline_blanks`. |
| 5 | BC parity gate (priority-only)             | 🚧 harnessed  | Loss/accuracy utilities and fixed-trace parity CLI landed; real trace gate still pending. |
| 6 | Combat blocks                               | 🚧 renderer slice | `<choose-block>` render/batch/model path started; sampler/action adapter still pending. |
| 7 | Targets / modes / mays / X / mana sources  | ⏳ blocked-by 6 |  |
| 8 | Delete legacy option/target heads          | ⏳ blocked-by 7 |  |

## What landed in each completed step

### Step 1 (`/home/user/magic-ai-inline-blanks`)

- `magic_ai/text_encoder/tokenizer.py` — `CHOOSE_KIND_TOKENS`,
  `BLANK_ANSWER_TOKENS`, `NUM_TOKENS`, `INLINE_BLANK_TOKENS`, `MAX_NUM=16`,
  `num_token()` helper. Appended to `ALL_CUSTOM_TOKENS`. Vocab 974 → 1004.
- `magic_ai/text_encoder/token_tables.py` — 14 single-id fields +
  `num_ids: list[int]`. Populated in `build_token_tables`.
- `magic_ai/text_encoder/native_token_tables.py` — held in `_Packed` and,
  after Step 3 native wiring, passed through `MageRegisterTokenTables`.
- `tests/test_text_encoder_tokenizer.py`, `tests/test_text_token_tables.py`,
  `tests/test_native_token_tables.py` — distinctness + round-trip.
- Persisted tokenizer rebuilt via `scripts/build_text_encoder_vocab.py`.

### Step 2 (`/home/user/magic-ai-inline-blanks`)

- `magic_ai/text_encoder/render.py` — `BlankAnchor` dataclass,
  `RenderError`, `RenderedSnapshot.blank_anchors`,
  `use_inline_blanks: bool` + `chosen_token_id: int | None` on
  `SnapshotRenderer.__init__` / `.render` / `render_snapshot`.
  Inline mode classifies pending options into per-card `<choose-play>` /
  `<use-ability>` blanks (emitted right after each card's `</card>`)
  plus a trailing `<choices>…</choices>` block with one `<pass>` blank
  per pass option. All anchors share `group_id=0`,
  `group_kind="CROSS_BLANK"`, `legal_token_ids=(chosen_token_id,)`.
- `tests/test_text_render.py` — five new tests including a re-render
  parity test under permuted option order.

### Step 3 — Python side (`/home/user/magic-ai-inline-blanks`)

- `magic_ai/text_encoder/render_plan.py` — `OP_EMIT_BLANK = 30`,
  `OP_EMIT_BLANK_LEGAL = 31`, `BLANK_GROUP_PER_BLANK/CROSS_BLANK/CONSTRAINED`,
  `RenderPlanWriter.emit_blank(...)`, `emit_blank_anchors(...)`.
- `magic_ai/text_encoder/batch.py` — `TextEncodedBatch` and
  `PackedTextBatch` gain `blank_positions`, `blank_kind`, `blank_group`,
  `blank_group_kind`, `blank_legal_ids`, `blank_legal_mask`. `collate`
  and `pack_batch` propagate; `pack_batch` rebases `blank_positions`.
- `magic_ai/text_encoder/assembler.py` — Python (non-native) path
  handles the two new opcodes; rejects stray `OP_EMIT_BLANK_LEGAL`.
- `magic_ai/text_encoder/native_assembler.py` — allocates/copies native
  blank buffers and passes `MageBlankAssemblerConfig` /
  `MagePackedBlankOutputs` through cffi/ctypes when
  `use_inline_blanks=True`.
- `magic_ai/text_encoder/replay_triton.py` —
  `TODO(step3-replay-triton-blanks)`; deferred to Step 4 since the
  replay buffer doesn't yet store these fields.
- `tests/test_text_assembler.py`, `tests/test_native_token_tables.py`,
  `tests/test_native_assembler_parity.py` — Python opcode coverage,
  cffi singleton/num-id round-trips, native blank-buffer allocation, and a
  real-game `MageEncodeTokensPacked(use_inline_blanks=True)` smoke test.

### Step 3 — mage-go side (`/home/user/mage-go`)

- `cmd/pylib/abi.h` — `MageTokenTables` extended with 14 singletons +
  `num_count` + `num_ids`; added `MageBlankAssemblerConfig` and
  `MagePackedBlankOutputs`.
- `cmd/pylib/render_plan.go` — `opEmitBlank=30`,
  `opEmitBlankLegal=31`; group-kind enum; native priority inline mode emits
  `<choose-play>` / `<use-ability>` anchors next to source cards and `<pass>`
  anchors at the trailing choice point when blank outputs are requested.
- `cmd/pylib/blank_assembler.go`, `cmd/pylib/token_assembler.go`,
  `cmd/pylib/main.go`, `cmd/pylib/encoder.go` — `blankCollector` is wired
  into `MageEncodeTokensPacked`; positions are absolute packed offsets,
  legal ids/masks are padded to caller `K,V`, and packed-row compaction
  rebases blank positions.
- `MageTokenTableLookup` extended with kind=12, kind=13.
- `cmd/pylib/mage/__init__.py` `_CDEF` regenerated manually for the new
  token-table and blank-output structs; `cmd/pylib/mage/libmage.so` rebuilt
  and reinstalled into this repo via
  `uv pip install --reinstall /home/user/mage-go`.
- Go tests pass; `go build -buildmode=c-shared` succeeds.

### Step 4 — Python side (`/home/user/magic-ai-inline-blanks`)

- `magic_ai/text_encoder/model.py` — added `TextEncoderConfig.use_inline_blanks`
  and `InlineBlankPolicy`, which gathers blank hidden states from padded or
  packed encoder outputs, scores only each blank's legal token ids against the
  tied input embedding rows, applies a learned per-kind temperature, and masks
  padded legal slots to `-inf`.
- `magic_ai/text_encoder/policy.py` — `TextPolicy` now owns an `MLMHead` and
  passes its dense + layer-norm pre-projection modules into
  `InlineBlankPolicy`. `TextPolicyOutput` carries `blank_logits` plus the
  batch's `blank_*` metadata when the flag is enabled. `encode_snapshots(...)`
  accepts `use_inline_blanks=True` and defaults `chosen_token_id` from the
  tokenizer's `<chosen>` id.
- `tests/test_text_encoder_model.py`, `tests/test_text_policy.py` — focused
  scorer mask/backprop coverage plus an end-to-end inline-blank snapshot
  forward smoke test. Targeted slice:
  `uv run pytest tests/test_text_encoder_model.py tests/test_text_policy.py tests/test_text_assembler.py -q`
  → **22 passed / 1 skipped**.

### Step 5 — BC utility scaffold (`/home/user/magic-ai-inline-blanks`)

- `magic_ai/text_encoder/training.py` — added
  `inline_blank_priority_loss(...)` and
  `inline_blank_priority_accuracy(...)` for priority-only `CROSS_BLANK`
  groups. The loss treats slot 0 of each blank's legal-token logits as the
  `<chosen>` score, masks padded / non-cross anchors, skips ignored target
  rows, and returns a differentiable zero when no rows are valid.
- `tests/test_text_encoder_training.py` — coverage for low/high CE,
  padding and non-cross masking, ignored rows, all-ignored zero loss, accuracy
  accounting, and gradient flow.
- Targeted slice:
  `uv run pytest tests/test_text_encoder_training.py -q`
  → **17 passed**.
- `scripts/inline_blank_bc_parity.py` — added a priority-only parity CLI that
  trains legacy option-head BC and inline cross-blank BC on the same fixed
  JSONL rows, then fails if held-out inline accuracy regresses by more than
  the configured pp threshold. It accepts trace rows with either
  `selected_option_index`, `selected_option_id`, `trace.indices`, or
  transcript-style `state`/`pending`/`action` fields; result JSON records the
  source hash, seed, model config, and training config. Includes a
  `--synthetic-fixture` smoke mode.
- `tests/test_inline_blank_bc_parity.py` — coverage for synthetic fixture
  generation, trace loading, selected option-id/action/trace resolution, and
  render-order target mapping for legacy vs inline rows.
- `scripts/train.py` — added `--priority-trace-jsonl-path`; sampled game
  transcripts now optionally append gate-ready priority JSONL rows with
  `state`, `pending`, and `action` fields while leaving the human-readable
  `--game-log-path` output unchanged.
- `scripts/play_text_rollout.py` — added `--priority-trace-jsonl-path` for a
  lightweight engine-backed trace smoke path that does not require slot-card
  embeddings; missing default deck file falls back to a small built-in deck.
- `tests/test_train.py` — coverage for JSONL trace append behavior and path
  defaulting.
- Smoke:
  `uv run python scripts/inline_blank_bc_parity.py --synthetic-fixture 8 --epochs 1 --batch-size 4 --d-model 32 --n-layers 1 --n-heads 4 --d-ff 64 --max-seq-len 512 --seed 0`
  → **passed**.
- Engine-backed smoke:
  `uv run python scripts/play_text_rollout.py --n-episodes 1 --max-turns 1 --device cpu --seed 0 --d-model 32 --n-layers 1 --n-heads 4 --max-tokens 2048 --priority-trace-jsonl-path /tmp/inline_blank_priority_trace.jsonl`
  then
  `uv run python scripts/inline_blank_bc_parity.py --trace-jsonl /tmp/inline_blank_priority_trace.jsonl --epochs 1 --batch-size 1 --d-model 32 --n-layers 1 --n-heads 4 --d-ff 64 --max-seq-len 2048 --seed 0`
  → **passed** on 3 trace rows.

## Open blockers

None for Steps 1-4. Step 5 still has harnesses but the migration is proceeding
without treating the accuracy gate as a blocker.

### Step 6 — Combat block renderer slice (`/home/user/magic-ai-inline-blanks`)

- `magic_ai/text_encoder/render.py` — inline mode now classifies `block`
  options into `<choose-block>` blanks next to each defender. Legal ids are
  `<none>` followed by the legal attacker `<card-ref:K>` ids in target order;
  anchors use `group_kind="CONSTRAINED"`.
- `magic_ai/text_encoder/policy.py` — `TextPolicy.encode_snapshots(...)`
  supplies `<none>` and `<card-ref:K>` token ids to the renderer for block
  blank legal vocabularies.
- `magic_ai/text_encoder/training.py` — added
  `inline_blank_per_blank_loss(...)` and
  `inline_blank_per_blank_accuracy(...)` for `PER_BLANK` and `CONSTRAINED`
  groups; block blanks use this target shape (`0=<none>`, `1..N=attackers`).
- `tests/test_text_render.py`, `tests/test_text_policy.py` — coverage for
  block anchor placement/legal ids and end-to-end inline block blank forward.
- `tests/test_text_encoder_training.py` — coverage for constrained
  per-blank loss/accuracy and masking.
- `magic_ai/actions.py` — added `action_from_inline_block_choices(...)` to
  map per-defender inline legal-slot selections back to the existing
  `{"blockers": ...}` action payload.
- `tests/test_actions.py` — coverage for inline block action decoding.

## Next steps

1. **Step 6 — Combat blocks.** Wire inline-block sampling into the live text
   actor path and replay scoring path.
2. **Step 7** — targets/modes/mays/X-cost/mana sources.
3. **Step 8** — delete `PolicyHead`, `TargetHead`, `option_*` /
   `target_*` batch fields, the legacy renderer branch, and the
   `use_inline_blanks` flag. Bump replay-buffer on-disk version.
