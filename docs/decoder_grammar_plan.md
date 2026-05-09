# Autoregressive Grammar Decoder — Transition Plan

## Status & motivation

The current text-encoder pipeline is *inline-only*: every pending decision is
materialized as a `<choose-*>` blank token at its natural position inside the
state text, scored by a tied-embedding "fill the blank" head, and sampled
group-by-group (see `docs/text_encoder_inline_blanks_plan.md` and
`docs/text_encoder_inline_blanks_status.md`). This unifies pretraining (MLM)
with the policy and removed the legacy option/target heads.

Inline blanks have hit their structural ceiling. The status doc enumerates
nine fidelity gaps that all share one root cause: **the action representation
is fixed by the renderer at encode time**, so any decision whose shape depends
on previously-chosen sub-answers (multi-target with shrinking legal sets,
damage-assignment order, multi-pip mana payment with constrained sources,
cross-blank combat constraints, X-as-digit-sequence, …) cannot be expressed
without either pre-enumerating combinations or punting to a follow-up snapshot.
The "single forward pass scores all blanks independently" factorization is
also a hard approximation that we already noted would need iterative-fill
escalation for blocks.

This plan transitions to a **bidirectional encoder + autoregressive grammar
decoder** architecture in which:

- The encoder consumes (game state) **plus** an explicit (decision spec) —
  the engine tells the model *what kind of answer it needs* rather than the
  model inferring it from blank tokens scattered through the state.
- The decoder emits a typed action program one token at a time, cross-attending
  to the encoder. The engine supplies an incremental next-token mask after
  every emitted token, so legal-by-construction holds at every prefix and
  cross-blank constraints are enforceable.
- Object identity stays a **pointer-attention** output over encoder positions
  (same idea as scoring `<card-ref:K>` against legal candidates today), not a
  giant flat vocabulary.

## Architecture

```
state tokens + decision-spec tokens + history token
        ↓
bidirectional encoder (existing ModernBERT trunk)
        ↓
encoded context  E ∈ R[T, D]
        ↓
autoregressive grammar decoder (small causal transformer, cross-attends to E)
        ↓
   per-step:  vocab logits  (grammar tokens: <ATTACK>, <BLOCK>, <YES>, <END>, …)
              pointer logits (dot-product attention over E for OBJ_REF / ACTION_REF)
              engine mask    (kills illegal next tokens given prefix + decision spec)
        ↓
sample/argmax → next token → engine advances mask state → repeat until <END>
```

### Encoder input layout

```
[CLS] [HIST]
[DECISION_TYPE: ...]  [SPEC tokens for this decision...]
[STATE_BEGIN]   ...rendered game state (today's renderer output, minus inline blanks)... [STATE_END]
```

`DECISION_TYPE` ∈ {PRIORITY, DECLARE_ATTACKERS, DECLARE_BLOCKERS,
CHOOSE_TARGETS, CHOOSE_MODE, MAY, CHOOSE_X}. The decision type also rides
as an additive embedding on every encoder token (so the trunk reinterprets
the same state under the asked question).

**Deferred for follow-up** (not in v1; engine handles them today and the
existing inline-blank fallback or engine autotapper continues to apply
until they're added):

- **DAMAGE_ORDER** — multi-blocked damage assignment order. Out of scope
  for v1; revisit once combat is otherwise stable.
- **PAY_MANA** — mana payment. v1 relies on the engine's autotapper; we
  do not surface a `PAY_MANA` decision to the model. Revisit once we want
  the model to choose alternative costs / non-obvious source assignments.
- **ORDER_TRIGGERS, MULLIGAN, replacement choices**, and other long-tail
  pending kinds — added one at a time after the v1 surfaces stabilize.

The decision-spec section enumerates only the **action-subject entities**
plus per-decision scalar parameters, not relations and not entities the
action merely *references*. Spec entries are role-tagged references —
`[LEGAL_ATTACKER] OBJ_REF:K`, `[LEGAL_BLOCKER] OBJ_REF:K`,
`[LEGAL_TARGET] OBJ_REF:K` (or `[LEGAL_TARGET] PLAYER_REF:p`,
`[LEGAL_TARGET] STACK_REF:s`), `[LEGAL_ACTION:i] <action descriptor>`,
`[MAX_VALUE] <NUM:n>` (for MODE / X). `OBJ_REF:K` reuses today's
`<card-ref:K>` token-id space; the encoder positions that hold those
tokens (in state text *or* in the spec section) are pointer targets.

**CHOOSE_TARGETS lists valid targets explicitly.** Targets vary in kind
(permanent, player, spell on stack, ability on stack, card in graveyard,
…) and not every legal target has a stable state-text anchor — stack
objects in particular don't always have a battlefield `<card-ref:K>`.
The spec emits one role-tagged line per legal target so the pointer head
has a definite anchor regardless of where (or whether) the target appears
in state text:

```
[DECISION_TYPE: CHOOSE_TARGETS]
[FOR_ACTION:i]                    # which pending action this resolves targets for
[LEGAL_TARGET] OBJ_REF:K          # permanent target, anchored at state card-ref
[LEGAL_TARGET] PLAYER_REF:0       # player target
[LEGAL_TARGET] PLAYER_REF:1
[LEGAL_TARGET] STACK_REF:s        # spell/ability on stack
```

**MODE and X carry a max-value scalar.** Both are bounded numeric choices.
Spec emits `<max-value>N</max-value>` where `N` is rendered as plain
BPE-tokenized digits — the same convention state text already uses for
life totals (`<life>20</life>`) and library counts. No new number-token
family is added to the encoder vocab.

```
[DECISION_TYPE: CHOOSE_MODE]   <max-value>3</max-value>     # modes 0..2
[DECISION_TYPE: CHOOSE_X]      <max-value>12</max-value>    # X ∈ 0..12
```

For X this also tells the digit grammar where to clamp once the
partial-prefix exceeds the cap.

**Other reference targets are resolved from state, not duplicated in spec.**
When declaring blockers, the spec lists `LEGAL_BLOCKER` only — attackers
are already in state with an `ATTACKING` status tag on their battlefield
card-ref. When the decoder emits `<BLOCK> BLOCKER:K <ATTACKER> …`, the
pointer head at the `<ATTACKER>` step ranges over all `<card-ref:K>`
positions in state, with the legal-edge bitmap as the mask. Rule of thumb:
spec adds information the state can't supply; if state already tags the
role (ATTACKING / BLOCKING / on-stack-targeting-X), don't duplicate it.

| Decision | Spec entities + params | Resolved from state |
|---|---|---|
| DECLARE_ATTACKERS | `LEGAL_ATTACKER` | defenders (players, planeswalkers) |
| DECLARE_BLOCKERS | `LEGAL_BLOCKER` | attackers (state ATTACKING tag) |
| CHOOSE_TARGETS | `FOR_ACTION:i`, `LEGAL_TARGET` per legal target (incl. players, stack objects) | — |
| PRIORITY | `LEGAL_ACTION:i` | sources/targets within each action |
| MAY | (fixed grammar: `<YES>`/`<NO>`) | — |
| CHOOSE_MODE | `<max-value>N</max-value>` (BPE digits) | — |
| CHOOSE_X | `<max-value>N</max-value>` (BPE digits) | — |

**Length is small.** Combinatorial relations (block edges,
controller-equivalence, …) never enter the encoder token stream — they
live in non-token side-tensors that the mask callback consults at decode
time. Encoder length grows by ~2 tokens per legal subject (and per legal
target for CHOOSE_TARGETS); MAY / MODE / X spec is constant-size.

### Decoder

A small causal transformer (target `~4–6 layers, d_model = trunk d_model / 2`
or shared d, no embedding tied with trunk so we can grow vocabulary
independently). Two output heads:

- **Grammar vocab head** — softmax over a small fixed grammar vocabulary
  (`<DECLARE_ATTACKERS>`, `<ATTACK>`, `<BLOCK>`, `<DEFENDER>`, `<TARGET>`,
  `<YES>`, `<NO>`, `<NONE>`, `<END>`, …) plus the **existing tokenizer's
  number tokens** (reused as-is for MODE / X output — whatever BPE
  produces for "0", "1", "12", etc., no new digit family). Order of ~25
  grammar tokens.
- **Pointer head** — `pointer_logit_i = decoder_h_t · W · E_i` over the encoded
  positions of *referenceable* tokens (legal attackers, defenders, action
  candidates, mana sources). The pointer-eligible positions are recorded by
  the renderer/native emitter into a `pointer_targets` index table per
  decision; the engine mask further restricts them at each step.

At each generation step the decoder produces a *combined* distribution over
{vocab tokens} ∪ {pointer positions}. The engine mask zeroes out everything
that isn't a legal next token under the decision-spec grammar.

### Engine-supplied incremental mask

The renderer no longer pre-enumerates per-blank legal sets. Combinatorial
relations live in **non-token side-tensors** that travel alongside the
encoder input but never enter encoder attention — for blocks, an
`[N_blockers, N_attackers]` legal-edge bitmap (bool, ≤128² ≈ 2 KB/env);
for mana payment, a `[N_pips, N_sources]` produces-color bitmap; for
"no two targets share controller", per-target controller equivalence
classes; for damage-assignment order, nothing (the grammar shrinks the
legal set as positions are decoded). The engine exposes a
`DecisionGrammar` callable: given (decision_spec, side-tensors, prefix),
return a mask over the combined vocab+pointer slot set.

**Invariant.** The encoder receives the vocabulary of referenceable
entities; the decoder mask receives the relations between them. The native side
implements this as a small state machine per decision type — e.g.:

```
DECLARE_ATTACKERS:
  Ø                              → {<ATTACK>, <END>}
  ATTACK                         → pointers(legal attackers \ chosen)
  ATTACK OBJ_REF:K               → {<DEFENDER>}                 (or auto-skip if defender is forced)
  ATTACK OBJ_REF:K DEF           → pointers(legal defenders[K])
  ATTACK OBJ_REF:K DEF P         → {<ATTACK>, <END>}

DECLARE_BLOCKERS:
  Ø                              → {<BLOCK>, <END>}
  BLOCK                          → pointers(blockers \ chosen)
  BLOCK BLOCKER:K                → pointers(legal attackers blocker K can block)
  BLOCK BLOCKER:K ATTACKER:A     → {<BLOCK>, <END>}

CHOOSE_TARGETS (single target slot, repeat for multi-target):
  Ø                              → pointers(legal targets) [∪ {<NONE>} if optional]

CHOOSE_MODE / CHOOSE_X (max=N):
  emit the integer using the tokenizer's native number tokenization
  (same BPE the renderer uses for <life>, library counts, etc.). The mask
  at each step enumerates token-prefixes that extend to some integer in
  [0, N], plus <END> whenever the prefix-so-far is itself a complete
  in-range integer. For small N most integers are a single BPE token, so
  this is typically one emit step + <END>; longer integers are
  multi-token but the mask logic is identical.

MAY:
  Ø                              → {<YES>, <NO>}
  <YES>|<NO>                     → {<END>}
```

The grammar is per-decision-type and small; each is a few dozen lines on the
Go side. **Cross-subject constraints** (a defender blocks at most one
attacker, "no two targets share a controller", must-attack / can't-attack
restrictions) are expressible because the mask sees the prefix.

### Output examples

```
<DECLARE_ATTACKERS> <ATTACK> OBJ_REF:32 <DEFENDER> PLAYER_REF:1
                    <ATTACK> OBJ_REF:52 <DEFENDER> PLAYER_REF:1 <END>

<DECLARE_BLOCKERS>  <BLOCK>  BLOCKER:12 <ATTACKER> OBJ_REF:52
                    <BLOCK>  BLOCKER:18 <ATTACKER> OBJ_REF:52 <END>

<MAY> <YES> <END>

<PRIORITY> ACTION_REF:i <END>          # i is one of the enumerated legal engine actions

<CHOOSE_TARGETS> OBJ_REF:42 <END>     # one target slot
<CHOOSE_MODE>    {tok:"2"} <END>      # mode 2 of N (whatever BPE produces for "2"; N from <max-value>)
<CHOOSE_X>       {tok:"12"} <END>     # X = 12 (single-token if BPE has it; otherwise N tokens then <END>)
```

For PRIORITY we keep the shortcut: legal engine actions are enumerated in the
decision spec and the decoder emits a single pointer pick. Compositional
generation only kicks in for combat / multi-target cases where the joint
matters. Mana payment is handled by the engine's autotapper in v1; the
model never sees a `PAY_MANA` decision.

### Object reference encoding

Pointer targets use the **encoder positions** of the entities listed in the
decision spec. Because the spec section is part of the encoder input, every
referenceable object has a definite anchor. We don't add per-object tokens to
the decoder vocab. This bounds decoder vocab to grammar tokens only and lets
us add new entity types without retraining the decoder embeddings.

### History / recurrence

The existing `RecurrentTextPolicy` LSTM remains. Its hidden vector is injected
as a `[HIST]` token at the start of the encoder input (a `Linear(h_t)`
projection). The LSTM update is unchanged — `h_t = LSTM(z_t, h_{t-1})` where
`z_t` is the encoder's CLS pool — but every decoder cross-attention step now
sees history-aware context.

### Value head

Unchanged. Pools the encoder's CLS / global state vector; trains on the same
returns target.

## What this lets us delete

- **Inline-blank rendering inside the state text.** State text reverts to the
  pure description used by MLM pretraining (no `<choose-*>` tokens scattered
  through hand/battlefield/choices). Decisions live in the decision-spec
  section instead.
- **`InlineBlankPolicy` and the per-blank legal-id batch fields**
  (`blank_kind`, `blank_legal_ids`, `blank_legal_mask`, `blank_group`,
  `blank_group_kind`, `blank_option_index`). Replaced by `decision_spec_*` and
  decoder-target tensors.
- **Per-blank loss helpers** (`inline_blank_priority_loss`,
  `inline_blank_per_blank_loss`, etc.). Replaced by a single autoregressive
  CE over the decoder output.
- **Group-kind enum and CONSTRAINED placeholder.** The grammar mask subsumes
  all of them.
- **Blank ordinal stability invariants.** Replay stores the *output token
  sequence* (grammar tokens + pointer slot ids), which is invariant under
  encoder-position permutation as long as pointer ids are recorded.

## Native (mage-go) changes

The render-plan opcode pipeline is unused on the live path; all production
encoding goes through `directTokenEmitter` (`direct_token_emitter.go`) which
writes tokens directly into the packed output buffer. This plan stays on
the direct-token-assembly path — no opcodes, no render-plan stream, no
new bytecode dialect.

### 1. Drop inline-blank emission from the state-text path

Delete `directTokenEmitter.emitBlank` / `emitBlankLegal`, the `<choose-*>`
/ `<choices>` token spans, the `MagePackedBlankOutputs` struct, and the
`blankCollector` machinery in `blank_assembler.go`. State-text emission
becomes "what is true" only.

### 2. Add direct-token decision-spec emission

`directTokenEmitter` (or a sibling `decisionSpecEmitter` writing into a
separate output buffer) gains methods that walk the engine's pending
decision and write spec tokens directly:

```go
e.emitOpenSpec(decisionType)
switch decisionType {
case decDeclareAttackers:
    for _, a := range pending.LegalAttackers {
        e.emitLegalAttacker(a.UUIDIdx)        // writes [LEGAL_ATTACKER] <card-ref:K>
    }
case decDeclareBlockers:
    for _, b := range pending.LegalBlockers {
        e.emitLegalBlocker(b.UUIDIdx)
    }
    // legal-edge bitmap written into a separate side-tensor, not the token stream
case decChooseTargets:
    e.emitForAction(pending.ActionIdx)
    for _, t := range pending.LegalTargets {
        e.emitLegalTarget(t.Kind, t.Ref)      // permanent / player / stack
    }
case decChooseMode, decChooseX:
    e.emitMaxValue(pending.MaxValue)
case decPriority:
    for i, opt := range pending.Options {
        e.emitLegalAction(i, opt)              // pre-enumerated action descriptor
    }
case decMay:
    /* nothing — fixed grammar */
}
e.emitCloseSpec()
```

ABI extension: `MageEncodeTokensPacked` gains output buffers for spec
tokens, spec length, decision type, pointer-anchor table, and the side-
tensor relation buffers (legal-edge bitmap for blocks). All are direct
allocations — same shape as today's token / card-ref output buffers.

### 3. Grammar-mask callback (also direct, no opcodes)

Add `MageDecisionMaskNext(handle, prefix_token_ids[], prefix_len) ->
{vocab_mask[V_grammar], pointer_mask[N_anchors]}`. Implementation: per
decision type, a hand-coded state machine in Go that reads the engine's
combat / target legality predicates directly. The handle wraps the
engine's pending-decision state and is invalidated when the next snapshot
is produced. Batched across the env to keep cgo-crossing cost down (one
call returns masks for all live envs at the current decoder step).

### 4. Token-table changes

Inline-blank singletons (`<choose-play>`, `<use-ability>`, `<pass>`,
`<choose-target>`, `<choose-block>`, `<choose-may>`, `<choose-mode>`,
`<choose-x-digit>`, `<choose-mana-source>`, `<chosen>`, `<yes>`, `<no>`,
`<none>`, `<x-end>`, `<num:0..15>`) are removed from the encoder's token
tables. The encoder's vocab gains a small set of spec-section tags
(`<decision-type>`, `<legal-attacker>`, `<legal-blocker>`,
`<legal-target>`, `<legal-action>`, `<for-action>`, `<max-value>`,
`<player-ref:0/1>`, `<stack-ref:k>`, `<spec-open>`, `<spec-close>`).
`<card-ref:K>` and `<num:k>` stay (the latter is reused for `MAX_VALUE`).
The grammar vocabulary (`<DECLARE_ATTACKERS>`, `<ATTACK>`, `<BLOCK>`,
`<END>`, …) is owned by the **decoder** and does not flow through the
encoder's token tables.

## Python changes

### Pipeline

```
GameStateSnapshot
    │
    ▼  render.py (state-only mode)
state_text  +  decision_spec  (ds_tokens, ds_pointer_anchors, decision_type)
    │
    ▼  assembler / native_assembler
EncodedBatch:
    state_tokens     [B, T_state]
    spec_tokens      [B, T_spec]
    pointer_anchors  [B, N]   (encoder positions for legal entities)
    pointer_kinds    [B, N]
    decision_type    [B]
    │
    ▼  encoder (existing trunk)
E ∈ [B, T, D]            (T = T_state + T_spec + history + cls)
    │
    ▼  GrammarDecoder (NEW)
for t in 1..L:
    h_t = decoder(prev_tokens, cross_attn=E)
    vocab_logits   = vocab_head(h_t)             [V_grammar]
    pointer_logits = pointer_head(h_t, E_anchors)[N]
    mask = engine.next_token_mask(prefix)        # native callback
    sample → next token → append → if <END> stop
```

### New / changed modules

- `magic_ai/text_encoder/decoder.py` — **new** `GrammarDecoder` causal
  transformer with the two-headed (vocab, pointer) output and a
  `step(prev_tokens, encoded, anchor_index, mask)` API.
- `magic_ai/text_encoder/decision_spec.py` — **new** dataclasses /
  TypedDicts for `DecisionSpec`, `PointerAnchor`, decision-type enum, and
  Python-side mirrors of the native opcodes for the non-native fallback path.
- `magic_ai/text_encoder/grammar.py` — **new** Python mirror of the
  per-decision-type grammar state machines (used by tests and the
  Python-only assembler; the live path calls the native callback).
- `magic_ai/text_encoder/render.py` — strip inline-blank rendering;
  add `render_decision_spec(snapshot)` returning `(spec_tokens, anchors,
  decision_type)`. Pure deletion + small addition.
- `magic_ai/text_encoder/batch.py` — drop `blank_*` fields, add
  `spec_*` and `pointer_*` fields and `decision_type`.
- `magic_ai/text_encoder/policy.py` — `TextPolicy.forward` runs the
  encoder over (state, spec, history) and dispatches to the decoder.
  `encode_snapshots` signature changes to return decoder-ready context.
- `magic_ai/text_encoder/actor_critic.py` — live sampling becomes an
  autoregressive loop calling `GrammarDecoder.step` and the native mask
  callback. Replay scoring becomes a teacher-forced cross-entropy over
  stored output tokens (encoder is rerun, decoder is parallelized over
  the prefix).
- `magic_ai/text_encoder/replay_buffer.py` — store
  `output_tokens [L]`, `output_pointer_ids [L]`, `output_kinds [L]` per
  decision instead of per-blank arrays. Bump on-disk version.
- `magic_ai/text_encoder/training.py` — single
  `decoder_cross_entropy_loss` over output tokens; entropy bonus is the
  per-step mean entropy under the live mask.
- `magic_ai/text_encoder/native_assembler.py` — drop blank tensors;
  add decision-spec tensor I/O; add the mask-callback FFI.
- `magic_ai/text_encoder/inline_blanks.py` — **delete**.
- `magic_ai/text_encoder/recurrent.py` — `[HIST]` token injection
  replaces the current LSTM-augmented logits combination; structural.

### Token preparation script (`scripts/build_text_encoder_vocab.py`)

The vocab build script registers every entry in
`magic_ai.text_encoder.tokenizer.ALL_CUSTOM_TOKENS` as a single-id
`additional_special_token`. Two changes:

- **Remove inline-blank tokens.** Drop `INLINE_BLANK_TOKENS`
  (`CHOOSE_KIND_TOKENS` + `BLANK_ANSWER_TOKENS` + `NUM_TOKENS`) from
  `ALL_CUSTOM_TOKENS`. ~30 tokens leave the encoder vocabulary; vocab
  shrinks from 1004 back toward ~974.
- **Add decision-spec tag tokens.** Register `<decision-type>`,
  `<spec-open>`, `<spec-close>`, `<legal-attacker>`, `<legal-blocker>`,
  `<legal-target>`, `<legal-action>`, `<for-action>`, `<max-value>`,
  `</max-value>`, `<player-ref:0>`, `<player-ref:1>`, `<stack-ref:k>`
  for k ∈ [0, MAX_STACK_REFS), and one decision-type-name token per
  enum value (`<priority>`, `<declare-attackers>`, `<declare-blockers>`,
  `<choose-targets>`, `<may>`, `<choose-mode>`, `<choose-x>`).
  `<card-ref:K>` and the existing mana / status / structural tokens
  stay unchanged.

Numbers inside `<max-value>...</max-value>` use the tokenizer's existing
BPE digit tokenization — no new digit family is added. The persisted
tokenizer is rebuilt by re-running this script after the change; smoke
test is `tests/test_text_encoder_tokenizer.py`-style round-trip on the
new tag set.

The decoder's grammar vocabulary is **not** registered through this
script. It lives only on the decoder side
(`magic_ai/text_encoder/decoder.py`) as a small enum of stable ids; the
encoder never sees those token ids.

### Forge extractor (`scripts/extract_forge_choice_situations.py`)

The extractor today stores a pre-choice snapshot + observed-event
metadata so the loader can reconstruct inline-blank candidates at load
time. For the decoder pipeline:

- **Stop emitting inline-blank-shaped targets.** The extractor's role
  becomes "capture (snapshot, decision_type, observed_action)"; legal
  candidate reconstruction moves out of the loader entirely (see below).
- **Persist the engine pending decision.** Add a serialized
  `PendingState` (or the subset needed to rebuild the decision spec —
  decision type, legal-attacker / legal-blocker / legal-target lists,
  legal-edge bitmap for blocks, max-value for mode/X) alongside the
  snapshot. This is what the spec-rendering path consumes.
- **Encode the observed action as the decoder target token sequence.**
  Translate Forge's observed-event description (the regex-parsed
  attack/block/cast/may strings) into a sequence of grammar tokens +
  pointer-anchor ids — exactly the sequence the decoder is asked to
  reproduce. Store as `output_token_ids: list[int]` and
  `output_pointer_ids: list[int]` (anchor index per pointer step,
  -1 for non-pointer steps). Bump `FORMAT_VERSION`.
- **Drop the inline-blank-specific imports** (`MANA_TOKENS`,
  `NUM_TOKENS`, render-blank metadata) once the new path is in.

Sharded torch and JSONL outputs both keep working; only the per-record
schema changes.

### Policy + value pretraining (`magic_ai/text_encoder/policy_value_pretrain.py`)

Currently this module tokenizes the snapshot, conservatively
reconstructs an inline-blank legal set from visible state, and trains
`InlineBlankPolicy` on the per-blank target derived from the observed
choice (priority / attack / block / may / choose). For the decoder
pipeline:

- **Replace inline-blank target reconstruction with decision-spec
  rendering.** The loader calls `render_decision_spec(snapshot,
  pending)` (using the persisted pending state from the extractor) to
  produce the encoder's spec tokens, pointer anchors, decision type, and
  side-tensors.
- **Replace per-blank loss with autoregressive decoder CE.** Drop
  `inline_blank_priority_loss` / `inline_blank_per_blank_loss` /
  `inline_blank_priority_accuracy` / `inline_blank_per_blank_accuracy`
  imports. Use the new `decoder_cross_entropy_loss` over the stored
  output token sequence in teacher-forced mode (encoder runs once,
  decoder runs the full sequence in parallel). Per-step accuracy is
  reported as the headline policy metric, plus full-sequence exact
  match for combat decisions where the joint matters.
- **Value head is unchanged.** It still pools the encoder's CLS / state
  vector; the value-target construction modes (`terminal`, `gae`,
  `vtrace`) stay as-is.
- **`ForgeChoiceBatch` schema** loses `blank_*` fields and gains
  `spec_tokens`, `pointer_anchors`, `pointer_kinds`, `decision_type`,
  `output_token_ids`, `output_pointer_ids`, and the relation
  side-tensors when present.
- **Loss weighting.** `policy_loss_weight` and `value_loss_weight`
  config fields stay; `policy_loss_weight` now scales the decoder
  CE rather than the per-blank sum. Default weights unchanged so the
  optimizer schedule transfers.

The training entry point (`scripts/pretrain_text_encoder.py`) only
needs CLI updates to forward the new dataclass fields; no
architectural change.

### MLM pretraining synergy

State text under the new pipeline is exactly what we already pretrain on
(no inline blanks to mask around). MLM keeps working unchanged. As a
follow-up we can pretrain the decoder against engine-recorded action
sequences from existing trajectories — a clean autoregressive LM objective
on the action grammar, no blank-coupling needed.

## Migration plan

Each step is independently shippable. The gate after each step is **R-NaD
training stability + BC accuracy parity** vs. the prior step's checkpoint
on the same fixed trace set.

The migration runs in a fresh worktree (mirroring the inline-blanks
worktree pattern) so `main` stays on the inline-blank pipeline until the
new path is fully online.

1. **Decision-spec rendering, Python only, alongside existing inline path.**
   Add `render_decision_spec` and `DecisionSpec` types. Continue rendering
   inline blanks — the new spec output is *additional*, not yet consumed.
   Tests: spec round-trip on fixture snapshots.

2. **Grammar state machines (Python).** Implement `grammar.py` for every
   decision type. Tests: enumerate prefixes and assert legal-mask membership
   matches engine legality for fixture decisions.

3. **GrammarDecoder module + value path.** Add `decoder.py` with the
   two-headed output. Wire end-to-end on a single snapshot through
   `TextPolicy` under a `use_grammar_decoder=True` flag (analogous to the
   inline `use_inline_blanks` flag in the inline-blank migration). Inline
   path still default. Forward-pass smoke test.

4. **BC parity on PRIORITY only.** Pre-enumerate legal actions in the spec
   for priority decisions; train BC under both pipelines on a fixed trace
   set. Require accuracy parity within 0.5pp before extending. Holdout:
   priority decisions are the bulk of the dataset; this catches encoder /
   decoder integration bugs early without compositional complexity.

5. **Combat (attackers + blockers, single-block only).** The headline gain
   — it's what the inline approach can't express. BC parity gate on
   block-heavy traces; we expect *better* than parity here because the
   joint is correctly modeled. Damage-assignment order for multi-block is
   deferred (engine default order applies).

6. **Targets.** Multi-target and shrinking legal sets handled via
   grammar; legal-target list is enumerated explicitly in the spec
   (covers permanents, players, stack objects).

7. **MAY / MODE / X.** MAY is fixed-grammar; MODE / X both consume a
   `<max-value>` from the spec and emit the integer using the
   tokenizer's native number tokenization, with the cap enforced by the
   mask over BPE-token prefixes. (PAY_MANA continues to use the engine
   autotapper; not in v1.)

8. **Native direct-token decision-spec emission.** Until step 8 the spec
   is produced from the Python rendering of the snapshot; this step
   moves it onto the native path: new `directTokenEmitter` (or sibling)
   methods write spec tokens directly into a separate output buffer,
   `MageEncodeTokensPacked` gains spec / pointer-anchor / side-tensor
   outputs, `MagePackedBlankOutputs` is deleted, inline-blank
   `<choose-*>` token spans and the `emitBlank*` methods are removed.
   Replay buffer version bumped.

9. **Native mask callback.** Replace the Python mask state machine with
   `MageDecisionMaskNext` on the live path. Keep the Python mirror for
   tests.

10. **Forge extractor + policy/value pretraining migration.** Update
    `scripts/build_text_encoder_vocab.py` to drop inline-blank tokens
    and add spec-tag tokens; rebuild the persisted tokenizer. Update
    `scripts/extract_forge_choice_situations.py` to persist pending
    state and observed-action token sequences (bump FORMAT_VERSION) and
    re-extract the Forge corpus. Migrate
    `magic_ai/text_encoder/policy_value_pretrain.py` to decoder CE +
    decision-spec rendering. Gate on policy accuracy parity vs. the
    inline-blank pretraining run on the same Forge corpus.

11. **Decoder pretraining (optional, follow-up).** Train the decoder
    autoregressively on engine-recorded action sequences from prior
    R-NaD / PPO trajectories (in addition to the Forge corpus from step
    10). Encoder is the existing MLM-pretrained trunk.

12. **Delete inline-blank surfaces.** Remove `InlineBlankPolicy`,
    `inline_blanks.py`, blank batch / replay / native fields, the
    inline-blank token-table singletons, and the `use_grammar_decoder`
    flag. State-text rendering becomes blank-free; the decoder is the
    only policy path.

### Deferred (post-v1)

- **DAMAGE_ORDER** — multi-blocked damage-assignment order. Engine
  default order applies until this lands.
- **PAY_MANA** — model-driven mana payment. Engine autotapper applies
  until this lands.
- **ORDER_TRIGGERS, MULLIGAN, replacement effects, scry/surveil
  ordering**, and other long-tail pending kinds.

## Open questions

- **Decoder size.** Sizing tradeoff between adding decoder params vs.
  growing the trunk. Default proposal: 4 layers, d_model = trunk d_model,
  no parameter tying. Revisit at the BC-parity gates.
- **Mask callback overhead.** Each decoder step calls into native to get
  the next mask. For combat decisions of length ~20 tokens × batch-size
  256, that's 5120 cgo calls per env-step. We should batch the call
  (one mask request per step, fanned across the batch in Go) — needs an
  ABI shape that takes `[B, prefix_len]` and returns `[B, V+N]`. Cheap on
  the Go side; the cost is the cgo crossing.
- **Pointer-anchor capacity.** Worst-case combat with ≥30 attackers and
  ≥30 blockers could push `N_anchors` past today's blank cap of 64. Cap
  at 128 and add a deterministic truncation policy (engine prunes to
  top-K by stable id, mark the truncated suffix unreachable in the mask).
- **Replay storage size.** Output token sequences are slightly larger than
  per-blank arrays at typical decision lengths (~10 tokens vs. ~1–4
  blanks), but no `legal_ids` tensor is stored, so net replay size goes
  *down*. Validated on combat-heavy fixtures before step 8.
- **Beam search / temperature schedules.** Out of scope; greedy /
  multinomial is fine for R-NaD. Revisit if exploration proves
  insufficient.
- **Cross-attention to history.** Choosing between "history as a single
  prepended token" vs. "history as a separate cross-attention key set" —
  prepended token is simpler and likely sufficient given the LSTM already
  carries the heavy lifting; defer the alternative until profiling
  motivates it.

## Why this is worth the move

The inline-blanks plan was a clean unification of policy + MLM, but it
treats every decision as "fill K independent slots", and it embeds the
question into the state. That works for priority and simple per-blank
decisions; it does not work for any decision whose answer space *is* the
combinatorial structure of the question (combat, ordering, multi-pip
payment with constraints). Status doc gaps 1, 3, 4, 5, 9 are all the same
gap.

The decoder pipeline keeps everything we built — ModernBERT trunk, MLM
pretraining, card-ref pointers, history LSTM, native packed encoding —
and replaces only the head: a small autoregressive module with engine-
masked output. Compositional decisions become first-class; the grammar
masks make legality cheap to enforce; replay storage simplifies; and the
state text reverts to the form that MLM pretraining already understands.
