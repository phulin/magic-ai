# Inline-Blank Decisions for the Text Encoder

## Summary

Replace the current option-head / target-head policy machinery with **inline
blank tokens** that the encoder fills in via the (already-existing) tied MLM
head. Each pending decision is rendered as a `<choose-*>` token at its
natural position in the state text; at decision time the model produces a
distribution over a per-blank legal-token vocabulary, and an action sample
is the concatenation of the chosen tokens at every blank.

This unifies the policy and pretraining objectives (both are MLM over the
same closed vocabulary), makes compound decisions like block assignments
first-class, and removes the separate gather pools and dedicated heads for
options/targets.

## Current implementation snapshot

The migration is currently inline-only on the Python policy side: legacy
text-policy option/target heads and batch fields have been removed, and live
native text rollout uses `MageEncodeTokensPacked` blank outputs. The packed
text-token ABI no longer exposes option/target scratch buffers; remaining
option/target arrays belong to the separate slot-encoder/native action path.

Implemented blank surfaces:

- `CROSS_BLANK`: `<choose-play>`, `<use-ability>`, and `<pass>` priority
  anchors scored with singleton `<chosen>` legal vocabularies and one softmax
  across the anchor positions.
- `PER_BLANK`: `<choose-target>`, `<choose-may>`, `<choose-mode>`,
  `<choose-x-digit>` for bounded `number` choices,
  `<choose-mana-source>` for `mana_color`, binary attacker-declaration
  blanks, and `<choose-block>` with `<none>` plus legal attacker
  `<card-ref:K>` ids.

Known fidelity gaps before this is a complete representation of all game
choices: damage assignment order is not wired; targets currently cover visible
card refs, player ids, and selected-column fallback for standalone object
choices, but not every target class; arbitrary multi-target
cardinality/constraints are not fully modeled; X is a bounded single blank
rather than a digit sequence; mana-color blanks choose colors rather than
source objects; several engine pending kinds still need dedicated or generic
blank encodings. The detailed live status and gap list lives in
`docs/text_encoder_inline_blanks_status.md`.

## Motivation

The current architecture (`magic_ai/text_encoder/model.py`) uses
position-anchored gather pools to extract per-option and per-target vectors,
then runs them through dedicated `PolicyHead` / `TargetHead` MLPs. This
forces every decision into a flat list of options, which is awkward for:

- **Block assignments.** A block step is one decision with one blank per
  attacker; the current option-list shape requires either pre-enumerating
  all assignment combinations (combinatorial blow-up) or a sequence of
  per-attacker option lists with no spatial coupling.
- **Multi-target spells.** "Choose any number of target creatures" or
  "no two of which share a controller" don't factor as one option-pick.
- **Priority / mode / may / X choices.** Each is a slightly different head
  shape today; they'd all be the same shape under inline blanks.

A second motivation is that MLM pretraining (`mlm.py`) already trains the
tied LM head over the closed vocabulary that includes `<card-ref:K>`. If
the policy head *is* the LM head, MLM pretraining transfers to policy
quality directly with no head re-init.

## Concept

At render time, every pending decision is emitted as a blank token:

- **`<choose-target>`** — spatially co-located with the spell/ability
  whose target slot it fills. The same kind token appears once per slot;
  the *slot index* is the blank's ordinal within its group, not a token
  variant. Legal vocabulary = card-refs of legal targets (+ `<none>` if
  the slot is optional).
- **`<choose-block>`** — emitted next to each *defender* (not each
  attacker). Per-defender framing is necessary because two defenders may
  block the same attacker (multi-block). Legal vocabulary = card-refs of
  legal attackers this defender could block + `<none>`. A separate
  *damage-assignment-order* sub-decision is rendered per multi-blocked
  attacker as an ordered sequence of `<choose-damage-order>` blanks
  whose vocabulary is the still-unordered blockers (so it's effectively
  a permutation decoded one position at a time).
- **`<choose-mode>`, `<choose-may>`, `<choose-x>`** — emitted in a
  dedicated trailing `<choices>…</choices>` zone, one per pending choice.
  Mode vocabulary = `<num:0..N-1>` (small N, bounded by mode count);
  may = `<yes>/<no>`; X is decomposed into a sequence of digit blanks
  (`<choose-x-digit>` with vocab `<num:0..9>` + a terminator) so the
  vocabulary stays bounded regardless of X.
- **Priority is a single decision, not a per-card vector.** Every
  candidate priority action is rendered as a tagged anchor at its
  natural position: cards in hand get a trailing `<choose-play>`,
  activatable abilities get a trailing `<use-ability>`, and a single
  `<pass>` blank lives in the choices zone. **All priority anchors in a
  snapshot share one `blank_group` and the policy is one softmax across
  those anchor positions** — i.e., the "logit" at each anchor is the
  score for selecting that specific priority action; we softmax across
  the group rather than treating each anchor as an independent
  yes/no. This avoids the illegal multi-yes / no-action samples that
  the per-anchor factorization would admit. Mana-payment sub-decisions
  triggered by playing a card are deferred to subsequent snapshots
  (each pay-step is its own decision rendered in `<choices>` with one
  `<choose-mana-source>` blank per cost pip; no in-snapshot recursion).

The encoder runs once over the rendered text. At each blank position we
score the legal-token subset directly against gathered embedding rows
(see "Model" below) — not via a full-vocab projection followed by a
gather, which would be wasteful at K × V and unnecessary since we know
the legal candidates per blank up front.

**Group-level policy shapes.** A `blank_group` is the unit of policy
factorization, not the individual blank. Two group shapes cover all current
decision kinds:

- **Per-blank softmax** (most decisions): each blank in the group is its
  own categorical over its legal-id list; group log-prob = sum of
  per-blank log-probs. Used for targets, modes, mays, X-digits,
  block-choice (per defender), damage-order positions, mana sources.
- **Cross-blank softmax** (priority, "choose one of these anchors"):
  the group has one categorical whose support is *all anchor positions*
  in the group; the logit at position k comes from a fixed scoring
  token (e.g. `<chosen>`) at that blank's position via the LM head.
  Exactly one anchor in the group is the chosen action.
`blank_group_kind ∈ {PER_BLANK, CROSS_BLANK}` rides
alongside `blank_group` in the batch.

Group kind describes scoring topology only. Game semantics live in the typed
blank token (`<choose-block>`, `<choose-target>`, `<choose-may>`, etc.) plus
the per-blank legal-id set. Complete Magic-fidelity for compound choices whose
legal sets shrink as earlier blanks are filled will need richer decision
metadata or an engine-backed legality filter, but not a third group kind.

## Data flow

```
GameStateSnapshot
    │
    ▼  render.py
text + anchors (card-refs, blank positions, blank kinds, blank legal-id sets)
    │
    ▼  assembler.py / native_assembler.py
TextEncodedBatch:
    tokens           [B, T]
    blank_positions  [B, K]
    blank_kind       [B, K]
    blank_legal_ids  [B, K, V_max]   (CSR or padded)
    blank_legal_mask [B, K, V_max]
    blank_group      [B, K]
    │
    ▼  model.py
hidden = trunk(tokens)                            # [B, T, D]
blank_h = gather(hidden, blank_positions)         # [B, K, D]
blank_h = mlm decoder pre-projection(blank_h)      # [B, K, D]
legal_emb = token_embedding(blank_legal_ids)       # [B, K, V_max, D]
logits = dot(legal_emb, blank_h)                   # [B, K, V_max]
logits = logits.masked_fill(~blank_legal_mask, -inf)
value = value_head(global_pool(hidden))
    │
    ▼  sampler
per-group sample → action token-ids per blank
    │
    ▼  engine adapter
mapped back to engine action
```

## Vocabulary additions (`token_tables.py`, `tokenizer.py`)

New custom tokens (stable ids; native side mirrors via
`native_token_tables.py`):

Each kind is a **single token**, not a family parameterized by index.
Slot/attacker/defender/mode-index identity comes from the blank's
ordinal within its `blank_group`, established by render order. This
keeps the custom-vocab small and avoids re-allocating ids if MAX_ATK
or MAX_TARGETS changes.

| Token                  | Use                                           |
|------------------------|-----------------------------------------------|
| `<choose-target>`      | Spell-target-slot blank                       |
| `<choose-block>`       | Per-defender block-choice blank               |
| `<choose-damage-order>`| Per-position damage-order blank               |
| `<choose-mode>`        | Mode-choice blank                             |
| `<choose-may>`         | Optional-trigger blank                        |
| `<choose-x-digit>`     | One digit of an X-cost answer                 |
| `<choose-mana-source>` | Per-pip mana-payment blank                    |
| `<choose-play>`        | Priority anchor on a card in hand             |
| `<use-ability>`        | Priority anchor on an activatable ability     |
| `<pass>`               | Priority anchor for "pass priority"           |
| `<chosen>`             | Scoring token for cross-blank softmax groups  |
| `<yes>`, `<no>`        | Boolean answer vocab                          |
| `<none>`               | Optional-choice "decline" answer              |
| `<x-end>`              | Terminator for X-digit sequence               |
| `<num:k>` (k=0..15)    | Small-integer answer vocab (mode index, digit)|

`<card-ref:K>` already exists and is the answer vocab for target/blocker
blanks. Choose two upper bounds carefully:

- `MAX_BLANKS_PER_SNAPSHOT` — caps `K` for batch shaping.
- `MAX_LEGAL_PER_BLANK` — caps `V_max`. For block/target slots this is the
  number of legal candidates; sized to e.g. 64 with a fallback truncation
  policy (engine pre-prunes to top-K by stable order if exceeded).

## Render layer (`render.py`)

Add a render mode flag (`use_inline_blanks: bool`) to keep the legacy
option-list output available during migration.

For each decision-kind the renderer:

1. Emits the `<choose-*>` token at the natural position.
2. Records a `BlankAnchor(blank_index, kind, char_start, char_end,
   legal_token_ids: tuple[int, ...], group_id: int)`.
3. `RenderedSnapshot` gains `blank_anchors: list[BlankAnchor]`; the
   existing `option_anchors` / `target_anchors` are dropped in the
   inline-blank branch.

`group_id` semantics:

- `<choose-block>` blanks are `PER_BLANK` groups over `<none>` plus legal
  attacker card refs. Cross-blank combat constraints are not represented by a
  third group kind; they require richer decision metadata or an engine-backed
  legality filter.
- Each multi-blocked attacker's `<choose-damage-order>` sequence is its
  own `PER_BLANK` group; positions are decoded left-to-right (legal set
  shrinks as blockers are committed).
- All target slots for one spell share a `PER_BLANK` group; target constraints
  such as "no two of which share a controller" need richer decision metadata
  or an engine-backed legality filter.
- All priority anchors in a snapshot (every `<choose-play>`,
  `<use-ability>`, and the single `<pass>`) share **one `CROSS_BLANK`
  group**. Exactly one is selected.
- Mode/may/X-digit/mana-source each form their own `PER_BLANK` groups.

**Stable blank numbering.** Blank ordinals must be deterministic across
re-renders of the same logical snapshot — replay-buffer entries store
chosen-token-id-per-blank by ordinal, and importance sampling for
off-policy correction relies on the live forward pass producing the
same ordinal layout. Render order is fixed by (zone, controller,
engine-stable object id) and does not depend on Python dict iteration
order; this is asserted in tests via re-render parity on a fixture
trace.

## Assembler & batch (`assembler.py`, `batch.py`, `native_assembler.py`)

`TextEncodedBatch` field changes:

- Drop: `option_positions`, `option_mask`, `target_positions`, `target_mask`.
- Add: `blank_positions [B,K] int32`, `blank_kind [B,K] int32`,
  `blank_legal_ids [B,K,V_max] int32`, `blank_legal_mask [B,K,V_max] bool`,
  `blank_group [B,K] int32`.

Native-assembler ABI (`text_encoder_render_plan_abi.md`): add two opcodes:

- `EMIT_BLANK(kind_id)` — writes the kind token at the cursor and records
  its position into the blank table.
- `EMIT_BLANK_LEGAL(token_id)` — appends a token id to the current blank's
  legal list. Followed by `END_BLANK_LEGAL`.

Triton packed-append kernels in `replay_triton.py` need their column list
updated; layout is preserved (still ragged-by-snapshot, packed across the
batch), so kernel logic is the same.

## Model (`model.py`)

Remove `PolicyHead` and `TargetHead`. Add:

```python
class InlineBlankPolicy(nn.Module):
    """Score legal-token candidates at each blank by gathering embedding rows
    and computing a single inner product per (blank, candidate). Avoids the
    [B, K, V] full-vocab projection."""

    def __init__(
        self,
        embed: nn.Embedding,                  # tied input embeddings
        mlm_decoder_norm: nn.Module,          # MLM head's pre-projection norm
        mlm_decoder_proj: nn.Module,          # MLM head's pre-projection dense
        kind_temperature: nn.Embedding,       # [num_blank_kinds, 1]
    ) -> None:
        super().__init__()
        self.embed = embed
        self.norm = mlm_decoder_norm
        self.proj = mlm_decoder_proj
        self.kind_temperature = kind_temperature

    def forward(
        self,
        hidden: Tensor,                  # [B, T, D]
        blank_positions: Tensor,         # [B, K]
        blank_kind: Tensor,              # [B, K]
        blank_legal_ids: Tensor,         # [B, K, V_max]
        blank_legal_mask: Tensor,        # [B, K, V_max]
    ) -> Tensor:                         # [B, K, V_max] logits
        blank_h = _gather_at(hidden, blank_positions)            # [B, K, D]
        blank_h = self.proj(self.norm(blank_h))                  # [B, K, D]
        legal_emb = self.embed(blank_legal_ids)                  # [B, K, V_max, D]
        logits = (legal_emb * blank_h.unsqueeze(-2)).sum(-1)     # [B, K, V_max]
        # Per-kind learned temperature (init 1.0); kind embedding is a scalar.
        temp = self.kind_temperature(blank_kind).squeeze(-1).unsqueeze(-1)
        logits = logits * temp
        return logits.masked_fill(~blank_legal_mask, float("-inf"))
```

Notes:

- The MLM head in `mlm.py` already has a pre-projection norm + dense; we
  share those parameters rather than duplicate. The output projection is
  the tied embedding matmul, which we replace with a per-blank legal-id
  gather + inner product.
- For `CROSS_BLANK` priority groups the "candidate vocabulary" at every
  blank is the singleton `<chosen>`, and the per-anchor logit is one
  scalar per position; we then softmax across the K positions in the
  group (handled in the sampler).
- Memory budget: replacing `[B, K, V]` (~ 50k cols) with
  `[B, K, V_max, D]` (V_max ≤ 64, D ≤ 768) is ~10× smaller in fp16 at
  typical shapes and avoids the K × V GEMM entirely.

The value head stays as-is and pools the global state vector (current
global pool is fine).

## Legality enforcement

The output is guaranteed-legal by construction at two layers; nothing
post-hoc clamps, retries, or falls back to a default action.

**Layer 1 — per-blank candidate masking (hard).** Every blank carries a
`blank_legal_ids: [V_max]` list and a `blank_legal_mask: [V_max]` bool
mask, both produced by the engine at render time. The model's logits
tensor is shaped `[B, K, V_max]` (legal-only support, not the full
vocabulary), and the mask is applied via `masked_fill(~mask, -inf)`
before softmax. Two consequences:

- **Sampling cannot produce a non-legal token.** Even with numerical
  underflow, `-inf` logits stay zero-probability after softmax;
  multinomial sampling never selects them.
- **Argmax under temperature → 0 or top-k truncation is also safe**, because
  the candidate set itself excludes illegal tokens — the only way a
  non-legal token enters the candidate list is an engine bug, which is
  caught by the assertion in the next bullet.

The engine adapter additionally asserts `chosen_id ∈ legal_ids` in
debug builds; in release this is a no-op since the mask makes it
unreachable. Empty legal sets (no legal action for a blank) are an
engine bug — render must not emit a blank with zero legal candidates;
a `RenderError` is raised at render time if it tries.

**Layer 2 — optional engine legality projection.**
Per-blank masking only enforces *per-blank* legality. Group-level
constraints (e.g. "a defender blocks at most one attacker", "no two
targets share a controller") should be enforced by richer decision metadata
or an engine-backed legality filter. The current basic representation keeps
block choices as independent `PER_BLANK` categoricals over legal attacker ids.

If group constraints can never be satisfied (e.g. impossible
combinatorial state), the engine must not have rendered the decision
in the first place; render-time validation catches this.

**Layer 3 — engine adapter (defense in depth).** When mapping chosen
tokens back to engine actions, the adapter validates the assembled
action against the engine's legality predicate before submitting. A
mismatch raises rather than silently falling back; any mismatch
indicates a render/engine drift bug and we'd rather hear about it
loudly than play an unintended action.

**What is *not* masked.** The MLM head's pre-projection (the per-blank
hidden state) is unrestricted; only the candidate-scoring step is
masked. This means the trunk can attend to anything in the rendered
text — including illegal-target descriptions — and learn from full
context, while still being prevented from emitting them. Pretraining
under MLM uses the unrestricted full-vocab head; the policy path
swaps in the legal-only candidate gather.

**Cross-blank softmax legality (priority).** The candidate set *is* the
set of anchor positions in the group, all of which are by construction
legal priority actions (render only emits an anchor for a card the
engine reports as playable, an ability it reports as activatable, and
the always-legal `<pass>`). So for `CROSS_BLANK` groups, masking
reduces to selecting among the anchor positions; no per-anchor legality
mask is needed.

## Action sampling

Group blanks by `blank_group`. For each group, dispatch on
`blank_group_kind`:

- **`PER_BLANK`** (mode, may, X-digits, single-target, blocker choice,
  mana-source, damage-order positions): sample each blank's softmax independently.
  `group_logp = sum(per_blank_logp)`. Damage-order positions are
  decoded left-to-right with the legal set shrinking after each pick;
  the position-i softmax is over remaining unassigned blockers.
- **`CROSS_BLANK`** (priority): one categorical over the K anchor
  positions in the group. Logits are the per-anchor scalar from the
  `<chosen>` candidate (a single sigmoid-style logit per anchor); we
  softmax across positions. `group_logp = log_softmax(scores)[selected]`.
  This is the primary fix to the per-anchor yes/no formulation: a single
  priority decision can't be factored as independent per-card yeses.

The factor model is still an approximation to the true joint. If BC
accuracy on block steps regresses materially vs. the legacy head,
escalate to **iterative fill**: pick the highest-confidence blank,
replace its `<choose-block>` token in the input with the chosen
`<card-ref:K>`, re-encode, repeat. N forward passes per combat step;
only worth it if parallel-fill measurably underperforms. A cheaper
middle ground is **one-step refinement**: do parallel-fill, then a
single re-encode with all chosen tokens substituted, and re-score the
joint under that conditional — costs 2 forward passes total and
captures most of the inter-blank dependency.

## Training (`actor_critic.py`, `training.py`)

**BC distillation.** Target = engine-chosen token id at each blank.
Loss per snapshot:

```
loss = sum over blanks k of
    cross_entropy(logits[b, k], target_id_in_legal[b, k])
```

with reduction = `mean` over blanks (not snapshots) so dense-blank
snapshots don't dominate.

**RL (R-NaD / PPO).** The action vector is the concatenation of chosen
token ids per blank. The behavior policy log-prob is `sum_k logp_k`
(under parallel fill; under iterative fill, log-probs are taken under
the *conditional* distribution at each step). Advantage scales the joint
log-prob exactly as today; entropy bonus is summed per-blank but
weighted per-group-size to avoid over-rewarding "many priority blanks
all of which obviously pick `<no>`" snapshots — concretely:

```
entropy_term = mean over groups g of mean over blanks in g of H_k
```

**Value targets** are unchanged.

## Replay buffer (`replay_buffer.py`)

Swap `option_*` / `target_*` columns for `blank_*` columns listed in the
batch section. Dataclass is frozen; bump the on-disk version tag and
add a one-shot upgrade path that drops legacy buffers (acceptable since
buffers are short-lived training state).

## Rollout (`rollout.py`, policy facade `policy.py`)

`encode_snapshots` now returns blank metadata instead of option metadata.
The Python rollout's action-decoding path (`_choose_action`) walks blanks
group-by-group, samples, and assembles the engine action via a
blank-kind dispatch:

- Priority `CROSS_BLANK` group: the single selected anchor's kind +
  ordinal determines the action — `<choose-play>` ⇒ play the
  corresponding hand card; `<use-ability>` ⇒ activate that ability
  instance; `<pass>` ⇒ pass. No tie-break needed; the cross-blank
  softmax produces exactly one selection.
- `<choose-block>` group: assemble `defender_id → attacker_id` dict
  from the per-defender choices (`<none>` ⇒ defender doesn't block).
- `<choose-damage-order>` group per multi-blocked attacker: ordered
  list of blocker ids in the chosen permutation.
- `<choose-target>` group: target list for the spell, one per slot in
  render order.
- `<choose-mode>` group: selected mode index.
- `<choose-may>` group: yes/no.
- `<choose-x-digit>` group: digits decoded into an integer up to the
  `<x-end>` terminator.
- `<choose-mana-source>` group: per-pip source assignment.

## Migration plan

Each step is independently shippable; after each, parity vs. legacy on
the BC validation set is the gate.

1. **Token table.** Add the new tokens to `token_tables.py` and the
   tokenizer custom-tokens list. Mirror to native via
   `native_token_tables.py`. Tests: round-trip encode/decode.
2. **Render priority blanks (flag-gated).** Implement
   `<choose-play>`/`<use-ability>`/`<pass>` rendering in `render.py`
   behind `use_inline_blanks=True`. Emit `BlankAnchor`s.
3. **Batch + native assembler plumbing.** Add the `blank_*` fields to
   `TextEncodedBatch`/`PackedTextBatch`; add the `EMIT_BLANK*` opcodes
   to the Go assembler and Python emitter. Triton kernel column update.
4. **`InlineBlankPolicy` + value head wiring.** Keep legacy heads in
   place; add the new path under the same flag. End-to-end forward
   pass on a single snapshot.
5. **BC parity on priority-only.** Train a BC head on a fixed trace set
   under both rendering modes. Require ≤ 0.5pp accuracy regression
   before extending.
6. **Combat blocks.** Add `<choose-blocker:a>` rendering, per-attacker
   group, parallel-fill + greedy projection sampler. BC parity gate;
   if it fails, add iterative-fill behind a sub-flag.
7. **Targets, modes, mays, X.** Add the remaining blank kinds. BC parity
   gate.
8. **Delete legacy paths.** Remove `PolicyHead`, `TargetHead`,
   `option_*` / `target_*` fields, and the legacy renderer branch.
   Remove the flag.

## Open questions

- **Single-pass vs iterative for blocks.** Resolved per-step at the
  parity gate; default parallel-fill, escalate to one-step refinement
  before full iterative fill.
- **`MAX_LEGAL_PER_BLANK`.** Block decisions in long games can have
  many legal blockers; survey real games for the 99.9th percentile.
  If a hard cap is needed, prune by stable engine order with a
  `<truncated>` sentinel that masks to non-zero probability so the
  policy can learn to back off in saturated states.
- **`<chosen>` scoring token vs. dedicated head for cross-blank.** The
  current sketch uses a `<chosen>` token's embedding inner-producted
  with the blank's hidden state as the priority logit. An alternative
  is a tiny dedicated linear (1 scalar per anchor); the LM-head reuse
  is cleaner but the dedicated head is more expressive. Decide at the
  priority-step parity gate.
- **Mana payment ordering.** Pay-cost decisions are deferred to
  follow-on snapshots in this plan. If that produces too many tiny
  decisions per turn, revisit by emitting a `<choose-mana-source>`
  group inline at play time.
- **Pretraining synergy.** Consider adding action-fill to the MLM
  pretraining mixture once the rendering supports it: mask `<choose-*>`
  blanks alongside oracle-text spans during pretraining on replayed
  trajectories. Out of scope for this plan; flagged for follow-up.

## Changes from initial sketch (review pass)

This plan was revised after a review pass. Material changes vs. the
initial sketch:

1. **Priority is no longer per-card yes/no.** The original framing had
   one `<yes>/<no>` blank per hand card + ability and a tie-break rule
   in rollout. That admits illegal multi-yes / no-action samples and
   the tie-break is a hack. Replaced with a single `CROSS_BLANK` group
   spanning all priority anchors.
2. **Block framing inverted.** Per-attacker blanks can't express
   multi-block. Now per-defender, with a separate damage-assignment-
   order sub-decision per multi-blocked attacker.
3. **`blank_kind` is a flat enum, not a parameterized family.** No more
   `<choose-blocker:0>`, `<choose-blocker:1>`, …; identity comes from
   ordinal within the group. Keeps custom-vocab small and stable.
4. **LM-head compute path changed.** Score against gathered legal
   embeddings, not via a full-vocab projection plus gather. Removes a
   K × V GEMM and avoids holding `[B, K, V]` in memory.
5. **Damage-assignment order, mana payment, X-cost** added to the
   render surface (originally missing).
6. **Stable blank-numbering invariant** called out explicitly with a
   render-parity test, since replay-buffer entries reference choices
   by ordinal.
7. **One-step refinement** added as a middle option between
   parallel-fill and full iterative-fill.
