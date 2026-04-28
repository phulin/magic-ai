# Transformer State/Action Encoder — Detailed Plan

## 0. TL;DR

Replace the frozen-Qwen-embedding `GameStateEncoder` and the scalar-feature `ActionOptionsEncoder` with a single bidirectional transformer that ingests **rendered game-state + action text** as a token stream. Tokenization uses the **ModernBERT tokenizer** (~50k BPE, byte-level fallback, case-sensitive) plus custom tokens for mana symbols, status flags, zone delimiters, and intra-snapshot card references. **The trunk is trained from scratch** at ~14M params (ModernBERT-shaped: RoPE, GeGLU, alternating local/global attention). Finetuning from `ModernBERT-base` is a documented upgrade path (§8), not the default. Per-card / per-option / per-target representations come from gathering hidden states at known anchor positions. Ship behind a `policy.encoder = "text" | "slot"` flag; keep the slot encoder as baseline until parity is shown on win rate and unseen-card generalization.

## 1. Goals & non-goals

**Goals**
- Replace `GameStateEncoder` (`magic_ai/game_state.py:163`) and `ActionOptionsEncoder` (`magic_ai/actions.py:239`) with a transformer fed raw oracle text plus structured game-state markup.
- Trainable end-to-end; no offline 4096-dim Qwen vectors in the hot path.
- Generalize to unseen cards (the open problem from `project_embedding_experiments.md`).

**Non-goals (v1)**
- Replacing the native C FFI rollout encoder (`magic_ai/native_encoder.py`). v1 keeps it for the slot path; the text path runs in Python rollouts.
- Pretraining a language model from scratch on Magic text. We either start from random init or finetune a small off-the-shelf encoder (§8).

## 2. Tokenizer

**Base**: ModernBERT tokenizer via HF `AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")`. Tokenizer choice is independent of weight init — we use this tokenizer with a scratch-initialized trunk. Rationale:
- **~50k BPE vocab**: small embedding matrix (~50k × 384 ≈ 19M params would dominate a 14M trunk if the vocab were 128k+; 50k keeps the embedding under 20M and the total model honest).
- **Byte-level fallback**: anything Scryfall throws at us tokenizes without `<unk>` (em-dashes, accented names, exotic punctuation).
- **2024-era English corpus**: meaningfully better compression on prose than GPT-2 / BERT-era tokenizers.
- **Forward-compatible**: if we later want to warm-start the trunk from `ModernBERT-base` (§8), the tokenizer already matches — no vocab migration needed.
- Note: ModernBERT BPE is case-sensitive. Renderer must use canonical Scryfall casing — see §9.

**Custom added tokens** (each maps to a single id, never split):

*Structural delimiters*
- `<bos>`, `<eos>`, `<pad>`
- `<state>` … `</state>` — the whole snapshot
- `<self>`, `<opp>` — player scope
- Zone openers + closers: `<hand></hand>`, `<battlefield></battlefield>`, `<graveyard></graveyard>`, `<exile></exile>`, `<library></library>`, `<stack></stack>`, `<command></command>`. Closers are kept (not strictly required given fixed zone order) because they make attention-mask debugging easier and keep empty zones distinguishable.
- `<card>` … `</card>` — one card record
- `<actions>` … `</actions>` — legal action list
- `<option>` … `</option>` — one action option
- `<target>` … `</target>` — one target inside an option
- `<sep>` — generic field separator inside a card record

*Status flags* (bare tokens inside a `<card>` block)
- `<tapped>`, `<untapped>`, `<sick>` (summoning sick), `<attacking>`, `<blocking>`, `<monstrous>`, `<flipped>`, `<facedown>`
- `<+1+1>`, `<-1-1>`, `<counter>` followed by an integer literal
- `<attached-to>` followed by a `<card-ref:K>`

*Mana / cost symbols* — one token per Scryfall symbol:
- `{W} {U} {B} {R} {G} {C} {S} {X} {0} {1} {2} … {20}`
- Hybrid: `{W/U} {U/B} {B/R} {R/G} {G/W} {W/B} {U/R} {B/G} {R/W} {G/U}`
- Phyrexian: `{W/P} {U/P} {B/P} {R/P} {G/P}`
- 2-hybrid: `{2/W} {2/U} {2/B} {2/R} {2/G}`
- `{T}` (tap), `{Q}` (untap), `{E}` (energy)
- Loyalty: `[+1] [-1] [+2] …` up to ±20

*Card identity*
- `<card-ref:K>` for K = 0…N-1 within the current snapshot. Lets oracle text and target lists *point* at the same card object across the prompt. Positional within snapshot, not global card IDs (a fresh game reuses 0..N-1). N capped at 64.

*Numerics*
- 0–20 already exist as BPE tokens. For life / mana / counters we render decimal digits and accept BPE behavior. If magnitude perception turns out to be a problem in eval, add `<num>` followed by single-digit tokens.

**Implementation**
- New module `magic_ai/text_encoder/tokenizer.py`. Loads ModernBERT's tokenizer, calls `add_special_tokens(...)` to register the custom tokens below, then `save_pretrained("data/text_encoder_tokenizer/")` so train and inference agree byte-for-byte. We pin the upstream revision in code so a ModernBERT update can't silently shift the base vocab.
- Provide `tokenize_snapshot(snapshot, actions) -> TokenizedExample` and the inverse `render_snapshot(snapshot, actions) -> str` (exposed for debugging — humans should be able to read the prompt).

## 3. Snapshot → text rendering

A single deterministic renderer (`magic_ai/text_encoder/render.py`) converts `GameStateSnapshot` + legal `ActionOptions` into a token stream. Example layout:

```
<bos><state>
  <self> life=20 mana={W}{W}{U} </self>
  <opp>  life=17 mana=          </opp>
  <self><battlefield>
    <card-ref:0><card> Llanowar Elves <sep> Creature — Elf Druid <sep>
      1/1 <sep> {T}: Add {G}. <sep> <tapped> <sick> </card>
    <card-ref:1><card> Forest <sep> Basic Land — Forest <sep>
      {T}: Add {G}. <sep> <untapped> </card>
  </battlefield></self>
  <opp><battlefield>
    <card-ref:2><card> Grizzly Bears <sep> Creature — Bear <sep> 2/2 <sep> </card>
  </battlefield></opp>
  <self><hand>
    <card-ref:3><card> Lightning Bolt <sep> Instant <sep>
      {R} <sep> Lightning Bolt deals 3 damage to any target. </card>
  </hand></self>
  <self><graveyard>...</graveyard></self>
  <opp><graveyard>...</graveyard></opp>
  <stack> ... </stack>
<actions>
  <option> cast <card-ref:3> cost {R} <target><card-ref:2></target> </option>
  <option> activate <card-ref:1> cost {T} </option>
  <option> pass </option>
  <option> attack with <card-ref:0> </option>
</actions>
</state><eos>
```

Rules:
- Oracle text is rendered verbatim from Scryfall (cached by `scripts/build_card_embeddings.py`), with mana symbols left as `{X}` so the tokenizer maps them to our added tokens.
- Card names rendered once per `<card>` block. Inside oracle text the card's own name is left alone (oracle text already self-references by name); the binding to `<card-ref:K>` is anchored by emitting the ref token immediately before `<card>`.
- Empty zones rendered as `<hand></hand>` so zone identity is always present — "no cards in hand" is a distinguishable state.
- Determinism: stable card ordering inside each zone (battlefield by entry order, hand by draw order). Token positions are a pure function of state so RnaD replay is reproducible.

**Length budget**: a busy mid-game (12 permanents + 7 hand + graveyards) can exceed 1500 tokens. Plan for a 2048-token context window. Truncate library to count only (it is hidden anyway), truncate opponent's hand to count only.

**Duplicate-card optimization**: render long oracle text once per *unique* card; subsequent copies emit only `<card-ref:K><card> Name <sep> ...stats... </card>` and rely on attention to match by name. Quantify the savings in stage 2 of §7 before relying on it.

## 4. Model architecture

`magic_ai/text_encoder/model.py`:

- `TextStateEncoder(nn.Module)`
  - `tok_emb`: `nn.Embedding(vocab_size, d_model)`. All rows N(0, 0.02) on the scratch default. On the finetune path (§8) BPE rows come from `ModernBERT-base.embeddings.tok_embeddings`; custom-token rows are mean-init from related vocab tokens.
  - **Default trunk (scratch)**: ModernBERT-shaped — N=6 layers, d_model=384, 6 heads, FF=1536, GeGLU, RoPE, alternating local/global attention. ~14M trunk params (plus ~19M embedding matrix). Bidirectional, no causal mask.
  - **Finetune upgrade path (§8)**: load `ModernBERT-base` weights (22L / d=768 / ~150M). Same tokenizer means no vocab migration; only the custom-token rows need to be added to the embedding matrix.
  - Output: `[B, T, d_model]` contextual hidden states.

- **Pooling heads** (consume the encoder output):
  - **Per-card vectors**: gather hidden states at the `<card-ref:K>` token positions. Replaces `embed_slot_vectors` (`game_state.py:359`).
  - **Per-option vectors**: gather hidden states at each `<option>` token. Replaces option scalars in `actions.py`.
  - **Per-target vectors**: gather hidden states at each `<target>` token (or at the `<card-ref:K>` inside it — pick one in implementation, document the choice). Targets and the cards they point at share information through the encoder forward pass — a free win the scalar encoder cannot do.
  - **Global state vector**: hidden state at `<state>` (or `<bos>`). Feeds the value head.

- **Action scoring** (parallels `action_query` at `model.py:2143`):
  - `policy_logit = MLP([option_vec, state_vec]) → scalar` per option.
  - `target_logit = MLP([target_vec, option_vec, state_vec]) → scalar` per target slot, gated by which option is being scored.

- **Value head**: `MLP(state_vec) → scalar` plus the QRE/log-π tensors RnaD needs (same shapes as today).

## 5. Wiring into the existing system

1. **New batch type**: `TextEncodedBatch` in `magic_ai/text_encoder/batch.py` carrying `token_ids [B, T]`, `attention_mask [B, T]`, `card_ref_positions [B, max_cards]`, `option_positions [B, max_opts]`, `target_positions [B, max_opts, max_targets]`, plus existing reward / legal-mask / step bookkeeping.

2. **Python tokenize-on-CPU path in the rollout worker.** Slow (~few ms/step) but unblocks training and lets us validate before any native port. Native port is a follow-up, not v1.

3. **Model integration** (`magic_ai/model.py`):
   - Behind config flag `policy.encoder = "text" | "slot"`. Keep the slot encoder available for A/B comparison.
   - `_compute_forward` (`model.py:1423`) branches on encoder type. Text path skips `embed_slot_vectors` and `ActionOptionsEncoder`; both replaced by the gather ops in §4.
   - `evaluate_replay_batch` (`model.py:1050`) needs the same branch. Store token ids in the rollout buffer (preferred over re-rendering): deterministic and avoids the renderer cost on every update. ~4 KB/step at 2048 ctx × int16; budget sanity-checked in §9.

4. **Rollout buffer**: extend `RolloutBuffer.ingest_parsed_batch` (`model.py:750`) to accept a `TextEncodedBatch` variant. Storage is a ragged int32 tensor of token ids per step.

5. **Training loop**: no changes to RnaD math. The batched-trajectory loss from commit `fd60fb7` works as-is on the new policy/value outputs.

## 6. Action representation

The current scalar encoder (`actions.py:239–587`) hand-engineers ability index, mana cost split, target type, etc. The transformer plan replaces these by *rendering the same information as text*:

- `cast Lightning Bolt cost {R} target <card-ref:2>`
- `activate <card-ref:1> ability 0 cost {T}` — ability index is still emitted as a literal small int; it disambiguates cards with multiple activated abilities.
- `attack with <card-ref:0>` / `block <card-ref:0> with <card-ref:5>`
- `pass`, `mulligan`, `keep`, `choose mode 2`, `pay {2}{R}`, `pay {1} life`

Side-channel scalars: for values the renderer cannot represent precisely (e.g. "X = 5" in `Fireball`), we both render `X=5` *and* pass X as a small float through a tiny MLP that adds into `option_vec`. This is the only structured-feature concession; the existing `_clip_norm` machinery (`actions.py:656`) goes away.

## 7. Curriculum & training plan

1. **Tokenizer build & freeze.** `scripts/build_text_encoder_vocab.py`. Outputs `data/text_encoder_vocab.json` and a round-trip test asserting every Scryfall card in `data/card_oracle_embeddings.json` tokenizes with no `<unk>`.

2. **Renderer test suite** (`tests/test_text_render.py`): for ~50 hand-picked snapshots, assert renderer is deterministic, length is within budget, and every card in state has exactly one `<card-ref:K>`.

3. **Supervised value pretrain (recommended)**: use the existing self-play replay buffer to train encoder + value head only on `(snapshot → outcome)` regression for ~1 epoch. Gets the transformer past random-init before RnaD signal kicks in.

4. **Behavior cloning warm-start**: distill the current slot-encoder policy into the text encoder for a few thousand steps. Metric: action-distribution KL between the two. Avoids an RnaD cold-start where the transformer is uniform over thousands of legal actions.

5. **RnaD finetune**: switch to standard rnad loss. Expect ~3–5× wall-clock per step vs current encoder (transformer forward dominates). Mitigations: `torch.compile` the encoder forward (we already compile decision distribution per commit `f24cf26`), bf16, FlashAttention.

6. **Eval**:
   - **Unseen-card generalization** (the headline metric): hold out N cards from training decks, swap them in at eval time, compare win rate vs slot-encoder baseline.
   - Reuse `scripts/eval_card_embeddings.py` triplet/synonym/cluster harness on per-card hidden states pooled at `<card-ref:K>`.
   - Action-prediction top-1 accuracy on a held-out replay set vs slot baseline.

## 8. Pretrained-LM option (decision point)

Both paths use the ModernBERT tokenizer. The choice is whether to also load pretrained weights.

- **Scratch 14M trunk (default)**: N=6 / d=384 ModernBERT-shaped trunk, random init. Faster step time, faster iteration loop, no upstream dependency. Big enough to fit Magic's domain given the supervised + BC warm-start in §7.
- **Finetune ModernBERT-base (upgrade)**: load pretrained 22L/d=768 weights, extend the token embedding matrix with our custom tokens (mean-init from related vocab tokens — e.g. `{W}` from the mean of `white` + `mana`; `<tapped>` from `tapped`), finetune end-to-end. ~10× the params and meaningfully slower per step; pursued only if the scratch model plateaus on unseen-card generalization.
- **Middle option**: initialize the 14M trunk from the **first 6 layers** of `ModernBERT-base` (same shape, partial pretrained init). Cheap to try; revisit if the scratch/finetune ablation suggests pretrained weights help but the full 22L model is too slow.

Ablation at stage 3 of §7: same supervised warm-start run, scratch vs finetune-base vs first-6-layers, compare value loss after 1 epoch and per-step wall-clock.

## 9. Risks & open questions

- **Sequence length blow-up** on cluttered boards. Mitigations: per-zone token budget (e.g. opponent graveyard truncated to last 6 cards + `…+N more`), duplicate-card body suppression (§3). Renderer reports length stats during stage 2 of §7 — set the cap from data, not guess.
- **Replay storage cost**. 2048 tokens × int16 × ~1M steps ≈ 4 GB RAM. Acceptable; fall back to re-rendering if it bites.
- **Native encoder divergence**: until the Rust/C native path also emits text, sample collection is Python-bound and rollouts will be slower. Pragmatic v1: "Python rollouts only for the text path; slot-encoder native rollouts kept for benchmarking."
- **Card-ref binding**: the model has to learn that `<card-ref:0>` mentioned in `<actions>` refers to the `<card>` block tagged `<card-ref:0>` earlier. RoPE + bidirectional attention should handle this; sanity-check with an attention-map probe on a small example before stage 5.
- **Hard cases for the renderer**: X-spells, modal choices, split cards, MDFCs, adventure cards, sagas, planeswalkers (loyalty + abilities), tokens with no oracle ID. Enumerate in the renderer test suite before declaring v1 done; each has historically broken state encoders.
- **Tokenizer drift**: ModernBERT BPE is case-sensitive. Card names like "Llanowar Elves" tokenize differently from "llanowar elves". Renderer must use canonical Scryfall casing. Pin the upstream tokenizer revision so a HF update can't shift base-vocab ids out from under a checkpoint.
- **Context length**: ModernBERT supports 8192 tokens natively, so the 2048-token budget from §3 leaves a comfortable margin and we can lift it later without re-init.

## 10. Concrete first PRs

1. `scripts/build_text_encoder_vocab.py` + `magic_ai/text_encoder/tokenizer.py` + round-trip test. (No model code yet.) Locks the token set before anything depends on it.
2. `magic_ai/text_encoder/render.py` + golden-file tests against ~50 snapshots from existing replays.
3. `magic_ai/text_encoder/model.py` (encoder + pooling) + a unit test that runs forward on a tokenized batch and produces correctly-shaped per-card / per-option / per-target tensors.
4. `magic_ai/text_encoder/batch.py` + `RolloutBuffer` integration behind `policy.encoder="text"` flag.
5. Supervised value-pretrain script + first comparative eval vs slot encoder.
6. RnaD integration + first self-play run.

Each PR is independently mergeable and revertable; the slot encoder stays the default until stage 5 shows the text encoder is at least at parity on win rate.
