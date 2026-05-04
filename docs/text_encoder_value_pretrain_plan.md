# Value-Head Pretrain (Stage 3a) — Detailed Plan

## 0. TL;DR

After the MLM pretrain phase (`run_mlm_pretrain` in `scripts/train.py:1415`)
finishes and before R-NaD/PPO begins, run a **second supervised phase** that
trains the encoder + value head to predict the perspective player's
win/loss/draw outcome from a single decision-point snapshot. Save a
`--post-value-checkpoint` so an RL run can resume from {pretrained encoder,
pretrained value head, random policy/target heads}. This is §7 stage 3 of
`docs/text_encoder_plan.md` ("Supervised value pretrain") promoted to a
checkpointed phase that mirrors how MLM is wired today.

## 1. Goals & non-goals

**Goals**
- Warm the value head and continue tuning the encoder against a real,
  game-grounded scalar before RL begins.
- Reuse the existing `*.jsonl.gz` recorder corpus under `data/games/` —
  each game already has a `META` record with `winnerId` and per-priority
  decision-point snapshots, which is everything we need.
- Land the artifact as a checkpoint alongside the existing
  `--post-mlm-checkpoint` so the three phases can be run independently
  and resumed independently.

**Non-goals (v1)**
- Training the policy / target heads (that's stage 4 BC distillation,
  separate plan).
- Training across recurrence (LSTM history). Value pretrain treats every
  decision point as i.i.d. — same simplification `TextEncoderTrainer.value_step`
  already makes (`magic_ai/text_encoder/training.py:193`).
- Bootstrapping value targets from later states (TD / GAE). The label is
  the **terminal outcome of the game** the snapshot came from, signed by
  perspective. TD bootstrapping is what the RL phase is for.

## 2. Data: snapshot → outcome

Source: the same `data/games/*.jsonl.gz` corpus
`scripts/jsonl_games_to_bin.py` already consumes for MLM. Each file is one
recorded self-play game.

Per-game extraction:
1. Read the `META` line. Extract `winnerId` (string player UUID, or `null`
   for a draw — `winnerName == "Game is a draw"`).
2. For every `EVENT` record where `_is_decision(rec)` returns true
   (`scripts/jsonl_games_to_bin.py:598` predicate), read `snapshot.priorityPlayerId`.
3. Compute the perspective-relative label:
   ```
   label =  +1.0  if winnerId == priorityPlayerId
            -1.0  if winnerId is not null and != priorityPlayerId
             0.0  if winnerId is null  (draw)
   ```
   This matches the convention already in the value head (signed scalar in
   `[-1, +1]`); the MSE loss in `value_loss` (`training.py:38`) handles draws
   directly.
4. Render the snapshot to tokens with the existing `_DirectEmitter` path so
   the bytes are identical to what the live encoder sees during rollouts.

### File format

Per-game layout in a new directory `data/games_value_bin/`:

- `<gameId>.bin` — flat `uint16` tokens, one decision-point span wrapped in
  `<bos>...<eos>`, concatenated back-to-back. **Byte-identical to
  `data/games_bin/`** so `BinTokenDataset` keeps working as-is and the same
  artifact can be used for MLM without re-conversion.
- `<gameId>.json` — small sidecar with game-level metadata and a span
  index. Schema:
  ```json
  {
    "game_id": "a76e5950-...",
    "winner_id": "78518d52-..." | null,
    "is_draw": false,
    "players": [
      {"id": "78518d52-...", "name": "PlayerA"},
      {"id": "...",          "name": "PlayerB"}
    ],
    "spans": [
      {"offset": 0,     "length": 612, "perspective_id": "78518d52-...", "label": 1.0},
      {"offset": 612,   "length": 588, "perspective_id": "...",          "label": -1.0},
      ...
    ]
  }
  ```
  `offset` and `length` are in `uint16` token units within `<gameId>.bin`.
  `label` is the perspective-signed scalar (`+1.0` / `-1.0` / `0.0`) so
  readers don't have to recompute it from `winner_id` + `perspective_id`,
  but both are present so future label schemes (e.g. discounted-outcome,
  margin-of-life) can be derived without rebuilding the bin.

Why JSON over a parallel binary `.idx`: the index is small (a few KB per
game, a few MB for the whole corpus), human-inspectable for debugging, and
trivially extensible without bumping a binary version field. Loading cost
is negligible compared to the token mmap.

A new dataset class `ValueLabeledBinDataset` in
`magic_ai/text_encoder/value_pretrain.py` mmaps each `.bin`, loads each
`.json` once at startup, and yields `(tokens [seq_len], label scalar)`
pairs by indexing into the mmap with `offset`/`length` from the JSON.
Span lengths in the recorder already exceed `--text-max-tokens`; we
right-truncate to `seq_len` and keep the `<bos>` prefix (consistent with
how `BinTokenDataset.iter_epoch` samples fixed spans).

### Sampling

- **Random decision points, balanced perspective.** Sample a game uniformly,
  then a decision point uniformly within the game. No per-game caps in v1
  — long games are naturally over-represented but they also have more
  diverse states, which is what we want.
- **Drop draws? No.** Keep them; MSE pulls predictions toward 0 for draws,
  which is the correct behavior. If draws dominate the corpus (the head
  collapses to ~0) we revisit; the smoke run below catches this.
- **Label imbalance check.** Print the `(win, loss, draw)` count at startup
  so a corpus that's e.g. 90% draws fails loud rather than producing a
  collapsed value head.

## 3. Trainer

New module `magic_ai/text_encoder/value_pretrain.py`. Mirrors `mlm.py` in
shape so review-by-diff is cheap.

```
class ValuePretrainConfig:
    data_dir: Path                  # data/games_value_bin/
    seq_len: int
    batch_size: int
    pad_token_id: int

class ValueLabeledBinDataset:
    # mmaps *.bin + *.idx, samples (tokens, label) pairs.

class ValuePretrainHead(nn.Module):
    # Reuses the same shape as the production value head:
    #   MLP(d_model -> d_ff -> 1), tanh on output to keep predictions in
    #   [-1, +1]. Initialized fresh — there is no HF analogue to warm-init
    #   from.

class ValuePretrainTrainer:
    def __init__(self, encoder, head, cfg, lr, ...): ...
    def step(self, token_ids, labels) -> dict[str, float]: ...
```

Key loss/optim choices:

- **Loss**: MSE between `head(state_vec)` and the scalar label, where
  `state_vec` is the hidden state at the `<state>` (or `<bos>`) anchor —
  same gather as the production value head in `model.py`. Reuse
  `value_loss(...)` from `training.py:38`.
- **Optimizer**: AdamW, lr `1e-4` default (lower than MLM's `2e-4` because
  the encoder is already past random init), `weight_decay=0.01`,
  `betas=(0.9, 0.95)`. Grad clip `1.0`.
- **Single optimizer over encoder + value head.** No frozen encoder option
  in v1; the encoder is still small enough that finetuning end-to-end is
  cheap and we want the encoder to adapt its `<state>` representation to
  the value task.
- **Metrics logged per step**: MSE, label-sign accuracy
  (`sign(pred) == sign(label)` ignoring draws), mean prediction, mean
  absolute prediction (collapse detector), mean label.
- **Eval split**: hold out 5% of games (deterministic by `hash(gameId) % 20 == 0`)
  and report eval MSE + sign accuracy every N steps. Catches overfitting
  on small corpora.

## 4. Wiring into `train.py`

Three edits, each parallel to an existing MLM edit:

1. **Argparse** (next to the `--pretrain-mlm-*` block at
   `scripts/train.py:1935`):
   - `--pretrain-value-dir PATH` — directory with `*.bin` + `*.idx`. If
     unset, the value pretrain phase is skipped.
   - `--pretrain-value-epochs INT` (default 1)
   - `--pretrain-value-batch-size INT` (default 128)
   - `--pretrain-value-seq-len INT` (default same as MLM)
   - `--pretrain-value-lr FLOAT` (default 1e-4)
   - `--pretrain-value-grad-clip FLOAT` (default 1.0)
   - `--pretrain-value-log-every INT` (default 50)
   - `--pretrain-value-amp` (bool flag, mirrors `--pretrain-mlm-amp`)
   - `--post-value-checkpoint PATH` — like `--post-mlm-checkpoint`.

2. **`run_value_pretrain(args, text_backend, device)`** function next to
   `run_mlm_pretrain` (`scripts/train.py:1415`). It:
   - Pulls `encoder = text_backend.policy.policy.text_policy.encoder` and
     `value_head = text_backend.policy.policy.text_policy.value_head`
     (so the same head used in RL is the one we pretrain — no
     reinitialization on phase boundary).
   - Builds `ValuePretrainConfig` + dataset + trainer.
   - Iterates epochs × batches with the same AMP / wandb / log-every
     scaffolding as `run_mlm_pretrain`.
   - Logs under `pretrain_value/*` so wandb panels stay separate from MLM.

3. **Phase orchestration** in `main()` (around `scripts/train.py:1601`):

   ```
   resumed_post_mlm   = _is_post_mlm_checkpoint(checkpoint_cpu)
   resumed_post_value = _is_post_value_checkpoint(checkpoint_cpu)

   run_mlm_now   = args.pretrain_mlm_dir   is not None and not (resumed_post_mlm or resumed_post_value)
   run_value_now = args.pretrain_value_dir is not None and not resumed_post_value

   if run_mlm_now:   run_mlm_pretrain(...)
   if run_value_now: run_value_pretrain(...)

   if run_mlm_now or run_value_now or resumed_post_mlm or resumed_post_value:
       # reset RL optimizer (value-head moments from supervised phase do
       # not transfer to the RL Adam betas — RnaD uses b1=0)
       optimizer = ...rebuild...
       if run_value_now and args.post_value_checkpoint is not None:
           save_checkpoint(..., post_value=True)
       attach_rl_lr_warmup(...)
   ```

   `_is_post_value_checkpoint` is a one-liner reading
   `metadata["post_value"]`, parallel to `_is_post_mlm_checkpoint` at
   `scripts/train.py:1036`. Add `post_value: bool = False` to
   `save_checkpoint` (`scripts/train.py:4822`) and stamp the metadata.

   The skip semantics: a `post_value` checkpoint implies "MLM and value
   pretrain are both done", so resuming from it skips both phases. A
   `post_mlm` checkpoint implies "MLM done, value not done"; resuming it
   *and* passing `--pretrain-value-dir` runs only the value phase, then
   optionally writes `--post-value-checkpoint`. This mirrors the existing
   resume-from-mlm flow and lets us bisect failures across phases.

## 5. Corpus builder

`scripts/jsonl_games_to_bin.py` is extended with a `--with-value-labels`
flag. When set, `convert_one` additionally:

1. Captures `winnerId` and `players` from the `META` record.
2. Tracks the per-decision `(offset, length, perspective_id, label)` as
   spans are emitted.
3. Writes a `<gameId>.json` sidecar alongside the `<gameId>.bin` carrying
   the schema in §2.

The bin half stays byte-identical regardless of the flag, so a single
artifact directory can drive both pretraining phases. Recommended layout:
`data/games_value_bin/` (run with `--with-value-labels`) for value
pretrain; existing `data/games_bin/` (no flag) keeps working for MLM.

Round-trip test: decode one `(span, label)` pair, retokenize with the
live tokenizer, and assert the same byte stream the live encoder
produces for that snapshot.

## 6. Test plan

1. **Unit: `ValueLabeledBinDataset`**. Synthesize a `*.bin` + `*.idx` pair
   in-memory, assert random sampling and epoch iteration return the right
   `(tokens, label)` tuples and respect `seq_len` truncation.
2. **Unit: `ValuePretrainTrainer.step`**. Synthetic batch where the label
   is `+1` whenever a marker token id is present and `-1` otherwise.
   Assert MSE drops ≥ 50% over 200 steps on a small encoder. Same shape as
   `scripts/pretrain_text_encoder.py`'s smoke (`scripts/pretrain_text_encoder.py:259`),
   reusable as a CLI smoke.
3. **Integration: `run_value_pretrain` end-to-end**. Convert ~10 recorded
   games into `data/games_value_bin/`, run for 100 steps with batch=4,
   assert (a) no NaNs, (b) eval sign-accuracy > 0.5 on a held-out fragment
   (i.e. better than always-predict-zero on a non-pure-draw corpus).
4. **Phase ordering**: with `--pretrain-mlm-dir A --pretrain-value-dir B
   --post-mlm-checkpoint M --post-value-checkpoint V`, resuming from `V`
   skips both phases; resuming from `M` runs only the value phase. Add
   to `tests/test_train_phase_ordering.py` (new) parallel to whatever
   tests already cover the MLM resume path.

## 7. Eval (before declaring v1 done)

- **Sign accuracy on a held-out shard** of the recorder corpus —
  perspective-correct win/loss prediction. Target: ≥ 0.65 on a corpus
  drawn from a non-uniform-skill self-play run. Random would be ~0.5
  after draws are dropped.
- **Calibration**: 10-bin reliability curve of `pred vs empirical
  win-rate`. Saved as a wandb plot at end of phase. Expect a roughly
  monotone curve; flat curves indicate value-head collapse.
- **Stage-3 ablation in §7 of the parent plan**: scratch-encoder vs
  HF-warm-started encoder, ± value pretrain. Same eval set, four runs.
  This is the actual decision point for whether the phase ships on by
  default.

## 8. Risks & open questions

- **Distribution shift between recorded corpus and RL self-play.** The
  recorder corpus we have today is from earlier checkpoints / earlier
  agents; the value head we pretrain is calibrated to *those* policies'
  win rates from each state, not the current policy's. RL fine-tuning
  re-calibrates, but the warm start may push the value head into a basin
  that's actively wrong for the new policy. Mitigation: rebuild
  `data/games_value_bin/` from the latest opponent-pool snapshot before
  each long RL run rather than treating the bin dir as static.
- **Draw-heavy corpora.** Early self-play often draws (timeouts, infinite
  loops). MSE on a 90%-draw corpus produces a value head that always
  predicts ~0. The startup label-imbalance print (§2) makes this loud.
  If it bites we add a `--pretrain-value-balance` flag that subsamples
  the majority class.
- **Recorded snapshots vs live snapshots.** Anything the live engine emits
  that the recorder dropped (or vice versa) silently changes the encoder's
  input distribution between phases. The round-trip test in §5 is the
  guard. We should also add a CI smoke that converts one game with the
  current recorder + tokenizer and bytewise-compares against a frozen
  golden artifact, refreshed deliberately on schema changes.
- **Value head shape divergence.** If the production value head changes
  (e.g. switches to a categorical distributional head) the pretrain phase
  has to follow. Keep `ValuePretrainHead` as a *reference* to
  `text_policy.value_head` rather than its own module — pretraining
  trains the same parameters that RL will use, so a head refactor
  automatically applies to both phases.

## 9. PR slicing

1. **`scripts/jsonl_games_to_bin.py --with-value-labels`** — extend the
   existing converter with the optional sidecar emit + round-trip test.
   No model/training changes. Lands first; produces the paired
   `.bin`/`.json` artifact under `data/games_value_bin/`.
2. **`magic_ai/text_encoder/value_pretrain.py`** — `ValueLabeledBinDataset`,
   `ValuePretrainTrainer`, unit tests. Standalone.
3. **`scripts/train.py` wiring** — argparse, `run_value_pretrain`,
   `_is_post_value_checkpoint`, `save_checkpoint(post_value=True)`,
   phase-ordering edits. Smallest diff; depends on (1) and (2).
4. **Eval script + wandb panel** — sign-accuracy + calibration plot,
   runnable standalone against any `value_pretrain` checkpoint. Useful
   for the §7 ablation in `text_encoder_plan.md`.

Each PR is independently mergeable and the value pretrain phase is
opt-in (`--pretrain-value-dir` unset = no behavior change), so RL runs
that don't want it stay byte-identical to today.
