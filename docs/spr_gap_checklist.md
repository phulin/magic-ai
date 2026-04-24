# SPR Paper vs. Codebase — Gap Checklist

Reference: `spr_paper.tex` (Schwarzer et al., SPR). Repo: PPO-based MTG agent over structured/tokenized state. Partial SPR implementation lives in `magic_ai/model.py` and `magic_ai/ppo.py`.

## Already in place
- [x] Online + target encoder with EMA — `target_game_state_encoder`, `target_action_encoder`, `target_feature_projection`, `target_lstm` (`magic_ai/model.py:201–212`)
- [x] EMA update step — `magic_ai/ppo.py:138`
- [x] Configurable EMA decay — CLI `--spr-ema-decay` (default 0.99)
- [x] Action-conditioned predictor MLP — `spr_predictor` (`magic_ai/model.py:219–223`)
- [x] Episode boundary masking via `has_next` (`magic_ai/model.py:1259`, denom :1264)
- [x] SPR loss wired into PPO update — `--spr-coef` (default 0.1), applied at `magic_ai/ppo.py:115`
- [x] Target nets serialized in checkpoints

## Gaps to close (priority order)

### 1. K-step latent rollout (biggest functional gap)
- [x] Replace single `next_idx` lookup with iterative unroll of `spr_transition` for K steps
- [x] Chain `next_idx` lookups to gather `z_target_{t+1..t+K}`
- [x] Per-step `has_next` mask, accumulated multiplicatively across k
- [x] Add `--spr-k` CLI flag (default 5)

### 2. Projection heads (BYOL-style g_o / g_m + predictor q)
- [x] Add online projection `spr_g_online` after encoder, before predictor
- [x] Add EMA target projection `spr_g_target` mirroring `spr_g_online`
- [x] Add `spr_q` predictor on top of online projection
- [x] Include `spr_g_target` in EMA update list

### 3. Cosine similarity loss
- [x] Replace MSE-on-L2-normalized vectors with `-F.cosine_similarity(...)` per step
- [x] Remove hand-rolled normalization in favor of cosine

### 4. Richer action conditioning
- [x] Step-0 action keeps full embedding (trace + opt/tgt mean + may bit) via cached forward
- [x] Chained-step actions fed at every k of the K-step rollout (paper-style)
- [ ] (Future) avoid simplification at k>=1 by caching option/target vectors for chained step indices

### 5. Loss weight + tuning
- [ ] Sweep `--spr-coef` upward toward paper's λ=2 once integration is validated
- [ ] Re-tune EMA decay (paper uses τ=0 with aug, τ=0.99 without)

### 6. Minor architectural niceties
- [ ] Normalize activations to [0,1] at encoder / transition outputs (paper detail)
- [ ] Add `τ=0` branch behind a flag for experimentation

## Out of scope / architectural mismatches
- [ ] **PPO vs DQN/Rainbow**: SPR's data-efficiency argument relies on off-policy replay reuse. Switching algorithm families is a rewrite, not a tweak. Skip unless explicitly redesigning.
- [ ] **Data augmentation (DrQ shifts + color jitter)**: not meaningful for structured MTG state. Open design question whether a structured analogue (feature dropout, card-order shuffles) is worth exploring.

## Validation plan
- [ ] Unit test: K-step rollout matches single-step when K=1
- [ ] Confirm target encoder params do not receive gradients
- [ ] A/B eval: baseline vs. K=5 + projections + cosine, on existing eval harness
- [ ] Profile training step cost (paper notes ~modest overhead)
