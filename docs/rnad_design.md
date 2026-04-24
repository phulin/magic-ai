# R-NaD / DeepNash Training for magic-ai — Design

## Goal
Add `--trainer rnad` alongside the existing PPO trainer. Share the rollout, buffer, model, and opponent-pool infrastructure, but replace the policy-optimization core with Regularized Nash Dynamics: reward transformation + NeuRD policy gradient + two-player v-trace + an outer-loop fixed-point iteration that updates the regularization policy.

## Why R-NaD for MTG
MTG is two-player, zero-sum, imperfect-information — R-NaD's target regime. PPO's clipped-ratio + GAE converges to *some* self-play fixed point but is not an equilibrium-finding algorithm; R-NaD has last-iterate convergence guarantees toward an ε-Nash equilibrium and is designed for bluff/deception play, which MTG exhibits (mulligan, hidden hand, bluff attacks). The existing LSTM policy, snapshot opponent pool, GPU rollout buffer, and episode-level buffer layout all carry over.

## Paper → codebase mapping

| Paper concept | Landing spot |
|---|---|
| Reward transform $r^i - \eta\log(\pi/\pi_{\text{reg}})$ (§161) | New `rnad.transform_rewards()` applied per-step before v-trace |
| Two-player v-trace $\Upsilon$ (§170–182) | New `rnad.two_player_vtrace()` replacing `ppo.gae_returns` |
| NeuRD policy loss (§188) | New `rnad.neurd_loss()` replacing clipped-ratio PPO loss |
| Fixed-point outer iteration $\pi_{m,\text{reg}} \leftarrow \pi_{m,\text{fix}}$ | Outer loop dispatched from `scripts/train_ppo.py`; reuses snapshot infra |
| Smooth $\alpha_n$ interpolation between consecutive reg policies (§164) | Keep prev + current reg in memory; blend inside reward-transform |
| Polyak target network $\gamma_{\text{avg}}$ (§191) | EMA copy of `PPOPolicy`; SPR code already has the EMA pattern |
| Full-episode batching | `RolloutBuffer` already stores full episodes; sample by episode |
| Softmax thresholding fine-tune / post-process (§197, §279) | New `rnad.threshold_discretize()` applied in fine-tune + eval |

## Decisions on the open questions

1. **Hierarchical vs joint NeuRD — decision: per-head independent.** The policy has four heads (option, target, may, none/blocker). Treat each as a parallel NeuRD instance sharing the joint reward transform. This matches the paper's own four-head architecture (§52) and avoids Cartesian enumeration of legal joint actions. If evaluation shows the approximation underperforms, revisit joint enumeration — do not pay that complexity up front.

2. **Reg policy in opponent pool — decision: self-play against current target only.** Matches the paper exactly. Keep the existing opponent-pool snapshots for PPO; R-NaD ignores them during training rollouts and only uses the pool for TrueSkill evaluation vs PPO snapshots. This keeps the theoretical convergence argument intact and removes a tuning knob.

3. **Δ_m and M schedule — decision: Δ_m=25k, M=20, η=0.2 as starting defaults.** Scaled down ~100× from the paper's 7.21M-step budget to match single-GPU capacity. Expose as CLI flags; tune after first end-to-end run.

4. **Draw-penalty interaction — decision: set draw_penalty=0 for R-NaD.** With η=0.2 the entropy-regularization term will swamp a ±0.1 draw signal and makes the reward no longer cleanly zero-sum. Keep draws as zero reward under R-NaD; re-enable only if empirical draw rate is pathologically high.

5. **Behavior-policy logging — decision: recompute logits at update time.** Don't bloat the buffer with cached per-action logits. `evaluate_replay_batch` already re-forwards stored observations with LSTM state; extend it to also run the two reg policies in the same pass. Memory stays flat; cost is 3× forward passes per update (online, reg_m, reg_{m-1}), which is acceptable.

6. **Test-time heuristics — decision: ship threshold/discretize only in v1.** Stratego-specific heuristics (memory heuristic §292, value-bounds heuristic §295, eagerness §288) don't transfer directly to MTG and would each be a separate project. Threshold+discretize is already needed for fine-tuning and is trivial to reuse at eval.

## Architecture

### New module: `magic_ai/rnad.py`

Mirrors `ppo.py`'s shape. Exports:

- **`RNaDConfig`** dataclass. Fields: `eta=0.2`, `delta_m=25_000`, `num_outer_iterations=20`, `vtrace_rho_bar=1.0`, `vtrace_c_bar=1.0`, `neurd_beta=2.0`, `neurd_clip=10_000.0`, `grad_clip=10_000.0`, `target_ema_gamma=0.001`, `reg_interpolation_fraction=0.5` (gives $\alpha_n = \min(1, 2n/\Delta_m)$), `finetune_eps=0.03`, `finetune_n_disc=16`.

- **`transform_rewards(rewards, logp_theta, logp_reg_cur, logp_reg_prev, alpha, eta, perspective_idx)`** → transformed per-step reward (§161–164). Sign-flip $(1 - 2\cdot\mathbf{1}_{i=\psi_t})$ reuses perspective logic from `gae_returns`.

- **`two_player_vtrace(trajectory, logp_theta, logp_mu, values, transformed_rewards, perspective_idx, rho_bar, c_bar)`** → `(v_hat, q_hat)`. Backward recursion over full episode (no bootstrap). Importance ratio applied only on own-turn steps; opponent-turn rewards accumulate multiplicatively via $\xi_t$ (§172). Produces $\hat Q_t$ for every legal action under each head (§179–180), not just the played action.

- **`neurd_loss(logits, q_hat, legal_mask, beta, clip)`** (§188). `Clip(Q, c_neurd)` multiplied by gradient of joint-head logits (stop-grad on Q); the $\hat\nabla$ operator gates gradient contributions when logits leave $[-\beta, \beta]$. Per-head summation, averaged over effective timesteps.

- **`critic_loss(v_theta, v_hat, perspective_mask)`** — L1 regression over own-turn steps (§186).

- **`threshold_discretize(probs, eps, n_disc)`** — §197 / §279–282. Used both in fine-tune sampling and at eval.

- **`rnad_update(policy, target_policy, reg_cur, reg_prev, batch, config, step_in_m)`** — orchestrates one optimizer step: run transforms, v-trace, losses, Adam, grad-clip, Polyak update of target.

### Buffer changes (`magic_ai/buffer.py`)
No new storage. Existing fields (observations, legal-action masks, sampled-action traces, old log-probs, LSTM state) are sufficient once we recompute per-action logits under both reg snapshots at update time.

### Reg-policy storage
Reuse `OpponentPool` snapshot serialization for $\pi_{m,\text{reg}}$. Hold two frozen `PPOPolicy` copies in RAM (prev + current reg); persist to `checkpoints/rnad/reg_m{N}.pt` so training can resume. Load reg params once per outer step $m$; no gradient, `eval()` mode.

### Target network
Add `target_policy: PPOPolicy`. Polyak-averaged with γ=0.001 every optimizer step (§191). V-trace inputs come from the target network; NeuRD gradients flow into the online network. Crib EMA mechanics from the existing SPR code in `model.py`.

### Action-space nuance: per-head NeuRD
MTG's joint action is `(option, target, may, none/blocker)`. Under the per-head decision above:
- Each head computes its own $\hat Q$ (over its own legal subset) and its own NeuRD loss.
- Reward transform uses the **joint** log-ratio $\log \pi_\theta(a|o) - \log \pi_{\text{reg}}(a|o)$, summed across heads, at the joint-action level.
- The critic is a single scalar value head, unchanged.

### Reward shape
Base terminal reward remains ±1 (winner/loser). Draw → 0. Entropy-regularization bonus added per-step via reward transform. Existing perspective-flipped accumulation in `gae_returns` is replaced by the v-trace recursion, not reused.

### Outer loop (sketch)
```
init online θ; target ← θ; reg_prev ← uniform-ish; reg_cur ← uniform-ish
for m in range(M):
    for n in range(Δ_m):
        rollouts ← self-play with target policy
        α_n ← min(1, 2n/Δ_m)
        rewards' ← transform_rewards(..., α_n, reg_cur, reg_prev, η)
        v̂, Q̂ ← two_player_vtrace(target logits, μ logp, v_target, rewards')
        loss ← critic_loss(v_θ, v̂) + Σ_head neurd_loss(logits_θ, Q̂)
        Adam step; grad-clip; Polyak update target
    π_{m,fix} ← target; persist snapshot
    reg_prev ← reg_cur; reg_cur ← π_{m,fix}
fine-tune: repeat loop with threshold_discretize projection at sampling time
```

### CLI and dispatch
Add `--trainer {ppo,rnad}` to `scripts/train_ppo.py`. All rollout flags (`--num-envs`, `--rollout-steps`, `--max-steps-per-game`, sharding) apply to both. New R-NaD-specific flags: `--rnad-eta`, `--rnad-delta-m`, `--rnad-m`, `--rnad-neurd-beta`, `--rnad-finetune-eps`, `--rnad-finetune-ndisc`, `--rnad-target-ema`. `--draw-penalty` is overridden to 0 when `--trainer rnad` unless explicitly set.

## Validation plan

1. **Unit tests** (`tests/test_rnad.py`):
   - `two_player_vtrace` on a hand-crafted 3-step trajectory matches a Python-reference closed-form.
   - `transform_rewards` matches eq. §161 on symbolic inputs, including sign flip.
   - `neurd_loss` gradient matches autograd of $-\sum \pi Q$ in the unclipped regime; outside-$[-β,β]$ logits produce zero gradient.
   - `threshold_discretize` sums to 1, drops entries below ε, quantizes to $1/n$ grid.

2. **Toy matching-pennies**: add a degenerate MTG scenario (or pure-NFG harness) where the Nash equilibrium is known (e.g., uniform). Confirm convergence within a few outer iterations. This is the paper's own worked example.

3. **Head-to-head vs PPO**: train for M=5, Δ_m=25k. Evaluate via TrueSkill against the current PPO snapshot pool. Target: within 1 σ of PPO after matched compute. A blowout is not expected at this scale — the paper's advantage is asymptotic.

4. **Encoder parity**: `encoder_parity.py` must continue to pass — no new buffer fields means this should be automatic.

## Scope / non-goals for v1
- Single-GPU PyTorch; no JAX, no pmap, no Sebulba redesign.
- No joint-action NeuRD enumeration (per-head instead).
- No MTG-specific search / lookahead heuristics analogous to Stratego's memory or value-bounds heuristics.
- No deployment-phase head (MTG has no symmetric deployment phase).

## Deliverable layout
```
magic_ai/rnad.py                    # new — losses, estimators, config, update
scripts/train_ppo.py                # add --trainer dispatch + R-NaD flags
tests/test_rnad.py                  # new unit + integration tests
docs/rnad_design.md                 # this file
docs/rnad_implementation_plan.md    # checklist
checkpoints/rnad/                   # reg snapshots, target EMA
```

Estimated effort: 2–3 weeks for a working v1, plus 1 week of tuning and a head-to-head eval.
