"""Training-facing actor-critic wrapper for the text encoder policy."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical

from magic_ai.actions import (
    COLORS,
    ActionRequest,
    action_from_attackers,
    action_from_blockers,
    action_from_choice_accepted,
    action_from_choice_color,
    action_from_choice_ids,
    action_from_choice_index,
    action_from_priority_candidate,
    build_priority_candidates,
    selected_option_id,
)
from magic_ai.game_state import PendingState
from magic_ai.model import TRACE_KIND_TO_ID, ActionTrace, PolicyStep, TraceKind
from magic_ai.replay_decisions import (
    ReplayPerChoice,
    ReplayScoringForward,
    direct_decision_logits_from_forward,
    score_may_decisions_from_forward,
)
from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.policy import EncodedSnapshots
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
    RecurrentTextPolicyOutput,
    _cast_encoded,
)
from magic_ai.text_encoder.replay_buffer import TextReplayBatch, TextReplayBuffer


@dataclass(frozen=True)
class TextActorCriticStep:
    output: RecurrentTextPolicyOutput
    h_out: Tensor
    c_out: Tensor


@dataclass(frozen=True)
class CachedReplayForward:
    """Per-policy cache of a single replay-batch forward.

    Produced by :meth:`TextActorCritic.precompute_replay_forward` and consumed
    by :meth:`TextActorCritic.evaluate_replay_batch_per_choice` so R-NaD's
    LSTM-recompute and per-choice scoring share one encoder forward instead of
    running it twice on the same flat batch.
    """

    flat_rows: tuple[int, ...]
    batch: TextReplayBatch
    encoded: EncodedSnapshots
    h_concat: Tensor


@dataclass(frozen=True)
class TextDecisionLayout:
    trace_kind: TraceKind
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    pending: PendingState


class TextActorCritic(nn.Module):
    """Thin stateful wrapper around :class:`RecurrentTextPolicy`.

    The wrapper owns live per-env, per-player recurrent state. Replay storage
    remains external in ``TextReplayBuffer`` so PPO/R-NaD can swap algorithms
    without changing the model wrapper.
    """

    def __init__(self, cfg: RecurrentTextPolicyConfig) -> None:
        super().__init__()
        self.policy = RecurrentTextPolicy(cfg)
        self.lstm_layers = cfg.lstm_layers
        self.lstm_hidden = cfg.lstm_hidden
        self.none_head = nn.Linear(self.lstm_hidden, 1)
        self.may_head = nn.Linear(self.lstm_hidden, 1)
        self.register_buffer("live_lstm_h", torch.zeros(self.lstm_layers, 0, self.lstm_hidden))
        self.register_buffer("live_lstm_c", torch.zeros(self.lstm_layers, 0, self.lstm_hidden))
        self._num_envs = 0
        self._players_per_env = 2
        self.spr_enabled = False
        self.rollout_buffer: TextReplayBuffer | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def clone_for_rnad(self) -> TextActorCritic:
        """Deep-copy parameter state, share the replay buffer.

        See :meth:`magic_ai.training_interfaces.RNaDTrainablePolicy.clone_for_rnad`
        for the contract. Live per-env LSTM cache is reset on the clone since
        target/reg policies never sample from a live env.
        """

        original_buffer = self.rollout_buffer
        self.rollout_buffer = None
        try:
            clone = copy.deepcopy(self)
        finally:
            self.rollout_buffer = original_buffer
        clone.rollout_buffer = original_buffer
        clone._num_envs = 0
        clone._players_per_env = self._players_per_env
        clone.live_lstm_h = torch.zeros(
            self.lstm_layers, 0, self.lstm_hidden, dtype=torch.float32, device=clone.device
        )
        clone.live_lstm_c = torch.zeros_like(clone.live_lstm_h)
        return clone

    def count_active_replay_steps(
        self,
        per_episode_replay_rows: Sequence[Sequence[int]],
    ) -> tuple[int, int]:
        """Compute R-NaD ``(cl_count, pl_count)`` totals from text replay rows.

        TextReplayBuffer stores decisions as ``(capacity, max_decision_groups,
        max_cached_choices)`` per step (no flat ``decision_start`` table), so
        the per-choice count is just the active-cell sum of ``decision_mask``
        over the selected rows.
        """

        if not per_episode_replay_rows:
            return 0, 0
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")

        rb = self.rollout_buffer
        flat_rows: list[int] = [r for ep in per_episode_replay_rows for r in ep]
        device = rb.trace_kind_id.device
        step_indices = torch.tensor(flat_rows, dtype=torch.long, device=device)
        cl_count_total = int(step_indices.numel())
        trace_kind = rb.trace_kind_id[step_indices]
        may_count_total = int((trace_kind == TRACE_KIND_TO_ID["may"]).sum().item())
        flat_count_total = int(rb.decision_mask[step_indices].sum().item())
        return cl_count_total, may_count_total + flat_count_total

    def init_lstm_env_states(self, num_envs: int, *, players_per_env: int = 2) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if players_per_env < 1:
            raise ValueError("players_per_env must be at least 1")
        self._num_envs = int(num_envs)
        self._players_per_env = int(players_per_env)
        slots = self._num_envs * self._players_per_env
        shape = (self.lstm_layers, slots, self.lstm_hidden)
        self.live_lstm_h = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.live_lstm_c = torch.zeros_like(self.live_lstm_h)

    def reset_lstm_env_states(self, env_indices: list[int]) -> None:
        if self._num_envs == 0:
            raise RuntimeError("LSTM env states have not been initialized")
        slots = self._state_slots(env_indices, None)
        self.live_lstm_h[:, slots] = 0
        self.live_lstm_c[:, slots] = 0

    def lstm_env_state_inputs(
        self,
        env_indices: list[int],
        perspective_player_indices: list[int],
    ) -> tuple[Tensor, Tensor]:
        slots = self._state_slots(env_indices, perspective_player_indices)
        return (
            self.live_lstm_h[:, slots].contiguous(),
            self.live_lstm_c[:, slots].contiguous(),
        )

    def forward_live(
        self,
        batch: TextEncodedBatch,
        *,
        env_indices: list[int],
        perspective_player_indices: list[int],
    ) -> TextActorCriticStep:
        h_in, c_in = self.lstm_env_state_inputs(env_indices, perspective_player_indices)
        output, (h_out, c_out) = self.policy(batch, h_in=h_in, c_in=c_in)
        self.scatter_lstm_env_states(env_indices, perspective_player_indices, h_out, c_out)
        return TextActorCriticStep(output=output, h_out=h_out, c_out=c_out)

    def sample_text_batch(
        self,
        batch: TextEncodedBatch,
        *,
        env_indices: list[int],
        perspective_player_indices: list[int],
        layouts: list[TextDecisionLayout],
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        """Sample a live text-encoded batch and append replay rows.

        The caller owns render-plan emission and assembly. This method owns
        recurrent state, decision-group sampling, and replay-buffer writes so
        PPO/R-NaD can re-score the exact sampled text rows later.
        """

        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        n = int(batch.token_ids.shape[0])
        if n == 0:
            return []
        if len(env_indices) != n or len(perspective_player_indices) != n or len(layouts) != n:
            raise ValueError("batch, env_indices, perspective_player_indices, and layouts differ")

        moved = _move_text_batch(batch, self.device)
        h_in, c_in = self.lstm_env_state_inputs(env_indices, perspective_player_indices)
        output, (h_out, c_out) = self.policy(moved, h_in=h_in, c_in=c_in)
        self.scatter_lstm_env_states(env_indices, perspective_player_indices, h_out, c_out)

        none_logits = self.none_head(output.state_hidden).squeeze(-1)
        may_logits = self.may_head(output.state_hidden).squeeze(-1)
        # Phase 1: queue all GPU work without ever forcing a sync. We collect
        # selected-col 0-D tensors, per-step log_prob/entropy/value tensors,
        # and Python metadata. Old code synced once per step (selected cols
        # via .tolist(), log_prob via .cpu(), value via .cpu(), plus a per-
        # group mask.any().item()); for n=32 envs × ~3 groups that was ~120
        # syncs/call dominating the loop. We defer everything to one batched
        # sync at the end.
        per_step_log_prob: list[Tensor] = []
        per_step_entropy: list[Tensor] = []
        per_step_value: list[Tensor] = []
        per_step_may_sample: list[Tensor | None] = []
        per_step_trace_kind: list[TraceKind] = []
        per_step_layout: list[TextDecisionLayout] = []
        per_step_selected_tensors: list[list[Tensor]] = []
        for step_idx, layout in enumerate(layouts):
            trace_kind = layout.trace_kind
            value = output.values[step_idx]
            may_sample_t: Tensor | None = None
            selected_tensors: list[Tensor] = []

            if trace_kind == "may":
                may_logit = may_logits[step_idx]
                may_dist = Bernoulli(logits=may_logit)
                may_sample_t = (
                    (may_logit >= 0).to(dtype=output.values.dtype)
                    if deterministic
                    else may_dist.sample()
                )
                log_prob = may_dist.log_prob(may_sample_t)
                entropy = may_dist.entropy()
            else:
                log_prob = output.values.new_zeros(())
                entropy = output.values.new_zeros(())
                nb = self.device.type == "cuda"
                option_idx_dev = layout.decision_option_idx.to(self.device, non_blocking=nb)
                target_idx_dev = layout.decision_target_idx.to(self.device, non_blocking=nb)
                mask_dev = layout.decision_mask.to(self.device, non_blocking=nb)
                if mask_dev.dtype != torch.bool:
                    mask_dev = mask_dev.to(dtype=torch.bool)
                # ``layout.uses_none_head`` is already CPU; just materialize
                # to a Python list without going through the device.
                uses_none_flags = layout.uses_none_head.bool().tolist()
                num_groups = int(layout.decision_option_idx.shape[0])
                # Vectorize visibility across ALL groups for this step instead
                # of computing per-group inside the loop.
                opt_visibility = output.option_mask[step_idx]
                tgt_visibility = output.target_mask[step_idx]
                num_opts_view = int(opt_visibility.shape[0])
                num_tgts_view = int(tgt_visibility.shape[1])
                # ``decision_option_idx`` / ``decision_target_idx`` carry the
                # engine's option/target indices in the encoder's full
                # ``max_options`` / ``max_targets_per_option`` space.
                # ``to_text_encoded_batch`` trims those dims to the per-batch
                # active extent, so naked ``opt_visibility[opt_clamped_all]``
                # OOBs when an engine index lands past the trimmed extent
                # (the option_position simply wasn't emitted as a token).
                # Treat any out-of-range index as invisible.
                opt_in_range = (option_idx_dev >= 0) & (option_idx_dev < num_opts_view)
                tgt_in_range = (target_idx_dev >= 0) & (target_idx_dev < num_tgts_view)
                if num_opts_view > 0:
                    opt_clamped_all = option_idx_dev.clamp(min=0, max=num_opts_view - 1)
                    opt_visible_all = opt_visibility[opt_clamped_all] & opt_in_range
                else:
                    opt_clamped_all = option_idx_dev.clamp(min=0)
                    opt_visible_all = torch.zeros_like(option_idx_dev, dtype=torch.bool)
                if num_opts_view > 0 and num_tgts_view > 0:
                    tgt_clamped_all = target_idx_dev.clamp(min=0, max=num_tgts_view - 1)
                    tgt_visible_all = (
                        tgt_visibility[opt_clamped_all, tgt_clamped_all] & tgt_in_range
                    )
                else:
                    tgt_clamped_all = target_idx_dev.clamp(min=0)
                    tgt_visible_all = torch.zeros_like(target_idx_dev, dtype=torch.bool)
                target_required_all = target_idx_dev >= 0
                visible_all = opt_visible_all & (~target_required_all | tgt_visible_all)
                if any(uses_none_flags):
                    col0 = torch.zeros_like(visible_all)
                    for gi, uses in enumerate(uses_none_flags):
                        if uses:
                            col0[gi, 0] = True
                    visible_all = visible_all | col0
                mask_all = mask_dev & visible_all  # [G, C]

                for group_idx in range(num_groups):
                    mask = mask_all[group_idx]
                    # Skip the per-group ``bool(mask.any())`` sync. If a layout
                    # group's mask is empty after visibility-trim, the
                    # downstream Categorical raises with a clear NaN error,
                    # which is enough to surface the bug without burning a
                    # sync per group on the happy path.
                    logits = self._direct_live_decision_logits(
                        output,
                        none_logits,
                        step_idx=step_idx,
                        option_idx=option_idx_dev[group_idx],
                        target_idx=target_idx_dev[group_idx],
                        uses_none=uses_none_flags[group_idx],
                    )
                    valid_cols = mask.nonzero(as_tuple=False).squeeze(-1)
                    if valid_cols.numel() == 0:
                        # The model's option-visibility view disagrees with
                        # the engine: this row has engine-valid choices but
                        # the rendered text didn't emit <option> tokens for
                        # them, so ``policy_logits`` at those positions is
                        # all -inf (the policy head masks invalid options).
                        # Happens for trace_kinds whose decision options
                        # aren't priority options (notably ``choice_ids``).
                        # Fall back to uniform sampling over engine-valid
                        # columns; the policy can't score these meaningfully
                        # but we keep the rollout alive.
                        engine_cols = mask_dev[group_idx].nonzero(as_tuple=False).squeeze(-1)
                        if engine_cols.numel() == 0:
                            # Fully empty engine mask — nothing to choose.
                            # Append a sentinel col 0 so the layout stays
                            # consistent and let the engine reject if it
                            # cares.
                            selected_tensors.append(
                                torch.zeros((), device=self.device, dtype=torch.long)
                            )
                            continue
                        n_valid = engine_cols.numel()
                        if deterministic:
                            sel_in_valid = torch.zeros((), device=self.device, dtype=torch.long)
                        else:
                            sel_in_valid = torch.randint(
                                low=0, high=n_valid, size=(), device=self.device
                            )
                        chosen = engine_cols[sel_in_valid]
                        # Uniform log-prob over the n_valid engine choices,
                        # zero entropy contribution (model has no view).
                        uniform_log_prob = output.values.new_tensor(
                            -float(torch.log(torch.tensor(float(n_valid))))
                        )
                        log_prob = log_prob + uniform_log_prob
                        selected_tensors.append(chosen)
                        continue
                    valid_logits = logits[valid_cols]
                    dist = Categorical(logits=valid_logits)
                    if deterministic:
                        selected_t = torch.argmax(valid_logits)
                    else:
                        selected_t = dist.sample()
                    log_prob = log_prob + dist.log_prob(selected_t)
                    entropy = entropy + dist.entropy()
                    selected_tensors.append(valid_cols[selected_t])

            per_step_log_prob.append(log_prob)
            per_step_entropy.append(entropy)
            per_step_value.append(value)
            per_step_may_sample.append(may_sample_t)
            per_step_trace_kind.append(trace_kind)
            per_step_layout.append(layout)
            per_step_selected_tensors.append(selected_tensors)

        # Phase 2: one batched GPU→CPU sync to materialize everything.
        flat_selected: list[Tensor] = [t for sublist in per_step_selected_tensors for t in sublist]
        if flat_selected:
            selected_flat_cpu = torch.stack(flat_selected).detach().cpu().tolist()
        else:
            selected_flat_cpu = []
        log_prob_cpu = torch.stack(per_step_log_prob).detach().cpu().tolist()
        value_cpu = torch.stack(per_step_value).detach().cpu().tolist()
        may_sample_cpu: list[float | None] = []
        if any(t is not None for t in per_step_may_sample):
            stacked_may = (
                torch.stack(
                    [
                        t if t is not None else output.values.new_zeros(())
                        for t in per_step_may_sample
                    ]
                )
                .detach()
                .cpu()
                .tolist()
            )
            for raw, original in zip(stacked_may, per_step_may_sample, strict=True):
                may_sample_cpu.append(raw if original is not None else None)
        else:
            may_sample_cpu = [None] * len(per_step_may_sample)

        # Phase 3: build PolicyStep results in Python from the materialized
        # values. This is pure CPU/Python work after this point.
        results: list[PolicyStep] = []
        cursor = 0
        for step_idx in range(len(layouts)):
            trace_kind = per_step_trace_kind[step_idx]
            layout = per_step_layout[step_idx]
            log_prob = per_step_log_prob[step_idx]
            entropy = per_step_entropy[step_idx]
            value = per_step_value[step_idx]
            num_selected = len(per_step_selected_tensors[step_idx])
            selected_cols = selected_flat_cpu[cursor : cursor + num_selected]
            cursor += num_selected

            may_selected = 0
            if trace_kind == "may":
                may_val = may_sample_cpu[step_idx]
                assert may_val is not None
                may_selected = int(may_val >= 0.5)
                action = action_from_choice_accepted(bool(may_selected))
                trace = ActionTrace("may", binary=(float(may_selected),))
            else:
                trace, action = _decode_text_action(trace_kind, layout.pending, selected_cols)

            replay_idx = self.rollout_buffer.append(
                encoded=moved,
                batch_index=step_idx,
                trace_kind_id=TRACE_KIND_TO_ID[trace_kind],
                decision_option_idx=layout.decision_option_idx,
                decision_target_idx=layout.decision_target_idx,
                decision_mask=layout.decision_mask,
                uses_none_head=layout.uses_none_head,
                selected_indices=torch.tensor(selected_cols, dtype=torch.long),
                may_selected=float(may_selected),
                old_log_prob=float(log_prob_cpu[step_idx]),
                value=float(value_cpu[step_idx]),
                perspective_player_idx=int(perspective_player_indices[step_idx]),
                lstm_h_in=h_in[:, step_idx].detach(),
                lstm_c_in=c_in[:, step_idx].detach(),
            )
            results.append(
                PolicyStep(
                    action=action,
                    trace=trace,
                    log_prob=log_prob,
                    value=value,
                    entropy=entropy,
                    replay_idx=replay_idx,
                    selected_choice_cols=tuple(selected_cols),
                    may_selected=may_selected,
                )
            )
        return results

    def evaluate_replay_batch(
        self,
        replay_rows: list[int],
        *,
        return_extras: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        if return_extras:
            raise ValueError("TextActorCritic does not implement SPR replay extras")
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        batch = self.rollout_buffer.gather(replay_rows)
        if batch.lstm_h_in is None or batch.lstm_c_in is None:
            output, _state = self.policy(batch.encoded)
        else:
            h_in = batch.lstm_h_in.permute(1, 0, 2).contiguous()
            c_in = batch.lstm_c_in.permute(1, 0, 2).contiguous()
            output, _state = self.policy(batch.encoded, h_in=h_in, c_in=c_in)

        n = int(batch.trace_kind_id.shape[0])
        log_probs = output.values.new_zeros(n)
        entropies = output.values.new_zeros(n)

        forward = self._replay_scoring_forward(output)
        may_mask = batch.trace_kind_id == TRACE_KIND_TO_ID["may"]
        may_log_probs, may_entropies, _may_logits_per_step, _may_selected_per_step = (
            score_may_decisions_from_forward(
                forward,
                may_selected=batch.may_selected,
                may_mask=may_mask,
            )
        )
        log_probs = log_probs + may_log_probs
        entropies = entropies + may_entropies

        decision_log_probs, decision_entropies = cast(
            tuple[Tensor, Tensor],
            self._evaluate_decision_groups(output, batch),
        )
        log_probs = log_probs + decision_log_probs
        entropies = entropies + decision_entropies
        return log_probs, entropies, output.values, None

    def compute_spr_loss(
        self,
        step_indices: Tensor,
        *,
        extras: Any | None = None,
    ) -> Tensor:
        del step_indices, extras
        raise ValueError("TextActorCritic does not implement SPR")

    def update_spr_target(self, decay: float | None = None) -> None:
        del decay

    def evaluate_replay_batch_per_choice(
        self,
        replay_rows: list[int],
        *,
        lstm_state_override: tuple[Tensor, Tensor] | None = None,
        hidden_override: Tensor | None = None,
        cached: CachedReplayForward | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, ReplayPerChoice]:
        if cached is not None:
            if tuple(int(r) for r in replay_rows) != cached.flat_rows:
                raise ValueError("cached.flat_rows must match replay_rows")
            batch = cached.batch
            output = self.policy.forward_from_encoded(cached.encoded, cached.h_concat)
        else:
            if self.rollout_buffer is None:
                raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
            batch = self.rollout_buffer.gather(replay_rows)
            if hidden_override is not None:
                output, _state = self.policy(
                    batch.encoded,
                    state_hidden_override=hidden_override.to(self.device),
                )
            elif lstm_state_override is not None:
                output, _state = self.policy(
                    batch.encoded,
                    h_in=lstm_state_override[0],
                    c_in=lstm_state_override[1],
                )
            elif batch.lstm_h_in is None or batch.lstm_c_in is None:
                output, _state = self.policy(batch.encoded)
            else:
                h_in = batch.lstm_h_in.permute(1, 0, 2).contiguous()
                c_in = batch.lstm_c_in.permute(1, 0, 2).contiguous()
                output, _state = self.policy(batch.encoded, h_in=h_in, c_in=c_in)

        n = int(batch.trace_kind_id.shape[0])
        log_probs = output.values.new_zeros(n)
        entropies = output.values.new_zeros(n)
        forward = self._replay_scoring_forward(output)
        may_mask = batch.trace_kind_id == TRACE_KIND_TO_ID["may"]
        may_log_probs, may_entropies, may_logits_per_step, may_selected_per_step = (
            score_may_decisions_from_forward(
                forward,
                may_selected=batch.may_selected,
                may_mask=may_mask,
            )
        )
        log_probs = log_probs + may_log_probs
        entropies = entropies + may_entropies

        decision_log_probs, decision_entropies, per_choice = cast(
            tuple[Tensor, Tensor, ReplayPerChoice],
            self._evaluate_decision_groups(
                output,
                batch,
                return_per_choice=True,
            ),
        )
        log_probs = log_probs + decision_log_probs
        entropies = entropies + decision_entropies
        per_choice = ReplayPerChoice(
            flat_logits=per_choice.flat_logits,
            flat_log_probs=per_choice.flat_log_probs,
            group_idx=per_choice.group_idx,
            choice_cols=per_choice.choice_cols,
            is_sampled_flat=per_choice.is_sampled_flat,
            may_is_active=may_mask,
            may_logits_per_step=may_logits_per_step,
            may_selected_per_step=may_selected_per_step,
            decision_group_id_flat=per_choice.decision_group_id_flat,
            step_for_decision_group=per_choice.step_for_decision_group,
        )
        return log_probs, entropies, output.values, per_choice

    def recompute_lstm_states_for_episode(
        self,
        replay_rows: list[int],
    ) -> tuple[Tensor, Tensor]:
        result = self.recompute_lstm_states_for_episodes([replay_rows])
        return result[0]

    def recompute_lstm_states_for_episodes(
        self,
        episodes: list[list[int]],
        *,
        strategy: str = "legacy",
    ) -> list[tuple[Tensor, Tensor]]:
        del strategy
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if any(len(ep) == 0 for ep in episodes):
            raise ValueError("each episode must contain at least one row")
        return [self._recompute_lstm_states_for_rows(ep) for ep in episodes]

    def recompute_lstm_outputs_for_episodes(
        self,
        episodes: list[list[int]],
        *,
        chunk_size: int = 200,
        compiled_lstm: Callable[..., Any] | None = None,
    ) -> list[Tensor]:
        del chunk_size, compiled_lstm
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if any(len(ep) == 0 for ep in episodes):
            raise ValueError("each episode must contain at least one row")
        return [self._recompute_lstm_outputs_for_rows(ep) for ep in episodes]

    def precompute_replay_forward(
        self,
        episodes_replay_rows: Sequence[Sequence[int]],
    ) -> CachedReplayForward:
        """Run encoder once on the flat replay batch + per-episode LSTM scan.

        R-NaD's batched-trajectory loss runs an LSTM recompute and a per-choice
        scoring forward against the same replay rows under the same parameters.
        Both used to call :meth:`TextPolicy.encode_only` independently, paying
        the encoder cost twice. This helper consolidates them: the encoder runs
        once on the concatenated flat batch (under the same bf16 autocast as
        the live forward path), and the cached :class:`CachedReplayForward` is
        consumed by ``evaluate_replay_batch_per_choice(cached=...)`` to skip
        the second forward.
        """

        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        if not episodes_replay_rows:
            raise ValueError("episodes_replay_rows must be non-empty")
        flat_rows: list[int] = []
        ep_lens: list[int] = []
        for ep in episodes_replay_rows:
            ep_list = [int(r) for r in ep]
            if not ep_list:
                raise ValueError("each episode must contain at least one row")
            flat_rows.extend(ep_list)
            ep_lens.append(len(ep_list))
        batch = self.rollout_buffer.gather(flat_rows)

        device_type = batch.encoded.token_ids.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            encoded = self.policy.text_policy.encode_only(batch.encoded)
        target_dtype = self.policy.in_proj.weight.dtype
        encoded = _cast_encoded(encoded, target_dtype)

        # cuDNN's fused LSTM doesn't expose intermediate cell states for a
        # multi-episode batched call without padding-and-masking, and the
        # episodes here have ragged lengths. The per-episode LSTM scan stays;
        # what we save is the encoder forward, which is the dominant cost.
        state = self.policy.in_proj(encoded.state_vector)
        device = state.device
        h_chunks: list[Tensor] = []
        cursor = 0
        for n_ep in ep_lens:
            x = state[cursor : cursor + n_ep].unsqueeze(0)
            h0 = torch.zeros(self.lstm_layers, 1, self.lstm_hidden, device=device)
            c0 = torch.zeros_like(h0)
            out, _ = self.policy.lstm(x, (h0, c0))
            h_chunks.append(out.squeeze(0).contiguous())
            cursor += n_ep
        h_concat = torch.cat(h_chunks, dim=0)

        return CachedReplayForward(
            flat_rows=tuple(flat_rows),
            batch=batch,
            encoded=encoded,
            h_concat=h_concat,
        )

    def scatter_lstm_env_states(
        self,
        env_indices: list[int],
        perspective_player_indices: list[int],
        h_out: Tensor,
        c_out: Tensor,
    ) -> None:
        slots = self._state_slots(env_indices, perspective_player_indices)
        expected = (self.lstm_layers, len(slots), self.lstm_hidden)
        if tuple(h_out.shape) != expected or tuple(c_out.shape) != expected:
            raise ValueError(
                f"state outputs must have shape {expected}, got {tuple(h_out.shape)} "
                f"and {tuple(c_out.shape)}"
            )
        self.live_lstm_h[:, slots] = h_out.detach()
        self.live_lstm_c[:, slots] = c_out.detach()

    def _recompute_lstm_states_for_rows(self, replay_rows: list[int]) -> tuple[Tensor, Tensor]:
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        batch = self.rollout_buffer.gather(replay_rows)
        encoded = self.policy.text_policy.encode_only(batch.encoded)
        projected = self.policy.in_proj(encoded.state_vector)
        h = torch.zeros(self.lstm_layers, 1, self.lstm_hidden, device=projected.device)
        c = torch.zeros_like(h)
        h_list: list[Tensor] = []
        c_list: list[Tensor] = []
        for t in range(projected.shape[0]):
            h_list.append(h)
            c_list.append(c)
            _out, (h_t, c_t) = self.policy.lstm(
                projected[t : t + 1].unsqueeze(0),
                (h.contiguous(), c.contiguous()),
            )
            h = h_t
            c = c_t
        return torch.cat(h_list, dim=1).contiguous(), torch.cat(c_list, dim=1).contiguous()

    def _recompute_lstm_outputs_for_rows(self, replay_rows: list[int]) -> Tensor:
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        batch = self.rollout_buffer.gather(replay_rows)
        encoded = self.policy.text_policy.encode_only(batch.encoded)
        projected = self.policy.in_proj(encoded.state_vector).unsqueeze(0)
        h = torch.zeros(self.lstm_layers, 1, self.lstm_hidden, device=projected.device)
        c = torch.zeros_like(h)
        output, _state = self.policy.lstm(projected, (h, c))
        return output.squeeze(0).contiguous()

    def _evaluate_decision_groups(
        self,
        output: RecurrentTextPolicyOutput,
        batch: TextReplayBatch,
        *,
        return_per_choice: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, ReplayPerChoice]:
        decision_count = batch.decision_count
        n = int(decision_count.shape[0])
        log_probs = output.values.new_zeros(n)
        entropies = output.values.new_zeros(n)
        forward = self._replay_scoring_forward(output)
        device = forward.values.device

        decision_count_long = decision_count.to(dtype=torch.long)
        g_total = int(decision_count_long.sum().item())
        if g_total == 0:
            if not return_per_choice:
                return log_probs, entropies
            empty_long = torch.zeros(0, dtype=torch.long, device=device)
            empty_bool = torch.zeros(0, dtype=torch.bool, device=device)
            return (
                log_probs,
                entropies,
                ReplayPerChoice(
                    flat_logits=output.values.new_zeros(0),
                    flat_log_probs=output.values.new_zeros(0),
                    group_idx=empty_long,
                    choice_cols=empty_long,
                    is_sampled_flat=empty_bool,
                    may_is_active=torch.zeros(n, dtype=torch.bool, device=device),
                    may_logits_per_step=output.values.new_zeros(n),
                    may_selected_per_step=output.values.new_zeros(n),
                    decision_group_id_flat=empty_long,
                    step_for_decision_group=empty_long,
                ),
            )

        # Flatten (step, group) pairs in step-major / group-ascending order.
        # ``decision_group_id`` is the position in this flat list, so ordering
        # has to match what the per-choice consumers in ``rnad`` expect.
        steps_t = torch.arange(n, device=device).repeat_interleave(decision_count_long)
        group_starts = torch.cumsum(decision_count_long, dim=0) - decision_count_long
        groups_t = torch.arange(g_total, device=device) - group_starts[steps_t]

        flat_option_idx = batch.decision_option_idx[steps_t, groups_t]
        flat_target_idx = batch.decision_target_idx[steps_t, groups_t]
        flat_masks = batch.decision_mask[steps_t, groups_t]
        flat_uses_none = batch.uses_none_head[steps_t, groups_t]
        flat_selected = batch.selected_indices[steps_t, groups_t].to(dtype=torch.long)

        if not bool(flat_masks.any(dim=-1).all()):
            raise ValueError("decision group must include at least one valid choice")

        all_logits = direct_decision_logits_from_forward(
            forward,
            step_positions=steps_t,
            option_idx=flat_option_idx,
            target_idx=flat_target_idx,
            masks=flat_masks,
            uses_none=flat_uses_none,
        )

        masked_logits = all_logits.masked_fill(~flat_masks, float("-inf"))
        log_probs_dense = torch.log_softmax(masked_logits, dim=-1)
        probs_dense = log_probs_dense.exp()
        # ``-inf * 0`` at masked positions would be NaN; zero them explicitly.
        entropy_terms = torch.where(
            flat_masks, probs_dense * log_probs_dense, log_probs_dense.new_zeros(())
        )

        per_group_log_prob = log_probs_dense.gather(-1, flat_selected.unsqueeze(-1)).squeeze(-1)
        per_group_entropy = -entropy_terms.sum(dim=-1)

        log_probs = log_probs.scatter_add(0, steps_t, per_group_log_prob)
        entropies = entropies.scatter_add(0, steps_t, per_group_entropy)

        if not return_per_choice:
            return log_probs, entropies

        # ``nonzero`` returns row-major (group, col-ascending), matching the
        # original per-group ``mask.nonzero()`` concatenation order.
        flat_indices = flat_masks.nonzero(as_tuple=False)
        decision_group_id_flat = flat_indices[:, 0]
        choice_cols = flat_indices[:, 1]
        flat_logits = all_logits[decision_group_id_flat, choice_cols]
        flat_log_probs = log_probs_dense[decision_group_id_flat, choice_cols]
        is_sampled_flat = choice_cols == flat_selected[decision_group_id_flat]
        group_idx_out = steps_t[decision_group_id_flat]

        return (
            log_probs,
            entropies,
            ReplayPerChoice(
                flat_logits=flat_logits,
                flat_log_probs=flat_log_probs,
                group_idx=group_idx_out,
                choice_cols=choice_cols,
                is_sampled_flat=is_sampled_flat,
                may_is_active=torch.zeros(n, dtype=torch.bool, device=device),
                may_logits_per_step=output.values.new_zeros(n),
                may_selected_per_step=output.values.new_zeros(n),
                decision_group_id_flat=decision_group_id_flat,
                step_for_decision_group=steps_t,
            ),
        )

    def _replay_scoring_forward(self, output: RecurrentTextPolicyOutput) -> ReplayScoringForward:
        return ReplayScoringForward(
            values=output.values,
            option_vectors=output.option_vectors,
            target_vectors=output.target_vectors,
            none_logits=self.none_head(output.state_hidden).squeeze(-1),
            may_logits=self.may_head(output.state_hidden).squeeze(-1),
            hidden=output.state_hidden,
            option_logits=output.policy_logits,
            target_logits=output.target_logits,
        )

    def _direct_decision_logits(
        self,
        forward: ReplayScoringForward,
        *,
        step_idx: int,
        group_idx: int,
        batch: TextReplayBatch,
    ) -> Tensor:
        device = forward.values.device
        return direct_decision_logits_from_forward(
            forward,
            step_positions=torch.tensor([step_idx], dtype=torch.long, device=device),
            option_idx=batch.decision_option_idx[step_idx, group_idx].unsqueeze(0),
            target_idx=batch.decision_target_idx[step_idx, group_idx].unsqueeze(0),
            masks=batch.decision_mask[step_idx, group_idx].unsqueeze(0),
            uses_none=batch.uses_none_head[step_idx, group_idx].unsqueeze(0),
        )[0]

    def _direct_live_decision_logits(
        self,
        output: RecurrentTextPolicyOutput,
        none_logits: Tensor,
        *,
        step_idx: int,
        option_idx: Tensor,
        target_idx: Tensor,
        uses_none: bool,
    ) -> Tensor:
        # Vectorized gather: avoid the per-column Python loop and the two
        # ``.item()`` syncs it carried. ``option_idx`` / ``target_idx`` may
        # contain -1 sentinels for masked columns; clamp before indexing and
        # use ``torch.where`` to fill those slots with -inf afterwards.
        # Guard against zero-sized option/target dims (a step with no live
        # options or no live targets) — gather would still raise even though
        # the values get masked away by ``torch.where``.
        num_options = int(output.policy_logits.shape[1])
        num_targets = int(output.target_logits.shape[2])
        # ``option_idx`` / ``target_idx`` use the engine's full-width
        # ``max_options`` / ``max_targets_per_option`` ranges; the model's
        # logits axes are trimmed to the per-batch active extent. Clamp to
        # the trimmed extent and mask out-of-range positions with -inf.
        opt_in_range = (option_idx >= 0) & (option_idx < num_options)
        tgt_in_range = (target_idx >= 0) & (target_idx < num_targets)
        opt_clamped = option_idx.clamp(min=0, max=max(num_options - 1, 0))
        tgt_clamped = target_idx.clamp(min=0, max=max(num_targets - 1, 0))
        if num_options > 0:
            option_vals = output.policy_logits[step_idx, opt_clamped]
        else:
            option_vals = output.values.new_full(option_idx.shape, float("-inf"))
        if num_options > 0 and num_targets > 0:
            target_vals = output.target_logits[step_idx, opt_clamped, tgt_clamped]
        else:
            target_vals = option_vals
        logits = torch.where(target_idx >= 0, target_vals, option_vals)
        neg_inf = output.values.new_tensor(float("-inf"))
        logits = torch.where(opt_in_range, logits, neg_inf)
        logits = torch.where((target_idx >= 0) & ~tgt_in_range, neg_inf, logits)
        if uses_none:
            logits = logits.clone()
            logits[0] = none_logits[step_idx]
        return logits

    def _state_slots(
        self,
        env_indices: list[int],
        perspective_player_indices: list[int] | None,
    ) -> Tensor:
        if self._num_envs == 0:
            raise RuntimeError("LSTM env states have not been initialized")
        if perspective_player_indices is not None and len(env_indices) != len(
            perspective_player_indices
        ):
            raise ValueError("env_indices and perspective_player_indices must have equal length")
        slots: list[int] = []
        if perspective_player_indices is None:
            for env_idx in env_indices:
                self._validate_env_idx(env_idx)
                start = int(env_idx) * self._players_per_env
                slots.extend(range(start, start + self._players_per_env))
        else:
            for env_idx, player_idx in zip(env_indices, perspective_player_indices, strict=True):
                self._validate_env_idx(env_idx)
                if player_idx < 0 or player_idx >= self._players_per_env:
                    raise IndexError("perspective player index out of range")
                slots.append(int(env_idx) * self._players_per_env + int(player_idx))
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def _validate_env_idx(self, env_idx: int) -> None:
        if env_idx < 0 or env_idx >= self._num_envs:
            raise IndexError("env index out of range")


def build_text_decision_layout(
    trace_kind: TraceKind,
    pending: PendingState,
    *,
    max_options: int,
    max_targets_per_option: int,
    max_cached_choices: int,
) -> TextDecisionLayout:
    if trace_kind == "may":
        return _layout(trace_kind, pending, [], [], [], [], max_cached_choices)

    if trace_kind == "priority":
        candidates = build_priority_candidates(
            pending,
            max_targets_per_option=max_targets_per_option,
        )[:max_cached_choices]
        if not candidates:
            return _layout(trace_kind, pending, [], [], [], [], max_cached_choices)
        option_row = [-1] * max_cached_choices
        target_row = [-1] * max_cached_choices
        mask_row = [False] * max_cached_choices
        for col, cand in enumerate(candidates):
            option_row[col] = cand.option_index
            if cand.target_index is not None:
                target_row[col] = cand.target_index
            mask_row[col] = True
        return _layout(
            trace_kind,
            pending,
            [option_row],
            [target_row],
            [mask_row],
            [False],
            max_cached_choices,
        )

    options = pending.get("options", [])[:max_options]
    if not options:
        return _layout(trace_kind, pending, [], [], [], [], max_cached_choices)

    if trace_kind == "attackers":
        option_rows = []
        target_rows = []
        mask_rows = []
        for i, _option in enumerate(options):
            option_row = [-1] * max_cached_choices
            target_row = [-1] * max_cached_choices
            mask_row = [False] * max_cached_choices
            if max_cached_choices > 0:
                mask_row[0] = True
            if max_cached_choices > 1:
                option_row[1] = i
                mask_row[1] = True
            option_rows.append(option_row)
            target_rows.append(target_row)
            mask_rows.append(mask_row)
        return _layout(
            trace_kind,
            pending,
            option_rows,
            target_rows,
            mask_rows,
            [True] * len(option_rows),
            max_cached_choices,
        )

    if trace_kind == "blockers":
        option_rows = []
        target_rows = []
        mask_rows = []
        for i, option in enumerate(options):
            option_row = [-1] * max_cached_choices
            target_row = [-1] * max_cached_choices
            mask_row = [False] * max_cached_choices
            if max_cached_choices > 0:
                mask_row[0] = True
            target_count = min(
                len(option.get("valid_targets", [])),
                max_targets_per_option,
                max_cached_choices - 1,
            )
            for target_idx in range(target_count):
                col = target_idx + 1
                option_row[col] = i
                target_row[col] = target_idx
                mask_row[col] = True
            option_rows.append(option_row)
            target_rows.append(target_row)
            mask_rows.append(mask_row)
        return _layout(
            trace_kind,
            pending,
            option_rows,
            target_rows,
            mask_rows,
            [True] * len(option_rows),
            max_cached_choices,
        )

    option_row = [-1] * max_cached_choices
    target_row = [-1] * max_cached_choices
    mask_row = [False] * max_cached_choices
    for i in range(min(len(options), max_cached_choices)):
        option_row[i] = i
        mask_row[i] = True
    return _layout(
        trace_kind,
        pending,
        [option_row],
        [target_row],
        [mask_row],
        [False],
        max_cached_choices,
    )


def infer_text_trace_kind(pending: PendingState) -> TraceKind:
    kind = pending.get("kind", "") or ""
    if kind in ("priority", "attackers", "blockers", "may"):
        return cast(TraceKind, kind)
    if kind == "mana_color":
        return "choice_color"
    if kind in ("cards_from_hand", "card_from_library"):
        return "choice_ids"
    return "choice_index"


def _layout(
    trace_kind: TraceKind,
    pending: PendingState,
    option_rows: list[list[int]],
    target_rows: list[list[int]],
    mask_rows: list[list[bool]],
    uses_none: list[bool],
    max_cached_choices: int,
) -> TextDecisionLayout:
    group_count = len(option_rows)
    if group_count == 0:
        shape = (0, max_cached_choices)
        decision_option_idx = torch.empty(shape, dtype=torch.long)
        decision_target_idx = torch.empty(shape, dtype=torch.long)
        decision_mask = torch.empty(shape, dtype=torch.bool)
        uses_none_head = torch.empty((0,), dtype=torch.bool)
    else:
        decision_option_idx = torch.tensor(option_rows, dtype=torch.long)
        decision_target_idx = torch.tensor(target_rows, dtype=torch.long)
        decision_mask = torch.tensor(mask_rows, dtype=torch.bool)
        uses_none_head = torch.tensor(uses_none, dtype=torch.bool)
    return TextDecisionLayout(
        trace_kind=trace_kind,
        decision_option_idx=decision_option_idx,
        decision_target_idx=decision_target_idx,
        decision_mask=decision_mask,
        uses_none_head=uses_none_head,
        pending=pending,
    )


def _decode_text_action(
    trace_kind: TraceKind,
    pending: PendingState,
    selected: list[int],
) -> tuple[ActionTrace, ActionRequest]:
    selected_idx = selected[0] if selected else 0
    if trace_kind == "priority":
        candidates = build_priority_candidates(pending)
        if not candidates:
            return ActionTrace("priority", indices=(0,)), {"kind": "pass"}
        selected_idx = min(selected_idx, len(candidates) - 1)
        return (
            ActionTrace("priority", indices=(selected_idx,)),
            action_from_priority_candidate(candidates[selected_idx]),
        )
    if trace_kind == "attackers":
        binary = tuple(float(value == 1) for value in selected)
        return (
            ActionTrace("attackers", binary=binary),
            action_from_attackers(pending, [value == 1.0 for value in binary]),
        )
    if trace_kind == "blockers":
        indices = tuple(value - 1 for value in selected)
        return ActionTrace("blockers", indices=indices), action_from_blockers(
            pending,
            list(indices),
        )
    if trace_kind == "choice_ids":
        target_id = selected_option_id(pending, selected_idx)
        return (
            ActionTrace("choice_ids", indices=(selected_idx,)),
            action_from_choice_ids([target_id] if target_id else []),
        )
    if trace_kind == "choice_color":
        options = pending.get("options", [])
        if 0 <= selected_idx < len(options):
            option = options[selected_idx]
            color = option.get("color", option.get("id", COLORS[selected_idx % len(COLORS)]))
        else:
            color = COLORS[selected_idx % len(COLORS)]
        return (
            ActionTrace("choice_color", indices=(selected_idx,)),
            action_from_choice_color(str(color)),
        )
    return (
        ActionTrace("choice_index", indices=(selected_idx,)),
        action_from_choice_index(selected_idx),
    )


def _move_text_batch(batch: TextEncodedBatch, device: torch.device) -> TextEncodedBatch:
    # ``non_blocking=True`` only takes effect when the source is pinned;
    # the native assembler allocates its outputs in pinned memory when CUDA
    # is available. Bool conversion runs on the device after the copy so
    # the H2D transfer happens on the actual storage (uint8) rather than
    # an unpinned bool intermediate.
    nb = device.type == "cuda"
    option_mask = batch.option_mask.to(device, non_blocking=nb)
    target_mask = batch.target_mask.to(device, non_blocking=nb)
    if option_mask.dtype != torch.bool:
        option_mask = option_mask.to(dtype=torch.bool)
    if target_mask.dtype != torch.bool:
        target_mask = target_mask.to(dtype=torch.bool)
    return TextEncodedBatch(
        token_ids=batch.token_ids.to(device, non_blocking=nb),
        attention_mask=batch.attention_mask.to(device, non_blocking=nb),
        card_ref_positions=batch.card_ref_positions.to(device, non_blocking=nb),
        option_positions=batch.option_positions.to(device, non_blocking=nb),
        option_mask=option_mask,
        target_positions=batch.target_positions.to(device, non_blocking=nb),
        target_mask=target_mask,
        seq_lengths=batch.seq_lengths.to(device, non_blocking=nb),
    )


__all__ = [
    "TextActorCritic",
    "TextActorCriticStep",
    "TextDecisionLayout",
    "build_text_decision_layout",
    "infer_text_trace_kind",
]
