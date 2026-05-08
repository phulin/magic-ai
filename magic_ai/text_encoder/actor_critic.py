"""Training-facing actor-critic wrapper for the text encoder policy."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical

from magic_ai.actions import (
    COLORS,
    TRACE_KIND_TO_ID,
    ActionRequest,
    ActionTrace,
    PolicyStep,
    TraceKind,
    action_from_attackers,
    action_from_choice_accepted,
    action_from_choice_color,
    action_from_choice_ids,
    action_from_choice_index,
    action_from_inline_block_choices,
    action_from_priority_candidate,
    build_decision_layout_rows,
    build_priority_candidates,
    selected_option_id,
)
from magic_ai.game_state import PendingState
from magic_ai.lstm_recompute import lstm_recompute_per_step_h_out_per_player
from magic_ai.replay_decisions import (
    ReplayPerChoice,
    ReplayScoringForward,
    score_may_decisions_from_forward,
)
from magic_ai.slot_encoder.model import _clone_detaching_buffer
from magic_ai.text_encoder.batch import (
    PackedTextBatch,
    TextEncodedBatch,
    pack_batch,
    subtract_packed_offsets,
)
from magic_ai.text_encoder.policy import EncodedSnapshots
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
    RecurrentTextPolicyOutput,
)
from magic_ai.text_encoder.render_plan import BLANK_GROUP_CROSS_BLANK, BLANK_GROUP_PER_BLANK
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
class _PendingDecisionCoverageCheck:
    unhandled_count_cpu: Tensor
    ready_event: torch.cuda.Event | None
    handled_groups: Tensor
    group_steps: Tensor
    trace_kind_id: Tensor
    decision_count: Tensor


@dataclass(frozen=True)
class TextDecisionLayout:
    trace_kind: TraceKind
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    pending: PendingState


@dataclass(frozen=True)
class NativeTextReplayPayload:
    encoded: PackedTextBatch
    trace_kind_id: Tensor
    decision_count: Tensor
    decision_count_host: tuple[int, ...] | None
    total_decision_groups: int | None
    total_stored_decision_groups: int | None
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    selected_indices: Tensor
    behavior_action_log_prob: Tensor | None
    may_selected: Tensor
    old_log_prob: Tensor
    value: Tensor
    perspective_player_idx: Tensor
    lstm_h_in: Tensor
    lstm_c_in: Tensor
    projected_state: Tensor | None = None


@dataclass(frozen=True)
class NativeTextSampleBatch:
    decision_counts: list[int]
    selected_choice_cols: list[int]
    may_selected: list[int]
    old_log_prob: list[float]
    value: list[float]
    replay_rows: list[int]
    replay_payload: NativeTextReplayPayload | None = None


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
        self._pending_decision_coverage_checks: list[_PendingDecisionCoverageCheck] = []
        self._compiled_inline_decision_batch: Callable[..., Any] | None = None
        self._compiled_combat_groups_batch: Callable[..., Any] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _raise_unhandled_vectorized_decisions(
        self,
        *,
        handled_groups: Tensor,
        group_steps: Tensor,
        trace_kind_id: Tensor,
        decision_count: Tensor,
    ) -> None:
        unhandled = (~handled_groups).nonzero(as_tuple=False).squeeze(-1)
        shown = unhandled[:16].detach().cpu()
        group_steps_cpu = group_steps.detach().cpu()
        trace_kind_cpu = trace_kind_id.detach().cpu()
        decision_count_cpu = decision_count.detach().cpu()
        detail = []
        id_to_trace = {int(v): k for k, v in TRACE_KIND_TO_ID.items()}
        for group_idx_t in shown:
            group_idx = int(group_idx_t)
            step_idx = int(group_steps_cpu[group_idx])
            trace_id = int(trace_kind_cpu[step_idx])
            detail.append(
                {
                    "group": group_idx,
                    "step": step_idx,
                    "trace": id_to_trace.get(trace_id, str(trace_id)),
                    "decision_count": int(decision_count_cpu[step_idx]),
                }
            )
        raise RuntimeError(
            "native text sampling has unhandled vectorized decision groups: "
            f"total={int(unhandled.numel())} examples={detail}"
        )

    def _drain_decision_coverage_checks(self) -> None:
        pending: list[_PendingDecisionCoverageCheck] = []
        for check in self._pending_decision_coverage_checks:
            if check.ready_event is not None and not check.ready_event.query():
                pending.append(check)
                continue
            if int(check.unhandled_count_cpu.item()) > 0:
                self._pending_decision_coverage_checks = pending
                self._raise_unhandled_vectorized_decisions(
                    handled_groups=check.handled_groups,
                    group_steps=check.group_steps,
                    trace_kind_id=check.trace_kind_id,
                    decision_count=check.decision_count,
                )
        self._pending_decision_coverage_checks = pending

    def _enqueue_decision_coverage_check(
        self,
        *,
        handled_groups: Tensor,
        group_steps: Tensor,
        trace_kind_id: Tensor,
        decision_count: Tensor,
    ) -> None:
        unhandled_count = (~handled_groups).sum().to(dtype=torch.int32)
        if handled_groups.device.type != "cuda":
            count_cpu = unhandled_count.detach().cpu()
            if int(count_cpu.item()) > 0:
                self._raise_unhandled_vectorized_decisions(
                    handled_groups=handled_groups,
                    group_steps=group_steps,
                    trace_kind_id=trace_kind_id,
                    decision_count=decision_count,
                )
            return

        count_cpu = torch.empty((), dtype=torch.int32, pin_memory=True)
        count_cpu.copy_(unhandled_count.detach(), non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(handled_groups.device))
        self._pending_decision_coverage_checks.append(
            _PendingDecisionCoverageCheck(
                unhandled_count_cpu=count_cpu,
                ready_event=event,
                handled_groups=handled_groups.detach(),
                group_steps=group_steps.detach(),
                trace_kind_id=trace_kind_id.detach(),
                decision_count=decision_count.detach(),
            )
        )
        self._drain_decision_coverage_checks()

    def _sample_inline_decision_batch(
        self,
        output: RecurrentTextPolicyOutput,
        batch: TextEncodedBatch | PackedTextBatch,
        *,
        option_idx: Tensor,
        target_idx: Tensor,
        decision_mask: Tensor,
        trace_kind_id: Tensor,
        decision_count: Tensor,
        decision_rows: int,
        deterministic: bool,
        profile_timings: dict[str, float] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        if (
            profile_timings is None
            and output.values.device.type == "cuda"
            and self.policy.cfg.compile_forward
        ):
            if self._compiled_inline_decision_batch is None:
                self._compiled_inline_decision_batch = torch.compile(_sample_inline_decision_batch)
            return cast(
                tuple[Tensor, Tensor, Tensor, Tensor] | None,
                self._compiled_inline_decision_batch(
                    output,
                    batch,
                    option_idx=option_idx,
                    target_idx=target_idx,
                    decision_mask=decision_mask,
                    trace_kind_id=trace_kind_id,
                    decision_count=decision_count,
                    decision_rows=decision_rows,
                    deterministic=deterministic,
                ),
            )
        if profile_timings is not None:
            return _sample_inline_decision_batch_profiled(
                output,
                batch,
                option_idx=option_idx,
                target_idx=target_idx,
                decision_mask=decision_mask,
                trace_kind_id=trace_kind_id,
                decision_count=decision_count,
                decision_rows=decision_rows,
                deterministic=deterministic,
                profile_timings=profile_timings,
            )
        return _sample_inline_decision_batch(
            output,
            batch,
            option_idx=option_idx,
            target_idx=target_idx,
            decision_mask=decision_mask,
            trace_kind_id=trace_kind_id,
            decision_count=decision_count,
            decision_rows=decision_rows,
            deterministic=deterministic,
        )

    def _sample_inline_combat_groups_batch(
        self,
        output: RecurrentTextPolicyOutput,
        batch: TextEncodedBatch | PackedTextBatch,
        *,
        option_idx: Tensor,
        decision_mask: Tensor,
        trace_kind_id: Tensor,
        decision_count: Tensor,
        group_steps: Tensor,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        if output.values.device.type == "cuda" and self.policy.cfg.compile_forward:
            if self._compiled_combat_groups_batch is None:
                self._compiled_combat_groups_batch = torch.compile(
                    _sample_inline_combat_groups_batch
                )
            return cast(
                tuple[Tensor, Tensor, Tensor, Tensor] | None,
                self._compiled_combat_groups_batch(
                    output,
                    batch,
                    option_idx=option_idx,
                    decision_mask=decision_mask,
                    trace_kind_id=trace_kind_id,
                    decision_count=decision_count,
                    group_steps=group_steps,
                    deterministic=deterministic,
                ),
            )
        return _sample_inline_combat_groups_batch(
            output,
            batch,
            option_idx=option_idx,
            decision_mask=decision_mask,
            trace_kind_id=trace_kind_id,
            decision_count=decision_count,
            group_steps=group_steps,
            deterministic=deterministic,
        )

    def clone_for_rnad(self) -> TextActorCritic:
        """Deep-copy parameter state, share the replay buffer.

        See :meth:`magic_ai.training_interfaces.RNaDTrainablePolicy.clone_for_rnad`
        for the contract. Live per-env LSTM cache is reset on the clone since
        target/reg policies never sample from a live env.
        """

        _clone = _clone_detaching_buffer(self, "rollout_buffer")
        clone = cast(TextActorCritic, _clone)
        clone._num_envs = 0  # type: ignore[attr-defined]
        clone._players_per_env = self._players_per_env  # type: ignore[attr-defined]
        clone._pending_decision_coverage_checks = []  # type: ignore[attr-defined]
        clone._compiled_inline_decision_batch = None  # type: ignore[attr-defined]
        clone._compiled_combat_groups_batch = None  # type: ignore[attr-defined]
        clone.live_lstm_h = torch.zeros(  # type: ignore[attr-defined]
            int(self.lstm_layers),
            0,
            int(self.lstm_hidden),
            dtype=torch.float32,
            device=clone.device,  # type: ignore[attr-defined]
        )
        clone.live_lstm_c = torch.zeros_like(clone.live_lstm_h)  # type: ignore[attr-defined]
        return clone

    def count_active_replay_steps(
        self,
        per_episode_replay_rows: Sequence[Sequence[int]],
    ) -> tuple[int, int]:
        """Compute R-NaD ``(cl_count, pl_count)`` totals from text replay rows.

        TextReplayBuffer stores decision groups in ReplayCore's flat decision
        arena, so the per-choice count is just the active-cell sum over the
        selected rows' decision masks.
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
        flat_count_total = int(rb.core.valid_choice_count(step_indices).item())
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
        batch: TextEncodedBatch | None,
        *,
        env_indices: list[int],
        perspective_player_indices: list[int],
        layouts: list[TextDecisionLayout],
        deterministic: bool = False,
        packed_batch: PackedTextBatch | None = None,
    ) -> list[PolicyStep]:
        """Sample a live text-encoded batch and append replay rows.

        The caller owns render-plan emission and assembly. This method owns
        recurrent state, decision-group sampling, and replay-buffer writes so
        PPO/R-NaD can re-score the exact sampled text rows later.
        """

        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        if batch is None and packed_batch is None:
            raise ValueError("batch or packed_batch is required")
        n = (
            int(packed_batch.seq_lengths.shape[0])
            if packed_batch is not None
            else int(cast(TextEncodedBatch, batch).token_ids.shape[0])
        )
        if n == 0:
            return []
        if len(env_indices) != n or len(perspective_player_indices) != n or len(layouts) != n:
            raise ValueError("batch, env_indices, perspective_player_indices, and layouts differ")

        moved = _move_text_batch(batch, self.device) if batch is not None else None
        moved_packed = (
            _move_packed_text_batch(packed_batch, self.device) if packed_batch is not None else None
        )
        h_in, c_in = self.lstm_env_state_inputs(env_indices, perspective_player_indices)
        if moved_packed is not None:
            output, (h_out, c_out) = self.policy.forward_packed(moved_packed, h_in=h_in, c_in=c_in)
        else:
            output, (h_out, c_out) = self.policy(moved, h_in=h_in, c_in=c_in)
        self.scatter_lstm_env_states(env_indices, perspective_player_indices, h_out, c_out)

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
                inline_batch = (
                    moved_packed if moved_packed is not None else cast(TextEncodedBatch, moved)
                )
                inline_may_sample = _sample_inline_may_for_step(
                    output,
                    inline_batch,
                    step_idx=step_idx,
                    deterministic=deterministic,
                )
                if inline_may_sample is None:
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
                    may_sample_t, log_prob, entropy = inline_may_sample
            else:
                log_prob = output.values.new_zeros(())
                entropy = output.values.new_zeros(())
                inline_batch = (
                    moved_packed if moved_packed is not None else cast(TextEncodedBatch, moved)
                )
                inline_choice_index_sample = (
                    _sample_inline_choice_index_for_step(
                        output,
                        inline_batch,
                        step_idx=step_idx,
                        deterministic=deterministic,
                    )
                    if trace_kind == "choice_index"
                    else None
                )
                inline_choice_color_sample = (
                    _sample_inline_choice_index_for_step(
                        output,
                        inline_batch,
                        step_idx=step_idx,
                        deterministic=deterministic,
                    )
                    if trace_kind == "choice_color"
                    else None
                )
                if inline_choice_color_sample is not None:
                    selected_tensors, log_prob, entropy = inline_choice_color_sample
                    per_step_log_prob.append(log_prob)
                    per_step_entropy.append(entropy)
                    per_step_value.append(value)
                    per_step_may_sample.append(may_sample_t)
                    per_step_trace_kind.append(trace_kind)
                    per_step_layout.append(layout)
                    per_step_selected_tensors.append(selected_tensors)
                    continue
                if inline_choice_index_sample is not None:
                    selected_tensors, log_prob, entropy = inline_choice_index_sample
                    per_step_log_prob.append(log_prob)
                    per_step_entropy.append(entropy)
                    per_step_value.append(value)
                    per_step_may_sample.append(may_sample_t)
                    per_step_trace_kind.append(trace_kind)
                    per_step_layout.append(layout)
                    per_step_selected_tensors.append(selected_tensors)
                    continue
                inline_priority_sample = (
                    _sample_inline_priority_for_step(
                        output,
                        inline_batch,
                        layout,
                        step_idx=step_idx,
                        deterministic=deterministic,
                    )
                    if trace_kind == "priority"
                    else None
                )
                if inline_priority_sample is not None:
                    selected_tensors, log_prob, entropy = inline_priority_sample
                    per_step_log_prob.append(log_prob)
                    per_step_entropy.append(entropy)
                    per_step_value.append(value)
                    per_step_may_sample.append(may_sample_t)
                    per_step_trace_kind.append(trace_kind)
                    per_step_layout.append(layout)
                    per_step_selected_tensors.append(selected_tensors)
                    continue
                inline_block_sample = (
                    _sample_inline_blockers_for_step(
                        output,
                        inline_batch,
                        layout,
                        step_idx=step_idx,
                        deterministic=deterministic,
                    )
                    if trace_kind == "blockers"
                    else None
                )
                inline_attacker_sample = (
                    _sample_inline_attackers_for_step(
                        output,
                        inline_batch,
                        layout,
                        step_idx=step_idx,
                        deterministic=deterministic,
                    )
                    if trace_kind == "attackers"
                    else None
                )
                if inline_attacker_sample is not None:
                    selected_tensors, log_prob, entropy = inline_attacker_sample
                    per_step_log_prob.append(log_prob)
                    per_step_entropy.append(entropy)
                    per_step_value.append(value)
                    per_step_may_sample.append(may_sample_t)
                    per_step_trace_kind.append(trace_kind)
                    per_step_layout.append(layout)
                    per_step_selected_tensors.append(selected_tensors)
                    continue
                if inline_block_sample is not None:
                    selected_tensors, log_prob, entropy = inline_block_sample
                    per_step_log_prob.append(log_prob)
                    per_step_entropy.append(entropy)
                    per_step_value.append(value)
                    per_step_may_sample.append(may_sample_t)
                    per_step_trace_kind.append(trace_kind)
                    per_step_layout.append(layout)
                    per_step_selected_tensors.append(selected_tensors)
                    continue
                raise ValueError(
                    "text live sampling requires inline blank metadata for "
                    f"trace_kind={trace_kind!r} at batch row {step_idx}"
                )

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
            selected_log_probs = [0.0 for _ in selected_cols]

            may_selected = 0
            if trace_kind == "may":
                may_val = may_sample_cpu[step_idx]
                assert may_val is not None
                may_selected = int(may_val >= 0.5)
                action = action_from_choice_accepted(bool(may_selected))
                trace = ActionTrace("may", binary=(float(may_selected),))
            else:
                trace, action = _decode_text_action(trace_kind, layout.pending, selected_cols)

            append_kwargs = {
                "batch_index": step_idx,
                "trace_kind_id": TRACE_KIND_TO_ID[trace_kind],
                "decision_option_idx": layout.decision_option_idx,
                "decision_target_idx": layout.decision_target_idx,
                "decision_mask": layout.decision_mask,
                "uses_none_head": layout.uses_none_head,
                "selected_indices": torch.tensor(selected_cols, dtype=torch.long),
                "behavior_action_log_prob": torch.tensor(selected_log_probs, dtype=torch.float32),
                "may_selected": float(may_selected),
                "old_log_prob": float(log_prob_cpu[step_idx]),
                "value": float(value_cpu[step_idx]),
                "perspective_player_idx": int(perspective_player_indices[step_idx]),
                "lstm_h_in": h_in[:, step_idx].detach(),
                "lstm_c_in": c_in[:, step_idx].detach(),
            }
            if packed_batch is not None:
                replay_idx = self.rollout_buffer.append_packed(
                    encoded=packed_batch,
                    **append_kwargs,
                )
            else:
                replay_idx = self.rollout_buffer.append(
                    encoded=cast(TextEncodedBatch, moved),
                    **append_kwargs,
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
                    selected_action_log_probs=tuple(selected_log_probs),
                )
            )
        if (
            results
            and self.rollout_buffer.projected_state is not None
            and output.lstm_input is not None
        ):
            rows = torch.tensor(
                [r.replay_idx for r in results], dtype=torch.long, device=self.device
            )
            self.rollout_buffer.write_projected_state(rows, output.lstm_input.detach())
        return results

    def sample_native_tensor_batch(
        self,
        *,
        native_batch: Any,
        env_indices: list[int],
        perspective_player_indices: list[int],
        text_batch: TextEncodedBatch | None = None,
        packed_batch: PackedTextBatch | None = None,
        deterministic: bool = False,
        append_replay: bool = True,
        return_replay_payload: bool = False,
        profile_timings: dict[str, float] | None = None,
    ) -> NativeTextSampleBatch:
        """Sample a native text batch without Python loops over tensor rows/groups."""

        self._drain_decision_coverage_checks()
        if text_batch is None and packed_batch is None:
            raise ValueError("text_batch or packed_batch is required")
        batch_size = (
            int(packed_batch.seq_lengths.shape[0])
            if packed_batch is not None
            else int(cast(TextEncodedBatch, text_batch).token_ids.shape[0])
        )
        if batch_size == 0:
            return NativeTextSampleBatch([], [], [], [], [], [])
        if len(env_indices) != batch_size or len(perspective_player_indices) != batch_size:
            raise ValueError("batch, env_indices, and perspective_player_indices differ")

        profile_last = time.perf_counter()

        def mark_profile(name: str) -> None:
            nonlocal profile_last
            if profile_timings is None:
                return
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            profile_timings[name] = profile_timings.get(name, 0.0) + (now - profile_last)
            profile_last = now

        moved_text = _move_text_batch(text_batch, self.device) if text_batch is not None else None
        moved_packed = (
            _move_packed_text_batch(packed_batch, self.device) if packed_batch is not None else None
        )
        mark_profile("move_text")
        h_in, c_in = self.lstm_env_state_inputs(env_indices, perspective_player_indices)
        mark_profile("lstm_state_in")
        device_type = self.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            if moved_packed is not None:
                output, (h_out, c_out) = self.policy.forward_packed(
                    moved_packed, h_in=h_in, c_in=c_in
                )
                replay_encoded = moved_packed
            else:
                moved = cast(TextEncodedBatch, moved_text)
                output, (h_out, c_out) = self.policy(moved, h_in=h_in, c_in=c_in)
                replay_encoded = pack_batch(moved)
            mark_profile("forward")
            self.scatter_lstm_env_states(env_indices, perspective_player_indices, h_out, c_out)
            mark_profile("lstm_state_out")

            trace_kind_id = native_batch.trace_kind_id.to(self.device)
            has_may_decision = bool(
                (native_batch.trace_kind_id[:batch_size] == TRACE_KIND_TO_ID["may"]).any().item()
            )
            decision_count = native_batch.decision_count.to(self.device)
            decision_rows = int(native_batch.decision_rows_written)
            max_cached_choices = int(native_batch.decision_mask.shape[1])
            device = output.values.device
            mark_profile("native_metadata_to_device")

            may_logits = self.may_head(output.state_hidden).squeeze(-1)
            step_log_probs = output.values.new_zeros(batch_size)
            step_entropies = output.values.new_zeros(batch_size)
            may_selected_t = output.values.new_zeros(batch_size)
            inline_may_active = torch.zeros(batch_size, dtype=torch.bool, device=device)
            inline_batch = (
                moved_packed if moved_packed is not None else cast(TextEncodedBatch, moved_text)
            )
            mark_profile("decision_init")

            if decision_rows > 0:
                option_idx = native_batch.decision_option_idx[:decision_rows].to(
                    device, non_blocking=device.type == "cuda"
                )
                target_idx = native_batch.decision_target_idx[:decision_rows].to(
                    device, non_blocking=device.type == "cuda"
                )
                decision_mask = native_batch.decision_mask[:decision_rows].to(
                    device, dtype=torch.bool, non_blocking=device.type == "cuda"
                )
                uses_none = torch.empty(0, dtype=torch.bool, device=device)
                mark_profile("decision_tensors_to_device")
                group_starts = decision_count.cumsum(0) - decision_count
                group_steps = torch.repeat_interleave(
                    torch.arange(batch_size, dtype=torch.long, device=device),
                    decision_count.to(device=device, dtype=torch.int32),
                    output_size=decision_rows,
                )
                handled_groups = torch.zeros(decision_rows, dtype=torch.bool, device=device)
                selected = torch.zeros(decision_rows, dtype=torch.long, device=device)
                decision_row_mask = decision_count > 0
                single_group_mask = decision_count == 1
                full_option_idx = torch.full(
                    (batch_size, max_cached_choices),
                    -1,
                    dtype=option_idx.dtype,
                    device=device,
                )
                full_target_idx = torch.full_like(full_option_idx, -1)
                full_decision_mask = torch.zeros(
                    (batch_size, max_cached_choices), dtype=torch.bool, device=device
                )
                row_group_starts = group_starts[decision_row_mask]
                full_option_idx[decision_row_mask] = option_idx[row_group_starts]
                full_target_idx[decision_row_mask] = target_idx[row_group_starts]
                full_decision_mask[decision_row_mask] = decision_mask[row_group_starts]
                mark_profile("decision_full_layout")
                inline_fast = self._sample_inline_decision_batch(
                    output,
                    inline_batch,
                    option_idx=full_option_idx,
                    target_idx=full_target_idx,
                    decision_mask=full_decision_mask,
                    trace_kind_id=trace_kind_id,
                    decision_count=decision_count,
                    decision_rows=batch_size,
                    deterministic=deterministic,
                    profile_timings=profile_timings,
                )
                mark_profile("decision_inline_fast")
                if inline_fast is not None:
                    selected_by_row, log_prob, entropy, active = inline_fast
                    single_active = active & single_group_mask
                    single_flat_rows = group_starts[single_active]
                    if int(single_flat_rows.numel()) > 0:
                        selected[single_flat_rows] = selected_by_row[single_active]
                        handled_groups[single_flat_rows] = True
                    step_log_probs = step_log_probs + torch.where(
                        single_active, log_prob, torch.zeros_like(log_prob)
                    )
                    step_entropies = step_entropies + torch.where(
                        single_active, entropy, torch.zeros_like(entropy)
                    )
                combat_fast = self._sample_inline_combat_groups_batch(
                    output,
                    inline_batch,
                    option_idx=option_idx,
                    decision_mask=decision_mask,
                    trace_kind_id=trace_kind_id,
                    decision_count=decision_count,
                    group_steps=group_steps,
                    deterministic=deterministic,
                )
                if combat_fast is not None:
                    combat_selected, combat_log_prob, combat_entropy, combat_active_groups = (
                        combat_fast
                    )
                    update_groups = combat_active_groups & ~handled_groups
                    selected = torch.where(update_groups, combat_selected, selected)
                    handled_groups = handled_groups | combat_active_groups
                    step_log_probs = step_log_probs + combat_log_prob
                    step_entropies = step_entropies + combat_entropy
                mark_profile("decision_accept_check")
                self._enqueue_decision_coverage_check(
                    handled_groups=handled_groups,
                    group_steps=group_steps,
                    trace_kind_id=trace_kind_id,
                    decision_count=decision_count,
                )
            else:
                option_idx = torch.empty((0, max_cached_choices), dtype=torch.long, device=device)
                target_idx = torch.empty_like(option_idx)
                decision_mask = torch.empty(
                    (0, max_cached_choices), dtype=torch.bool, device=device
                )
                uses_none = torch.empty(0, dtype=torch.bool, device=device)
                selected = torch.empty(0, dtype=torch.long, device=device)
            group_log_probs = output.values.new_zeros(decision_rows)
            needs_replay_metadata = return_replay_payload or (
                append_replay and self.rollout_buffer is not None
            )
            if needs_replay_metadata and int(uses_none.shape[0]) != decision_rows:
                uses_none = native_batch.uses_none_head[:decision_rows].to(
                    device, dtype=torch.bool, non_blocking=device.type == "cuda"
                )
            mark_profile("decision_post_decision")

            may_mask = trace_kind_id == TRACE_KIND_TO_ID["may"]
            if has_may_decision:
                inline_may_sample = _sample_inline_may_batch(
                    output,
                    inline_batch,
                    may_mask=may_mask,
                    deterministic=deterministic,
                )
                if inline_may_sample is not None:
                    selected_may, log_prob, entropy, inline_may_active = inline_may_sample
                    may_selected_t = torch.where(inline_may_active, selected_may, may_selected_t)
                    active_f = inline_may_active.to(dtype=output.values.dtype)
                    step_log_probs = step_log_probs + log_prob * active_f
                    step_entropies = step_entropies + entropy * active_f

                fallback_may_mask = may_mask & ~inline_may_active
                if deterministic:
                    fallback_may_selected = (may_logits >= 0).to(dtype=output.values.dtype)
                else:
                    fallback_may_selected = torch.bernoulli(torch.sigmoid(may_logits))
                may_selected_t = torch.where(
                    fallback_may_mask, fallback_may_selected, may_selected_t
                )
                may_log_prob = -torch.nn.functional.binary_cross_entropy_with_logits(
                    may_logits,
                    fallback_may_selected,
                    reduction="none",
                )
                with torch.autocast(device_type=device_type, enabled=False):
                    may_prob = torch.sigmoid(may_logits.float())
                    may_entropy = torch.nn.functional.binary_cross_entropy(
                        may_prob,
                        may_prob,
                        reduction="none",
                    )
                may_mask_f = fallback_may_mask.to(dtype=output.values.dtype)
                step_log_probs = step_log_probs + may_log_prob * may_mask_f
                step_entropies = step_entropies + may_entropy * may_mask_f
            del step_entropies
            mark_profile("decision_may")
            mark_profile("decision_sampling")

        replay_payload = (
            NativeTextReplayPayload(
                encoded=replay_encoded,
                trace_kind_id=trace_kind_id.detach(),
                decision_count=decision_count.detach(),
                decision_count_host=tuple(int(x) for x in decision_count.detach().cpu().tolist()),
                total_decision_groups=decision_rows,
                total_stored_decision_groups=None,
                decision_option_idx=option_idx.detach(),
                decision_target_idx=target_idx.detach(),
                decision_mask=decision_mask.detach(),
                uses_none_head=uses_none.detach(),
                selected_indices=selected.detach(),
                behavior_action_log_prob=group_log_probs.detach(),
                may_selected=may_selected_t.detach(),
                old_log_prob=step_log_probs.detach(),
                value=output.values.detach(),
                perspective_player_idx=torch.tensor(
                    perspective_player_indices, dtype=torch.long, device=self.device
                ),
                lstm_h_in=h_in.detach(),
                lstm_c_in=c_in.detach(),
                projected_state=output.lstm_input.detach()
                if output.lstm_input is not None
                else None,
            )
            if return_replay_payload
            else None
        )
        mark_profile("replay_payload")

        if append_replay and self.rollout_buffer is not None:
            perspective_t = torch.tensor(
                perspective_player_indices, dtype=torch.long, device=self.device
            )
            replay_rows = self.rollout_buffer.append_batch(
                encoded=replay_encoded,
                trace_kind_id=trace_kind_id,
                decision_count=decision_count,
                decision_count_host=tuple(int(x) for x in decision_count.detach().cpu().tolist()),
                total_decision_groups=decision_rows,
                total_stored_decision_groups=None,
                decision_option_idx=option_idx,
                decision_target_idx=target_idx,
                decision_mask=decision_mask,
                uses_none_head=uses_none,
                selected_indices=selected,
                behavior_action_log_prob=group_log_probs.detach(),
                may_selected=may_selected_t,
                old_log_prob=step_log_probs.detach(),
                value=output.values.detach(),
                perspective_player_idx=perspective_t,
                lstm_h_in=h_in.detach(),
                lstm_c_in=c_in.detach(),
            )
            if self.rollout_buffer.projected_state is not None and output.lstm_input is not None:
                self.rollout_buffer.write_projected_state(replay_rows, output.lstm_input.detach())
        else:
            replay_rows = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        mark_profile("append_replay")

        host = torch.cat(
            [
                decision_count.to(dtype=torch.float32),
                may_selected_t.to(dtype=torch.float32),
                step_log_probs.detach(),
                output.values.detach(),
                replay_rows.to(dtype=torch.float32),
                selected.to(dtype=torch.float32),
            ]
        ).cpu()
        mark_profile("host_return")
        b = batch_size
        decision_counts = host[:b].tolist()
        may_selected = host[b : 2 * b].tolist()
        old_log_prob = host[2 * b : 3 * b].tolist()
        value = host[3 * b : 4 * b].tolist()
        replay_rows_cpu = host[4 * b : 5 * b].tolist()
        selected_choice_cols = host[5 * b :].tolist()
        return NativeTextSampleBatch(
            decision_counts=[int(x) for x in decision_counts],
            selected_choice_cols=[int(x) for x in selected_choice_cols],
            may_selected=[int(x) for x in may_selected],
            old_log_prob=[float(x) for x in old_log_prob],
            value=[float(x) for x in value],
            replay_rows=[int(x) for x in replay_rows_cpu],
            replay_payload=replay_payload,
        )

    def evaluate_replay_batch(
        self,
        replay_rows: list[int] | Tensor,
        *,
        return_extras: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        if return_extras:
            raise ValueError("TextActorCritic does not implement SPR replay extras")
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        batch = self.rollout_buffer.gather(replay_rows)
        device_type = self.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            if batch.lstm_h_in is None or batch.lstm_c_in is None:
                output, _state = self.policy.forward_packed(batch.encoded)
            else:
                h_in = batch.lstm_h_in.permute(1, 0, 2).contiguous()
                c_in = batch.lstm_c_in.permute(1, 0, 2).contiguous()
                output, _state = self.policy.forward_packed(batch.encoded, h_in=h_in, c_in=c_in)

            n = int(batch.trace_kind_id.shape[0])
            log_probs = output.values.new_zeros(n)
            entropies = output.values.new_zeros(n)

            forward = self._replay_scoring_forward(output)
            may_mask = batch.trace_kind_id == TRACE_KIND_TO_ID["may"]
            inline_may_log_probs, inline_may_entropies, inline_may_mask, _, _ = (
                _score_inline_may_decisions(output, batch)
            )
            may_log_probs, may_entropies, _may_logits_per_step, _may_selected_per_step = (
                score_may_decisions_from_forward(
                    forward,
                    may_selected=batch.may_selected,
                    may_mask=may_mask & ~inline_may_mask,
                )
            )
            log_probs = log_probs + may_log_probs + inline_may_log_probs
            entropies = entropies + may_entropies + inline_may_entropies

            decision_log_probs, decision_entropies = cast(
                tuple[Tensor, Tensor],
                self._evaluate_decision_groups(output, batch),
            )
            log_probs = log_probs + decision_log_probs
            entropies = entropies + decision_entropies
        return log_probs.float(), entropies.float(), output.values.float(), None

    def write_ppo_targets(
        self,
        replay_rows: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> None:
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        self.rollout_buffer.write_ppo_targets(replay_rows, old_log_probs, returns, advantages)

    def gather_ppo_targets(self, replay_rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        return self.rollout_buffer.gather_ppo_targets(replay_rows)

    def gather_replay_old_log_prob_value(
        self,
        replay_rows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        idx = replay_rows.to(device=self.rollout_buffer.device)
        return self.rollout_buffer.old_log_prob[idx], self.rollout_buffer.value[idx]

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
                output, _state = self.policy.forward_packed(
                    batch.encoded,
                    state_hidden_override=hidden_override.to(self.device),
                )
            elif lstm_state_override is not None:
                output, _state = self.policy.forward_packed(
                    batch.encoded,
                    h_in=lstm_state_override[0],
                    c_in=lstm_state_override[1],
                )
            elif batch.lstm_h_in is None or batch.lstm_c_in is None:
                output, _state = self.policy.forward_packed(batch.encoded)
            else:
                h_in = batch.lstm_h_in.permute(1, 0, 2).contiguous()
                c_in = batch.lstm_c_in.permute(1, 0, 2).contiguous()
                output, _state = self.policy.forward_packed(batch.encoded, h_in=h_in, c_in=c_in)

        n = int(batch.trace_kind_id.shape[0])
        forward = self._replay_scoring_forward(output)
        log_probs = torch.zeros(n, dtype=torch.float32, device=forward.values.device)
        entropies = torch.zeros(n, dtype=torch.float32, device=forward.values.device)
        may_mask = batch.trace_kind_id == TRACE_KIND_TO_ID["may"]
        (
            inline_may_log_probs,
            inline_may_entropies,
            inline_may_mask,
            inline_may_logits,
            inline_may_selected,
        ) = _score_inline_may_decisions(output, batch)
        may_log_probs, may_entropies, may_logits_per_step, may_selected_per_step = (
            score_may_decisions_from_forward(
                forward,
                may_selected=batch.may_selected,
                may_mask=may_mask & ~inline_may_mask,
            )
        )
        log_probs = log_probs + may_log_probs + inline_may_log_probs
        entropies = entropies + may_entropies + inline_may_entropies
        may_logits_per_step = torch.where(
            inline_may_mask,
            inline_may_logits,
            may_logits_per_step,
        )
        may_selected_per_step = torch.where(
            inline_may_mask,
            inline_may_selected,
            may_selected_per_step,
        )

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
        log_probs = torch.nan_to_num(log_probs.float(), nan=-1.0e9, neginf=-1.0e9, posinf=1.0e9)
        entropies = torch.nan_to_num(entropies.float(), nan=0.0, neginf=0.0, posinf=1.0e9)
        values = torch.nan_to_num(output.values.float(), nan=0.0, neginf=-1.0e9, posinf=1.0e9)
        per_choice = ReplayPerChoice(
            flat_logits=torch.nan_to_num(
                per_choice.flat_logits.float(), nan=0.0, neginf=-1.0e9, posinf=1.0e9
            ),
            flat_log_probs=torch.nan_to_num(
                per_choice.flat_log_probs.float(), nan=-1.0e9, neginf=-1.0e9, posinf=1.0e9
            ),
            group_idx=per_choice.group_idx,
            choice_cols=per_choice.choice_cols,
            is_sampled_flat=per_choice.is_sampled_flat,
            may_is_active=may_mask,
            may_logits_per_step=may_logits_per_step,
            may_selected_per_step=may_selected_per_step,
            decision_group_id_flat=per_choice.decision_group_id_flat,
            step_for_decision_group=per_choice.step_for_decision_group,
            behavior_action_log_prob_per_decision_group=(
                per_choice.behavior_action_log_prob_per_decision_group
            ),
        )
        return log_probs, entropies, values, per_choice

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
        """Run encoder + per-player LSTM recompute for a batch of episodes.

        Replaces the old per-episode sequential scan with a fully batched
        per-player recompute that matches rollout semantics: each player's
        LSTM state advances only on their own turns and resets to zero at
        game start. Gradients flow through the encoder, ``in_proj``, and the
        LSTM so all parameters receive signal from the R-NaD loss.
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
            blank_row_mask = (batch.decision_count > 0) | (
                batch.trace_kind_id == TRACE_KIND_TO_ID["may"]
            )
            encoded = self.policy.text_policy.encode_packed_replay_only(
                batch.encoded,
                blank_row_mask=blank_row_mask,
            )
            state = self.policy.in_proj(encoded.state_vector)
            perspective = batch.perspective_player_idx.to(device=state.device)
            h_concat = lstm_recompute_per_step_h_out_per_player(
                self.policy.lstm,
                state,
                perspective,
                ep_lens,
            )

        return CachedReplayForward(
            flat_rows=tuple(flat_rows),
            batch=batch,
            encoded=encoded,
            h_concat=h_concat,
        )

    @torch.no_grad()
    def refresh_lstm_states(
        self,
        episodes_replay_rows: Sequence[Sequence[int]],
    ) -> None:
        """Refresh ``lstm_h_in`` in the replay buffer using cached projected features.

        Runs the per-player LSTM recompute on the cached ``in_proj`` outputs
        stored at rollout time, then writes fresh per-step ``h_in`` back.
        ``lstm_c_in`` is zeroed because per-step cell states are not captured
        during the fused cuDNN forward. Only supports ``lstm_layers == 1``.

        Safe to call between PPO epochs: runs under ``torch.no_grad()`` and
        only writes to the buffer, so it does not affect the gradient graph.
        """
        if self.rollout_buffer is None or self.rollout_buffer.projected_state is None:
            return
        if self.rollout_buffer.lstm_h_in is None:
            return
        if self.lstm_layers != 1:
            raise ValueError("refresh_lstm_states only supports lstm_layers == 1")

        flat_rows = [int(r) for ep in episodes_replay_rows for r in ep]
        ep_lens = [len(list(ep)) for ep in episodes_replay_rows]
        if not flat_rows:
            return

        rows_t = torch.tensor(flat_rows, dtype=torch.long, device=self.device)
        target_dtype = next(self.policy.lstm.parameters()).dtype

        projected = self.rollout_buffer.projected_state[rows_t].to(
            device=self.device, dtype=target_dtype
        )
        perspective = self.rollout_buffer.perspective_player_idx[rows_t].to(dtype=torch.int32)

        h_out = lstm_recompute_per_step_h_out_per_player(
            self.policy.lstm,
            projected,
            perspective,
            ep_lens,
        )

        # Shift within each player's sub-sequence per episode:
        # h_in[step k] = h_out[step k-1]; h_in[first step] = 0.
        n_eps = len(ep_lens)
        ep_lens_t = torch.tensor(ep_lens, dtype=torch.long, device=self.device)
        ep_ids = torch.repeat_interleave(torch.arange(n_eps, device=self.device), ep_lens_t)
        virt_ids = ep_ids * 2 + perspective
        sort_idx = torch.argsort(virt_ids, stable=True)
        virt_lengths = torch.bincount(virt_ids, minlength=2 * n_eps)

        h_out_sorted = h_out[sort_idx]
        h_in_sorted = h_out_sorted.new_zeros(h_out_sorted.shape)

        seq_starts = torch.cat(
            [
                h_out_sorted.new_zeros(1, dtype=torch.long),
                virt_lengths.cumsum(0)[:-1],
            ]
        )
        is_first = h_out_sorted.new_zeros(len(flat_rows), dtype=torch.bool)
        is_first[seq_starts[virt_lengths > 0]] = True

        h_in_sorted[1:] = torch.where(
            is_first[1:].unsqueeze(-1),
            h_in_sorted.new_zeros(1, self.lstm_hidden),
            h_out_sorted[:-1],
        )

        h_in_flat = h_in_sorted.new_empty(h_in_sorted.shape)
        h_in_flat[sort_idx] = h_in_sorted

        buf_device = self.rollout_buffer.device
        buf_rows = rows_t.to(device=buf_device)
        lstm_h_in = self.rollout_buffer.lstm_h_in
        lstm_c_in = self.rollout_buffer.lstm_c_in
        assert lstm_h_in is not None and lstm_c_in is not None
        lstm_h_in[buf_rows] = h_in_flat.to(device=buf_device, dtype=torch.float32).unsqueeze(1)
        lstm_c_in[buf_rows] = 0.0

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
        self.live_lstm_h[:, slots] = h_out.detach().to(dtype=self.live_lstm_h.dtype)
        self.live_lstm_c[:, slots] = c_out.detach().to(dtype=self.live_lstm_c.dtype)

    def _run_lstm_scan(self, replay_rows: list[int]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the LSTM scan from scratch for the given replay rows.

        Returns (projected [T, lstm_hidden], h_in_seq [layers, T, hidden],
        c_in_seq [layers, T, hidden], out_seq [T, hidden]) where h_in_seq and
        c_in_seq are the per-step hidden inputs and out_seq is the per-step
        top-layer output.
        """
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer has not been set")
        batch = self.rollout_buffer.gather(replay_rows)
        encoded = self.policy.text_policy.encode_packed_only(batch.encoded)
        projected = self.policy.in_proj(encoded.state_vector)
        h = torch.zeros(self.lstm_layers, 1, self.lstm_hidden, device=projected.device)
        c = torch.zeros_like(h)
        h_list: list[Tensor] = []
        c_list: list[Tensor] = []
        out_list: list[Tensor] = []
        for t in range(projected.shape[0]):
            h_list.append(h)
            c_list.append(c)
            out_t, (h_t, c_t) = self.policy.lstm(
                projected[t : t + 1].unsqueeze(0),
                (h.contiguous(), c.contiguous()),
            )
            out_list.append(out_t.squeeze(0).squeeze(0))
            h = h_t
            c = c_t
        h_seq = torch.cat(h_list, dim=1).contiguous()  # (layers, T, hidden)
        c_seq = torch.cat(c_list, dim=1).contiguous()  # (layers, T, hidden)
        out_seq = torch.stack(out_list, dim=0).contiguous()  # (T, hidden)
        return projected, h_seq, c_seq, out_seq

    def _recompute_lstm_states_for_rows(self, replay_rows: list[int]) -> tuple[Tensor, Tensor]:
        _projected, h_seq, c_seq, _out_seq = self._run_lstm_scan(replay_rows)
        return h_seq, c_seq

    def _recompute_lstm_outputs_for_rows(self, replay_rows: list[int]) -> Tensor:
        _projected, _h_seq, _c_seq, out_seq = self._run_lstm_scan(replay_rows)
        return out_seq

    def _evaluate_decision_groups(
        self,
        output: RecurrentTextPolicyOutput,
        batch: TextReplayBatch,
        *,
        return_per_choice: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, ReplayPerChoice]:
        decision_count = batch.decision_count
        n = int(decision_count.shape[0])
        forward = self._replay_scoring_forward(output)
        device = forward.values.device
        log_probs = torch.zeros(n, dtype=torch.float32, device=device)
        entropies = torch.zeros(n, dtype=torch.float32, device=device)

        g_total = int(batch.step_for_decision_group.shape[0])
        if g_total == 0:
            if not return_per_choice:
                return log_probs, entropies
            empty_long = torch.zeros(0, dtype=torch.long, device=device)
            empty_bool = torch.zeros(0, dtype=torch.bool, device=device)
            return (
                log_probs,
                entropies,
                ReplayPerChoice(
                    flat_logits=torch.zeros(0, dtype=torch.float32, device=device),
                    flat_log_probs=torch.zeros(0, dtype=torch.float32, device=device),
                    group_idx=empty_long,
                    choice_cols=empty_long,
                    is_sampled_flat=empty_bool,
                    may_is_active=torch.zeros(n, dtype=torch.bool, device=device),
                    may_logits_per_step=torch.zeros(n, dtype=torch.float32, device=device),
                    may_selected_per_step=torch.zeros(n, dtype=torch.float32, device=device),
                    decision_group_id_flat=empty_long,
                    step_for_decision_group=empty_long,
                    behavior_action_log_prob_per_decision_group=output.values.new_zeros(0),
                ),
            )

        choice_log_probs, choice_entropies, choice_group_mask, choice_per_choice = (
            _evaluate_inline_choice_index_replay_groups(
                output,
                batch,
                return_per_choice=return_per_choice,
                trace_kind_id=TRACE_KIND_TO_ID["choice_index"],
            )
        )
        color_log_probs, color_entropies, color_group_mask, color_per_choice = (
            _evaluate_inline_choice_index_replay_groups(
                output,
                batch,
                return_per_choice=return_per_choice,
                trace_kind_id=TRACE_KIND_TO_ID["choice_color"],
                group_skip_mask=choice_group_mask,
            )
        )
        choice_ids_log_probs, choice_ids_entropies, choice_ids_group_mask, choice_ids_per_choice = (
            _evaluate_inline_choice_index_replay_groups(
                output,
                batch,
                return_per_choice=return_per_choice,
                trace_kind_id=TRACE_KIND_TO_ID["choice_ids"],
                group_skip_mask=choice_group_mask | color_group_mask,
            )
        )
        priority_log_probs, priority_entropies, priority_group_mask, priority_per_choice = (
            _evaluate_inline_priority_replay_groups(
                output,
                batch,
                return_per_choice=return_per_choice,
                group_skip_mask=choice_group_mask | color_group_mask | choice_ids_group_mask,
            )
        )
        blocker_log_probs, blocker_entropies, blocker_group_mask, blocker_per_choice = (
            _evaluate_inline_blocker_replay_groups(
                output,
                batch,
                return_per_choice=return_per_choice,
                group_skip_mask=choice_group_mask | color_group_mask | priority_group_mask,
            )
        )
        attacker_log_probs, attacker_entropies, attacker_group_mask, attacker_per_choice = (
            _evaluate_inline_blocker_replay_groups(
                output,
                batch,
                return_per_choice=return_per_choice,
                trace_kind_id=TRACE_KIND_TO_ID["attackers"],
                blank_group_kind_id=BLANK_GROUP_PER_BLANK,
                group_skip_mask=(
                    choice_group_mask | color_group_mask | priority_group_mask | blocker_group_mask
                ),
            )
        )
        log_probs = (
            log_probs
            + choice_log_probs
            + color_log_probs
            + choice_ids_log_probs
            + priority_log_probs
            + blocker_log_probs
            + attacker_log_probs
        )
        entropies = (
            entropies
            + choice_entropies
            + color_entropies
            + choice_ids_entropies
            + priority_entropies
            + blocker_entropies
            + attacker_entropies
        )

        inline_group_mask = (
            choice_group_mask
            | color_group_mask
            | choice_ids_group_mask
            | priority_group_mask
            | blocker_group_mask
            | attacker_group_mask
        )
        if return_per_choice:
            assert choice_per_choice is not None
            assert color_per_choice is not None
            assert choice_ids_per_choice is not None
            assert priority_per_choice is not None
            assert blocker_per_choice is not None
            assert attacker_per_choice is not None
            inline_per_choice = _concat_replay_per_choice(
                _concat_replay_per_choice(
                    _concat_replay_per_choice(
                        _concat_replay_per_choice(
                            _concat_replay_per_choice(choice_per_choice, color_per_choice),
                            choice_ids_per_choice,
                        ),
                        priority_per_choice,
                    ),
                    blocker_per_choice,
                ),
                attacker_per_choice,
            )
        else:
            inline_per_choice = None
        legacy_group_mask = ~inline_group_mask
        if not bool(legacy_group_mask.any().item()):
            if not return_per_choice:
                return log_probs, entropies
            assert inline_per_choice is not None
            return log_probs, entropies, inline_per_choice

        legacy_group_ids = legacy_group_mask.nonzero(as_tuple=False).squeeze(-1)
        kinds = batch.trace_kind_id[batch.step_for_decision_group[legacy_group_ids]]
        first_group = int(legacy_group_ids[0].detach().cpu().item())
        first_step = int(batch.step_for_decision_group[first_group].detach().cpu().item())
        first_blanks = batch.encoded.blank_positions[first_step] >= 0
        first_blank_indices = first_blanks.nonzero(as_tuple=False).squeeze(-1)
        first_selected = int(batch.selected_indices[first_group].detach().cpu().item())
        first_option_row = batch.decision_option_idx[first_group].detach().cpu().tolist()
        first_target_row = batch.decision_target_idx[first_group].detach().cpu().tolist()
        first_blank_options = (
            batch.encoded.blank_option_index[first_step, first_blank_indices]
            .detach()
            .cpu()
            .tolist()
            if first_blank_indices.numel() > 0
            else []
        )
        first_blank_group_kind = (
            batch.encoded.blank_group_kind[first_step, first_blank_indices].detach().cpu().tolist()
            if first_blank_indices.numel() > 0
            else []
        )
        first_blank_legal_counts = (
            batch.encoded.blank_legal_mask[first_step, first_blank_indices]
            .sum(dim=-1)
            .detach()
            .cpu()
            .tolist()
            if first_blank_indices.numel() > 0
            else []
        )
        raise ValueError(
            "text replay batch contains decision groups without inline blank scoring "
            f"(group_ids={legacy_group_ids.detach().cpu().tolist()}, "
            f"trace_kind_ids={kinds.detach().cpu().tolist()}, "
            f"first_group={first_group}, first_step={first_step}, "
            f"first_selected={first_selected}, first_option_row={first_option_row}, "
            f"first_target_row={first_target_row}, "
            f"first_blank_options={first_blank_options}, "
            f"first_blank_group_kind={first_blank_group_kind}, "
            f"first_blank_legal_counts={first_blank_legal_counts})"
        )

    def _replay_scoring_forward(self, output: RecurrentTextPolicyOutput) -> ReplayScoringForward:
        empty_options = output.values.new_empty((output.values.shape[0], 0, 0))
        empty_targets = output.values.new_empty((output.values.shape[0], 0, 0, 0))
        device_type = output.state_hidden.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            none_logits = self.none_head(output.state_hidden).squeeze(-1)
            may_logits = self.may_head(output.state_hidden).squeeze(-1)
        return ReplayScoringForward(
            values=output.values.float(),
            option_vectors=empty_options,
            target_vectors=empty_targets,
            none_logits=none_logits.float(),
            may_logits=may_logits.float(),
            hidden=output.state_hidden.float(),
        )

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
    options = pending.get("options", [])[:max_options]
    option_count = len(options)
    if trace_kind == "priority":
        priority_candidates = build_priority_candidates(
            pending,
            max_targets_per_option=max_targets_per_option,
        )
    else:
        priority_candidates = []
    if trace_kind == "blockers":
        target_counts_per_option = [
            min(
                len(option.get("valid_targets", [])),
                max_targets_per_option,
                max_cached_choices - 1,
            )
            for option in options
        ]
    else:
        target_counts_per_option = []
    option_rows, target_rows, mask_rows, uses_none = build_decision_layout_rows(
        trace_kind,
        max_cached_choices=max_cached_choices,
        option_count=option_count,
        priority_candidates=priority_candidates,
        target_counts_per_option=target_counts_per_option,
    )
    return _layout(
        trace_kind, pending, option_rows, target_rows, mask_rows, uses_none, max_cached_choices
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
        return ActionTrace("blockers", indices=indices), action_from_inline_block_choices(
            pending,
            list(range(len(selected))),
            selected,
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


def _sample_inline_may_for_step(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    step_idx: int,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor] | None:
    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None
    row_group_kind = batch.blank_group_kind[step_idx].to(device=blank_logits.device)
    row_option_index = batch.blank_option_index[step_idx].to(device=blank_logits.device)
    row_legal_mask = batch.blank_legal_mask[step_idx].to(
        device=blank_logits.device, dtype=torch.bool
    )
    may_support = (
        (row_group_kind == BLANK_GROUP_PER_BLANK)
        & (row_option_index < 0)
        & row_legal_mask[..., :2].all(dim=-1)
    )
    may_blanks = may_support.nonzero(as_tuple=False).squeeze(-1)
    if may_blanks.numel() != 1:
        return None
    logits = blank_logits[step_idx, may_blanks[0], :2]
    dist = Categorical(logits=logits)
    selected = torch.argmax(logits) if deterministic else dist.sample()
    may_selected = selected.to(dtype=output.values.dtype)
    return may_selected, dist.log_prob(selected), dist.entropy()


def _sample_inline_may_batch(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    may_mask: Tensor,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None
    device = blank_logits.device
    row_group_kind = batch.blank_group_kind.to(device=device)
    row_option_index = batch.blank_option_index.to(device=device)
    row_legal_mask = batch.blank_legal_mask.to(device=device, dtype=torch.bool)
    if int(row_group_kind.shape[1]) == 0 or int(row_legal_mask.shape[2]) < 2:
        return None
    support = (
        (row_group_kind == BLANK_GROUP_PER_BLANK)
        & (row_option_index < 0)
        & row_legal_mask[..., :2].all(dim=-1)
        & may_mask.to(device=device, dtype=torch.bool).unsqueeze(1)
    )
    active = support.sum(dim=1) == 1
    blank_idx = support.to(dtype=torch.int32).argmax(dim=1)
    row_idx = torch.arange(blank_logits.shape[0], device=device)
    logits = blank_logits[row_idx, blank_idx, :2]
    logits = torch.where(active.unsqueeze(1), logits, torch.zeros_like(logits))
    selected = (
        torch.argmax(logits, dim=-1)
        if deterministic
        else (logits - torch.empty_like(logits).exponential_().log()).argmax(dim=-1)
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    selected_may = selected.to(dtype=output.values.dtype)
    log_prob = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
    entropy = torch.zeros_like(log_prob)
    return selected_may, log_prob, entropy, active


def _sample_inline_choice_index_for_step(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    step_idx: int,
    deterministic: bool,
) -> tuple[list[Tensor], Tensor, Tensor] | None:
    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None
    row_group_kind = batch.blank_group_kind[step_idx].to(device=blank_logits.device)
    row_positions = (
        batch.blank_positions[step_idx].to(device=blank_logits.device)
        if batch.blank_positions.numel() > 0
        else torch.zeros_like(row_group_kind)
    )
    row_option_index = batch.blank_option_index[step_idx].to(device=blank_logits.device)
    row_legal_mask = batch.blank_legal_mask[step_idx].to(
        device=blank_logits.device, dtype=torch.bool
    )
    support = (
        (row_positions >= 0)
        & (row_group_kind == BLANK_GROUP_PER_BLANK)
        & (row_option_index < 0)
        & row_legal_mask.any(dim=-1)
    )
    blanks = support.nonzero(as_tuple=False).squeeze(-1)
    if blanks.numel() != 1:
        return None
    blank_idx = blanks[0]
    legal_slots = row_legal_mask[blank_idx].nonzero(as_tuple=False).squeeze(-1)
    if legal_slots.numel() == 0:
        return None
    logits = blank_logits[step_idx, blank_idx, legal_slots]
    dist = Categorical(logits=logits)
    chosen_in_legal = torch.argmax(logits) if deterministic else dist.sample()
    return (
        [legal_slots[chosen_in_legal]],
        dist.log_prob(chosen_in_legal),
        dist.entropy(),
    )


def _score_inline_may_decisions(
    output: RecurrentTextPolicyOutput,
    batch: TextReplayBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    n = int(batch.trace_kind_id.shape[0])
    log_probs = output.values.new_zeros(n)
    entropies = output.values.new_zeros(n)
    logits_per_step = output.values.new_zeros(n)
    selected_per_step = output.values.new_zeros(n)
    device = output.values.device
    active = torch.zeros(n, dtype=torch.bool, device=device)

    blank_logits = output.blank_logits
    if (
        n == 0
        or blank_logits is None
        or blank_logits.numel() == 0
        or batch.encoded.blank_option_index.numel() == 0
    ):
        return log_probs, entropies, active, logits_per_step, selected_per_step

    blank_group_kind = batch.encoded.blank_group_kind.to(device=device)
    blank_option_index = batch.encoded.blank_option_index.to(device=device)
    blank_legal_mask = batch.encoded.blank_legal_mask.to(device=device, dtype=torch.bool)
    if int(blank_group_kind.shape[1]) == 0 or int(blank_legal_mask.shape[2]) < 2:
        return log_probs, entropies, active, logits_per_step, selected_per_step

    may_support = (
        (blank_group_kind == BLANK_GROUP_PER_BLANK)
        & (blank_option_index < 0)
        & blank_legal_mask[..., :2].all(dim=-1)
    )
    match_count = may_support.sum(dim=-1)
    blank_idx = may_support.to(dtype=torch.int32).argmax(dim=-1)
    selected = batch.may_selected.to(device=device, dtype=torch.int32)
    selected_in_range = (selected >= 0) & (selected < 2)
    active = (
        (batch.trace_kind_id.to(device=device) == TRACE_KIND_TO_ID["may"])
        & (match_count == 1)
        & selected_in_range
    )
    row_logits = blank_logits[torch.arange(n, dtype=torch.long, device=device), blank_idx, :2]
    log_probs_dense = torch.log_softmax(row_logits, dim=-1)
    probs = log_probs_dense.exp()
    entropy_dense = -(probs * log_probs_dense).sum(dim=-1)
    safe_selected = selected.clamp(min=0, max=1)
    selected_log_prob = log_probs_dense.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    log_probs = torch.where(active, selected_log_prob, log_probs)
    entropies = torch.where(active, entropy_dense, entropies)
    logits_per_step = torch.where(active, row_logits[:, 1] - row_logits[:, 0], logits_per_step)
    selected_per_step = torch.where(
        active, selected.to(dtype=selected_per_step.dtype), selected_per_step
    )
    return log_probs, entropies, active, logits_per_step, selected_per_step


def _sample_inline_decision_batch(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    option_idx: Tensor,
    target_idx: Tensor,
    decision_mask: Tensor,
    trace_kind_id: Tensor,
    decision_count: Tensor,
    decision_rows: int,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    priority = _sample_inline_priority_batch(
        output,
        batch,
        option_idx=option_idx,
        target_idx=target_idx,
        decision_mask=decision_mask,
        trace_kind_id=trace_kind_id,
        decision_count=decision_count,
        decision_rows=decision_rows,
        deterministic=deterministic,
    )
    choice = _sample_inline_choice_batch(
        output,
        batch,
        decision_mask=decision_mask,
        trace_kind_id=trace_kind_id,
        decision_count=decision_count,
        decision_rows=decision_rows,
        deterministic=deterministic,
    )
    attacker = _sample_inline_per_blank_binary_batch(
        output,
        batch,
        option_idx=option_idx,
        decision_mask=decision_mask,
        trace_kind_id=trace_kind_id,
        decision_count=decision_count,
        decision_rows=decision_rows,
        deterministic=deterministic,
    )
    outputs = [sample for sample in (priority, choice, attacker) if sample is not None]
    if not outputs:
        return None
    selected, log_prob, entropy, active = outputs[0]
    for next_selected, next_log_prob, next_entropy, next_active in outputs[1:]:
        selected = torch.where(next_active, next_selected, selected)
        log_prob = torch.where(next_active, next_log_prob, log_prob)
        entropy = torch.where(next_active, next_entropy, entropy)
        active = active | next_active
    return selected, log_prob, entropy, active


def _sample_inline_decision_batch_profiled(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    option_idx: Tensor,
    target_idx: Tensor,
    decision_mask: Tensor,
    trace_kind_id: Tensor,
    decision_count: Tensor,
    decision_rows: int,
    deterministic: bool,
    profile_timings: dict[str, float],
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    profile_last = time.perf_counter()

    def mark_profile(name: str, device: torch.device) -> None:
        nonlocal profile_last
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        profile_timings[name] = profile_timings.get(name, 0.0) + (now - profile_last)
        profile_last = now

    device = output.values.device
    priority = _sample_inline_priority_batch(
        output,
        batch,
        option_idx=option_idx,
        target_idx=target_idx,
        decision_mask=decision_mask,
        trace_kind_id=trace_kind_id,
        decision_count=decision_count,
        decision_rows=decision_rows,
        deterministic=deterministic,
    )
    mark_profile("decision_priority_batch", device)
    choice = _sample_inline_choice_batch(
        output,
        batch,
        decision_mask=decision_mask,
        trace_kind_id=trace_kind_id,
        decision_count=decision_count,
        decision_rows=decision_rows,
        deterministic=deterministic,
    )
    mark_profile("decision_choice_batch", device)
    attacker = _sample_inline_per_blank_binary_batch(
        output,
        batch,
        option_idx=option_idx,
        decision_mask=decision_mask,
        trace_kind_id=trace_kind_id,
        decision_count=decision_count,
        decision_rows=decision_rows,
        deterministic=deterministic,
    )
    mark_profile("decision_binary_batch", device)
    outputs = [sample for sample in (priority, choice, attacker) if sample is not None]
    if not outputs:
        return None
    selected, log_prob, entropy, active = outputs[0]
    for next_selected, next_log_prob, next_entropy, next_active in outputs[1:]:
        selected = torch.where(next_active, next_selected, selected)
        log_prob = torch.where(next_active, next_log_prob, log_prob)
        entropy = torch.where(next_active, next_entropy, entropy)
        active = active | next_active
    mark_profile("decision_merge_batch", device)
    return selected, log_prob, entropy, active


def _sample_inline_choice_batch(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    decision_mask: Tensor,
    trace_kind_id: Tensor,
    decision_count: Tensor,
    decision_rows: int,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None
    n = int(trace_kind_id.shape[0])
    if decision_rows != n:
        return None
    device = blank_logits.device
    row_positions = batch.blank_positions.to(device=device)
    row_group_kind = batch.blank_group_kind.to(device=device)
    row_option_index = batch.blank_option_index.to(device=device)
    row_legal_mask = batch.blank_legal_mask.to(device=device, dtype=torch.bool)

    trace_kind = trace_kind_id.to(device=device)
    choice_trace = (
        (trace_kind == TRACE_KIND_TO_ID["choice_index"])
        | (trace_kind == TRACE_KIND_TO_ID["choice_ids"])
        | (trace_kind == TRACE_KIND_TO_ID["choice_color"])
    )
    eligible = choice_trace & (decision_count.to(device=device) == 1)
    support = (
        (row_positions >= 0)
        & (row_group_kind == BLANK_GROUP_PER_BLANK)
        & (row_option_index < 0)
        & row_legal_mask.any(dim=-1)
    )
    match_count = support.sum(dim=-1)
    blank_idx = support.to(dtype=torch.int32).argmax(dim=-1)
    row_idx = torch.arange(n, dtype=torch.long, device=device)
    legal_mask = row_legal_mask[row_idx, blank_idx]
    logits = blank_logits[row_idx, blank_idx]
    dummy_mask = torch.zeros_like(legal_mask)
    dummy_mask[:, 0] = True
    active = eligible & (match_count == 1)
    effective_mask = torch.where(active.unsqueeze(1), legal_mask, dummy_mask)
    safe_logits = torch.where(active.unsqueeze(1), logits, torch.zeros_like(logits))
    masked_logits = safe_logits.masked_fill(~effective_mask, float("-inf"))
    chosen = (
        torch.argmax(masked_logits, dim=-1)
        if deterministic
        else (masked_logits - torch.empty_like(masked_logits).exponential_().log()).argmax(dim=-1)
    )
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    log_prob = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
    decision_mask = decision_mask.to(device=device, dtype=torch.bool)
    choice_width = decision_mask.shape[1]
    selected_in_range = chosen < choice_width
    max_choice = torch.full_like(chosen, choice_width - 1)
    safe_selected = torch.minimum(chosen, max_choice)
    selected_valid = decision_mask.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    active = active & selected_in_range & selected_valid
    entropy = torch.zeros_like(log_prob)
    return chosen, log_prob, entropy, active


def _sample_inline_per_blank_binary_batch(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    option_idx: Tensor,
    decision_mask: Tensor,
    trace_kind_id: Tensor,
    decision_count: Tensor,
    decision_rows: int,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None
    n = int(trace_kind_id.shape[0])
    if decision_rows != n:
        return None
    device = blank_logits.device
    row_group_kind = batch.blank_group_kind.to(device=device)
    row_option_index = batch.blank_option_index.to(device=device)
    row_legal_mask = batch.blank_legal_mask.to(device=device, dtype=torch.bool)
    row_live = (
        batch.blank_positions.to(device=device) >= 0
        if batch.blank_positions.numel() > 0
        else torch.ones_like(row_group_kind, dtype=torch.bool)
    )

    trace_kind = trace_kind_id.to(device=device)
    binary_trace = (trace_kind == TRACE_KIND_TO_ID["attackers"]) | (
        trace_kind == TRACE_KIND_TO_ID["blockers"]
    )
    eligible = binary_trace & (decision_count.to(device=device) == 1)
    option_idx = option_idx.to(device=device, dtype=torch.int32)
    decision_mask = decision_mask.to(device=device, dtype=torch.bool)
    valid_option = torch.where(option_idx >= 0, option_idx, torch.full_like(option_idx, 2**30))
    first_option = valid_option.min(dim=-1).values
    has_option = first_option < 2**30
    desired_option = torch.where(
        (trace_kind == TRACE_KIND_TO_ID["blockers"]) & ~has_option,
        torch.zeros_like(first_option),
        first_option,
    )
    support = (
        row_live
        & (row_group_kind == BLANK_GROUP_PER_BLANK)
        & (row_option_index == desired_option.unsqueeze(1))
        & row_legal_mask.any(dim=-1)
    )
    match_count = support.sum(dim=-1)
    blank_idx = support.to(dtype=torch.int32).argmax(dim=-1)
    row_idx = torch.arange(n, dtype=torch.long, device=device)
    legal_mask = row_legal_mask[row_idx, blank_idx]
    choice_width = decision_mask.shape[1]
    legal_slots = torch.arange(legal_mask.shape[1], dtype=torch.long, device=device)
    choice_mask = decision_mask.gather(1, legal_slots.unsqueeze(0).expand(n, -1))
    valid_legal_mask = legal_mask & choice_mask
    legal_count = valid_legal_mask.sum(dim=-1)
    legal_enough = torch.where(
        trace_kind == TRACE_KIND_TO_ID["attackers"], legal_count >= 2, legal_count >= 1
    )
    active = (
        eligible
        & (has_option | (trace_kind == TRACE_KIND_TO_ID["blockers"]))
        & (match_count == 1)
        & legal_enough
    )
    logits = blank_logits[row_idx, blank_idx]
    dummy_mask = torch.zeros_like(valid_legal_mask)
    dummy_mask[:, 0] = True
    effective_mask = torch.where(active.unsqueeze(1), valid_legal_mask, dummy_mask)
    safe_logits = torch.where(active.unsqueeze(1), logits, torch.zeros_like(logits))
    masked_logits = safe_logits.masked_fill(~effective_mask, float("-inf"))
    chosen = (
        torch.argmax(masked_logits, dim=-1)
        if deterministic
        else (masked_logits - torch.empty_like(masked_logits).exponential_().log()).argmax(dim=-1)
    )
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    log_prob = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
    selected_in_range = chosen < choice_width
    max_choice = torch.full_like(chosen, choice_width - 1)
    safe_selected = torch.minimum(chosen, max_choice)
    selected_valid = decision_mask.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    active = active & selected_in_range & selected_valid
    entropy = torch.zeros_like(log_prob)
    return chosen, log_prob, entropy, active


def _sample_inline_combat_groups_batch(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    option_idx: Tensor,
    decision_mask: Tensor,
    trace_kind_id: Tensor,
    decision_count: Tensor,
    group_steps: Tensor,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    """Vectorized attackers/blockers sampler for multi-group combat rows."""

    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None
    decision_rows = int(option_idx.shape[0])
    if decision_rows == 0:
        return None
    device = blank_logits.device
    option_idx = option_idx.to(device=device, dtype=torch.int32)
    decision_mask = decision_mask.to(device=device, dtype=torch.bool)
    decision_count = decision_count.to(device=device)
    trace_kind = trace_kind_id.to(device=device)
    group_steps = group_steps.to(device=device)
    if int(group_steps.shape[0]) != decision_rows:
        return None

    row_group_kind = batch.blank_group_kind.to(device=device)
    row_option_index = batch.blank_option_index.to(device=device)
    row_legal_mask = batch.blank_legal_mask.to(device=device, dtype=torch.bool)

    group_trace = trace_kind[group_steps]
    is_attacker = group_trace == TRACE_KIND_TO_ID["attackers"]
    is_blocker = group_trace == TRACE_KIND_TO_ID["blockers"]
    combat_group = (is_attacker | is_blocker) & (decision_count[group_steps] > 1)
    valid_option = torch.where(option_idx >= 0, option_idx, torch.full_like(option_idx, 2**30))
    first_option = valid_option.min(dim=-1).values
    has_option = first_option < 2**30
    local_group = _local_decision_group_indices(group_steps)
    desired_option = torch.where(is_blocker & ~has_option, local_group, first_option)
    candidate_group = combat_group & (has_option | is_blocker)

    group_blank_kind = row_group_kind[group_steps]
    group_blank_options = row_option_index[group_steps]
    group_legal_mask = row_legal_mask[group_steps]
    support = (
        candidate_group.unsqueeze(1)
        & (group_blank_kind == BLANK_GROUP_PER_BLANK)
        & (group_blank_options == desired_option.unsqueeze(1))
        & group_legal_mask.any(dim=-1)
    )
    match_count = support.sum(dim=-1)
    blank_idx = support.to(dtype=torch.int32).argmax(dim=-1)
    row_idx = torch.arange(decision_rows, dtype=torch.long, device=device)
    legal_mask = group_legal_mask[row_idx, blank_idx]
    legal_count = legal_mask.sum(dim=-1)
    legal_enough = torch.where(is_attacker, legal_count >= 2, legal_count >= 1)
    active_candidate = candidate_group & (match_count == 1) & legal_enough

    logits = blank_logits[group_steps, blank_idx]
    dummy_mask = torch.zeros_like(legal_mask)
    dummy_mask[:, 0] = True
    effective_mask = torch.where(active_candidate.unsqueeze(1), legal_mask, dummy_mask)
    safe_logits = torch.where(active_candidate.unsqueeze(1), logits, torch.zeros_like(logits))
    masked_logits = safe_logits.masked_fill(~effective_mask, float("-inf"))
    chosen = (
        torch.argmax(masked_logits, dim=-1)
        if deterministic
        else (masked_logits - torch.empty_like(masked_logits).exponential_().log()).argmax(dim=-1)
    )
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    group_log_prob = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)

    choice_width = decision_mask.shape[1]
    selected_in_range = chosen < choice_width
    max_choice = torch.full_like(chosen, choice_width - 1)
    safe_selected = torch.minimum(chosen, max_choice)
    selected_valid = decision_mask.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    active_group = active_candidate & selected_in_range & selected_valid
    selected = chosen

    step_log_prob = output.values.new_zeros(int(trace_kind.shape[0]))
    group_log_prob = group_log_prob.to(dtype=step_log_prob.dtype)
    step_log_prob.scatter_add_(
        0,
        group_steps,
        torch.where(active_group, group_log_prob, torch.zeros_like(group_log_prob)),
    )
    step_entropy = torch.zeros_like(step_log_prob)
    return selected, step_log_prob, step_entropy, active_group


def _sample_inline_priority_batch(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    *,
    option_idx: Tensor,
    target_idx: Tensor,
    decision_mask: Tensor,
    trace_kind_id: Tensor,
    decision_count: Tensor,
    decision_rows: int,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    """Vectorized priority sampler for the hot one-decision-group-per-row path."""

    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None
    n = int(trace_kind_id.shape[0])
    if decision_rows != n or int(option_idx.shape[0]) != n:
        return None
    device = blank_logits.device
    row_group_kind = batch.blank_group_kind.to(device=device)
    row_option_index = batch.blank_option_index.to(device=device)
    row_legal_mask = batch.blank_legal_mask.to(device=device, dtype=torch.bool)

    eligible = (trace_kind_id.to(device=device) == TRACE_KIND_TO_ID["priority"]) & (
        decision_count.to(device=device) == 1
    )
    priority_support = (
        (row_group_kind == BLANK_GROUP_CROSS_BLANK)
        & (row_option_index >= 0)
        & row_legal_mask.any(dim=-1)
    )
    priority_scores = blank_logits[:, :, 0]
    dummy_priority = torch.zeros_like(priority_support)
    dummy_priority[:, 0] = True
    has_priority = priority_support.any(dim=-1)
    effective_priority = torch.where(
        (eligible & has_priority).unsqueeze(1), priority_support, dummy_priority
    )
    safe_priority_scores = torch.where(
        (eligible & has_priority).unsqueeze(1),
        priority_scores,
        torch.zeros_like(priority_scores),
    )
    priority_logits = safe_priority_scores.masked_fill(~effective_priority, float("-inf"))
    chosen_priority = (
        torch.argmax(priority_logits, dim=-1)
        if deterministic
        else (priority_logits - torch.empty_like(priority_logits).exponential_().log()).argmax(
            dim=-1
        )
    )
    priority_log_probs = torch.log_softmax(priority_logits, dim=-1)
    chosen_option_idx = row_option_index.gather(1, chosen_priority.unsqueeze(1)).squeeze(1)
    priority_log_prob = priority_log_probs.gather(1, chosen_priority.unsqueeze(1)).squeeze(1)

    option_idx = option_idx.to(device=device, dtype=torch.int32)
    target_idx = target_idx.to(device=device, dtype=torch.int32)
    decision_mask = decision_mask.to(device=device, dtype=torch.bool)
    candidate_mask = decision_mask & (option_idx == chosen_option_idx.unsqueeze(1))
    has_candidate = candidate_mask.any(dim=-1)
    target_required_mask = candidate_mask & (target_idx >= 0)
    target_required = target_required_mask.any(dim=-1)

    no_target_col = candidate_mask.to(dtype=torch.int32).argmax(dim=-1)
    selected = no_target_col
    log_prob = priority_log_prob
    entropy = torch.zeros_like(priority_log_prob)

    target_support = (
        (row_group_kind == BLANK_GROUP_PER_BLANK)
        & (row_option_index == chosen_option_idx.unsqueeze(1))
        & row_legal_mask.any(dim=-1)
    )
    target_count = target_support.sum(dim=-1)
    target_blank = target_support.to(dtype=torch.int32).argmax(dim=-1)
    row_target_mask = row_legal_mask[torch.arange(n, dtype=torch.long, device=device), target_blank]

    row_target_logits = blank_logits[
        torch.arange(n, dtype=torch.long, device=device),
        target_blank,
    ]
    dummy_target = torch.zeros_like(row_target_mask)
    dummy_target[:, 0] = True
    target_valid = target_required & (target_count == 1)
    effective_target = torch.where(target_valid.unsqueeze(1), row_target_mask, dummy_target)
    safe_target_logits = torch.where(
        target_valid.unsqueeze(1), row_target_logits, torch.zeros_like(row_target_logits)
    )
    target_logits = safe_target_logits.masked_fill(~effective_target, float("-inf"))
    chosen_in_legal = (
        torch.argmax(target_logits, dim=-1)
        if deterministic
        else (target_logits - torch.empty_like(target_logits).exponential_().log()).argmax(dim=-1)
    )
    target_log_probs = torch.log_softmax(target_logits, dim=-1)
    target_log_prob = target_log_probs.gather(1, chosen_in_legal.unsqueeze(1)).squeeze(1)

    target_col_mask = target_required_mask & (target_idx == chosen_in_legal.unsqueeze(1))
    target_col_count = target_col_mask.sum(dim=-1)
    target_col = target_col_mask.to(dtype=torch.int32).argmax(dim=-1)
    selected = torch.where(target_required, target_col, selected)
    log_prob = torch.where(
        target_required,
        log_prob + target_log_prob,
        log_prob,
    )
    active = (
        eligible
        & has_priority
        & has_candidate
        & ((~target_required) | (target_valid & (target_col_count == 1)))
    )
    return selected, log_prob, entropy, active


def _sample_inline_priority_for_step(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    layout: TextDecisionLayout,
    *,
    step_idx: int,
    deterministic: bool,
) -> tuple[list[Tensor], Tensor, Tensor] | None:
    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None
    if layout.decision_option_idx.shape[0] != 1:
        return None

    row_group_kind = batch.blank_group_kind[step_idx].to(device=blank_logits.device)
    row_option_index = batch.blank_option_index[step_idx].to(device=blank_logits.device)
    row_legal_mask = batch.blank_legal_mask[step_idx].to(
        device=blank_logits.device, dtype=torch.bool
    )
    row_logits = blank_logits[step_idx]

    priority_support = (
        (row_group_kind == BLANK_GROUP_CROSS_BLANK)
        & (row_option_index >= 0)
        & row_legal_mask.any(dim=-1)
    )
    priority_blanks = priority_support.nonzero(as_tuple=False).squeeze(-1)
    if priority_blanks.numel() == 0:
        return None

    priority_logits = row_logits[priority_blanks, 0]
    priority_dist = Categorical(logits=priority_logits)
    chosen_priority = torch.argmax(priority_logits) if deterministic else priority_dist.sample()
    chosen_blank = priority_blanks[chosen_priority]
    chosen_option_idx = row_option_index[chosen_blank]
    log_prob = priority_dist.log_prob(chosen_priority)
    entropy = priority_dist.entropy()

    option_row = layout.decision_option_idx[0].to(device=blank_logits.device)
    target_row = layout.decision_target_idx[0].to(device=blank_logits.device)
    mask_row = layout.decision_mask[0].to(device=blank_logits.device, dtype=torch.bool)
    candidate_mask = mask_row & (option_row == chosen_option_idx)
    if not bool(candidate_mask.any().item()):
        return None

    target_required = target_row >= 0
    target_candidate_mask = candidate_mask & target_required
    if not bool(target_candidate_mask.any().item()):
        cols = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
        return [cols[0]], log_prob, entropy

    target_support = (
        (row_group_kind == BLANK_GROUP_PER_BLANK)
        & (row_option_index == chosen_option_idx)
        & row_legal_mask.any(dim=-1)
    )
    target_blanks = target_support.nonzero(as_tuple=False).squeeze(-1)
    if target_blanks.numel() != 1:
        return None
    target_blank = target_blanks[0]
    legal_slots = row_legal_mask[target_blank].nonzero(as_tuple=False).squeeze(-1)
    if legal_slots.numel() == 0:
        return None
    target_logits = row_logits[target_blank, legal_slots]
    target_dist = Categorical(logits=target_logits)
    chosen_in_legal = torch.argmax(target_logits) if deterministic else target_dist.sample()
    chosen_legal_slot = legal_slots[chosen_in_legal]
    log_prob = log_prob + target_dist.log_prob(chosen_in_legal)
    entropy = entropy + target_dist.entropy()

    chosen_col_mask = target_candidate_mask & (target_row == chosen_legal_slot)
    chosen_cols = chosen_col_mask.nonzero(as_tuple=False).squeeze(-1)
    if chosen_cols.numel() != 1:
        return None
    return [chosen_cols[0]], log_prob, entropy


def _sample_inline_blockers_for_step(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    layout: TextDecisionLayout,
    *,
    step_idx: int,
    deterministic: bool,
) -> tuple[list[Tensor], Tensor, Tensor] | None:
    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None

    row_group_kind = batch.blank_group_kind[step_idx].to(device=blank_logits.device)
    row_option_index = batch.blank_option_index[step_idx].to(device=blank_logits.device)
    row_legal_mask = batch.blank_legal_mask[step_idx].to(
        device=blank_logits.device, dtype=torch.bool
    )
    row_logits = blank_logits[step_idx]
    blank_support = (row_group_kind == BLANK_GROUP_PER_BLANK) & (row_option_index >= 0)

    group_option_indices: list[int] = []
    for group_idx in range(int(layout.decision_option_idx.shape[0])):
        row = layout.decision_option_idx[group_idx]
        valid = row[row >= 0]
        if int(valid.numel()) == 0:
            group_option_indices.append(-1)
        else:
            group_option_indices.append(int(valid[0].item()))

    selected: list[Tensor] = []
    log_prob = output.values.new_zeros(())
    entropy = output.values.new_zeros(())
    for group_idx, option_idx in enumerate(group_option_indices):
        if option_idx < 0:
            option_idx = group_idx
        matches = (row_option_index == option_idx) & blank_support
        match_idx = matches.nonzero(as_tuple=False).squeeze(-1)
        if int(match_idx.numel()) != 1:
            return None
        blank_idx = match_idx[0]
        legal_slots = row_legal_mask[blank_idx].nonzero(as_tuple=False).squeeze(-1)
        if legal_slots.numel() == 0:
            return None
        logits = row_logits[blank_idx, legal_slots]
        dist = Categorical(logits=logits)
        chosen_in_legal = torch.argmax(logits) if deterministic else dist.sample()
        log_prob = log_prob + dist.log_prob(chosen_in_legal)
        entropy = entropy + dist.entropy()
        selected.append(legal_slots[chosen_in_legal])
    return selected, log_prob, entropy


def _sample_inline_attackers_for_step(
    output: RecurrentTextPolicyOutput,
    batch: TextEncodedBatch | PackedTextBatch,
    layout: TextDecisionLayout,
    *,
    step_idx: int,
    deterministic: bool,
) -> tuple[list[Tensor], Tensor, Tensor] | None:
    blank_logits = output.blank_logits
    if blank_logits is None or blank_logits.numel() == 0:
        return None
    if batch.blank_option_index.numel() == 0:
        return None

    row_group_kind = batch.blank_group_kind[step_idx].to(device=blank_logits.device)
    row_option_index = batch.blank_option_index[step_idx].to(device=blank_logits.device)
    row_legal_mask = batch.blank_legal_mask[step_idx].to(
        device=blank_logits.device, dtype=torch.bool
    )
    row_logits = blank_logits[step_idx]
    blank_support = (row_group_kind == BLANK_GROUP_PER_BLANK) & (row_option_index >= 0)

    selected: list[Tensor] = []
    log_prob = output.values.new_zeros(())
    entropy = output.values.new_zeros(())
    for group_idx in range(int(layout.decision_option_idx.shape[0])):
        row = layout.decision_option_idx[group_idx]
        valid = row[row >= 0]
        if int(valid.numel()) == 0:
            return None
        option_idx = int(valid[0].item())
        matches = (row_option_index == option_idx) & blank_support
        match_idx = matches.nonzero(as_tuple=False).squeeze(-1)
        if int(match_idx.numel()) != 1:
            return None
        blank_idx = match_idx[0]
        legal_slots = row_legal_mask[blank_idx].nonzero(as_tuple=False).squeeze(-1)
        if legal_slots.numel() < 2:
            return None
        logits = row_logits[blank_idx, legal_slots]
        dist = Categorical(logits=logits)
        chosen_in_legal = torch.argmax(logits) if deterministic else dist.sample()
        log_prob = log_prob + dist.log_prob(chosen_in_legal)
        entropy = entropy + dist.entropy()
        selected.append(legal_slots[chosen_in_legal])
    return selected, log_prob, entropy


def _local_decision_group_indices(steps_t: Tensor) -> Tensor:
    """Return each flattened decision group's ordinal within its source step."""

    total = int(steps_t.numel())
    if total == 0:
        return steps_t
    positions = torch.arange(total, dtype=torch.long, device=steps_t.device)
    segment_start = torch.ones(total, dtype=torch.bool, device=steps_t.device)
    segment_start[1:] = steps_t[1:] != steps_t[:-1]
    start_positions = torch.where(segment_start, positions, torch.zeros_like(positions))
    starts = torch.cummax(start_positions, dim=0).values
    return positions - starts


def _evaluate_inline_priority_replay_groups(
    output: RecurrentTextPolicyOutput,
    batch: TextReplayBatch,
    *,
    return_per_choice: bool,
    group_skip_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, ReplayPerChoice | None]:
    decision_count = batch.decision_count
    n = int(decision_count.shape[0])
    g_total = int(batch.step_for_decision_group.shape[0])
    device = output.values.device
    log_probs = torch.zeros(n, dtype=torch.float32, device=device)
    entropies = torch.zeros(n, dtype=torch.float32, device=device)
    group_mask = torch.zeros(g_total, dtype=torch.bool, device=device)

    blank_logits = output.blank_logits
    if (
        g_total == 0
        or blank_logits is None
        or blank_logits.numel() == 0
        or batch.encoded.blank_option_index.numel() == 0
    ):
        per_choice = _empty_replay_per_choice(output, n, device) if return_per_choice else None
        return log_probs, entropies, group_mask, per_choice

    steps_t = batch.step_for_decision_group.to(device=device)
    option_idx = batch.decision_option_idx.to(device=device, dtype=torch.int32)
    target_idx = batch.decision_target_idx.to(device=device, dtype=torch.int32)
    decision_mask = batch.decision_mask.to(device=device, dtype=torch.bool)
    selected = batch.selected_indices.to(device=device, dtype=torch.int32)
    trace_kind = batch.trace_kind_id.to(device=device)

    choice_width = int(decision_mask.shape[1])
    if choice_width == 0:
        per_choice = _empty_replay_per_choice(output, n, device) if return_per_choice else None
        return log_probs, entropies, group_mask, per_choice
    selected_in_range = (selected >= 0) & (selected < choice_width)
    safe_selected = selected.clamp(min=0, max=choice_width - 1)
    selected_masked = decision_mask.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    selected_option = option_idx.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    selected_target = target_idx.gather(1, safe_selected.unsqueeze(1)).squeeze(1)

    blank_option_index = batch.encoded.blank_option_index.to(device=device)
    blank_group_kind = batch.encoded.blank_group_kind.to(device=device)
    blank_legal_mask = batch.encoded.blank_legal_mask.to(device=device, dtype=torch.bool)

    row_blank_option = blank_option_index[steps_t]
    row_blank_kind = blank_group_kind[steps_t]
    row_legal_mask = blank_legal_mask[steps_t]
    if int(row_blank_kind.shape[1]) == 0:
        per_choice = _empty_replay_per_choice(output, n, device) if return_per_choice else None
        return log_probs, entropies, group_mask, per_choice
    cross_support = (
        (row_blank_kind == BLANK_GROUP_CROSS_BLANK)
        & (row_blank_option >= 0)
        & row_legal_mask.any(dim=-1)
    )
    cross_matches = cross_support & (row_blank_option == selected_option.unsqueeze(1))
    cross_count = cross_matches.sum(dim=-1)
    cross_blank_idx = cross_matches.to(dtype=torch.int32).argmax(dim=-1)

    priority_scores = blank_logits[steps_t, :, 0]
    dummy_cross = torch.zeros_like(cross_support)
    if int(dummy_cross.shape[1]) > 0:
        dummy_cross[:, 0] = True
    effective_cross = torch.where(
        cross_support.any(dim=-1, keepdim=True), cross_support, dummy_cross
    )
    safe_priority_scores = torch.where(
        cross_support.any(dim=-1, keepdim=True),
        priority_scores,
        torch.zeros_like(priority_scores),
    )
    cross_log_probs_dense = torch.log_softmax(
        safe_priority_scores.masked_fill(~effective_cross, float("-inf")),
        dim=-1,
    )
    cross_probs = cross_log_probs_dense.exp()
    safe_cross_log_probs = torch.where(
        effective_cross,
        cross_log_probs_dense,
        cross_log_probs_dense.new_zeros(()),
    )
    cross_entropy = -(cross_probs * safe_cross_log_probs).sum(dim=-1)
    cross_log_prob = cross_log_probs_dense.gather(1, cross_blank_idx.unsqueeze(1)).squeeze(1)

    target_required = selected_target >= 0
    target_support = (
        (row_blank_kind == BLANK_GROUP_PER_BLANK)
        & (row_blank_option == selected_option.unsqueeze(1))
        & row_legal_mask.any(dim=-1)
    )
    target_count = target_support.sum(dim=-1)
    target_blank_idx = target_support.to(dtype=torch.int32).argmax(dim=-1)
    target_legal_mask = row_legal_mask[
        torch.arange(g_total, dtype=torch.long, device=device),
        target_blank_idx,
    ]
    target_width = int(target_legal_mask.shape[1])
    if target_width == 0:
        per_choice = _empty_replay_per_choice(output, n, device) if return_per_choice else None
        return log_probs, entropies, group_mask, per_choice
    selected_target_in_range = (selected_target >= 0) & (selected_target < target_width)
    safe_selected_target = selected_target.clamp(min=0, max=target_width - 1)
    selected_target_legal = target_legal_mask.gather(
        1,
        safe_selected_target.unsqueeze(1),
    ).squeeze(1)

    target_group_valid = (~target_required) | (
        (target_count == 1) & selected_target_in_range & selected_target_legal
    )
    is_priority_step = trace_kind[steps_t] == TRACE_KIND_TO_ID["priority"]
    group_mask = (
        is_priority_step
        & selected_in_range
        & selected_masked
        & (cross_count == 1)
        & target_group_valid
    )
    if group_skip_mask is not None:
        group_mask = group_mask & ~group_skip_mask.to(device=device, dtype=torch.bool)

    row_target_logits = blank_logits[
        steps_t,
        target_blank_idx,
    ]
    dummy_target_mask = torch.zeros_like(target_legal_mask)
    dummy_target_mask[:, 0] = True
    effective_target_mask = torch.where(
        target_required.unsqueeze(1) & group_mask.unsqueeze(1),
        target_legal_mask,
        dummy_target_mask,
    )
    safe_target_logits = torch.where(
        target_required.unsqueeze(1) & group_mask.unsqueeze(1),
        row_target_logits,
        torch.zeros_like(row_target_logits),
    )
    target_log_probs_dense = torch.log_softmax(
        safe_target_logits.masked_fill(~effective_target_mask, float("-inf")),
        dim=-1,
    )
    target_probs = target_log_probs_dense.exp()
    safe_target_log_probs = torch.where(
        effective_target_mask,
        target_log_probs_dense,
        target_log_probs_dense.new_zeros(()),
    )
    target_entropy = -(target_probs * safe_target_log_probs).sum(dim=-1)
    target_log_prob = target_log_probs_dense.gather(
        1,
        safe_selected_target.unsqueeze(1),
    ).squeeze(1)
    target_log_prob = torch.where(
        target_required,
        target_log_prob,
        target_log_prob.new_zeros(()),
    )
    target_entropy = torch.where(target_required, target_entropy, target_entropy.new_zeros(()))

    per_group_log_prob = torch.where(
        group_mask,
        cross_log_prob + target_log_prob,
        cross_log_prob.new_zeros(()),
    )
    per_group_entropy = torch.where(
        group_mask,
        cross_entropy + target_entropy,
        cross_entropy.new_zeros(()),
    )
    log_probs = log_probs.scatter_add(0, steps_t, per_group_log_prob.to(dtype=log_probs.dtype))
    entropies = entropies.scatter_add(0, steps_t, per_group_entropy.to(dtype=entropies.dtype))

    if not return_per_choice:
        return log_probs, entropies, group_mask, None

    flat_indices = (group_mask.unsqueeze(1) & decision_mask).nonzero(as_tuple=False)
    if int(flat_indices.numel()) == 0:
        per_choice = _empty_replay_per_choice(output, n, device)
    else:
        group_ids = flat_indices[:, 0]
        choice_cols = flat_indices[:, 1]
        col_options = option_idx[group_ids, choice_cols]
        col_targets = target_idx[group_ids, choice_cols]
        col_target_required = col_targets >= 0
        col_cross_matches = cross_support[group_ids] & (
            row_blank_option[group_ids] == col_options.unsqueeze(1)
        )
        col_cross_blank = col_cross_matches.to(dtype=torch.int32).argmax(dim=-1)
        col_cross_log_probs = cross_log_probs_dense[group_ids, col_cross_blank]
        col_cross_logits = priority_scores[group_ids, col_cross_blank]

        col_target_matches = target_support[group_ids]
        col_target_blank = col_target_matches.to(dtype=torch.int32).argmax(dim=-1)
        safe_col_targets = col_targets.clamp(min=0, max=target_width - 1)
        col_target_log_probs = target_log_probs_dense[group_ids, safe_col_targets]
        col_target_logits = blank_logits[steps_t[group_ids], col_target_blank, safe_col_targets]
        flat_log_probs = torch.where(
            col_target_required,
            col_cross_log_probs + col_target_log_probs,
            col_cross_log_probs,
        )
        flat_logits = torch.where(
            col_target_required,
            col_cross_logits + col_target_logits,
            col_cross_logits,
        )
        per_choice = ReplayPerChoice(
            flat_logits=flat_logits,
            flat_log_probs=flat_log_probs,
            group_idx=steps_t[group_ids],
            choice_cols=choice_cols,
            is_sampled_flat=choice_cols == selected[group_ids],
            may_is_active=torch.zeros(n, dtype=torch.bool, device=device),
            may_logits_per_step=output.values.new_zeros(n),
            may_selected_per_step=output.values.new_zeros(n),
            decision_group_id_flat=group_ids,
            step_for_decision_group=steps_t[group_mask],
            behavior_action_log_prob_per_decision_group=batch.behavior_action_log_prob[
                group_mask
            ].to(device=device, dtype=output.values.dtype),
        )
    return log_probs, entropies, group_mask, per_choice


def _evaluate_inline_choice_index_replay_groups(
    output: RecurrentTextPolicyOutput,
    batch: TextReplayBatch,
    *,
    return_per_choice: bool,
    trace_kind_id: int,
    group_skip_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, ReplayPerChoice | None]:
    decision_count = batch.decision_count
    n = int(decision_count.shape[0])
    g_total = int(batch.step_for_decision_group.shape[0])
    device = output.values.device
    log_probs = torch.zeros(n, dtype=torch.float32, device=device)
    entropies = torch.zeros(n, dtype=torch.float32, device=device)
    group_mask = torch.zeros(g_total, dtype=torch.bool, device=device)

    blank_logits = output.blank_logits
    if (
        g_total == 0
        or blank_logits is None
        or blank_logits.numel() == 0
        or batch.encoded.blank_option_index.numel() == 0
    ):
        per_choice = _empty_replay_per_choice(output, n, device) if return_per_choice else None
        return log_probs, entropies, group_mask, per_choice

    steps_t = batch.step_for_decision_group.to(device=device)
    selected = batch.selected_indices.to(device=device, dtype=torch.int32)
    trace_kind = batch.trace_kind_id.to(device=device)
    decision_mask = batch.decision_mask.to(device=device, dtype=torch.bool)

    blank_group_kind = batch.encoded.blank_group_kind.to(device=device)
    blank_option_index = batch.encoded.blank_option_index.to(device=device)
    blank_legal_mask = batch.encoded.blank_legal_mask.to(device=device, dtype=torch.bool)
    if int(blank_group_kind.shape[1]) == 0 or int(blank_legal_mask.shape[2]) == 0:
        per_choice = _empty_replay_per_choice(output, n, device) if return_per_choice else None
        return log_probs, entropies, group_mask, per_choice

    row_group_kind = blank_group_kind[steps_t]
    row_option_index = blank_option_index[steps_t]
    row_legal_mask = blank_legal_mask[steps_t]
    support = (
        (row_group_kind == BLANK_GROUP_PER_BLANK)
        & (row_option_index < 0)
        & row_legal_mask.any(dim=-1)
    )
    match_count = support.sum(dim=-1)
    blank_idx = support.to(dtype=torch.int32).argmax(dim=-1)
    row_mask = row_legal_mask[
        torch.arange(g_total, dtype=torch.long, device=device),
        blank_idx,
    ]
    legal_width = int(row_mask.shape[1])
    choice_width = int(decision_mask.shape[1])
    selected_in_range = (selected >= 0) & (selected < legal_width)
    selected_in_decision_range = (selected >= 0) & (selected < choice_width)
    safe_selected = selected.clamp(min=0, max=legal_width - 1)
    selected_legal = row_mask.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    safe_decision_selected = selected.clamp(min=0, max=max(choice_width - 1, 0))
    selected_engine_valid = (
        decision_mask.gather(1, safe_decision_selected.unsqueeze(1)).squeeze(1)
        if choice_width > 0
        else torch.zeros_like(selected_legal)
    )
    group_mask = (
        (trace_kind[steps_t] == int(trace_kind_id))
        & (match_count == 1)
        & selected_in_range
        & selected_in_decision_range
        & selected_legal
        & selected_engine_valid
    )
    if group_skip_mask is not None:
        group_mask = group_mask & ~group_skip_mask.to(device=device, dtype=torch.bool)

    row_logits = blank_logits[steps_t, blank_idx]
    dummy_mask = torch.zeros_like(row_mask)
    dummy_mask[:, 0] = True
    effective_mask = torch.where(group_mask.unsqueeze(1), row_mask, dummy_mask)
    safe_logits = torch.where(group_mask.unsqueeze(1), row_logits, torch.zeros_like(row_logits))
    log_probs_dense = torch.log_softmax(
        safe_logits.masked_fill(~effective_mask, float("-inf")),
        dim=-1,
    )
    probs = log_probs_dense.exp()
    safe_log_probs = torch.where(effective_mask, log_probs_dense, log_probs_dense.new_zeros(()))
    per_group_log_prob = log_probs_dense.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    per_group_entropy = -(probs * safe_log_probs).sum(dim=-1)
    per_group_log_prob = torch.where(
        group_mask,
        per_group_log_prob,
        per_group_log_prob.new_zeros(()),
    )
    per_group_entropy = torch.where(
        group_mask,
        per_group_entropy,
        per_group_entropy.new_zeros(()),
    )
    log_probs = log_probs.scatter_add(0, steps_t, per_group_log_prob.to(dtype=log_probs.dtype))
    entropies = entropies.scatter_add(0, steps_t, per_group_entropy.to(dtype=entropies.dtype))

    if not return_per_choice:
        return log_probs, entropies, group_mask, None

    flat_width = min(choice_width, legal_width)
    flat_valid = group_mask.unsqueeze(1) & decision_mask[:, :flat_width] & row_mask[:, :flat_width]
    flat_indices = flat_valid.nonzero(as_tuple=False)
    if int(flat_indices.numel()) == 0:
        per_choice = _empty_replay_per_choice(output, n, device)
    else:
        group_ids = flat_indices[:, 0]
        choice_cols = flat_indices[:, 1]
        per_choice = ReplayPerChoice(
            flat_logits=row_logits[group_ids, choice_cols],
            flat_log_probs=log_probs_dense[group_ids, choice_cols],
            group_idx=steps_t[group_ids],
            choice_cols=choice_cols,
            is_sampled_flat=choice_cols == selected[group_ids],
            may_is_active=torch.zeros(n, dtype=torch.bool, device=device),
            may_logits_per_step=output.values.new_zeros(n),
            may_selected_per_step=output.values.new_zeros(n),
            decision_group_id_flat=group_ids,
            step_for_decision_group=steps_t[group_mask],
            behavior_action_log_prob_per_decision_group=batch.behavior_action_log_prob[
                group_mask
            ].to(device=device, dtype=output.values.dtype),
        )
    return log_probs, entropies, group_mask, per_choice


def _evaluate_inline_blocker_replay_groups(
    output: RecurrentTextPolicyOutput,
    batch: TextReplayBatch,
    *,
    return_per_choice: bool,
    trace_kind_id: int = TRACE_KIND_TO_ID["blockers"],
    blank_group_kind_id: int = BLANK_GROUP_PER_BLANK,
    group_skip_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, ReplayPerChoice | None]:
    """Score replayed per-option choices from inline blank logits when available."""

    decision_count = batch.decision_count
    n = int(decision_count.shape[0])
    g_total = int(batch.step_for_decision_group.shape[0])
    device = output.values.device
    log_probs = torch.zeros(n, dtype=torch.float32, device=device)
    entropies = torch.zeros(n, dtype=torch.float32, device=device)
    group_mask = torch.zeros(g_total, dtype=torch.bool, device=device)

    blank_logits = output.blank_logits
    if (
        g_total == 0
        or blank_logits is None
        or blank_logits.numel() == 0
        or batch.encoded.blank_option_index.numel() == 0
    ):
        per_choice = _empty_replay_per_choice(output, n, device) if return_per_choice else None
        return log_probs, entropies, group_mask, per_choice

    steps_t = batch.step_for_decision_group.to(device=device)
    option_idx = batch.decision_option_idx.to(device=device, dtype=torch.int32)
    selected = batch.selected_indices.to(device=device, dtype=torch.int32)
    trace_kind = batch.trace_kind_id.to(device=device)

    option_valid = option_idx >= 0
    has_option = option_valid.any(dim=-1)
    first_option_col = option_valid.to(dtype=torch.int32).argmax(dim=-1)
    group_option_idx = torch.where(
        has_option,
        option_idx.gather(1, first_option_col.unsqueeze(1)).squeeze(1),
        _local_decision_group_indices(steps_t),
    )

    blank_option_index = batch.encoded.blank_option_index.to(device=device)
    blank_group_kind = batch.encoded.blank_group_kind.to(device=device)
    blank_legal_mask = batch.encoded.blank_legal_mask.to(device=device, dtype=torch.bool)

    row_blank_option = blank_option_index[steps_t]
    row_blank_kind = blank_group_kind[steps_t]
    blank_support = (row_blank_kind == blank_group_kind_id) & (row_blank_option >= 0)
    matches = (row_blank_option == group_option_idx.unsqueeze(1)) & blank_support
    match_count = matches.sum(dim=-1)
    blank_idx = matches.to(dtype=torch.int32).argmax(dim=-1)

    row_legal_mask = blank_legal_mask[steps_t, blank_idx]
    legal_width = int(row_legal_mask.shape[1])
    if legal_width == 0:
        per_choice = _empty_replay_per_choice(output, n, device) if return_per_choice else None
        return log_probs, entropies, group_mask, per_choice
    selected_in_range = (selected >= 0) & (selected < legal_width)
    safe_selected = selected.clamp(min=0, max=legal_width - 1)
    selected_legal = row_legal_mask.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    is_matching_step = trace_kind[steps_t] == trace_kind_id
    group_mask = is_matching_step & (match_count == 1) & selected_in_range & selected_legal
    if group_skip_mask is not None:
        group_mask = group_mask & ~group_skip_mask.to(device=device, dtype=torch.bool)

    dummy_mask = torch.zeros_like(row_legal_mask)
    if legal_width > 0:
        dummy_mask[:, 0] = True
    effective_mask = torch.where(group_mask.unsqueeze(1), row_legal_mask, dummy_mask)
    row_logits = blank_logits[steps_t, blank_idx]
    safe_logits = torch.where(group_mask.unsqueeze(1), row_logits, torch.zeros_like(row_logits))
    masked_logits = safe_logits.masked_fill(~effective_mask, float("-inf"))
    log_probs_dense = torch.log_softmax(masked_logits, dim=-1)
    probs_dense = log_probs_dense.exp()
    safe_log_probs = torch.where(effective_mask, log_probs_dense, log_probs_dense.new_zeros(()))
    per_group_log_prob = log_probs_dense.gather(1, safe_selected.unsqueeze(1)).squeeze(1)
    per_group_entropy = -(probs_dense * safe_log_probs).sum(dim=-1)
    per_group_log_prob = torch.where(
        group_mask,
        per_group_log_prob,
        per_group_log_prob.new_zeros(()),
    )
    per_group_entropy = torch.where(
        group_mask,
        per_group_entropy,
        per_group_entropy.new_zeros(()),
    )

    log_probs = log_probs.scatter_add(0, steps_t, per_group_log_prob.to(dtype=log_probs.dtype))
    entropies = entropies.scatter_add(0, steps_t, per_group_entropy.to(dtype=entropies.dtype))

    if not return_per_choice:
        return log_probs, entropies, group_mask, None

    flat_indices = (group_mask.unsqueeze(1) & row_legal_mask).nonzero(as_tuple=False)
    if int(flat_indices.numel()) == 0:
        per_choice = _empty_replay_per_choice(output, n, device)
    else:
        group_ids = flat_indices[:, 0]
        choice_cols = flat_indices[:, 1]
        per_choice = ReplayPerChoice(
            flat_logits=row_logits[group_ids, choice_cols],
            flat_log_probs=log_probs_dense[group_ids, choice_cols],
            group_idx=steps_t[group_ids],
            choice_cols=choice_cols,
            is_sampled_flat=choice_cols == selected[group_ids],
            may_is_active=torch.zeros(n, dtype=torch.bool, device=device),
            may_logits_per_step=output.values.new_zeros(n),
            may_selected_per_step=output.values.new_zeros(n),
            decision_group_id_flat=group_ids,
            step_for_decision_group=steps_t[group_mask],
            behavior_action_log_prob_per_decision_group=batch.behavior_action_log_prob[
                group_mask
            ].to(device=device, dtype=output.values.dtype),
        )
    return log_probs, entropies, group_mask, per_choice


def _empty_replay_per_choice(
    output: RecurrentTextPolicyOutput,
    n: int,
    device: torch.device,
) -> ReplayPerChoice:
    empty_long = torch.zeros(0, dtype=torch.long, device=device)
    empty_bool = torch.zeros(0, dtype=torch.bool, device=device)
    return ReplayPerChoice(
        flat_logits=torch.zeros(0, dtype=torch.float32, device=device),
        flat_log_probs=torch.zeros(0, dtype=torch.float32, device=device),
        group_idx=empty_long,
        choice_cols=empty_long,
        is_sampled_flat=empty_bool,
        may_is_active=torch.zeros(n, dtype=torch.bool, device=device),
        may_logits_per_step=torch.zeros(n, dtype=torch.float32, device=device),
        may_selected_per_step=torch.zeros(n, dtype=torch.float32, device=device),
        decision_group_id_flat=empty_long,
        step_for_decision_group=empty_long,
        behavior_action_log_prob_per_decision_group=output.values.new_zeros(0),
    )


def _concat_replay_per_choice(left: ReplayPerChoice, right: ReplayPerChoice) -> ReplayPerChoice:
    return ReplayPerChoice(
        flat_logits=torch.cat((left.flat_logits, right.flat_logits), dim=0),
        flat_log_probs=torch.cat((left.flat_log_probs, right.flat_log_probs), dim=0),
        group_idx=torch.cat((left.group_idx, right.group_idx), dim=0),
        choice_cols=torch.cat((left.choice_cols, right.choice_cols), dim=0),
        is_sampled_flat=torch.cat((left.is_sampled_flat, right.is_sampled_flat), dim=0),
        may_is_active=left.may_is_active,
        may_logits_per_step=left.may_logits_per_step,
        may_selected_per_step=left.may_selected_per_step,
        decision_group_id_flat=torch.cat(
            (left.decision_group_id_flat, right.decision_group_id_flat),
            dim=0,
        ),
        step_for_decision_group=torch.cat(
            (left.step_for_decision_group, right.step_for_decision_group),
            dim=0,
        ),
        behavior_action_log_prob_per_decision_group=torch.cat(
            (
                left.behavior_action_log_prob_per_decision_group,
                right.behavior_action_log_prob_per_decision_group,
            ),
            dim=0,
        ),
    )


def _dense_from_packed_batch(
    batch: PackedTextBatch,
    *,
    max_tokens: int,
    pad_id: int,
) -> TextEncodedBatch:
    b = int(batch.seq_lengths.shape[0])
    token_ids = torch.full(
        (b, max_tokens),
        int(pad_id),
        dtype=batch.token_ids.dtype,
        device=batch.token_ids.device,
    )
    attention_mask = torch.zeros((b, max_tokens), dtype=torch.bool, device=batch.token_ids.device)
    in_range = batch.pos_in_seq < max_tokens
    seq_id = batch.seq_id[in_range]
    pos = batch.pos_in_seq[in_range]
    token_ids[seq_id, pos] = batch.token_ids[in_range]
    attention_mask[seq_id, pos] = True
    return TextEncodedBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        card_ref_positions=subtract_packed_offsets(batch.card_ref_positions, batch.state_positions),
        seq_lengths=batch.seq_lengths,
        total_tokens=batch.total_tokens,
        seq_lengths_host=batch.seq_lengths_host,
        blank_positions=subtract_packed_offsets(batch.blank_positions, batch.state_positions),
        blank_kind=batch.blank_kind,
        blank_group=batch.blank_group,
        blank_group_kind=batch.blank_group_kind,
        blank_option_index=batch.blank_option_index,
        blank_legal_ids=batch.blank_legal_ids,
        blank_legal_mask=batch.blank_legal_mask,
    )


def _move_text_batch(batch: TextEncodedBatch, device: torch.device) -> TextEncodedBatch:
    # ``non_blocking=True`` only takes effect when the source is pinned;
    # the native assembler allocates its outputs in pinned memory when CUDA
    # is available. Bool conversion runs on the device after the copy so
    # the H2D transfer happens on the actual storage (uint8) rather than
    # an unpinned bool intermediate.
    nb = device.type == "cuda"
    return TextEncodedBatch(
        token_ids=batch.token_ids.to(device, non_blocking=nb),
        attention_mask=batch.attention_mask.to(device, non_blocking=nb),
        card_ref_positions=batch.card_ref_positions.to(device, non_blocking=nb),
        seq_lengths=batch.seq_lengths.to(device, non_blocking=nb),
        total_tokens=batch.total_tokens,
        seq_lengths_host=batch.seq_lengths_host,
        blank_positions=batch.blank_positions.to(device, non_blocking=nb),
        blank_kind=batch.blank_kind.to(device, non_blocking=nb),
        blank_group=batch.blank_group.to(device, non_blocking=nb),
        blank_group_kind=batch.blank_group_kind.to(device, non_blocking=nb),
        blank_option_index=batch.blank_option_index.to(device, non_blocking=nb),
        blank_legal_ids=batch.blank_legal_ids.to(device, non_blocking=nb),
        blank_legal_mask=batch.blank_legal_mask.to(device, non_blocking=nb),
    )


def _move_packed_text_batch(batch: PackedTextBatch, device: torch.device) -> PackedTextBatch:
    nb = device.type == "cuda"
    seq_lengths = batch.seq_lengths.to(device, non_blocking=nb)
    cu_seqlens = batch.cu_seqlens.to(device, non_blocking=nb)
    token_count = int(batch.token_ids.shape[0])
    if int(batch.seq_id.shape[0]) != token_count or int(batch.pos_in_seq.shape[0]) != token_count:
        raise ValueError(
            "packed batch token metadata length must match token_ids length "
            f"(tokens={token_count}, seq_id={int(batch.seq_id.shape[0])}, "
            f"pos_in_seq={int(batch.pos_in_seq.shape[0])})"
        )
    seq_id = batch.seq_id.to(device, non_blocking=nb)
    pos_in_seq = batch.pos_in_seq.to(device, non_blocking=nb)
    return PackedTextBatch(
        token_ids=batch.token_ids.to(device, non_blocking=nb),
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu_seqlens,
        seq_lengths=seq_lengths,
        state_positions=batch.state_positions.to(device, non_blocking=nb),
        card_ref_positions=batch.card_ref_positions.to(device, non_blocking=nb),
        total_tokens=batch.total_tokens,
        seq_lengths_host=batch.seq_lengths_host,
        max_seqlen=batch.max_seqlen,
        blank_positions=batch.blank_positions.to(device, non_blocking=nb),
        blank_kind=batch.blank_kind.to(device, non_blocking=nb),
        blank_group=batch.blank_group.to(device, non_blocking=nb),
        blank_group_kind=batch.blank_group_kind.to(device, non_blocking=nb),
        blank_option_index=batch.blank_option_index.to(device, non_blocking=nb),
        blank_legal_ids=batch.blank_legal_ids.to(device, non_blocking=nb),
        blank_legal_mask=batch.blank_legal_mask.to(device, non_blocking=nb),
    )


__all__ = [
    "TextActorCritic",
    "TextActorCriticStep",
    "TextDecisionLayout",
    "build_text_decision_layout",
    "infer_text_trace_kind",
]
