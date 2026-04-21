"""Actor-critic model for mage-go legal action spaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli

from magic_ai.actions import (
    COLORS,
    OPTION_SCALAR_DIM,
    TARGET_SCALAR_DIM,
    ActionOptionsEncoder,
    ActionRequest,
    ParsedActionInputs,
    PendingState,
    action_from_attackers,
    action_from_blockers,
    action_from_choice_accepted,
    action_from_choice_color,
    action_from_choice_ids,
    action_from_choice_index,
    action_from_priority_candidate,
    selected_option_id,
)
from magic_ai.buffer import RolloutBuffer
from magic_ai.game_state import (
    GAME_INFO_DIM,
    ZONE_SLOT_COUNT,
    GameStateEncoder,
    GameStateSnapshot,
    ParsedGameState,
)

TraceKind = Literal[
    "priority",
    "attackers",
    "blockers",
    "choice_index",
    "choice_ids",
    "choice_color",
    "may",
]


@dataclass(frozen=True)
class ActionTrace:
    """Enough information to recompute a sampled action's log-probability."""

    kind: TraceKind
    indices: tuple[int, ...] = ()
    binary: tuple[float, ...] = ()


@dataclass(frozen=True)
class ParsedStep:
    """Pure-Python parsed policy inputs for one step.

    Holds the parsed game-state / action-options plus the decision-row layout.
    No tensors; ``RolloutBuffer.ingest_batch`` does the bulk CPU→GPU copy.
    """

    parsed_state: ParsedGameState
    parsed_action: ParsedActionInputs
    trace_kind: TraceKind
    decision_option_idx: list[list[int]]  # [G, C]
    decision_target_idx: list[list[int]]  # [G, C]
    decision_mask: list[list[bool]]  # [G, C]
    uses_none_head: list[bool]  # [G]
    pending: PendingState


@dataclass(frozen=True)
class CachedPolicyInput:
    """Slim handle into the rollout buffer for one sampled step.

    The actual parsed tensors live in ``PPOPolicy.rollout_buffer``; this record
    only stores the indices needed to gather them back during PPO evaluation.
    """

    buffer_idx: int
    decision_start: int
    decision_count: int
    trace_kind: TraceKind
    pending: PendingState | None


@dataclass(frozen=True)
class PolicyStep:
    action: ActionRequest
    trace: ActionTrace
    log_prob: Tensor
    value: Tensor
    entropy: Tensor
    cache: CachedPolicyInput | None = None


class PPOPolicy(nn.Module):
    """Actor-critic network that scores legal mage-go action options."""

    def __init__(
        self,
        game_state_encoder: GameStateEncoder,
        *,
        hidden_dim: int = 512,
        max_options: int = 64,
        max_targets_per_option: int = 4,
        rollout_capacity: int = 4096,
        decision_capacity: int | None = None,
    ) -> None:
        super().__init__()
        self.game_state_encoder = game_state_encoder
        self.action_encoder = ActionOptionsEncoder(
            game_state_encoder,
            max_options=max_options,
            max_targets_per_option=max_targets_per_option,
        )
        self.max_options = max_options
        self.max_targets_per_option = max_targets_per_option
        self.max_cached_choices = max(max_options, max_options * max(1, max_targets_per_option))

        input_dim = game_state_encoder.output_dim + game_state_encoder.d_model * 2
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_query = nn.Linear(hidden_dim, game_state_encoder.d_model)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.none_blocker_head = nn.Linear(hidden_dim, 1)
        self.may_head = nn.Linear(hidden_dim, 1)

        if decision_capacity is None:
            decision_capacity = rollout_capacity * 8

        self.rollout_buffer = RolloutBuffer(
            capacity=rollout_capacity,
            decision_capacity=decision_capacity,
            max_options=max_options,
            max_targets_per_option=max_targets_per_option,
            max_cached_choices=self.max_cached_choices,
            zone_slot_count=ZONE_SLOT_COUNT,
            game_info_dim=GAME_INFO_DIM,
            option_scalar_dim=OPTION_SCALAR_DIM,
            target_scalar_dim=TARGET_SCALAR_DIM,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_rollout_buffer(self) -> None:
        self.rollout_buffer.reset()

    def parse_inputs(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None = None,
    ) -> ParsedStep:
        """Parse a state + pending request into pure-Python index lists.

        No GPU allocations — ``act_batch`` does the bulk ingest into the
        rollout buffer in one pass per field.
        """

        parsed_state = self.game_state_encoder.parse_state(state, perspective_player_idx)
        player_idx = self.game_state_encoder._resolve_perspective_player_idx(
            state, perspective_player_idx
        )
        parsed_action = self.action_encoder.parse_pending(
            state,
            pending,
            perspective_player_idx=player_idx,
            card_id_to_slot=parsed_state.card_id_to_slot,
        )
        trace_kind = _trace_kind_for_pending(pending)
        option_idx, target_idx, mask, uses_none = self._build_decision_layout(
            trace_kind, pending, parsed_action
        )
        return ParsedStep(
            parsed_state=parsed_state,
            parsed_action=parsed_action,
            trace_kind=trace_kind,
            decision_option_idx=option_idx,
            decision_target_idx=target_idx,
            decision_mask=mask,
            uses_none_head=uses_none,
            pending=pending,
        )

    def act(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None = None,
        deterministic: bool = False,
    ) -> PolicyStep:
        parsed = self.parse_inputs(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
        )
        return self.act_batch([parsed], deterministic=deterministic)[0]

    def act_batch(
        self,
        parsed_steps: list[ParsedStep],
        *,
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        """Ingest parsed steps, run one forward, sample actions, write back.

        Sampling is batched: one ``Bernoulli`` over may-head logits and one
        ``Categorical`` over concatenated decision groups. Sampled values are
        written back to the buffer so ``evaluate_cached_batch`` can replay log
        probs from integer handles alone.
        """

        if not parsed_steps:
            return []

        device = self.device
        rb = self.rollout_buffer
        write = rb.ingest_batch(parsed_steps)
        n = len(parsed_steps)

        forward = self._forward_batch(write.step_indices)

        may_positions: list[int] = []
        group_step_positions: list[int] = []
        group_decision_indices: list[int] = []
        for step_idx, parsed in enumerate(parsed_steps):
            if parsed.trace_kind == "may":
                may_positions.append(step_idx)
                continue
            count = write.decision_counts[step_idx]
            start = write.decision_starts[step_idx]
            for k in range(count):
                group_step_positions.append(step_idx)
                group_decision_indices.append(start + k)

        may_log_probs: Tensor | None = None
        may_entropies: Tensor | None = None
        may_selected_cpu: list[float] = []
        if may_positions:
            may_pos_t = torch.tensor(may_positions, dtype=torch.long, device=device)
            may_logits = forward.may_logits[may_pos_t]
            may_dist = Bernoulli(logits=may_logits)
            if deterministic:
                may_sel = (may_logits >= 0).to(dtype=forward.values.dtype)
            else:
                may_sel = may_dist.sample()
            may_log_probs = may_dist.log_prob(may_sel)
            may_entropies = may_dist.entropy()
            may_selected_cpu = may_sel.detach().cpu().tolist()
            may_buf = write.step_indices[may_pos_t]
            rb.may_selected[may_buf] = may_sel.to(dtype=rb.may_selected.dtype)

        per_step_log_prob_sum: Tensor | None = None
        per_step_entropy_sum: Tensor | None = None
        decision_selected_cpu: list[int] = []
        if group_step_positions:
            pos_t = torch.tensor(group_step_positions, dtype=torch.long, device=device)
            idx_t = torch.tensor(group_decision_indices, dtype=torch.long, device=device)
            option_idx = rb.decision_option_idx[idx_t]
            target_idx = rb.decision_target_idx[idx_t]
            masks = rb.decision_mask[idx_t]
            uses_none = rb.uses_none_head[idx_t]

            group_idx, choice_cols, flat_logits, flat_log_probs, group_entropies = (
                self._flat_decision_distribution(
                    step_positions=pos_t,
                    option_idx=option_idx,
                    target_idx=target_idx,
                    masks=masks,
                    uses_none=uses_none,
                    option_vectors=forward.option_vectors,
                    target_vectors=forward.target_vectors,
                    query=forward.query,
                    none_logits=forward.none_logits,
                )
            )
            decision_selected, decision_log_probs = self._sample_flat_decisions(
                group_idx=group_idx,
                choice_cols=choice_cols,
                flat_logits=flat_logits,
                flat_log_probs=flat_log_probs,
                deterministic=deterministic,
            )
            per_step_log_prob_sum = torch.zeros(n, dtype=forward.values.dtype, device=device)
            per_step_entropy_sum = torch.zeros(n, dtype=forward.values.dtype, device=device)
            per_step_log_prob_sum.scatter_add_(0, pos_t, decision_log_probs)
            per_step_entropy_sum.scatter_add_(0, pos_t, group_entropies)
            decision_selected_cpu = decision_selected.detach().cpu().tolist()
            rb.selected_indices[idx_t] = decision_selected

        may_lookup = {step_idx: pos for pos, step_idx in enumerate(may_positions)}
        results: list[PolicyStep] = []
        offset = 0
        for step_idx, parsed in enumerate(parsed_steps):
            value = forward.values[step_idx]
            buffer_idx = int(write.step_indices[step_idx])
            decision_start = write.decision_starts[step_idx]
            decision_count = write.decision_counts[step_idx]
            cache = CachedPolicyInput(
                buffer_idx=buffer_idx,
                decision_start=decision_start,
                decision_count=decision_count,
                trace_kind=parsed.trace_kind,
                pending=parsed.pending,
            )

            if parsed.trace_kind == "may":
                pos = may_lookup[step_idx]
                assert may_log_probs is not None and may_entropies is not None
                sel_scalar = may_selected_cpu[pos]
                trace = ActionTrace("may", binary=(float(sel_scalar),))
                results.append(
                    PolicyStep(
                        action=action_from_choice_accepted(bool(sel_scalar >= 0.5)),
                        trace=trace,
                        log_prob=may_log_probs[pos],
                        value=value,
                        entropy=may_entropies[pos],
                        cache=cache,
                    )
                )
                continue

            if decision_count == 0:
                zero = torch.zeros((), device=device)
                results.append(
                    PolicyStep(
                        action=cast(ActionRequest, {"kind": "pass"}),
                        trace=ActionTrace("priority", indices=(0,)),
                        log_prob=zero,
                        value=value,
                        entropy=zero,
                        cache=cache,
                    )
                )
                continue

            assert per_step_log_prob_sum is not None and per_step_entropy_sum is not None
            step_selected = decision_selected_cpu[offset : offset + decision_count]
            offset += decision_count
            trace, action = self._decode_action(parsed.trace_kind, parsed.pending, step_selected)
            results.append(
                PolicyStep(
                    action=action,
                    trace=trace,
                    log_prob=per_step_log_prob_sum[step_idx],
                    value=value,
                    entropy=per_step_entropy_sum[step_idx],
                    cache=cache,
                )
            )
        return results

    def evaluate_cached_batch(
        self,
        cached_steps: list[CachedPolicyInput],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replay log-probs/entropies/values from buffer handles with gradients."""

        if not cached_steps:
            raise ValueError("cached_steps must not be empty")

        device = self.device
        rb = self.rollout_buffer
        n = len(cached_steps)

        step_indices = torch.tensor(
            [c.buffer_idx for c in cached_steps], dtype=torch.long, device=device
        )
        forward = self._forward_batch(step_indices)
        log_probs = torch.zeros(n, dtype=forward.values.dtype, device=device)
        entropies = torch.zeros(n, dtype=forward.values.dtype, device=device)

        may_step_positions: list[int] = []
        may_buffer_indices: list[int] = []
        for step_idx, cached in enumerate(cached_steps):
            if cached.trace_kind == "may":
                may_step_positions.append(step_idx)
                may_buffer_indices.append(cached.buffer_idx)
        if may_step_positions:
            may_pos_t = torch.tensor(may_step_positions, dtype=torch.long, device=device)
            may_buf_t = torch.tensor(may_buffer_indices, dtype=torch.long, device=device)
            may_logits = forward.may_logits[may_pos_t]
            may_selected_t = rb.may_selected[may_buf_t].to(dtype=forward.values.dtype)
            may_dist = Bernoulli(logits=may_logits)
            log_probs[may_pos_t] = may_dist.log_prob(may_selected_t)
            entropies[may_pos_t] = may_dist.entropy()

        group_step_positions: list[int] = []
        group_decision_indices: list[int] = []
        for step_idx, cached in enumerate(cached_steps):
            if cached.decision_count == 0:
                continue
            for k in range(cached.decision_count):
                group_step_positions.append(step_idx)
                group_decision_indices.append(cached.decision_start + k)

        if group_step_positions:
            pos_t = torch.tensor(group_step_positions, dtype=torch.long, device=device)
            idx_t = torch.tensor(group_decision_indices, dtype=torch.long, device=device)
            option_idx = rb.decision_option_idx[idx_t]
            target_idx = rb.decision_target_idx[idx_t]
            masks = rb.decision_mask[idx_t]
            uses_none = rb.uses_none_head[idx_t]
            selected = rb.selected_indices[idx_t]

            group_idx, choice_cols, _flat_logits, flat_log_probs, group_entropies = (
                self._flat_decision_distribution(
                    step_positions=pos_t,
                    option_idx=option_idx,
                    target_idx=target_idx,
                    masks=masks,
                    uses_none=uses_none,
                    option_vectors=forward.option_vectors,
                    target_vectors=forward.target_vectors,
                    query=forward.query,
                    none_logits=forward.none_logits,
                )
            )
            selected_mask = choice_cols == selected[group_idx]
            selected_flat_log_probs = torch.where(
                selected_mask,
                flat_log_probs,
                torch.zeros_like(flat_log_probs),
            )
            group_log_probs = torch.zeros_like(group_entropies)
            group_log_probs.scatter_add_(0, group_idx, selected_flat_log_probs)
            log_probs.scatter_add_(0, pos_t, group_log_probs)
            entropies.scatter_add_(0, pos_t, group_entropies)

        return log_probs, entropies, forward.values

    def _forward_batch(self, step_indices: Tensor) -> _ForwardBatch:
        rb = self.rollout_buffer
        slot_card_rows = rb.slot_card_rows[step_indices]
        slot_occupied = rb.slot_occupied[step_indices]
        slot_tapped = rb.slot_tapped[step_indices]
        game_info = rb.game_info[step_indices]
        option_mask = rb.option_mask[step_indices]

        slot_vectors = self.game_state_encoder.embed_slot_vectors(
            slot_card_rows, slot_occupied, slot_tapped
        )
        state_vector = self.game_state_encoder.state_vector_from_slots(slot_vectors, game_info)

        pending_vector, option_vectors, target_vectors = self.action_encoder.embed_from_parsed(
            slot_vectors=slot_vectors,
            pending_kind_id=rb.pending_kind_id[step_indices],
            option_kind_ids=rb.option_kind_ids[step_indices],
            option_scalars=rb.option_scalars[step_indices],
            option_mask=option_mask,
            option_ref_slot_idx=rb.option_ref_slot_idx[step_indices],
            option_ref_card_row=rb.option_ref_card_row[step_indices],
            target_mask=rb.target_mask[step_indices],
            target_type_ids=rb.target_type_ids[step_indices],
            target_scalars=rb.target_scalars[step_indices],
            target_ref_slot_idx=rb.target_ref_slot_idx[step_indices],
            target_ref_is_player=rb.target_ref_is_player[step_indices],
            target_ref_is_self=rb.target_ref_is_self[step_indices],
        )

        option_mask_f = option_mask.unsqueeze(-1)
        option_count = option_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (option_vectors * option_mask_f).sum(dim=1) / option_count

        features = torch.cat([state_vector, pending_vector, pooled], dim=-1)
        hidden = self.trunk(features)
        query = self.action_query(hidden) / math.sqrt(float(self.game_state_encoder.d_model))
        values = self.value_head(hidden).squeeze(-1)
        none_logits = self.none_blocker_head(hidden).squeeze(-1)
        may_logits = self.may_head(hidden).squeeze(-1)

        return _ForwardBatch(
            query=query,
            values=values,
            none_logits=none_logits,
            may_logits=may_logits,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
        )

    def _decision_logits(
        self,
        *,
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        query: Tensor,
        none_logits: Tensor,
    ) -> Tensor:
        return self._decision_logits_reference(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
            query=query,
            none_logits=none_logits,
        )

    def _decision_logits_reference(
        self,
        *,
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        query: Tensor,
        none_logits: Tensor,
    ) -> Tensor:
        """Compute masked logits for a concatenated set of decision groups.

        ``step_positions`` maps each of the ``total_groups`` rows to its
        position in the minibatch (0..n-1) so the corresponding
        option_vectors / target_vectors / query / none_logits row is selected.
        """

        d_model = option_vectors.shape[-1]
        max_targets = target_vectors.shape[-2]

        option_idx_clamped = option_idx.clamp_min(0)
        target_idx_clamped = target_idx.clamp_min(0)
        options_for_groups = option_vectors[step_positions]
        option_gather = torch.gather(
            options_for_groups,
            dim=1,
            index=option_idx_clamped.unsqueeze(-1).expand(-1, -1, d_model),
        )
        option_present = (option_idx >= 0).unsqueeze(-1)
        option_part = torch.where(option_present, option_gather, torch.zeros_like(option_gather))

        targets_for_groups = target_vectors[step_positions]
        opt_gather = torch.gather(
            targets_for_groups,
            dim=1,
            index=option_idx_clamped.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, max_targets, d_model),
        )
        target_gather = torch.gather(
            opt_gather,
            dim=2,
            index=target_idx_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d_model),
        ).squeeze(2)
        target_present = (target_idx >= 0).unsqueeze(-1)
        target_part = torch.where(target_present, target_gather, torch.zeros_like(target_gather))

        decision_vectors = option_part + target_part
        query_for_groups = query[step_positions]
        logits = torch.einsum("gcd,gd->gc", decision_vectors, query_for_groups)

        if uses_none.any():
            none_for_groups = none_logits[step_positions[uses_none]]
            logits[uses_none, 0] = none_for_groups

        return logits.masked_fill(~masks, -torch.inf)

    def _flat_decision_distribution(
        self,
        *,
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        query: Tensor,
        none_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return flat valid-choice logits plus grouped log-probs/entropies."""

        valid = masks.nonzero(as_tuple=False)
        if valid.numel() == 0:
            raise ValueError("decision groups must include at least one valid choice")

        device = masks.device
        group_idx = valid[:, 0]
        choice_cols = valid[:, 1]
        flat_step_positions = step_positions[group_idx]
        flat_logits = torch.empty(valid.shape[0], dtype=query.dtype, device=device)

        is_none = uses_none[group_idx] & (choice_cols == 0)
        if is_none.any():
            flat_logits[is_none] = none_logits[flat_step_positions[is_none]]

        is_scored = ~is_none
        if is_scored.any():
            scored_groups = group_idx[is_scored]
            scored_steps = flat_step_positions[is_scored]
            scored_cols = choice_cols[is_scored]

            scored_option_idx = option_idx[scored_groups, scored_cols]
            scored_target_idx = target_idx[scored_groups, scored_cols]

            scored_option_vectors = torch.zeros(
                (scored_groups.shape[0], option_vectors.shape[-1]),
                dtype=query.dtype,
                device=device,
            )
            has_option = scored_option_idx >= 0
            if has_option.any():
                scored_option_vectors[has_option] = option_vectors[
                    scored_steps[has_option],
                    scored_option_idx[has_option],
                ]

            scored_target_vectors = torch.zeros_like(scored_option_vectors)
            has_target = has_option & (scored_target_idx >= 0)
            if has_target.any():
                scored_target_vectors[has_target] = target_vectors[
                    scored_steps[has_target],
                    scored_option_idx[has_target],
                    scored_target_idx[has_target],
                ]

            decision_vectors = scored_option_vectors + scored_target_vectors
            flat_logits[is_scored] = (decision_vectors * query[scored_steps]).sum(dim=-1)

        group_count = step_positions.shape[0]
        group_max = torch.full((group_count,), -torch.inf, dtype=query.dtype, device=device)
        group_max.scatter_reduce_(0, group_idx, flat_logits, reduce="amax", include_self=True)

        stabilized = flat_logits - group_max[group_idx]
        exp_logits = stabilized.exp()
        group_exp_sum = torch.zeros(group_count, dtype=query.dtype, device=device)
        group_exp_sum.scatter_add_(0, group_idx, exp_logits)
        flat_log_probs = stabilized - group_exp_sum[group_idx].log()

        probs = flat_log_probs.exp()
        group_entropies = torch.zeros(group_count, dtype=query.dtype, device=device)
        group_entropies.scatter_add_(0, group_idx, -(probs * flat_log_probs))

        return group_idx, choice_cols, flat_logits, flat_log_probs, group_entropies

    def _sample_flat_decisions(
        self,
        *,
        group_idx: Tensor,
        choice_cols: Tensor,
        flat_logits: Tensor,
        flat_log_probs: Tensor,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor]:
        """Sample one valid choice per decision group."""

        device = flat_logits.device
        group_count = int(group_idx.max().item()) + 1
        choice_count = flat_logits.shape[0]
        flat_positions = torch.arange(choice_count, dtype=torch.long, device=device)

        counts = torch.bincount(group_idx, minlength=group_count)
        group_offsets = torch.zeros(group_count, dtype=torch.long, device=device)
        if group_count > 1:
            group_offsets[1:] = counts.cumsum(dim=0)[:-1]
        group_last = group_offsets + counts - 1

        if deterministic:
            group_max = torch.full(
                (group_count,),
                -torch.inf,
                dtype=flat_logits.dtype,
                device=device,
            )
            group_max.scatter_reduce_(0, group_idx, flat_logits, reduce="amax", include_self=True)
            sentinel = torch.full((choice_count,), choice_count, dtype=torch.long, device=device)
            first_best = torch.where(flat_logits == group_max[group_idx], flat_positions, sentinel)
            selected_flat = torch.full(
                (group_count,), choice_count, dtype=torch.long, device=device
            )
            selected_flat.scatter_reduce_(
                0,
                group_idx,
                first_best,
                reduce="amin",
                include_self=True,
            )
        else:
            probs = flat_log_probs.exp()
            flat_cumsum = probs.cumsum(dim=0)
            group_cumsum_offsets = torch.zeros(group_count, dtype=flat_logits.dtype, device=device)
            if group_count > 1:
                group_cumsum_offsets[1:] = flat_cumsum[group_last[:-1]]
            local_cumsum = flat_cumsum - group_cumsum_offsets[group_idx]
            thresholds = torch.rand(group_count, dtype=flat_logits.dtype, device=device)
            sentinel = torch.full((choice_count,), choice_count, dtype=torch.long, device=device)
            first_over = torch.where(
                local_cumsum >= thresholds[group_idx], flat_positions, sentinel
            )
            selected_flat = torch.full(
                (group_count,), choice_count, dtype=torch.long, device=device
            )
            selected_flat.scatter_reduce_(
                0,
                group_idx,
                first_over,
                reduce="amin",
                include_self=True,
            )
            selected_flat = torch.where(selected_flat == choice_count, group_last, selected_flat)

        selected_cols = choice_cols[selected_flat]
        selected_log_probs = flat_log_probs[selected_flat]
        return selected_cols, selected_log_probs

    def _decode_action(
        self,
        trace_kind: TraceKind,
        pending: PendingState,
        selected: list[int],
    ) -> tuple[ActionTrace, ActionRequest]:
        if trace_kind == "priority":
            selected_idx = selected[0]
            from magic_ai.actions import build_priority_candidates

            candidates_list = build_priority_candidates(
                pending, max_targets_per_option=self.max_targets_per_option
            )
            if not candidates_list:
                return ActionTrace("priority", indices=(0,)), cast(ActionRequest, {"kind": "pass"})
            selected_idx = min(selected_idx, len(candidates_list) - 1)
            return (
                ActionTrace("priority", indices=(selected_idx,)),
                action_from_priority_candidate(candidates_list[selected_idx]),
            )
        if trace_kind == "attackers":
            binary = tuple(float(v == 1) for v in selected)
            return (
                ActionTrace("attackers", binary=binary),
                action_from_attackers(pending, [value == 1.0 for value in binary]),
            )
        if trace_kind == "blockers":
            indices = tuple(v - 1 for v in selected)
            return (
                ActionTrace("blockers", indices=indices),
                action_from_blockers(pending, list(indices)),
            )
        if trace_kind == "choice_ids":
            selected_idx = selected[0]
            target_id = selected_option_id(pending, selected_idx)
            return (
                ActionTrace("choice_ids", indices=(selected_idx,)),
                action_from_choice_ids([target_id] if target_id else []),
            )
        if trace_kind == "choice_color":
            selected_idx = selected[0]
            option = pending.get("options", [])[selected_idx]
            color = option.get("color", option.get("id", COLORS[selected_idx % len(COLORS)]))
            return (
                ActionTrace("choice_color", indices=(selected_idx,)),
                action_from_choice_color(color),
            )
        selected_idx = selected[0]
        return (
            ActionTrace("choice_index", indices=(selected_idx,)),
            action_from_choice_index(selected_idx),
        )

    def _build_decision_layout(
        self,
        trace_kind: TraceKind,
        pending: PendingState,
        parsed: ParsedActionInputs,
    ) -> tuple[list[list[int]], list[list[int]], list[list[bool]], list[bool]]:
        """Return Python-side decision layout rows (padded to max_cached_choices).

        ``RolloutBuffer.ingest_batch`` flattens these into the buffer's
        decision storage in one torch.tensor call per field.
        """

        choices = self.max_cached_choices

        if trace_kind == "may":
            return [], [], [], []

        if trace_kind == "priority":
            candidates = parsed.priority_candidates[:choices]
            if not candidates:
                return [], [], [], []
            option_row = [-1] * choices
            target_row = [-1] * choices
            mask_row = [False] * choices
            for col, cand in enumerate(candidates):
                option_row[col] = cand.option_index
                if cand.target_index is not None:
                    target_row[col] = cand.target_index
                mask_row[col] = True
            return [option_row], [target_row], [mask_row], [False]

        option_count = min(parsed.num_present_options, self.max_options)

        if trace_kind == "attackers":
            if option_count == 0:
                return [], [], [], []
            option_idx_l: list[list[int]] = []
            target_idx_l: list[list[int]] = []
            mask_l: list[list[bool]] = []
            for i in range(option_count):
                option_row = [-1] * choices
                option_row[1] = i
                target_row = [-1] * choices
                mask_row = [False] * choices
                mask_row[0] = True
                mask_row[1] = True
                option_idx_l.append(option_row)
                target_idx_l.append(target_row)
                mask_l.append(mask_row)
            return option_idx_l, target_idx_l, mask_l, [False] * option_count

        if trace_kind == "blockers":
            options = pending.get("options", [])[: self.max_options]
            if not options:
                return [], [], [], []
            option_idx_l = []
            target_idx_l = []
            mask_l = []
            for i, option in enumerate(options):
                target_count = min(
                    len(option.get("valid_targets", [])), self.max_targets_per_option
                )
                option_row = [-1] * choices
                target_row = [-1] * choices
                mask_row = [False] * choices
                mask_row[0] = True  # col 0 will use none-blocker head
                for t in range(target_count):
                    col = t + 1
                    option_row[col] = i
                    target_row[col] = t
                    mask_row[col] = True
                option_idx_l.append(option_row)
                target_idx_l.append(target_row)
                mask_l.append(mask_row)
            return option_idx_l, target_idx_l, mask_l, [True] * len(options)

        # choice_index / choice_ids / choice_color
        if option_count == 0:
            return [], [], [], []
        option_row = [-1] * choices
        target_row = [-1] * choices
        mask_row = [False] * choices
        for i in range(option_count):
            option_row[i] = i
            mask_row[i] = True
        return [option_row], [target_row], [mask_row], [False]


@dataclass(frozen=True)
class _ForwardBatch:
    query: Tensor
    values: Tensor
    none_logits: Tensor
    may_logits: Tensor
    option_vectors: Tensor
    target_vectors: Tensor


def _trace_kind_for_pending(pending: PendingState) -> TraceKind:
    pending_kind = pending.get("kind", "")
    if pending_kind == "priority":
        return "priority"
    if pending_kind == "attackers":
        return "attackers"
    if pending_kind == "blockers":
        return "blockers"
    if pending_kind == "may":
        return "may"
    if pending_kind == "mana_color":
        return "choice_color"
    if pending_kind in {"cards_from_hand", "card_from_library", "permanent"}:
        return "choice_ids"
    return "choice_index"
