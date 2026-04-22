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
    LegalActionCandidate,
    ParsedActionBatch,
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
    ParsedGameStateBatch,
)
from magic_ai.native_encoder import NativeEncodedBatch

TraceKind = Literal[
    "priority",
    "attackers",
    "blockers",
    "choice_index",
    "choice_ids",
    "choice_color",
    "may",
]

TRACE_KIND_VALUES: tuple[TraceKind, ...] = (
    "priority",
    "attackers",
    "blockers",
    "choice_index",
    "choice_ids",
    "choice_color",
    "may",
)
TRACE_KIND_TO_ID: dict[TraceKind, int] = {
    trace_kind: idx for idx, trace_kind in enumerate(TRACE_KIND_VALUES)
}


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
    trace_kind_id: int
    decision_option_idx: list[list[int]]  # [G, C]
    decision_target_idx: list[list[int]]  # [G, C]
    decision_mask: list[list[bool]]  # [G, C]
    uses_none_head: list[bool]  # [G]
    pending: PendingState


@dataclass(frozen=True)
class ParsedBatch:
    """Batched parsed policy inputs for one actor forward."""

    parsed_state: ParsedGameStateBatch
    parsed_action: ParsedActionBatch
    trace_kinds: list[TraceKind]
    trace_kind_ids: Tensor  # [N]
    pendings: list[PendingState]
    decision_option_idx: Tensor  # [total_groups, max_cached_choices]
    decision_target_idx: Tensor  # [total_groups, max_cached_choices]
    decision_mask: Tensor  # [total_groups, max_cached_choices]
    uses_none_head: Tensor  # [total_groups]
    decision_starts: list[int]
    decision_counts: list[int]


@dataclass(frozen=True)
class PolicyStep:
    action: ActionRequest
    trace: ActionTrace
    log_prob: Tensor
    value: Tensor
    entropy: Tensor
    replay_idx: int | None = None


class PPOPolicy(nn.Module):
    """Actor-critic network that scores legal mage-go action options."""

    def __init__(
        self,
        game_state_encoder: GameStateEncoder,
        *,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
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
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        trunk_layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            trunk_layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU()))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
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

    def release_replay_rows(self, replay_rows: list[int]) -> None:
        if not replay_rows:
            return
        self.rollout_buffer.release_steps(replay_rows)

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
            trace_kind_id=TRACE_KIND_TO_ID[trace_kind],
            decision_option_idx=option_idx,
            decision_target_idx=target_idx,
            decision_mask=mask,
            uses_none_head=uses_none,
            pending=pending,
        )

    def parse_inputs_batch(
        self,
        states: list[GameStateSnapshot],
        pendings: list[PendingState],
        *,
        perspective_player_indices: list[int | None],
    ) -> ParsedBatch:
        parsed_state = self.game_state_encoder.parse_state_batch(
            states,
            perspective_player_indices,
        )
        resolved_player_indices = [
            self.game_state_encoder._resolve_perspective_player_idx(state, perspective_player_idx)
            for state, perspective_player_idx in zip(
                states, perspective_player_indices, strict=True
            )
        ]
        parsed_action = self.action_encoder.parse_pending_batch(
            states,
            pendings,
            perspective_player_indices=resolved_player_indices,
            card_id_to_slots=parsed_state.card_id_to_slots,
        )

        trace_kinds = [_trace_kind_for_pending(pending) for pending in pendings]
        trace_kind_ids = torch.tensor(
            [TRACE_KIND_TO_ID[trace_kind] for trace_kind in trace_kinds],
            dtype=torch.long,
        )

        flat_option_idx: list[list[int]] = []
        flat_target_idx: list[list[int]] = []
        flat_mask: list[list[bool]] = []
        flat_uses_none: list[bool] = []
        decision_starts: list[int] = []
        decision_counts: list[int] = []
        cursor = 0

        for step_idx, (trace_kind, pending) in enumerate(zip(trace_kinds, pendings, strict=True)):
            option_idx, target_idx, mask, uses_none = self._build_decision_layout_batch_step(
                trace_kind,
                pending,
                num_present_options=int(parsed_action.num_present_options[step_idx].item()),
                target_mask=parsed_action.target_mask[step_idx],
                priority_candidates=parsed_action.priority_candidates[step_idx],
            )
            decision_starts.append(cursor)
            decision_counts.append(len(option_idx))
            cursor += len(option_idx)
            flat_option_idx.extend(option_idx)
            flat_target_idx.extend(target_idx)
            flat_mask.extend(mask)
            flat_uses_none.extend(uses_none)

        if flat_option_idx:
            decision_option_idx = torch.tensor(flat_option_idx, dtype=torch.long)
            decision_target_idx = torch.tensor(flat_target_idx, dtype=torch.long)
            decision_mask = torch.tensor(flat_mask, dtype=torch.bool)
            uses_none_head = torch.tensor(flat_uses_none, dtype=torch.bool)
        else:
            decision_option_idx = torch.zeros((0, self.max_cached_choices), dtype=torch.long)
            decision_target_idx = torch.zeros((0, self.max_cached_choices), dtype=torch.long)
            decision_mask = torch.zeros((0, self.max_cached_choices), dtype=torch.bool)
            uses_none_head = torch.zeros((0,), dtype=torch.bool)

        return ParsedBatch(
            parsed_state=parsed_state,
            parsed_action=parsed_action,
            trace_kinds=trace_kinds,
            trace_kind_ids=trace_kind_ids,
            pendings=pendings,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )

    def _build_decision_layout_batch_step(
        self,
        trace_kind: TraceKind,
        pending: PendingState,
        *,
        num_present_options: int,
        target_mask: Tensor,
        priority_candidates: list[LegalActionCandidate],
    ) -> tuple[list[list[int]], list[list[int]], list[list[bool]], list[bool]]:
        choices = self.max_cached_choices

        if trace_kind == "may":
            return [], [], [], []

        if trace_kind == "priority":
            candidates = priority_candidates[:choices]
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

        option_count = min(num_present_options, self.max_options)

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
            return option_idx_l, target_idx_l, mask_l, [True] * option_count

        if trace_kind == "blockers":
            if option_count == 0:
                return [], [], [], []
            option_idx_l = []
            target_idx_l = []
            mask_l = []
            target_counts = target_mask[:option_count].count_nonzero(dim=-1).detach().cpu().tolist()
            for i, target_count in enumerate(target_counts):
                option_row = [-1] * choices
                target_row = [-1] * choices
                mask_row = [False] * choices
                mask_row[0] = True
                for t in range(int(target_count)):
                    col = t + 1
                    option_row[col] = i
                    target_row[col] = t
                    mask_row[col] = True
                option_idx_l.append(option_row)
                target_idx_l.append(target_row)
                mask_l.append(mask_row)
            return option_idx_l, target_idx_l, mask_l, [True] * option_count

        if option_count == 0:
            return [], [], [], []
        option_row = [-1] * choices
        target_row = [-1] * choices
        mask_row = [False] * choices
        for i in range(option_count):
            option_row[i] = i
            mask_row[i] = True
        return [option_row], [target_row], [mask_row], [False]

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

    def act_parsed_batch(
        self,
        parsed_batch: ParsedBatch | NativeEncodedBatch,
        *,
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        return self.act_batch(parsed_batch, deterministic=deterministic)

    def act_batch(
        self,
        parsed_steps: list[ParsedStep] | ParsedBatch | NativeEncodedBatch,
        *,
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        """Ingest parsed steps, run one forward, sample actions, write back.

        Sampling is batched: one ``Bernoulli`` over may-head logits and one
        ``Categorical`` over concatenated decision groups. Sampled values are
        written back to the buffer so ``evaluate_cached_batch`` can replay log
        probs from integer handles alone.
        """

        if isinstance(parsed_steps, NativeEncodedBatch):
            parsed_batch = self._parsed_batch_from_native(parsed_steps)
        elif isinstance(parsed_steps, ParsedBatch):
            parsed_batch = parsed_steps
        else:
            if not parsed_steps:
                return []
            parsed_batch = self._parsed_batch_from_steps(parsed_steps)

        if not parsed_batch.pendings:
            return []

        device = self.device
        rb = self.rollout_buffer
        write = rb.ingest_parsed_batch(parsed_batch)
        n = len(parsed_batch.pendings)

        forward = self._forward_batch(write.step_indices)

        may_positions: list[int] = []
        group_step_positions: list[int] = []
        group_decision_indices: list[int] = []
        for step_idx, trace_kind in enumerate(parsed_batch.trace_kinds):
            if trace_kind == "may":
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
        for step_idx, (trace_kind, pending) in enumerate(
            zip(parsed_batch.trace_kinds, parsed_batch.pendings, strict=True)
        ):
            value = forward.values[step_idx]
            replay_idx = int(write.step_indices[step_idx])
            decision_count = write.decision_counts[step_idx]

            if trace_kind == "may":
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
                        replay_idx=replay_idx,
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
                        replay_idx=replay_idx,
                    )
                )
                continue

            assert per_step_log_prob_sum is not None and per_step_entropy_sum is not None
            step_selected = decision_selected_cpu[offset : offset + decision_count]
            offset += decision_count
            trace, action = self._decode_action(trace_kind, pending, step_selected)
            results.append(
                PolicyStep(
                    action=action,
                    trace=trace,
                    log_prob=per_step_log_prob_sum[step_idx],
                    value=value,
                    entropy=per_step_entropy_sum[step_idx],
                    replay_idx=replay_idx,
                )
            )
        return results

    def _parsed_batch_from_steps(self, parsed_steps: list[ParsedStep]) -> ParsedBatch:
        parsed_state = ParsedGameStateBatch(
            slot_card_rows=torch.tensor(
                [parsed.parsed_state.slot_card_rows for parsed in parsed_steps],
                dtype=torch.long,
            ),
            slot_occupied=torch.tensor(
                [parsed.parsed_state.slot_occupied for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            slot_tapped=torch.tensor(
                [parsed.parsed_state.slot_tapped for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            game_info=torch.tensor(
                [parsed.parsed_state.game_info for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            card_id_to_slots=[parsed.parsed_state.card_id_to_slot for parsed in parsed_steps],
        )
        parsed_action = ParsedActionBatch(
            pending_kind_id=torch.tensor(
                [parsed.parsed_action.pending_kind_id for parsed in parsed_steps],
                dtype=torch.long,
            ),
            num_present_options=torch.tensor(
                [parsed.parsed_action.num_present_options for parsed in parsed_steps],
                dtype=torch.long,
            ),
            option_kind_ids=torch.tensor(
                [parsed.parsed_action.option_kind_ids for parsed in parsed_steps],
                dtype=torch.long,
            ),
            option_scalars=torch.tensor(
                [parsed.parsed_action.option_scalars for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            option_mask=torch.tensor(
                [parsed.parsed_action.option_mask for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            option_ref_slot_idx=torch.tensor(
                [parsed.parsed_action.option_ref_slot_idx for parsed in parsed_steps],
                dtype=torch.long,
            ),
            option_ref_card_row=torch.tensor(
                [parsed.parsed_action.option_ref_card_row for parsed in parsed_steps],
                dtype=torch.long,
            ),
            target_mask=torch.tensor(
                [parsed.parsed_action.target_mask for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            target_type_ids=torch.tensor(
                [parsed.parsed_action.target_type_ids for parsed in parsed_steps],
                dtype=torch.long,
            ),
            target_scalars=torch.tensor(
                [parsed.parsed_action.target_scalars for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            target_overflow=torch.tensor(
                [parsed.parsed_action.target_overflow for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            target_ref_slot_idx=torch.tensor(
                [parsed.parsed_action.target_ref_slot_idx for parsed in parsed_steps],
                dtype=torch.long,
            ),
            target_ref_is_player=torch.tensor(
                [parsed.parsed_action.target_ref_is_player for parsed in parsed_steps],
                dtype=torch.bool,
            ),
            target_ref_is_self=torch.tensor(
                [parsed.parsed_action.target_ref_is_self for parsed in parsed_steps],
                dtype=torch.bool,
            ),
            priority_candidates=[
                parsed.parsed_action.priority_candidates for parsed in parsed_steps
            ],
        )
        flat_option_idx: list[list[int]] = []
        flat_target_idx: list[list[int]] = []
        flat_mask: list[list[bool]] = []
        flat_uses_none: list[bool] = []
        decision_starts: list[int] = []
        decision_counts: list[int] = []
        cursor = 0
        for parsed in parsed_steps:
            decision_starts.append(cursor)
            count = len(parsed.decision_option_idx)
            decision_counts.append(count)
            cursor += count
            flat_option_idx.extend(parsed.decision_option_idx)
            flat_target_idx.extend(parsed.decision_target_idx)
            flat_mask.extend(parsed.decision_mask)
            flat_uses_none.extend(parsed.uses_none_head)
        if flat_option_idx:
            decision_option_idx = torch.tensor(flat_option_idx, dtype=torch.long)
            decision_target_idx = torch.tensor(flat_target_idx, dtype=torch.long)
            decision_mask = torch.tensor(flat_mask, dtype=torch.bool)
            uses_none_head = torch.tensor(flat_uses_none, dtype=torch.bool)
        else:
            decision_option_idx = torch.zeros((0, self.max_cached_choices), dtype=torch.long)
            decision_target_idx = torch.zeros((0, self.max_cached_choices), dtype=torch.long)
            decision_mask = torch.zeros((0, self.max_cached_choices), dtype=torch.bool)
            uses_none_head = torch.zeros((0,), dtype=torch.bool)
        return ParsedBatch(
            parsed_state=parsed_state,
            parsed_action=parsed_action,
            trace_kinds=[parsed.trace_kind for parsed in parsed_steps],
            trace_kind_ids=torch.tensor(
                [parsed.trace_kind_id for parsed in parsed_steps],
                dtype=torch.long,
            ),
            pendings=[parsed.pending for parsed in parsed_steps],
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )

    def _parsed_batch_from_native(self, native_batch: NativeEncodedBatch) -> ParsedBatch:
        return ParsedBatch(
            parsed_state=ParsedGameStateBatch(
                slot_card_rows=native_batch.slot_card_rows,
                slot_occupied=native_batch.slot_occupied,
                slot_tapped=native_batch.slot_tapped,
                game_info=native_batch.game_info,
                card_id_to_slots=[{} for _ in native_batch.pendings],
            ),
            parsed_action=ParsedActionBatch(
                pending_kind_id=native_batch.pending_kind_id,
                num_present_options=native_batch.num_present_options,
                option_kind_ids=native_batch.option_kind_ids,
                option_scalars=native_batch.option_scalars,
                option_mask=native_batch.option_mask,
                option_ref_slot_idx=native_batch.option_ref_slot_idx,
                option_ref_card_row=native_batch.option_ref_card_row,
                target_mask=native_batch.target_mask,
                target_type_ids=native_batch.target_type_ids,
                target_scalars=native_batch.target_scalars,
                target_overflow=native_batch.target_overflow,
                target_ref_slot_idx=native_batch.target_ref_slot_idx,
                target_ref_is_player=native_batch.target_ref_is_player,
                target_ref_is_self=native_batch.target_ref_is_self,
                priority_candidates=[[] for _ in native_batch.pendings],
            ),
            trace_kinds=[cast(TraceKind, trace_kind) for trace_kind in native_batch.trace_kinds],
            trace_kind_ids=native_batch.trace_kind_id,
            pendings=native_batch.pendings,
            decision_option_idx=native_batch.decision_option_idx,
            decision_target_idx=native_batch.decision_target_idx,
            decision_mask=native_batch.decision_mask,
            uses_none_head=native_batch.uses_none_head,
            decision_starts=native_batch.decision_start.detach().cpu().tolist(),
            decision_counts=native_batch.decision_count.detach().cpu().tolist(),
        )

    def evaluate_replay_batch(
        self,
        replay_rows: list[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replay log-probs/entropies/values from replay rows with gradients."""

        if not replay_rows:
            raise ValueError("replay_rows must not be empty")

        device = self.device
        rb = self.rollout_buffer
        step_indices = torch.tensor(replay_rows, dtype=torch.long, device=device)
        n = int(step_indices.numel())
        forward = self._forward_batch(step_indices)
        log_probs = torch.zeros(n, dtype=forward.values.dtype, device=device)
        entropies = torch.zeros(n, dtype=forward.values.dtype, device=device)

        trace_kind_ids = rb.trace_kind_id[step_indices]
        may_mask = trace_kind_ids == TRACE_KIND_TO_ID["may"]
        if may_mask.any():
            may_pos_t = may_mask.nonzero(as_tuple=False).squeeze(-1)
            may_buf_t = step_indices[may_pos_t]
            may_logits = forward.may_logits[may_pos_t]
            may_selected_t = rb.may_selected[may_buf_t].to(dtype=forward.values.dtype)
            may_dist = Bernoulli(logits=may_logits)
            log_probs[may_pos_t] = may_dist.log_prob(may_selected_t)
            entropies[may_pos_t] = may_dist.entropy()

        decision_starts = rb.decision_start[step_indices]
        decision_counts = rb.decision_count[step_indices]
        active_mask = decision_counts > 0
        if active_mask.any():
            pos_t = active_mask.nonzero(as_tuple=False).squeeze(-1)
            active_counts = decision_counts[pos_t]
            max_count = int(active_counts.max().item())
            offsets = torch.arange(max_count, dtype=torch.long, device=device).unsqueeze(0)
            expanded_offsets = offsets.expand(pos_t.shape[0], -1)
            valid_offsets = expanded_offsets < active_counts.unsqueeze(1)
            idx_t = (decision_starts[pos_t].unsqueeze(1) + expanded_offsets)[valid_offsets]
            pos_t = torch.repeat_interleave(pos_t, active_counts)
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
        self._validate_slot_card_rows(
            slot_card_rows, self.game_state_encoder.card_embedding_table.shape[0]
        )

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

    @staticmethod
    def _validate_slot_card_rows(slot_card_rows: Tensor, max_card_rows: int) -> None:
        bad = (slot_card_rows < 0) | (slot_card_rows >= max_card_rows)
        if not bad.any():
            return
        bad_pos = bad.nonzero(as_tuple=False)[0]
        batch = int(bad_pos[0].item())
        slot = int(bad_pos[1].item())
        row = int(slot_card_rows[batch, slot].item())
        raise ValueError(
            f"invalid slot card row: batch={batch} slot={slot} row={row} "
            f"bounds=[0, {max_card_rows})"
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

        self._validate_decision_indices(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            max_steps=option_vectors.shape[0],
            max_options=option_vectors.shape[1],
            max_targets=max_targets,
        )

        option_idx_clamped = option_idx.clamp(0, option_vectors.shape[1] - 1)
        target_idx_clamped = target_idx.clamp(0, max_targets - 1)
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
            self._validate_flat_scored_indices(
                scored_groups=scored_groups,
                scored_cols=scored_cols,
                scored_steps=scored_steps,
                scored_option_idx=scored_option_idx,
                scored_target_idx=scored_target_idx,
                max_steps=option_vectors.shape[0],
                max_options=option_vectors.shape[1],
                max_targets=target_vectors.shape[2],
            )

            scored_option_vectors = option_vectors[scored_steps, scored_option_idx]
            scored_target_vectors = torch.zeros_like(scored_option_vectors)
            has_target = scored_target_idx >= 0
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

    @staticmethod
    def _validate_flat_scored_indices(
        *,
        scored_groups: Tensor,
        scored_cols: Tensor,
        scored_steps: Tensor,
        scored_option_idx: Tensor,
        scored_target_idx: Tensor,
        max_steps: int,
        max_options: int,
        max_targets: int,
    ) -> None:
        bad = (
            (scored_steps < 0)
            | (scored_steps >= max_steps)
            | (scored_option_idx < 0)
            | (scored_option_idx >= max_options)
            | (scored_target_idx >= max_targets)
        )
        if not bad.any():
            return

        bad_pos = int(bad.nonzero(as_tuple=False)[0, 0].item())
        group = int(scored_groups[bad_pos].item())
        col = int(scored_cols[bad_pos].item())
        step = int(scored_steps[bad_pos].item())
        option = int(scored_option_idx[bad_pos].item())
        target = int(scored_target_idx[bad_pos].item())
        raise ValueError(
            "invalid decision gather index: "
            f"group={group} col={col} step={step} option={option} target={target} "
            f"bounds=(steps={max_steps}, options={max_options}, targets={max_targets})"
        )

    @staticmethod
    def _validate_decision_indices(
        *,
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        max_steps: int,
        max_options: int,
        max_targets: int,
    ) -> None:
        valid = masks.nonzero(as_tuple=False)
        if valid.numel() == 0:
            return
        groups = valid[:, 0]
        cols = valid[:, 1]
        scored = ~(uses_none[groups] & cols.eq(0))
        if not scored.any():
            return
        scored_groups = groups[scored]
        scored_cols = cols[scored]
        scored_steps = step_positions[scored_groups]
        PPOPolicy._validate_flat_scored_indices(
            scored_groups=scored_groups,
            scored_cols=scored_cols,
            scored_steps=scored_steps,
            scored_option_idx=option_idx[scored_groups, scored_cols],
            scored_target_idx=target_idx[scored_groups, scored_cols],
            max_steps=max_steps,
            max_options=max_options,
            max_targets=max_targets,
        )

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
            return option_idx_l, target_idx_l, mask_l, [True] * option_count

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
