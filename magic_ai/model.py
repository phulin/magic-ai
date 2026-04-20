"""Actor-critic model for mage-go legal action spaces."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical

from magic_ai.actions import (
    COLORS,
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
from magic_ai.game_state import GameStateEncoder, GameStateSnapshot

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
class PolicyStep:
    action: ActionRequest
    trace: ActionTrace
    log_prob: Tensor
    value: Tensor
    entropy: Tensor
    cache: CachedPolicyInput | None = None


@dataclass(frozen=True)
class CachedPolicyInput:
    """Parsed (non-differentiable) inputs for one policy step.

    Stores index/scalar tensors describing a game state + pending request plus
    the decision-row layout needed to compute logits at PPO update time. No
    trainable tensors live here; the encoder stack is re-run on every forward
    so all of its parameters receive gradients.

    ``selected_indices`` / ``may_selected`` are filled in after sampling and
    let the update step replay the chosen action's log-probability without the
    raw ``trace`` or ``pending`` dict.
    """

    # Game-state parsing
    slot_card_rows: Tensor  # Long [ZONE_SLOT_COUNT]
    slot_occupied: Tensor  # Float [ZONE_SLOT_COUNT]
    slot_tapped: Tensor  # Float [ZONE_SLOT_COUNT]
    game_info: Tensor  # Float [GAME_INFO_DIM]
    # Action-option parsing
    pending_kind_id: Tensor  # Long []
    option_kind_ids: Tensor  # Long [max_options]
    option_scalars: Tensor  # Float [max_options, OPTION_SCALAR_DIM]
    option_mask: Tensor  # Float [max_options]
    option_ref_slot_idx: Tensor  # Long [max_options]
    option_ref_card_row: Tensor  # Long [max_options]
    target_mask: Tensor  # Float [max_options, max_targets]
    target_type_ids: Tensor  # Long [max_options, max_targets]
    target_scalars: Tensor  # Float [max_options, max_targets, 2]
    target_overflow: Tensor  # Float [max_options]
    target_ref_slot_idx: Tensor  # Long [max_options, max_targets]
    target_ref_is_player: Tensor  # Bool [max_options, max_targets]
    target_ref_is_self: Tensor  # Bool [max_options, max_targets]
    # Decision layout (variable G per step)
    trace_kind: TraceKind
    decision_option_idx: Tensor  # Long [G, C]
    decision_target_idx: Tensor  # Long [G, C]
    decision_mask: Tensor  # Bool [G, C]
    uses_none_head: Tensor  # Bool [G]
    selected_indices: Tensor  # Long [G]
    may_selected: Tensor | None
    # Reference only (rollout action reconstruction, not used in PPO update)
    pending: PendingState | None


class PPOPolicy(nn.Module):
    """Actor-critic network that scores legal mage-go action options."""

    def __init__(
        self,
        game_state_encoder: GameStateEncoder,
        *,
        hidden_dim: int = 512,
        max_options: int = 64,
        max_targets_per_option: int = 4,
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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def parse_inputs(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None = None,
    ) -> CachedPolicyInput:
        """Parse a state + pending request into integer/scalar index tensors.

        Pure CPU-side Python work plus small GPU tensor allocations. The
        resulting cache holds no trainable tensors, so downstream forwards can
        be re-run during PPO updates with gradients flowing through the full
        encoder stack.
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
        decision_option_idx, decision_target_idx, decision_mask, uses_none_head = (
            self._build_decision_layout(trace_kind, pending, parsed_action)
        )

        device = self.device
        return CachedPolicyInput(
            slot_card_rows=parsed_state.slot_card_rows,
            slot_occupied=parsed_state.slot_occupied,
            slot_tapped=parsed_state.slot_tapped,
            game_info=parsed_state.game_info,
            pending_kind_id=parsed_action.pending_kind_id,
            option_kind_ids=parsed_action.option_kind_ids,
            option_scalars=parsed_action.option_scalars,
            option_mask=parsed_action.option_mask,
            option_ref_slot_idx=parsed_action.option_ref_slot_idx,
            option_ref_card_row=parsed_action.option_ref_card_row,
            target_mask=parsed_action.target_mask,
            target_type_ids=parsed_action.target_type_ids,
            target_scalars=parsed_action.target_scalars,
            target_overflow=parsed_action.target_overflow,
            target_ref_slot_idx=parsed_action.target_ref_slot_idx,
            target_ref_is_player=parsed_action.target_ref_is_player,
            target_ref_is_self=parsed_action.target_ref_is_self,
            trace_kind=trace_kind,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            selected_indices=torch.zeros(
                decision_option_idx.shape[0], dtype=torch.long, device=device
            ),
            may_selected=None,
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
        cached = self.parse_inputs(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
        )
        return self.act_batch([cached], deterministic=deterministic)[0]

    def act_batch(
        self,
        cached_steps: Sequence[CachedPolicyInput],
        *,
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        """Run one forward over a batch of parsed steps and sample actions."""

        if not cached_steps:
            return []

        forward = self._forward_batch(cached_steps)
        results: list[PolicyStep] = []
        for step_idx, cached in enumerate(cached_steps):
            results.append(
                self._sample_step(
                    cached,
                    value=forward.values[step_idx],
                    query=forward.query[step_idx],
                    none_logit=forward.none_logits[step_idx],
                    may_logit=forward.may_logits[step_idx],
                    option_vectors=forward.option_vectors[step_idx],
                    target_vectors=forward.target_vectors[step_idx],
                    deterministic=deterministic,
                )
            )
        return results

    def evaluate_cached_batch(
        self,
        cached_steps: Sequence[CachedPolicyInput],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-run the encoder stack with gradients and return log-prob/entropy/value.

        All parameters in ``game_state_encoder``, ``action_encoder``, trunk,
        and heads receive gradients. The raw card-embedding table is a
        (frozen) buffer on the module's device and is indexed directly from
        the cached integer tensors — no CPU↔GPU transfers of embedded vectors.
        """

        if not cached_steps:
            raise ValueError("cached_steps must not be empty")

        device = self.device
        forward = self._forward_batch(cached_steps)
        n = len(cached_steps)
        log_probs = torch.zeros(n, dtype=forward.values.dtype, device=device)
        entropies = torch.zeros(n, dtype=forward.values.dtype, device=device)

        # May steps: Bernoulli on the may-head logit.
        may_step_indices: list[int] = []
        may_selected_values: list[float] = []
        for step_idx, cached in enumerate(cached_steps):
            if cached.trace_kind == "may" and cached.may_selected is not None:
                may_step_indices.append(step_idx)
                may_selected_values.append(float(cached.may_selected.detach().cpu()))
        if may_step_indices:
            may_idx_t = torch.tensor(may_step_indices, dtype=torch.long, device=device)
            may_selected_t = torch.tensor(
                may_selected_values, dtype=forward.values.dtype, device=device
            )
            may_logits_batch = forward.may_logits[may_idx_t]
            may_dist = Bernoulli(logits=may_logits_batch)
            log_probs[may_idx_t] = may_dist.log_prob(may_selected_t)
            entropies[may_idx_t] = may_dist.entropy()

        # Non-may steps: concatenate decision groups across the minibatch.
        group_step_indices: list[int] = []
        group_option_idx: list[Tensor] = []
        group_target_idx: list[Tensor] = []
        group_masks: list[Tensor] = []
        group_uses_none: list[Tensor] = []
        group_selected: list[Tensor] = []
        for step_idx, cached in enumerate(cached_steps):
            g = cached.decision_option_idx.shape[0]
            if g == 0:
                continue
            group_option_idx.append(cached.decision_option_idx.to(device))
            group_target_idx.append(cached.decision_target_idx.to(device))
            group_masks.append(cached.decision_mask.to(device))
            group_uses_none.append(cached.uses_none_head.to(device))
            group_selected.append(cached.selected_indices.to(device))
            group_step_indices.extend([step_idx] * g)

        if group_step_indices:
            step_indices_t = torch.tensor(group_step_indices, dtype=torch.long, device=device)
            option_idx = torch.cat(group_option_idx, dim=0)
            target_idx = torch.cat(group_target_idx, dim=0)
            masks = torch.cat(group_masks, dim=0).bool()
            uses_none = torch.cat(group_uses_none, dim=0).bool()
            selected = torch.cat(group_selected, dim=0)

            logits = self._decision_logits(
                step_indices=step_indices_t,
                option_idx=option_idx,
                target_idx=target_idx,
                masks=masks,
                uses_none=uses_none,
                option_vectors=forward.option_vectors,
                target_vectors=forward.target_vectors,
                query=forward.query,
                none_logits=forward.none_logits,
            )
            dist = Categorical(logits=logits)
            group_log_probs = dist.log_prob(selected)
            group_entropies = dist.entropy()
            log_probs.scatter_add_(0, step_indices_t, group_log_probs)
            entropies.scatter_add_(0, step_indices_t, group_entropies)

        return log_probs, entropies, forward.values

    def _forward_batch(self, cached_steps: Sequence[CachedPolicyInput]) -> _ForwardBatch:
        device = self.device
        slot_card_rows = torch.stack([c.slot_card_rows.to(device) for c in cached_steps])
        slot_occupied = torch.stack([c.slot_occupied.to(device) for c in cached_steps])
        slot_tapped = torch.stack([c.slot_tapped.to(device) for c in cached_steps])
        game_info = torch.stack([c.game_info.to(device) for c in cached_steps])
        pending_kind_id = torch.stack([c.pending_kind_id.to(device) for c in cached_steps])
        option_kind_ids = torch.stack([c.option_kind_ids.to(device) for c in cached_steps])
        option_scalars = torch.stack([c.option_scalars.to(device) for c in cached_steps])
        option_mask = torch.stack([c.option_mask.to(device) for c in cached_steps])
        option_ref_slot_idx = torch.stack([c.option_ref_slot_idx.to(device) for c in cached_steps])
        option_ref_card_row = torch.stack([c.option_ref_card_row.to(device) for c in cached_steps])
        target_mask = torch.stack([c.target_mask.to(device) for c in cached_steps])
        target_type_ids = torch.stack([c.target_type_ids.to(device) for c in cached_steps])
        target_scalars = torch.stack([c.target_scalars.to(device) for c in cached_steps])
        target_ref_slot_idx = torch.stack([c.target_ref_slot_idx.to(device) for c in cached_steps])
        target_ref_is_player = torch.stack(
            [c.target_ref_is_player.to(device) for c in cached_steps]
        )
        target_ref_is_self = torch.stack([c.target_ref_is_self.to(device) for c in cached_steps])

        slot_vectors = self.game_state_encoder.embed_slot_vectors(
            slot_card_rows, slot_occupied, slot_tapped
        )
        state_vector = self.game_state_encoder.state_vector_from_slots(slot_vectors, game_info)

        pending_vector, option_vectors, target_vectors = self.action_encoder.embed_from_parsed(
            slot_vectors=slot_vectors,
            pending_kind_id=pending_kind_id,
            option_kind_ids=option_kind_ids,
            option_scalars=option_scalars,
            option_mask=option_mask,
            option_ref_slot_idx=option_ref_slot_idx,
            option_ref_card_row=option_ref_card_row,
            target_mask=target_mask,
            target_type_ids=target_type_ids,
            target_scalars=target_scalars,
            target_ref_slot_idx=target_ref_slot_idx,
            target_ref_is_player=target_ref_is_player,
            target_ref_is_self=target_ref_is_self,
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
        step_indices: Tensor,
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

        Returned shape is ``[total_groups, max_cached_choices]``. Invalid
        columns are filled with -inf; group rows with ``uses_none_head=True``
        replace column 0 with the step-wise ``none_blocker_head`` logit.
        """

        d_model = option_vectors.shape[-1]
        max_targets = target_vectors.shape[-2]
        total_groups, choices = option_idx.shape

        option_idx_clamped = option_idx.clamp_min(0)
        target_idx_clamped = target_idx.clamp_min(0)
        options_for_groups = option_vectors[step_indices]  # [G, max_options, d]
        option_gather = torch.gather(
            options_for_groups,
            dim=1,
            index=option_idx_clamped.unsqueeze(-1).expand(-1, -1, d_model),
        )  # [G, C, d]
        option_present = (option_idx >= 0).unsqueeze(-1)
        option_part = torch.where(option_present, option_gather, torch.zeros_like(option_gather))

        targets_for_groups = target_vectors[step_indices]  # [G, max_options, max_targets, d]
        opt_gather = torch.gather(
            targets_for_groups,
            dim=1,
            index=option_idx_clamped.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, max_targets, d_model),
        )  # [G, C, max_targets, d]
        target_gather = torch.gather(
            opt_gather,
            dim=2,
            index=target_idx_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d_model),
        ).squeeze(2)  # [G, C, d]
        target_present = (target_idx >= 0).unsqueeze(-1)
        target_part = torch.where(target_present, target_gather, torch.zeros_like(target_gather))

        decision_vectors = option_part + target_part  # [G, C, d]
        query_for_groups = query[step_indices]  # [G, d]
        logits = torch.einsum("gcd,gd->gc", decision_vectors, query_for_groups)

        if uses_none.any():
            none_for_groups = none_logits[step_indices[uses_none]]
            logits[uses_none, 0] = none_for_groups

        return logits.masked_fill(~masks, -torch.inf)

    def _sample_step(
        self,
        cached: CachedPolicyInput,
        *,
        value: Tensor,
        query: Tensor,
        none_logit: Tensor,
        may_logit: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        deterministic: bool,
    ) -> PolicyStep:
        device = value.device
        pending = cached.pending
        assert pending is not None, "act requires pending on the cached input"

        if cached.trace_kind == "may":
            dist = Bernoulli(logits=may_logit)
            selected = (
                torch.tensor(float(may_logit >= 0), device=device)
                if deterministic
                else dist.sample()
            )
            accepted = bool(selected.item() >= 0.5)
            trace = ActionTrace("may", binary=(float(selected.item()),))
            enriched = _replace(
                cached,
                may_selected=selected.detach().to(dtype=torch.float32),
            )
            return PolicyStep(
                action=action_from_choice_accepted(accepted),
                trace=trace,
                log_prob=dist.log_prob(selected),
                value=value,
                entropy=dist.entropy(),
                cache=enriched,
            )

        g = cached.decision_option_idx.shape[0]
        if g == 0:
            # Forced pass (priority with no candidates) or empty choice set.
            trace = ActionTrace("priority", indices=(0,))
            zero = torch.zeros((), device=device)
            return PolicyStep(
                action=cast(ActionRequest, {"kind": "pass"}),
                trace=trace,
                log_prob=zero,
                value=value,
                entropy=zero,
                cache=cached,
            )

        step_indices = torch.zeros(g, dtype=torch.long, device=device)
        logits = self._decision_logits(
            step_indices=step_indices,
            option_idx=cached.decision_option_idx.to(device),
            target_idx=cached.decision_target_idx.to(device),
            masks=cached.decision_mask.to(device).bool(),
            uses_none=cached.uses_none_head.to(device).bool(),
            option_vectors=option_vectors.unsqueeze(0),
            target_vectors=target_vectors.unsqueeze(0),
            query=query.unsqueeze(0),
            none_logits=none_logit.unsqueeze(0),
        )
        dist = Categorical(logits=logits)
        if deterministic:
            selected = torch.argmax(logits, dim=-1)
        else:
            selected = dist.sample()
        log_prob = dist.log_prob(selected).sum()
        entropy = dist.entropy().sum()

        enriched = _replace(cached, selected_indices=selected.detach())
        trace, action = self._decode_action(cached.trace_kind, pending, selected)
        return PolicyStep(
            action=action,
            trace=trace,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            cache=enriched,
        )

    def _decode_action(
        self,
        trace_kind: TraceKind,
        pending: PendingState,
        selected: Tensor,
    ) -> tuple[ActionTrace, ActionRequest]:
        if trace_kind == "priority":
            selected_idx = int(selected[0].item())
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
            binary = tuple(float(item.item() == 1) for item in selected)
            return (
                ActionTrace("attackers", binary=binary),
                action_from_attackers(pending, [value == 1.0 for value in binary]),
            )
        if trace_kind == "blockers":
            indices = tuple(int(item.item()) - 1 for item in selected)
            return (
                ActionTrace("blockers", indices=indices),
                action_from_blockers(pending, list(indices)),
            )
        if trace_kind == "choice_ids":
            selected_idx = int(selected[0].item())
            target_id = selected_option_id(pending, selected_idx)
            return (
                ActionTrace("choice_ids", indices=(selected_idx,)),
                action_from_choice_ids([target_id] if target_id else []),
            )
        if trace_kind == "choice_color":
            selected_idx = int(selected[0].item())
            option = pending.get("options", [])[selected_idx]
            color = option.get("color", option.get("id", COLORS[selected_idx % len(COLORS)]))
            return (
                ActionTrace("choice_color", indices=(selected_idx,)),
                action_from_choice_color(color),
            )
        selected_idx = int(selected[0].item())
        return (
            ActionTrace("choice_index", indices=(selected_idx,)),
            action_from_choice_index(selected_idx),
        )

    def _build_decision_layout(
        self,
        trace_kind: TraceKind,
        pending: PendingState,
        parsed: ParsedActionInputs,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = self.device
        choices = self.max_cached_choices

        def empty() -> tuple[Tensor, Tensor, Tensor, Tensor]:
            return (
                torch.zeros((0, choices), dtype=torch.long, device=device),
                torch.zeros((0, choices), dtype=torch.long, device=device),
                torch.zeros((0, choices), dtype=torch.bool, device=device),
                torch.zeros((0,), dtype=torch.bool, device=device),
            )

        if trace_kind == "may":
            return empty()

        if trace_kind == "priority":
            candidates = parsed.priority_candidates[:choices]
            if not candidates:
                return empty()
            option_idx = torch.full((1, choices), -1, dtype=torch.long, device=device)
            target_idx = torch.full((1, choices), -1, dtype=torch.long, device=device)
            mask = torch.zeros((1, choices), dtype=torch.bool, device=device)
            for col, cand in enumerate(candidates):
                option_idx[0, col] = cand.option_index
                if cand.target_index is not None:
                    target_idx[0, col] = cand.target_index
                mask[0, col] = True
            uses_none = torch.zeros((1,), dtype=torch.bool, device=device)
            return option_idx, target_idx, mask, uses_none

        option_count = min(parsed.num_present_options, self.max_options)

        if trace_kind == "attackers":
            if option_count == 0:
                return empty()
            option_idx = torch.full((option_count, choices), -1, dtype=torch.long, device=device)
            target_idx = torch.full((option_count, choices), -1, dtype=torch.long, device=device)
            mask = torch.zeros((option_count, choices), dtype=torch.bool, device=device)
            uses_none = torch.zeros((option_count,), dtype=torch.bool, device=device)
            for i in range(option_count):
                option_idx[i, 1] = i
                mask[i, 0] = True
                mask[i, 1] = True
            return option_idx, target_idx, mask, uses_none

        if trace_kind == "blockers":
            options = pending.get("options", [])[: self.max_options]
            if not options:
                return empty()
            groups = len(options)
            option_idx = torch.full((groups, choices), -1, dtype=torch.long, device=device)
            target_idx = torch.full((groups, choices), -1, dtype=torch.long, device=device)
            mask = torch.zeros((groups, choices), dtype=torch.bool, device=device)
            uses_none = torch.ones((groups,), dtype=torch.bool, device=device)
            for i, option in enumerate(options):
                target_count = min(
                    len(option.get("valid_targets", [])), self.max_targets_per_option
                )
                mask[i, 0] = True  # col 0 will use none-blocker head
                for t in range(target_count):
                    col = t + 1
                    option_idx[i, col] = i
                    target_idx[i, col] = t
                    mask[i, col] = True
            return option_idx, target_idx, mask, uses_none

        # choice_index / choice_ids / choice_color
        if option_count == 0:
            return empty()
        option_idx = torch.full((1, choices), -1, dtype=torch.long, device=device)
        target_idx = torch.full((1, choices), -1, dtype=torch.long, device=device)
        mask = torch.zeros((1, choices), dtype=torch.bool, device=device)
        for i in range(option_count):
            option_idx[0, i] = i
            mask[0, i] = True
        uses_none = torch.zeros((1,), dtype=torch.bool, device=device)
        return option_idx, target_idx, mask, uses_none


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


def _replace(cached: CachedPolicyInput, **changes: object) -> CachedPolicyInput:
    from dataclasses import replace

    return replace(cached, **changes)
