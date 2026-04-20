"""Actor-critic model for mage-go legal action spaces."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical

from magic_ai.actions import (
    COLORS,
    ActionOptionsEncoder,
    ActionRequest,
    EncodedActionOptions,
    PendingOptionState,
    PendingState,
    action_from_attackers,
    action_from_blockers,
    action_from_choice_accepted,
    action_from_choice_color,
    action_from_choice_ids,
    action_from_choice_index,
    action_from_priority_candidate,
)
from magic_ai.game_state import GameStateEncoder, GameStateSnapshot

if TYPE_CHECKING:
    from magic_ai.ppo import RolloutStep

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
    """Detached encoder output used for batched PPO head updates.

    Caching these tensors avoids rebuilding state/action encodings across PPO
    epochs. Gradients flow through the trunk and heads, but not back into the
    state/action encoders for cached updates.
    """

    features: Tensor
    decision_vectors: Tensor
    decision_mask: Tensor
    selected_indices: Tensor
    uses_none_head: Tensor
    may_selected: Tensor | None = None


@dataclass(frozen=True)
class PreparedPolicyInput:
    pending: PendingState
    features: Tensor
    encoded_options: EncodedActionOptions
    decision_vectors: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    trace_kind: TraceKind


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

    def act(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None = None,
        deterministic: bool = False,
    ) -> PolicyStep:
        """Sample or greedily choose a legal action for the current pending request."""

        prepared = self.prepare_action_input(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
        )
        return self.act_prepared_batch([prepared], deterministic=deterministic)[0]

    def prepare_action_input(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None = None,
    ) -> PreparedPolicyInput:
        common = self._encode_common(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
        )
        return self._prepared_from_common(pending, common)

    def act_prepared_batch(
        self,
        prepared_steps: Sequence[PreparedPolicyInput],
        *,
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        if not prepared_steps:
            return []

        device = next(self.parameters()).device
        features = torch.stack([step.features.to(device) for step in prepared_steps], dim=0)
        hidden = self.trunk(features)
        query = self.action_query(hidden) / math.sqrt(float(self.game_state_encoder.d_model))
        values = self.value_head(hidden).squeeze(-1)
        none_logits = self.none_blocker_head(hidden).squeeze(-1)
        may_logits = self.may_head(hidden).squeeze(-1)

        policy_steps: list[PolicyStep] = []
        for idx, prepared in enumerate(prepared_steps):
            policy_steps.append(
                self._act_one_prepared(
                    prepared,
                    query=query[idx],
                    value=values[idx],
                    none_logit=none_logits[idx],
                    may_logit=may_logits[idx],
                    deterministic=deterministic,
                )
            )
        return policy_steps

    def _act_legacy(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None = None,
        deterministic: bool = False,
    ) -> PolicyStep:
        outputs = self._forward_common(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
        )
        kind = pending.get("kind", "")
        if kind == "priority":
            return self._act_priority(pending, outputs, deterministic=deterministic)
        if kind == "attackers":
            return self._act_attackers(pending, outputs, deterministic=deterministic)
        if kind == "blockers":
            return self._act_blockers(pending, outputs, deterministic=deterministic)
        return self._act_choice(pending, outputs, deterministic=deterministic)

    def evaluate_rollout_step(self, step: RolloutStep) -> tuple[Tensor, Tensor, Tensor]:
        outputs = self._forward_common(
            step.state,
            step.pending,
            perspective_player_idx=step.perspective_player_idx,
        )
        log_prob, entropy = self._trace_log_prob_and_entropy(step.pending, step.trace, outputs)
        return log_prob, entropy, cast(Tensor, outputs["value"])

    def evaluate_cached_batch(
        self,
        cached_steps: Sequence[CachedPolicyInput],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate cached rollout encoder outputs as one trunk/head batch."""

        if not cached_steps:
            raise ValueError("cached_steps must not be empty")

        device = next(self.parameters()).device
        features = torch.stack([step.features.to(device) for step in cached_steps], dim=0)
        hidden = self.trunk(features)
        query = self.action_query(hidden) / math.sqrt(float(self.game_state_encoder.d_model))
        values = self.value_head(hidden).squeeze(-1)
        none_logits = self.none_blocker_head(hidden).squeeze(-1)
        may_logits = self.may_head(hidden).squeeze(-1)

        log_probs = torch.zeros(len(cached_steps), dtype=torch.float32, device=device)
        entropies = torch.zeros(len(cached_steps), dtype=torch.float32, device=device)
        group_vectors: list[Tensor] = []
        group_masks: list[Tensor] = []
        group_selected: list[Tensor] = []
        group_uses_none: list[Tensor] = []
        group_step_indices: list[int] = []

        for step_idx, cached in enumerate(cached_steps):
            decision_vectors = cached.decision_vectors.to(device)
            if decision_vectors.shape[0] == 0:
                may_selected = cached.may_selected
                if may_selected is not None:
                    selected = may_selected.to(device=device, dtype=torch.float32)
                    dist = Bernoulli(logits=may_logits[step_idx])
                    log_probs[step_idx] = dist.log_prob(selected)
                    entropies[step_idx] = dist.entropy()
                continue

            group_count = decision_vectors.shape[0]
            group_vectors.append(decision_vectors)
            group_masks.append(cached.decision_mask.to(device))
            group_selected.append(cached.selected_indices.to(device))
            group_uses_none.append(cached.uses_none_head.to(device))
            group_step_indices.extend([step_idx] * group_count)

        if group_vectors:
            vectors = torch.cat(group_vectors, dim=0)
            masks = torch.cat(group_masks, dim=0).bool()
            selected = torch.cat(group_selected, dim=0)
            uses_none = torch.cat(group_uses_none, dim=0).bool()
            step_indices = torch.tensor(group_step_indices, dtype=torch.long, device=device)
            group_query = query[step_indices]
            logits = torch.einsum("gcd,gd->gc", vectors, group_query)
            if uses_none.any():
                logits[uses_none, 0] = none_logits[step_indices[uses_none]]
            logits = logits.masked_fill(~masks, -torch.inf)
            dist = Categorical(logits=logits)
            group_log_probs = dist.log_prob(selected)
            group_entropies = dist.entropy()
            log_probs.scatter_add_(0, step_indices, group_log_probs)
            entropies.scatter_add_(0, step_indices, group_entropies)

        return log_probs, entropies, values

    def _forward_common(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None,
    ) -> dict[str, Tensor | EncodedActionOptions]:
        common = self._encode_common(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
        )
        features = cast(Tensor, common["features"])
        hidden = self.trunk(features)
        query = self.action_query(hidden) / math.sqrt(float(self.game_state_encoder.d_model))
        return {
            "features": features,
            "encoded_options": common["encoded_options"],
            "query": query,
            "value": self.value_head(hidden).squeeze(-1),
            "none_blocker_logit": self.none_blocker_head(hidden).squeeze(-1),
            "may_logit": self.may_head(hidden).squeeze(-1),
        }

    def _encode_common(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None,
    ) -> dict[str, Tensor | EncodedActionOptions]:
        state_vector, object_vectors = self.game_state_encoder.encode_state_with_references(
            state,
            perspective_player_idx=perspective_player_idx,
        )
        encoded_options = self.action_encoder(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
            precomputed_object_vectors=object_vectors,
        )
        option_mask = encoded_options["option_mask"].unsqueeze(-1)
        option_count = option_mask.sum().clamp_min(1.0)
        pooled_options = (encoded_options["option_vectors"] * option_mask).sum(dim=0) / option_count
        features = torch.cat(
            [state_vector, encoded_options["pending_vector"], pooled_options],
            dim=0,
        )
        return {
            "features": features,
            "encoded_options": encoded_options,
        }

    def _prepared_from_common(
        self,
        pending: PendingState,
        common: dict[str, Tensor | EncodedActionOptions],
    ) -> PreparedPolicyInput:
        device = cast(Tensor, common["features"]).device
        encoded = cast(EncodedActionOptions, common["encoded_options"])
        trace_kind = _trace_kind_for_pending(pending)
        decision_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        uses_none_head: list[bool] = []

        if trace_kind == "may":
            pass
        elif trace_kind == "priority":
            vectors: list[Tensor] = []
            for candidate in encoded["priority_candidates"][: self.max_cached_choices]:
                vector = encoded["option_vectors"][candidate.option_index]
                if candidate.target_index is not None:
                    vector = (
                        vector
                        + encoded["target_vectors"][candidate.option_index, candidate.target_index]
                    )
                vectors.append(vector)
            if vectors:
                decision_rows.append(_pad_vectors(vectors, self.max_cached_choices, device))
                mask_rows.append(_mask_row(len(vectors), self.max_cached_choices, device))
                uses_none_head.append(False)
        elif trace_kind == "attackers":
            option_count = min(len(pending.get("options", [])), self.max_options)
            zero = torch.zeros(self.game_state_encoder.d_model, dtype=torch.float32, device=device)
            for option_idx in range(option_count):
                decision_rows.append(
                    _pad_vectors(
                        [zero, encoded["option_vectors"][option_idx]],
                        self.max_cached_choices,
                        device,
                    )
                )
                mask_rows.append(_mask_row(2, self.max_cached_choices, device))
                uses_none_head.append(False)
        elif trace_kind == "blockers":
            options = pending.get("options", [])
            zero = torch.zeros(self.game_state_encoder.d_model, dtype=torch.float32, device=device)
            for option_idx, option in enumerate(options[: self.max_options]):
                target_count = min(
                    len(option.get("valid_targets", [])),
                    self.max_targets_per_option,
                )
                vectors = [
                    zero,
                    *[
                        encoded["target_vectors"][option_idx, target_idx]
                        for target_idx in range(target_count)
                    ],
                ]
                decision_rows.append(_pad_vectors(vectors, self.max_cached_choices, device))
                mask_rows.append(_mask_row(len(vectors), self.max_cached_choices, device))
                uses_none_head.append(True)
        else:
            option_count = min(len(pending.get("options", [])), self.max_options)
            vectors = [encoded["option_vectors"][option_idx] for option_idx in range(option_count)]
            if not vectors:
                vectors = [
                    torch.zeros(self.game_state_encoder.d_model, dtype=torch.float32, device=device)
                ]
            decision_rows.append(_pad_vectors(vectors, self.max_cached_choices, device))
            mask_rows.append(_mask_row(len(vectors), self.max_cached_choices, device))
            uses_none_head.append(False)

        if decision_rows:
            decision_vectors = torch.stack(decision_rows, dim=0).detach()
            decision_mask = torch.stack(mask_rows, dim=0).detach()
            uses_none_tensor = torch.tensor(uses_none_head, dtype=torch.bool, device=device)
        else:
            decision_vectors = torch.zeros(
                (0, self.max_cached_choices, self.game_state_encoder.d_model),
                dtype=torch.float32,
                device=device,
            )
            decision_mask = torch.zeros(
                (0, self.max_cached_choices), dtype=torch.bool, device=device
            )
            uses_none_tensor = torch.zeros((0,), dtype=torch.bool, device=device)

        return PreparedPolicyInput(
            pending=pending,
            features=cast(Tensor, common["features"]).detach(),
            encoded_options=encoded,
            decision_vectors=decision_vectors,
            decision_mask=decision_mask,
            uses_none_head=uses_none_tensor.detach(),
            trace_kind=trace_kind,
        )

    def _act_one_prepared(
        self,
        prepared: PreparedPolicyInput,
        *,
        query: Tensor,
        value: Tensor,
        none_logit: Tensor,
        may_logit: Tensor,
        deterministic: bool,
    ) -> PolicyStep:
        pending = prepared.pending
        if prepared.trace_kind == "may":
            dist = Bernoulli(logits=may_logit)
            selected = (
                torch.tensor(float(may_logit >= 0), device=may_logit.device)
                if deterministic
                else dist.sample()
            )
            accepted = bool(selected.item() >= 0.5)
            trace = ActionTrace("may", binary=(float(selected.item()),))
            return PolicyStep(
                action=action_from_choice_accepted(accepted),
                trace=trace,
                log_prob=dist.log_prob(selected),
                value=value,
                entropy=dist.entropy(),
                cache=_cache_from_prepared(prepared, trace),
            )

        if prepared.decision_vectors.shape[0] == 0:
            trace = ActionTrace("priority", indices=(0,))
            return _forced_pass(value, _cache_from_prepared(prepared, trace))

        logits = prepared.decision_vectors.to(query.device) @ query
        uses_none = prepared.uses_none_head.to(query.device)
        if uses_none.any():
            logits[uses_none, 0] = none_logit
        logits = logits.masked_fill(~prepared.decision_mask.to(query.device).bool(), -torch.inf)
        dist = Categorical(logits=logits)
        selected = torch.argmax(logits, dim=1) if deterministic else dist.sample()
        log_prob = dist.log_prob(selected).sum()
        entropy = dist.entropy().sum()

        if prepared.trace_kind == "priority":
            selected_idx = int(selected[0].item())
            candidates = prepared.encoded_options["priority_candidates"]
            if not candidates:
                trace = ActionTrace("priority", indices=(0,))
                return _forced_pass(value, _cache_from_prepared(prepared, trace))
            selected_idx = min(selected_idx, len(candidates) - 1)
            trace = ActionTrace("priority", indices=(selected_idx,))
            action = action_from_priority_candidate(candidates[selected_idx])
        elif prepared.trace_kind == "attackers":
            binary = tuple(float(item.item() == 1) for item in selected)
            trace = ActionTrace("attackers", binary=binary)
            action = action_from_attackers(pending, [value == 1.0 for value in binary])
        elif prepared.trace_kind == "blockers":
            indices = tuple(int(item.item()) - 1 for item in selected)
            trace = ActionTrace("blockers", indices=indices)
            action = action_from_blockers(pending, list(indices))
        elif prepared.trace_kind == "choice_ids":
            selected_idx = int(selected[0].item())
            selected_id = _selected_option_id(pending, selected_idx)
            trace = ActionTrace("choice_ids", indices=(selected_idx,))
            action = action_from_choice_ids([selected_id] if selected_id else [])
        elif prepared.trace_kind == "choice_color":
            selected_idx = int(selected[0].item())
            option = pending.get("options", [])[selected_idx]
            color = option.get("color", option.get("id", COLORS[selected_idx % len(COLORS)]))
            trace = ActionTrace("choice_color", indices=(selected_idx,))
            action = action_from_choice_color(color)
        else:
            selected_idx = int(selected[0].item())
            trace = ActionTrace("choice_index", indices=(selected_idx,))
            action = action_from_choice_index(selected_idx)

        return PolicyStep(
            action=action,
            trace=trace,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            cache=_cache_from_prepared(prepared, trace),
        )

    def _act_priority(
        self,
        pending: PendingState,
        outputs: dict[str, Tensor | EncodedActionOptions],
        *,
        deterministic: bool,
    ) -> PolicyStep:
        encoded = cast(EncodedActionOptions, outputs["encoded_options"])
        candidates = encoded["priority_candidates"]
        if not candidates:
            trace = ActionTrace("priority", indices=(0,))
            return _forced_pass(
                cast(Tensor, outputs["value"]),
                self._cache_from_outputs(pending, trace, outputs),
            )

        logits = self._priority_logits(encoded, cast(Tensor, outputs["query"]))
        dist = Categorical(logits=logits)
        selected = int(torch.argmax(logits).item()) if deterministic else int(dist.sample().item())
        trace = ActionTrace("priority", indices=(selected,))
        return PolicyStep(
            action=action_from_priority_candidate(candidates[selected]),
            trace=trace,
            log_prob=dist.log_prob(torch.tensor(selected, device=logits.device)),
            value=cast(Tensor, outputs["value"]),
            entropy=dist.entropy(),
            cache=self._cache_from_outputs(pending, trace, outputs),
        )

    def _act_attackers(
        self,
        pending: PendingState,
        outputs: dict[str, Tensor | EncodedActionOptions],
        *,
        deterministic: bool,
    ) -> PolicyStep:
        encoded = cast(EncodedActionOptions, outputs["encoded_options"])
        logits = self._option_logits(encoded, cast(Tensor, outputs["query"]))
        option_count = min(len(pending.get("options", [])), self.max_options)
        logits = logits[:option_count]
        if option_count == 0:
            selected = torch.zeros(0, device=logits.device)
            log_prob = torch.zeros((), device=logits.device)
            entropy = torch.zeros((), device=logits.device)
        else:
            dist = Bernoulli(logits=logits)
            selected = (logits >= 0).float() if deterministic else dist.sample()
            log_prob = dist.log_prob(selected).sum()
            entropy = dist.entropy().sum()
        trace = ActionTrace("attackers", binary=tuple(float(v) for v in selected.detach().cpu()))
        return PolicyStep(
            action=action_from_attackers(pending, selected),
            trace=trace,
            log_prob=log_prob,
            value=cast(Tensor, outputs["value"]),
            entropy=entropy,
            cache=self._cache_from_outputs(pending, trace, outputs),
        )

    def _act_blockers(
        self,
        pending: PendingState,
        outputs: dict[str, Tensor | EncodedActionOptions],
        *,
        deterministic: bool,
    ) -> PolicyStep:
        selected: list[int] = []
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        for option_idx, option in enumerate(pending.get("options", [])[: self.max_options]):
            logits = self._blocker_logits(option_idx, option, outputs)
            dist = Categorical(logits=logits)
            chosen = (
                int(torch.argmax(logits).item()) if deterministic else int(dist.sample().item())
            )
            selected.append(chosen - 1)
            log_probs.append(dist.log_prob(torch.tensor(chosen, device=logits.device)))
            entropies.append(dist.entropy())

        device = cast(Tensor, outputs["value"]).device
        log_prob = torch.stack(log_probs).sum() if log_probs else torch.zeros((), device=device)
        entropy = torch.stack(entropies).sum() if entropies else torch.zeros((), device=device)
        trace = ActionTrace("blockers", indices=tuple(selected))
        return PolicyStep(
            action=action_from_blockers(pending, selected),
            trace=trace,
            log_prob=log_prob,
            value=cast(Tensor, outputs["value"]),
            entropy=entropy,
            cache=self._cache_from_outputs(pending, trace, outputs),
        )

    def _act_choice(
        self,
        pending: PendingState,
        outputs: dict[str, Tensor | EncodedActionOptions],
        *,
        deterministic: bool,
    ) -> PolicyStep:
        pending_kind = pending.get("kind", "")
        if pending_kind == "may":
            return self._act_may(outputs, deterministic=deterministic)
        if pending_kind == "mana_color":
            return self._act_color(pending, outputs, deterministic=deterministic)
        if pending_kind in {"cards_from_hand", "card_from_library", "permanent"}:
            return self._act_choice_ids(pending, outputs, deterministic=deterministic)
        return self._act_choice_index(pending, outputs, deterministic=deterministic)

    def _act_choice_index(
        self,
        pending: PendingState,
        outputs: dict[str, Tensor | EncodedActionOptions],
        *,
        deterministic: bool,
    ) -> PolicyStep:
        logits = self._masked_option_logits(pending, outputs)
        dist = Categorical(logits=logits)
        selected = int(torch.argmax(logits).item()) if deterministic else int(dist.sample().item())
        trace = ActionTrace("choice_index", indices=(selected,))
        return PolicyStep(
            action=action_from_choice_index(selected),
            trace=trace,
            log_prob=dist.log_prob(torch.tensor(selected, device=logits.device)),
            value=cast(Tensor, outputs["value"]),
            entropy=dist.entropy(),
            cache=self._cache_from_outputs(pending, trace, outputs),
        )

    def _act_choice_ids(
        self,
        pending: PendingState,
        outputs: dict[str, Tensor | EncodedActionOptions],
        *,
        deterministic: bool,
    ) -> PolicyStep:
        logits = self._masked_option_logits(pending, outputs)
        dist = Categorical(logits=logits)
        selected = int(torch.argmax(logits).item()) if deterministic else int(dist.sample().item())
        selected_id = pending.get("options", [])[selected].get("id", "")
        if not selected_id:
            selected_id = pending.get("options", [])[selected].get("card_id", "")
        if not selected_id:
            selected_id = pending.get("options", [])[selected].get("permanent_id", "")
        trace = ActionTrace("choice_ids", indices=(selected,))
        return PolicyStep(
            action=action_from_choice_ids([selected_id] if selected_id else []),
            trace=trace,
            log_prob=dist.log_prob(torch.tensor(selected, device=logits.device)),
            value=cast(Tensor, outputs["value"]),
            entropy=dist.entropy(),
            cache=self._cache_from_outputs(pending, trace, outputs),
        )

    def _act_color(
        self,
        pending: PendingState,
        outputs: dict[str, Tensor | EncodedActionOptions],
        *,
        deterministic: bool,
    ) -> PolicyStep:
        logits = self._masked_option_logits(pending, outputs)
        dist = Categorical(logits=logits)
        selected = int(torch.argmax(logits).item()) if deterministic else int(dist.sample().item())
        option = pending.get("options", [])[selected]
        color = option.get("color", option.get("id", COLORS[selected % len(COLORS)]))
        trace = ActionTrace("choice_color", indices=(selected,))
        return PolicyStep(
            action=action_from_choice_color(color),
            trace=trace,
            log_prob=dist.log_prob(torch.tensor(selected, device=logits.device)),
            value=cast(Tensor, outputs["value"]),
            entropy=dist.entropy(),
            cache=self._cache_from_outputs(pending, trace, outputs),
        )

    def _act_may(
        self,
        outputs: dict[str, Tensor | EncodedActionOptions],
        *,
        deterministic: bool,
    ) -> PolicyStep:
        logit = cast(Tensor, outputs["may_logit"])
        dist = Bernoulli(logits=logit)
        selected = (
            torch.tensor(float(logit >= 0), device=logit.device) if deterministic else dist.sample()
        )
        accepted = bool(selected.item() >= 0.5)
        trace = ActionTrace("may", binary=(float(selected.item()),))
        return PolicyStep(
            action=action_from_choice_accepted(accepted),
            trace=trace,
            log_prob=dist.log_prob(selected),
            value=cast(Tensor, outputs["value"]),
            entropy=dist.entropy(),
            cache=self._cache_from_outputs(cast(PendingState, {"kind": "may"}), trace, outputs),
        )

    def _trace_log_prob_and_entropy(
        self,
        pending: PendingState,
        trace: ActionTrace,
        outputs: dict[str, Tensor | EncodedActionOptions],
    ) -> tuple[Tensor, Tensor]:
        device = cast(Tensor, outputs["value"]).device
        if trace.kind == "priority":
            encoded = cast(EncodedActionOptions, outputs["encoded_options"])
            logits = self._priority_logits(encoded, cast(Tensor, outputs["query"]))
            selected = torch.tensor(trace.indices[0], device=device)
            dist = Categorical(logits=logits)
            return dist.log_prob(selected), dist.entropy()
        if trace.kind == "attackers":
            encoded = cast(EncodedActionOptions, outputs["encoded_options"])
            logits = self._option_logits(
                encoded,
                cast(Tensor, outputs["query"]),
            )[: len(trace.binary)]
            selected = torch.tensor(trace.binary, dtype=torch.float32, device=device)
            dist = Bernoulli(logits=logits)
            return dist.log_prob(selected).sum(), dist.entropy().sum()
        if trace.kind == "blockers":
            log_probs: list[Tensor] = []
            entropies: list[Tensor] = []
            for option_idx, (option, target_idx) in enumerate(
                zip(pending.get("options", []), trace.indices, strict=False)
            ):
                logits = self._blocker_logits(option_idx, option, outputs)
                selected = torch.tensor(target_idx + 1, device=device)
                dist = Categorical(logits=logits)
                log_probs.append(dist.log_prob(selected))
                entropies.append(dist.entropy())
            return _sum_or_zero(log_probs, device), _sum_or_zero(entropies, device)
        if trace.kind == "may":
            selected = torch.tensor(trace.binary[0], dtype=torch.float32, device=device)
            dist = Bernoulli(logits=cast(Tensor, outputs["may_logit"]))
            return dist.log_prob(selected), dist.entropy()

        logits = self._masked_option_logits(pending, outputs)
        selected = torch.tensor(trace.indices[0], device=device)
        dist = Categorical(logits=logits)
        return dist.log_prob(selected), dist.entropy()

    def _cache_from_outputs(
        self,
        pending: PendingState,
        trace: ActionTrace,
        outputs: dict[str, Tensor | EncodedActionOptions],
    ) -> CachedPolicyInput:
        device = cast(Tensor, outputs["value"]).device
        encoded = cast(EncodedActionOptions, outputs["encoded_options"])
        empty_vectors = torch.zeros(
            (0, self.max_cached_choices, self.game_state_encoder.d_model),
            dtype=torch.float32,
            device=device,
        )
        empty_mask = torch.zeros((0, self.max_cached_choices), dtype=torch.bool, device=device)
        empty_selected = torch.zeros((0,), dtype=torch.long, device=device)
        empty_uses_none = torch.zeros((0,), dtype=torch.bool, device=device)

        if trace.kind == "may":
            selected = torch.tensor(float(trace.binary[0]), dtype=torch.float32, device=device)
            return CachedPolicyInput(
                features=cast(Tensor, outputs["features"]).detach(),
                decision_vectors=empty_vectors,
                decision_mask=empty_mask,
                selected_indices=empty_selected,
                uses_none_head=empty_uses_none,
                may_selected=selected.detach(),
            )

        decision_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        selected_indices: list[int] = []
        uses_none_head: list[bool] = []

        if trace.kind == "priority":
            candidates = encoded["priority_candidates"]
            if candidates:
                vectors: list[Tensor] = []
                for candidate in candidates[: self.max_cached_choices]:
                    vector = encoded["option_vectors"][candidate.option_index]
                    if candidate.target_index is not None:
                        vector = (
                            vector
                            + encoded["target_vectors"][
                                candidate.option_index, candidate.target_index
                            ]
                        )
                    vectors.append(vector)
                decision_rows.append(_pad_vectors(vectors, self.max_cached_choices, device))
                mask_rows.append(_mask_row(len(vectors), self.max_cached_choices, device))
                selected_indices.append(min(trace.indices[0], len(vectors) - 1))
                uses_none_head.append(False)
        elif trace.kind == "attackers":
            option_count = min(len(trace.binary), self.max_options)
            zero = torch.zeros(self.game_state_encoder.d_model, dtype=torch.float32, device=device)
            for option_idx in range(option_count):
                decision_rows.append(
                    _pad_vectors(
                        [zero, encoded["option_vectors"][option_idx]],
                        self.max_cached_choices,
                        device,
                    )
                )
                mask_rows.append(_mask_row(2, self.max_cached_choices, device))
                selected_indices.append(1 if trace.binary[option_idx] >= 0.5 else 0)
                uses_none_head.append(False)
        elif trace.kind == "blockers":
            options = pending.get("options", [])
            zero = torch.zeros(self.game_state_encoder.d_model, dtype=torch.float32, device=device)
            for option_idx, selected_target_idx in enumerate(trace.indices[: self.max_options]):
                if option_idx >= len(options):
                    break
                target_count = min(
                    len(options[option_idx].get("valid_targets", [])),
                    self.max_targets_per_option,
                )
                vectors = [
                    zero,
                    *[
                        encoded["target_vectors"][option_idx, target_idx]
                        for target_idx in range(target_count)
                    ],
                ]
                decision_rows.append(_pad_vectors(vectors, self.max_cached_choices, device))
                mask_rows.append(_mask_row(len(vectors), self.max_cached_choices, device))
                selected_indices.append(max(0, min(selected_target_idx + 1, len(vectors) - 1)))
                uses_none_head.append(True)
        else:
            option_count = min(len(pending.get("options", [])), self.max_options)
            if option_count > 0:
                vectors = [
                    encoded["option_vectors"][option_idx] for option_idx in range(option_count)
                ]
                decision_rows.append(_pad_vectors(vectors, self.max_cached_choices, device))
                mask_rows.append(_mask_row(len(vectors), self.max_cached_choices, device))
                selected_indices.append(min(trace.indices[0], len(vectors) - 1))
                uses_none_head.append(False)

        if decision_rows:
            decision_vectors = torch.stack(decision_rows, dim=0).detach()
            decision_mask = torch.stack(mask_rows, dim=0).detach()
            selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
            uses_none_tensor = torch.tensor(uses_none_head, dtype=torch.bool, device=device)
        else:
            decision_vectors = empty_vectors
            decision_mask = empty_mask
            selected_tensor = empty_selected
            uses_none_tensor = empty_uses_none

        return CachedPolicyInput(
            features=cast(Tensor, outputs["features"]).detach(),
            decision_vectors=decision_vectors,
            decision_mask=decision_mask,
            selected_indices=selected_tensor.detach(),
            uses_none_head=uses_none_tensor.detach(),
        )

    def _priority_logits(self, encoded: EncodedActionOptions, query: Tensor) -> Tensor:
        vectors: list[Tensor] = []
        for candidate in encoded["priority_candidates"]:
            vector = encoded["option_vectors"][candidate.option_index]
            if candidate.target_index is not None:
                vector = (
                    vector
                    + encoded["target_vectors"][candidate.option_index, candidate.target_index]
                )
            vectors.append(vector)
        if not vectors:
            return torch.zeros(1, device=query.device)
        return torch.stack(vectors, dim=0) @ query

    def _option_logits(self, encoded: EncodedActionOptions, query: Tensor) -> Tensor:
        return encoded["option_vectors"] @ query

    def _masked_option_logits(
        self,
        pending: PendingState,
        outputs: dict[str, Tensor | EncodedActionOptions],
    ) -> Tensor:
        encoded = cast(EncodedActionOptions, outputs["encoded_options"])
        option_count = min(len(pending.get("options", [])), self.max_options)
        if option_count == 0:
            return torch.zeros(1, device=cast(Tensor, outputs["value"]).device)
        return self._option_logits(encoded, cast(Tensor, outputs["query"]))[:option_count]

    def _blocker_logits(
        self,
        option_idx: int,
        option: object,
        outputs: dict[str, Tensor | EncodedActionOptions],
    ) -> Tensor:
        encoded = cast(EncodedActionOptions, outputs["encoded_options"])
        query = cast(Tensor, outputs["query"])
        if isinstance(option, dict):
            pending_option = cast(PendingOptionState, option)
            target_count = min(
                len(pending_option.get("valid_targets", [])),
                self.max_targets_per_option,
            )
        else:
            target_count = 0
        none_logit = cast(Tensor, outputs["none_blocker_logit"]).unsqueeze(0)
        if target_count == 0:
            return none_logit
        target_logits = encoded["target_vectors"][option_idx, :target_count] @ query
        return torch.cat([none_logit, target_logits], dim=0)


def _sum_or_zero(values: Iterable[Tensor], device: torch.device) -> Tensor:
    items = list(values)
    return torch.stack(items).sum() if items else torch.zeros((), device=device)


def _cache_from_prepared(prepared: PreparedPolicyInput, trace: ActionTrace) -> CachedPolicyInput:
    device = prepared.features.device
    if trace.kind == "may":
        selected = torch.tensor(float(trace.binary[0]), dtype=torch.float32, device=device)
        return CachedPolicyInput(
            features=prepared.features.detach(),
            decision_vectors=prepared.decision_vectors.detach(),
            decision_mask=prepared.decision_mask.detach(),
            selected_indices=torch.zeros((0,), dtype=torch.long, device=device),
            uses_none_head=prepared.uses_none_head.detach(),
            may_selected=selected.detach(),
        )

    if trace.kind == "attackers":
        selected = [1 if value >= 0.5 else 0 for value in trace.binary]
    elif trace.kind == "blockers":
        selected = [value + 1 for value in trace.indices]
    elif trace.indices:
        selected = [trace.indices[0]]
    else:
        selected = []

    return CachedPolicyInput(
        features=prepared.features.detach(),
        decision_vectors=prepared.decision_vectors.detach(),
        decision_mask=prepared.decision_mask.detach(),
        selected_indices=torch.tensor(selected, dtype=torch.long, device=device),
        uses_none_head=prepared.uses_none_head.detach(),
    )


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


def _selected_option_id(pending: PendingState, selected_idx: int) -> str:
    options = pending.get("options", [])
    if not 0 <= selected_idx < len(options):
        return ""
    option = options[selected_idx]
    return option.get("id", "") or option.get("card_id", "") or option.get("permanent_id", "")


def _pad_vectors(vectors: Sequence[Tensor], width: int, device: torch.device) -> Tensor:
    if not vectors:
        raise ValueError("vectors must not be empty")
    row = torch.zeros((width, vectors[0].shape[-1]), dtype=torch.float32, device=device)
    count = min(len(vectors), width)
    row[:count] = torch.stack(list(vectors[:count]), dim=0)
    return row


def _mask_row(count: int, width: int, device: torch.device) -> Tensor:
    row = torch.zeros((width,), dtype=torch.bool, device=device)
    row[: min(count, width)] = True
    return row


def _forced_pass(value: Tensor, cache: CachedPolicyInput | None = None) -> PolicyStep:
    zero = torch.zeros((), device=value.device)
    return PolicyStep(
        action=cast(ActionRequest, {"kind": "pass"}),
        trace=ActionTrace("priority", indices=(0,)),
        log_prob=zero,
        value=value,
        entropy=zero,
        cache=cache,
    )
