"""Actor-critic model for mage-go legal action spaces."""

from __future__ import annotations

import math
from collections.abc import Iterable
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

    def _forward_common(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None,
    ) -> dict[str, Tensor | EncodedActionOptions]:
        state_vector = self.game_state_encoder(
            state,
            perspective_player_idx=perspective_player_idx,
        )
        encoded_options = self.action_encoder(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
        )
        option_mask = encoded_options["option_mask"].unsqueeze(-1)
        option_count = option_mask.sum().clamp_min(1.0)
        pooled_options = (encoded_options["option_vectors"] * option_mask).sum(dim=0) / option_count
        features = torch.cat(
            [state_vector, encoded_options["pending_vector"], pooled_options],
            dim=0,
        )
        hidden = self.trunk(features)
        query = self.action_query(hidden) / math.sqrt(float(self.game_state_encoder.d_model))
        return {
            "encoded_options": encoded_options,
            "query": query,
            "value": self.value_head(hidden).squeeze(-1),
            "none_blocker_logit": self.none_blocker_head(hidden).squeeze(-1),
            "may_logit": self.may_head(hidden).squeeze(-1),
        }

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
            return _forced_pass(cast(Tensor, outputs["value"]))

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


def _forced_pass(value: Tensor) -> PolicyStep:
    zero = torch.zeros((), device=value.device)
    return PolicyStep(
        action=cast(ActionRequest, {"kind": "pass"}),
        trace=ActionTrace("priority", indices=(0,)),
        log_prob=zero,
        value=value,
        entropy=zero,
    )
