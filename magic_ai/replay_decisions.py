"""Backend-neutral helpers for replay decision-group scoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Bernoulli


@dataclass(frozen=True)
class ReplayPerChoice:
    """Per-choice decision-group tensors from replay policy evaluation.

    ``flat_logits`` and ``flat_log_probs`` are concatenated across all legal
    choices in all decision groups. ``group_idx`` maps each flat entry back to
    the replay-batch step index, while ``decision_group_id_flat`` preserves the
    distinct decision-group identity for multi-decision steps.
    """

    flat_logits: Tensor
    flat_log_probs: Tensor
    group_idx: Tensor
    choice_cols: Tensor
    is_sampled_flat: Tensor
    may_is_active: Tensor
    may_logits_per_step: Tensor
    may_selected_per_step: Tensor
    decision_group_id_flat: Tensor
    step_for_decision_group: Tensor


@dataclass(frozen=True)
class ReplayScoringForward:
    """Backend-neutral tensors needed to score replay rows.

    Slot replay uses ``query`` with option/target vectors. Text replay can
    instead expose direct ``option_logits`` / ``target_logits`` while still
    sharing values, none/may logits, and recurrent hidden state.
    """

    values: Tensor
    option_vectors: Tensor
    target_vectors: Tensor
    none_logits: Tensor
    may_logits: Tensor
    hidden: Tensor
    query: Tensor | None = None
    option_logits: Tensor | None = None
    target_logits: Tensor | None = None


def score_may_decisions(
    *,
    may_logits: Tensor,
    may_selected: Tensor,
    may_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Score per-step Bernoulli may decisions from replay rows.

    Returns ``(log_probs, entropies, may_logits_per_step, may_selected_per_step)``
    with zeros for non-may steps. Active logits keep gradient so downstream
    per-choice losses can operate on ``may_logits_per_step``.
    """

    selected = may_selected.to(dtype=may_logits.dtype, device=may_logits.device)
    log_probs = may_logits.new_zeros(may_mask.shape[0])
    entropies = may_logits.new_zeros(may_mask.shape[0])
    may_logits_per_step = may_logits.new_zeros(may_mask.shape[0])
    may_selected_per_step = may_logits.new_zeros(may_mask.shape[0])

    may_pos = may_mask.nonzero(as_tuple=False).squeeze(-1)
    active_logits = may_logits[may_pos]
    active_selected = selected[may_pos]
    dist = Bernoulli(logits=active_logits)
    log_probs[may_pos] = dist.log_prob(active_selected)
    entropies[may_pos] = dist.entropy()

    # Assignment into a cloned tensor preserves gradient from active logits.
    may_logits_per_step = may_logits_per_step.clone()
    may_logits_per_step[may_pos] = active_logits
    may_selected_per_step[may_pos] = active_selected
    return log_probs, entropies, may_logits_per_step, may_selected_per_step


def score_may_decisions_from_forward(
    forward: ReplayScoringForward,
    *,
    may_selected: Tensor,
    may_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Score may decisions using the common replay-forward shape."""

    return score_may_decisions(
        may_logits=forward.may_logits,
        may_selected=may_selected,
        may_mask=may_mask,
    )


def decision_logits_reference(
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
    validate: bool = False,
) -> Tensor:
    """Compute masked logits for a concatenated set of decision groups."""

    d_model = option_vectors.shape[-1]
    max_targets = target_vectors.shape[-2]

    if validate:
        validate_decision_indices(
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
        index=option_idx_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, max_targets, d_model),
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


def decision_logits_from_forward(
    forward: ReplayScoringForward,
    *,
    step_positions: Tensor,
    option_idx: Tensor,
    target_idx: Tensor,
    masks: Tensor,
    uses_none: Tensor,
    validate: bool = False,
) -> Tensor:
    """Compute decision logits using the common replay-forward shape."""

    if forward.query is None:
        raise ValueError("ReplayScoringForward.query is required for vector replay scoring")
    return decision_logits_reference(
        step_positions=step_positions,
        option_idx=option_idx,
        target_idx=target_idx,
        masks=masks,
        uses_none=uses_none,
        option_vectors=forward.option_vectors,
        target_vectors=forward.target_vectors,
        query=forward.query,
        none_logits=forward.none_logits,
        validate=validate,
    )


def direct_decision_logits_from_forward(
    forward: ReplayScoringForward,
    *,
    step_positions: Tensor,
    option_idx: Tensor,
    target_idx: Tensor,
    masks: Tensor,
    uses_none: Tensor,
    validate: bool = False,
) -> Tensor:
    """Compute replay logits from direct option/target heads.

    This is the text-backend counterpart to ``decision_logits_from_forward``:
    choices with ``target_idx >= 0`` use target logits, other option choices
    use option logits, and none choices use ``none_logits``.
    """

    if forward.option_logits is None or forward.target_logits is None:
        raise ValueError(
            "ReplayScoringForward.option_logits and target_logits are required "
            "for direct replay scoring"
        )

    if validate:
        validate_decision_indices(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            max_steps=forward.option_logits.shape[0],
            max_options=forward.option_logits.shape[1],
            max_targets=forward.target_logits.shape[2],
        )

    option_logits = forward.option_logits
    target_logits = forward.target_logits
    option_idx_clamped = option_idx.clamp(0, option_logits.shape[1] - 1)
    target_idx_clamped = target_idx.clamp(0, target_logits.shape[2] - 1)

    option_for_groups = option_logits[step_positions]
    option_choice_logits = torch.gather(option_for_groups, 1, option_idx_clamped)

    target_for_groups = target_logits[step_positions]
    option_targets = torch.gather(
        target_for_groups,
        dim=1,
        index=option_idx_clamped.unsqueeze(-1).expand(-1, -1, target_logits.shape[2]),
    )
    target_choice_logits = torch.gather(
        option_targets,
        dim=2,
        index=target_idx_clamped.unsqueeze(-1),
    ).squeeze(-1)

    has_target = target_idx >= 0
    has_option = option_idx >= 0
    logits = torch.where(has_target, target_choice_logits, option_choice_logits)
    logits = torch.where(has_option, logits, torch.full_like(logits, -torch.inf))
    if uses_none.any():
        logits[uses_none, 0] = forward.none_logits[step_positions[uses_none]]
    return logits.masked_fill(~masks, -torch.inf)


def flat_decision_distribution_impl(
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
    """Compile-friendly flat valid-choice logits plus grouped log-probs."""

    valid = masks.nonzero(as_tuple=False)
    device = masks.device
    group_idx = valid[:, 0]
    choice_cols = valid[:, 1]
    flat_step_positions = step_positions[group_idx]

    is_none = uses_none[group_idx].bool() & (choice_cols == 0)
    is_scored = ~is_none
    scored_pos = is_scored.nonzero(as_tuple=False).squeeze(-1)
    scored_groups = group_idx[scored_pos]
    scored_steps = flat_step_positions[scored_pos]
    scored_cols = choice_cols[scored_pos]

    scored_option_idx = option_idx[scored_groups, scored_cols]
    scored_target_idx = target_idx[scored_groups, scored_cols]

    scored_option_vectors = option_vectors[scored_steps, scored_option_idx]
    has_target = scored_target_idx >= 0
    hat_pos = has_target.nonzero(as_tuple=False).squeeze(-1)
    target_present = target_vectors[
        scored_steps[hat_pos],
        scored_option_idx[hat_pos],
        scored_target_idx[hat_pos],
    ]
    scored_target_vectors = torch.zeros_like(scored_option_vectors).index_put(
        (hat_pos,), target_present
    )

    decision_vectors = scored_option_vectors + scored_target_vectors
    scored_values = (decision_vectors * query[scored_steps]).sum(dim=-1)
    none_full = none_logits[flat_step_positions]
    flat_logits = none_full.scatter(0, scored_pos, scored_values)

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


def flat_decision_distribution_from_forward(
    forward: ReplayScoringForward,
    *,
    step_positions: Tensor,
    option_idx: Tensor,
    target_idx: Tensor,
    masks: Tensor,
    uses_none: Tensor,
    validate: bool = False,
    compiled_fn: Callable[..., tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return flat valid-choice logits using the common replay-forward shape."""

    if forward.query is None:
        raise ValueError("ReplayScoringForward.query is required for vector replay scoring")
    return flat_decision_distribution(
        step_positions=step_positions,
        option_idx=option_idx,
        target_idx=target_idx,
        masks=masks,
        uses_none=uses_none,
        option_vectors=forward.option_vectors,
        target_vectors=forward.target_vectors,
        query=forward.query,
        none_logits=forward.none_logits,
        validate=validate,
        compiled_fn=compiled_fn,
    )


def flat_decision_distribution(
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
    validate: bool = False,
    compiled_fn: Callable[..., tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return flat valid-choice logits plus grouped log-probs/entropies."""

    valid = masks.nonzero(as_tuple=False)
    if valid.numel() == 0:
        raise ValueError("decision groups must include at least one valid choice")

    if validate:
        group_idx_v = valid[:, 0]
        choice_cols_v = valid[:, 1]
        is_none_v = uses_none[group_idx_v] & (choice_cols_v == 0)
        is_scored_v = ~is_none_v
        scored_groups_v = group_idx_v[is_scored_v]
        scored_cols_v = choice_cols_v[is_scored_v]
        scored_steps_v = step_positions[group_idx_v][is_scored_v]
        validate_flat_scored_indices(
            scored_groups=scored_groups_v,
            scored_cols=scored_cols_v,
            scored_steps=scored_steps_v,
            scored_option_idx=option_idx[scored_groups_v, scored_cols_v],
            scored_target_idx=target_idx[scored_groups_v, scored_cols_v],
            max_steps=option_vectors.shape[0],
            max_options=option_vectors.shape[1],
            max_targets=target_vectors.shape[2],
        )

    fn = compiled_fn or flat_decision_distribution_impl
    return fn(
        step_positions,
        option_idx,
        target_idx,
        masks,
        uses_none,
        option_vectors,
        target_vectors,
        query,
        none_logits,
    )


def validate_flat_scored_indices(
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


def validate_decision_indices(
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
    validate_flat_scored_indices(
        scored_groups=scored_groups,
        scored_cols=scored_cols,
        scored_steps=scored_steps,
        scored_option_idx=option_idx[scored_groups, scored_cols],
        scored_target_idx=target_idx[scored_groups, scored_cols],
        max_steps=max_steps,
        max_options=max_options,
        max_targets=max_targets,
    )
