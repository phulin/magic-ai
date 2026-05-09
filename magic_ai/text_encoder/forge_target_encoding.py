"""Translate observed Forge events into decoder-target token sequences.

Step 10 of ``docs/decoder_grammar_plan.md``. Given a ``PendingState`` (the
engine's pending decision at extraction time) and the observed-event
metadata produced by :mod:`scripts.extract_forge_choice_situations`, build
the flat (token_ids, pointer_subjects, is_pointer) sequence the
autoregressive decoder is asked to reproduce.

Each translator returns a sequence that is grammar-legal under
:func:`magic_ai.text_encoder.grammar.next_mask` for the corresponding
``DecisionType``. The translators are deliberately small + functional so
they're directly testable.

Conventions:
- ``output_pointer_subjects[i]`` is the *subject_index* of the chosen
  anchor (i.e. the entity's local 0-based ordinal within its kind on
  this decision), not its encoder position. The training-time collator
  resolves subject_index → encoder_position via the row's rendered
  ``DecisionSpec.anchors``.
- For non-pointer steps, ``output_pointer_subjects[i] = -1`` and
  ``output_is_pointer[i] = False``.
- Translators return ``None`` when the observed event cannot be mapped
  to the pending decision (e.g. ambiguous attacker text). Callers should
  drop those rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from magic_ai.game_state import PendingOptionState, PendingState
from magic_ai.text_encoder.decision_spec import DecisionType
from magic_ai.text_encoder.grammar import GrammarVocab, bpe_digit_str_to_grammar_ids


@dataclass(frozen=True)
class DecoderTarget:
    """Flat decoder target sequence for one decision."""

    decision_type: int
    output_token_ids: list[int]
    output_pointer_subjects: list[int]
    output_is_pointer: list[bool]


# Pending-kind → DecisionType mirror (matches render_spec._PENDING_KIND_TO_DECISION_TYPE).
_PENDING_KIND_TO_DECISION_TYPE: dict[str, DecisionType] = {
    "priority": DecisionType.PRIORITY,
    "attackers": DecisionType.DECLARE_ATTACKERS,
    "blockers": DecisionType.DECLARE_BLOCKERS,
    "permanent": DecisionType.CHOOSE_TARGETS,
    "cards_from_hand": DecisionType.CHOOSE_TARGETS,
    "card_from_library": DecisionType.CHOOSE_TARGETS,
    "may": DecisionType.MAY,
    "mode": DecisionType.CHOOSE_MODE,
    "number": DecisionType.CHOOSE_X,
}


def pending_decision_type(pending: PendingState) -> DecisionType | None:
    """Return the ``DecisionType`` for a pending kind, or None if deferred."""

    return _PENDING_KIND_TO_DECISION_TYPE.get(pending["kind"])


# --------------------------------------------------------------------------- #
# Per-decision-type translators                                               #
# --------------------------------------------------------------------------- #


def _vocab_step(token: int) -> tuple[int, int, bool]:
    return int(token), -1, False


def _pointer_step(subject_index: int) -> tuple[int, int, bool]:
    # The token id slot for pointer steps is unused by the decoder loss
    # (it scores the pointer head, not the vocab head). Storing PAD keeps
    # the array a stable rectangle and makes it obvious in dumps that the
    # vocab id is not the supervised signal at this step.
    return int(GrammarVocab.PAD), int(subject_index), True


def _flatten(steps: list[tuple[int, int, bool]]) -> tuple[list[int], list[int], list[bool]]:
    tokens = [s[0] for s in steps]
    subjects = [s[1] for s in steps]
    is_ptr = [s[2] for s in steps]
    return tokens, subjects, is_ptr


def _option_index_for_priority(
    options: list[PendingOptionState], observed: dict[str, Any]
) -> int | None:
    """Map an observed priority event to its option index in ``pending.options``.

    The Forge extractor's observed event for priority decisions is a
    ``{raw, event_type}`` dict where ``raw`` is the human-readable log
    line ("PlayerA played Forest", "PlayerB cast Lightning Bolt", …).
    We match by card name + action verb.
    """

    raw = str(observed.get("raw") or "")
    lower = raw.lower()
    is_play = " played " in lower
    is_cast = " cast " in lower
    is_activate = " activated " in lower
    is_pass = "pass" in lower and not (is_play or is_cast or is_activate)

    for i, option in enumerate(options):
        kind = option.get("kind") or ""
        name = option.get("card_name") or option.get("label") or ""
        if kind == "pass" and is_pass:
            return i
        if not name or name not in raw:
            continue
        if kind == "play" and is_play:
            return i
        if kind == "cast" and is_cast:
            return i
        if kind == "activate" and is_activate:
            return i
    return None


def translate_priority(pending: PendingState, observed: dict[str, Any]) -> DecoderTarget | None:
    options = pending["options"]
    idx = _option_index_for_priority(options, observed)
    if idx is None:
        return None
    tokens, subjects, is_ptr = _flatten(
        [
            _vocab_step(GrammarVocab.PRIORITY_OPEN),
            _pointer_step(idx),
            _vocab_step(GrammarVocab.END),
        ]
    )
    return DecoderTarget(
        decision_type=int(DecisionType.PRIORITY),
        output_token_ids=tokens,
        output_pointer_subjects=subjects,
        output_is_pointer=is_ptr,
    )


def _attacker_subject_indices(
    options: list[PendingOptionState], observed: dict[str, Any]
) -> list[int] | None:
    """Pick attacker subject indices from the observed attack-event text."""

    text = str(observed.get("attackers_text") or "")
    if not text:
        return None
    chosen: list[int] = []
    for i, option in enumerate(options):
        name = option.get("card_name") or option.get("label") or ""
        if name and name in text:
            chosen.append(i)
    return chosen if chosen else None


def translate_attackers(pending: PendingState, observed: dict[str, Any]) -> DecoderTarget | None:
    options = pending["options"]
    chosen = _attacker_subject_indices(options, observed)
    if not chosen:
        return None
    # v1 spec emits two DEFENDER anchors for player_idx 0/1; the attacker
    # always attacks the opponent (subject_index = 1 in the perspective-
    # rotated snapshot, where index 0 is "self").
    defender_subject = 1
    steps: list[tuple[int, int, bool]] = [_vocab_step(GrammarVocab.DECLARE_ATTACKERS_OPEN)]
    for atk in chosen:
        steps.append(_vocab_step(GrammarVocab.ATTACK))
        steps.append(_pointer_step(atk))
        steps.append(_vocab_step(GrammarVocab.DEFENDER))
        steps.append(_pointer_step(defender_subject))
    steps.append(_vocab_step(GrammarVocab.END))
    tokens, subjects, is_ptr = _flatten(steps)
    return DecoderTarget(
        decision_type=int(DecisionType.DECLARE_ATTACKERS),
        output_token_ids=tokens,
        output_pointer_subjects=subjects,
        output_is_pointer=is_ptr,
    )


def _blocker_assignments(
    pending: PendingState, observed: dict[str, Any]
) -> list[tuple[int, int]] | None:
    """List of (blocker_subject_index, attacker_subject_index) pairs.

    The attacker subject_index aligns with ``render_spec.py``'s spec-side
    LEGAL_ATTACKER ordering (first-seen across blocker options'
    valid_targets), which matches the indexing in
    ``DecisionSpec.legal_edge_bitmap`` columns.
    """

    options = pending["options"]
    assignments = list(observed.get("assignments") or [])
    if not assignments:
        return None

    # Build the same attacker order as render_spec: union of valid_targets
    # in first-seen order across blocker options.
    attacker_index: dict[str, int] = {}
    attacker_names_in_order: list[str] = []
    for option in options:
        for target in option.get("valid_targets") or []:
            aid = target["id"]
            if aid not in attacker_index:
                idx = len(attacker_index)
                attacker_index[aid] = idx
                # Pull a display name to match against the raw log text.
                attacker_names_in_order.append(str(target.get("label") or ""))

    pairs: list[tuple[int, int]] = []
    for assignment in assignments:
        if assignment.get("kind") != "block":
            continue
        blockers_text = str(assignment.get("blockers_text") or "")
        attacker_text = str(assignment.get("attacker_text") or "")
        # Find the attacker by label match.
        attacker_subject = -1
        for atk_idx, label in enumerate(attacker_names_in_order):
            if label and label in attacker_text:
                attacker_subject = atk_idx
                break
        if attacker_subject < 0:
            return None
        # Find the blocker option(s) whose card name appears in blockers_text.
        for opt_idx, option in enumerate(options):
            name = option.get("card_name") or option.get("label") or ""
            if name and name in blockers_text:
                pairs.append((opt_idx, attacker_subject))
    return pairs if pairs else None


def translate_blockers(pending: PendingState, observed: dict[str, Any]) -> DecoderTarget | None:
    pairs = _blocker_assignments(pending, observed)
    if not pairs:
        return None
    steps: list[tuple[int, int, bool]] = [_vocab_step(GrammarVocab.DECLARE_BLOCKERS_OPEN)]
    for blk, atk in pairs:
        steps.append(_vocab_step(GrammarVocab.BLOCK))
        steps.append(_pointer_step(blk))
        steps.append(_vocab_step(GrammarVocab.ATTACKER))
        steps.append(_pointer_step(atk))
    steps.append(_vocab_step(GrammarVocab.END))
    tokens, subjects, is_ptr = _flatten(steps)
    return DecoderTarget(
        decision_type=int(DecisionType.DECLARE_BLOCKERS),
        output_token_ids=tokens,
        output_pointer_subjects=subjects,
        output_is_pointer=is_ptr,
    )


def translate_choose_target(
    pending: PendingState, observed: dict[str, Any]
) -> DecoderTarget | None:
    options = pending["options"]
    raw = str(observed.get("raw") or "")
    target_subject = -1
    for i, option in enumerate(options):
        name = option.get("card_name") or option.get("label") or ""
        if name and name in raw:
            target_subject = i
            break
    if target_subject < 0:
        # Fall back to the first option only when the observed event
        # carries no usable text — e.g. extractor-synthesized "choose"
        # placeholders. Same conservative default as the legacy loader.
        if not options:
            return None
        target_subject = 0
    steps = [
        _vocab_step(GrammarVocab.CHOOSE_TARGETS_OPEN),
        _pointer_step(target_subject),
        _vocab_step(GrammarVocab.END),
    ]
    tokens, subjects, is_ptr = _flatten(steps)
    return DecoderTarget(
        decision_type=int(DecisionType.CHOOSE_TARGETS),
        output_token_ids=tokens,
        output_pointer_subjects=subjects,
        output_is_pointer=is_ptr,
    )


def translate_may(pending: PendingState, observed: dict[str, Any]) -> DecoderTarget | None:
    del pending
    accepted = bool(observed.get("accepted"))
    yes_no = GrammarVocab.YES if accepted else GrammarVocab.NO
    steps = [
        _vocab_step(GrammarVocab.MAY_OPEN),
        _vocab_step(yes_no),
        _vocab_step(GrammarVocab.END),
    ]
    tokens, subjects, is_ptr = _flatten(steps)
    return DecoderTarget(
        decision_type=int(DecisionType.MAY),
        output_token_ids=tokens,
        output_pointer_subjects=subjects,
        output_is_pointer=is_ptr,
    )


def translate_choose_mode(pending: PendingState, observed: dict[str, Any]) -> DecoderTarget | None:
    chosen_index = observed.get("chosen_index")
    if chosen_index is None:
        return None
    digits = bpe_digit_str_to_grammar_ids(str(int(chosen_index)))
    steps = [_vocab_step(GrammarVocab.CHOOSE_MODE_OPEN)]
    steps.extend(_vocab_step(d) for d in digits)
    steps.append(_vocab_step(GrammarVocab.END))
    tokens, subjects, is_ptr = _flatten(steps)
    return DecoderTarget(
        decision_type=int(DecisionType.CHOOSE_MODE),
        output_token_ids=tokens,
        output_pointer_subjects=subjects,
        output_is_pointer=is_ptr,
    )


def translate_choose_x(pending: PendingState, observed: dict[str, Any]) -> DecoderTarget | None:
    chosen_value = observed.get("chosen_value")
    if chosen_value is None:
        return None
    digits = bpe_digit_str_to_grammar_ids(str(int(chosen_value)))
    steps = [_vocab_step(GrammarVocab.CHOOSE_X_OPEN)]
    steps.extend(_vocab_step(d) for d in digits)
    steps.append(_vocab_step(GrammarVocab.END))
    tokens, subjects, is_ptr = _flatten(steps)
    return DecoderTarget(
        decision_type=int(DecisionType.CHOOSE_X),
        output_token_ids=tokens,
        output_pointer_subjects=subjects,
        output_is_pointer=is_ptr,
    )


_TRANSLATORS = {
    DecisionType.PRIORITY: translate_priority,
    DecisionType.DECLARE_ATTACKERS: translate_attackers,
    DecisionType.DECLARE_BLOCKERS: translate_blockers,
    DecisionType.CHOOSE_TARGETS: translate_choose_target,
    DecisionType.MAY: translate_may,
    DecisionType.CHOOSE_MODE: translate_choose_mode,
    DecisionType.CHOOSE_X: translate_choose_x,
}


def translate(pending: PendingState, observed: dict[str, Any]) -> DecoderTarget | None:
    """Dispatch translator for ``pending.kind`` and return ``DecoderTarget``."""

    decision_type = pending_decision_type(pending)
    if decision_type is None:
        return None
    return _TRANSLATORS[decision_type](pending, observed)


__all__ = [
    "DecoderTarget",
    "pending_decision_type",
    "translate",
    "translate_attackers",
    "translate_blockers",
    "translate_choose_mode",
    "translate_choose_target",
    "translate_choose_x",
    "translate_may",
    "translate_priority",
]
