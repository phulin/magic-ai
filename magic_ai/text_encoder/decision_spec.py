"""Decision-spec types for the autoregressive grammar decoder.

A ``DecisionSpec`` describes the question the engine is asking the model:
which decision type, which subjects/targets/actions are legal, and any
scalar parameters (max-value for MODE/X). The encoder consumes the spec
tokens alongside the state-text tokens; the decoder consumes pointer
anchors (encoder positions of referenceable entities) and the
side-tensors (relations between subjects, e.g. block legal-edge bitmaps).

This module defines the Python-side types only. The renderer
(``render.py::render_decision_spec``) produces them; the native side
will eventually emit the same shape directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from magic_ai.game_state import PendingOptionState


def blocker_attacker_order(options: Iterable[PendingOptionState]) -> list[str]:
    """Canonical attacker order for ``DECLARE_BLOCKERS``: union of
    ``valid_targets`` across blocker options in stable first-seen order.

    Single source of truth shared by ``render_spec`` (which lays out
    LEGAL_ATTACKER anchors and the legal-edge bitmap columns) and
    ``forge_target_encoding`` (which translates observed assignments
    into pointer subject indices). Keep these aligned by calling here
    rather than reimplementing the loop.
    """

    attacker_index: dict[str, int] = {}
    order: list[str] = []
    for option in options:
        for target in option.get("valid_targets") or []:
            aid = str(target["id"])
            if aid not in attacker_index:
                attacker_index[aid] = len(order)
                order.append(aid)
    return order


class DecisionType(IntEnum):
    """Stable integer ids for each decision kind.

    Stable across Python and (eventually) the native side; persisted
    into replay rows and the Forge extractor's records. Add new kinds
    by appending — never reorder.
    """

    PRIORITY = 0
    DECLARE_ATTACKERS = 1
    DECLARE_BLOCKERS = 2
    CHOOSE_TARGETS = 3
    MAY = 4
    CHOOSE_MODE = 5
    CHOOSE_X = 6


class AnchorKind(IntEnum):
    """What an encoder position refers to.

    Pointer-head outputs are restricted to anchor positions whose
    ``AnchorKind`` matches what the current grammar step expects (e.g.
    DECLARE_ATTACKERS only ranges over ``LEGAL_ATTACKER`` anchors).
    """

    LEGAL_ATTACKER = 0
    LEGAL_BLOCKER = 1
    LEGAL_TARGET = 2
    LEGAL_ACTION = 3
    DEFENDER = 4  # players + planeswalkers a creature can attack into


@dataclass(frozen=True)
class PointerAnchor:
    """One referenceable entity in the decision spec.

    ``token_position`` is the position in the encoder's combined token
    sequence (state tokens + spec tokens) where this entity's anchor
    token lives. The decoder pointer head dot-products its hidden
    state with the encoder hidden at this position.

    ``subject_index`` is the entity's local 0-based ordinal within
    its kind on this decision (e.g. blocker #3 of N legal blockers);
    used by side-tensors and by the grammar mask to reason about
    relations.
    """

    kind: AnchorKind
    token_position: int
    subject_index: int
    # Stable engine handle (option index for actions, card-ref index for
    # objects, player index for players). The decoder samples a pointer
    # position; the actor_critic translates it back to an engine action
    # via this handle. Renderer-driven; opaque to the decoder.
    handle: int = -1


@dataclass
class DecisionSpec:
    """Per-decision data fed to encoder + decoder.

    Token streams are *additive* to the state-text token stream — the
    encoder concatenates ``[state_tokens] + [spec_tokens]``. Pointer
    anchors hold encoder positions in that combined stream; renderers
    must offset by the state-text length when populating.

    Side-tensors carry relations that don't belong in the token
    stream (legal-edge bitmaps, etc.) — the grammar mask consults them
    at decode time.
    """

    decision_type: DecisionType
    # Spec tokens (encoder-vocab ids). Concatenated to the state-text
    # token stream by the assembler.
    spec_tokens: list[int] = field(default_factory=list)
    # Per-anchor metadata. ``token_position`` is the absolute position
    # in the combined ``[state_tokens] + [spec_tokens]`` stream.
    anchors: list[PointerAnchor] = field(default_factory=list)
    # Scalar params. -1 means "not applicable to this decision type".
    max_value: int = -1
    for_action_index: int = -1  # CHOOSE_TARGETS: which legal action this resolves
    # Relations between subjects. Layout depends on decision_type:
    #   DECLARE_BLOCKERS: legal_edge_bitmap[N_blockers, N_attackers] bool
    #     where attackers are looked up by `subject_index` against the
    #     state-text DEFENDER anchors (not the spec section).
    # None for decisions that have no cross-subject relations.
    legal_edge_bitmap: np.ndarray | None = None

    def anchors_of_kind(self, kind: AnchorKind) -> list[PointerAnchor]:
        return [a for a in self.anchors if a.kind == kind]


def empty_spec(decision_type: DecisionType) -> DecisionSpec:
    """Convenience constructor for tests and edge cases."""

    return DecisionSpec(decision_type=decision_type)


def encode_decoder_target(
    spec: DecisionSpec,
    grammar_token_ids: Sequence[int],
    pointer_subject_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a teacher-forced decoder target sequence.

    Returns (output_token_ids[L], output_pointer_ids[L]) where
    ``output_pointer_ids[i] == -1`` for non-pointer steps and an
    anchor's ``subject_index`` (within the appropriate kind for that
    step) for pointer steps. Storage-only helper used by the replay
    buffer and the Forge extractor — the actual id-to-position mapping
    is recovered at training time from the spec's anchor list.
    """

    if len(grammar_token_ids) != len(pointer_subject_indices):
        raise ValueError(
            f"length mismatch: tokens={len(grammar_token_ids)} "
            f"pointers={len(pointer_subject_indices)}"
        )
    return (
        np.asarray(grammar_token_ids, dtype=np.int32),
        np.asarray(pointer_subject_indices, dtype=np.int32),
    )


__all__ = [
    "AnchorKind",
    "DecisionSpec",
    "DecisionType",
    "PointerAnchor",
    "empty_spec",
    "encode_decoder_target",
]
