"""Render a ``GameStateSnapshot`` into a ``DecisionSpec``.

Step 1 of the decoder-grammar migration (``docs/decoder_grammar_plan.md``).
This module is purely additive — it does not touch the inline-blank
renderer (``render.py``). The decision spec is fed to the encoder
*alongside* the state-text token stream; pointer anchor positions
recorded here are relative to the spec section (offset 0 = first spec
token) and the assembler shifts them by the state-token length when it
concatenates the two streams.

The tokenizer's spec-tag token ids are pre-resolved once per
``DecisionSpecRenderer`` instance and cached on the object — the
per-anchor loop never does string-to-id lookups.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from transformers import PreTrainedTokenizerFast

from magic_ai.game_state import (
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
)
from magic_ai.text_encoder.decision_spec import (
    AnchorKind,
    DecisionSpec,
    DecisionType,
    PointerAnchor,
    blocker_attacker_order,
)

# Pending-kind → decision-type dispatch.
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

# Decision-type-name token (the `<dt-...>` payload after `<decision-type>`).
_DECISION_TYPE_NAME_TOKEN: dict[DecisionType, str] = {
    DecisionType.PRIORITY: "<dt-priority>",
    DecisionType.DECLARE_ATTACKERS: "<dt-declare-attackers>",
    DecisionType.DECLARE_BLOCKERS: "<dt-declare-blockers>",
    DecisionType.CHOOSE_TARGETS: "<dt-choose-targets>",
    DecisionType.MAY: "<dt-may>",
    DecisionType.CHOOSE_MODE: "<dt-choose-mode>",
    DecisionType.CHOOSE_X: "<dt-choose-x>",
}


class _SpecVocab:
    """Pre-resolved token ids for every spec-section structural tag.

    Resolved once at construction; the per-decision render loop only
    appends ints. ``PreTrainedTokenizerFast.convert_tokens_to_ids`` is
    used because the spec tags are registered as ``additional_special_tokens``
    in the persisted tokenizer.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        def tid(token: str) -> int:
            ids = tokenizer.convert_tokens_to_ids(token)
            # `convert_tokens_to_ids` returns a single int for a single token,
            # but its declared return type is `int | list[int]`.
            assert isinstance(ids, int)
            return ids

        self.spec_open = tid("<spec-open>")
        self.spec_close = tid("<spec-close>")
        self.decision_type = tid("<decision-type>")
        self.legal_attacker = tid("<legal-attacker>")
        self.legal_blocker = tid("<legal-blocker>")
        self.legal_target = tid("<legal-target>")
        self.legal_action = tid("<legal-action>")
        self.for_action_open = tid("<for-action>")
        self.max_value_open = tid("<max-value>")
        self.max_value_close = tid("</max-value>")
        self.player_ref = (tid("<player-ref:0>"), tid("<player-ref:1>"))
        self.dt_name = {dt: tid(name) for dt, name in _DECISION_TYPE_NAME_TOKEN.items()}


# ``</for-action>`` is not yet registered (the plan only listed
# ``<for-action>``). We open with ``<for-action>`` followed by digit
# tokens; close-tag fallback below uses ``<spec-close>``-style scoping
# implicitly since the next anchor tag delimits the section.
def _digit_token_ids(tokenizer: PreTrainedTokenizerFast, value: int) -> list[int]:
    """BPE-tokenize ``str(value)`` for ``<max-value>`` / ``<for-action>`` digits."""

    return [int(t) for t in tokenizer.encode(str(value), add_special_tokens=False)]


class DecisionSpecRenderer:
    """Stateful renderer; cache the vocab once, render many snapshots."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self._tokenizer = tokenizer
        self._vocab = _SpecVocab(tokenizer)

    def render(
        self,
        snapshot: GameStateSnapshot,
        *,
        card_refs: dict[str, int],
    ) -> DecisionSpec:
        pending: PendingState = snapshot["pending"]  # type: ignore[typeddict-item]
        kind = pending["kind"]
        if kind in ("mana_color", "mulligan"):
            raise NotImplementedError(f"decision kind {kind!r} deferred to post-v1")
        if kind not in _PENDING_KIND_TO_DECISION_TYPE:
            raise ValueError(f"unknown pending kind: {kind!r}")
        decision_type = _PENDING_KIND_TO_DECISION_TYPE[kind]
        options = pending["options"]
        v = self._vocab

        tokens: list[int] = [v.spec_open, v.decision_type, v.dt_name[decision_type]]
        anchors: list[PointerAnchor] = []
        max_value = -1
        legal_edge_bitmap: np.ndarray | None = None

        if decision_type is DecisionType.PRIORITY:
            for opt_idx in range(len(options)):
                anchors.append(
                    PointerAnchor(
                        kind=AnchorKind.LEGAL_ACTION,
                        token_position=len(tokens),
                        subject_index=opt_idx,
                        handle=opt_idx,
                    )
                )
                tokens.append(v.legal_action)

        elif decision_type is DecisionType.DECLARE_ATTACKERS:
            for opt_idx in range(len(options)):
                anchors.append(
                    PointerAnchor(
                        kind=AnchorKind.LEGAL_ATTACKER,
                        token_position=len(tokens),
                        subject_index=opt_idx,
                        handle=opt_idx,
                    )
                )
                tokens.append(v.legal_attacker)
            # v1: emit two implicit defenders pointing at the
            # `<player-ref:0/1>` tokens we drop into the spec. Future
            # work should enumerate planeswalker defenders from state
            # too. For the attacker-controller, only the *opponent* is a
            # legal defender; we still emit both refs as anchors so the
            # decoder mask can pick — illegality of self-defense is
            # enforced by the grammar mask, not by anchor omission.
            for player_idx in (0, 1):
                anchors.append(
                    PointerAnchor(
                        kind=AnchorKind.DEFENDER,
                        token_position=len(tokens),
                        subject_index=player_idx,
                        handle=player_idx,
                    )
                )
                tokens.append(v.player_ref[player_idx])

        elif decision_type is DecisionType.DECLARE_BLOCKERS:
            # Canonical attacker order shared with forge_target_encoding via
            # decision_spec.blocker_attacker_order.
            attacker_order = blocker_attacker_order(options)
            attacker_index = {aid: i for i, aid in enumerate(attacker_order)}

            # Emit one <legal-blocker> per blocker option.
            for opt_idx in range(len(options)):
                anchors.append(
                    PointerAnchor(
                        kind=AnchorKind.LEGAL_BLOCKER,
                        token_position=len(tokens),
                        subject_index=opt_idx,
                        handle=opt_idx,
                    )
                )
                tokens.append(v.legal_blocker)
            # v1 trade-off: state already tags attackers via the
            # `ATTACKING` status, so the spec could in principle skip
            # them. But re-emitting them as spec anchors gives the
            # pointer head a definite position to score against without
            # needing the state-text card-ref index to be wired through
            # the spec data structure. Cost is ~1 token per attacker.
            for atk_idx, _aid in enumerate(attacker_order):
                anchors.append(
                    PointerAnchor(
                        kind=AnchorKind.LEGAL_ATTACKER,
                        token_position=len(tokens),
                        subject_index=atk_idx,
                        handle=atk_idx,
                    )
                )
                tokens.append(v.legal_attacker)

            # Legal-edge bitmap: rows = blockers (option order), cols =
            # attackers (first-seen order).
            n_blockers = len(options)
            n_attackers = len(attacker_order)
            bitmap = np.zeros((n_blockers, n_attackers), dtype=np.bool_)
            for blocker_idx, option in enumerate(options):
                for target in option.get("valid_targets") or []:
                    bitmap[blocker_idx, attacker_index[str(target["id"])]] = True
            legal_edge_bitmap = bitmap

        elif decision_type is DecisionType.CHOOSE_TARGETS:
            # v1: assume single-target per option; emit one
            # <legal-target> anchor per option's first valid target.
            for opt_idx, option in enumerate(options):
                anchors.append(
                    PointerAnchor(
                        kind=AnchorKind.LEGAL_TARGET,
                        token_position=len(tokens),
                        subject_index=opt_idx,
                        handle=opt_idx,
                    )
                )
                tokens.append(v.legal_target)

        elif decision_type is DecisionType.MAY:
            # No anchors — the decoder grammar emits <YES>/<NO>.
            pass

        elif decision_type is DecisionType.CHOOSE_MODE:
            max_value = len(options)
            tokens.append(v.max_value_open)
            tokens.extend(_digit_token_ids(self._tokenizer, max_value))
            tokens.append(v.max_value_close)

        elif decision_type is DecisionType.CHOOSE_X:
            amount = pending.get("amount")
            max_value = int(amount) if amount is not None else max(0, len(options) - 1)
            tokens.append(v.max_value_open)
            tokens.extend(_digit_token_ids(self._tokenizer, max_value))
            tokens.append(v.max_value_close)

        tokens.append(v.spec_close)

        # ``card_refs`` is not consulted for v1 (every anchor position is
        # within the spec section). Kept in the signature so the caller
        # contract matches the eventual native path, which will need the
        # state-side card-ref index to anchor LEGAL_ATTACKER positions
        # at the in-state ``<card-ref:K>`` tokens for DECLARE_BLOCKERS.
        del card_refs

        return DecisionSpec(
            decision_type=decision_type,
            spec_tokens=tokens,
            anchors=anchors,
            max_value=max_value,
            for_action_index=-1,
            legal_edge_bitmap=legal_edge_bitmap,
        )


def render_decision_spec(
    snapshot: GameStateSnapshot,
    *,
    card_refs: dict[str, int],
    tokenizer: PreTrainedTokenizerFast,
) -> DecisionSpec:
    """Convenience wrapper for one-shot rendering.

    Construct a :class:`DecisionSpecRenderer` once and call ``.render``
    repeatedly when rendering many snapshots — that path resolves the
    spec-tag vocab once instead of per call.
    """

    return DecisionSpecRenderer(tokenizer).render(snapshot, card_refs=card_refs)


# Suppress unused-import warning for types referenced only in signatures.
_ = (PendingOptionState, Sequence)


__all__ = [
    "DecisionSpecRenderer",
    "render_decision_spec",
]
