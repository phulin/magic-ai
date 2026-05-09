"""Smoke test: synthesize one fake decision per pending kind, run them
through the V2 decoder-target encoder, assert the produced sequences are
non-empty and grammar-legal under :func:`grammar.next_mask`.

Lets us validate the V2 extractor → V2 loader pipeline end-to-end without
needing a Forge JVM or the input corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np

# Allow direct invocation as ``uv run python scripts/test_forge_target_encoding_smoke.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.game_state import PendingOptionState, PendingState
from magic_ai.text_encoder.decision_spec import (
    AnchorKind,
    DecisionSpec,
    DecisionType,
    PointerAnchor,
)
from magic_ai.text_encoder.forge_target_encoding import (
    DecoderTarget,
    pending_decision_type,
    translate,
)
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, next_mask


def _opt(**kwargs: object) -> PendingOptionState:
    return cast(PendingOptionState, dict(kwargs))


def _pending(kind: str, options: list[PendingOptionState], **extras: object) -> PendingState:
    return cast(PendingState, {"kind": kind, "player_idx": 0, "options": options, **extras})


def _spec_for(target: DecoderTarget, n_options: int) -> DecisionSpec:
    """Build a synthetic DecisionSpec sized for the target sequence."""

    dt = DecisionType(target.decision_type)
    if dt is DecisionType.PRIORITY:
        return DecisionSpec(
            decision_type=dt,
            anchors=[
                PointerAnchor(kind=AnchorKind.LEGAL_ACTION, token_position=10 + i, subject_index=i)
                for i in range(n_options)
            ],
        )
    if dt is DecisionType.DECLARE_ATTACKERS:
        return DecisionSpec(
            decision_type=dt,
            anchors=(
                [
                    PointerAnchor(
                        kind=AnchorKind.LEGAL_ATTACKER,
                        token_position=10 + i,
                        subject_index=i,
                    )
                    for i in range(n_options)
                ]
                + [
                    PointerAnchor(kind=AnchorKind.DEFENDER, token_position=20 + i, subject_index=i)
                    for i in range(2)
                ]
            ),
        )
    if dt is DecisionType.DECLARE_BLOCKERS:
        bitmap = np.ones((n_options, 1), dtype=np.bool_)
        return DecisionSpec(
            decision_type=dt,
            anchors=(
                [
                    PointerAnchor(
                        kind=AnchorKind.LEGAL_BLOCKER,
                        token_position=10 + i,
                        subject_index=i,
                    )
                    for i in range(n_options)
                ]
                + [
                    PointerAnchor(
                        kind=AnchorKind.LEGAL_ATTACKER,
                        token_position=30,
                        subject_index=0,
                    )
                ]
            ),
            legal_edge_bitmap=bitmap,
        )
    if dt is DecisionType.CHOOSE_TARGETS:
        return DecisionSpec(
            decision_type=dt,
            anchors=[
                PointerAnchor(kind=AnchorKind.LEGAL_TARGET, token_position=10 + i, subject_index=i)
                for i in range(n_options)
            ],
        )
    if dt is DecisionType.MAY:
        return DecisionSpec(decision_type=dt)
    if dt is DecisionType.CHOOSE_MODE:
        return DecisionSpec(decision_type=dt, max_value=n_options)
    if dt is DecisionType.CHOOSE_X:
        return DecisionSpec(decision_type=dt, max_value=12)
    raise ValueError(f"unknown decision type {dt}")


def _assert_grammar_legal(spec: DecisionSpec, target: DecoderTarget) -> None:
    prefix_tokens: list[int] = []
    prefix_pointers: list[int] = []
    for tok, subj, is_ptr in zip(
        target.output_token_ids,
        target.output_pointer_subjects,
        target.output_is_pointer,
        strict=True,
    ):
        mask = next_mask(spec, prefix_tokens, prefix_pointers)
        if is_ptr:
            assert mask.pointer_mask is not None
            assert bool(mask.pointer_mask[subj]), (
                f"pointer subject {subj} illegal at step {len(prefix_tokens)}"
            )
        else:
            assert 0 <= tok < GRAMMAR_VOCAB_SIZE
            assert bool(mask.vocab_mask[tok]), f"token {tok} illegal at step {len(prefix_tokens)}"
        prefix_tokens.append(tok)
        prefix_pointers.append(subj)


def main() -> int:
    cases: list[tuple[str, PendingState, dict[str, object]]] = [
        (
            "priority",
            _pending(
                "priority",
                [
                    _opt(id="play:1", kind="play", card_name="Forest"),
                    _opt(id="pass", kind="pass"),
                ],
            ),
            {"raw": "PlayerA played Forest.", "event_type": "ZONE"},
        ),
        (
            "attackers",
            _pending(
                "attackers",
                [_opt(id="atk:1", kind="attacker", card_name="Bear")],
            ),
            {"attackers_text": "Bear", "actor_name": "PlayerA"},
        ),
        (
            "blockers",
            _pending(
                "blockers",
                [
                    _opt(
                        id="block:1",
                        kind="block",
                        card_name="Wall",
                        valid_targets=[{"id": "atk:1", "label": "Bear"}],
                    )
                ],
            ),
            {
                "assignments": [
                    {
                        "kind": "block",
                        "actor_name": "PlayerB",
                        "attacker_text": "Bear",
                        "blockers_text": "Wall",
                    }
                ]
            },
        ),
        (
            "may",
            _pending("may", []),
            {"accepted": True},
        ),
        (
            "permanent",
            _pending(
                "permanent",
                [
                    _opt(id="t:1", kind="permanent", card_name="Forest"),
                    _opt(id="t:2", kind="permanent", card_name="Mountain"),
                ],
            ),
            {"raw": "Choose Mountain"},
        ),
    ]

    for name, pending, observed in cases:
        dt = pending_decision_type(pending)
        assert dt is not None, f"pending kind {name!r} not mapped"
        target = translate(pending, observed)
        assert target is not None, f"translator returned None for {name!r}"
        assert target.output_token_ids, f"empty token sequence for {name!r}"
        assert (
            len(target.output_token_ids)
            == len(target.output_pointer_subjects)
            == len(target.output_is_pointer)
        )
        spec = _spec_for(target, len(pending["options"]) or 2)
        _assert_grammar_legal(spec, target)
        sys.stdout.write(
            f"OK [{name}] decision_type={DecisionType(target.decision_type).name} "
            f"len={len(target.output_token_ids)}\n"
        )
    sys.stdout.write("V2 decoder-target encoding smoke passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
