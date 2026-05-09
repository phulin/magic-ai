"""Smoke test for ``register_decision_spec_token_table``.

Verifies the magic-ai-side glue that derives spec-tag ids from the
tokenizer and ships them through ``mage.register_decision_spec_tokens``
to the Go side. End-to-end registration smoke; does NOT yet exercise
``mage.encode_decision_spec`` against ``render_decision_spec`` parity
(that needs a `BatchHandle` constructed against a real game, which is
out of scope here).
"""

from __future__ import annotations

import pytest
from magic_ai.text_encoder.tokenizer import load_tokenizer

mage = pytest.importorskip("mage")
from magic_ai.text_encoder.native_decision_spec import (  # noqa: E402
    register_decision_spec_token_table,
)


def test_register_round_trip() -> None:
    tokenizer = load_tokenizer()
    register_decision_spec_token_table(tokenizer, max_value_digit_max=64)
    # Re-register; should overwrite without leaking.
    register_decision_spec_token_table(tokenizer, max_value_digit_max=128)


def test_token_ids_are_singletons() -> None:
    """Every spec-tag token must round-trip to a single id (not BPE-split)."""

    tokenizer = load_tokenizer()
    tags = [
        "<spec-open>",
        "<spec-close>",
        "<decision-type>",
        "<legal-attacker>",
        "<legal-blocker>",
        "<legal-target>",
        "<legal-action>",
        "<for-action>",
        "<max-value>",
        "</max-value>",
        "<player-ref:0>",
        "<player-ref:1>",
        "<dt-priority>",
        "<dt-declare-attackers>",
        "<dt-declare-blockers>",
        "<dt-choose-targets>",
        "<dt-may>",
        "<dt-choose-mode>",
        "<dt-choose-x>",
    ]
    for tag in tags:
        ids = tokenizer.encode(tag, add_special_tokens=False)
        assert len(ids) == 1, f"{tag!r} → {ids} (BPE split, must be a singleton)"
