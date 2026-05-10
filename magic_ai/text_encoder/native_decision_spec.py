"""Wire decision-spec tag ids from the magic-ai tokenizer into the native side.

Magic-ai loads the persisted ModernBERT tokenizer (with `DECISION_SPEC_TOKENS`
registered as additional special tokens — see
``scripts/build_text_encoder_vocab.py``); the native decision-spec emitter
(`mage-go/cmd/pylib/decision_spec_emitter.go`) needs the *id* of each tag so it
can write the correct token-stream bytes. This module bridges the two:
takes a ``PreTrainedTokenizerFast`` and calls ``mage.register_decision_spec_tokens``
with the resolved ids + a precomputed digit-token lookup table for
``<max-value>`` rendering.

Module-global keepalive: cffi expects the buffers backing the registration
to outlive the registration itself.
"""

from __future__ import annotations

from typing import Any, cast

import mage
from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.tokenizer import MAX_STACK_REFS

# Maximum integer the renderer needs to encode inside ``<max-value>...</max-value>``.
# X usually caps well below 100; CHOOSE_MODE rarely exceeds 8. Pick a generous
# bound; the lookup table is a few KB total.
DEFAULT_MAX_VALUE_DIGIT_MAX = 1024

# Names mirror render_spec.py and mage-go/.../decision_spec_emitter.go (which
# in turn mirror docs/decoder_grammar_plan.md). Order is critical for the
# decision-type-name array (one entry per DecisionType enum value).
_DT_NAME_TOKENS = (
    "<dt-priority>",
    "<dt-declare-attackers>",
    "<dt-declare-blockers>",
    "<dt-choose-targets>",
    "<dt-may>",
    "<dt-choose-mode>",
    "<dt-choose-x>",
)


_active_keepalive: tuple[Any, ...] | None = None


def register_decision_spec_token_table(
    tokenizer: PreTrainedTokenizerFast,
    *,
    max_value_digit_max: int = DEFAULT_MAX_VALUE_DIGIT_MAX,
) -> None:
    """Resolve all spec-tag ids from ``tokenizer`` and ship them to the Go side.

    Idempotent — replaces any prior registration. Holds the cffi keepalive
    buffers on a module-global so the native side's borrowed pointers stay
    valid for the process lifetime.
    """

    global _active_keepalive

    def _id(token: str) -> int:
        # convert_tokens_to_ids returns Union[int, list[int]] depending on
        # whether the input is a single token or a list. We always pass a
        # single string, so cast to int.
        tid = cast(int, tokenizer.convert_tokens_to_ids(token))
        if tid == tokenizer.unk_token_id:
            raise ValueError(
                f"tokenizer has no id for {token!r}; "
                "rebuild via scripts/build_text_encoder_vocab.py"
            )
        return tid

    dt_name_ids = [_id(t) for t in _DT_NAME_TOKENS]
    stack_ref_ids = [_id(f"<stack-ref:{k}>") for k in range(MAX_STACK_REFS)]

    # Precomputed digit-token lookup: for each i ∈ [0, max_value_digit_max],
    # store the BPE token-id sequence for str(i). Concatenate into a flat
    # int32 buffer with length-prefixed offsets — the Go side iterates
    # offsets[i]:offsets[i+1] and treats max_value_digit_max as the
    # *inclusive* upper bound (so it reads up to offsets[max_value_digit_max+1]).
    # ``offsets`` therefore needs ``max_value_digit_max + 2`` entries: one
    # leading 0 plus a right-boundary for every value in the inclusive range.
    digits_flat: list[int] = []
    digit_offsets: list[int] = [0]
    for i in range(max_value_digit_max + 1):
        ids = tokenizer.encode(str(i), add_special_tokens=False)
        digits_flat.extend(int(x) for x in ids)
        digit_offsets.append(len(digits_flat))

    keepalive = mage.register_decision_spec_tokens(
        spec_open_id=_id("<spec-open>"),
        spec_close_id=_id("<spec-close>"),
        decision_type_id=_id("<decision-type>"),
        legal_attacker_id=_id("<legal-attacker>"),
        legal_blocker_id=_id("<legal-blocker>"),
        legal_target_id=_id("<legal-target>"),
        legal_action_id=_id("<legal-action>"),
        for_action_id=_id("<for-action>"),
        max_value_open_id=_id("<max-value>"),
        max_value_close_id=_id("</max-value>"),
        player_ref0_id=_id("<player-ref:0>"),
        player_ref1_id=_id("<player-ref:1>"),
        dt_name_ids=dt_name_ids,
        stack_ref_ids=stack_ref_ids,
        max_value_digit_max=max_value_digit_max,
        max_value_digits=digits_flat,
        max_value_digit_offsets=digit_offsets,
    )
    _active_keepalive = keepalive


__all__ = [
    "DEFAULT_MAX_VALUE_DIGIT_MAX",
    "register_decision_spec_token_table",
]
