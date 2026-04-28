"""Text encoder package — tokenizer + (forthcoming) renderer / model.

Only the tokenizer is implemented in this PR; see ``docs/text_encoder_plan.md``.
"""

from magic_ai.text_encoder.tokenizer import (
    ALL_CUSTOM_TOKENS,
    CARD_REF_TOKENS,
    LOYALTY_TOKENS,
    MANA_TOKENS,
    MAX_CARD_REFS,
    MAX_LOYALTY,
    MODERNBERT_REPO,
    MODERNBERT_REVISION,
    STATUS_TOKENS,
    STRUCTURAL_TOKENS,
    TOKENIZER_DIR,
    card_ref_token,
    load_tokenizer,
)

__all__ = [
    "ALL_CUSTOM_TOKENS",
    "CARD_REF_TOKENS",
    "LOYALTY_TOKENS",
    "MANA_TOKENS",
    "MAX_CARD_REFS",
    "MAX_LOYALTY",
    "MODERNBERT_REPO",
    "MODERNBERT_REVISION",
    "STATUS_TOKENS",
    "STRUCTURAL_TOKENS",
    "TOKENIZER_DIR",
    "card_ref_token",
    "load_tokenizer",
]
