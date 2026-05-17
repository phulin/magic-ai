from __future__ import annotations

from pathlib import Path

import torch
from magic_ai.text_encoder.card_cache import build_card_cache
from magic_ai.text_encoder.render import DEFAULT_ORACLE_PATH, load_oracle_text
from magic_ai.text_encoder.token_table_export import (
    TOKEN_TABLE_ARTIFACT_SCHEMA_VERSION,
    card_cache_fingerprint,
    token_tables_artifact,
)
from magic_ai.text_encoder.token_tables import Frag, build_token_tables
from magic_ai.text_encoder.tokenizer import MAX_DICT_ENTRIES, load_tokenizer


def test_token_table_artifact_matches_python_tables() -> None:
    tokenizer = load_tokenizer()
    oracle = load_oracle_text(Path(DEFAULT_ORACLE_PATH))
    cache = build_card_cache(["Forest", "Lightning Bolt"], oracle, tokenizer)
    tables = build_token_tables(tokenizer, cache)

    artifact = token_tables_artifact(tokenizer, cache, tables=tables)
    exported = artifact["tables"]

    assert artifact["schema_version"] == TOKEN_TABLE_ARTIFACT_SCHEMA_VERSION
    assert artifact["tokenizer"]["unk_token_id"] == tokenizer.unk_token_id
    assert (
        exported["structural"]["tokens"][
            exported["structural"]["offsets"][int(Frag.BOS_STATE)] : exported["structural"][
                "offsets"
            ][int(Frag.BOS_STATE) + 1]
        ]
        == tables.structural[Frag.BOS_STATE]
    )
    assert exported["card_ref"][:8] == tables.card_ref[:8]
    assert len(exported["dict_entry"]) == MAX_DICT_ENTRIES
    assert exported["dict_entry"] == tables.dict_entry
    assert exported["row_to_name"] == cache.row_to_name

    forest_row = cache.row_to_name.index("Forest")
    body_offsets = exported["card_body"]["offsets"]
    assert exported["card_body"]["tokens"][body_offsets[forest_row] : body_offsets[forest_row + 1]]
    assert exported["card_name"]["offsets"][cache.num_rows] == len(exported["card_name"]["tokens"])


def test_card_cache_fingerprint_changes_with_tokens() -> None:
    tokenizer = load_tokenizer()
    oracle = load_oracle_text(Path(DEFAULT_ORACLE_PATH))
    cache = build_card_cache(["Forest"], oracle, tokenizer)
    changed = build_card_cache(["Forest"], oracle, tokenizer)
    changed = type(changed)(
        token_buffer=torch.cat([changed.token_buffer, torch.tensor([123], dtype=torch.int32)]),
        offsets=changed.offsets,
        row_to_name=changed.row_to_name,
        engine_card_set_hash=changed.engine_card_set_hash,
        content_hash=changed.content_hash,
        cache_schema_version=changed.cache_schema_version,
    )

    assert card_cache_fingerprint(cache) != card_cache_fingerprint(changed)
