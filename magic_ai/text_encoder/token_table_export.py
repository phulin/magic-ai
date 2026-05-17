"""Export closed-vocabulary token tables for non-Python assemblers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import orjson
import torch
from transformers import PreTrainedTokenizerFast

from magic_ai.text_encoder.card_cache import CardTokenCache
from magic_ai.text_encoder.native_decision_spec import DEFAULT_MAX_VALUE_DIGIT_MAX
from magic_ai.text_encoder.token_tables import Frag, TokenTables, build_token_tables
from magic_ai.text_encoder.tokenizer import MAX_STACK_REFS

TOKEN_TABLE_ARTIFACT_SCHEMA_VERSION = 1


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_jsonable(payload: Any) -> str:
    return _sha256_hex(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))


def tokenizer_fingerprint(tokenizer: PreTrainedTokenizerFast) -> str:
    """Hash tokenizer state that affects token IDs."""

    backend = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    specials = tokenizer.special_tokens_map
    added = tokenizer.get_added_vocab()
    payload = {
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "special_tokens": {str(k): str(v) for k, v in sorted(specials.items())},
        "added_vocab": {str(k): int(v) for k, v in sorted(added.items())},
        "backend_sha256": _sha256_hex(backend),
    }
    return _hash_jsonable(payload)


def custom_token_id_fingerprint(tokenizer: PreTrainedTokenizerFast) -> str:
    """Hash just the augmented vocabulary assignments."""

    added = tokenizer.get_added_vocab()
    return _hash_jsonable({str(k): int(v) for k, v in sorted(added.items())})


def card_cache_fingerprint(cache: CardTokenCache) -> str:
    """Hash card-cache row order, offsets, token contents, and metadata."""

    h = hashlib.sha256()
    h.update(f"schema:{cache.cache_schema_version}\n".encode())
    h.update(f"engine:{cache.engine_card_set_hash}\n".encode())
    h.update(f"content:{cache.content_hash or ''}\n".encode())
    for name in cache.row_to_name:
        raw = name.encode("utf-8")
        h.update(len(raw).to_bytes(4, "little"))
        h.update(raw)
    h.update(cache.offsets.detach().cpu().contiguous().numpy().tobytes())
    h.update(cache.token_buffer.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _flatten(seqs: list[list[int]]) -> tuple[list[int], list[int]]:
    offsets = [0]
    flat: list[int] = []
    for seq in seqs:
        flat.extend(int(x) for x in seq)
        offsets.append(len(flat))
    return offsets, flat


def _table_sequences_by_index(
    mapping: dict[int, list[int]], min_id: int, max_id: int
) -> list[list[int]]:
    return [list(mapping.get(i, [])) for i in range(min_id, max_id + 1)]


def _tuple_key_sequences(
    mapping: dict[tuple[int, int], list[int]],
    outer_count: int,
    inner_count: int,
) -> list[list[int]]:
    out: list[list[int]] = []
    for outer in range(outer_count):
        for inner in range(inner_count):
            out.append(list(mapping.get((outer, inner), [])))
    return out


def _serialize_sequence_table(seqs: list[list[int]]) -> dict[str, list[int]]:
    offsets, tokens = _flatten(seqs)
    return {"offsets": offsets, "tokens": tokens}


def _token_id(tokenizer: PreTrainedTokenizerFast, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int):
        raise TypeError(f"expected one token id for {token!r}")
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no id for {token!r}")
    return int(token_id)


def token_tables_artifact(
    tokenizer: PreTrainedTokenizerFast,
    cache: CardTokenCache,
    *,
    tables: TokenTables | None = None,
) -> dict[str, Any]:
    tables = tables if tables is not None else build_token_tables(tokenizer, cache)

    structural = _table_sequences_by_index(
        {int(k): v for k, v in tables.structural.items()},
        int(min(Frag)),
        int(max(Frag)),
    )
    zone_count = max((z for z, _owner in tables.zone_open), default=-1) + 1
    owner_count = 2
    turn_step = [
        list(tables.turn_step[(turn, step)])
        for turn in range(tables.turn_min, tables.turn_max + 1)
        for step in range(len({s for _turn, s in tables.turn_step}))
    ]
    life_owner = [
        list(tables.life_owner[(life, owner)])
        for life in range(tables.life_min, tables.life_max + 1)
        for owner in range(owner_count)
    ]

    body_offsets, body_tokens = _flatten(tables.card_body)
    name_offsets, name_tokens = _flatten(tables.card_name)
    digit_sequences = [
        [int(t) for t in tokenizer.encode(str(i), add_special_tokens=False)]
        for i in range(DEFAULT_MAX_VALUE_DIGIT_MAX + 1)
    ]
    artifact: dict[str, Any] = {
        "schema_version": TOKEN_TABLE_ARTIFACT_SCHEMA_VERSION,
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "unk_token_id": tokenizer.unk_token_id,
            "fingerprint": tokenizer_fingerprint(tokenizer),
            "custom_token_id_fingerprint": custom_token_id_fingerprint(tokenizer),
        },
        "card_cache": {
            "fingerprint": card_cache_fingerprint(cache),
            "engine_card_set_hash": cache.engine_card_set_hash,
            "content_hash": cache.content_hash,
            "cache_schema_version": cache.cache_schema_version,
            "num_rows": cache.num_rows,
        },
        "tables": {
            "pad_id": tables.pad_id,
            "option_id": tables.option_id,
            "target_open_id": tables.target_open_id,
            "target_close_id": tables.target_close_id,
            "dict_open_id": tables.dict_open_id,
            "dict_close_id": tables.dict_close_id,
            "card_open_id": tables.card_open_id,
            "self_id": tables.self_id,
            "opp_id": tables.opp_id,
            "stack_open_id": tables.stack_open_id,
            "stack_close_id": tables.stack_close_id,
            "command_open_id": tables.command_open_id,
            "command_close_id": tables.command_close_id,
            "turn_min": tables.turn_min,
            "turn_max": tables.turn_max,
            "life_min": tables.life_min,
            "life_max": tables.life_max,
            "count_min": tables.count_min,
            "count_max": tables.count_max,
            "ability_min": tables.ability_min,
            "ability_max": tables.ability_max,
            "step_count": len({step for _turn, step in tables.turn_step}),
            "zone_count": zone_count,
            "owner_count": owner_count,
            "structural": _serialize_sequence_table(structural),
            "zone_open": _serialize_sequence_table(
                _tuple_key_sequences(tables.zone_open, zone_count, owner_count)
            ),
            "zone_close": _serialize_sequence_table(
                _tuple_key_sequences(tables.zone_close, zone_count, owner_count)
            ),
            "action_verb": _serialize_sequence_table(
                _table_sequences_by_index(tables.action_verb, 0, max(tables.action_verb))
            ),
            "mana_glyph": _serialize_sequence_table(tables.mana_glyph),
            "turn_step": _serialize_sequence_table(turn_step),
            "life_owner": _serialize_sequence_table(life_owner),
            "ability": _serialize_sequence_table(
                _table_sequences_by_index(tables.ability, tables.ability_min, tables.ability_max)
            ),
            "count": _serialize_sequence_table(
                _table_sequences_by_index(tables.count, tables.count_min, tables.count_max)
            ),
            "card_closer": tables.card_closer,
            "status_tapped": tables.status_tapped,
            "status_untapped": tables.status_untapped,
            "card_ref": tables.card_ref,
            "dict_entry": tables.dict_entry,
            "card_body": {"offsets": body_offsets, "tokens": body_tokens},
            "card_name": {"offsets": name_offsets, "tokens": name_tokens},
            "row_to_name": list(cache.row_to_name),
        },
        "decision_spec": {
            "spec_open_id": _token_id(tokenizer, "<spec-open>"),
            "spec_close_id": _token_id(tokenizer, "<spec-close>"),
            "decision_type_id": _token_id(tokenizer, "<decision-type>"),
            "legal_attacker_id": _token_id(tokenizer, "<legal-attacker>"),
            "legal_blocker_id": _token_id(tokenizer, "<legal-blocker>"),
            "legal_target_id": _token_id(tokenizer, "<legal-target>"),
            "legal_action_id": _token_id(tokenizer, "<legal-action>"),
            "max_value_open_id": _token_id(tokenizer, "<max-value>"),
            "max_value_close_id": _token_id(tokenizer, "</max-value>"),
            "player_ref_ids": [
                _token_id(tokenizer, "<player-ref:0>"),
                _token_id(tokenizer, "<player-ref:1>"),
            ],
            "decision_type_name_ids": [
                _token_id(tokenizer, "<dt-priority>"),
                _token_id(tokenizer, "<dt-declare-attackers>"),
                _token_id(tokenizer, "<dt-declare-blockers>"),
                _token_id(tokenizer, "<dt-choose-targets>"),
                _token_id(tokenizer, "<dt-may>"),
                _token_id(tokenizer, "<dt-choose-mode>"),
                _token_id(tokenizer, "<dt-choose-x>"),
            ],
            "stack_ref_ids": [
                _token_id(tokenizer, f"<stack-ref:{k}>") for k in range(MAX_STACK_REFS)
            ],
            "max_value_digit_max": DEFAULT_MAX_VALUE_DIGIT_MAX,
            "max_value_digits": _serialize_sequence_table(digit_sequences),
        },
    }
    artifact["artifact_fingerprint"] = _hash_jsonable(artifact)
    return artifact


def export_token_tables(
    output: Path | str,
    tokenizer: PreTrainedTokenizerFast,
    cache: CardTokenCache,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = token_tables_artifact(tokenizer, cache)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    return path


def load_card_cache_for_export(path: Path | str) -> CardTokenCache:
    from magic_ai.text_encoder.card_cache import load_card_cache

    cache = load_card_cache(path)
    return CardTokenCache(
        token_buffer=cache.token_buffer.detach().cpu().to(dtype=torch.int32).contiguous(),
        offsets=cache.offsets.detach().cpu().to(dtype=torch.int64).contiguous(),
        row_to_name=cache.row_to_name,
        engine_card_set_hash=cache.engine_card_set_hash,
        content_hash=cache.content_hash,
        cache_schema_version=cache.cache_schema_version,
    )


__all__ = [
    "TOKEN_TABLE_ARTIFACT_SCHEMA_VERSION",
    "card_cache_fingerprint",
    "custom_token_id_fingerprint",
    "export_token_tables",
    "load_card_cache_for_export",
    "token_tables_artifact",
    "tokenizer_fingerprint",
]
