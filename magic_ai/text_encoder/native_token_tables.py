"""Wire serialization for shipping token tables into the Go-side ``mage`` lib.

Phase 3 of the assembler-port. Takes a :class:`TokenTables` and copies its
contents into flat ``torch.int32`` / ``torch.int64`` buffers laid out the
way the native side expects (entry K spans tokens[offsets[K]:offsets[K+1]]).
The buffers are then bound into a cffi ``MageTokenTables`` struct and passed
through ``MageRegisterTokenTables``.

The packed buffers are kept alive on a module-global registration so the
native side can read them as long as it needs to. This module is a single-
writer registry; calling :func:`register_native_token_tables` with a fresh
table replaces (and frees) the prior registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mage
import torch

from magic_ai.text_encoder.token_tables import (
    Frag,
    TokenTables,
)


def _pack(
    items: list[list[int]], dtype: torch.dtype = torch.int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate variable-length token lists into ``tokens`` + ``offsets``.

    ``offsets`` has length ``len(items) + 1`` so ``offsets[-1]`` is the total
    token-buffer length. ``offsets`` is always int32 (small) regardless of
    ``dtype`` for the tokens. Returns the offsets in int32 by default; the
    caller can request int64 for very large totals (per-card body table).
    """
    sizes = [len(x) for x in items]
    total = sum(sizes)
    tokens = torch.empty(total, dtype=dtype)
    cursor = 0
    for sz, item in zip(sizes, items, strict=True):
        if sz:
            tokens[cursor : cursor + sz] = torch.as_tensor(item, dtype=dtype)
            cursor += sz
    offs = [0]
    running = 0
    for sz in sizes:
        running += sz
        offs.append(running)
    return tokens, torch.as_tensor(offs, dtype=torch.int32)


def _pack_int64_offsets(items: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = [len(x) for x in items]
    total = sum(sizes)
    tokens = torch.empty(total, dtype=torch.int32)
    cursor = 0
    for sz, item in zip(sizes, items, strict=True):
        if sz:
            tokens[cursor : cursor + sz] = torch.as_tensor(item, dtype=torch.int32)
            cursor += sz
    offs = [0]
    running = 0
    for sz in sizes:
        running += sz
        offs.append(running)
    return tokens, torch.as_tensor(offs, dtype=torch.int64)


@dataclass
class _Packed:
    """Packed buffers + the cffi struct holding pointers into them.

    Held on a module-global so the underlying tensors outlive the native
    registration. Anything stored here is borrowed by Go.
    """

    structural_tokens: torch.Tensor
    structural_offsets: torch.Tensor
    turn_step_tokens: torch.Tensor
    turn_step_offsets: torch.Tensor
    life_owner_tokens: torch.Tensor
    life_owner_offsets: torch.Tensor
    ability_tokens: torch.Tensor
    ability_offsets: torch.Tensor
    count_tokens: torch.Tensor
    count_offsets: torch.Tensor
    zone_open_tokens: torch.Tensor
    zone_open_offsets: torch.Tensor
    zone_close_tokens: torch.Tensor
    zone_close_offsets: torch.Tensor
    action_verb_tokens: torch.Tensor
    action_verb_offsets: torch.Tensor
    mana_glyph_tokens: torch.Tensor
    mana_glyph_offsets: torch.Tensor
    card_ref_ids: torch.Tensor
    card_closer: torch.Tensor
    status_tapped: torch.Tensor
    status_untapped: torch.Tensor
    card_body_tokens: torch.Tensor
    card_body_offsets: torch.Tensor
    card_name_tokens: torch.Tensor
    card_name_offsets: torch.Tensor
    dict_entry_ids: torch.Tensor
    struct: Any = field(default=None)


_active_registration: _Packed | None = None


def _i32_ptr(ffi: Any, t: torch.Tensor) -> Any:
    if t.numel() == 0:
        return ffi.NULL
    if t.dtype != torch.int32:
        raise TypeError(f"expected int32 tensor, got {t.dtype}")
    return ffi.cast("int32_t *", t.data_ptr())


def _i64_ptr(ffi: Any, t: torch.Tensor) -> Any:
    if t.numel() == 0:
        return ffi.NULL
    if t.dtype != torch.int64:
        raise TypeError(f"expected int64 tensor, got {t.dtype}")
    return ffi.cast("int64_t *", t.data_ptr())


def register_native_token_tables(tables: TokenTables) -> None:
    """Pack ``tables`` into flat tensors and register them with the mage lib.

    Replaces any prior registration. The packed tensors are held on a
    module-global so they outlive the native side's borrowed pointers.
    """

    global _active_registration

    mage._ensure_loaded()
    ffi = mage._ffi
    lib = mage._lib

    # Order keys deterministically so the native side's index math matches.
    fragment_count = len(Frag)
    structural_items: list[list[int]] = [
        list(tables.structural[Frag(i)]) for i in range(fragment_count)
    ]
    structural_tokens, structural_offsets = _pack(structural_items)

    step_count = (
        max((step_id for (_, step_id) in tables.turn_step.keys()), default=-1) + 1
        if tables.turn_step
        else 0
    )
    turn_step_items: list[list[int]] = []
    for turn in range(tables.turn_min, tables.turn_max + 1):
        for step_id in range(step_count):
            turn_step_items.append(list(tables.turn_step[(turn, step_id)]))
    turn_step_tokens, turn_step_offsets = _pack(turn_step_items)

    owner_count = (
        max((owner for (_, owner) in tables.life_owner.keys()), default=-1) + 1
        if tables.life_owner
        else 0
    )
    life_owner_items: list[list[int]] = []
    for life in range(tables.life_min, tables.life_max + 1):
        for owner in range(owner_count):
            life_owner_items.append(list(tables.life_owner[(life, owner)]))
    life_owner_tokens, life_owner_offsets = _pack(life_owner_items)

    ability_items = [
        list(tables.ability[n]) for n in range(tables.ability_min, tables.ability_max + 1)
    ]
    ability_tokens, ability_offsets = _pack(ability_items)

    count_items = [list(tables.count[n]) for n in range(tables.count_min, tables.count_max + 1)]
    count_tokens, count_offsets = _pack(count_items)

    zone_count = (
        max((zone for (zone, _) in tables.zone_open.keys()), default=-1) + 1
        if tables.zone_open
        else 0
    )
    zone_open_items: list[list[int]] = []
    zone_close_items: list[list[int]] = []
    for zone in range(zone_count):
        for owner in range(owner_count):
            zone_open_items.append(list(tables.zone_open[(zone, owner)]))
            zone_close_items.append(list(tables.zone_close[(zone, owner)]))
    zone_open_tokens, zone_open_offsets = _pack(zone_open_items)
    zone_close_tokens, zone_close_offsets = _pack(zone_close_items)

    action_verb_count = (
        (max(tables.action_verb.keys(), default=-1) + 1) if tables.action_verb else 0
    )
    action_verb_items = [list(tables.action_verb[k]) for k in range(action_verb_count)]
    action_verb_tokens, action_verb_offsets = _pack(action_verb_items)

    mana_color_count = len(tables.mana_glyph)
    mana_glyph_tokens, mana_glyph_offsets = _pack([list(g) for g in tables.mana_glyph])

    card_ref_ids = torch.as_tensor(tables.card_ref, dtype=torch.int32)

    card_closer = torch.as_tensor(tables.card_closer, dtype=torch.int32)
    status_tapped = torch.as_tensor(tables.status_tapped, dtype=torch.int32)
    status_untapped = torch.as_tensor(tables.status_untapped, dtype=torch.int32)

    card_body_tokens, card_body_offsets = _pack_int64_offsets([list(b) for b in tables.card_body])
    card_name_tokens, card_name_offsets = _pack_int64_offsets([list(n) for n in tables.card_name])
    dict_entry_ids = torch.as_tensor(tables.dict_entry, dtype=torch.int32)

    packed = _Packed(
        structural_tokens=structural_tokens,
        structural_offsets=structural_offsets,
        turn_step_tokens=turn_step_tokens,
        turn_step_offsets=turn_step_offsets,
        life_owner_tokens=life_owner_tokens,
        life_owner_offsets=life_owner_offsets,
        ability_tokens=ability_tokens,
        ability_offsets=ability_offsets,
        count_tokens=count_tokens,
        count_offsets=count_offsets,
        zone_open_tokens=zone_open_tokens,
        zone_open_offsets=zone_open_offsets,
        zone_close_tokens=zone_close_tokens,
        zone_close_offsets=zone_close_offsets,
        action_verb_tokens=action_verb_tokens,
        action_verb_offsets=action_verb_offsets,
        mana_glyph_tokens=mana_glyph_tokens,
        mana_glyph_offsets=mana_glyph_offsets,
        card_ref_ids=card_ref_ids,
        card_closer=card_closer,
        status_tapped=status_tapped,
        status_untapped=status_untapped,
        card_body_tokens=card_body_tokens,
        card_body_offsets=card_body_offsets,
        card_name_tokens=card_name_tokens,
        card_name_offsets=card_name_offsets,
        dict_entry_ids=dict_entry_ids,
    )

    struct = ffi.new(
        "MageTokenTables *",
        {
            "fragment_count": fragment_count,
            "structural_tokens": _i32_ptr(ffi, structural_tokens),
            "structural_offsets": _i32_ptr(ffi, structural_offsets),
            "turn_min": tables.turn_min,
            "turn_max": tables.turn_max,
            "step_count": step_count,
            "turn_step_tokens": _i32_ptr(ffi, turn_step_tokens),
            "turn_step_offsets": _i32_ptr(ffi, turn_step_offsets),
            "life_min": tables.life_min,
            "life_max": tables.life_max,
            "owner_count": owner_count,
            "life_owner_tokens": _i32_ptr(ffi, life_owner_tokens),
            "life_owner_offsets": _i32_ptr(ffi, life_owner_offsets),
            "ability_min": tables.ability_min,
            "ability_max": tables.ability_max,
            "ability_tokens": _i32_ptr(ffi, ability_tokens),
            "ability_offsets": _i32_ptr(ffi, ability_offsets),
            "count_min": tables.count_min,
            "count_max": tables.count_max,
            "count_tokens": _i32_ptr(ffi, count_tokens),
            "count_offsets": _i32_ptr(ffi, count_offsets),
            "zone_count": zone_count,
            "zone_open_tokens": _i32_ptr(ffi, zone_open_tokens),
            "zone_open_offsets": _i32_ptr(ffi, zone_open_offsets),
            "zone_close_tokens": _i32_ptr(ffi, zone_close_tokens),
            "zone_close_offsets": _i32_ptr(ffi, zone_close_offsets),
            "action_verb_count": action_verb_count,
            "action_verb_tokens": _i32_ptr(ffi, action_verb_tokens),
            "action_verb_offsets": _i32_ptr(ffi, action_verb_offsets),
            "mana_color_count": mana_color_count,
            "mana_glyph_tokens": _i32_ptr(ffi, mana_glyph_tokens),
            "mana_glyph_offsets": _i32_ptr(ffi, mana_glyph_offsets),
            "card_ref_count": card_ref_ids.numel(),
            "card_ref_ids": _i32_ptr(ffi, card_ref_ids),
            "pad_id": tables.pad_id,
            "option_id": tables.option_id,
            "target_open_id": tables.target_open_id,
            "target_close_id": tables.target_close_id,
            "tapped_id": tables.tapped_id,
            "untapped_id": tables.untapped_id,
            "card_closer_len": card_closer.numel(),
            "card_closer": _i32_ptr(ffi, card_closer),
            "status_tapped_len": status_tapped.numel(),
            "status_tapped": _i32_ptr(ffi, status_tapped),
            "status_untapped_len": status_untapped.numel(),
            "status_untapped": _i32_ptr(ffi, status_untapped),
            "card_row_count": len(tables.card_body),
            "card_body_tokens": _i32_ptr(ffi, card_body_tokens),
            "card_body_offsets": _i64_ptr(ffi, card_body_offsets),
            "card_name_tokens": _i32_ptr(ffi, card_name_tokens),
            "card_name_offsets": _i64_ptr(ffi, card_name_offsets),
            "dict_open_id": tables.dict_open_id,
            "dict_close_id": tables.dict_close_id,
            "card_open_id": tables.card_open_id,
            "dict_entry_ids": _i32_ptr(ffi, dict_entry_ids),
            "self_id": tables.self_id,
            "opp_id": tables.opp_id,
            "stack_open_id": tables.stack_open_id,
            "stack_close_id": tables.stack_close_id,
            "command_open_id": tables.command_open_id,
            "command_close_id": tables.command_close_id,
        },
    )
    packed.struct = struct

    rc = lib.MageRegisterTokenTables(struct)
    if int(rc) != 0:
        raise RuntimeError(f"MageRegisterTokenTables failed with code {int(rc)}")

    # Replace prior registration only after the call succeeded so a failed
    # re-registration leaves the lib in a known-good state.
    _active_registration = packed


# Lookup-kind tags for MageTokenTableLookup (mirror the Go-side switch).
LOOKUP_FRAGMENT = 0
LOOKUP_TURN_STEP = 1
LOOKUP_LIFE_OWNER = 2
LOOKUP_ABILITY = 3
LOOKUP_COUNT = 4
LOOKUP_ZONE_OPEN = 5
LOOKUP_ZONE_CLOSE = 6
LOOKUP_ACTION_VERB = 7
LOOKUP_MANA_GLYPH = 8
LOOKUP_CARD_BODY = 9
LOOKUP_CARD_NAME = 10
LOOKUP_CARD_REF = 11


def native_lookup(kind: int, k0: int = 0, k1: int = 0) -> list[int]:
    """Round-trip a single (kind, k0, k1) lookup through the Go side."""
    mage._ensure_loaded()
    raw = mage._lib.MageTokenTableLookup(int(kind), int(k0), int(k1))
    return mage._take_raw(raw)


def native_summary() -> dict[str, Any]:
    """JSON-decoded snapshot of the currently registered native tables."""
    mage._ensure_loaded()
    raw = mage._lib.MageTokenTableSummary()
    return mage._take_raw(raw)


__all__ = [
    "LOOKUP_ABILITY",
    "LOOKUP_ACTION_VERB",
    "LOOKUP_CARD_BODY",
    "LOOKUP_CARD_NAME",
    "LOOKUP_CARD_REF",
    "LOOKUP_COUNT",
    "LOOKUP_FRAGMENT",
    "LOOKUP_LIFE_OWNER",
    "LOOKUP_MANA_GLYPH",
    "LOOKUP_TURN_STEP",
    "LOOKUP_ZONE_CLOSE",
    "LOOKUP_ZONE_OPEN",
    "native_lookup",
    "native_summary",
    "register_native_token_tables",
]
