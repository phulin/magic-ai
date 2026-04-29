"""Native vs. Python render-plan parity dump.

Runs the same bolts-vs-bears mid-turn-6 state through both the Python
``emit_render_plan`` path and the native (Go) ``MageEncodeTokens`` path, and
prints both decoded streams + a diff line. Intended for byte-equal sanity
after touching either emitter.
"""

# ruff: noqa: E402, E501
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mage
from magic_ai.actions import build_priority_candidates
from magic_ai.game_state import GameStateSnapshot, PendingState
from magic_ai.native_encoder import NativeBatchEncoder
from magic_ai.text_encoder.assembler import assemble_batch
from magic_ai.text_encoder.card_cache import build_card_cache
from magic_ai.text_encoder.native_token_tables import register_native_token_tables
from magic_ai.text_encoder.render import DEFAULT_ORACLE_PATH, load_oracle_text
from magic_ai.text_encoder.render_plan import emit_render_plan
from magic_ai.text_encoder.token_tables import build_token_tables
from magic_ai.text_encoder.tokenizer import load_tokenizer


def main() -> None:
    deck_cfg = json.loads((REPO_ROOT / "decks" / "bears.json").read_text())
    deck_a = deck_cfg["player_a"]
    deck_b = deck_cfg["player_b"]

    tokenizer = load_tokenizer()
    oracle = load_oracle_text(Path(DEFAULT_ORACLE_PATH))
    names = sorted(oracle.keys())
    cache = build_card_cache(names, oracle, tokenizer, missing_policy="raise")
    name_to_row = {n: idx for idx, n in enumerate(cache.row_to_name)}

    def card_row_lookup(name: str) -> int:
        return name_to_row.get(name, 0)

    def tokenize(s: str) -> list[int]:
        return list(tokenizer.encode(s, add_special_tokens=False))

    rng = random.Random(13)
    game = mage.new_game(deck_a, deck_b, seed=42)
    target_turn = 6
    for _ in range(2000):
        game.refresh_state()
        if game.is_over:
            break
        if int(game.state.get("turn", 0) or 0) >= target_turn:
            p = game.pending or game.legal()
            if p is not None and (p.get("kind") or "") == "priority":
                break
        pending = cast(PendingState | None, game.pending or game.legal())
        if pending is None:
            try:
                game.step({"kind": "pass"})
            except Exception:
                break
            continue
        if (pending.get("kind") or "") == "priority":
            cands = build_priority_candidates(pending)
            if not cands:
                break
            action: dict[str, Any] = cast(dict, rng.choice(cands).to_action_request())
        else:
            opts = list(pending.get("options") or [])
            if not opts:
                break
            action = dict(rng.choice(opts))
        try:
            game.step(action)
        except Exception:
            break

    game.refresh_state()
    snapshot = cast(GameStateSnapshot, game.state)
    pending = cast(PendingState | None, game.pending or game.legal())
    legal = list((pending or {}).get("options") or [])

    # --- Python path ---
    py_plan = emit_render_plan(
        snapshot,
        legal,
        card_row_lookup=card_row_lookup,
        tokenize=tokenize,
        oracle=oracle,
        dedup_card_bodies=True,
    )
    py_batch = assemble_batch([py_plan], cache, tokenizer, max_tokens=4096)
    py_n = int(py_batch.seq_lengths[0])
    py_ids = py_batch.token_ids[0, :py_n].tolist()

    # --- Native path ---
    # Build + register the token tables so the Go-side assembler knows the
    # exact id sequences for every (frag, scalar, zone, ...) combination.
    tables = build_token_tables(tokenizer, cache=cache)
    register_native_token_tables(tables)

    from magic_ai.text_encoder.rollout import _load_mage_ffi

    lib, ffi = _load_mage_ffi()
    from magic_ai.text_encoder.rollout import (
        GAME_INFO_DIM,
        OPTION_SCALAR_DIM,
        TARGET_SCALAR_DIM,
        ZONE_SLOT_COUNT,
    )

    encoder = NativeBatchEncoder(
        max_options=64,
        max_targets_per_option=4,
        max_cached_choices=64,
        zone_slot_count=ZONE_SLOT_COUNT,
        game_info_dim=GAME_INFO_DIM,
        option_scalar_dim=OPTION_SCALAR_DIM,
        target_scalar_dim=TARGET_SCALAR_DIM,
        lib=lib,
        ffi=ffi,
        card_name_to_row={n: i for i, n in enumerate(cache.row_to_name)},
        emit_render_plan=True,
        render_plan_capacity=4096,
        dedup_card_bodies=True,
    )
    if not encoder.is_available:
        print("native encoder unavailable; aborting parity check")
        return
    perspective = int((pending or {}).get("player_idx", 0) or 0)
    native = encoder.encode_handles([game], perspective_player_indices=[perspective])
    assert native.render_plan is not None and native.render_plan_lengths is not None
    nat_plan = native.render_plan[0, : int(native.render_plan_lengths[0])].clone()
    nat_batch = assemble_batch([nat_plan], cache, tokenizer, max_tokens=4096)
    nat_n = int(nat_batch.seq_lengths[0])
    nat_ids = nat_batch.token_ids[0, :nat_n].tolist()

    print(f"=== Python emit path ===  seq_len={py_n}")
    print(tokenizer.decode(py_ids))
    print()
    print(f"=== Native (Go) emit path ===  seq_len={nat_n}")
    print(tokenizer.decode(nat_ids))
    print()
    if py_ids == nat_ids:
        print("PARITY: byte-equal token streams.")
    else:
        # Find first divergence.
        for i in range(min(len(py_ids), len(nat_ids))):
            if py_ids[i] != nat_ids[i]:
                print(
                    f"DIVERGES at index {i}: "
                    f"py={py_ids[i]!r} ({tokenizer.convert_ids_to_tokens(py_ids[i])!r}) "
                    f"vs nat={nat_ids[i]!r} ({tokenizer.convert_ids_to_tokens(nat_ids[i])!r})"
                )
                ctx = 6
                lo = max(0, i - ctx)
                hi_py = min(len(py_ids), i + ctx)
                hi_nat = min(len(nat_ids), i + ctx)
                print(
                    f"  py  [{lo}:{hi_py}] = {[tokenizer.convert_ids_to_tokens(t) for t in py_ids[lo:hi_py]]}"
                )
                print(
                    f"  nat [{lo}:{hi_nat}] = {[tokenizer.convert_ids_to_tokens(t) for t in nat_ids[lo:hi_nat]]}"
                )
                break
        else:
            print(f"DIVERGES in length only: py={len(py_ids)} nat={len(nat_ids)}")


if __name__ == "__main__":
    main()
