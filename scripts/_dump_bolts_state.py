"""One-off: dump a single assembled token stream from a real bolts.dec game."""

# ruff: noqa: E402 — sys.path tweak before imports is intentional for direct invocation.
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
from magic_ai.text_encoder.assembler import assemble_batch
from magic_ai.text_encoder.card_cache import build_card_cache
from magic_ai.text_encoder.render import DEFAULT_ORACLE_PATH, load_oracle_text
from magic_ai.text_encoder.render_plan import emit_render_plan
from magic_ai.text_encoder.tokenizer import load_tokenizer


def main() -> None:
    deck_path = REPO_ROOT / "decks" / "bears.json"
    deck_cfg = json.loads(deck_path.read_text())
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

    # Step well into the game with random play so attackers / blockers /
    # tapped permanents / mana pool / multi-turn state actually appear.
    target_turn = 6
    for _ in range(2000):
        game.refresh_state()
        if game.is_over:
            break
        if int(game.state.get("turn", 0) or 0) >= target_turn:
            # Stop on the first priority window of the target turn so we
            # land on an interesting decision rather than mid-cleanup.
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
        # Use build_priority_candidates so multi-target spells get one
        # candidate per (option, target) and to_action_request emits the
        # exact shape the engine expects.
        if (pending.get("kind") or "") == "priority":
            candidates = build_priority_candidates(pending)
            if not candidates:
                break
            cand = rng.choice(candidates)
            action: dict[str, Any] = cast(dict, cand.to_action_request())
        else:
            options = list(pending.get("options") or [])
            if not options:
                break
            action = dict(rng.choice(options))
        try:
            game.step(action)
        except Exception:
            break

    game.refresh_state()
    snapshot = cast(GameStateSnapshot, game.state)
    pending = cast(PendingState | None, game.pending or game.legal())
    legal = list((pending or {}).get("options") or [])

    plan = emit_render_plan(
        snapshot,
        legal,
        card_row_lookup=card_row_lookup,
        tokenize=tokenize,
        oracle=oracle,
        dedup_card_bodies=True,
    )
    batch = assemble_batch([plan], cache, tokenizer, max_tokens=4096)
    n = int(batch.seq_lengths[0])
    ids = batch.token_ids[0, :n].tolist()

    print(
        f"turn={snapshot.get('turn')} step={snapshot.get('step')} "
        f"pending_kind={(pending or {}).get('kind')!r} "
        f"n_options={len(legal)}"
    )
    print(f"seq_len = {n}")
    print()
    print("=== full decoded text ===")
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
