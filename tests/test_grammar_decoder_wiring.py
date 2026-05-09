"""Smoke tests for wiring ``GrammarDecoder`` into ``TextPolicy``.

Step 3 of ``docs/decoder_grammar_plan.md``. Confirms that:

* ``TextPolicy(use_grammar_decoder=True)`` runs encoder + decoder in
  teacher-forced mode and gradients flow.
* ``TextPolicy(use_grammar_decoder=False)`` keeps producing inline-blank
  logits with the same shapes as before — i.e. the additive flag does
  not regress the inline-blank path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
    PlayerState,
)
from magic_ai.text_encoder.decoder import GrammarDecoderConfig
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, GrammarVocab
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.policy import TextPolicy
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    OracleEntry,
    load_oracle_text,
)
from magic_ai.text_encoder.tokenizer import TOKENIZER_DIR, load_tokenizer

ORACLE_PATH = Path(DEFAULT_ORACLE_PATH)


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR.exists():
        pytest.skip(f"tokenizer artifact missing at {TOKENIZER_DIR}")
    return load_tokenizer()


@pytest.fixture(scope="module")
def oracle() -> dict[str, OracleEntry]:
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle JSON missing at {ORACLE_PATH}")
    return load_oracle_text(ORACLE_PATH)


@pytest.fixture(scope="module")
def real_card_name() -> str:
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle JSON missing at {ORACLE_PATH}")
    payload = json.loads(ORACLE_PATH.read_text())
    for record in payload.get("cards", []):
        name = record.get("name")
        if name:
            return cast(str, name)
    pytest.skip("no card names in oracle JSON")


def _card(cid: str, name: str, *, tapped: bool | None = None) -> GameCardState:
    out: dict[str, object] = {"ID": cid, "Name": name}
    if tapped is not None:
        out["Tapped"] = tapped
    return cast(GameCardState, out)


def _player(pid: str, name: str, *, hand: list[GameCardState] | None = None) -> PlayerState:
    out: dict[str, object] = {
        "ID": pid,
        "Name": name,
        "Life": 20,
        "LibraryCount": 53,
        "HandCount": len(hand or []),
        "GraveyardCount": 0,
        "Hand": hand or [],
        "Battlefield": [],
        "Graveyard": [],
        "ManaPool": {
            "White": 0,
            "Blue": 0,
            "Black": 0,
            "Red": 0,
            "Green": 0,
            "Colorless": 0,
        },
    }
    return cast(PlayerState, out)


def _snapshot_priority(name: str) -> GameStateSnapshot:
    snap: dict[str, object] = {
        "turn": 1,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", hand=[_card("h0", name)]),
            _player("p2", "Opp"),
        ],
        "pending": cast(
            PendingState,
            {
                "kind": "priority",
                "player_idx": 0,
                "options": [
                    cast(PendingOptionState, {"id": "p", "kind": "pass"}),
                    cast(PendingOptionState, {"id": "a", "kind": "ability"}),
                ],
            },
        ),
    }
    return cast(GameStateSnapshot, snap)


def _snapshot_may(name: str) -> GameStateSnapshot:
    snap: dict[str, object] = {
        "turn": 2,
        "active_player": "p1",
        "step": "Upkeep",
        "players": [
            _player("p1", "Self", hand=[_card("h0", name)]),
            _player("p2", "Opp"),
        ],
        "pending": cast(
            PendingState,
            {
                "kind": "may",
                "player_idx": 0,
                "options": [
                    cast(PendingOptionState, {"id": "yes", "kind": "yes"}),
                    cast(PendingOptionState, {"id": "no", "kind": "no"}),
                ],
            },
        ),
    }
    return cast(GameStateSnapshot, snap)


def _small_cfg(tokenizer) -> TextEncoderConfig:
    return TextEncoderConfig(
        vocab_size=len(tokenizer),
        pad_id=int(tokenizer.pad_token_id),
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )


def _small_decoder_cfg() -> GrammarDecoderConfig:
    return GrammarDecoderConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)


def test_grammar_decoder_teacher_forced_smoke(
    tokenizer, oracle: dict[str, OracleEntry], real_card_name: str
) -> None:
    snapshots = [_snapshot_priority(real_card_name), _snapshot_may(real_card_name)]
    policy = TextPolicy(
        _small_cfg(tokenizer),
        decoder_cfg=_small_decoder_cfg(),
    )
    batch = TextPolicy.encode_snapshots(
        snapshots,
        oracle,
        tokenizer,
    )

    # Combined-stream invariants.
    b = batch.token_ids.shape[0]
    assert b == 2
    assert batch.spec_lens.shape == (b,)
    assert batch.decision_type.shape == (b,)
    assert int(batch.decision_type[0]) >= 0
    assert int(batch.decision_type[1]) >= 0
    # Spec tokens live at the tail of token_ids and the last `T_spec_i` rows
    # of token_ids should equal the spec_tokens slice (modulo dtype).
    for i in range(b):
        t_total = int(batch.seq_lengths[i])
        t_spec = int(batch.spec_lens[i])
        if t_spec == 0:
            continue
        tail = batch.token_ids[i, t_total - t_spec : t_total].to(torch.int32)
        assert torch.equal(tail, batch.spec_tokens[i, :t_spec])

    # Pointer anchor positions are within the live (combined) stream.
    n_anchors = batch.pointer_anchor_positions.shape[1]
    if n_anchors:
        for i in range(b):
            for a in range(n_anchors):
                pos = int(batch.pointer_anchor_positions[i, a])
                if pos < 0:
                    continue
                assert pos < int(batch.seq_lengths[i])

    # Teacher-forced decoder forward.
    target = torch.tensor(
        [
            [GrammarVocab.PRIORITY_OPEN, GrammarVocab.END],
            [GrammarVocab.MAY_OPEN, GrammarVocab.YES],
        ],
        dtype=torch.int64,
    )
    vocab_logits, pointer_logits = policy.forward_decoder_teacher_forced(batch, target)
    t_enc = batch.token_ids.shape[1]
    assert vocab_logits.shape == (b, target.shape[1], GRAMMAR_VOCAB_SIZE)
    assert pointer_logits.shape == (b, target.shape[1], t_enc)
    assert torch.isfinite(vocab_logits).all()
    assert torch.isfinite(pointer_logits).all()

    loss = vocab_logits.sum() + pointer_logits.sum()
    loss.backward()

    grads = [
        float(p.grad.detach().abs().sum().item()) for p in policy.parameters() if p.grad is not None
    ]
    assert grads
    assert any(g > 0 for g in grads), "no decoder gradient flowed"


# test_grammar_decoder_off_preserves_inline_path was removed in Phase 6 of the
# inline-blank cutover: the ``use_grammar_decoder`` flag is gone (the decoder is
# always wired) and the inline-blank head no longer exists.
