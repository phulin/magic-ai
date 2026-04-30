"""Smoke tests for ``magic_ai.text_encoder.policy.TextPolicy``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from magic_ai.game_state import (
    GameCardState,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
    PlayerState,
    TargetState,
)
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.policy import (
    TextPolicy,
    TextPolicyOutput,
    build_text_policy,
)
from magic_ai.text_encoder.render import (
    DEFAULT_ORACLE_PATH,
    OracleEntry,
    load_oracle_text,
)
from magic_ai.text_encoder.tokenizer import (
    TOKENIZER_DIR,
    card_ref_token,
    load_tokenizer,
)

ORACLE_PATH = Path(DEFAULT_ORACLE_PATH)


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR.exists():
        pytest.skip(
            f"tokenizer artifact missing at {TOKENIZER_DIR}; "
            "run scripts/build_text_encoder_vocab.py first"
        )
    return load_tokenizer()


@pytest.fixture(scope="module")
def oracle() -> dict[str, OracleEntry]:
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle JSON missing at {ORACLE_PATH}")
    return load_oracle_text(ORACLE_PATH)


@pytest.fixture(scope="module")
def real_card_names() -> list[str]:
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle JSON missing at {ORACLE_PATH}")
    payload = json.loads(ORACLE_PATH.read_text())
    names: list[str] = []
    for record in payload.get("cards", []):
        name = record.get("name")
        if name:
            names.append(cast(str, name))
        if len(names) >= 4:
            break
    if len(names) < 2:
        pytest.skip("not enough card names in oracle JSON")
    return names


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _card(cid: str, name: str, *, tapped: bool | None = None) -> GameCardState:
    out: dict[str, object] = {"ID": cid, "Name": name}
    if tapped is not None:
        out["Tapped"] = tapped
    return cast(GameCardState, out)


def _player(
    pid: str,
    name: str,
    *,
    life: int = 20,
    hand: list[GameCardState] | None = None,
    battlefield: list[GameCardState] | None = None,
    library_count: int = 53,
) -> PlayerState:
    out: dict[str, object] = {
        "ID": pid,
        "Name": name,
        "Life": life,
        "LibraryCount": library_count,
        "HandCount": len(hand or []),
        "GraveyardCount": 0,
        "Hand": hand or [],
        "Battlefield": battlefield or [],
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


def _snapshot_with_action(names: list[str]) -> GameStateSnapshot:
    a, b = names[0], names[1]
    self_bf = [_card("c1", a, tapped=False)]
    opp_bf = [_card("c2", b, tapped=True)]
    self_hand = [_card("c3", a)]
    snap: dict[str, object] = {
        "turn": 2,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", hand=self_hand, battlefield=self_bf),
            _player("p2", "Opp", battlefield=opp_bf, life=18),
        ],
        "pending": cast(
            PendingState,
            {
                "kind": "main",
                "player_idx": 0,
                "options": [
                    cast(
                        PendingOptionState,
                        {
                            "id": "opt0",
                            "kind": "cast",
                            "card_id": "c3",
                            "card_name": a,
                            "mana_cost": "{R}",
                            "valid_targets": [
                                cast(TargetState, {"id": "c2", "label": b}),
                                cast(TargetState, {"id": "c1", "label": a}),
                            ],
                        },
                    ),
                    cast(PendingOptionState, {"id": "p", "kind": "pass"}),
                ],
            },
        ),
    }
    return cast(GameStateSnapshot, snap)


def _snapshot_simple(names: list[str]) -> GameStateSnapshot:
    a = names[0]
    snap: dict[str, object] = {
        "turn": 1,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", hand=[_card("h0", a), _card("h1", a)]),
            _player("p2", "Opp"),
        ],
    }
    return cast(GameStateSnapshot, snap)


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def _small_cfg(tokenizer: Any) -> TextEncoderConfig:
    """Tiny encoder config for smoke tests: real vocab but small model."""
    return TextEncoderConfig(
        vocab_size=len(tokenizer),
        pad_id=int(tokenizer.pad_token_id),
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
    )


def test_text_policy_end_to_end(
    tokenizer, oracle: dict[str, OracleEntry], real_card_names: list[str]
) -> None:
    snapshots = [_snapshot_with_action(real_card_names), _snapshot_simple(real_card_names)]

    policy = build_text_policy(tokenizer, _small_cfg(tokenizer))
    cfg = policy.cfg

    batch = TextPolicy.encode_snapshots(
        snapshots,
        actions_per_snapshot=None,
        oracle=oracle,
        tokenizer=tokenizer,
    )

    # Sanity: round-trip card-ref index recovery on a real tokenizer.
    for b in range(batch.token_ids.shape[0]):
        for k in range(batch.card_ref_positions.shape[1]):
            pos = int(batch.card_ref_positions[b, k])
            if pos < 0:
                continue
            tok_str = tokenizer.convert_ids_to_tokens(int(batch.token_ids[b, pos]))
            assert tok_str == card_ref_token(k), (
                f"row {b} K={k} pos={pos} -> {tok_str!r}, expected {card_ref_token(k)!r}"
            )

    out: TextPolicyOutput = policy(batch)

    b_size, max_opts = batch.option_positions.shape
    max_targets = batch.target_positions.shape[2]
    d = cfg.d_model
    k = batch.card_ref_positions.shape[1]

    # Shapes
    assert out.policy_logits.shape == (b_size, max_opts)
    assert out.target_logits.shape == (b_size, max_opts, max_targets)
    assert out.values.shape == (b_size,)
    assert out.card_vectors.shape == (b_size, k, d)
    assert out.card_mask.shape == (b_size, k)
    assert out.option_vectors.shape == (b_size, max_opts, d)
    assert out.option_mask.shape == (b_size, max_opts)
    assert out.target_vectors.shape == (b_size, max_opts, max_targets, d)
    assert out.target_mask.shape == (b_size, max_opts, max_targets)
    assert out.state_vector.shape == (b_size, d)

    # Finiteness on valid slots; -inf on masked-out logits.
    assert torch.isfinite(out.values).all()
    assert torch.isfinite(out.state_vector).all()
    assert torch.isfinite(out.policy_logits[out.option_mask]).all()
    assert (out.policy_logits[~out.option_mask] == float("-inf")).all()
    assert torch.isfinite(out.target_logits[out.target_mask]).all()
    assert (out.target_logits[~out.target_mask] == float("-inf")).all()

    # At least one valid option in this batch.
    assert out.option_mask.any()


def test_text_policy_backward(
    tokenizer, oracle: dict[str, OracleEntry], real_card_names: list[str]
) -> None:
    snapshots = [_snapshot_with_action(real_card_names), _snapshot_simple(real_card_names)]
    policy = build_text_policy(tokenizer, _small_cfg(tokenizer))
    batch = TextPolicy.encode_snapshots(snapshots, None, oracle, tokenizer)

    out = policy(batch)

    loss = (
        out.values.sum()
        + out.policy_logits[out.option_mask].sum()
        + out.target_logits[out.target_mask].sum()
    )
    loss.backward()

    grad_norms = [
        float(p.grad.detach().abs().sum().item()) for p in policy.parameters() if p.grad is not None
    ]
    assert grad_norms, "no parameters received gradients"
    assert any(g > 0 for g in grad_norms), "all gradients zero"


def test_build_text_policy_validates_config(tokenizer) -> None:
    bad = TextEncoderConfig(vocab_size=len(tokenizer) + 1)
    with pytest.raises(ValueError):
        build_text_policy(tokenizer, bad)
    bad_pad = TextEncoderConfig(vocab_size=len(tokenizer), pad_id=int(tokenizer.pad_token_id) + 1)
    with pytest.raises(ValueError):
        build_text_policy(tokenizer, bad_pad)
