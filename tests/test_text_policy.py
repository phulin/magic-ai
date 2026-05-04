"""Smoke tests for ``magic_ai.text_encoder.policy.TextPolicy``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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
from magic_ai.text_encoder import model as text_model
from magic_ai.text_encoder.batch import TextEncodedBatch
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


def _snapshot_with_blockers(names: list[str]) -> GameStateSnapshot:
    blocker_name, attacker_name = names[0], names[1]
    blocker = _card("blocker-1", blocker_name, tapped=False)
    attacker = _card("attacker-1", attacker_name, tapped=True)
    snap: dict[str, object] = {
        "turn": 4,
        "active_player": "p2",
        "step": "Declare Blockers",
        "players": [
            _player("p1", "Self", battlefield=[blocker]),
            _player("p2", "Opp", battlefield=[attacker]),
        ],
        "pending": cast(
            PendingState,
            {
                "kind": "blockers",
                "player_idx": 0,
                "options": [
                    cast(
                        PendingOptionState,
                        {
                            "id": "block-opt",
                            "kind": "block",
                            "permanent_id": blocker["ID"],
                            "valid_targets": [
                                cast(TargetState, {"id": attacker["ID"], "label": attacker_name})
                            ],
                        },
                    )
                ],
            },
        ),
    }
    return cast(GameStateSnapshot, snap)


def _snapshot_with_may(names: list[str]) -> GameStateSnapshot:
    a = names[0]
    snap: dict[str, object] = {
        "turn": 3,
        "active_player": "p1",
        "step": "Upkeep",
        "players": [
            _player("p1", "Self", battlefield=[_card("c1", a, tapped=False)]),
            _player("p2", "Opp"),
        ],
        "pending": cast(PendingState, {"kind": "may", "player_idx": 0, "options": []}),
    }
    return cast(GameStateSnapshot, snap)


def _snapshot_with_mode(names: list[str]) -> GameStateSnapshot:
    a = names[0]
    snap: dict[str, object] = {
        "turn": 3,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", battlefield=[_card("c1", a, tapped=False)]),
            _player("p2", "Opp"),
        ],
        "pending": cast(
            PendingState,
            {
                "kind": "mode",
                "player_idx": 0,
                "options": [
                    cast(PendingOptionState, {"id": "mode-0", "kind": "choice"}),
                    cast(PendingOptionState, {"id": "mode-1", "kind": "choice"}),
                ],
            },
        ),
    }
    return cast(GameStateSnapshot, snap)


def _snapshot_with_number(names: list[str]) -> GameStateSnapshot:
    a = names[0]
    snap: dict[str, object] = {
        "turn": 3,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", battlefield=[_card("c1", a, tapped=False)]),
            _player("p2", "Opp"),
        ],
        "pending": cast(
            PendingState,
            {
                "kind": "number",
                "player_idx": 0,
                "options": [
                    cast(PendingOptionState, {"id": "x-0", "kind": "choice"}),
                    cast(PendingOptionState, {"id": "x-1", "kind": "choice"}),
                    cast(PendingOptionState, {"id": "x-2", "kind": "choice"}),
                ],
            },
        ),
    }
    return cast(GameStateSnapshot, snap)


def _snapshot_with_mana_color(names: list[str]) -> GameStateSnapshot:
    a = names[0]
    snap: dict[str, object] = {
        "turn": 3,
        "active_player": "p1",
        "step": "Precombat Main",
        "players": [
            _player("p1", "Self", battlefield=[_card("c1", a, tapped=False)]),
            _player("p2", "Opp"),
        ],
        "pending": cast(
            PendingState,
            {
                "kind": "mana_color",
                "player_idx": 0,
                "options": [
                    cast(PendingOptionState, {"id": "white", "kind": "choice"}),
                    cast(PendingOptionState, {"id": "blue", "kind": "choice"}),
                    cast(PendingOptionState, {"id": "black", "kind": "choice"}),
                    cast(PendingOptionState, {"id": "red", "kind": "choice"}),
                    cast(PendingOptionState, {"id": "green", "kind": "choice"}),
                    cast(PendingOptionState, {"id": "colorless", "kind": "choice"}),
                ],
            },
        ),
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


def test_text_policy_inline_blank_forward(
    tokenizer, oracle: dict[str, OracleEntry], real_card_names: list[str]
) -> None:
    cfg = _small_cfg(tokenizer)
    cfg.use_inline_blanks = True
    policy = build_text_policy(tokenizer, cfg)
    batch = TextPolicy.encode_snapshots(
        [_snapshot_with_action(real_card_names)],
        actions_per_snapshot=None,
        oracle=oracle,
        tokenizer=tokenizer,
        use_inline_blanks=True,
    )

    assert batch.blank_positions.shape[1] > 0
    assert batch.blank_legal_mask.any()

    out = policy(batch)

    assert out.blank_logits is not None
    assert out.blank_logits.shape == batch.blank_legal_ids.shape
    assert torch.isfinite(out.blank_logits[batch.blank_legal_mask]).all()
    assert (out.blank_logits[~batch.blank_legal_mask] == float("-inf")).all()
    assert out.blank_group is batch.blank_group
    assert out.blank_group_kind is batch.blank_group_kind


def test_text_policy_inline_block_blank_forward(
    tokenizer, oracle: dict[str, OracleEntry], real_card_names: list[str]
) -> None:
    cfg = _small_cfg(tokenizer)
    cfg.use_inline_blanks = True
    policy = build_text_policy(tokenizer, cfg)
    batch = TextPolicy.encode_snapshots(
        [_snapshot_with_blockers(real_card_names)],
        actions_per_snapshot=None,
        oracle=oracle,
        tokenizer=tokenizer,
        use_inline_blanks=True,
    )

    assert batch.blank_positions.shape == (1, 1)
    assert batch.blank_legal_ids.shape == (1, 1, 2)
    assert int(batch.blank_option_index[0, 0]) == 0
    none_id = tokenizer.convert_tokens_to_ids("<none>")
    assert int(batch.blank_legal_ids[0, 0, 0]) == int(none_id)

    out = policy(batch)

    assert out.blank_logits is not None
    assert out.blank_logits.shape == batch.blank_legal_ids.shape
    assert out.blank_option_index is batch.blank_option_index
    assert torch.isfinite(out.blank_logits[batch.blank_legal_mask]).all()


def test_text_policy_inline_may_blank_forward(
    tokenizer, oracle: dict[str, OracleEntry], real_card_names: list[str]
) -> None:
    cfg = _small_cfg(tokenizer)
    cfg.use_inline_blanks = True
    policy = build_text_policy(tokenizer, cfg)
    batch = TextPolicy.encode_snapshots(
        [_snapshot_with_may(real_card_names)],
        actions_per_snapshot=None,
        oracle=oracle,
        tokenizer=tokenizer,
        use_inline_blanks=True,
    )

    assert batch.blank_positions.shape == (1, 1)
    no_id = tokenizer.convert_tokens_to_ids("<no>")
    yes_id = tokenizer.convert_tokens_to_ids("<yes>")
    assert int(batch.blank_legal_ids[0, 0, 0]) == int(no_id)
    assert int(batch.blank_legal_ids[0, 0, 1]) == int(yes_id)

    out = policy(batch)

    assert out.blank_logits is not None
    assert out.blank_logits.shape == batch.blank_legal_ids.shape
    assert torch.isfinite(out.blank_logits[batch.blank_legal_mask]).all()


def test_text_policy_inline_mode_blank_forward(
    tokenizer, oracle: dict[str, OracleEntry], real_card_names: list[str]
) -> None:
    cfg = _small_cfg(tokenizer)
    cfg.use_inline_blanks = True
    policy = build_text_policy(tokenizer, cfg)
    batch = TextPolicy.encode_snapshots(
        [_snapshot_with_mode(real_card_names)],
        actions_per_snapshot=None,
        oracle=oracle,
        tokenizer=tokenizer,
        use_inline_blanks=True,
    )

    assert batch.blank_positions.shape == (1, 1)
    num0_id = tokenizer.convert_tokens_to_ids("<num:0>")
    num1_id = tokenizer.convert_tokens_to_ids("<num:1>")
    assert int(batch.blank_legal_ids[0, 0, 0]) == int(num0_id)
    assert int(batch.blank_legal_ids[0, 0, 1]) == int(num1_id)

    out = policy(batch)

    assert out.blank_logits is not None
    assert out.blank_logits.shape == batch.blank_legal_ids.shape
    assert torch.isfinite(out.blank_logits[batch.blank_legal_mask]).all()


def test_text_policy_inline_number_blank_forward(
    tokenizer, oracle: dict[str, OracleEntry], real_card_names: list[str]
) -> None:
    cfg = _small_cfg(tokenizer)
    cfg.use_inline_blanks = True
    policy = build_text_policy(tokenizer, cfg)
    batch = TextPolicy.encode_snapshots(
        [_snapshot_with_number(real_card_names)],
        actions_per_snapshot=None,
        oracle=oracle,
        tokenizer=tokenizer,
        use_inline_blanks=True,
    )

    assert batch.blank_positions.shape == (1, 1)
    for k in range(3):
        num_id = tokenizer.convert_tokens_to_ids(f"<num:{k}>")
        assert int(batch.blank_legal_ids[0, 0, k]) == int(num_id)

    out = policy(batch)

    assert out.blank_logits is not None
    assert out.blank_logits.shape == batch.blank_legal_ids.shape
    assert torch.isfinite(out.blank_logits[batch.blank_legal_mask]).all()


def test_text_policy_inline_mana_color_blank_forward(
    tokenizer, oracle: dict[str, OracleEntry], real_card_names: list[str]
) -> None:
    cfg = _small_cfg(tokenizer)
    cfg.use_inline_blanks = True
    policy = build_text_policy(tokenizer, cfg)
    batch = TextPolicy.encode_snapshots(
        [_snapshot_with_mana_color(real_card_names)],
        actions_per_snapshot=None,
        oracle=oracle,
        tokenizer=tokenizer,
        use_inline_blanks=True,
    )

    assert batch.blank_positions.shape == (1, 1)
    for k, symbol in enumerate(("W", "U", "B", "R", "G", "C")):
        mana_id = tokenizer.convert_tokens_to_ids(f"<mana:{symbol}>")
        assert int(batch.blank_legal_ids[0, 0, k]) == int(mana_id)

    out = policy(batch)

    assert out.blank_logits is not None
    assert out.blank_logits.shape == batch.blank_legal_ids.shape
    assert torch.isfinite(out.blank_logits[batch.blank_legal_mask]).all()


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


def test_hf_text_policy_copies_checkpoint_into_local_encoder(monkeypatch) -> None:
    class DummyConfig:
        hidden_size = 16
        num_attention_heads = 4
        num_hidden_layers = 3
        intermediate_size = 32
        max_position_embeddings = 128

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> DummyConfig:
            return DummyConfig()

    class DummyHfModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16, num_hidden_layers=3, pad_token_id=None)
            self.resized_to: int | None = None
            self._state = self._make_state(vocab_size=8)

        @staticmethod
        def _make_state(vocab_size: int) -> dict[str, torch.Tensor]:
            state: dict[str, torch.Tensor] = {
                "embeddings.tok_embeddings.weight": torch.arange(
                    vocab_size * 16, dtype=torch.float32
                ).reshape(vocab_size, 16),
                "embeddings.norm.weight": torch.full((16,), 0.25),
                "final_norm.weight": torch.full((16,), 0.75),
            }
            for layer in range(3):
                prefix = f"layers.{layer}"
                if layer > 0:
                    state[f"{prefix}.attn_norm.weight"] = torch.full((16,), 0.1 + layer)
                state[f"{prefix}.attn.Wqkv.weight"] = torch.full((48, 16), 1.0 + layer)
                state[f"{prefix}.attn.Wo.weight"] = torch.full((16, 16), 2.0 + layer)
                state[f"{prefix}.mlp_norm.weight"] = torch.full((16,), 3.0 + layer)
                state[f"{prefix}.mlp.Wi.weight"] = torch.full((64, 16), 4.0 + layer)
                state[f"{prefix}.mlp.Wo.weight"] = torch.full((16, 32), 5.0 + layer)
            return state

        def resize_token_embeddings(
            self, vocab_size: int, pad_to_multiple_of: int | None = None
        ) -> None:
            del pad_to_multiple_of
            self.resized_to = vocab_size
            self._state = self._make_state(vocab_size=vocab_size)

        def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
            del args, kwargs
            return self._state

    class DummyHfWrapper:
        def __init__(self) -> None:
            self.model = DummyHfModel()

        def resize_token_embeddings(
            self, vocab_size: int, pad_to_multiple_of: int | None = None
        ) -> None:
            self.model.resize_token_embeddings(vocab_size, pad_to_multiple_of=pad_to_multiple_of)

    class DummyAutoModel:
        last_model: DummyHfModel | None = None

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> DummyHfWrapper:
            wrapper = DummyHfWrapper()
            cls.last_model = wrapper.model
            return wrapper

    monkeypatch.setattr(text_model, "AutoConfig", DummyAutoConfig)
    monkeypatch.setattr(text_model, "AutoModelForMaskedLM", DummyAutoModel)

    cfg = text_model.text_encoder_config_from_hf(
        model_name="dummy/checkpoint",
        vocab_size=20,
        pad_id=0,
        truncate_layers=2,
    )
    policy = TextPolicy(cfg)

    assert cfg.d_model == 16
    assert cfg.n_layers == 2
    assert cfg.n_heads == 4
    assert cfg.d_ff == 32
    assert DummyAutoModel.last_model is not None
    assert DummyAutoModel.last_model.resized_to == 20
    assert isinstance(policy.encoder, text_model.TextStateEncoder)
    assert torch.equal(
        policy.encoder.tok_emb.weight.detach().cpu(),
        DummyAutoModel.last_model.state_dict()["embeddings.tok_embeddings.weight"],
    )
    assert torch.equal(
        cast(text_model.EncoderBlock, policy.encoder.blocks[1]).attn.qkv.weight.detach().cpu(),
        DummyAutoModel.last_model.state_dict()["layers.1.attn.Wqkv.weight"],
    )

    batch = TextEncodedBatch(
        token_ids=torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64),
        card_ref_positions=torch.full((2, 256), -1, dtype=torch.int64),
        option_positions=torch.tensor([[0], [0]], dtype=torch.int64),
        option_mask=torch.ones((2, 1), dtype=torch.bool),
        target_positions=torch.full((2, 1, 0), -1, dtype=torch.int64),
        target_mask=torch.zeros((2, 1, 0), dtype=torch.bool),
        seq_lengths=torch.tensor([2, 1], dtype=torch.int64),
    )
    out = policy(batch)
    assert out.state_vector.shape == (2, 16)
    assert out.option_vectors.shape == (2, 1, 16)
