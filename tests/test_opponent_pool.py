from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import trueskill
from magic_ai.opponent_pool import (
    OpponentEntry,
    OpponentPool,
    _disable_text_replay_capture,
    load_opponent_weights,
    opponent_policy_state_dict,
    save_snapshot,
)
from magic_ai.text_encoder.lstm_stateful_text_policy import LSTMStatefulTextPolicy
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.recurrent import RecurrentTextPolicyConfig


class _PolicyWithRuntimeBuffers(nn.Module):
    def __init__(self, slots: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.register_buffer("live_lstm_h", torch.zeros(1, slots, 8))
        self.register_buffer("live_lstm_c", torch.zeros(1, slots, 8))


def _text_policy() -> LSTMStatefulTextPolicy:
    cfg = RecurrentTextPolicyConfig(
        encoder=TextEncoderConfig(
            vocab_size=16,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_seq_len=8,
            pad_id=0,
        ),
        lstm_hidden=8,
        lstm_layers=1,
    )
    return LSTMStatefulTextPolicy(cfg)


def test_add_snapshot_uses_default_trueskill_prior_for_each_snapshot() -> None:
    pool = OpponentPool()
    first = pool.add_snapshot(Path("snapshot_g000100_p010.0.pt"), "g000100_p010.0")
    first.rating = pool.env.create_rating(mu=31.0, sigma=5.0)

    second = pool.add_snapshot(Path("snapshot_g000200_p020.0.pt"), "g000200_p020.0")

    assert second.rating.mu == 25.0
    assert second.rating.sigma == 25.0 / 3.0


def test_opponent_state_dict_excludes_runtime_lstm_buffers() -> None:
    policy = _PolicyWithRuntimeBuffers(slots=64)

    state = opponent_policy_state_dict(policy)

    assert "linear.weight" in state
    assert "live_lstm_h" not in state
    assert "live_lstm_c" not in state


def test_disable_text_replay_capture_restores_buffer_after_eval_context() -> None:
    policy = _text_policy()
    rollout_buffer = object()
    policy.rollout_buffer = cast(Any, rollout_buffer)

    with _disable_text_replay_capture(policy, object()):
        assert policy.rollout_buffer is None

    assert policy.rollout_buffer is rollout_buffer


def test_disable_text_replay_capture_restores_buffer_after_exception() -> None:
    policy = _text_policy()
    rollout_buffer = object()
    policy.rollout_buffer = cast(Any, rollout_buffer)

    try:
        with _disable_text_replay_capture(policy):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert policy.rollout_buffer is rollout_buffer


def test_save_snapshot_excludes_runtime_lstm_buffers(tmp_path: Path) -> None:
    policy = _PolicyWithRuntimeBuffers(slots=64)

    path = save_snapshot(policy, tmp_path, "p001.0")
    payload = torch.load(path, map_location="cpu")

    assert "linear.weight" in payload["policy"]
    assert "live_lstm_h" not in payload["policy"]
    assert "live_lstm_c" not in payload["policy"]


def test_load_opponent_weights_ignores_legacy_runtime_lstm_buffers(tmp_path: Path) -> None:
    src = _PolicyWithRuntimeBuffers(slots=64)
    dst = _PolicyWithRuntimeBuffers(slots=100)
    with torch.no_grad():
        src.linear.weight.fill_(0.25)
        src.linear.bias.fill_(0.5)

    snapshot_path = tmp_path / "legacy.pt"
    torch.save({"policy": src.state_dict()}, snapshot_path)
    entry = OpponentEntry(
        path=snapshot_path,
        tag="legacy",
        rating=trueskill.TrueSkill().create_rating(),
    )

    load_opponent_weights(cast(Any, dst), entry, torch.device("cpu"))

    torch.testing.assert_close(dst.linear.weight, src.linear.weight)
    torch.testing.assert_close(dst.linear.bias, src.linear.bias)
    assert tuple(dst.get_buffer("live_lstm_h").shape) == (1, 100, 8)
    assert entry.cached_policy is not None
    assert "live_lstm_h" not in entry.cached_policy
    assert "live_lstm_c" not in entry.cached_policy
