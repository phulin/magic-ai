from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import trueskill
from magic_ai.opponent_pool import (
    OpponentEntry,
    load_opponent_weights,
    opponent_policy_state_dict,
    save_snapshot,
)


class _PolicyWithRuntimeBuffers(nn.Module):
    def __init__(self, slots: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.register_buffer("live_lstm_h", torch.zeros(1, slots, 8))
        self.register_buffer("live_lstm_c", torch.zeros(1, slots, 8))


def test_opponent_state_dict_excludes_runtime_lstm_buffers() -> None:
    policy = _PolicyWithRuntimeBuffers(slots=64)

    state = opponent_policy_state_dict(policy)

    assert "linear.weight" in state
    assert "live_lstm_h" not in state
    assert "live_lstm_c" not in state


def test_save_snapshot_excludes_runtime_lstm_buffers(tmp_path: Path) -> None:
    policy = _PolicyWithRuntimeBuffers(slots=64)

    path = save_snapshot(policy, tmp_path, "g000001_p001.0")
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
