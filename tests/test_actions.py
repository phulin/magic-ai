"""Tests for shared action decoding helpers."""

from __future__ import annotations

import torch
from magic_ai.actions import action_from_inline_block_choices
from magic_ai.game_state import PendingState


def test_action_from_inline_block_choices_decodes_none_and_attackers() -> None:
    pending: PendingState = {
        "kind": "blockers",
        "player_idx": 0,
        "options": [
            {
                "kind": "block",
                "permanent_id": "blocker-a",
                "valid_targets": [{"id": "attacker-a"}, {"id": "attacker-b"}],
            },
            {
                "kind": "block",
                "permanent_id": "blocker-b",
                "valid_targets": [{"id": "attacker-a"}],
            },
        ],
    }

    action = action_from_inline_block_choices(pending, [0, 1], [2, 0])

    assert action == {"blockers": [{"blocker": "blocker-a", "attacker": "attacker-b"}]}


def test_action_from_inline_block_choices_accepts_tensors_and_skips_invalid_options() -> None:
    pending: PendingState = {
        "kind": "blockers",
        "player_idx": 0,
        "options": [
            {
                "kind": "block",
                "permanent_id": "blocker-a",
                "valid_targets": [{"id": "attacker-a"}],
            }
        ],
    }

    action = action_from_inline_block_choices(
        pending,
        torch.tensor([99, 0]),
        torch.tensor([1, 1]),
    )

    assert action == {"blockers": [{"blocker": "blocker-a", "attacker": "attacker-a"}]}
