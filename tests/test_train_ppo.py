from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from magic_ai.ppo import PPOStats, RolloutStep, gae_returns
from scripts.train_ppo import (
    _current_transcript_snapshot,
    load_deck_dir,
    log_args_to_wandb_summary,
    log_ppo_stats,
    sample_decks,
    validate_args,
    validate_deck_embeddings,
)


class TrainPPOTests(unittest.TestCase):
    def test_current_transcript_snapshot_refreshes_state_before_legal_lookup(self) -> None:
        class StubGame:
            def __init__(self) -> None:
                self.state: dict[str, object] = {"step": "Precombat Main"}
                self.pending = None
                self.refresh_calls = 0

            def refresh_state(self) -> dict[str, object]:
                self.refresh_calls += 1
                self.state = {"step": "Declare Attackers"}
                return self.state

            def legal(self) -> dict[str, object]:
                return {"kind": "priority", "player_idx": 1, "options": []}

        game = StubGame()

        state, pending = _current_transcript_snapshot(game)

        self.assertEqual(game.refresh_calls, 1)
        self.assertEqual(state["step"], "Declare Attackers")
        self.assertEqual(pending["player_idx"], 1)

    def test_gae_returns_flips_bootstrap_sign_for_opponent_steps(self) -> None:
        steps = [
            RolloutStep(perspective_player_idx=0, old_log_prob=0.0, value=0.2),
            RolloutStep(perspective_player_idx=1, old_log_prob=0.0, value=-0.1),
            RolloutStep(perspective_player_idx=1, old_log_prob=0.0, value=0.3),
        ]

        returns = gae_returns(
            steps,
            winner_idx=1,
            gamma=1.0,
            gae_lambda=1.0,
        )

        expected = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(returns, expected))

    def test_validate_deck_embeddings_reports_missing_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_path = Path(tmpdir) / "embeddings.json"
            embeddings_path.write_text(
                json.dumps(
                    {
                        "cards": [
                            {"name": "Mountain", "embedding": [0.0]},
                            {"name": "Lightning Bolt", "embedding": [1.0]},
                        ]
                    }
                )
            )

            deck_a = {"cards": [{"name": "Mountain", "count": 2}]}
            deck_b = {"cards": [{"name": "Missing Card", "count": 3}]}

            with self.assertRaisesRegex(ValueError, "Missing Card .*player_b=3"):
                validate_deck_embeddings(embeddings_path, (deck_a, deck_b))

    def test_load_deck_dir_loads_sorted_json_decks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deck_dir = Path(tmpdir)
            (deck_dir / "b.json").write_text(
                json.dumps({"name": "B", "cards": [{"name": "Island", "count": 1}]})
            )
            (deck_dir / "a.json").write_text(
                json.dumps({"name": "A", "cards": [{"name": "Forest", "count": 1}]})
            )

            decks = load_deck_dir(deck_dir)

        self.assertEqual([deck["name"] for deck in decks], ["A", "B"])

    def test_sample_decks_is_deterministic_for_seed(self) -> None:
        deck_pool = [
            {"name": "A", "cards": []},
            {"name": "B", "cards": []},
            {"name": "C", "cards": []},
        ]

        first = sample_decks(deck_pool, seed=17)
        second = sample_decks(deck_pool, seed=17)

        self.assertEqual(first, second)

    def test_log_args_to_wandb_summary_serializes_namespace(self) -> None:
        args = Namespace(
            checkpoint_dir=Path("checkpoints/latest"),
            no_wandb=False,
            wandb_run_name=None,
            layers=[128, 256],
            metadata={"seed": 7},
        )

        class StubRun:
            def __init__(self) -> None:
                self.summary: dict[str, object] = {}

        run = StubRun()

        log_args_to_wandb_summary(args, run=run)

        self.assertEqual(
            run.summary,
            {
                "args/checkpoint_dir": "checkpoints/latest",
                "args/no_wandb": False,
                "args/wandb_run_name": None,
                "args/layers": [128, 256],
                "args/metadata": {"seed": 7},
            },
        )

    def test_log_ppo_stats_includes_total_rollout_steps(self) -> None:
        stats = PPOStats(
            loss=1.0,
            policy_loss=2.0,
            value_loss=3.0,
            entropy=4.0,
            approx_kl=5.0,
            clip_fraction=6.0,
            spr_loss=7.0,
        )
        payloads: list[dict[str, object]] = []

        log_ppo_stats(
            stats,
            games=8,
            steps=13,
            total_rollout_steps=21,
            total_generated_rollout_steps=34,
            value_metrics={"return_mean": 0.5},
            log_fn=payloads.append,
            run_active=True,
        )

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["rollout_steps"], 13)
        self.assertEqual(payloads[0]["total_rollout_steps"], 21)
        self.assertEqual(payloads[0]["total_generated_rollout_steps"], 34)
        self.assertEqual(payloads[0]["games"], 8)
        self.assertEqual(payloads[0]["return_mean"], 0.5)

    def test_validate_args_requires_no_validate_for_torch_compile(self) -> None:
        args = Namespace(
            episodes=1,
            num_envs=1,
            rollout_steps=1,
            max_steps_per_game=1,
            minibatch_size=1,
            hidden_layers=1,
            gae_lambda=0.95,
            torch_compile=True,
            no_validate=False,
            deck_json=None,
            deck_dir=None,
            eval_rounds_per_snapshot=0,
            eval_games_per_round=0,
            eval_num_envs=None,
        )

        with self.assertRaisesRegex(ValueError, "--torch-compile requires --no-validate"):
            validate_args(args)

    def test_validate_args_rejects_invalid_gae_lambda(self) -> None:
        args = Namespace(
            episodes=1,
            num_envs=1,
            rollout_steps=1,
            max_steps_per_game=1,
            minibatch_size=1,
            hidden_layers=1,
            gae_lambda=1.5,
            torch_compile=False,
            no_validate=False,
            deck_json=None,
            deck_dir=None,
            eval_rounds_per_snapshot=0,
            eval_games_per_round=0,
            eval_num_envs=None,
        )

        with self.assertRaisesRegex(ValueError, "--gae-lambda must be in \\[0, 1\\]"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
