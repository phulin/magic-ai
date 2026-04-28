from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import cast

import torch
from magic_ai.game_state import GameStateEncoder
from magic_ai.model import PPOPolicy
from magic_ai.opponent_pool import OpponentPool, SnapshotSchedule
from magic_ai.ppo import PPOStats, RolloutStep, gae_returns
from scripts.train import (
    RetrospectiveLogSchedule,
    TrainingResumeState,
    _current_transcript_snapshot,
    _restore_opponent_pool,
    build_slot_backend,
    checkpoint_encoder_kind,
    load_deck_dir,
    load_training_checkpoint,
    log_args_to_wandb_summary,
    log_ppo_stats,
    log_retrospective_table,
    retrospective_rating_rows,
    sample_decks,
    save_checkpoint,
    validate_args,
    validate_checkpoint_encoder,
    validate_deck_embeddings,
)


class TrainPPOTests(unittest.TestCase):
    def test_load_training_checkpoint_supports_legacy_path_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy.pt"
            torch.save({"args": {"output": Path("checkpoints/ppo.pt")}}, checkpoint_path)

            checkpoint = load_training_checkpoint(checkpoint_path)

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint["args"]["output"], Path("checkpoints/ppo.pt"))

    def test_legacy_checkpoint_defaults_to_slots_encoder(self) -> None:
        self.assertEqual(checkpoint_encoder_kind({"metadata": {}}), "slots")

    def test_validate_checkpoint_encoder_rejects_mismatch(self) -> None:
        args = Namespace(encoder="text")
        checkpoint = {"metadata": {"encoder": "slots"}}

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint encoder 'slots' is incompatible with --encoder text",
        ):
            validate_checkpoint_encoder(args, checkpoint)

    def test_decode_action_choice_color_falls_back_when_transcript_options_are_short(self) -> None:
        policy = PPOPolicy.__new__(PPOPolicy)

        trace, action = policy._decode_action(
            "choice_color",
            {"kind": "mana_color", "options": [{"color": "white"}]},
            [3],
        )

        self.assertEqual(trace.kind, "choice_color")
        self.assertEqual(trace.indices, (3,))
        self.assertEqual(action, {"selected_color": "red"})

    def test_current_transcript_snapshot_refreshes_live_game_before_snapshot(self) -> None:
        class StubGame:
            def __init__(self) -> None:
                self.state: dict[str, object] = {"step": "Precombat Main"}
                self.pending: dict[str, object] | None = {
                    "kind": "priority",
                    "player_idx": 0,
                    "options": [],
                }
                self.refresh_calls = 0

            def refresh_state(self) -> dict[str, object]:
                self.refresh_calls += 1
                self.state = {"step": "Declare Attackers"}
                self.pending = {"kind": "priority", "player_idx": 1, "options": []}
                return self.state

            def legal(self) -> dict[str, object]:
                return {"kind": "priority", "player_idx": 99, "options": []}

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

    def test_retrospective_log_schedule_fires_every_five_percent(self) -> None:
        schedule = RetrospectiveLogSchedule.build(episodes := 100)
        self.assertEqual(schedule.total_episodes, episodes)

        self.assertEqual(schedule.fire(4), [])
        self.assertEqual(schedule.fire(5), [(5, 5)])
        self.assertEqual(schedule.fire(16), [(10, 10), (15, 15)])

    def test_log_retrospective_table_includes_all_snapshot_ratings(self) -> None:
        pool = OpponentPool()
        first = pool.add_snapshot(Path("snapshot_g000005_p005.0.pt"), "g000005_p005.0")
        second = pool.add_snapshot(Path("snapshot_g000010_p010.0.pt"), "g000010_p010.0")
        pool.record_match(second, first, rated_won=True)

        rows = retrospective_rating_rows(pool, total_episodes=100)

        class StubRun:
            def __init__(self) -> None:
                self.payloads: list[dict[str, object]] = []

            def log(self, payload: dict[str, object]) -> None:
                self.payloads.append(payload)

        class StubTable:
            def __init__(self, *, columns: list[str], data: list[list[object]]) -> None:
                self.columns = columns
                self.data = data

        run = StubRun()

        log_retrospective_table(
            run,
            horizon_pct=10,
            horizon_step_count=10,
            ratings=rows,
            table_factory=StubTable,
        )

        self.assertEqual(len(run.payloads), 1)
        payload = run.payloads[0]
        self.assertEqual(payload["retrospective/horizon_pct"], 10)
        self.assertEqual(payload["retrospective/horizon_step_count"], 10)
        table = payload["retrospective/current_curve"]
        self.assertIsInstance(table, StubTable)
        table = cast(StubTable, table)
        self.assertEqual(
            table.columns,
            [
                "horizon_pct",
                "horizon_step_count",
                "snapshot_pct",
                "snapshot_step_count",
                "mu",
                "sigma",
                "conservative",
                "n_games",
            ],
        )
        self.assertEqual(len(table.data), 2)
        self.assertEqual(table.data[0][0], 10)
        self.assertEqual(table.data[0][2], 5.0)
        self.assertEqual(table.data[0][3], 5)
        self.assertEqual(table.data[0][7], 1)

    def test_build_slot_backend_constructs_current_slot_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_path = Path(tmpdir) / "embeddings.json"
            embeddings_path.write_text(
                json.dumps({"cards": [{"name": "Mountain", "embedding": [0.0, 0.1]}]})
            )
            args = Namespace(
                embeddings=embeddings_path,
                d_model=8,
                rollout_buffer_capacity=None,
                rollout_steps=16,
                max_steps_per_game=4,
                num_envs=2,
                hidden_dim=16,
                hidden_layers=1,
                max_options=4,
                max_targets_per_option=2,
                lstm=True,
                spr=False,
                spr_action_dim=16,
                spr_ema_decay=0.99,
                spr_k=3,
                spr_proj_dim=16,
                no_validate=False,
                torch_compile=False,
                batch_workers=1,
            )

            backend = build_slot_backend(args, torch.device("cpu"))

        self.assertIsInstance(backend.policy, PPOPolicy)
        self.assertEqual(backend.batch_workers, 1)
        self.assertIsNone(backend.batch_pool)
        self.assertEqual(backend.staging_buffer.max_steps_per_trajectory, 4)
        self.assertEqual(backend.policy.live_lstm_h.shape[1], 2)

    def test_save_checkpoint_serializes_resume_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "ppo.pt"
            opponent_pool_dir = Path(tmpdir) / "runs" / "run-123" / "opponent_pool"
            encoder = GameStateEncoder({"Mountain": [0.1, 0.2, 0.3]}, d_model=8)
            policy = PPOPolicy(
                encoder,
                hidden_dim=16,
                hidden_layers=1,
                max_options=4,
                max_targets_per_option=2,
                rollout_capacity=16,
            )
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            args = Namespace(
                output=checkpoint_path,
                opponent_pool_dir=opponent_pool_dir,
                encoder="slots",
            )
            pool = OpponentPool()
            snapshot_path = opponent_pool_dir / "snapshot_g000100_p010.0.pt"
            pool.add_snapshot(snapshot_path, "g000100_p010.0")

            save_checkpoint(
                checkpoint_path,
                policy,
                optimizer,
                args,
                opponent_pool=pool,
                snapshot_schedule=SnapshotSchedule(
                    total_episodes=10, thresholds=[1, 2], next_idx=1
                ),
                retrospective_schedule=RetrospectiveLogSchedule(
                    total_episodes=10,
                    thresholds=[1, 2],
                    horizon_pcts=[5, 10],
                    next_idx=1,
                ),
                resume_state=TrainingResumeState(
                    completed_games=7,
                    last_saved_games=6,
                    total_rollout_steps=11,
                    total_generated_rollout_steps=13,
                ),
                wandb_run_id="run-123",
            )

            checkpoint = load_training_checkpoint(checkpoint_path)

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint["args"]["output"], str(checkpoint_path))
        self.assertEqual(checkpoint["metadata"]["wandb_run_id"], "run-123")
        self.assertEqual(checkpoint["metadata"]["encoder"], "slots")
        self.assertEqual(
            checkpoint["metadata"]["run_artifact_dir"],
            str(opponent_pool_dir.parent),
        )
        self.assertEqual(checkpoint["training_state"]["completed_games"], 7)
        self.assertEqual(checkpoint["training_state"]["snapshot_schedule_next_idx"], 1)
        self.assertEqual(checkpoint["training_state"]["retrospective_schedule_next_idx"], 1)
        self.assertEqual(
            checkpoint["training_state"]["opponent_pool"]["entries"][0]["tag"],
            "g000100_p010.0",
        )

    def test_restore_opponent_pool_loads_checkpoint_state_and_new_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_dir = Path(tmpdir) / "opponent_pool"
            snapshot_dir.mkdir(parents=True)
            existing_snapshot = snapshot_dir / "snapshot_g000100_p010.0.pt"
            extra_snapshot = snapshot_dir / "snapshot_g000200_p020.0.pt"
            existing_snapshot.write_bytes(b"")
            extra_snapshot.write_bytes(b"")
            checkpoint = {
                "training_state": {
                    "opponent_pool": {
                        "entries": [
                            {
                                "path": str(existing_snapshot),
                                "tag": "g000100_p010.0",
                                "mu": 31.0,
                                "sigma": 5.0,
                                "n_games": 9,
                            }
                        ],
                    }
                }
            }

            pool = _restore_opponent_pool(checkpoint, snapshot_dir)

        self.assertEqual(len(pool.entries), 2)
        self.assertEqual(pool.entries[0].tag, "g000100_p010.0")
        self.assertEqual(pool.entries[1].tag, "g000200_p020.0")
        self.assertEqual(pool.entries[0].rating.mu, 31.0)
        self.assertEqual(pool.entries[0].n_games, 9)
        # New snapshot seeds from the previous entry's mean, but not confidence.
        self.assertAlmostEqual(pool.entries[1].rating.mu, 31.0)
        self.assertAlmostEqual(pool.entries[1].rating.sigma, 25.0 / 3.0)

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
            eval_games_per_snapshot=0,
            eval_recency_tau=4.0,
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
            eval_games_per_snapshot=0,
            eval_recency_tau=4.0,
            eval_num_envs=None,
        )

        with self.assertRaisesRegex(ValueError, "--gae-lambda must be in \\[0, 1\\]"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
