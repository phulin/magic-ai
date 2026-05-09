from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import scripts.train as train_mod
import torch
from magic_ai.actions import ActionTrace, PolicyStep
from magic_ai.game_state import GameStateSnapshot, PendingState
from magic_ai.opponent_pool import OpponentPool, SnapshotSchedule
from magic_ai.returns import gae_returns
from magic_ai.rnad import RNaDStats
from magic_ai.rollout import PPOStats, RolloutStep
from magic_ai.slot_encoder.game_state import GameStateEncoder
from magic_ai.slot_encoder.model import PPOPolicy
from scripts.train import (
    RetrospectiveLogSchedule,
    SlotTrainingBackend,
    TrainingResumeState,
    _build_opponent_schedules,
    _checkpoint_has_policy,
    _current_transcript_snapshot,
    _prune_pool_to_schedule,
    _restore_opponent_pool,
    _should_run_mlm_pretrain,
    append_priority_trace_jsonl,
    append_sample_game_log,
    build_slot_backend,
    checkpoint_encoder_kind,
    initialize_game_log,
    load_deck_dir,
    load_training_checkpoint,
    log_args_to_wandb_summary,
    log_ppo_stats,
    log_retrospective_table,
    main,
    priority_trace_jsonl_path,
    retrospective_rating_rows,
    rnad_value_metrics,
    sample_decks,
    save_checkpoint,
    token_length_percentile_metrics,
    train_selected_backend,
    train_text_envs,
    validate_args,
    validate_checkpoint_encoder,
    validate_deck_embeddings,
)

# Inline-blank pipeline constant retained as a no-op shim; Phase 6 deleted the producer.
BLANK_GROUP_CROSS_BLANK = 0


class TrainPPOTests(unittest.TestCase):
    def test_load_training_checkpoint_supports_legacy_path_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy.pt"
            torch.save({"args": {"output": Path("checkpoints/ppo.pt")}}, checkpoint_path)

            checkpoint = load_training_checkpoint(checkpoint_path)

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint["args"]["output"], Path("checkpoints/ppo.pt"))

    def test_load_training_checkpoint_rejects_missing_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "missing.pt"

            with self.assertRaisesRegex(FileNotFoundError, "--checkpoint path does not exist"):
                load_training_checkpoint(checkpoint_path)

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

    def test_validate_checkpoint_encoder_rejects_policy_snapshot(self) -> None:
        args = Namespace(encoder="text")
        checkpoint = {"state_dict": {}}

        with self.assertRaisesRegex(
            ValueError,
            "policy snapshot, not a training checkpoint",
        ):
            validate_checkpoint_encoder(args, checkpoint)

    def test_checkpoint_has_policy_detects_training_checkpoints(self) -> None:
        self.assertFalse(_checkpoint_has_policy(None))
        self.assertFalse(_checkpoint_has_policy({"state_dict": {}}))
        self.assertTrue(_checkpoint_has_policy({"policy": {}}))

    def test_pretrain_gates_skip_full_policy_checkpoint(self) -> None:
        args = Namespace(pretrain_mlm_dir=Path("choices"))
        checkpoint = {"policy": {}, "metadata": {"encoder": "text"}}

        self.assertFalse(_should_run_mlm_pretrain(args, checkpoint))

    def test_pretrain_gates_run_without_policy_checkpoint(self) -> None:
        args = Namespace(pretrain_mlm_dir=Path("choices"))

        self.assertTrue(_should_run_mlm_pretrain(args, None))

    def test_pretrain_gates_skip_post_policy_value_checkpoint(self) -> None:
        args = Namespace(pretrain_mlm_dir=Path("choices"))
        checkpoint = {"policy": {}, "metadata": {"encoder": "text", "post_mlm": True}}

        self.assertFalse(_should_run_mlm_pretrain(args, checkpoint))

    def test_validate_checkpoint_encoder_rejects_text_config_mismatch(self) -> None:
        args = Namespace(
            encoder="text",
            text_max_tokens=128,
            text_d_model=32,
            text_layers=1,
            text_heads=4,
            text_d_ff=64,
            hidden_layers=1,
            max_options=4,
            max_targets_per_option=2,
        )
        checkpoint = {
            "metadata": {
                "encoder": "text",
                "text_config": {
                    "text_max_tokens": 128,
                    "text_d_model": 16,
                    "text_layers": 1,
                    "text_heads": 4,
                    "text_d_ff": 64,
                    "hidden_layers": 1,
                    "max_options": 4,
                    "max_targets_per_option": 2,
                },
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "text checkpoint text_d_model=16 is incompatible with --text-d-model 32",
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

    def test_game_log_is_rewritten_and_appended(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game_logs.txt"
            path.write_text("stale")

            initialize_game_log(path)
            append_sample_game_log(
                path,
                [
                    train_mod.TranscriptAction(
                        state=cast(
                            GameStateSnapshot,
                            {
                                "players": [
                                    {"ID": "A", "Name": "A", "Life": 20},
                                    {"ID": "B", "Name": "B", "Life": 20},
                                ],
                                "active_player": "A",
                                "turn": 1,
                                "step": "Precombat Main",
                            },
                        ),
                        pending=cast(
                            PendingState,
                            {"kind": "priority", "player_idx": 0, "options": []},
                        ),
                        action={"kind": "pass"},
                    )
                ],
                episode_idx=7,
                winner_idx=0,
                max_actions=80,
                encoder="text",
            )

            text = path.read_text()

        self.assertNotIn("stale", text)
        self.assertIn("encoder=text episode=7 winner=0", text)
        self.assertIn("pass", text)

    def test_priority_trace_jsonl_appends_only_priority_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "priority.jsonl"
            transcript = [
                train_mod.TranscriptAction(
                    state=cast(
                        GameStateSnapshot,
                        {
                            "players": [
                                {"ID": "A", "Name": "A", "Life": 20},
                                {"ID": "B", "Name": "B", "Life": 20},
                            ],
                            "active_player": "A",
                            "turn": 1,
                            "step": "Precombat Main",
                        },
                    ),
                    pending=cast(
                        PendingState,
                        {"kind": "priority", "player_idx": 0, "options": []},
                    ),
                    action={"kind": "pass"},
                ),
                train_mod.TranscriptAction(
                    state=cast(
                        GameStateSnapshot,
                        {
                            "players": [],
                            "active_player": "A",
                            "turn": 1,
                            "step": "Declare Attackers",
                        },
                    ),
                    pending=cast(
                        PendingState,
                        {"kind": "attackers", "player_idx": 0, "options": []},
                    ),
                    action={"attackers": []},
                ),
            ]

            append_priority_trace_jsonl(
                path,
                transcript,
                episode_idx=3,
                winner_idx=1,
                encoder="text",
            )

            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["episode_idx"], 3)
        self.assertEqual(rows[0]["action_idx"], 0)
        self.assertEqual(rows[0]["winner_idx"], 1)
        self.assertEqual(rows[0]["encoder"], "text")
        self.assertEqual(rows[0]["pending"]["kind"], "priority")
        self.assertEqual(rows[0]["action"], {"kind": "pass"})

    def test_priority_trace_jsonl_path_defaults_to_none(self) -> None:
        self.assertIsNone(priority_trace_jsonl_path(Namespace()))
        path = Path("/tmp/priority.jsonl")
        self.assertEqual(priority_trace_jsonl_path(Namespace(priority_trace_jsonl_path=path)), path)

    def test_gae_returns_flips_bootstrap_sign_for_opponent_steps(self) -> None:
        steps = [
            RolloutStep(perspective_player_idx=0, old_log_prob=0.0, value=0.2),
            RolloutStep(perspective_player_idx=1, old_log_prob=0.0, value=-0.1),
            RolloutStep(perspective_player_idx=1, old_log_prob=0.0, value=0.3),
        ]

        returns = gae_returns(
            steps,
            terminal_reward_p0=-1.0,  # winner = p1
            zero_sum=True,
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

    def test_log_ppo_stats_includes_token_metrics(self) -> None:
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
            token_metrics={"tokens_per_encoded_state/p50": 128.0},
            log_fn=payloads.append,
            run_active=True,
        )

        self.assertEqual(payloads[0]["tokens_per_encoded_state/p50"], 128.0)

    def test_token_length_percentile_metrics(self) -> None:
        row_token_length = [10, 30, 20, 40]
        rows = [0, 2, 1, 3]

        metrics = token_length_percentile_metrics(row_token_length, rows)

        self.assertEqual(metrics["tokens_per_encoded_state/p25"], 17.5)
        self.assertEqual(metrics["tokens_per_encoded_state/p50"], 25.0)
        self.assertEqual(metrics["tokens_per_encoded_state/p75"], 32.5)

    def test_rnad_value_metrics_skips_cadence_gated_diagnostics(self) -> None:
        state = Namespace(
            last_stats=[
                RNaDStats(
                    loss=0.5,
                    critic_loss=1.0,
                    policy_loss=2.0,
                    v_hat_mean=3.0,
                    grad_norm=6.0,
                    transformed_reward_mean=4.0,
                    q_clip_fraction=0.25,
                    sampled_log_ratio_mean=100.0,
                    sampled_log_ratio_absmax=101.0,
                    is_bias_up_mean=102.0,
                    is_bias_down_mean=103.0,
                    v_target_reg_share=104.0,
                    policy_drift_diagnostics_computed=False,
                    v_target_reg_share_computed=False,
                )
            ],
            outer_iteration=5,
            gradient_step=8,
        )

        metrics = rnad_value_metrics(cast(Any, state))

        self.assertEqual(metrics["rnad/v_hat_mean"], 3.0)
        self.assertEqual(metrics["rnad/transformed_reward_mean"], 4.0)
        self.assertEqual(metrics["rnad/grad_norm"], 6.0)
        self.assertEqual(metrics["rnad/q_clip_fraction"], 0.25)
        self.assertEqual(metrics["rnad/outer_iteration"], 5)
        self.assertEqual(metrics["rnad/gradient_step"], 8)
        self.assertNotIn("rnad/sampled_log_ratio_mean", metrics)
        self.assertNotIn("rnad/sampled_log_ratio_absmax", metrics)
        self.assertNotIn("rnad/is_bias_up_mean", metrics)
        self.assertNotIn("rnad/is_bias_down_mean", metrics)
        self.assertNotIn("rnad/v_target_reg_share", metrics)

    def test_rnad_value_metrics_logs_successful_update_stats(self) -> None:
        state = Namespace(
            last_stats=[
                RNaDStats(
                    loss=0.5,
                    critic_loss=1.0,
                    policy_loss=2.0,
                    v_hat_mean=3.0,
                    grad_norm=4.0,
                    transformed_reward_mean=5.0,
                    sampled_log_ratio_mean=6.0,
                    sampled_log_ratio_absmax=7.0,
                    is_bias_up_mean=8.0,
                    is_bias_down_mean=9.0,
                    v_target_reg_share=10.0,
                    policy_drift_diagnostics_computed=True,
                    v_target_reg_share_computed=True,
                )
            ],
            outer_iteration=7,
            gradient_step=11,
        )

        metrics = rnad_value_metrics(cast(Any, state))

        self.assertEqual(metrics["rnad/v_hat_mean"], 3.0)
        self.assertEqual(metrics["rnad/grad_norm"], 4.0)
        self.assertEqual(metrics["rnad/transformed_reward_mean"], 5.0)
        self.assertEqual(metrics["rnad/sampled_log_ratio_mean"], 6.0)
        self.assertEqual(metrics["rnad/sampled_log_ratio_absmax"], 7.0)
        self.assertEqual(metrics["rnad/is_bias_up_mean"], 8.0)
        self.assertEqual(metrics["rnad/is_bias_down_mean"], 9.0)
        self.assertEqual(metrics["rnad/v_target_reg_share"], 10.0)
        self.assertEqual(metrics["rnad/outer_iteration"], 7)
        self.assertEqual(metrics["rnad/gradient_step"], 11)

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

    def test_main_slots_branch_uses_native_training_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = Namespace(
                encoder="slots",
                learning_rate=1e-3,
                trainer="ppo",
                deck_json=None,
                deck_dir=None,
                jumpstart_dir=None,
                embeddings=root / "embeddings.json",
                torch_threads=None,
                checkpoint=None,
                no_wandb=True,
                wandb_project="magic-ai",
                wandb_run_name=None,
                output=root / "ppo.pt",
                disable_opponent_pool=True,
                device="cpu",
            )
            policy = torch.nn.Linear(2, 1)
            backend = SlotTrainingBackend(
                policy=cast(PPOPolicy, policy),
                native_encoder=cast(Any, object()),
                staging_buffer=cast(Any, object()),
                batch_pool=None,
                batch_workers=1,
            )

            with (
                patch("scripts.train.parse_args", return_value=args),
                patch("scripts.train.validate_args"),
                patch("scripts.train.load_deck_pool", return_value=[{"cards": []}]),
                patch("scripts.train.validate_deck_embeddings"),
                patch("scripts.train.load_training_checkpoint", return_value=None),
                patch("scripts.train.build_slot_backend", return_value=backend) as build_slot,
                patch("scripts.train.importlib.import_module", return_value=object()),
                patch.object(
                    train_mod.ShardedNativeRolloutDriver,
                    "for_mage",
                    return_value=object(),
                ),
                patch.object(train_mod, "train_text_envs") as train_text,
                patch.object(
                    train_mod,
                    "train_native_batched_envs",
                    return_value=(TrainingResumeState(completed_games=1), None),
                ) as train_native,
                patch.object(train_mod, "save_checkpoint") as save,
                patch.object(train_mod.wandb, "finish"),
            ):
                main()

        build_slot.assert_called_once()
        train_text.assert_not_called()
        train_native.assert_called_once()
        save.assert_called_once()

    def test_main_text_branch_uses_common_save_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = Namespace(
                encoder="text",
                learning_rate=1e-3,
                trainer="ppo",
                deck_json=None,
                deck_dir=None,
                jumpstart_dir=None,
                embeddings=root / "embeddings.json",
                torch_threads=None,
                checkpoint=None,
                no_wandb=True,
                wandb_project="magic-ai",
                wandb_run_name=None,
                output=root / "text.pt",
                device="cpu",
            )
            policy = torch.nn.Linear(2, 1)
            backend = Namespace(policy=policy)

            with (
                patch("scripts.train.parse_args", return_value=args),
                patch("scripts.train.validate_args"),
                patch("scripts.train.load_deck_pool", return_value=[{"cards": []}]),
                patch("scripts.train.validate_deck_embeddings"),
                patch("scripts.train.load_training_checkpoint", return_value=None),
                patch("scripts.train.build_text_backend", return_value=backend) as build_text,
                patch("scripts.train.importlib.import_module", return_value=object()),
                patch.object(
                    train_mod,
                    "train_text_envs",
                    return_value=(TrainingResumeState(completed_games=1), None),
                ) as train_text,
                patch.object(train_mod, "train_native_batched_envs") as train_native,
                patch.object(
                    train_mod,
                    "_build_opponent_schedules",
                    return_value=(None, None, None),
                ),
                patch.object(train_mod, "build_text_opponent_policy"),
                patch.object(train_mod, "save_checkpoint") as save,
                patch.object(train_mod.wandb, "finish") as finish,
            ):
                main()

        build_text.assert_called_once()
        train_text.assert_called_once()
        train_native.assert_not_called()
        save.assert_called_once()
        finish.assert_called_once()

    def test_train_text_envs_single_game_rollout_smoke(self) -> None:
        class FakeReplayBuffer:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.row_token_length_host = [8]

            def reset(self) -> None:
                self.reset_calls += 1

        class FakePolicy(torch.nn.Module):
            spr_enabled = False

            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([0.0]))
                self.init_count = 0
                self.reset_indices: list[list[int]] = []

            def init_lstm_env_states(self, num_envs: int) -> None:
                self.init_count += num_envs

            def reset_lstm_env_states(self, env_indices: list[int]) -> None:
                self.reset_indices.append(list(env_indices))

        class FakeGame:
            def __init__(self) -> None:
                self.is_over = False
                self.winner = ""
                self.state = {
                    "players": [{"ID": "A", "Name": "A"}, {"ID": "B", "Name": "B"}],
                    "active_player": "A",
                    "turn": 1,
                    "step": "Precombat Main",
                }
                self.pending = {
                    "kind": "priority",
                    "player_idx": 0,
                    "options": [{"kind": "pass"}],
                }
                self.actions: list[dict[str, object]] = []
                self.closed = False

            def refresh_state(self) -> dict[str, object]:
                return cast(dict[str, object], self.state)

            def legal(self) -> dict[str, object]:
                return cast(dict[str, object], self.pending)

            def step(self, action: dict[str, object]) -> None:
                self.actions.append(action)

            def close(self) -> None:
                self.closed = True

        class FakeMage:
            def __init__(self) -> None:
                self.game = FakeGame()

            def new_game(self, *_args: object, **_kwargs: object) -> FakeGame:
                return self.game

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                episodes=1,
                max_steps_per_game=1,
                seed=3,
                name_a="A",
                name_b="B",
                no_shuffle=True,
                hand_size=7,
                deterministic_rollout=True,
                rollout_steps=1,
                ppo_epochs=1,
                minibatch_size=1,
                clip_epsilon=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                max_grad_norm=0.5,
                spr=False,
                spr_coef=0.0,
                gamma=1.0,
                gae_lambda=1.0,
                draw_penalty=1.0,
                save_every=0,
                output=Path(tmpdir) / "unused.pt",
                sample_actions=80,
                game_log_path=Path(tmpdir) / "game_logs.txt",
            )
            initialize_game_log(args.game_log_path)
            policy = FakePolicy()
            backend = Namespace(policy=policy, replay_buffer=FakeReplayBuffer())
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            policy_step = PolicyStep(
                action={"kind": "pass"},
                trace=ActionTrace("priority", indices=(0,)),
                log_prob=torch.tensor(-0.5),
                value=torch.tensor(0.1),
                entropy=torch.tensor(0.0),
                replay_idx=0,
                selected_choice_cols=(0,),
            )

            with (
                patch(
                    "scripts.train.sample_text_policy_batch", return_value=[policy_step]
                ) as sample,
                patch(
                    "scripts.train.ppo_update",
                    return_value=PPOStats(
                        loss=1.0,
                        policy_loss=0.5,
                        value_loss=0.25,
                        entropy=0.0,
                        approx_kl=0.0,
                        clip_fraction=0.0,
                    ),
                ) as update,
            ):
                resume, rnad_state = train_text_envs(
                    args,
                    FakeMage(),
                    [{"cards": []}],
                    cast(Any, backend),
                    optimizer,
                )
            game_log_text = args.game_log_path.read_text()

        self.assertIsNone(rnad_state)
        self.assertEqual(resume.completed_games, 1)
        self.assertEqual(resume.total_generated_rollout_steps, 1)
        self.assertEqual(resume.total_rollout_steps, 1)
        self.assertGreaterEqual(backend.replay_buffer.reset_calls, 2)
        self.assertEqual(policy.init_count, 1)
        self.assertEqual(policy.reset_indices, [[0]])
        sample.assert_called_once()
        update.assert_called_once()
        self.assertIn("encoder=text episode=0 winner=draw", game_log_text)

    def test_train_selected_backend_native_text_dispatches_to_native_loop(self) -> None:
        args = Namespace(
            encoder="text",
            native_render_plan=True,
        )
        backend = Namespace(native_encoder=object(), batch_workers=1)
        optimizer = torch.optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=1e-3)
        expected_state = TrainingResumeState(completed_games=1)

        with (
            patch.object(
                train_mod.ShardedNativeRolloutDriver,
                "for_mage",
                return_value=object(),
            ) as rollout,
            patch.object(
                train_mod,
                "train_text_native_batched_envs",
                return_value=(expected_state, None),
            ) as train_native_text,
            patch.object(train_mod, "train_text_envs") as train_text,
            patch.object(
                train_mod,
                "_build_opponent_schedules",
                return_value=(None, None, None),
            ),
            patch.object(train_mod, "build_text_opponent_policy"),
        ):
            result = train_selected_backend(
                args,
                object(),
                [{"cards": []}],
                optimizer,
                device=torch.device("cpu"),
                checkpoint_cpu=None,
                text_backend=cast(Any, backend),
            )

        self.assertEqual(result.resume_state, expected_state)
        rollout.assert_called_once()
        train_native_text.assert_called_once()
        train_text.assert_not_called()

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
                    total_wandb_logs=17,
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
        self.assertEqual(checkpoint["training_state"]["total_wandb_logs"], 17)
        self.assertEqual(checkpoint["training_state"]["snapshot_schedule_next_idx"], 1)
        self.assertEqual(checkpoint["training_state"]["retrospective_schedule_next_idx"], 1)
        self.assertEqual(
            checkpoint["training_state"]["opponent_pool"]["entries"][0]["tag"],
            "g000100_p010.0",
        )

    def test_save_checkpoint_serializes_text_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / "text.pt"
            cache_path = root / "card_tokens.pt"
            cache_path.write_bytes(b"card-cache")
            policy = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            args = Namespace(
                output=checkpoint_path,
                opponent_pool_dir=root / "opponent_pool",
                encoder="text",
                card_token_cache=cache_path,
                text_max_tokens=128,
                text_d_model=16,
                text_layers=1,
                text_heads=2,
                text_d_ff=64,
                hidden_layers=1,
                max_options=4,
                max_targets_per_option=2,
            )

            save_checkpoint(checkpoint_path, policy, optimizer, args)
            checkpoint = load_training_checkpoint(checkpoint_path)

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        metadata = checkpoint["metadata"]
        self.assertEqual(metadata["encoder"], "text")
        self.assertEqual(metadata["text_config"]["text_max_tokens"], 128)
        self.assertEqual(metadata["text_config"]["max_targets_per_option"], 2)
        self.assertEqual(metadata["card_token_cache"]["path"], str(cache_path))
        self.assertEqual(
            metadata["card_token_cache"]["sha256"],
            hashlib.sha256(b"card-cache").hexdigest(),
        )
        self.assertIn("modernbert_revision", metadata["tokenizer"])

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
        # New snapshots seed from the default TrueSkill prior.
        self.assertAlmostEqual(pool.entries[1].rating.mu, 25.0)
        self.assertAlmostEqual(pool.entries[1].rating.sigma, 25.0 / 3.0)

    def test_prune_pool_to_schedule_keeps_entries_closest_to_new_thresholds(self) -> None:
        pool = OpponentPool()
        # Original 1M-episode run: 1%/2%/4%/6%/... -> at 10k, 20k, 40k, 60k, 80k.
        for games in (10_000, 20_000, 40_000, 60_000, 80_000, 100_000):
            pool.add_snapshot(
                Path(f"snapshot_g{games:06d}.pt"),
                f"g{games:06d}_p001.0",
                snapshot_games=games,
            )

        # Extending to 2M episodes: new schedule is 20k, 40k, 80k, 120k, 160k, ...
        # At completed_games=100k, new thresholds <= completed are 20k, 40k, 80k.
        # Closest existing: 20k -> 20000, 40k -> 40000, 80k -> 80000.
        new_schedule = SnapshotSchedule.build(2_000_000)
        _prune_pool_to_schedule(pool, new_schedule, completed_games=100_000)

        kept_games = [e.snapshot_games for e in pool.entries]
        self.assertEqual(kept_games, [20_000, 40_000, 80_000])

    def test_build_opponent_schedules_resets_next_idx_for_extended_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_dir = Path(tmpdir) / "opponent_pool"
            snapshot_dir.mkdir(parents=True)
            checkpoint = {
                "training_state": {
                    "completed_games": 100_000,
                    "snapshot_schedule_next_idx": 999,  # stale; from old --episodes
                    "retrospective_schedule_next_idx": 999,
                    "opponent_pool": {
                        "entries": [
                            {
                                "path": str(snapshot_dir / f"snapshot_g{g:06d}.pt"),
                                "tag": f"g{g:06d}_p001.0",
                                "mu": 25.0,
                                "sigma": 8.0,
                                "n_games": 0,
                                "snapshot_games": g,
                            }
                            for g in (10_000, 20_000, 40_000, 60_000, 80_000, 100_000)
                        ],
                    },
                }
            }
            args = Namespace(
                episodes=2_000_000,
                opponent_pool_dir=snapshot_dir,
                disable_opponent_pool=False,
            )
            pool, snap_sched, retro_sched = _build_opponent_schedules(args, checkpoint)

        assert pool is not None and snap_sched is not None and retro_sched is not None
        # snap thresholds at 2M: 20k, 40k, 80k, 120k, ... -> 3 are <= 100k
        self.assertEqual(snap_sched.next_idx, 3)
        # Pool pruned to those three closest entries.
        self.assertEqual([e.snapshot_games for e in pool.entries], [20_000, 40_000, 80_000])
        # Retro thresholds at 2M (5% step): 100k, 200k, ... -> exactly 1 <= 100k.
        self.assertEqual(retro_sched.next_idx, 1)

    def test_validate_args_requires_no_validate_for_torch_compile(self) -> None:
        args = Namespace(
            episodes=1,
            num_envs=1,
            rollout_steps=1,
            rollout_min_ready_batch=1,
            rollout_ready_wait_ms=0.0,
            max_steps_per_game=1,
            minibatch_size=1,
            hidden_layers=1,
            gae_lambda=0.95,
            torch_compile=True,
            no_validate=False,
            deck_json=None,
            deck_dir=None,
            jumpstart_dir=None,
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
            rollout_min_ready_batch=1,
            rollout_ready_wait_ms=0.0,
            max_steps_per_game=1,
            minibatch_size=1,
            hidden_layers=1,
            gae_lambda=1.5,
            torch_compile=False,
            no_validate=False,
            deck_json=None,
            deck_dir=None,
            jumpstart_dir=None,
            eval_games_per_snapshot=0,
            eval_recency_tau=4.0,
            eval_num_envs=None,
        )

        with self.assertRaisesRegex(ValueError, "--gae-lambda must be in \\[0, 1\\]"):
            validate_args(args)

    def test_parse_args_defaults_native_text_rollout_and_assembler_on(self) -> None:
        with patch("sys.argv", ["train.py", "--encoder", "text"]):
            args = train_mod.parse_args()

        self.assertTrue(args.native_render_plan)
        self.assertTrue(args.text_native_assembler)
        self.assertIsNone(args.minibatch_token_limit)

    def test_validate_args_sets_text_minibatch_token_limit_default(self) -> None:
        args = Namespace(
            episodes=1,
            num_envs=1,
            rollout_steps=1,
            rollout_min_ready_batch=1,
            rollout_ready_wait_ms=0.0,
            max_steps_per_game=1,
            minibatch_size=512,
            minibatch_token_limit=None,
            hidden_layers=1,
            gae_lambda=0.95,
            torch_compile=False,
            no_validate=False,
            deck_json=None,
            deck_dir=None,
            jumpstart_dir=None,
            eval_games_per_snapshot=0,
            eval_recency_tau=4.0,
            eval_num_envs=None,
            encoder="text",
            trainer="ppo",
            spr=True,
            native_render_plan=True,
            text_native_assembler=True,
            text_max_tokens=1024,
        )

        validate_args(args)

        self.assertEqual(
            args.minibatch_token_limit,
            train_mod.DEFAULT_TEXT_MINIBATCH_TOKEN_LIMIT,
        )

    def test_validate_args_can_disable_minibatch_token_limit(self) -> None:
        args = Namespace(
            episodes=1,
            num_envs=1,
            rollout_steps=1,
            rollout_min_ready_batch=1,
            rollout_ready_wait_ms=0.0,
            max_steps_per_game=1,
            minibatch_size=512,
            minibatch_token_limit=0,
            hidden_layers=1,
            gae_lambda=0.95,
            torch_compile=False,
            no_validate=False,
            deck_json=None,
            deck_dir=None,
            jumpstart_dir=None,
            eval_games_per_snapshot=0,
            eval_recency_tau=4.0,
            eval_num_envs=None,
            encoder="text",
            trainer="ppo",
            spr=True,
            native_render_plan=True,
            text_native_assembler=True,
            text_max_tokens=1024,
        )

        validate_args(args)

        self.assertIsNone(args.minibatch_token_limit)

    def test_validate_args_text_native_rollout_can_be_disabled_for_ppo(self) -> None:
        args = Namespace(
            episodes=1,
            num_envs=1,
            rollout_steps=1,
            rollout_min_ready_batch=1,
            rollout_ready_wait_ms=0.0,
            max_steps_per_game=1,
            minibatch_size=1,
            hidden_layers=1,
            gae_lambda=0.95,
            torch_compile=False,
            no_validate=False,
            deck_json=None,
            deck_dir=None,
            jumpstart_dir=None,
            eval_games_per_snapshot=0,
            eval_recency_tau=4.0,
            eval_num_envs=None,
            encoder="text",
            trainer="ppo",
            spr=True,
            native_render_plan=False,
            text_native_assembler=True,
        )

        validate_args(args)
        self.assertFalse(args.spr)
        self.assertFalse(args.native_render_plan)

    def test_validate_args_text_rnad_requires_native_text_rollout(self) -> None:
        args = Namespace(
            episodes=1,
            num_envs=1,
            rollout_steps=1,
            rollout_min_ready_batch=1,
            rollout_ready_wait_ms=0.0,
            max_steps_per_game=1,
            minibatch_size=1,
            hidden_layers=1,
            gae_lambda=0.95,
            torch_compile=False,
            no_validate=False,
            deck_json=None,
            deck_dir=None,
            jumpstart_dir=None,
            eval_games_per_snapshot=0,
            eval_recency_tau=4.0,
            eval_num_envs=None,
            encoder="text",
            trainer="rnad",
            spr=True,
            native_render_plan=False,
            text_native_assembler=False,
        )

        args.trainer = "rnad"
        with self.assertRaisesRegex(ValueError, "requires --native-text-rollout"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
