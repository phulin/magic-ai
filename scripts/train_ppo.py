#!/usr/bin/env python3
"""Train a PPO self-play policy against the mage-go Python engine."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from dotenv import load_dotenv

load_dotenv()

import wandb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.actions import ActionRequest  # noqa: E402
from magic_ai.buffer import NativeTrajectoryBuffer  # noqa: E402
from magic_ai.game_state import (  # noqa: E402
    GameStateEncoder,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
)
from magic_ai.model import PPOPolicy  # noqa: E402
from magic_ai.native_encoder import NativeBatchEncoder  # noqa: E402
from magic_ai.native_rollout import (  # noqa: E402
    NativeRolloutDriver,
    NativeRolloutUnavailable,
)
from magic_ai.opponent_pool import (  # noqa: E402
    OpponentEntry,
    OpponentPool,
    SnapshotSchedule,
    build_opponent_policy,
    opponent_policy_state_dict,
    run_eval_matches,
    save_snapshot,
    snapshot_tag,
)
from magic_ai.ppo import PPOStats, RolloutStep, gae_returns, ppo_update  # noqa: E402

DEFAULT_DECK = {
    "name": "bolt-mountain",
    "cards": [
        {"name": "Mountain", "count": 24},
        {"name": "Lightning Bolt", "count": 36},
    ],
}


@dataclass(frozen=True)
class TranscriptAction:
    state: GameStateSnapshot
    pending: PendingState
    action: ActionRequest


@dataclass
class LiveGame:
    game: Any
    transcript_game: Any | None
    slot_idx: int
    episode_idx: int
    episode_steps: list[RolloutStep]
    transcript: list[TranscriptAction]
    action_count: int = 0


@dataclass
class WinFractionStats:
    p1_wins: int = 0
    p2_wins: int = 0
    draws: int = 0

    def record(self, bucket: str) -> None:
        if bucket == "p1":
            self.p1_wins += 1
        elif bucket == "p2":
            self.p2_wins += 1
        else:
            self.draws += 1

    @property
    def total_games(self) -> int:
        return self.p1_wins + self.p2_wins + self.draws

    def as_wandb_metrics(self) -> dict[str, float]:
        total = self.total_games
        if total == 0:
            return {}
        denom = float(total)
        return {
            "p1_win_fraction": self.p1_wins / denom,
            "p2_win_fraction": self.p2_wins / denom,
            "draw_fraction": self.draws / denom,
            "window_games": float(total),
        }

    def reset(self) -> None:
        self.p1_wins = 0
        self.p2_wins = 0
        self.draws = 0


def main() -> None:
    args = parse_args()
    validate_args(args)
    deck_pool = load_deck_pool(args.deck_json, args.deck_dir)
    validate_deck_embeddings(args.embeddings, deck_pool)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    game_state_encoder = GameStateEncoder.from_embedding_json(args.embeddings, d_model=args.d_model)
    rollout_capacity = args.rollout_buffer_capacity or max(
        4096, args.rollout_steps + 400 * args.num_envs
    )
    policy = PPOPolicy(
        game_state_encoder,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        rollout_capacity=rollout_capacity,
        use_lstm=args.lstm,
        spr_enabled=args.spr,
        spr_action_dim=args.spr_action_dim,
        spr_ema_decay=args.spr_ema_decay,
        validate=not args.no_validate,
        compile_forward=args.torch_compile,
    ).to(device)
    policy.init_lstm_env_states(args.num_envs)
    if device.type == "cuda":
        # Force cuBLAS handle creation before rollout ingestion creates temporary
        # CUDA copy tensors that PyTorch may hold in its caching allocator.
        _ = torch.empty((1, 1), device=device) @ torch.empty((1, 1), device=device)
    native_encoder = NativeBatchEncoder.for_policy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    staging_decision_capacity_per_env = max(
        1,
        (policy.rollout_buffer.decision_capacity // max(1, policy.rollout_buffer.capacity))
        * args.max_steps_per_game,
    )
    staging_buffer = NativeTrajectoryBuffer(
        num_envs=args.num_envs,
        max_steps_per_trajectory=args.max_steps_per_game,
        decision_capacity_per_env=staging_decision_capacity_per_env,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        max_cached_choices=policy.max_cached_choices,
        zone_slot_count=policy.rollout_buffer.slot_card_rows.shape[1],
        game_info_dim=policy.rollout_buffer.game_info.shape[1],
        option_scalar_dim=policy.rollout_buffer.option_scalars.shape[2],
        target_scalar_dim=policy.rollout_buffer.target_scalars.shape[3],
        recurrent_layers=policy.hidden_layers if policy.use_lstm else 0,
        recurrent_hidden_dim=policy.hidden_dim if policy.use_lstm else 0,
    ).to(device)

    if args.checkpoint and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["policy"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    mage = importlib.import_module("mage")
    try:
        native_rollout = NativeRolloutDriver.for_mage(mage)
    except NativeRolloutUnavailable as exc:
        raise SystemExit(f"native rollout is unavailable: {exc}") from exc

    opponent_pool: OpponentPool | None = None
    snapshot_schedule: SnapshotSchedule | None = None
    opponent_policy: PPOPolicy | None = None
    if not args.disable_opponent_pool:
        opponent_pool = OpponentPool()
        snapshot_schedule = SnapshotSchedule.build(args.episodes)
        opponent_policy = build_opponent_policy(policy, device)

    train_native_batched_envs(
        args,
        mage,
        deck_pool,
        policy,
        native_encoder,
        optimizer,
        native_rollout,
        staging_buffer,
        opponent_pool=opponent_pool,
        snapshot_schedule=snapshot_schedule,
        opponent_policy=opponent_policy,
    )

    save_checkpoint(args.output, policy, optimizer, args)
    print(f"saved checkpoint -> {args.output}")
    wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO self-play training with mage-go.")
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ppo.pt"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--deck-json", type=Path, default=None)
    parser.add_argument(
        "--deck-dir",
        type=Path,
        default=None,
        help="directory of deck JSON files to sample randomly per game",
    )
    parser.add_argument("--episodes", type=int, default=65536)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument(
        "--rollout-buffer-capacity",
        type=int,
        default=None,
        help="rows in the rollout GPU buffer; default max(4096, rollout-steps + 400*num-envs)",
    )
    parser.add_argument("--max-steps-per-game", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hand-size", type=int, default=7)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--deterministic-rollout", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument(
        "--lstm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use an LSTM policy core (default: on; pass --no-lstm to disable)",
    )
    parser.add_argument(
        "--spr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add a self-predictive (SPR) auxiliary loss on the LSTM latent "
        "(default: on; pass --no-spr to disable)",
    )
    parser.add_argument("--spr-coef", type=float, default=0.1)
    parser.add_argument("--spr-ema-decay", type=float, default=0.99)
    parser.add_argument("--spr-action-dim", type=int, default=32)
    parser.add_argument("--max-options", type=int, default=64)
    parser.add_argument("--max-targets-per-option", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=8192)
    parser.add_argument(
        "--sample-actions",
        type=int,
        default=80,
        help="maximum actions to print from a sample rollout game at each PPO update",
    )
    parser.add_argument("--wandb-project", default="magic-ai")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="compile the pure tensor policy core with torch.compile(dynamic=True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip runtime tensor validation checks (required for torch.compile)",
    )
    parser.add_argument(
        "--disable-opponent-pool",
        action="store_true",
        help="skip snapshotting, TrueSkill eval, and opponent-pool logging",
    )
    parser.add_argument(
        "--opponent-pool-dir",
        type=Path,
        default=Path("checkpoints/opponent_pool"),
        help="directory to store frozen opponent snapshots",
    )
    parser.add_argument(
        "--eval-rounds-per-snapshot",
        type=int,
        default=5,
        help="number of random opponents to evaluate against each time a snapshot is taken",
    )
    parser.add_argument(
        "--eval-games-per-round",
        type=int,
        default=50,
        help="eval games played against each sampled opponent",
    )
    parser.add_argument(
        "--eval-num-envs",
        type=int,
        default=None,
        help="parallel envs during eval; defaults to --num-envs",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be at least 1")
    if args.max_steps_per_game < 1:
        raise ValueError("--max-steps-per-game must be at least 1")
    if args.minibatch_size < 1:
        raise ValueError("--minibatch-size must be at least 1")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be in [0, 1]")
    if args.hidden_layers < 1:
        raise ValueError("--hidden-layers must be at least 1")
    if args.torch_compile and not args.no_validate:
        raise ValueError("--torch-compile requires --no-validate")
    if args.deck_json is not None and args.deck_dir is not None:
        raise ValueError("--deck-json and --deck-dir are mutually exclusive")
    if args.eval_rounds_per_snapshot < 0:
        raise ValueError("--eval-rounds-per-snapshot must be non-negative")
    if args.eval_games_per_round < 0:
        raise ValueError("--eval-games-per-round must be non-negative")
    if args.eval_num_envs is not None and args.eval_num_envs < 1:
        raise ValueError("--eval-num-envs must be at least 1")


def log_ppo_stats(
    stats: PPOStats,
    *,
    games: int,
    steps: int,
    win_stats: WinFractionStats | None = None,
    value_metrics: dict[str, float] | None = None,
) -> None:
    """Log PPO update metrics to wandb (if active)."""
    if wandb.run is None:
        return
    payload = {
        "loss": stats.loss,
        "policy_loss": stats.policy_loss,
        "value_loss": stats.value_loss,
        "entropy": stats.entropy,
        "approx_kl": stats.approx_kl,
        "clip_fraction": stats.clip_fraction,
        "spr_loss": stats.spr_loss,
        "games": games,
        "rollout_steps": steps,
    }
    if win_stats is not None:
        payload.update(win_stats.as_wandb_metrics())
    if value_metrics is not None:
        payload.update(value_metrics)
    wandb.log(payload)


def rollout_value_metrics(
    steps: list[RolloutStep],
    returns: torch.Tensor,
) -> dict[str, float]:
    """Summarize rollout return targets and sampled value predictions."""
    return_values = returns.detach().to(dtype=torch.float32, device="cpu")
    predicted_values = torch.tensor(
        [step.value for step in steps],
        dtype=torch.float32,
    )
    return {
        "return_mean": float(return_values.mean().item()),
        "return_std": float(return_values.std(unbiased=False).item()),
        "value_mean": float(predicted_values.mean().item()),
        "value_std": float(predicted_values.std(unbiased=False).item()),
    }


def train_native_batched_envs(
    args: argparse.Namespace,
    mage: Any,
    deck_pool: list[dict[str, Any]],
    policy: PPOPolicy,
    native_encoder: NativeBatchEncoder,
    optimizer: torch.optim.Optimizer,
    native_rollout: NativeRolloutDriver,
    staging_buffer: NativeTrajectoryBuffer,
    *,
    opponent_pool: OpponentPool | None = None,
    snapshot_schedule: SnapshotSchedule | None = None,
    opponent_policy: PPOPolicy | None = None,
) -> None:
    if not native_encoder.is_available:
        raise SystemExit("native rollout requires MageEncodeBatch")
    eval_rng = random.Random(args.seed ^ 0x5EED5)

    pending_steps: list[RolloutStep] = []
    pending_returns: list[torch.Tensor] = []
    completed_games = 0
    last_saved_games = 0
    next_episode_idx = 0
    live_games: list[LiveGame] = []
    free_slots = list(range(args.num_envs - 1, -1, -1))
    win_stats = WinFractionStats()
    policy.reset_rollout_buffer()

    def start_game(slot_idx: int, episode_idx: int) -> LiveGame:
        staging_buffer.reset_env(slot_idx)
        policy.reset_lstm_env_states([slot_idx])
        seed = args.seed + episode_idx
        deck_a, deck_b = sample_decks(deck_pool, seed)
        return LiveGame(
            game=mage.new_game(
                deck_a,
                deck_b,
                name_a=args.name_a,
                name_b=args.name_b,
                seed=seed,
                shuffle=not args.no_shuffle,
                hand_size=args.hand_size,
            ),
            transcript_game=(
                mage.new_game(
                    deck_a,
                    deck_b,
                    name_a=args.name_a,
                    name_b=args.name_b,
                    seed=seed,
                    shuffle=not args.no_shuffle,
                    hand_size=args.hand_size,
                )
                if slot_idx == 0
                else None
            ),
            slot_idx=slot_idx,
            episode_idx=episode_idx,
            episode_steps=[],
            transcript=[],
        )

    def maybe_start_games() -> None:
        nonlocal next_episode_idx
        while free_slots and next_episode_idx < args.episodes:
            live_games.append(start_game(free_slots.pop(), next_episode_idx))
            next_episode_idx += 1

    def finish_games(finished: list[tuple[LiveGame, int]]) -> None:
        nonlocal completed_games
        if not finished:
            return

        envs = [env for env, _ in finished]
        replay_rows_by_env = policy.append_staged_episodes_to_rollout(
            staging_buffer,
            [env.slot_idx for env in envs],
        )
        for (env, winner_idx), replay_rows in zip(finished, replay_rows_by_env, strict=True):
            env.game.close()
            if env.transcript_game is not None:
                env.transcript_game.close()
            step_count = staging_buffer.active_step_count(env.slot_idx)
            if step_count:
                player_indices = (
                    staging_buffer.perspective_player_idx[env.slot_idx, :step_count]
                    .detach()
                    .cpu()
                    .tolist()
                )
                old_log_probs = (
                    staging_buffer.old_log_prob[env.slot_idx, :step_count].detach().cpu().tolist()
                )
                values = staging_buffer.value[env.slot_idx, :step_count].detach().cpu().tolist()
                env.episode_steps = [
                    RolloutStep(
                        perspective_player_idx=int(player_idx),
                        old_log_prob=float(old_log_prob),
                        value=float(value),
                        replay_idx=replay_idx,
                    )
                    for player_idx, old_log_prob, value, replay_idx in zip(
                        player_indices, old_log_probs, values, replay_rows, strict=True
                    )
                ]
            if env.episode_steps:
                returns = gae_returns(
                    env.episode_steps,
                    winner_idx=winner_idx,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
                pending_steps.extend(env.episode_steps)
                pending_returns.append(returns)
            if env.slot_idx == 0:
                print_sample_game(
                    env.transcript,
                    winner_idx=winner_idx,
                    max_actions=args.sample_actions,
                )
            staging_buffer.reset_env(env.slot_idx)
            free_slots.append(env.slot_idx)
            if winner_idx == 0:
                win_stats.p1_wins += 1
            elif winner_idx == 1:
                win_stats.p2_wins += 1
            else:
                win_stats.draws += 1
            completed_games += 1

    maybe_start_games()
    while live_games:
        ready_t, over_t, player_t, winner_t = native_rollout.poll([env.game for env in live_games])
        ready_envs: list[LiveGame] = []
        ready_players: list[int] = []
        still_live: list[LiveGame] = []
        finished_games: list[tuple[LiveGame, int]] = []
        for idx, env in enumerate(live_games):
            if int(over_t[idx]) or env.action_count >= args.max_steps_per_game:
                finished_games.append((env, int(winner_t[idx]) if int(over_t[idx]) else -1))
                continue
            still_live.append(env)
            if int(ready_t[idx]):
                ready_envs.append(env)
                ready_players.append(int(player_t[idx]))
        live_games = still_live
        finish_games(finished_games)

        if ready_envs:
            ready_env_indices = [env.slot_idx for env in ready_envs]
            parsed_batch = native_encoder.encode_handles(
                [env.game for env in ready_envs],
                perspective_player_indices=ready_players,
            )
            lstm_state_inputs = policy.lstm_env_state_inputs(ready_env_indices)
            with torch.no_grad():
                policy_steps = policy.sample_native_batch(
                    parsed_batch,
                    env_indices=ready_env_indices,
                    deterministic=args.deterministic_rollout,
                )
            log_probs = torch.stack([policy_step.log_prob for policy_step in policy_steps])
            values = torch.stack([policy_step.value for policy_step in policy_steps])

            starts: list[int] = []
            counts: list[int] = []
            selected_cols: list[int] = []
            may_selected: list[int] = []
            cursor = 0
            for env, player_idx, policy_step in zip(
                ready_envs,
                ready_players,
                policy_steps,
                strict=True,
            ):
                cols = list(policy_step.selected_choice_cols)
                if env.transcript_game is not None:
                    transcript_pending = env.transcript_game.pending or env.transcript_game.legal()
                    if transcript_pending is None:
                        raise RuntimeError("transcript shadow game is missing a pending action")
                    transcript_action = copy.deepcopy(policy_step.action)
                    if policy_step.trace.kind != "may":
                        _trace, decoded_action = policy._decode_action(
                            policy_step.trace.kind,
                            cast(PendingState, transcript_pending),
                            cols,
                        )
                        transcript_action = copy.deepcopy(decoded_action)
                    env.transcript.append(
                        TranscriptAction(
                            state=cast(GameStateSnapshot, copy.deepcopy(env.transcript_game.state)),
                            pending=cast(PendingState, copy.deepcopy(transcript_pending)),
                            action=transcript_action,
                        )
                    )
                    env.transcript_game.step(cast(dict[Any, Any], transcript_action))
                starts.append(cursor)
                counts.append(len(cols))
                selected_cols.extend(cols)
                may_selected.append(policy_step.may_selected)
                cursor += len(cols)
                env.action_count += 1

            selected_choice_cols_flat = torch.tensor(
                selected_cols,
                dtype=torch.long,
                device=policy.device,
            )

            staging_buffer.stage_batch(
                ready_env_indices,
                parsed_batch,
                selected_choice_cols_flat=selected_choice_cols_flat,
                may_selected=may_selected,
                old_log_probs=log_probs,
                values=values,
                perspective_player_indices=ready_players,
                lstm_h_in=lstm_state_inputs[0] if lstm_state_inputs is not None else None,
                lstm_c_in=lstm_state_inputs[1] if lstm_state_inputs is not None else None,
            )

            native_rollout.step_by_choice(
                [env.game for env in ready_envs],
                decision_starts=starts,
                decision_counts=counts,
                selected_choice_cols=selected_cols,
                may_selected=may_selected,
                max_options=args.max_options,
                max_targets_per_option=args.max_targets_per_option,
            )

        if len(pending_steps) >= args.rollout_steps:
            rollout_returns = torch.cat(pending_returns)
            stats = ppo_update(
                policy,
                optimizer,
                pending_steps,
                rollout_returns,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_epsilon=args.clip_epsilon,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                spr_coef=args.spr_coef if args.spr else 0.0,
            )
            print(
                "update",
                f"games={completed_games}",
                f"steps={len(pending_steps)}",
                f"loss={stats.loss:.4f}",
                f"policy={stats.policy_loss:.4f}",
                f"value={stats.value_loss:.4f}",
                f"entropy={stats.entropy:.4f}",
                f"kl={stats.approx_kl:.4f}",
                f"clip={stats.clip_fraction:.3f}",
                flush=True,
            )
            log_ppo_stats(
                stats,
                games=completed_games,
                steps=len(pending_steps),
                win_stats=win_stats,
                value_metrics=rollout_value_metrics(pending_steps, rollout_returns),
            )
            policy.reset_rollout_buffer()
            pending_steps.clear()
            pending_returns.clear()
            win_stats.reset()

        if (
            args.save_every
            and completed_games > 0
            and completed_games % args.save_every == 0
            and completed_games != last_saved_games
        ):
            save_checkpoint(args.output, policy, optimizer, args)
            last_saved_games = completed_games

        if (
            opponent_pool is not None
            and snapshot_schedule is not None
            and opponent_policy is not None
        ):
            fired = snapshot_schedule.fire(completed_games)
            for threshold in fired:
                take_snapshot_and_eval(
                    args=args,
                    threshold=threshold,
                    policy=policy,
                    opponent_policy=opponent_policy,
                    opponent_pool=opponent_pool,
                    native_encoder=native_encoder,
                    native_rollout=native_rollout,
                    mage=mage,
                    deck_pool=deck_pool,
                    rng=eval_rng,
                )

        maybe_start_games()

    if pending_steps:
        rollout_returns = torch.cat(pending_returns)
        stats = ppo_update(
            policy,
            optimizer,
            pending_steps,
            rollout_returns,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            spr_coef=args.spr_coef if args.spr else 0.0,
        )
        print(
            "final_update",
            f"games={completed_games}",
            f"steps={len(pending_steps)}",
            f"loss={stats.loss:.4f}",
            f"policy={stats.policy_loss:.4f}",
            f"value={stats.value_loss:.4f}",
            f"entropy={stats.entropy:.4f}",
            flush=True,
        )
        log_ppo_stats(
            stats,
            games=completed_games,
            steps=len(pending_steps),
            win_stats=win_stats,
            value_metrics=rollout_value_metrics(pending_steps, rollout_returns),
        )
        policy.reset_rollout_buffer()


def take_snapshot_and_eval(
    *,
    args: argparse.Namespace,
    threshold: int,
    policy: PPOPolicy,
    opponent_policy: PPOPolicy,
    opponent_pool: OpponentPool,
    native_encoder: NativeBatchEncoder,
    native_rollout: NativeRolloutDriver,
    mage: Any,
    deck_pool: list[dict[str, Any]],
    rng: random.Random,
) -> None:
    tag = snapshot_tag(threshold, args.episodes)
    snapshot_path = save_snapshot(policy, args.opponent_pool_dir, tag)
    opponent_pool.add_snapshot(
        snapshot_path,
        tag,
    ).cached_policy = opponent_policy_state_dict(policy)
    print(
        f"pool: snapshot {tag} -> {snapshot_path} (pool size={len(opponent_pool.entries)})",
        flush=True,
    )

    if args.eval_rounds_per_snapshot == 0 or args.eval_games_per_round == 0:
        if wandb.run is not None:
            wandb.log(
                {
                    **opponent_pool.main_rating_metrics(),
                    "eval/snapshot_games": float(threshold),
                }
            )
        return

    sampled_opponents: list[OpponentEntry] = []
    for _round_idx in range(args.eval_rounds_per_snapshot):
        opponent = opponent_pool.sample(rng)
        if opponent is None:
            break
        sampled_opponents.append(opponent)

    if not sampled_opponents:
        return

    eval_num_envs = (
        args.eval_num_envs
        if args.eval_num_envs is not None
        else args.eval_rounds_per_snapshot * args.eval_games_per_round
    )
    seed_base = args.seed + threshold * 1000
    metrics = run_eval_matches(
        main_policy=policy,
        opponent_policy=opponent_policy,
        opponents=sampled_opponents,
        pool=opponent_pool,
        native_encoder=native_encoder,
        native_rollout=native_rollout,
        mage=mage,
        deck_pool=deck_pool,
        num_games_per_opponent=args.eval_games_per_round,
        num_envs=eval_num_envs,
        max_steps_per_game=args.max_steps_per_game,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        hand_size=args.hand_size,
        name_a=args.name_a,
        name_b=args.name_b,
        no_shuffle=args.no_shuffle,
        seed_base=seed_base,
        rng=rng,
    )
    main_rating = opponent_pool.main_rating
    assert main_rating is not None
    for round_idx, opponent in enumerate(sampled_opponents):
        print(
            f"eval: snapshot_tag={tag} round={round_idx} "
            f"opponent={opponent.tag} "
            f"main_win={metrics[f'eval/round_{round_idx}_main_win_fraction']:.2f} "
            f"main_rating=mu={main_rating.mu:.2f},"
            f"sigma={main_rating.sigma:.2f}",
            flush=True,
        )

    if wandb.run is not None:
        payload = {
            **metrics,
            "eval/snapshot_games": float(threshold),
            "eval/new_snapshot_tag": tag,
        }
        wandb.log(payload)

    if wandb.run is not None:
        wandb.log(
            {
                **opponent_pool.main_rating_metrics(),
                "eval/snapshot_games": float(threshold),
                "eval/new_snapshot_tag": tag,
            }
        )


def load_deck_pool(deck_json: Path | None, deck_dir: Path | None) -> list[dict[str, Any]]:
    if deck_dir is not None:
        return load_deck_dir(deck_dir)
    return list(load_decks(deck_json))


def load_deck_dir(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"deck directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"deck directory path is not a directory: {path}")

    decks: list[dict[str, Any]] = []
    for deck_path in sorted(path.glob("*.json")):
        payload = json.loads(deck_path.read_text())
        decks.append(cast(dict[str, Any], payload))

    if not decks:
        raise ValueError(f"deck directory contains no JSON decks: {path}")
    return decks


def sample_decks(
    deck_pool: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not deck_pool:
        raise ValueError("deck pool must contain at least one deck")
    rng = random.Random(seed)
    return rng.choice(deck_pool), rng.choice(deck_pool)


def load_decks(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        return dict(DEFAULT_DECK), dict(DEFAULT_DECK)

    payload = json.loads(path.read_text())
    if "player_a" in payload or "player_b" in payload:
        return (
            cast(dict[str, Any], payload.get("player_a", DEFAULT_DECK)),
            cast(dict[str, Any], payload.get("player_b", DEFAULT_DECK)),
        )
    return cast(dict[str, Any], payload), cast(dict[str, Any], payload)


def validate_deck_embeddings(
    embeddings_path: Path,
    decks: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> None:
    embedded_names = load_embedded_card_names(embeddings_path)
    missing: dict[str, dict[str, int]] = {}
    for idx, deck in enumerate(decks):
        label = deck_label(deck, idx, len(decks))
        for name, count in deck_card_counts(deck).items():
            if card_name_key(name) in embedded_names:
                continue
            missing.setdefault(name, {})[label] = count

    if not missing:
        return

    details = []
    for name in sorted(missing, key=str.casefold):
        counts = ", ".join(f"{label}={count}" for label, count in sorted(missing[name].items()))
        details.append(f"{name} ({counts})")
    raise ValueError(
        f"{embeddings_path} is missing embeddings for {len(missing)} deck cards: "
        + "; ".join(details)
    )


def load_embedded_card_names(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    names: set[str] = set()
    for record in payload.get("cards", []):
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if isinstance(name, str) and record.get("embedding") is not None:
            names.add(card_name_key(name))
    return names


def deck_card_counts(deck: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    cards = deck.get("cards", [])
    if not isinstance(cards, list):
        return counts
    for card in cards:
        if not isinstance(card, dict):
            continue
        name = card.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        count = card.get("count", 1)
        counts[name] = counts.get(name, 0) + int(count)
    return counts


def deck_label(deck: dict[str, Any], idx: int, deck_count: int) -> str:
    name = deck.get("name")
    if isinstance(name, str) and name.strip():
        return name
    if deck_count == 2:
        return "player_a" if idx == 0 else "player_b"
    return f"deck_{idx + 1}"


def card_name_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def save_checkpoint(
    path: Path,
    policy: PPOPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def print_sample_game(
    transcript: list[TranscriptAction],
    *,
    winner_idx: int,
    max_actions: int,
) -> None:
    print()
    if not transcript:
        print("(no actions)")
    else:
        condensed = condense_transcript_lines(transcript, max_actions=max_actions)
        if condensed:
            turn_width = max(len(line["turn"]) for line in condensed)
            step_width = max(len(line["step"]) for line in condensed)
            player_width = max(len(line["player"]) for line in condensed)
            life_width = max(len(line["life"]) for line in condensed)
            for line in condensed:
                print(
                    f"{line['turn']:<{turn_width}}  "
                    f"{line['step']:<{step_width}}  "
                    f"{line['player']:<{player_width}}  "
                    f"{line['life']:<{life_width}}  "
                    f"{line['action']}"
                )
        remaining = len(transcript) - max_actions
        if remaining > 0:
            print(f"... {remaining} more actions")
    if winner_idx >= 0:
        print(f"== PLAYER {winner_idx + 1} WINS ==")
    else:
        print("== DRAW ==")
    print()


def condense_transcript_lines(
    transcript: list[TranscriptAction],
    *,
    max_actions: int,
) -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    for item in transcript[:max_actions]:
        line = {
            "turn": format_turn_number(item.state),
            "step": format_step_name(str(item.state.get("step", ""))),
            "player": f"P{int(item.pending.get('player_idx', 0)) + 1}",
            "life": format_life_totals(item.state),
            "action": describe_action(item),
        }
        if (
            line["action"] == "pass"
            and lines
            and lines[-1]["action"].startswith("pass")
            and lines[-1]["turn"] == line["turn"]
            and lines[-1]["step"] == line["step"]
            and lines[-1]["player"] == line["player"]
            and lines[-1]["life"] == line["life"]
        ):
            prev = lines[-1]["action"]
            if prev == "pass":
                lines[-1]["action"] = "pass x2"
            else:
                count = int(prev.rsplit("x", 1)[1])
                lines[-1]["action"] = f"pass x{count + 1}"
            continue
        lines.append(line)
    return lines


def format_step_name(step: str) -> str:
    normalized = " ".join(step.split()).casefold()
    aliases = {
        "untap": "untap",
        "upkeep": "upk",
        "draw": "draw",
        "precombat main": "pre",
        "begin combat": "bcom",
        "declare attackers": "atk",
        "declare blockers": "blk",
        "combat damage": "dmg",
        "end combat": "ecom",
        "end of combat": "ecom",
        "postcombat main": "post",
        "end step": "end",
        "cleanup": "clnp",
    }
    if normalized in aliases:
        return aliases[normalized]
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "unknown"


def format_turn_number(state: GameStateSnapshot) -> str:
    raw_turn = int(state.get("turn", 0))
    turn = max(1, (raw_turn + 1) // 2)
    suffix = "?"
    active_player = str(state.get("active_player", ""))
    players = state.get("players", [])
    if players:
        player_a = players[0]
        player_a_ids = {str(player_a.get("ID", "")), str(player_a.get("Name", ""))}
        suffix = "A" if active_player in player_a_ids else "B"
    return f"{turn}{suffix}"


def format_life_totals(state: GameStateSnapshot) -> str:
    players = state.get("players", [])
    p1_life = int(players[0].get("Life", 0)) if players else 0
    p2_life = int(players[1].get("Life", 0)) if len(players) > 1 else 0
    return f"{p1_life:>2}-{p2_life:<2}"


def describe_action(item: TranscriptAction) -> str:
    action = item.action
    pending = item.pending
    action_kind = action.get("kind", "")
    if action_kind == "pass":
        return "pass"
    if action_kind == "play_land":
        return f"play {_card_name_for_id(pending, action.get('card_id', ''))}"
    if action_kind == "cast_spell":
        name = _card_name_for_id(pending, action.get("card_id", ""))
        return _with_targets(f"play {name}", item, action.get("targets", []))
    if action_kind == "activate_ability":
        name = _card_label_for_id(item, action.get("permanent_id", ""))
        ability_index = int(action.get("ability_index", 0))
        return _with_targets(
            f"activate {name} ability {ability_index}",
            item,
            action.get("targets", []),
        )

    if "attackers" in action:
        attackers = [_card_label_for_id(item, attacker_id) for attacker_id in action["attackers"]]
        if not attackers:
            return "attack with no creatures"
        return "attack with " + ", ".join(attackers)

    if "blockers" in action:
        assignments = []
        for assignment in action["blockers"]:
            blocker = _card_label_for_id(item, assignment.get("blocker", ""))
            attacker = _card_label_for_id(item, assignment.get("attacker", ""))
            assignments.append(f"{blocker} blocks {attacker}")
        if not assignments:
            return "block with no creatures"
        return "; ".join(assignments)

    if "selected_ids" in action:
        selected = [
            _card_name_for_id(pending, selected_id) for selected_id in action["selected_ids"]
        ]
        return "choose " + (", ".join(selected) if selected else "nothing")
    if "selected_index" in action:
        idx = int(action["selected_index"])
        return f"choose {_option_label(pending, idx)}"
    if "selected_color" in action:
        return f"choose {action['selected_color']}"
    if "accepted" in action:
        return "accept" if action["accepted"] else "decline"
    return str(dict(action))


def _with_targets(
    prefix: str,
    item: TranscriptAction,
    target_ids: list[str],
) -> str:
    if not target_ids:
        return prefix
    targets = [_target_label_for_id(item, target_id) for target_id in target_ids]
    return f"{prefix}, target {', '.join(targets)}"


def _card_name_for_id(pending: PendingState, object_id: str) -> str:
    if not object_id:
        return "unknown"
    for option in pending.get("options", []):
        if _option_ids(option) & {object_id}:
            return option.get("card_name") or option.get("label") or object_id
        for target in option.get("valid_targets") or []:
            if target.get("id") == object_id:
                return target.get("label") or object_id
    return object_id


def _target_label_for_id(item: TranscriptAction, target_id: str) -> str:
    if not target_id:
        return "unknown"
    for player_idx, player in enumerate(item.state.get("players", [])):
        if target_id in {player.get("ID"), player.get("Name")}:
            return f"P{player_idx + 1}"
    state_label = _card_label_for_id(item, target_id)
    if state_label != target_id:
        return state_label
    label = _card_name_for_id(item.pending, target_id)
    return label if label != target_id else target_id


def _card_label_for_id(item: TranscriptAction, object_id: str) -> str:
    if not object_id:
        return "unknown"
    card = _state_card_for_id(item.state, object_id)
    if card is None:
        return _card_name_for_id(item.pending, object_id)

    name = str(card.get("Name") or object_id)
    power = card.get("Power", card.get("power"))
    toughness = card.get("Toughness", card.get("toughness"))
    if power is not None and toughness is not None:
        return f"{name} {power}/{toughness}"
    return name


def _state_card_for_id(state: GameStateSnapshot, object_id: str) -> dict[str, Any] | None:
    for player in state.get("players", []):
        for zone_name in ("Battlefield", "Hand", "Graveyard"):
            for card in player.get(zone_name) or []:
                if card.get("ID") == object_id:
                    return cast(dict[str, Any], card)
    return None


def _option_label(pending: PendingState, idx: int) -> str:
    options = pending.get("options", [])
    if not 0 <= idx < len(options):
        return str(idx)
    option = options[idx]
    return option.get("card_name") or option.get("label") or option.get("id") or str(idx)


def _option_ids(option: PendingOptionState) -> set[str]:
    values = {
        option.get("id", ""),
        option.get("card_id", ""),
        option.get("permanent_id", ""),
    }
    return {value for value in values if value}


if __name__ == "__main__":
    main()
