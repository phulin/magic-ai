#!/usr/bin/env python3
"""Train a PPO self-play policy against the mage-go Python engine."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from dotenv import load_dotenv

load_dotenv()

import wandb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.buffer import NativeTrajectoryBuffer  # noqa: E402
from magic_ai.game_state import GameStateEncoder  # noqa: E402
from magic_ai.model import PPOPolicy  # noqa: E402
from magic_ai.native_encoder import NativeBatchEncoder  # noqa: E402
from magic_ai.native_rollout import (  # noqa: E402
    NativeRolloutDriver,
    NativeRolloutUnavailable,
)
from magic_ai.ppo import PPOStats, RolloutStep, ppo_update, terminal_returns  # noqa: E402

DEFAULT_DECK = {
    "name": "bolt-mountain",
    "cards": [
        {"name": "Mountain", "count": 24},
        {"name": "Lightning Bolt", "count": 36},
    ],
}


@dataclass
class LiveGame:
    game: Any
    slot_idx: int
    episode_idx: int
    episode_steps: list[RolloutStep]
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
    deck_a, deck_b = load_decks(args.deck_json)
    validate_deck_embeddings(args.embeddings, deck_a, deck_b)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    device = torch.device(args.device)
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
    ).to(device)
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

    train_native_batched_envs(
        args,
        mage,
        deck_a,
        deck_b,
        policy,
        native_encoder,
        optimizer,
        native_rollout,
        staging_buffer,
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
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=512)
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--max-options", type=int, default=64)
    parser.add_argument("--max-targets-per-option", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--sample-actions",
        type=int,
        default=80,
        help="maximum actions to print from a sample rollout game at each PPO update",
    )
    parser.add_argument("--wandb-project", default="magic-ai")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
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
    if args.hidden_layers < 1:
        raise ValueError("--hidden-layers must be at least 1")


def log_ppo_stats(
    stats: PPOStats,
    *,
    games: int,
    steps: int,
    win_stats: WinFractionStats | None = None,
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
        "games": games,
        "rollout_steps": steps,
    }
    if win_stats is not None:
        payload.update(win_stats.as_wandb_metrics())
    wandb.log(payload)


def train_native_batched_envs(
    args: argparse.Namespace,
    mage: Any,
    deck_a: dict[str, Any],
    deck_b: dict[str, Any],
    policy: PPOPolicy,
    native_encoder: NativeBatchEncoder,
    optimizer: torch.optim.Optimizer,
    native_rollout: NativeRolloutDriver,
    staging_buffer: NativeTrajectoryBuffer,
) -> None:
    if not native_encoder.is_available:
        raise SystemExit("native rollout requires MageEncodeBatch")

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
        return LiveGame(
            game=mage.new_game(
                deck_a,
                deck_b,
                name_a=args.name_a,
                name_b=args.name_b,
                seed=args.seed + episode_idx,
                shuffle=not args.no_shuffle,
                hand_size=args.hand_size,
            ),
            slot_idx=slot_idx,
            episode_idx=episode_idx,
            episode_steps=[],
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
                returns = terminal_returns(
                    env.episode_steps,
                    winner_idx=winner_idx,
                    gamma=args.gamma,
                )
                pending_steps.extend(env.episode_steps)
                pending_returns.append(returns)
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
            parsed_batch = native_encoder.encode_handles(
                [env.game for env in ready_envs],
                perspective_player_indices=ready_players,
            )
            with torch.no_grad():
                policy_steps = policy.sample_native_batch(
                    parsed_batch,
                    deterministic=args.deterministic_rollout,
                )
            log_probs = torch.stack([policy_step.log_prob for policy_step in policy_steps])
            values = torch.stack([policy_step.value for policy_step in policy_steps])

            starts: list[int] = []
            counts: list[int] = []
            selected_cols: list[int] = []
            may_selected: list[int] = []
            selected_choice_cols: list[tuple[int, ...]] = []
            cursor = 0
            for env, player_idx, policy_step in zip(
                ready_envs, ready_players, policy_steps, strict=True
            ):
                cols = list(policy_step.selected_choice_cols)
                starts.append(cursor)
                counts.append(len(cols))
                selected_cols.extend(cols)
                may_selected.append(policy_step.may_selected)
                selected_choice_cols.append(tuple(policy_step.selected_choice_cols))
                cursor += len(cols)
                env.action_count += 1

            staging_buffer.stage_batch(
                [env.slot_idx for env in ready_envs],
                parsed_batch,
                selected_choice_cols=selected_choice_cols,
                may_selected=may_selected,
                old_log_probs=log_probs,
                values=values,
                perspective_player_indices=ready_players,
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
            stats = ppo_update(
                policy,
                optimizer,
                pending_steps,
                torch.cat(pending_returns),
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_epsilon=args.clip_epsilon,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
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

        maybe_start_games()

    if pending_steps:
        stats = ppo_update(
            policy,
            optimizer,
            pending_steps,
            torch.cat(pending_returns),
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
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
        log_ppo_stats(stats, games=completed_games, steps=len(pending_steps), win_stats=win_stats)
        policy.reset_rollout_buffer()


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
    deck_a: dict[str, Any],
    deck_b: dict[str, Any],
) -> None:
    embedded_names = load_embedded_card_names(embeddings_path)
    missing: dict[str, dict[str, int]] = {}
    for label, deck in (("player_a", deck_a), ("player_b", deck_b)):
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


if __name__ == "__main__":
    main()
