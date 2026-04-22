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

from magic_ai.actions import ActionRequest  # noqa: E402
from magic_ai.game_state import (  # noqa: E402
    GameStateEncoder,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
)
from magic_ai.model import PPOPolicy  # noqa: E402
from magic_ai.ppo import (  # noqa: E402
    PPOStats,
    RolloutStep,
    merge_pending_into_state,
    ppo_update,
    rollout_step_from_policy,
    terminal_returns,
)

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
    episode_idx: int
    episode_steps: list[RolloutStep]
    episode_transcript: list[TranscriptAction]
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
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        rollout_capacity=rollout_capacity,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    if args.checkpoint and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["policy"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    mage = importlib.import_module("mage")
    deck_a, deck_b = load_decks(args.deck_json)
    if args.num_envs > 1:
        train_batched_envs(args, mage, deck_a, deck_b, policy, optimizer)
        save_checkpoint(args.output, policy, optimizer, args)
        print(f"saved checkpoint -> {args.output}")
        return

    pending_steps: list[RolloutStep] = []
    pending_returns: list[torch.Tensor] = []
    last_sample_transcript: list[TranscriptAction] = []
    completed_games = 0
    win_stats = WinFractionStats()

    for episode_idx in range(args.episodes):
        seed = args.seed + episode_idx
        game = mage.new_game(
            deck_a,
            deck_b,
            name_a=args.name_a,
            name_b=args.name_b,
            seed=seed,
            shuffle=not args.no_shuffle,
            hand_size=args.hand_size,
        )
        episode_steps: list[RolloutStep] = []
        episode_transcript: list[TranscriptAction] = []
        winner = ""
        try:
            for _ in range(args.max_steps_per_game):
                pending = game.pending or game.legal()
                if game.is_over:
                    winner = str(game.winner)
                    break
                if pending is None:
                    game.refresh_state()
                    continue

                state = merge_pending_into_state(game.state, pending)
                action, step = rollout_step_from_policy(
                    policy,
                    state,
                    cast(Any, pending),
                    deterministic=args.deterministic_rollout,
                )
                episode_steps.append(step)
                episode_transcript.append(
                    TranscriptAction(
                        state=state,
                        pending=cast(PendingState, pending),
                        action=action,
                    )
                )
                game.step(cast(dict[Any, Any], action))

            if game.is_over:
                winner = str(game.winner)
            elif not winner:
                winner = ""
        finally:
            game.close()

        if not episode_steps:
            continue

        returns = terminal_returns(episode_steps, winner=winner, gamma=args.gamma)
        pending_steps.extend(episode_steps)
        pending_returns.append(returns)
        last_sample_transcript = episode_transcript
        completed_games += 1
        win_stats.record(
            classify_winner(
                cast(GameStateSnapshot, game.state),
                winner,
                args.name_a,
                args.name_b,
            )
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
            print_sample_game(
                episode_transcript,
                winner=winner,
                max_actions=args.sample_actions,
            )
            print(
                "update",
                f"episode={episode_idx + 1}",
                f"games={completed_games}",
                f"steps={len(pending_steps)}",
                f"winner={winner or 'draw'}",
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
            policy.release_replay_rows(
                [step.replay_idx for step in pending_steps if step.replay_idx is not None]
            )
            pending_steps.clear()
            pending_returns.clear()
            win_stats.reset()

        if args.save_every and (episode_idx + 1) % args.save_every == 0:
            save_checkpoint(args.output, policy, optimizer, args)

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
        print_sample_game(
            last_sample_transcript,
            winner="",
            max_actions=args.sample_actions,
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


def classify_winner(
    state: GameStateSnapshot,
    winner: str,
    name_a: str,
    name_b: str,
) -> str:
    if not winner:
        return "draw"

    players = state.get("players", [])
    if players:
        player_a = players[0]
        player_b = players[1] if len(players) > 1 else {}
        p1_ids = {str(player_a.get("ID", "")), str(player_a.get("Name", "")), name_a}
        p2_ids = {str(player_b.get("ID", "")), str(player_b.get("Name", "")), name_b}
        if winner in p1_ids:
            return "p1"
        if winner in p2_ids:
            return "p2"

    if winner == name_a:
        return "p1"
    if winner == name_b:
        return "p2"
    return "draw"


def train_batched_envs(
    args: argparse.Namespace,
    mage: Any,
    deck_a: dict[str, Any],
    deck_b: dict[str, Any],
    policy: PPOPolicy,
    optimizer: torch.optim.Optimizer,
) -> None:
    pending_steps: list[RolloutStep] = []
    pending_returns: list[torch.Tensor] = []
    last_sample_transcript: list[TranscriptAction] = []
    completed_games = 0
    last_saved_games = 0
    next_episode_idx = 0
    live_games: list[LiveGame] = []
    win_stats = WinFractionStats()

    def start_game(episode_idx: int) -> LiveGame:
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
            episode_idx=episode_idx,
            episode_steps=[],
            episode_transcript=[],
        )

    def maybe_start_games() -> None:
        nonlocal next_episode_idx
        while len(live_games) < args.num_envs and next_episode_idx < args.episodes:
            live_games.append(start_game(next_episode_idx))
            next_episode_idx += 1

    def finish_game(env: LiveGame, winner: str) -> None:
        nonlocal completed_games, last_sample_transcript
        win_stats.record(
            classify_winner(
                cast(GameStateSnapshot, env.game.state),
                winner,
                args.name_a,
                args.name_b,
            )
        )
        env.game.close()
        if env.episode_steps:
            returns = terminal_returns(env.episode_steps, winner=winner, gamma=args.gamma)
            pending_steps.extend(env.episode_steps)
            pending_returns.append(returns)
            last_sample_transcript = env.episode_transcript
        completed_games += 1

    maybe_start_games()
    while live_games:
        ready: list[tuple[LiveGame, GameStateSnapshot, PendingState]] = []
        remaining_games: list[LiveGame] = []
        for env in live_games:
            pending = env.game.pending or env.game.legal()
            if env.game.is_over:
                finish_game(env, str(env.game.winner))
                continue
            if env.action_count >= args.max_steps_per_game:
                finish_game(env, "")
                continue
            if pending is None:
                env.game.refresh_state()
                remaining_games.append(env)
                continue
            state = merge_pending_into_state(env.game.state, pending)
            ready.append((env, state, cast(PendingState, pending)))
            remaining_games.append(env)

        live_games = remaining_games
        if ready:
            cached_inputs = [
                policy.parse_inputs(
                    state,
                    pending,
                    perspective_player_idx=int(pending.get("player_idx", 0)),
                )
                for _env, state, pending in ready
            ]
            with torch.no_grad():
                policy_steps = policy.act_batch(
                    cached_inputs,
                    deterministic=args.deterministic_rollout,
                )
            for (env, state, pending), policy_step in zip(ready, policy_steps, strict=True):
                player_idx = int(pending.get("player_idx", 0))
                player = state["players"][player_idx]
                env.episode_steps.append(
                    RolloutStep(
                        state=state,
                        pending=pending,
                        perspective_player_idx=player_idx,
                        player_id=player.get("ID", ""),
                        player_name=player.get("Name", ""),
                        trace=policy_step.trace,
                        old_log_prob=float(policy_step.log_prob.detach().cpu()),
                        value=float(policy_step.value.detach().cpu()),
                        replay_idx=policy_step.replay_idx,
                    )
                )
                env.episode_transcript.append(
                    TranscriptAction(
                        state=state,
                        pending=pending,
                        action=policy_step.action,
                    )
                )
                env.action_count += 1
                env.game.step(cast(dict[Any, Any], policy_step.action))

        still_live: list[LiveGame] = []
        for env in live_games:
            if env.game.is_over:
                finish_game(env, str(env.game.winner))
            else:
                still_live.append(env)
        live_games = still_live

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
            print_sample_game(
                last_sample_transcript,
                winner="",
                max_actions=args.sample_actions,
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
            policy.release_replay_rows(
                [step.replay_idx for step in pending_steps if step.replay_idx is not None]
            )
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
        print_sample_game(
            last_sample_transcript,
            winner="",
            max_actions=args.sample_actions,
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
        policy.release_replay_rows(
            [step.replay_idx for step in pending_steps if step.replay_idx is not None]
        )


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
    winner: str,
    max_actions: int,
) -> None:
    print("sample_game")
    if not transcript:
        print("(no actions)")
        return

    current_turn: int | None = None
    for idx, item in enumerate(transcript[:max_actions]):
        turn = max(0, int(item.state.get("turn", 1)) - 1)
        if turn != current_turn:
            current_turn = turn
            print(f"== TURN {turn} ==")
        player_idx = int(item.pending.get("player_idx", 0))
        print(f"Player{player_idx + 1}: {describe_action(item)}")

    remaining = len(transcript) - max_actions
    if remaining > 0:
        print(f"... {remaining} more actions")
    if winner:
        print(f"winner: {winner}")


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
        name = _card_name_for_id(pending, action.get("permanent_id", ""))
        ability_index = int(action.get("ability_index", 0))
        return _with_targets(
            f"activate {name} ability {ability_index}",
            item,
            action.get("targets", []),
        )

    if "attackers" in action:
        attackers = [_card_name_for_id(pending, attacker_id) for attacker_id in action["attackers"]]
        if not attackers:
            return "attack with no creatures"
        return "attack with " + ", ".join(attackers)

    if "blockers" in action:
        assignments = []
        for assignment in action["blockers"]:
            blocker = _card_label_for_id(item, assignment.get("blocker", ""))
            attacker = _target_label_for_id(item, assignment.get("attacker", ""))
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
            return f"Player{player_idx + 1}"
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
        return object_id

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
