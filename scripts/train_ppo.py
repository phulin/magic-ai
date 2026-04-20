#!/usr/bin/env python3
"""Train a PPO self-play policy against the mage-go Python engine."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.game_state import GameStateEncoder
from magic_ai.model import PPOPolicy
from magic_ai.ppo import (
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


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    device = torch.device(args.device)
    game_state_encoder = GameStateEncoder.from_embedding_json(args.embeddings, d_model=args.d_model)
    policy = PPOPolicy(
        game_state_encoder,
        hidden_dim=args.hidden_dim,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    if args.checkpoint and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["policy"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    mage = importlib.import_module("mage")
    deck_a, deck_b = load_decks(args.deck_json)
    pending_steps: list[RolloutStep] = []
    pending_returns: list[torch.Tensor] = []
    completed_games = 0

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
        completed_games += 1

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
            pending_steps.clear()
            pending_returns.clear()

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

    save_checkpoint(args.output, policy, optimizer, args)
    print(f"saved checkpoint -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO self-play training with mage-go.")
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ppo.pt"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--deck-json", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=512)
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
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
