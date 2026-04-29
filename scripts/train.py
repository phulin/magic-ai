#!/usr/bin/env python3
"""Train a PPO self-play policy against the mage-go Python engine."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import itertools
import json
import random
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from dotenv import load_dotenv

load_dotenv()

import wandb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.actions import (  # noqa: E402
    OPTION_SCALAR_DIM,
    TARGET_SCALAR_DIM,
    ActionRequest,
)
from magic_ai.buffer import NativeTrajectoryBuffer  # noqa: E402
from magic_ai.game_state import (  # noqa: E402
    GAME_INFO_DIM,
    ZONE_SLOT_COUNT,
    GameStateEncoder,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
)
from magic_ai.model import PPOPolicy  # noqa: E402
from magic_ai.native_encoder import NativeBatchEncoder, NativeEncodingError  # noqa: E402,F401
from magic_ai.native_rollout import (  # noqa: E402
    NativeRolloutDriver,  # noqa: F401
    NativeRolloutUnavailable,
)
from magic_ai.opponent_pool import (  # noqa: E402
    OpponentEntry,
    OpponentPool,
    SnapshotSchedule,
    build_opponent_policy,
    distribute_games_by_recency,
    opponent_policy_state_dict,
    run_eval_matches,
    save_snapshot,
    snapshot_tag,
)
from magic_ai.ppo import PPOStats, RolloutStep, gae_returns, ppo_update  # noqa: E402
from magic_ai.rnad import RNaDConfig  # noqa: E402
from magic_ai.rnad_trainer import (  # noqa: E402
    EpisodeBatch,
    RNaDTrainerState,
    build_trainer_state,
    run_rnad_update,
)
from magic_ai.sharded_native import (  # noqa: E402
    ShardedNativeBatchEncoder,
    ShardedNativeRolloutDriver,
)
from magic_ai.text_encoder.actor_critic import (  # noqa: E402
    TextActorCritic,  # noqa: E402
    TextDecisionLayout,
    build_text_decision_layout,
    infer_text_trace_kind,
)
from magic_ai.text_encoder.assembler import (  # noqa: E402
    AssemblerTokens,
    assemble_batch,
    build_assembler_tokens,
)
from magic_ai.text_encoder.card_cache import (  # noqa: E402
    DEFAULT_ORACLE_DB_PATH,
    CardTokenCache,
    build_card_cache,
    fetch_registered_card_names_from_engine,
    load_card_cache,
    load_oracle_db,
)
from magic_ai.text_encoder.model import TextEncoderConfig  # noqa: E402
from magic_ai.text_encoder.recurrent import RecurrentTextPolicyConfig  # noqa: E402
from magic_ai.text_encoder.render import OracleEntry  # noqa: E402
from magic_ai.text_encoder.render_plan import emit_render_plan  # noqa: E402
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer  # noqa: E402
from magic_ai.text_encoder.tokenizer import (  # noqa: E402
    MODERNBERT_REPO,
    MODERNBERT_REVISION,
    TOKENIZER_DIR,
    load_tokenizer,
)

DEFAULT_DECK = {
    "name": "bolt-mountain",
    "cards": [
        {"name": "Mountain", "count": 24},
        {"name": "Lightning Bolt", "count": 36},
    ],
}


# Optional subphase profiling hook. The profile script binds this to a
# real recorder; in normal runs it stays a no-op (one Python attribute
# read + branch per call site, negligible).
_subphase_record: Callable[[str, float], None] | None = None


def _record_phase(name: str, t0: float) -> None:
    rec = _subphase_record
    if rec is not None:
        rec(name, time.perf_counter() - t0)


@dataclass(frozen=True)
class TranscriptAction:
    state: GameStateSnapshot
    pending: PendingState
    action: ActionRequest


@dataclass
class LiveGame:
    game: Any
    slot_idx: int
    episode_idx: int
    episode_steps: list[RolloutStep]
    transcript: list[TranscriptAction]
    transcript_enabled: bool = False
    action_count: int = 0


@dataclass(frozen=True)
class TrainingResumeState:
    completed_games: int = 0
    last_saved_games: int = 0
    total_rollout_steps: int = 0
    total_generated_rollout_steps: int = 0
    total_wandb_logs: int = 0


@dataclass
class RetrospectiveLogSchedule:
    """Fires wandb retrospective-rating logs at fixed run-percent horizons."""

    total_episodes: int
    thresholds: list[int]
    horizon_pcts: list[int]
    next_idx: int = 0

    @classmethod
    def build(cls, total_episodes: int, *, pct_step: int = 5) -> RetrospectiveLogSchedule:
        if total_episodes < 1:
            raise ValueError("total_episodes must be at least 1")
        if pct_step < 1:
            raise ValueError("pct_step must be at least 1")

        thresholds: list[int] = []
        horizon_pcts: list[int] = []
        for pct in range(pct_step, 101, pct_step):
            threshold = max(1, int(round(total_episodes * pct / 100.0)))
            if thresholds and threshold <= thresholds[-1]:
                threshold = thresholds[-1] + 1
            if threshold > total_episodes:
                break
            thresholds.append(threshold)
            horizon_pcts.append(pct)
        return cls(
            total_episodes=total_episodes,
            thresholds=thresholds,
            horizon_pcts=horizon_pcts,
        )

    def fire(self, completed_games: int) -> list[tuple[int, int]]:
        """Return ``(horizon_pct, threshold_games)`` crossed since the last call."""
        fired: list[tuple[int, int]] = []
        while (
            self.next_idx < len(self.thresholds)
            and completed_games >= self.thresholds[self.next_idx]
        ):
            fired.append((self.horizon_pcts[self.next_idx], self.thresholds[self.next_idx]))
            self.next_idx += 1
        return fired


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


@dataclass
class SlotTrainingBackend:
    policy: PPOPolicy
    native_encoder: ShardedNativeBatchEncoder
    staging_buffer: NativeTrajectoryBuffer
    batch_pool: ThreadPoolExecutor | None
    batch_workers: int


@dataclass
class TextTrainingBackend:
    policy: TextActorCritic
    replay_buffer: TextReplayBuffer
    cache: CardTokenCache
    oracle: dict[str, OracleEntry]
    tokenizer: Any
    # Pre-resolved assembler structural-token table (and its lazily-attached
    # status prefixes + per-cache body_lists memo). Built once at backend
    # init and threaded through every assemble_batch call so we don't
    # re-encode the static fragment table or re-derive body_lists per call.
    assembler_tokens: AssemblerTokens
    native_encoder: ShardedNativeBatchEncoder | None = None
    batch_pool: ThreadPoolExecutor | None = None
    batch_workers: int = 1


@dataclass
class TrainingRunResult:
    resume_state: TrainingResumeState
    rnad_state: RNaDTrainerState | None
    opponent_pool: OpponentPool | None = None
    snapshot_schedule: SnapshotSchedule | None = None
    retrospective_schedule: RetrospectiveLogSchedule | None = None


def build_slot_backend(args: argparse.Namespace, device: torch.device) -> SlotTrainingBackend:
    game_state_encoder = GameStateEncoder.from_embedding_json(args.embeddings, d_model=args.d_model)
    rollout_capacity = args.rollout_buffer_capacity or max(
        4096, args.rollout_steps + args.max_steps_per_game * args.num_envs
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
        spr_k=args.spr_k,
        spr_proj_dim=args.spr_proj_dim,
        validate=not args.no_validate,
        compile_forward=args.torch_compile,
    ).to(device)
    policy.init_lstm_env_states(args.num_envs)
    if device.type == "cuda":
        # Force cuBLAS handle creation before rollout ingestion creates temporary
        # CUDA copy tensors that PyTorch may hold in its caching allocator.
        _ = torch.empty((1, 1), device=device) @ torch.empty((1, 1), device=device)

    batch_workers = max(1, getattr(args, "batch_workers", 1))
    batch_pool = (
        ThreadPoolExecutor(max_workers=batch_workers, thread_name_prefix="mage-batch")
        if batch_workers > 1
        else None
    )
    native_encoder = ShardedNativeBatchEncoder.for_policy(
        policy, workers=batch_workers, pool=batch_pool
    )
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
    return SlotTrainingBackend(
        policy=policy,
        native_encoder=native_encoder,
        staging_buffer=staging_buffer,
        batch_pool=batch_pool,
        batch_workers=batch_workers,
    )


def build_text_backend(args: argparse.Namespace, device: torch.device) -> TextTrainingBackend:
    tokenizer = load_tokenizer()
    cache_path = Path(args.card_token_cache)
    oracle_db_path = Path(getattr(args, "oracle_db", DEFAULT_ORACLE_DB_PATH))
    registered_names = fetch_registered_card_names_from_engine()
    oracle = load_oracle_db(oracle_db_path, names=registered_names)
    if cache_path.exists():
        cache = load_card_cache(cache_path)
    else:
        cache = build_card_cache(
            registered_names,
            oracle,
            tokenizer,
            oracle_db_path=oracle_db_path,
            missing_policy="warn",
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("text tokenizer must define pad_token_id")
    cfg = TextEncoderConfig(
        vocab_size=len(tokenizer),
        pad_id=int(pad_id),
        d_model=args.text_d_model,
        n_layers=args.text_layers,
        n_heads=args.text_heads,
        d_ff=args.text_d_ff,
        max_seq_len=args.text_max_tokens,
    )
    recurrent_cfg = RecurrentTextPolicyConfig(
        encoder=cfg,
        lstm_hidden=args.text_d_model,
        lstm_layers=args.hidden_layers,
    )
    policy = TextActorCritic(recurrent_cfg).to(device)
    policy.init_lstm_env_states(args.num_envs)
    rollout_capacity = args.rollout_buffer_capacity or max(
        4096, args.rollout_steps + args.max_steps_per_game * args.num_envs
    )
    replay_buffer = TextReplayBuffer(
        capacity=rollout_capacity,
        max_tokens=args.text_max_tokens,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        max_decision_groups=args.max_decision_groups,
        max_cached_choices=args.max_cached_choices,
        recurrent_layers=args.hidden_layers,
        recurrent_hidden_dim=args.text_d_model,
        device=device,
    )
    policy.rollout_buffer = replay_buffer
    batch_workers = max(1, getattr(args, "batch_workers", 1))
    batch_pool = (
        ThreadPoolExecutor(max_workers=batch_workers, thread_name_prefix="mage-text-batch")
        if batch_workers > 1
        else None
    )
    native_encoder = None
    if getattr(args, "native_render_plan", False):
        native_encoder = ShardedNativeBatchEncoder.for_text(
            max_options=args.max_options,
            max_targets_per_option=args.max_targets_per_option,
            max_cached_choices=replay_buffer.max_cached_choices,
            zone_slot_count=ZONE_SLOT_COUNT,
            game_info_dim=GAME_INFO_DIM,
            option_scalar_dim=OPTION_SCALAR_DIM,
            target_scalar_dim=TARGET_SCALAR_DIM,
            card_name_to_row=_build_text_card_name_to_row(cache),
            emit_render_plan=True,
            render_plan_capacity=args.render_plan_capacity,
            validate=not getattr(args, "no_validate", False),
            workers=batch_workers,
            pool=batch_pool,
            dedup_card_bodies=bool(getattr(args, "card_body_dedup", False)),
        )
    backend = TextTrainingBackend(
        policy=policy,
        replay_buffer=replay_buffer,
        cache=cache,
        oracle=oracle,
        tokenizer=tokenizer,
        assembler_tokens=build_assembler_tokens(tokenizer),
        native_encoder=native_encoder,
        batch_pool=batch_pool,
        batch_workers=batch_workers,
    )
    # Phase 5: register the closed-vocabulary token tables with the
    # mage-go side so the rollout hot path can dispatch through
    # MageEncodeTokens. Falls back gracefully if the native lib doesn't
    # have MageRegisterTokenTables (older libmage.so), with a clear
    # warning so users know to rebuild.
    if getattr(args, "text_native_assembler", True):
        try:
            from magic_ai.text_encoder.native_token_tables import (
                register_native_token_tables,
            )
            from magic_ai.text_encoder.token_tables import build_token_tables

            tables = build_token_tables(tokenizer, cache)
            register_native_token_tables(tables)
            print("native token assembler registered (MageEncodeTokens path)")
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(
                f"warning: MageRegisterTokenTables unavailable ({exc}); "
                "falling back to Python assembler. Rebuild libmage.so to enable."
            )
            args.text_native_assembler = False
    return backend


def sample_text_policy_batch(
    args: argparse.Namespace,
    backend: TextTrainingBackend,
    snapshots: list[GameStateSnapshot],
    pendings: list[PendingState],
    *,
    env_indices: list[int],
    perspective_player_indices: list[int],
    deterministic: bool = False,
) -> list[Any]:
    """Emit/assemble text batches and sample policy actions with replay rows."""

    n = len(snapshots)
    if n == 0:
        return []
    if len(pendings) != n or len(env_indices) != n or len(perspective_player_indices) != n:
        raise ValueError("snapshots, pendings, env_indices, and players differ")
    if getattr(args, "native_render_plan", False):
        raise NotImplementedError("native render-plan emission is not wired into train.py yet")

    card_row_lookup = _build_text_card_row_lookup(backend.cache)

    def tokenize(text: str) -> list[int]:
        return list(backend.tokenizer.encode(text, add_special_tokens=False))

    plans = []
    layouts = []
    max_cached_choices = backend.replay_buffer.max_cached_choices
    for snapshot, pending in zip(snapshots, pendings, strict=True):
        snapshot_for_render = copy.deepcopy(snapshot)
        snapshot_for_render["pending"] = pending
        options = list(pending.get("options", []) or [])
        plans.append(
            emit_render_plan(
                snapshot_for_render,
                options,
                card_row_lookup=card_row_lookup,
                tokenize=tokenize,
                oracle=backend.oracle,
            )
        )
        trace_kind = infer_text_trace_kind(pending)
        layouts.append(
            build_text_decision_layout(
                trace_kind,
                pending,
                max_options=args.max_options,
                max_targets_per_option=args.max_targets_per_option,
                max_cached_choices=max_cached_choices,
            )
        )

    encoded = assemble_batch(
        plans,
        backend.cache,
        backend.tokenizer,
        max_tokens=args.text_max_tokens,
        on_overflow="truncate",
        assembler_tokens=backend.assembler_tokens,
    )
    return backend.policy.sample_text_batch(
        encoded,
        env_indices=env_indices,
        perspective_player_indices=perspective_player_indices,
        layouts=layouts,
        deterministic=deterministic,
    )


def _build_text_card_row_lookup(cache: CardTokenCache) -> Callable[[str], int]:
    name_to_row = _build_text_card_name_to_row(cache)

    def lookup(name: str) -> int:
        return name_to_row.get(name, 0)

    return lookup


def _build_text_card_name_to_row(cache: CardTokenCache) -> dict[str, int]:
    name_to_row: dict[str, int] = {}
    for idx, name in enumerate(cache.row_to_name):
        if idx == 0 or not name:
            continue
        name_to_row.setdefault(name, idx)
    return name_to_row


def _current_transcript_snapshot(game: Any) -> tuple[GameStateSnapshot, PendingState]:
    # Native batch rollout advances the engine without updating the Python
    # wrapper's cached state, so refresh before every transcript snapshot.
    game.refresh_state()
    pending = cast(PendingState | None, game.pending or game.legal())
    state = cast(GameStateSnapshot, copy.deepcopy(game.state))
    if pending is None:
        raise RuntimeError("live game is missing a pending action for transcript capture")
    return state, copy.deepcopy(pending)


def _wandb_summary_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_wandb_summary_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _wandb_summary_value(item) for key, item in value.items()}
    return str(value)


def log_args_to_wandb_summary(args: argparse.Namespace, run: Any | None = None) -> None:
    active_run = wandb.run if run is None else run
    if active_run is None:
        return
    for key, value in vars(args).items():
        active_run.summary[f"args/{key}"] = _wandb_summary_value(value)


def load_training_checkpoint(
    path: Path | None,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return cast(
        dict[str, Any],
        torch.load(path, map_location=map_location, weights_only=False),
    )


def _checkpoint_metadata(checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    metadata = checkpoint.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _training_state_dict(checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    training_state = checkpoint.get("training_state", {})
    return training_state if isinstance(training_state, dict) else {}


def _checkpoint_wandb_run_id(checkpoint: dict[str, Any] | None) -> str | None:
    run_id = _checkpoint_metadata(checkpoint).get("wandb_run_id")
    return str(run_id) if isinstance(run_id, str) and run_id else None


def checkpoint_encoder_kind(checkpoint: dict[str, Any] | None) -> str:
    """Return serialized encoder kind, defaulting legacy checkpoints to slots."""

    encoder = _checkpoint_metadata(checkpoint).get("encoder")
    if encoder is None:
        return "slots"
    if isinstance(encoder, str) and encoder:
        return encoder
    raise ValueError("checkpoint metadata field 'encoder' must be a non-empty string")


def validate_checkpoint_encoder(
    args: argparse.Namespace, checkpoint: dict[str, Any] | None
) -> None:
    if checkpoint is None:
        return
    checkpoint_encoder = checkpoint_encoder_kind(checkpoint)
    if checkpoint_encoder != args.encoder:
        raise ValueError(
            f"checkpoint encoder '{checkpoint_encoder}' is incompatible with "
            f"--encoder {args.encoder}"
        )
    if checkpoint_encoder != "text":
        return
    metadata = _checkpoint_metadata(checkpoint)
    text_config = metadata.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("text checkpoint metadata is missing text_config")
    for key in (
        "text_max_tokens",
        "text_d_model",
        "text_layers",
        "text_heads",
        "text_d_ff",
        "hidden_layers",
        "max_options",
        "max_targets_per_option",
    ):
        saved = text_config.get(key)
        requested = getattr(args, key, None)
        if saved != requested:
            raise ValueError(
                f"text checkpoint {key}={saved!r} is incompatible with "
                f"--{key.replace('_', '-')} {requested!r}"
            )


def _default_run_artifact_dir(output_path: Path, run_id: str | None) -> Path:
    label = run_id or output_path.stem
    return output_path.parent / "runs" / label


def _resolve_run_artifact_dir(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    wandb_run_id: str | None,
) -> Path:
    metadata = _checkpoint_metadata(checkpoint)
    saved_dir = metadata.get("run_artifact_dir")
    if isinstance(saved_dir, str) and saved_dir:
        return Path(saved_dir)
    return _default_run_artifact_dir(args.output, wandb_run_id)


def _restore_opponent_pool(
    checkpoint: dict[str, Any] | None,
    snapshot_dir: Path,
) -> OpponentPool:
    training_state = _training_state_dict(checkpoint)
    pool_state = training_state.get("opponent_pool")
    if isinstance(pool_state, dict):
        pool = OpponentPool.from_state_dict(pool_state)
    else:
        pool = OpponentPool()

    known_paths = {entry.path.resolve() for entry in pool.entries if entry.path}
    if snapshot_dir.exists():
        for snapshot_path in sorted(snapshot_dir.glob("snapshot_*.pt")):
            resolved = snapshot_path.resolve()
            if resolved in known_paths:
                continue
            tag = snapshot_path.stem.removeprefix("snapshot_")
            pool.add_snapshot(snapshot_path, tag)
            known_paths.add(resolved)
    return pool


def _resume_state_from_checkpoint(checkpoint: dict[str, Any] | None) -> TrainingResumeState:
    training_state = _training_state_dict(checkpoint)
    return TrainingResumeState(
        completed_games=int(training_state.get("completed_games", 0)),
        last_saved_games=int(training_state.get("last_saved_games", 0)),
        total_rollout_steps=int(training_state.get("total_rollout_steps", 0)),
        total_generated_rollout_steps=int(training_state.get("total_generated_rollout_steps", 0)),
        total_wandb_logs=int(training_state.get("total_wandb_logs", 0)),
    )


def _restore_rnad_state(
    state: RNaDTrainerState,
    checkpoint: dict[str, Any] | None,
) -> None:
    """Pull R-NaD outer-loop state out of a resumed PPO checkpoint.

    No-op when ``checkpoint`` is missing the ``rnad_state`` key (e.g. a PPO
    checkpoint is being used to bootstrap an R-NaD run).

    Honors the serialized ``reg_snapshot_dir``: if it exists and differs
    from the current configured dir, the reg snapshots are loaded from the
    saved path (and ``state.reg_snapshot_dir`` is repointed so subsequent
    outer-iteration snapshots land in the same place). This covers the
    common case where checkpoints are copied between machines with
    different ``--output`` paths.
    """

    payload = _training_state_dict(checkpoint).get("rnad_state")
    if not isinstance(payload, dict):
        return
    total_wandb_logs = int(_training_state_dict(checkpoint).get("total_wandb_logs", 0))
    target_sd = payload.get("target")
    if isinstance(target_sd, dict):
        target_module = cast(PPOPolicy, state.target)
        target_module.load_state_dict(target_sd)
        for p in target_module.parameters():
            p.requires_grad_(False)
        target_module.eval()
    outer = int(payload.get("outer_iteration", 0))
    grad_step = int(payload.get("gradient_step", 0))
    finetuning = bool(payload.get("is_finetuning", False))

    saved_dir_raw = payload.get("reg_snapshot_dir")
    saved_dir = Path(saved_dir_raw) if isinstance(saved_dir_raw, str) and saved_dir_raw else None
    # Prefer the saved dir when it actually has the reg snapshot we need.
    if saved_dir is not None and (saved_dir / f"reg_m{outer:03d}.pt").exists():
        if saved_dir != state.reg_snapshot_dir:
            print(
                f"step={total_wandb_logs} "
                f"[rnad] resuming from saved reg_snapshot_dir {saved_dir!s} "
                f"(configured was {state.reg_snapshot_dir!s})",
                flush=True,
            )
            state.reg_snapshot_dir = saved_dir

    try:
        from magic_ai.rnad_trainer import resume_from_snapshot_dir

        resume_from_snapshot_dir(state, outer_iteration=outer, gradient_step=grad_step)
    except (FileNotFoundError, KeyError) as err:
        print(
            f"step={total_wandb_logs} "
            f"[rnad] failed to restore reg snapshots from "
            f"{state.reg_snapshot_dir!s}: {err}; continuing with in-memory regs",
            flush=True,
        )
        state.outer_iteration = outer
        state.gradient_step = grad_step
    state.is_finetuning = finetuning


def train_selected_backend(
    args: argparse.Namespace,
    mage: Any,
    deck_pool: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    checkpoint_cpu: dict[str, Any] | None,
    slot_backend: SlotTrainingBackend | None = None,
    text_backend: TextTrainingBackend | None = None,
    batch_pool: ThreadPoolExecutor | None = None,
) -> TrainingRunResult:
    """Dispatch the selected encoder backend and return checkpoint state."""

    if getattr(args, "encoder", "slots") == "text":
        if text_backend is None:
            raise ValueError("text_backend is required when --encoder text")
        if getattr(args, "native_render_plan", False):
            if text_backend.native_encoder is None:
                raise ValueError("text backend is missing native render-plan encoder")
            try:
                native_rollout = ShardedNativeRolloutDriver.for_mage(
                    mage, workers=text_backend.batch_workers, pool=batch_pool
                )
            except NativeRolloutUnavailable as exc:
                raise SystemExit(f"native rollout is unavailable: {exc}") from exc
            resume_state, rnad_state = train_text_native_batched_envs(
                args,
                mage,
                deck_pool,
                text_backend,
                optimizer,
                native_rollout,
                text_backend.native_encoder,
                resume_state=_resume_state_from_checkpoint(checkpoint_cpu),
                resume_checkpoint=checkpoint_cpu,
            )
        else:
            resume_state, rnad_state = train_text_envs(
                args,
                mage,
                deck_pool,
                text_backend,
                optimizer,
                resume_state=_resume_state_from_checkpoint(checkpoint_cpu),
            )
        return TrainingRunResult(resume_state=resume_state, rnad_state=rnad_state)

    if slot_backend is None:
        raise ValueError("slot_backend is required when --encoder slots")
    slot_policy = slot_backend.policy
    try:
        native_rollout = ShardedNativeRolloutDriver.for_mage(
            mage, workers=slot_backend.batch_workers, pool=batch_pool
        )
    except NativeRolloutUnavailable as exc:
        raise SystemExit(f"native rollout is unavailable: {exc}") from exc

    opponent_pool: OpponentPool | None = None
    snapshot_schedule: SnapshotSchedule | None = None
    retrospective_schedule: RetrospectiveLogSchedule | None = None
    opponent_policy: PPOPolicy | None = None
    if not args.disable_opponent_pool:
        opponent_pool = _restore_opponent_pool(checkpoint_cpu, args.opponent_pool_dir)
        snapshot_schedule = SnapshotSchedule.build(args.episodes)
        retrospective_schedule = RetrospectiveLogSchedule.build(args.episodes)
        training_state = _training_state_dict(checkpoint_cpu)
        snapshot_schedule.next_idx = min(
            int(training_state.get("snapshot_schedule_next_idx", 0)),
            len(snapshot_schedule.thresholds),
        )
        retrospective_schedule.next_idx = min(
            int(training_state.get("retrospective_schedule_next_idx", 0)),
            len(retrospective_schedule.thresholds),
        )
        opponent_policy = build_opponent_policy(slot_policy, device)

    resume_state, rnad_state = train_native_batched_envs(
        args,
        mage,
        deck_pool,
        slot_policy,
        slot_backend.native_encoder,
        optimizer,
        native_rollout,
        slot_backend.staging_buffer,
        opponent_pool=opponent_pool,
        snapshot_schedule=snapshot_schedule,
        retrospective_schedule=retrospective_schedule,
        opponent_policy=opponent_policy,
        resume_state=_resume_state_from_checkpoint(checkpoint_cpu),
        resume_checkpoint=checkpoint_cpu,
    )
    return TrainingRunResult(
        resume_state=resume_state,
        rnad_state=rnad_state,
        opponent_pool=opponent_pool,
        snapshot_schedule=snapshot_schedule,
        retrospective_schedule=retrospective_schedule,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.learning_rate is None:
        args.learning_rate = 5e-5 if args.trainer == "rnad" else 3e-4
    deck_pool = load_deck_pool(args.deck_json, args.deck_dir)
    validate_deck_embeddings(args.embeddings, deck_pool)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    checkpoint_cpu = load_training_checkpoint(args.checkpoint, map_location="cpu")
    validate_checkpoint_encoder(args, checkpoint_cpu)
    checkpoint_wandb_run_id = _checkpoint_wandb_run_id(checkpoint_cpu)

    if not args.no_wandb:
        init_kwargs: dict[str, Any] = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": vars(args),
        }
        if checkpoint_wandb_run_id is not None:
            init_kwargs["id"] = checkpoint_wandb_run_id
            init_kwargs["resume"] = "must"
        wandb.init(
            **init_kwargs,
        )

    active_wandb_run_id = wandb.run.id if wandb.run is not None else checkpoint_wandb_run_id
    run_artifact_dir = _resolve_run_artifact_dir(
        args=args,
        checkpoint=checkpoint_cpu,
        wandb_run_id=active_wandb_run_id,
    )
    args.opponent_pool_dir = run_artifact_dir / "opponent_pool"
    if not args.no_wandb:
        log_args_to_wandb_summary(args)

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    encoder_kind = getattr(args, "encoder", "slots")
    batch_pool: ThreadPoolExecutor | None = None
    slot_backend: SlotTrainingBackend | None = None
    text_backend: TextTrainingBackend | None = None
    if encoder_kind == "slots":
        slot_backend = build_slot_backend(args, device)
        policy = slot_backend.policy
        batch_pool = slot_backend.batch_pool
    else:
        text_backend = build_text_backend(args, device)
        policy = text_backend.policy
        batch_pool = getattr(text_backend, "batch_pool", None)
    # Paper §199: R-NaD uses Adam with b1=0.0 (no momentum). This is
    # load-bearing for stability: nonzero b1 lets policy updates accumulate
    # directional drift across batches, which combined with NeuRD's raw-logit
    # gradient and the [-beta, beta] gate causes logit saturation and
    # frozen policies. PPO keeps the standard b1=0.9 default.
    if args.trainer == "rnad":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(0.0, 0.999),
            eps=1e-8,
        )
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    checkpoint = load_training_checkpoint(args.checkpoint, map_location=device)
    if checkpoint is not None:
        policy.load_state_dict(checkpoint["policy"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    mage = importlib.import_module("mage")
    result = train_selected_backend(
        args,
        mage,
        deck_pool,
        optimizer,
        device=device,
        checkpoint_cpu=checkpoint_cpu,
        slot_backend=slot_backend,
        text_backend=text_backend,
        batch_pool=batch_pool,
    )

    save_checkpoint(
        args.output,
        policy,
        optimizer,
        args,
        opponent_pool=result.opponent_pool,
        snapshot_schedule=result.snapshot_schedule,
        retrospective_schedule=result.retrospective_schedule,
        resume_state=result.resume_state,
        wandb_run_id=active_wandb_run_id,
        run_artifact_dir=run_artifact_dir,
        rnad_state=result.rnad_state,
    )
    print(f"step={result.resume_state.total_wandb_logs} saved checkpoint -> {args.output}")
    wandb.finish()
    if batch_pool is not None:
        batch_pool.shutdown(wait=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO self-play training with mage-go.")
    parser.add_argument(
        "--encoder",
        choices=("slots", "text"),
        default="slots",
        help="state encoder backend to train; slots preserves the existing native slot path",
    )
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
    parser.add_argument(
        "--trainer",
        choices=("ppo", "rnad"),
        default="ppo",
        help="training algorithm: 'ppo' (default) or 'rnad' (Regularized Nash "
        "Dynamics / DeepNash). The 'rnad' path is scaffolded via CLI and "
        "primitives in magic_ai/rnad.py; the full training loop integration "
        "lands in phases 4-7 of docs/rnad_implementation_plan.md.",
    )
    parser.add_argument(
        "--rnad-eta",
        type=float,
        default=0.2,
        help="R-NaD reward-transform regularization strength (paper default 0.2)",
    )
    parser.add_argument(
        "--rnad-delta-m",
        type=int,
        default=1_000,
        help="R-NaD gradient steps per outer iteration. Paper §199 uses "
        "10k-100k on 768 TPU learners; this default is scaled for "
        "single-GPU rollout-batch cadence. Scale up proportionally with "
        "rollout-batch size if you have more compute.",
    )
    parser.add_argument(
        "--rnad-m",
        type=int,
        default=50,
        help="R-NaD number of outer fixed-point iterations (paper: ~200)",
    )
    parser.add_argument(
        "--rnad-neurd-beta",
        type=float,
        default=2.0,
        help="R-NaD NeuRD logit magnitude threshold",
    )
    parser.add_argument(
        "--rnad-neurd-clip",
        type=float,
        default=10_000.0,
        help="R-NaD NeuRD Q clip",
    )
    parser.add_argument(
        "--rnad-target-ema",
        type=float,
        default=0.005,
        help="R-NaD target-network Polyak averaging rate. Paired with the "
        "scaled --rnad-delta-m default so target tracks online inside one "
        "outer iter (delta_m * target_ema ~= 5). Paper uses 1e-3 with "
        "delta_m=10k-100k.",
    )
    parser.add_argument(
        "--rnad-finetune-eps",
        type=float,
        default=0.03,
        help="R-NaD fine-tune / test-time probability threshold",
    )
    parser.add_argument(
        "--rnad-finetune-ndisc",
        type=int,
        default=16,
        help="R-NaD fine-tune / test-time probability quanta",
    )
    parser.add_argument(
        "--rnad-q-corr-rho-bar",
        type=float,
        default=100.0,
        help="R-NaD full-NeuRD clip on the joint inverse sampling weight "
        "1/mu_t in the per-action Q estimator. Magic actions factor as "
        "mu_t = ∏_k mu_k so the unclipped weight can blow up "
        "multiplicatively in the number of decision groups.",
    )
    parser.add_argument(
        "--cuda-memory-snapshot",
        type=Path,
        default=None,
        help="Record CUDA allocator history and dump a snapshot to this path "
        "if the first training update OOMs. Load the resulting .pickle into "
        "https://pytorch.org/memory_viz to see every live allocation with the "
        "Python stack that produced it. Off by default (recording has overhead).",
    )
    parser.add_argument(
        "--rnad-bptt-chunk-size",
        type=int,
        default=200,
        help="R-NaD chunked-BPTT chunk length (DeepNash R-NaD §'Full games "
        "learning'): trajectories are split into chunks of this many steps; "
        "each chunk runs as one fused cuDNN nn.LSTM call with full BPTT "
        "inside the chunk, state detached at chunk boundaries. Default 200 "
        "matches --max-steps-per-game so the whole trace is one chunk.",
    )
    parser.add_argument("--episodes", type=int, default=65536)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument(
        "--rollout-buffer-capacity",
        type=int,
        default=None,
        help=(
            "rows in the rollout GPU buffer; default "
            "max(4096, rollout-steps + max-steps-per-game * num-envs)"
        ),
    )
    parser.add_argument("--max-steps-per-game", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hand-size", type=int, default=7)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--deterministic-rollout", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument(
        "--batch-workers",
        type=int,
        default=1,
        help="parallel worker threads for Go-side engine step/encode batches "
        "(default: 1 = serial; cgo releases the GIL so N threads run in parallel)",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument(
        "--card-token-cache",
        type=Path,
        default=Path("data/text_encoder_card_tokens.pt"),
        help="text encoder card-token cache (.pt); built in memory from oracle if missing",
    )
    parser.add_argument(
        "--oracle-db",
        type=Path,
        default=DEFAULT_ORACLE_DB_PATH,
        help="Scryfall oracle-cards.json bulk dump for text encoder oracle text",
    )
    parser.add_argument("--text-max-tokens", type=int, default=4096)
    parser.add_argument("--text-d-model", type=int, default=128)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--text-heads", type=int, default=4)
    parser.add_argument("--text-d-ff", type=int, default=512)
    parser.add_argument(
        "--native-render-plan",
        action="store_true",
        help="use mage-go native render-plan emission for text encoder rollouts",
    )
    parser.add_argument("--render-plan-capacity", type=int, default=4096)
    parser.add_argument(
        "--card-body-dedup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="emit each unique card body once at the top of the snapshot and "
        "reference it from per-zone occurrences (~3-4x token reduction on "
        "card-heavy snapshots; default: off)",
    )
    parser.add_argument(
        "--text-native-assembler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the text-encoder assembler natively in mage-go via "
        "MageEncodeTokens (default: on; pass --no-text-native-assembler to "
        "fall back to the Python assemble_batch path)",
    )
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
    parser.add_argument("--spr-k", type=int, default=5)
    parser.add_argument("--spr-proj-dim", type=int, default=256)
    parser.add_argument("--max-options", type=int, default=64)
    parser.add_argument("--max-targets-per-option", type=int, default=4)
    parser.add_argument(
        "--max-decision-groups",
        type=int,
        default=16,
        help="Cap on per-step decision groups in the replay buffer. Combat "
        "steps (attackers/blockers) emit one group per creature; rows that "
        "would exceed this cap raise at append time.",
    )
    parser.add_argument(
        "--max-cached-choices",
        type=int,
        default=None,
        help="Cap on per-group cached choices in the replay buffer. Priority "
        "candidates beyond this width are truncated when building the "
        "decision layout. The native encoder requires this to be at least "
        "max(max_options, max_targets_per_option + 1); when unset, it "
        "defaults to that floor.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="optimizer learning rate; default depends on --trainer "
        "(rnad: 5e-5 per paper §199; ppo: 3e-4)",
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    # Default treats a draw as equivalent to a loss for both players so that
    # agents don't learn to drag out games (e.g. mill stalls, infinite loops)
    # to avoid losing.
    parser.add_argument(
        "--draw-penalty",
        type=float,
        default=1.0,
        help=(
            "terminal reward magnitude applied to both players on a draw "
            "(default 1.0 = treat draw as a loss)"
        ),
    )
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
        "--eval-games-per-snapshot",
        type=int,
        default=None,
        help="total eval games played each time a snapshot is taken, distributed "
        "across the opponent pool with a recency bias. Defaults to "
        "max(100, episodes // 2500) so eval time stays near 2%% of training "
        "(~50 snapshots * ~2%% of per-snapshot training budget), with a 100-game "
        "floor to keep win-rate estimates meaningful",
    )
    parser.add_argument(
        "--eval-recency-tau",
        type=float,
        default=4.0,
        help="decay constant (in checkpoint positions) for recency-biased eval-game "
        "distribution; 0 = uniform across all checkpoints",
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
    if getattr(args, "text_max_tokens", 1) < 1:
        raise ValueError("--text-max-tokens must be at least 1")
    if getattr(args, "text_d_model", 1) < 1:
        raise ValueError("--text-d-model must be at least 1")
    if getattr(args, "text_layers", 1) < 1:
        raise ValueError("--text-layers must be at least 1")
    if getattr(args, "text_heads", 1) < 1:
        raise ValueError("--text-heads must be at least 1")
    if getattr(args, "text_d_ff", 1) < 1:
        raise ValueError("--text-d-ff must be at least 1")
    if getattr(args, "render_plan_capacity", 1) < 1:
        raise ValueError("--render-plan-capacity must be at least 1")
    if getattr(args, "max_decision_groups", 1) < 1:
        raise ValueError("--max-decision-groups must be at least 1")
    max_options = getattr(args, "max_options", 1)
    max_targets = getattr(args, "max_targets_per_option", 1)
    cached_floor = max(max_options, max_targets + 1)
    cached = getattr(args, "max_cached_choices", None)
    if cached is None:
        args.max_cached_choices = cached_floor
    elif cached < cached_floor:
        raise ValueError(
            f"--max-cached-choices ({cached}) must be >= "
            f"max(max_options={max_options}, "
            f"max_targets_per_option+1={max_targets + 1}); "
            "raise it or lower --max-options / --max-targets-per-option."
        )
    if getattr(args, "encoder", "slots") == "text":
        if args.trainer == "rnad" and not getattr(args, "native_render_plan", False):
            raise ValueError("--encoder text --trainer rnad requires --native-render-plan")
        args.spr = False
    if args.torch_compile and not args.no_validate:
        raise ValueError("--torch-compile requires --no-validate")
    if args.deck_json is not None and args.deck_dir is not None:
        raise ValueError("--deck-json and --deck-dir are mutually exclusive")
    if args.eval_games_per_snapshot is not None and args.eval_games_per_snapshot < 0:
        raise ValueError("--eval-games-per-snapshot must be non-negative")
    if args.eval_num_envs is not None and args.eval_num_envs < 1:
        raise ValueError("--eval-num-envs must be at least 1")
    if args.trainer == "rnad":
        if args.rnad_eta <= 0.0:
            raise ValueError("--rnad-eta must be positive")
        if args.rnad_delta_m < 1:
            raise ValueError("--rnad-delta-m must be at least 1")
        if args.rnad_m < 1:
            raise ValueError("--rnad-m must be at least 1")
        if args.rnad_neurd_beta <= 0.0:
            raise ValueError("--rnad-neurd-beta must be positive")
        if not 0.0 < args.rnad_target_ema < 1.0:
            raise ValueError("--rnad-target-ema must be in (0, 1)")
        if not 0.0 <= args.rnad_finetune_eps < 1.0:
            raise ValueError("--rnad-finetune-eps must be in [0, 1)")
        if args.rnad_finetune_ndisc < 1:
            raise ValueError("--rnad-finetune-ndisc must be at least 1")
        # Polyak target-tracking sanity: target needs to actually move
        # toward online inside one outer iteration, otherwise reg snapshots
        # are pinned to the initial random network forever and the R-NaD
        # outer-loop convergence guarantee (paper §38) does not apply.
        # ``delta_m * target_ema`` is the integrated tracking strength per
        # outer iter; below ~0.5, the EMA half-life exceeds the outer
        # iteration length and ``rnad_m`` will stay at 0 in practice.
        track = args.rnad_delta_m * args.rnad_target_ema
        if track < 0.5:
            print(
                "step=0",
                f"warning: --rnad-delta-m ({args.rnad_delta_m}) * "
                f"--rnad-target-ema ({args.rnad_target_ema}) = {track:.3f} "
                "< 0.5: target network will not meaningfully track online "
                "within one outer iteration, so reg snapshots stay pinned "
                "near the initial random policy. Either increase "
                "--rnad-target-ema or --rnad-delta-m. Paper uses "
                "delta_m * target_ema ~= 10-100 (delta_m=10k-100k, ema=1e-3).",
                flush=True,
            )


def log_ppo_stats(
    stats: PPOStats,
    *,
    games: int,
    steps: int,
    total_rollout_steps: int,
    total_generated_rollout_steps: int,
    win_stats: WinFractionStats | None = None,
    value_metrics: dict[str, float] | None = None,
    log_fn: Callable[[dict[str, Any]], None] | None = None,
    run_active: bool | None = None,
) -> None:
    """Log PPO update metrics to wandb (if active)."""
    is_run_active = wandb.run is not None if run_active is None else run_active
    if not is_run_active:
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
        "total_rollout_steps": total_rollout_steps,
        "total_generated_rollout_steps": total_generated_rollout_steps,
    }
    if win_stats is not None:
        payload.update(win_stats.as_wandb_metrics())
    if value_metrics is not None:
        payload.update(value_metrics)
    logger = wandb.log if log_fn is None else log_fn
    logger(payload)


def _snapshot_games_from_tag(tag: str) -> int | None:
    match = re.match(r"g(\d+)_p", tag)
    if match is None:
        return None
    return int(match.group(1))


def _snapshot_pct_from_tag(tag: str, total_episodes: int) -> float | None:
    match = re.search(r"_p(\d+(?:\.\d+)?)$", tag)
    if match is not None:
        return float(match.group(1))
    snapshot_games = _snapshot_games_from_tag(tag)
    if snapshot_games is None:
        return None
    return 100.0 * snapshot_games / max(1, total_episodes)


def retrospective_rating_rows(
    opponent_pool: OpponentPool,
    *,
    total_episodes: int,
) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for entry in opponent_pool.entries:
        mu = float(entry.rating.mu)
        sigma = float(entry.rating.sigma)
        rows.append(
            {
                "snapshot_pct": _snapshot_pct_from_tag(entry.tag, total_episodes),
                "snapshot_step_count": _snapshot_games_from_tag(entry.tag),
                "mu": mu,
                "sigma": sigma,
                "conservative": mu - 3.0 * sigma,
                "n_games": int(entry.n_games),
            }
        )
    return rows


def log_retrospective_table(
    run: Any,
    *,
    horizon_pct: int,
    horizon_step_count: int,
    ratings: list[dict[str, float | int | None]],
    table_factory: Callable[..., Any] | None = None,
    log_fn: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Log the current snapshot TrueSkill curve as a wandb table."""

    columns = [
        "horizon_pct",
        "horizon_step_count",
        "snapshot_pct",
        "snapshot_step_count",
        "mu",
        "sigma",
        "conservative",
        "n_games",
    ]
    data = [
        [
            horizon_pct,
            horizon_step_count,
            rating["snapshot_pct"],
            rating["snapshot_step_count"],
            rating["mu"],
            rating["sigma"],
            rating["conservative"],
            rating.get("n_games"),
        ]
        for rating in ratings
    ]
    make_table = wandb.Table if table_factory is None else table_factory
    table = make_table(columns=columns, data=data)
    payload = {
        "retrospective/current_curve": table,
        "retrospective/horizon_pct": horizon_pct,
        "retrospective/horizon_step_count": horizon_step_count,
    }
    if log_fn is None:
        run.log(payload)
    else:
        log_fn(payload)


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
    native_encoder: ShardedNativeBatchEncoder,
    optimizer: torch.optim.Optimizer,
    native_rollout: ShardedNativeRolloutDriver,
    staging_buffer: NativeTrajectoryBuffer,
    *,
    opponent_pool: OpponentPool | None = None,
    snapshot_schedule: SnapshotSchedule | None = None,
    retrospective_schedule: RetrospectiveLogSchedule | None = None,
    opponent_policy: PPOPolicy | None = None,
    resume_state: TrainingResumeState | None = None,
    resume_checkpoint: dict[str, Any] | None = None,
) -> tuple[TrainingResumeState, RNaDTrainerState | None]:
    if not native_encoder.is_available:
        raise SystemExit("native rollout requires MageEncodeBatch")
    eval_rng = random.Random(args.seed ^ 0x5EED5)

    pending_steps: list[RolloutStep] = []
    pending_returns: list[torch.Tensor] = []
    pending_episodes: list[EpisodeBatch] = []  # R-NaD: one entry per finished game
    rnad_state: RNaDTrainerState | None = None
    if args.trainer == "rnad":
        rnad_state = build_trainer_state(
            policy,
            config=RNaDConfig(
                eta=args.rnad_eta,
                delta_m=args.rnad_delta_m,
                num_outer_iterations=args.rnad_m,
                neurd_beta=args.rnad_neurd_beta,
                neurd_clip=args.rnad_neurd_clip,
                target_ema_gamma=args.rnad_target_ema,
                finetune_eps=args.rnad_finetune_eps,
                finetune_n_disc=args.rnad_finetune_ndisc,
                learning_rate=args.learning_rate,
                q_corr_rho_bar=args.rnad_q_corr_rho_bar,
                bptt_chunk_size=args.rnad_bptt_chunk_size,
                step_minibatch_size=args.minibatch_size,
            ),
            reg_snapshot_dir=args.output.parent / "rnad",
            device=policy.device,
        )
        _restore_rnad_state(rnad_state, resume_checkpoint)
        # The target network runs rollouts under R-NaD; give it its own set
        # of LSTM env buffers (online retains its own for evaluate_replay_batch).
        cast(PPOPolicy, rnad_state.target).init_lstm_env_states(args.num_envs)
    restored_state = resume_state or TrainingResumeState()
    completed_games = restored_state.completed_games
    last_saved_games = restored_state.last_saved_games
    total_rollout_steps = restored_state.total_rollout_steps
    total_generated_rollout_steps = restored_state.total_generated_rollout_steps
    total_wandb_logs = restored_state.total_wandb_logs
    run_step = getattr(wandb.run, "step", None) if wandb.run is not None else None
    if isinstance(run_step, int):
        total_wandb_logs = max(total_wandb_logs, run_step)
    next_episode_idx = completed_games
    live_games: list[LiveGame] = []
    free_slots = list(range(args.num_envs - 1, -1, -1))
    win_stats = WinFractionStats()
    transcript_warning_emitted = False
    policy.reset_rollout_buffer()

    # Rollouts sample from the target policy under R-NaD (paper §157-§191),
    # and from the online policy under PPO. The online policy always owns
    # the rollout buffer; target is a Polyak-averaged EMA living alongside.
    sampling_policy: PPOPolicy = (
        cast(PPOPolicy, rnad_state.target) if rnad_state is not None else policy
    )

    def start_game(slot_idx: int, episode_idx: int) -> LiveGame:
        staging_buffer.reset_env(slot_idx)
        policy.reset_lstm_env_states([slot_idx])
        if sampling_policy is not policy:
            sampling_policy.reset_lstm_env_states([slot_idx])
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
            slot_idx=slot_idx,
            episode_idx=episode_idx,
            episode_steps=[],
            transcript=[],
            transcript_enabled=slot_idx == 0,
        )

    def maybe_start_games() -> None:
        nonlocal next_episode_idx
        while free_slots and next_episode_idx < args.episodes:
            live_games.append(start_game(free_slots.pop(), next_episode_idx))
            next_episode_idx += 1

    def cli_step_prefix() -> str:
        return f"step={total_wandb_logs}"

    def tracked_wandb_log(payload: dict[str, Any]) -> None:
        nonlocal total_wandb_logs
        if wandb.run is None:
            return
        wandb.log(payload)
        total_wandb_logs += 1

    def disable_transcript(env: LiveGame, reason: str) -> None:
        nonlocal transcript_warning_emitted
        env.transcript_enabled = False
        if not transcript_warning_emitted:
            print(
                cli_step_prefix(),
                f"warning: disabling sample transcript capture: {reason}",
                flush=True,
            )
            transcript_warning_emitted = True

    def finish_games(finished: list[tuple[LiveGame, int]]) -> None:
        nonlocal completed_games, total_generated_rollout_steps
        if not finished:
            return

        envs = [env for env, _ in finished]
        replay_rows_by_env = policy.append_staged_episodes_to_rollout(
            staging_buffer,
            [env.slot_idx for env in envs],
        )

        # Batch the per-episode D2H: previously this issued 3 .cpu().tolist()
        # calls per finished episode (3*N CUDA syncs). Instead, gather the
        # three fields across every finished env on-device, stack them once,
        # and do a single host transfer.
        slot_idxs = [env.slot_idx for env in envs]
        device = staging_buffer.device
        slot_t = torch.tensor(slot_idxs, dtype=torch.long, device=device)
        step_counts = staging_buffer.step_count[slot_t]
        step_counts_h = step_counts.tolist()
        max_steps = int(staging_buffer.max_steps_per_trajectory)
        step_arange = torch.arange(max_steps, device=device).unsqueeze(0)
        valid_mask = step_arange < step_counts.unsqueeze(1)
        flat_player = staging_buffer.perspective_player_idx[slot_t][valid_mask]
        flat_log = staging_buffer.old_log_prob[slot_t][valid_mask]
        flat_val = staging_buffer.value[slot_t][valid_mask]
        host = torch.stack([flat_player.to(torch.float32), flat_log, flat_val], dim=0).cpu()
        host_player = host[0].long().tolist()
        host_log = host[1].tolist()
        host_val = host[2].tolist()

        cursor = 0
        for (env, winner_idx), replay_rows, step_count in zip(
            finished, replay_rows_by_env, step_counts_h, strict=True
        ):
            env.game.close()
            if step_count:
                end = cursor + step_count
                player_indices = host_player[cursor:end]
                old_log_probs = host_log[cursor:end]
                values = host_val[cursor:end]
                cursor = end
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
                    draw_penalty=args.draw_penalty,
                )
                pending_steps.extend(env.episode_steps)
                pending_returns.append(returns)
                if rnad_state is not None:
                    pending_episodes.append(
                        EpisodeBatch(
                            steps=list(env.episode_steps),
                            winner_idx=int(winner_idx),
                        )
                    )
                total_generated_rollout_steps += len(env.episode_steps)
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
    last_step_time = time.monotonic()
    while live_games:
        ready_t, over_t, player_t, winner_t = native_rollout.poll([env.game for env in live_games])
        _t = time.perf_counter()
        # Pull poll results to host once. Each .tolist() is one transfer; the
        # alternative (int(t[idx]) per env) issues 4*N tiny syncs per poll.
        ready_l = ready_t.tolist()
        over_l = over_t.tolist()
        player_l = player_t.tolist()
        winner_l = winner_t.tolist()
        ready_envs: list[LiveGame] = []
        ready_players: list[int] = []
        still_live: list[LiveGame] = []
        finished_games: list[tuple[LiveGame, int]] = []
        for idx, env in enumerate(live_games):
            is_over = bool(over_l[idx])
            if is_over or env.action_count >= args.max_steps_per_game:
                finished_games.append((env, int(winner_l[idx]) if is_over else -1))
                continue
            still_live.append(env)
            if ready_l[idx]:
                ready_envs.append(env)
                ready_players.append(int(player_l[idx]))
        live_games = still_live
        _record_phase("partition", _t)
        _t = time.perf_counter()
        finish_games(finished_games)
        _record_phase("finish", _t)

        if ready_envs:
            _t = time.perf_counter()
            ready_env_indices = [env.slot_idx for env in ready_envs]
            ready_games = [env.game for env in ready_envs]
            _record_phase("ready_lists", _t)
            parsed_batch = native_encoder.encode_handles(
                ready_games,
                perspective_player_indices=ready_players,
            )
            lstm_state_inputs = sampling_policy.lstm_env_state_inputs(ready_env_indices)
            finetune_active = rnad_state is not None and rnad_state.is_finetuning
            finetune_eps = args.rnad_finetune_eps if finetune_active else 0.0
            finetune_n_disc = args.rnad_finetune_ndisc if finetune_active else 0
            with torch.no_grad():
                policy_steps = sampling_policy.sample_native_batch(
                    parsed_batch,
                    env_indices=ready_env_indices,
                    deterministic=args.deterministic_rollout,
                    finetune_eps=finetune_eps,
                    finetune_n_disc=finetune_n_disc,
                )
            _t = time.perf_counter()
            log_probs = torch.stack([policy_step.log_prob for policy_step in policy_steps])
            values = torch.stack([policy_step.value for policy_step in policy_steps])
            _record_phase("stack_steps", _t)

            # Batch the per-env bookkeeping with comprehensions, then walk
            # the envs once for the rare per-env work (transcripts and the
            # action_count bump). The hot path is now a few torch ops on
            # CPU + one async H2D copy, instead of N Python list-extends and
            # a synchronous torch.tensor(..., device=cuda) build.
            _t = time.perf_counter()
            counts = [len(s.selected_choice_cols) for s in policy_steps]
            may_selected = [s.may_selected for s in policy_steps]
            starts: list[int] = list(itertools.accumulate(counts, initial=0))[:-1]
            selected_cols: list[int] = [c for s in policy_steps for c in s.selected_choice_cols]
            selected_choice_cols_flat = torch.tensor(selected_cols, dtype=torch.long).to(
                policy.device, non_blocking=True
            )
            _record_phase("build_cols", _t)

            _t = time.perf_counter()
            for env, policy_step in zip(ready_envs, policy_steps, strict=True):
                env.action_count += 1
                if not env.transcript_enabled:
                    continue
                try:
                    transcript_state, transcript_pending = _current_transcript_snapshot(env.game)
                    transcript_action = copy.deepcopy(policy_step.action)
                    if policy_step.trace.kind != "may":
                        _trace, decoded_action = policy._decode_action(
                            policy_step.trace.kind,
                            transcript_pending,
                            list(policy_step.selected_choice_cols),
                        )
                        transcript_action = copy.deepcopy(decoded_action)
                    env.transcript.append(
                        TranscriptAction(
                            state=transcript_state,
                            pending=transcript_pending,
                            action=transcript_action,
                        )
                    )
                except Exception as exc:
                    disable_transcript(
                        env,
                        f"{exc} while snapshotting live game for action {policy_step.action!r}",
                    )
            _record_phase("env_loop", _t)

            staging_buffer.stage_batch(
                ready_env_indices,
                parsed_batch,
                selected_choice_cols_flat=selected_choice_cols_flat,
                may_selected=may_selected,
                old_log_probs=log_probs,
                values=values,
                perspective_player_indices=ready_players,
                decision_counts=counts,
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
            rollout_step_count = len(pending_steps)
            if rnad_state is not None:
                # If --cuda-memory-snapshot is set, also print live/reserved
                # before each update and record the allocator history so an
                # OOM dumps a snapshot loadable at pytorch.org/memory_viz.
                snapshot_armed = args.cuda_memory_snapshot is not None and torch.cuda.is_available()
                if snapshot_armed:
                    print(
                        cli_step_prefix(),
                        f"[mem] before run_rnad_update[{total_rollout_steps}]: "
                        f"{torch.cuda.memory_allocated() / 1e9:.2f} GB live, "
                        f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved",
                        flush=True,
                    )
                    torch.cuda.memory._record_memory_history(max_entries=100_000)
                try:
                    stats = run_rnad_update(
                        policy,
                        optimizer,
                        rnad_state,
                        pending_episodes,
                    )
                except torch.OutOfMemoryError:
                    if snapshot_armed:
                        path = Path(args.cuda_memory_snapshot)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            torch.cuda.memory._dump_snapshot(str(path))
                            print(
                                cli_step_prefix(),
                                f"[mem] OOM in run_rnad_update; allocator history dumped "
                                f"to {path}. Load at https://pytorch.org/memory_viz",
                                flush=True,
                            )
                        except Exception as dump_exc:
                            print(
                                cli_step_prefix(),
                                f"[mem] _dump_snapshot failed: "
                                f"{type(dump_exc).__name__}: {dump_exc}",
                                flush=True,
                            )
                    raise
                finally:
                    if snapshot_armed:
                        torch.cuda.memory._record_memory_history(enabled=None)
            else:
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
            now = time.monotonic()
            elapsed = now - last_step_time
            last_step_time = now
            fields = [
                cli_step_prefix(),
                f"update[{args.trainer}]",
                f"games={completed_games}",
                f"steps={rollout_step_count}",
                f"dt={elapsed:.1f}s",
                f"loss={stats.loss:.4f}",
                f"policy={stats.policy_loss:.4f}",
                f"value={stats.value_loss:.4f}",
            ]
            if rnad_state is None:
                fields.extend(
                    [
                        f"entropy={stats.entropy:.4f}",
                        f"kl={stats.approx_kl:.4f}",
                        f"clip={stats.clip_fraction:.3f}",
                    ]
                )
            else:
                fields.append(f"rnad_m={rnad_state.outer_iteration}")
            print(*fields, flush=True)
            total_rollout_steps += rollout_step_count
            _t = time.perf_counter()
            value_metrics = rollout_value_metrics(pending_steps, rollout_returns)
            if rnad_state is not None and rnad_state.last_stats:
                rs = rnad_state.last_stats[0]
                value_metrics.update(
                    {
                        "rnad/sampled_log_ratio_mean": rs.sampled_log_ratio_mean,
                        "rnad/sampled_log_ratio_absmax": rs.sampled_log_ratio_absmax,
                        "rnad/is_bias_up_mean": rs.is_bias_up_mean,
                        "rnad/is_bias_down_mean": rs.is_bias_down_mean,
                        "rnad/v_target_reg_share": rs.v_target_reg_share,
                        "rnad/q_clip_fraction": rs.q_clip_fraction,
                        "rnad/v_hat_mean": rs.v_hat_mean,
                        "rnad/transformed_reward_mean": rs.transformed_reward_mean,
                        "rnad/grad_norm": rs.grad_norm,
                        "rnad/outer_iteration": rnad_state.outer_iteration,
                        "rnad/gradient_step": rnad_state.gradient_step,
                    }
                )
            log_ppo_stats(
                stats,
                games=completed_games,
                steps=rollout_step_count,
                total_rollout_steps=total_rollout_steps,
                total_generated_rollout_steps=total_generated_rollout_steps,
                win_stats=win_stats,
                value_metrics=value_metrics,
                log_fn=tracked_wandb_log,
                run_active=True,
            )
            policy.reset_rollout_buffer()
            pending_steps.clear()
            pending_returns.clear()
            pending_episodes.clear()
            win_stats.reset()
            _record_phase("post_update", _t)

        if (
            args.save_every
            and completed_games > 0
            and completed_games % args.save_every == 0
            and completed_games != last_saved_games
        ):
            save_checkpoint(
                args.output,
                policy,
                optimizer,
                args,
                opponent_pool=opponent_pool,
                snapshot_schedule=snapshot_schedule,
                retrospective_schedule=retrospective_schedule,
                resume_state=TrainingResumeState(
                    completed_games=completed_games,
                    last_saved_games=completed_games,
                    total_rollout_steps=total_rollout_steps,
                    total_generated_rollout_steps=total_generated_rollout_steps,
                    total_wandb_logs=total_wandb_logs,
                ),
                rnad_state=rnad_state,
            )
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
                    step_prefix=cli_step_prefix(),
                    log_fn=tracked_wandb_log,
                )

        if opponent_pool is not None and retrospective_schedule is not None:
            for horizon_pct, horizon_step_count in retrospective_schedule.fire(completed_games):
                if wandb.run is not None:
                    log_retrospective_table(
                        wandb.run,
                        horizon_pct=horizon_pct,
                        horizon_step_count=horizon_step_count,
                        ratings=retrospective_rating_rows(
                            opponent_pool,
                            total_episodes=args.episodes,
                        ),
                        log_fn=tracked_wandb_log,
                    )

        _t = time.perf_counter()
        maybe_start_games()
        _record_phase("maybe_start", _t)

    if pending_steps:
        rollout_returns = torch.cat(pending_returns)
        rollout_step_count = len(pending_steps)
        if rnad_state is not None and pending_episodes:
            stats = run_rnad_update(
                policy,
                optimizer,
                rnad_state,
                pending_episodes,
            )
        else:
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
            cli_step_prefix(),
            f"final_update[{args.trainer}]",
            f"games={completed_games}",
            f"steps={rollout_step_count}",
            f"loss={stats.loss:.4f}",
            f"policy={stats.policy_loss:.4f}",
            f"value={stats.value_loss:.4f}",
            f"entropy={stats.entropy:.4f}",
            flush=True,
        )
        total_rollout_steps += rollout_step_count
        log_ppo_stats(
            stats,
            games=completed_games,
            steps=rollout_step_count,
            total_rollout_steps=total_rollout_steps,
            total_generated_rollout_steps=total_generated_rollout_steps,
            win_stats=win_stats,
            value_metrics=rollout_value_metrics(pending_steps, rollout_returns),
            log_fn=tracked_wandb_log,
            run_active=True,
        )
        policy.reset_rollout_buffer()
        pending_episodes.clear()

    return (
        TrainingResumeState(
            completed_games=completed_games,
            last_saved_games=last_saved_games,
            total_rollout_steps=total_rollout_steps,
            total_generated_rollout_steps=total_generated_rollout_steps,
            total_wandb_logs=total_wandb_logs,
        ),
        rnad_state,
    )


def train_text_envs(
    args: argparse.Namespace,
    mage: Any,
    deck_pool: list[dict[str, Any]],
    backend: TextTrainingBackend,
    optimizer: torch.optim.Optimizer,
    *,
    resume_state: TrainingResumeState | None = None,
) -> tuple[TrainingResumeState, None]:
    """Slow Python text-encoder PPO loop.

    This intentionally stays separate from the native slot loop. It provides a
    correctness path for `--encoder text`; performance work can later replace
    the single-env Python stepping with native render-plan batching.
    """

    restored_state = resume_state or TrainingResumeState()
    completed_games = restored_state.completed_games
    last_saved_games = restored_state.last_saved_games
    total_rollout_steps = restored_state.total_rollout_steps
    total_generated_rollout_steps = restored_state.total_generated_rollout_steps
    total_wandb_logs = restored_state.total_wandb_logs
    run_step = getattr(wandb.run, "step", None) if wandb.run is not None else None
    if isinstance(run_step, int):
        total_wandb_logs = max(total_wandb_logs, run_step)

    pending_steps: list[RolloutStep] = []
    pending_returns: list[torch.Tensor] = []
    win_stats = WinFractionStats()
    backend.replay_buffer.reset()
    backend.policy.init_lstm_env_states(1)
    last_step_time = time.monotonic()

    def cli_step_prefix() -> str:
        return f"step={total_wandb_logs}"

    def tracked_wandb_log(payload: dict[str, Any]) -> None:
        nonlocal total_wandb_logs
        if wandb.run is None:
            return
        wandb.log(payload)
        total_wandb_logs += 1

    def run_update(*, final: bool = False) -> None:
        nonlocal total_rollout_steps, last_step_time
        if not pending_steps:
            return
        rollout_returns = torch.cat(pending_returns)
        rollout_step_count = len(pending_steps)
        stats = ppo_update(
            backend.policy,
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
        now = time.monotonic()
        elapsed = now - last_step_time
        last_step_time = now
        print(
            cli_step_prefix(),
            "final_update[ppo,text]" if final else "update[ppo,text]",
            f"games={completed_games}",
            f"steps={rollout_step_count}",
            f"dt={elapsed:.1f}s",
            f"loss={stats.loss:.4f}",
            f"policy={stats.policy_loss:.4f}",
            f"value={stats.value_loss:.4f}",
            f"entropy={stats.entropy:.4f}",
            f"kl={stats.approx_kl:.4f}",
            f"clip={stats.clip_fraction:.3f}",
            flush=True,
        )
        total_rollout_steps += rollout_step_count
        log_ppo_stats(
            stats,
            games=completed_games,
            steps=rollout_step_count,
            total_rollout_steps=total_rollout_steps,
            total_generated_rollout_steps=total_generated_rollout_steps,
            win_stats=win_stats,
            value_metrics=rollout_value_metrics(pending_steps, rollout_returns),
            log_fn=tracked_wandb_log,
            run_active=True,
        )
        backend.replay_buffer.reset()
        pending_steps.clear()
        pending_returns.clear()
        win_stats.reset()

    for episode_idx in range(completed_games, args.episodes):
        backend.policy.reset_lstm_env_states([0])
        seed = args.seed + episode_idx
        deck_a, deck_b = sample_decks(deck_pool, seed)
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
        winner_idx = -1
        try:
            for _action_idx in range(args.max_steps_per_game):
                game.refresh_state()
                if game.is_over:
                    winner_idx = _winner_idx_from_game(game)
                    break
                pending = cast(PendingState | None, game.pending or game.legal())
                if pending is None:
                    game.step({"kind": "pass"})
                    continue
                snapshot = cast(GameStateSnapshot, copy.deepcopy(game.state))
                player_idx = int(pending.get("player_idx", 0) or 0)
                if player_idx not in (0, 1):
                    player_idx = 0
                policy_steps = sample_text_policy_batch(
                    args,
                    backend,
                    [snapshot],
                    [pending],
                    env_indices=[0],
                    perspective_player_indices=[player_idx],
                    deterministic=args.deterministic_rollout,
                )
                if not policy_steps:
                    game.step({"kind": "pass"})
                    continue
                policy_step = policy_steps[0]
                game.step(dict(policy_step.action))
                episode_steps.append(
                    RolloutStep(
                        perspective_player_idx=player_idx,
                        old_log_prob=float(policy_step.log_prob.detach().cpu()),
                        value=float(policy_step.value.detach().cpu()),
                        replay_idx=policy_step.replay_idx,
                    )
                )
            game.refresh_state()
            if game.is_over:
                winner_idx = _winner_idx_from_game(game)
        finally:
            try:
                game.close()
            except Exception:
                pass

        completed_games += 1
        if winner_idx == 0:
            win_stats.p1_wins += 1
        elif winner_idx == 1:
            win_stats.p2_wins += 1
        else:
            win_stats.draws += 1
        if episode_steps:
            pending_steps.extend(episode_steps)
            pending_returns.append(
                gae_returns(
                    episode_steps,
                    winner_idx=winner_idx,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    draw_penalty=args.draw_penalty,
                )
            )
            total_generated_rollout_steps += len(episode_steps)
        if len(pending_steps) >= args.rollout_steps:
            run_update()

        if (
            args.save_every
            and completed_games > 0
            and completed_games % args.save_every == 0
            and completed_games != last_saved_games
        ):
            save_checkpoint(
                args.output,
                backend.policy,
                optimizer,
                args,
                resume_state=TrainingResumeState(
                    completed_games=completed_games,
                    last_saved_games=completed_games,
                    total_rollout_steps=total_rollout_steps,
                    total_generated_rollout_steps=total_generated_rollout_steps,
                    total_wandb_logs=total_wandb_logs,
                ),
            )
            last_saved_games = completed_games

    run_update(final=True)
    return (
        TrainingResumeState(
            completed_games=completed_games,
            last_saved_games=last_saved_games,
            total_rollout_steps=total_rollout_steps,
            total_generated_rollout_steps=total_generated_rollout_steps,
            total_wandb_logs=total_wandb_logs,
        ),
        None,
    )


def sample_native_text_policy_batch(
    args: argparse.Namespace,
    backend: TextTrainingBackend,
    native_batch: Any,
    *,
    env_indices: list[int],
    perspective_player_indices: list[int],
    deterministic: bool = False,
    sampling_policy: TextActorCritic | None = None,
    text_batch: Any | None = None,
) -> list[Any]:
    if text_batch is not None:
        # Phase 5: native assembler already produced the TextEncodedBatch.
        # Skip the render-plan slice + Python ``assemble_batch`` call.
        encoded = text_batch
    else:
        if native_batch.render_plan is None or native_batch.render_plan_lengths is None:
            raise NativeEncodingError("native encoder did not return render plans")
        if native_batch.render_plan_overflow is not None and bool(
            native_batch.render_plan_overflow.any()
        ):
            overflow_idx = int(
                native_batch.render_plan_overflow.nonzero(as_tuple=False)[0, 0].item()
            )
            raise NativeEncodingError(f"native render plan overflowed for batch row {overflow_idx}")

        # ``render_plan_lengths`` is a small CPU int64 tensor — ``.tolist()``
        # is a single C-loop materialization, no GPU sync. Slice the CPU
        # int32 render-plan tensor directly into the assembler; no numpy
        # round-trip.
        lengths = native_batch.render_plan_lengths.tolist()
        render_plan_cpu = native_batch.render_plan
        plans: list[torch.Tensor] = [
            render_plan_cpu[row_idx, : int(length)] for row_idx, length in enumerate(lengths)
        ]
        encoded = assemble_batch(
            plans,
            backend.cache,
            backend.tokenizer,
            max_tokens=args.text_max_tokens,
            on_overflow="truncate",
            assembler_tokens=backend.assembler_tokens,
        )

    # The native encoder writes these directly into pinned CPU scratch — no
    # device transfer needed, no detach (we're already under no_grad).
    layouts: list[Any] = []
    starts = native_batch.decision_start.tolist()
    counts = native_batch.decision_count.tolist()
    for row_idx, trace_kind_raw in enumerate(native_batch.trace_kinds):
        trace_kind = cast(Any, trace_kind_raw)
        start = int(starts[row_idx])
        count = int(counts[row_idx])
        end = start + count
        pending = cast(PendingState, {"kind": trace_kind, "options": []})
        layouts.append(
            TextDecisionLayout(
                trace_kind=trace_kind,
                decision_option_idx=native_batch.decision_option_idx[start:end],
                decision_target_idx=native_batch.decision_target_idx[start:end],
                decision_mask=native_batch.decision_mask[start:end],
                uses_none_head=native_batch.uses_none_head[start:end],
                pending=pending,
            )
        )

    actor = sampling_policy if sampling_policy is not None else backend.policy
    return actor.sample_text_batch(
        encoded,
        env_indices=env_indices,
        perspective_player_indices=perspective_player_indices,
        layouts=layouts,
        deterministic=deterministic,
    )


def train_text_native_batched_envs(
    args: argparse.Namespace,
    mage: Any,
    deck_pool: list[dict[str, Any]],
    backend: TextTrainingBackend,
    optimizer: torch.optim.Optimizer,
    native_rollout: ShardedNativeRolloutDriver,
    native_encoder: ShardedNativeBatchEncoder,
    *,
    resume_state: TrainingResumeState | None = None,
    resume_checkpoint: dict[str, Any] | None = None,
) -> tuple[TrainingResumeState, RNaDTrainerState | None]:
    if not native_encoder.is_available:
        raise SystemExit("native text rollout requires MageEncodeBatch")

    restored_state = resume_state or TrainingResumeState()
    completed_games = restored_state.completed_games
    last_saved_games = restored_state.last_saved_games
    total_rollout_steps = restored_state.total_rollout_steps
    total_generated_rollout_steps = restored_state.total_generated_rollout_steps
    total_wandb_logs = restored_state.total_wandb_logs
    run_step = getattr(wandb.run, "step", None) if wandb.run is not None else None
    if isinstance(run_step, int):
        total_wandb_logs = max(total_wandb_logs, run_step)

    pending_steps: list[RolloutStep] = []
    pending_returns: list[torch.Tensor] = []
    pending_episodes: list[EpisodeBatch] = []
    rnad_state: RNaDTrainerState | None = None
    if args.trainer == "rnad":
        rnad_state = build_trainer_state(
            backend.policy,
            config=RNaDConfig(
                eta=args.rnad_eta,
                delta_m=args.rnad_delta_m,
                num_outer_iterations=args.rnad_m,
                neurd_beta=args.rnad_neurd_beta,
                neurd_clip=args.rnad_neurd_clip,
                target_ema_gamma=args.rnad_target_ema,
                finetune_eps=args.rnad_finetune_eps,
                finetune_n_disc=args.rnad_finetune_ndisc,
                learning_rate=args.learning_rate,
                q_corr_rho_bar=args.rnad_q_corr_rho_bar,
                bptt_chunk_size=args.rnad_bptt_chunk_size,
                step_minibatch_size=args.minibatch_size,
            ),
            reg_snapshot_dir=args.output.parent / "rnad",
            device=backend.policy.device,
        )
        _restore_rnad_state(rnad_state, resume_checkpoint)
        cast(TextActorCritic, rnad_state.target).init_lstm_env_states(args.num_envs)
    win_stats = WinFractionStats()
    backend.replay_buffer.reset()
    backend.policy.init_lstm_env_states(args.num_envs)
    sampling_policy: TextActorCritic = (
        cast(TextActorCritic, rnad_state.target) if rnad_state is not None else backend.policy
    )
    next_episode_idx = completed_games
    live_games: list[LiveGame] = []
    free_slots = list(range(args.num_envs - 1, -1, -1))
    last_step_time = time.monotonic()

    def cli_step_prefix() -> str:
        return f"step={total_wandb_logs}"

    def tracked_wandb_log(payload: dict[str, Any]) -> None:
        nonlocal total_wandb_logs
        if wandb.run is None:
            return
        wandb.log(payload)
        total_wandb_logs += 1

    def start_game(slot_idx: int, episode_idx: int) -> LiveGame:
        backend.policy.reset_lstm_env_states([slot_idx])
        if sampling_policy is not backend.policy:
            sampling_policy.reset_lstm_env_states([slot_idx])
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
            slot_idx=slot_idx,
            episode_idx=episode_idx,
            episode_steps=[],
            transcript=[],
        )

    def maybe_start_games() -> None:
        nonlocal next_episode_idx
        if len(pending_steps) >= args.rollout_steps:
            return
        while free_slots and next_episode_idx < args.episodes:
            live_games.append(start_game(free_slots.pop(), next_episode_idx))
            next_episode_idx += 1

    def finish_games(finished: list[tuple[LiveGame, int]]) -> None:
        nonlocal completed_games, total_generated_rollout_steps
        for env, winner_idx in finished:
            try:
                env.game.close()
            except Exception:
                pass
            if env.episode_steps:
                pending_steps.extend(env.episode_steps)
                pending_returns.append(
                    gae_returns(
                        env.episode_steps,
                        winner_idx=winner_idx,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        draw_penalty=args.draw_penalty,
                    )
                )
                if rnad_state is not None:
                    pending_episodes.append(
                        EpisodeBatch(
                            steps=list(env.episode_steps),
                            winner_idx=int(winner_idx),
                        )
                    )
                total_generated_rollout_steps += len(env.episode_steps)
            free_slots.append(env.slot_idx)
            if winner_idx == 0:
                win_stats.p1_wins += 1
            elif winner_idx == 1:
                win_stats.p2_wins += 1
            else:
                win_stats.draws += 1
            completed_games += 1

    def run_update(*, final: bool = False) -> None:
        nonlocal total_rollout_steps, last_step_time
        if not pending_steps:
            return
        rollout_returns = torch.cat(pending_returns)
        rollout_step_count = len(pending_steps)
        if rnad_state is not None and pending_episodes:
            stats = run_rnad_update(
                backend.policy,
                optimizer,
                rnad_state,
                pending_episodes,
            )
            trainer_label = "rnad,text,native"
        else:
            stats = ppo_update(
                backend.policy,
                optimizer,
                pending_steps,
                rollout_returns,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_epsilon=args.clip_epsilon,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                spr_coef=0.0,
            )
            trainer_label = "ppo,text,native"
        now = time.monotonic()
        elapsed = now - last_step_time
        last_step_time = now
        fields = [
            cli_step_prefix(),
            f"final_update[{trainer_label}]" if final else f"update[{trainer_label}]",
            f"games={completed_games}",
            f"steps={rollout_step_count}",
            f"dt={elapsed:.1f}s",
            f"loss={stats.loss:.4f}",
            f"policy={stats.policy_loss:.4f}",
            f"value={stats.value_loss:.4f}",
        ]
        if rnad_state is None:
            fields.extend(
                [
                    f"entropy={stats.entropy:.4f}",
                    f"kl={stats.approx_kl:.4f}",
                    f"clip={stats.clip_fraction:.3f}",
                ]
            )
        else:
            fields.append(f"rnad_m={rnad_state.outer_iteration}")
        print(*fields, flush=True)
        total_rollout_steps += rollout_step_count
        value_metrics = rollout_value_metrics(pending_steps, rollout_returns)
        if rnad_state is not None and rnad_state.last_stats:
            rs = rnad_state.last_stats[0]
            value_metrics.update(
                {
                    "rnad/sampled_log_ratio_mean": rs.sampled_log_ratio_mean,
                    "rnad/sampled_log_ratio_absmax": rs.sampled_log_ratio_absmax,
                    "rnad/is_bias_up_mean": rs.is_bias_up_mean,
                    "rnad/is_bias_down_mean": rs.is_bias_down_mean,
                    "rnad/v_target_reg_share": rs.v_target_reg_share,
                    "rnad/q_clip_fraction": rs.q_clip_fraction,
                    "rnad/v_hat_mean": rs.v_hat_mean,
                    "rnad/transformed_reward_mean": rs.transformed_reward_mean,
                    "rnad/grad_norm": rs.grad_norm,
                    "rnad/outer_iteration": rnad_state.outer_iteration,
                    "rnad/gradient_step": rnad_state.gradient_step,
                }
            )
        log_ppo_stats(
            stats,
            games=completed_games,
            steps=rollout_step_count,
            total_rollout_steps=total_rollout_steps,
            total_generated_rollout_steps=total_generated_rollout_steps,
            win_stats=win_stats,
            value_metrics=value_metrics,
            log_fn=tracked_wandb_log,
            run_active=True,
        )
        backend.replay_buffer.reset()
        pending_steps.clear()
        pending_returns.clear()
        pending_episodes.clear()
        win_stats.reset()

    maybe_start_games()
    while live_games:
        ready_t, over_t, player_t, winner_t = native_rollout.poll([env.game for env in live_games])
        ready_l = ready_t.tolist()
        over_l = over_t.tolist()
        player_l = player_t.tolist()
        winner_l = winner_t.tolist()
        ready_envs: list[LiveGame] = []
        ready_players: list[int] = []
        still_live: list[LiveGame] = []
        finished_games: list[tuple[LiveGame, int]] = []
        for idx, env in enumerate(live_games):
            is_over = bool(over_l[idx])
            if is_over or env.action_count >= args.max_steps_per_game:
                finished_games.append((env, int(winner_l[idx]) if is_over else -1))
                continue
            still_live.append(env)
            if ready_l[idx]:
                ready_envs.append(env)
                ready_players.append(int(player_l[idx]))
        live_games = still_live
        finish_games(finished_games)

        if ready_envs:
            ready_env_indices = [env.slot_idx for env in ready_envs]
            ready_games = [env.game for env in ready_envs]
            text_batch: Any | None = None
            if getattr(args, "text_native_assembler", True):
                native_batch, nat_outputs = native_encoder.encode_tokens(
                    ready_games,
                    perspective_player_indices=ready_players,
                    max_tokens=args.text_max_tokens,
                    max_options=args.max_options,
                    max_targets=args.max_targets_per_option,
                    max_card_refs=256,
                )
                text_batch = nat_outputs.to_text_encoded_batch()
            else:
                native_batch = native_encoder.encode_handles(
                    ready_games,
                    perspective_player_indices=ready_players,
                )
            with torch.no_grad():
                policy_steps = sample_native_text_policy_batch(
                    args,
                    backend,
                    native_batch,
                    env_indices=ready_env_indices,
                    perspective_player_indices=ready_players,
                    deterministic=args.deterministic_rollout,
                    sampling_policy=sampling_policy,
                    text_batch=text_batch,
                )
            counts = [len(s.selected_choice_cols) for s in policy_steps]
            starts: list[int] = list(itertools.accumulate(counts, initial=0))[:-1]
            selected_cols: list[int] = [c for s in policy_steps for c in s.selected_choice_cols]
            may_selected = [s.may_selected for s in policy_steps]
            # Batch the per-step D2H of log_prob/value into one stack+.tolist()
            # so we incur 2 GPU syncs total instead of 2*N.
            if policy_steps:
                log_probs_cpu = (
                    torch.stack([s.log_prob for s in policy_steps]).detach().cpu().tolist()
                )
                values_cpu = torch.stack([s.value for s in policy_steps]).detach().cpu().tolist()
            else:
                log_probs_cpu = []
                values_cpu = []
            for step_idx, (env, player_idx, policy_step) in enumerate(
                zip(ready_envs, ready_players, policy_steps, strict=True)
            ):
                env.action_count += 1
                env.episode_steps.append(
                    RolloutStep(
                        perspective_player_idx=int(player_idx),
                        old_log_prob=float(log_probs_cpu[step_idx]),
                        value=float(values_cpu[step_idx]),
                        replay_idx=policy_step.replay_idx,
                    )
                )
            native_rollout.step_by_choice(
                ready_games,
                decision_starts=starts,
                decision_counts=counts,
                selected_choice_cols=selected_cols,
                may_selected=may_selected,
                max_options=args.max_options,
                max_targets_per_option=args.max_targets_per_option,
            )

        if len(pending_steps) >= args.rollout_steps and not live_games:
            run_update()
        maybe_start_games()

    run_update(final=True)
    return (
        TrainingResumeState(
            completed_games=completed_games,
            last_saved_games=last_saved_games,
            total_rollout_steps=total_rollout_steps,
            total_generated_rollout_steps=total_generated_rollout_steps,
            total_wandb_logs=total_wandb_logs,
        ),
        rnad_state,
    )


def _winner_idx_from_game(game: Any) -> int:
    winner_id = getattr(game, "winner", None) or ""
    players = (getattr(game, "state", None) or {}).get("players", []) or []
    for idx, player in enumerate(players):
        if player.get("ID", "") == winner_id or player.get("Name", "") == winner_id:
            return idx
    return -1


def take_snapshot_and_eval(
    *,
    args: argparse.Namespace,
    threshold: int,
    policy: PPOPolicy,
    opponent_policy: PPOPolicy,
    opponent_pool: OpponentPool,
    native_encoder: ShardedNativeBatchEncoder,
    native_rollout: ShardedNativeRolloutDriver,
    mage: Any,
    deck_pool: list[dict[str, Any]],
    rng: random.Random,
    step_prefix: str,
    log_fn: Callable[[dict[str, Any]], None],
) -> None:
    tag = snapshot_tag(threshold, args.episodes)
    snapshot_path = save_snapshot(policy, args.opponent_pool_dir, tag)
    current_entry = opponent_pool.add_snapshot(snapshot_path, tag)
    current_entry.cached_policy = opponent_policy_state_dict(policy)
    print(
        step_prefix,
        f"pool: snapshot {tag} -> {snapshot_path} (pool size={len(opponent_pool.entries)})",
        flush=True,
    )

    eval_games_per_snapshot = (
        args.eval_games_per_snapshot
        if args.eval_games_per_snapshot is not None
        else max(100, args.episodes // 2500)
    )

    # Historical opponents only — exclude the freshly-added snapshot from its
    # own eval. Playing it against itself is pure variance and would fool
    # TrueSkill into tightening σ on the current checkpoint.
    historical_opponents = opponent_pool.entries[:-1]

    if eval_games_per_snapshot == 0 or not historical_opponents:
        log_fn(
            {
                **opponent_pool.current_rating_metrics(),
                "eval/snapshot_games": float(threshold),
            }
        )
        return

    game_opponents = distribute_games_by_recency(
        historical_opponents,
        eval_games_per_snapshot,
        args.eval_recency_tau,
    )
    if not game_opponents:
        return

    unique_opponents: list[OpponentEntry] = []
    seen: set[str] = set()
    for opp in game_opponents:
        if opp.tag not in seen:
            seen.add(opp.tag)
            unique_opponents.append(opp)

    eval_num_envs = (
        args.eval_num_envs if args.eval_num_envs is not None else eval_games_per_snapshot
    )
    seed_base = args.seed + threshold * 1000
    metrics = run_eval_matches(
        main_policy=policy,
        opponent_policy=opponent_policy,
        game_opponents=game_opponents,
        pool=opponent_pool,
        current_entry=current_entry,
        native_encoder=native_encoder,
        native_rollout=native_rollout,
        mage=mage,
        deck_pool=deck_pool,
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
    for opponent in unique_opponents:
        games = int(metrics.get(f"eval/opp_{opponent.tag}_games", 0.0))
        main_win = metrics.get(f"eval/opp_{opponent.tag}_main_win_fraction", 0.0)
        opponent_mu = metrics.get(f"eval/opp_{opponent.tag}_rating_mu", opponent.rating.mu)
        opponent_sigma = metrics.get(
            f"eval/opp_{opponent.tag}_rating_sigma",
            opponent.rating.sigma,
        )
        print(
            step_prefix,
            f"eval: snapshot_tag={tag} opponent={opponent.tag} "
            f"games={games} main_win={main_win:.2f} "
            f"rating=mu={opponent_mu:.2f},sigma={opponent_sigma:.2f}",
            flush=True,
        )

    log_fn(
        {
            **metrics,
            **opponent_pool.current_rating_metrics(),
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
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    opponent_pool: OpponentPool | None = None,
    snapshot_schedule: SnapshotSchedule | None = None,
    retrospective_schedule: RetrospectiveLogSchedule | None = None,
    resume_state: TrainingResumeState | None = None,
    wandb_run_id: str | None = None,
    run_artifact_dir: Path | None = None,
    rnad_state: RNaDTrainerState | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    effective_run_artifact_dir = run_artifact_dir or args.opponent_pool_dir.parent
    effective_wandb_run_id = wandb_run_id
    if effective_wandb_run_id is None and wandb.run is not None:
        effective_wandb_run_id = wandb.run.id
    serialized_args = {key: _wandb_summary_value(value) for key, value in vars(args).items()}
    training_state = {
        "completed_games": resume_state.completed_games if resume_state is not None else 0,
        "last_saved_games": resume_state.last_saved_games if resume_state is not None else 0,
        "total_rollout_steps": resume_state.total_rollout_steps if resume_state is not None else 0,
        "total_generated_rollout_steps": (
            resume_state.total_generated_rollout_steps if resume_state is not None else 0
        ),
        "total_wandb_logs": resume_state.total_wandb_logs if resume_state is not None else 0,
        "snapshot_schedule_next_idx": snapshot_schedule.next_idx if snapshot_schedule else 0,
        "retrospective_schedule_next_idx": (
            retrospective_schedule.next_idx if retrospective_schedule else 0
        ),
        "opponent_pool": opponent_pool.state_dict() if opponent_pool is not None else None,
    }
    if rnad_state is not None:
        # Reg snapshots are persisted separately under reg_snapshot_dir as
        # reg_m{N}.pt; here we only serialize the live trainer state the
        # outer loop needs to resume in-place (target EMA, counters).
        training_state["rnad_state"] = {
            "outer_iteration": rnad_state.outer_iteration,
            "gradient_step": rnad_state.gradient_step,
            "is_finetuning": rnad_state.is_finetuning,
            "target": cast(PPOPolicy, rnad_state.target).state_dict(),
            "reg_snapshot_dir": str(rnad_state.reg_snapshot_dir),
        }
    metadata = {
        "encoder": getattr(args, "encoder", "slots"),
        "wandb_run_id": effective_wandb_run_id,
        "run_artifact_dir": str(effective_run_artifact_dir),
    }
    if getattr(args, "encoder", "slots") == "text":
        cache_path = Path(getattr(args, "card_token_cache", ""))
        metadata["text_config"] = {
            "text_max_tokens": getattr(args, "text_max_tokens", None),
            "text_d_model": getattr(args, "text_d_model", None),
            "text_layers": getattr(args, "text_layers", None),
            "text_heads": getattr(args, "text_heads", None),
            "text_d_ff": getattr(args, "text_d_ff", None),
            "hidden_layers": getattr(args, "hidden_layers", None),
            "max_options": getattr(args, "max_options", None),
            "max_targets_per_option": getattr(args, "max_targets_per_option", None),
        }
        metadata["tokenizer"] = {
            "path": str(TOKENIZER_DIR),
            "modernbert_repo": MODERNBERT_REPO,
            "modernbert_revision": MODERNBERT_REVISION,
            "sha256": _hash_path_if_exists(TOKENIZER_DIR),
        }
        metadata["card_token_cache"] = {
            "path": str(cache_path),
            "sha256": _hash_path_if_exists(cache_path),
        }
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": serialized_args,
            "training_state": training_state,
            "metadata": metadata,
        },
        path,
    )


def _hash_path_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        return None
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
        "first strike damage": "fsd",
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
