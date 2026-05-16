#!/usr/bin/env python3
"""Train a PPO self-play policy against the mage-go Python engine."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import itertools
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from typing import Any, TextIO, cast

# Inductor cache: pin to a project-local dir so torch.compile output survives
# across runs. First run pays the full ~minutes compile cost; subsequent runs
# load .so / cached FX graphs from disk. Set BEFORE importing torch so
# inductor reads the right value when its config module initializes.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "torch_inductor"),
)
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Let Dynamo trace through nn.LSTM (cuDNN op stays opaque) instead of breaking
# the graph at every LSTM call. Without this the recurrent text policy is
# compiled as fragments around the LSTM, halving the speedup and doubling
# cold-compile time. Typed as Literal[False] in the stubs; setattr around it.
setattr(torch._dynamo.config, "allow_rnn", True)  # noqa: B010

load_dotenv()

import wandb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.actions import (  # noqa: E402
    OPTION_SCALAR_DIM,
    TARGET_SCALAR_DIM,
    ActionRequest,
    ActionTrace,
    PolicyStep,
    TraceKind,
)
from magic_ai.game_state import (  # noqa: E402
    GAME_INFO_DIM,
    ZONE_SLOT_COUNT,
    GameStateSnapshot,
    PendingOptionState,
    PendingState,
)
from magic_ai.model_state import is_actor_runtime_state_key  # noqa: E402
from magic_ai.native.policy_version import PolicyVersionManager  # noqa: E402
from magic_ai.native.sharded import (  # noqa: E402
    ShardedNativeBatchEncoder,
    ShardedNativeRolloutDriver,
)
from magic_ai.opponent_pool import (  # noqa: E402
    OpponentEntry,
    OpponentPool,
    SnapshotSchedule,
    build_opponent_policy,
    build_text_opponent_policy,
    distribute_games_by_recency,
    opponent_policy_state_dict,
    run_eval_matches,
    save_snapshot,
    snapshot_games_from_tag,
    snapshot_tag,
)
from magic_ai.ppo import ppo_update  # noqa: E402
from magic_ai.returns import gae_returns, gae_returns_batched  # noqa: E402
from magic_ai.rnad import RNaDConfig  # noqa: E402
from magic_ai.rnad_trainer import (  # noqa: E402
    EpisodeBatch,
    RNaDTrainerState,
    build_trainer_state,
    run_rnad_update,
)
from magic_ai.rollout import (  # noqa: E402
    PPOStats,
    RolloutStep,
    terminal_reward_for_finish,
)
from magic_ai.slot_encoder.buffer import NativeTrajectoryBuffer  # noqa: E402
from magic_ai.slot_encoder.game_state import GameStateEncoder  # noqa: E402
from magic_ai.slot_encoder.model import PPOPolicy  # noqa: E402
from magic_ai.slot_encoder.native_encoder import (  # noqa: E402,F401
    NativeBatchEncoder,
    NativeEncodingError,
)
from magic_ai.slot_encoder.native_rollout import (  # noqa: E402
    NativeRolloutDriver,  # noqa: F401
    NativeRolloutUnavailable,
)

# Inline-blank → grammar-decoder cutover (Session B):
#
# The IMPALA actor / inference-server pipeline now speaks the decoder wire
# shape (:class:`NativeTextDecoderBatch`) end-to-end. The four legacy
# inline-blank stubs that lived here (TextDecisionLayout,
# build_text_decision_layout, infer_text_trace_kind, _decode_text_action)
# have been removed:
#
# * ``TextDecisionLayout`` / ``build_text_decision_layout`` →
#   :class:`DecoderDecisionLayout` (sampled directly by
#   :meth:`LSTMStatefulTextPolicy.sample_batch`).
# * ``infer_text_trace_kind`` → :func:`_decision_type_to_trace_kind` defined
#   later in this module.
# * ``_decode_text_action`` → engine-side translation in Go via
#   ``mage.batch_step_by_decoder_action``; transcripts use
#   :func:`decode_decoder_action` from
#   :mod:`magic_ai.text_encoder.decoder_action`.
from magic_ai.text_encoder.card_cache import (  # noqa: E402
    DEFAULT_ORACLE_DB_PATH,
    CardTokenCache,
    build_card_cache,
    card_cache_is_current,
    fetch_registered_card_names_from_engine,
    load_card_cache,
    load_oracle_db,
    save_card_cache,
)
from magic_ai.text_encoder.decoder import GrammarDecoderConfig  # noqa: E402
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE  # noqa: E402
from magic_ai.text_encoder.lstm_stateful_text_policy import (  # noqa: E402
    LSTMStatefulTextPolicy,
)
from magic_ai.text_encoder.model import (  # noqa: E402
    DEFAULT_HF_ENCODER_MODEL,
    TextEncoderConfig,
    text_encoder_config_from_hf,
)
from magic_ai.text_encoder.native_assembler import DEFAULT_T_SPEC_MAX  # noqa: E402
from magic_ai.text_encoder.policy_value_pretrain import (  # noqa: E402
    ForgeChoiceDataset,
    ForgePolicyValueConfig,
    ForgePolicyValueTrainer,
    _batch_to_device,
    batches_per_epoch,
)
from magic_ai.text_encoder.recurrent import RecurrentTextPolicyConfig  # noqa: E402
from magic_ai.text_encoder.render import OracleEntry  # noqa: E402
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer  # noqa: E402
from magic_ai.text_encoder.tokenizer import (  # noqa: E402
    MAX_CARD_REFS,
    MODERNBERT_REPO,
    MODERNBERT_REVISION,
    TOKENIZER_DIR,
    load_tokenizer,
)

# Inline-blank pipeline used a fixed ``<num:k>`` token table; the decoder
# pipeline emits digits via the grammar so MAX_NUM is no longer a tokenizer
# constant. Keep a local upper bound for the few legacy rollout helpers in
# this script that haven't been ported to the decoder yet.
MAX_NUM = 64

DEFAULT_DECK = {
    "name": "bolt-mountain",
    "cards": [
        {"name": "Mountain", "count": 24},
        {"name": "Lightning Bolt", "count": 36},
    ],
}

DEFAULT_GAME_LOG_PATH = Path("/tmp/game_logs.txt")
DEFAULT_TEXT_MINIBATCH_TOKEN_LIMIT = 262_144


# Optional subphase profiling hook. The profile script binds this to a
# real recorder; in normal runs it stays a no-op (one Python attribute
# read + branch per call site, negligible).
_subphase_record: Callable[[str, float], None] | None = None


def _record_phase(name: str, t0: float) -> None:
    rec = _subphase_record
    if rec is not None:
        rec(name, time.perf_counter() - t0)


def _maybe_install_sync_debug() -> None:
    """Activate torch.cuda.set_sync_debug_mode and route warnings to stderr
    with a Python traceback, gated by ``MAGIC_AI_SYNC_DEBUG=warn|error``.

    Each unique (file, lineno) site fires once.
    """

    import os as _os
    import traceback as _tb
    import warnings as _warn

    mode = _os.environ.get("MAGIC_AI_SYNC_DEBUG", "").strip().lower()
    if mode not in {"warn", "error"}:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.set_sync_debug_mode(mode)
    _warn.simplefilter("always")
    _seen: set[tuple[str, int]] = set()

    def _show(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        del file, line
        key = (str(filename), int(lineno))
        if key in _seen:
            return
        _seen.add(key)
        print(
            f"\n[sync-debug:{category.__name__}] {filename}:{lineno}: {message}",
            flush=True,
        )
        _tb.print_stack(limit=30)

    _warn.showwarning = cast(Any, _show)
    print(
        f"[sync-debug] torch.cuda.set_sync_debug_mode({mode!r}) installed; "
        "Python traceback will be printed once per call site.",
        flush=True,
    )


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
    actor_id: int = -1
    behavior_policy_version: int = 0
    inference_policy_version: int = 0
    target_policy_version: int = -1


def _read_life_totals(game: Any) -> tuple[int, int]:
    """Refresh the engine's state cache and return (life_p0, life_p1).

    Life totals are clamped to ``>= 0`` by the consumer (the tiebreak helper
    in ``magic_ai.ppo``); this just extracts the raw integers from the
    snapshot dict.
    """
    try:
        game.refresh_state()
    except Exception:
        pass
    state = getattr(game, "state", None) or {}
    players = state.get("players", []) or []
    l0 = int(players[0].get("Life", 0)) if len(players) > 0 else 0
    l1 = int(players[1].get("Life", 0)) if len(players) > 1 else 0
    return l0, l1


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
    timeouts: int = 0

    def record(self, bucket: str) -> None:
        if bucket == "p1":
            self.p1_wins += 1
        elif bucket == "p2":
            self.p2_wins += 1
        elif bucket == "timeout":
            self.timeouts += 1
        else:
            self.draws += 1

    @property
    def total_games(self) -> int:
        return self.p1_wins + self.p2_wins + self.draws + self.timeouts

    def as_wandb_metrics(self) -> dict[str, float]:
        total = self.total_games
        if total == 0:
            return {}
        denom = float(total)
        return {
            "p1_win_fraction": self.p1_wins / denom,
            "p2_win_fraction": self.p2_wins / denom,
            "draw_fraction": self.draws / denom,
            "timeout_fraction": self.timeouts / denom,
            "window_games": float(total),
        }

    def reset(self) -> None:
        self.p1_wins = 0
        self.p2_wins = 0
        self.draws = 0
        self.timeouts = 0


@dataclass
class SlotTrainingBackend:
    policy: PPOPolicy
    native_encoder: ShardedNativeBatchEncoder
    staging_buffer: NativeTrajectoryBuffer
    batch_pool: ThreadPoolExecutor | None
    batch_workers: int


@dataclass
class TextTrainingBackend:
    policy: LSTMStatefulTextPolicy
    replay_buffer: TextReplayBuffer
    cache: CardTokenCache
    oracle: dict[str, OracleEntry]
    tokenizer: Any
    native_encoder: ShardedNativeBatchEncoder | None = None
    batch_pool: ThreadPoolExecutor | None = None
    batch_workers: int = 1


class NativeTextTrajectoryBuffer:
    """Per-env staging for the native batched IMPALA text decoder rollout.

    Actor replies are written into a tensor staging table. Each active env-step
    owns one staged row id; variable-length token streams live in a padded
    per-row slab and are compacted only when the finished episode is appended
    into the replay ring.
    """

    def __init__(
        self,
        replay_buffer: TextReplayBuffer,
        *,
        num_envs: int,
        max_steps: int,
        validate: bool = True,
    ) -> None:
        self._replay_buffer = replay_buffer
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.validate = bool(validate)
        self.device = replay_buffer.device
        self._lock = threading.Lock()
        self.timing_stats: Any | None = None
        shape = (self.num_envs, self.max_steps)
        self._row_id = torch.full(shape, -1, dtype=torch.long)
        self._token_length = torch.zeros(shape, dtype=torch.int32)
        self._perspective_player_idx = torch.zeros(shape, dtype=torch.int16)
        self._step_counts = torch.zeros((self.num_envs,), dtype=torch.int32)
        # Per-env aggregate token counts, kept in sync with the metadata table
        # so replay admission checks are O(number of envs being committed).
        self._token_counts = torch.zeros((self.num_envs,), dtype=torch.int64)
        self._staged_capacity = 0
        self._next_staged_row = 0
        self._free_row_ids = torch.empty((0,), dtype=torch.long)
        self._pin_staging = replay_buffer.device.type == "cuda" and torch.cuda.is_available()
        self._token_ids: torch.Tensor | None = None
        self._card_ref_positions: torch.Tensor | None = None
        self._decoder_decision_type: torch.Tensor | None = None
        self._decoder_output_token_ids: torch.Tensor | None = None
        self._decoder_output_pointer_pos: torch.Tensor | None = None
        self._decoder_output_is_pointer: torch.Tensor | None = None
        self._decoder_output_pad_mask: torch.Tensor | None = None
        self._decoder_log_probs: torch.Tensor | None = None
        self._decoder_value: torch.Tensor | None = None
        self._pointer_anchor_positions: torch.Tensor | None = None
        self._pointer_anchor_kinds: torch.Tensor | None = None
        self._pointer_anchor_subjects: torch.Tensor | None = None
        self._pointer_anchor_handles: torch.Tensor | None = None
        self._vocab_mask: torch.Tensor | None = None
        self._pointer_mask: torch.Tensor | None = None
        self._lstm_h_in: torch.Tensor | None = None
        self._lstm_c_in: torch.Tensor | None = None

    @property
    def step_count_host(self) -> list[int]:
        """Host mirror for compatibility with the slot-buffer surface."""

        return self._step_counts.tolist()

    def _new_staging_tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        fill: int | float | bool | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"dtype": dtype}
        if self._pin_staging:
            kwargs["pin_memory"] = True
        if fill is None:
            return torch.empty(shape, **kwargs)
        return torch.full(shape, fill, **kwargs)

    def _grow_staging_tensor(
        self,
        current: torch.Tensor | None,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        fill: int | float | bool | None = None,
    ) -> torch.Tensor:
        grown = self._new_staging_tensor(shape, dtype=dtype, fill=fill)
        if current is not None and int(current.shape[0]) > 0:
            grown[: int(current.shape[0])] = current
        return grown

    def _ensure_staged_capacity(self, needed: int, *, has_lstm: bool) -> None:
        if needed <= self._staged_capacity:
            if has_lstm and self._lstm_h_in is None:
                self._lstm_h_in = self._new_staging_tensor(
                    (
                        self._staged_capacity,
                        self._replay_buffer.recurrent_layers,
                        self._replay_buffer.recurrent_hidden_dim,
                    ),
                    dtype=torch.float32,
                    fill=0.0,
                )
                self._lstm_c_in = self._new_staging_tensor(
                    (
                        self._staged_capacity,
                        self._replay_buffer.recurrent_layers,
                        self._replay_buffer.recurrent_hidden_dim,
                    ),
                    dtype=torch.float32,
                    fill=0.0,
                )
            return
        max_rows = self.num_envs * self.max_steps
        new_capacity = min(max(needed, max(1024, self._staged_capacity * 2)), max_rows)
        if new_capacity < needed:
            raise RuntimeError("native text staging row capacity exhausted")

        replay = self._replay_buffer
        l_dec = replay.max_decoder_len
        n_anchor = replay.max_anchors
        self._token_ids = self._grow_staging_tensor(
            self._token_ids,
            (new_capacity, replay.max_tokens),
            dtype=torch.int32,
            fill=0,
        )
        self._card_ref_positions = self._grow_staging_tensor(
            self._card_ref_positions,
            (new_capacity, replay.max_card_refs),
            dtype=torch.int32,
            fill=-1,
        )
        self._decoder_decision_type = self._grow_staging_tensor(
            self._decoder_decision_type,
            (new_capacity,),
            dtype=torch.int32,
            fill=-1,
        )
        self._decoder_output_token_ids = self._grow_staging_tensor(
            self._decoder_output_token_ids,
            (new_capacity, l_dec),
            dtype=torch.int32,
            fill=0,
        )
        self._decoder_output_pointer_pos = self._grow_staging_tensor(
            self._decoder_output_pointer_pos,
            (new_capacity, l_dec),
            dtype=torch.int32,
            fill=-1,
        )
        self._decoder_output_is_pointer = self._grow_staging_tensor(
            self._decoder_output_is_pointer,
            (new_capacity, l_dec),
            dtype=torch.bool,
            fill=False,
        )
        self._decoder_output_pad_mask = self._grow_staging_tensor(
            self._decoder_output_pad_mask,
            (new_capacity, l_dec),
            dtype=torch.bool,
            fill=False,
        )
        self._decoder_log_probs = self._grow_staging_tensor(
            self._decoder_log_probs,
            (new_capacity, l_dec),
            dtype=torch.float32,
            fill=0.0,
        )
        self._decoder_value = self._grow_staging_tensor(
            self._decoder_value,
            (new_capacity,),
            dtype=torch.float32,
            fill=0.0,
        )
        self._pointer_anchor_positions = self._grow_staging_tensor(
            self._pointer_anchor_positions,
            (new_capacity, n_anchor),
            dtype=torch.int32,
            fill=-1,
        )
        self._pointer_anchor_kinds = self._grow_staging_tensor(
            self._pointer_anchor_kinds,
            (new_capacity, n_anchor),
            dtype=torch.int32,
            fill=-1,
        )
        self._pointer_anchor_subjects = self._grow_staging_tensor(
            self._pointer_anchor_subjects,
            (new_capacity, n_anchor),
            dtype=torch.int32,
            fill=-1,
        )
        self._pointer_anchor_handles = self._grow_staging_tensor(
            self._pointer_anchor_handles,
            (new_capacity, n_anchor),
            dtype=torch.int32,
            fill=-1,
        )
        self._vocab_mask = self._grow_staging_tensor(
            self._vocab_mask,
            (new_capacity, l_dec, GRAMMAR_VOCAB_SIZE),
            dtype=torch.bool,
            fill=False,
        )
        self._pointer_mask = self._grow_staging_tensor(
            self._pointer_mask,
            (new_capacity, l_dec, n_anchor),
            dtype=torch.bool,
            fill=False,
        )
        if has_lstm or self._lstm_h_in is not None:
            self._lstm_h_in = self._grow_staging_tensor(
                self._lstm_h_in,
                (new_capacity, replay.recurrent_layers, replay.recurrent_hidden_dim),
                dtype=torch.float32,
                fill=0.0,
            )
            self._lstm_c_in = self._grow_staging_tensor(
                self._lstm_c_in,
                (new_capacity, replay.recurrent_layers, replay.recurrent_hidden_dim),
                dtype=torch.float32,
                fill=0.0,
            )
        self._staged_capacity = new_capacity

    def _allocate_staged_rows_locked(self, n: int, *, has_lstm: bool) -> torch.Tensor:
        n = int(n)
        free_n = int(self._free_row_ids.numel())
        if free_n >= n:
            self._ensure_staged_capacity(self._staged_capacity, has_lstm=has_lstm)
            row_ids = self._free_row_ids[-n:].clone()
            self._free_row_ids = self._free_row_ids[:-n]
            return row_ids
        fresh_n = n - free_n
        start = self._next_staged_row
        end = start + fresh_n
        self._ensure_staged_capacity(end, has_lstm=has_lstm)
        fresh = torch.arange(start, end, dtype=torch.long)
        self._next_staged_row = end
        if free_n == 0:
            return fresh
        row_ids = torch.cat((self._free_row_ids, fresh), dim=0)
        self._free_row_ids = self._free_row_ids[:0]
        return row_ids

    def _release_staged_rows_locked(self, row_ids: torch.Tensor) -> None:
        valid = row_ids.to(dtype=torch.long)
        valid = valid[valid >= 0]
        if int(valid.numel()) > 0:
            self._free_row_ids = torch.cat((self._free_row_ids, valid), dim=0)

    # ------------------------------------------------------------------ stage

    def stage_batch(
        self,
        env_indices: list[int],
        host_decoder: Any,
        *,
        packed_parent: Any,
        perspective_player_indices: Sequence[int],
        lstm_h_in: torch.Tensor | None = None,
        lstm_c_in: torch.Tensor | None = None,
    ) -> None:
        """Append per-env decoder rows + their encoded snapshots.

        ``host_decoder`` is a :class:`DecoderHostView` carrying the full
        decoder batch on host. ``packed_parent`` is the matching
        multi-row :class:`PackedTextBatch` for the whole actor tick,
        host-side. LSTM state slices come from the inference reply already
        on host.
        """

        n = len(env_indices)
        if int(host_decoder.decision_type.shape[0]) != n:
            raise ValueError("host_decoder row count must match env_indices length")
        if int(packed_parent.seq_lengths.shape[0]) != n:
            raise ValueError("packed_parent row count must match env_indices length")
        if len(perspective_player_indices) != n:
            raise ValueError("perspective length must match env_indices length")

        seq_lengths = packed_parent.seq_lengths.to(dtype=torch.int32, device="cpu")
        token_offsets = torch.zeros((n,), dtype=torch.int32)
        if n > 1:
            token_offsets[1:] = seq_lengths.cumsum(0)[:-1]
        env_t = torch.as_tensor(env_indices, dtype=torch.long)
        perspective_t = torch.as_tensor(perspective_player_indices, dtype=torch.int16)
        if self.validate:
            replay = self._replay_buffer
            if int(seq_lengths.max().item()) > replay.max_tokens:
                raise ValueError("staged text row exceeds replay max_tokens")
            if int(host_decoder.output_token_ids.shape[1]) != replay.max_decoder_len:
                raise ValueError("decoder max length does not match replay buffer")
            if int(packed_parent.card_ref_positions.shape[1]) != replay.max_card_refs:
                raise ValueError("card-ref width does not match replay buffer")
            if int(packed_parent.pointer_anchor_positions.shape[1]) != replay.max_anchors:
                raise ValueError("pointer-anchor width does not match replay buffer")
            if int(host_decoder.vocab_mask.shape[2]) != GRAMMAR_VOCAB_SIZE:
                raise ValueError("vocab-mask width does not match grammar vocab")
            if int(host_decoder.pointer_mask.shape[2]) != replay.max_anchors:
                raise ValueError("pointer-mask width does not match replay buffer")

        with self._lock:
            if self.validate:
                if bool(((env_t < 0) | (env_t >= self.num_envs)).any().item()):
                    raise IndexError("env index out of range")
                if int(torch.unique(env_t).numel()) != n:
                    raise ValueError("env_indices must be unique within a staged batch")
                if bool((self._step_counts[env_t] >= self.max_steps).any().item()):
                    raise RuntimeError("native text staging buffer is full for at least one env")
            row_ids = self._allocate_staged_rows_locked(n, has_lstm=lstm_h_in is not None)
            step_t = self._step_counts[env_t].to(dtype=torch.long)
            self._row_id[env_t, step_t] = row_ids
            self._token_length[env_t, step_t] = seq_lengths
            self._perspective_player_idx[env_t, step_t] = perspective_t
            self._step_counts[env_t] += 1
            self._token_counts.index_add_(0, env_t, seq_lengths.to(dtype=torch.int64))
            self._write_staged_rows(
                row_ids=row_ids,
                packed_parent=packed_parent,
                decoder=host_decoder,
                token_offsets=token_offsets,
                seq_lengths=seq_lengths,
                lstm_h_in=lstm_h_in,
                lstm_c_in=lstm_c_in,
            )

    def _write_staged_rows(
        self,
        *,
        row_ids: torch.Tensor,
        packed_parent: Any,
        decoder: Any,
        token_offsets: torch.Tensor,
        seq_lengths: torch.Tensor,
        lstm_h_in: torch.Tensor | None,
        lstm_c_in: torch.Tensor | None,
    ) -> None:
        token_ids = self._token_ids
        card_ref_positions = self._card_ref_positions
        decoder_decision_type = self._decoder_decision_type
        decoder_output_token_ids = self._decoder_output_token_ids
        decoder_output_pointer_pos = self._decoder_output_pointer_pos
        decoder_output_is_pointer = self._decoder_output_is_pointer
        decoder_output_pad_mask = self._decoder_output_pad_mask
        decoder_log_probs = self._decoder_log_probs
        decoder_value = self._decoder_value
        pointer_anchor_positions = self._pointer_anchor_positions
        pointer_anchor_kinds = self._pointer_anchor_kinds
        pointer_anchor_subjects = self._pointer_anchor_subjects
        pointer_anchor_handles = self._pointer_anchor_handles
        vocab_mask = self._vocab_mask
        pointer_mask = self._pointer_mask
        if (
            token_ids is None
            or card_ref_positions is None
            or decoder_decision_type is None
            or decoder_output_token_ids is None
            or decoder_output_pointer_pos is None
            or decoder_output_is_pointer is None
            or decoder_output_pad_mask is None
            or decoder_log_probs is None
            or decoder_value is None
            or pointer_anchor_positions is None
            or pointer_anchor_kinds is None
            or pointer_anchor_subjects is None
            or pointer_anchor_handles is None
            or vocab_mask is None
            or pointer_mask is None
        ):
            raise RuntimeError("staging tensors were not allocated")

        max_len = int(seq_lengths.max().item())
        if max_len > 0:
            pos = torch.arange(max_len, dtype=torch.long)
            live = pos.unsqueeze(0) < seq_lengths.to(dtype=torch.long).unsqueeze(1)
            src = token_offsets.to(dtype=torch.long).unsqueeze(1) + pos.unsqueeze(0)
            src = torch.where(live, src, torch.zeros_like(src))
            values = packed_parent.token_ids[src].to(dtype=torch.int32)
            token_ids[row_ids.unsqueeze(1), pos.unsqueeze(0)] = values
        card_ref_positions[row_ids] = packed_parent.card_ref_positions.to(torch.int32)
        pointer_anchor_positions[row_ids] = packed_parent.pointer_anchor_positions.to(torch.int32)
        pointer_anchor_kinds[row_ids] = packed_parent.pointer_anchor_kinds.to(torch.int32)
        pointer_anchor_subjects[row_ids] = packed_parent.pointer_anchor_subjects.to(torch.int32)
        pointer_anchor_handles[row_ids] = packed_parent.pointer_anchor_handles.to(torch.int32)
        decoder_decision_type[row_ids] = decoder.decision_type.to(torch.int32)
        decoder_output_token_ids[row_ids] = decoder.output_token_ids.to(torch.int32)
        decoder_output_pointer_pos[row_ids] = decoder.output_pointer_pos.to(torch.int32)
        decoder_output_is_pointer[row_ids] = decoder.output_is_pointer.to(torch.bool)
        decoder_output_pad_mask[row_ids] = decoder.output_pad_mask.to(torch.bool)
        decoder_log_probs[row_ids] = decoder.log_probs.to(torch.float32)
        decoder_value[row_ids] = decoder.value.to(torch.float32)
        vocab_mask[row_ids] = decoder.vocab_mask.to(torch.bool)
        pointer_mask[row_ids] = decoder.pointer_mask.to(torch.bool)
        if lstm_h_in is not None and lstm_c_in is not None:
            if self._lstm_h_in is None or self._lstm_c_in is None:
                raise RuntimeError("recurrent staging tensors were not allocated")
            self._lstm_h_in[row_ids] = lstm_h_in.permute(1, 0, 2).to(torch.float32)
            self._lstm_c_in[row_ids] = lstm_c_in.permute(1, 0, 2).to(torch.float32)

    def reset_env(self, env_idx: int) -> None:
        with self._lock:
            env_i = int(env_idx)
            count = int(self._step_counts[env_i].item())
            if count > 0:
                self._release_staged_rows_locked(self._row_id[env_i, :count].clone())
            self._row_id[env_i].fill_(-1)
            self._token_length[env_i].zero_()
            self._perspective_player_idx[env_i].zero_()
            self._step_counts[env_i] = 0
            self._token_counts[env_i] = 0

    def active_step_count(self, env_idx: int) -> int:
        return int(self._step_counts[int(env_idx)].item())

    # ---------------------------------------------------------------- commit

    def append_envs_to_replay_returning_tensor(
        self,
        env_indices: list[int],
        replay_buffer: TextReplayBuffer,
        *,
        seal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Commit the staged steps for ``env_indices`` into the replay buffer.

        Returns ``(flat_rows, counts_per_env)`` on the buffer's device.
        After this returns, the staged slots for these envs are cleared.
        """

        del seal  # decoder commit always seals via commit_decoder_decision + commit
        device = replay_buffer.device
        timing_stats = self.timing_stats
        finish_part_start = time.perf_counter()
        with self._lock:
            if timing_stats is not None:
                timing_stats.add(
                    "finish_append_wait_lock",
                    time.perf_counter() - finish_part_start,
                )
            finish_part_start = time.perf_counter()
            env_t = torch.as_tensor(env_indices, dtype=torch.long)
            counts_cpu = self._step_counts[env_t].to(dtype=torch.long)
            counts_t = counts_cpu.to(device=device)
            total = int(counts_cpu.sum().item())
            if total == 0:
                return torch.zeros(0, dtype=torch.long, device=device), counts_t
            env_order = torch.repeat_interleave(
                torch.arange(int(env_t.numel()), dtype=torch.long),
                counts_cpu,
                output_size=total,
            )
            step_base = torch.repeat_interleave(
                counts_cpu.cumsum(0) - counts_cpu,
                counts_cpu,
                output_size=total,
            )
            step_t = torch.arange(total, dtype=torch.long) - step_base
            source_env = env_t[env_order]
            staged_row_ids = self._row_id[source_env, step_t].clone()
            token_lengths = self._token_length[source_env, step_t].clone()
            perspective_player_idx = self._perspective_player_idx[source_env, step_t].clone()
            self._row_id[env_t].fill_(-1)
            self._token_length[env_t].zero_()
            self._perspective_player_idx[env_t].zero_()
            self._step_counts[env_t] = 0
            self._token_counts[env_t] = 0
            if timing_stats is not None:
                timing_stats.add(
                    "finish_append_reset_slots",
                    time.perf_counter() - finish_part_start,
                )

        if (
            self._token_ids is None
            or self._card_ref_positions is None
            or self._decoder_decision_type is None
            or self._decoder_output_token_ids is None
            or self._decoder_output_pointer_pos is None
            or self._decoder_output_is_pointer is None
            or self._decoder_output_pad_mask is None
            or self._decoder_log_probs is None
            or self._decoder_value is None
            or self._pointer_anchor_positions is None
            or self._pointer_anchor_kinds is None
            or self._pointer_anchor_subjects is None
            or self._pointer_anchor_handles is None
            or self._vocab_mask is None
            or self._pointer_mask is None
        ):
            raise RuntimeError("staging tensors were not allocated")

        try:
            rows = replay_buffer.append_staged_decoder_rows_compact(
                staged_row_ids=staged_row_ids,
                staged_token_ids=self._token_ids,
                staged_card_ref_positions=self._card_ref_positions,
                staged_decision_type=self._decoder_decision_type,
                staged_output_token_ids=self._decoder_output_token_ids,
                staged_output_pointer_pos=self._decoder_output_pointer_pos,
                staged_output_is_pointer=self._decoder_output_is_pointer,
                staged_output_pad_mask=self._decoder_output_pad_mask,
                staged_log_probs=self._decoder_log_probs,
                staged_value=self._decoder_value,
                staged_pointer_anchor_positions=self._pointer_anchor_positions,
                staged_pointer_anchor_kinds=self._pointer_anchor_kinds,
                staged_pointer_anchor_subjects=self._pointer_anchor_subjects,
                staged_pointer_anchor_handles=self._pointer_anchor_handles,
                staged_vocab_mask=self._vocab_mask,
                staged_pointer_mask=self._pointer_mask,
                staged_lstm_h_in=self._lstm_h_in,
                staged_lstm_c_in=self._lstm_c_in,
                token_lengths=token_lengths,
                perspective_player_idx=perspective_player_idx,
                trace_kind_lut=_decision_type_trace_kind_lut(device),
                timing_stats=timing_stats,
            )
        finally:
            with self._lock:
                self._release_staged_rows_locked(staged_row_ids)
        return rows, counts_t

    def append_envs_to_replay(
        self,
        env_indices: list[int],
        replay_buffer: TextReplayBuffer,
    ) -> list[list[int]]:
        rows, counts = self.append_envs_to_replay_returning_tensor(env_indices, replay_buffer)
        if int(rows.numel()) == 0:
            return [[] for _ in env_indices]
        offsets = [0]
        for c in counts.detach().cpu().tolist():
            offsets.append(offsets[-1] + int(c))
        rows_h = rows.detach().cpu().tolist()
        return [
            [int(r) for r in rows_h[offsets[i] : offsets[i + 1]]] for i in range(len(env_indices))
        ]

    def append_env_to_replay(self, env_idx: int, replay_buffer: TextReplayBuffer) -> list[int]:
        return self.append_envs_to_replay([env_idx], replay_buffer)[0]

    def can_append_envs_to_replay(
        self,
        env_indices: list[int],
        replay_buffer: TextReplayBuffer,
    ) -> bool:
        """Check whether ``append_envs_to_replay`` would currently fit.

        Sums staged row + token counts for the proposed env slots and asks
        the replay buffer whether a reservation of that size is currently
        available. The learner uses this to defer FinishedEnv commits when
        the buffer is too full, without holding the per-env staging lock
        across the whole reservation attempt.
        """

        with self._lock:
            env_t = torch.as_tensor(env_indices, dtype=torch.long)
            total_rows = int(self._step_counts[env_t].sum().item())
            total_tokens = int(self._token_counts[env_t].sum().item())
        if total_rows == 0:
            return True
        return replay_buffer.can_reserve(
            row_count=total_rows, token_count=total_tokens, decision_count=0
        )


@dataclass
class TrainingRunResult:
    resume_state: TrainingResumeState
    rnad_state: RNaDTrainerState | None
    opponent_pool: OpponentPool | None = None
    snapshot_schedule: SnapshotSchedule | None = None
    retrospective_schedule: RetrospectiveLogSchedule | None = None


def build_slot_backend(args: argparse.Namespace, device: torch.device) -> SlotTrainingBackend:
    game_state_encoder = GameStateEncoder.from_embedding_json(args.embeddings, d_model=args.d_model)
    rollout_capacity = args.rollout_buffer_capacity or max(4096, 2 * args.rollout_steps)
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


def _resolve_grammar_decoder_config(
    args: argparse.Namespace,
    encoder_cfg: TextEncoderConfig,
) -> GrammarDecoderConfig:
    layers = getattr(args, "decoder_layers", None)
    if layers is None:
        layers = min(int(encoder_cfg.n_layers), GrammarDecoderConfig.n_layers)
    heads = getattr(args, "decoder_heads", None)
    if heads is None:
        head_cap = min(int(encoder_cfg.n_heads), GrammarDecoderConfig.n_heads)
        heads = next((h for h in range(head_cap, 0, -1) if encoder_cfg.d_model % h == 0), 1)
    d_ff = getattr(args, "decoder_d_ff", None)
    if d_ff is None:
        d_ff = int(encoder_cfg.d_ff)
    max_decode_len_arg = getattr(args, "decoder_max_decode_len", None)
    max_decode_len = int(
        GrammarDecoderConfig.max_decode_len if max_decode_len_arg is None else max_decode_len_arg
    )
    if int(encoder_cfg.d_model) % int(heads) != 0:
        raise ValueError(
            f"--decoder-heads ({heads}) must divide encoder d_model ({encoder_cfg.d_model})"
        )
    return GrammarDecoderConfig(
        d_model=int(encoder_cfg.d_model),
        n_layers=int(layers),
        n_heads=int(heads),
        d_ff=int(d_ff),
        max_decode_len=max_decode_len,
        compile_mask_update=bool(getattr(args, "compile_decoder_mask_update", False)),
    )


def build_text_backend(args: argparse.Namespace, device: torch.device) -> TextTrainingBackend:
    tokenizer = load_tokenizer()
    cache_path = Path(args.card_token_cache)
    oracle_db_path = Path(getattr(args, "oracle_db", DEFAULT_ORACLE_DB_PATH))
    registered_names = fetch_registered_card_names_from_engine()
    oracle = load_oracle_db(oracle_db_path, names=registered_names)
    if cache_path.exists():
        cache = load_card_cache(cache_path)
        if not card_cache_is_current(cache, registered_names, oracle):
            cache = build_card_cache(
                registered_names,
                oracle,
                tokenizer,
                oracle_db_path=oracle_db_path,
                missing_policy="warn",
            )
            save_card_cache(cache, cache_path)
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
    # The native assembler caps *state* tokens at ``text_max_tokens`` and
    # then appends up to ``DEFAULT_T_SPEC_MAX`` decision-spec tokens per row,
    # so the per-row combined length the encoder and replay buffer have to
    # absorb is ``text_max_tokens + DEFAULT_T_SPEC_MAX``. Size the RoPE
    # cache (and downstream replay storage) accordingly.
    combined_max_tokens = int(args.text_max_tokens) + DEFAULT_T_SPEC_MAX
    if getattr(args, "text_encoder_backend", "scratch") == "hf":
        cfg = text_encoder_config_from_hf(
            model_name=args.text_hf_model,
            revision=args.text_hf_revision,
            truncate_layers=args.text_hf_layers,
            vocab_size=len(tokenizer),
            pad_id=int(pad_id),
            max_seq_len=combined_max_tokens,
            trust_remote_code=args.text_hf_trust_remote_code,
        )
        if getattr(args, "skip_text_hf_init", False):
            cfg = replace(cfg, hf_model_name=None)
    else:
        cfg = TextEncoderConfig(
            vocab_size=len(tokenizer),
            pad_id=int(pad_id),
            d_model=args.text_d_model,
            n_layers=args.text_layers,
            n_heads=args.text_heads,
            d_ff=args.text_d_ff,
            max_seq_len=combined_max_tokens,
        )
    decoder_cfg = _resolve_grammar_decoder_config(args, cfg)
    recurrent_cfg = RecurrentTextPolicyConfig(
        encoder=cfg,
        lstm_hidden=cfg.d_model,
        compile_forward=bool(getattr(args, "torch_compile", False)),
        grammar_decoder_cfg=decoder_cfg,
    )
    policy = LSTMStatefulTextPolicy(recurrent_cfg).to(device)
    policy.init_lstm_env_states(args.num_envs)
    rollout_capacity = args.rollout_buffer_capacity or int(
        getattr(
            args,
            "replay_ring_capacity",
            max(6 * int(args.rollout_steps), 2 * int(args.num_envs)),
        )
    )
    # Replay buffer lives on CPU; the rollout finish path writes via host
    # memcpy (no per-row GPU dispatch storm), and gathered training batches
    # are H→D'd once per update by the consumer. Triton gather/append are
    # CUDA-only kernels — keep them off when storage is host-side.
    replay_buffer = TextReplayBuffer(
        capacity=rollout_capacity,
        max_tokens=combined_max_tokens,
        max_options=args.max_options,
        max_targets_per_option=args.max_targets_per_option,
        max_decision_groups=args.max_decision_groups,
        max_cached_choices=args.max_cached_choices,
        max_decoder_len=decoder_cfg.max_decode_len,
        recurrent_layers=recurrent_cfg.lstm_layers,
        recurrent_hidden_dim=cfg.d_model,
        lstm_proj_hidden=recurrent_cfg.lstm_hidden,
        device=torch.device("cpu"),
        validate=not getattr(args, "no_validate", False),
        materialize_gather_seq_id=True,
        use_triton_append=False,
        use_triton_gather=False,
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
            emit_render_plan=False,
            render_plan_capacity=4096,
            validate=not getattr(args, "no_validate", False),
            workers=batch_workers,
            pool=batch_pool,
            dedup_card_bodies=bool(getattr(args, "card_body_dedup", False)),
            shard_packed_tokens=bool(getattr(args, "shard_packed_tokens", False)),
        )
    backend = TextTrainingBackend(
        policy=policy,
        replay_buffer=replay_buffer,
        cache=cache,
        oracle=oracle,
        tokenizer=tokenizer,
        native_encoder=native_encoder,
        batch_pool=batch_pool,
        batch_workers=batch_workers,
    )
    # Register the closed-vocabulary token tables with the mage-go side so
    # the rollout hot path can dispatch through MageEncodeTokensPacked. Falls
    # back gracefully if the native lib doesn't have MageRegisterTokenTables
    # (older libmage.so), with a clear warning so users know to rebuild.
    if getattr(args, "text_native_assembler", True):
        try:
            from magic_ai.text_encoder.native_token_tables import (
                register_native_token_tables,
            )
            from magic_ai.text_encoder.token_tables import build_token_tables

            tables = build_token_tables(tokenizer, cache)
            register_native_token_tables(tables, tokenizer=tokenizer)
            print("native token assembler registered (MageEncodeTokensPacked path)")
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(
                f"warning: MageRegisterTokenTables unavailable ({exc}); "
                "native packed-token assembly is unavailable. Rebuild libmage.so to enable."
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
) -> list[PolicyStep]:
    """Encode snapshots, sample one decoder action per row, return PolicyStep list.

    Cutover note: the inline-blank tokens (``<chosen>`` / ``<yes>`` / ``<num:k>``
    …) are gone; the renderer is now decision-agnostic and the grammar decoder
    samples per-row token sequences. We translate each sampled sequence into
    an :class:`ActionRequest` via ``decode_decoder_action``.
    """

    from magic_ai.text_encoder.decoder_action import decode_decoder_action
    from magic_ai.text_encoder.policy import TextPolicy

    n = len(snapshots)
    if n == 0:
        return []
    if len(pendings) != n or len(env_indices) != n or len(perspective_player_indices) != n:
        raise ValueError("snapshots, pendings, env_indices, and players differ")

    encoded = TextPolicy.encode_snapshots(snapshots, backend.oracle, backend.tokenizer)
    if int(encoded.token_ids.shape[1]) > int(args.text_max_tokens):
        encoded = _truncate_text_batch(encoded, max_tokens=int(args.text_max_tokens))

    device = backend.policy.device
    moved = type(encoded)(
        token_ids=encoded.token_ids.to(device=device, dtype=torch.long),
        attention_mask=encoded.attention_mask.to(device=device, dtype=torch.long),
        card_ref_positions=encoded.card_ref_positions.to(device=device, dtype=torch.long),
        seq_lengths=encoded.seq_lengths.to(device=device, dtype=torch.long),
        spec_tokens=encoded.spec_tokens.to(device=device),
        spec_lens=encoded.spec_lens.to(device=device),
        decision_type=encoded.decision_type.to(device=device, dtype=torch.long),
        pointer_anchor_positions=encoded.pointer_anchor_positions.to(device=device),
        pointer_anchor_kinds=encoded.pointer_anchor_kinds.to(device=device),
        pointer_anchor_subjects=encoded.pointer_anchor_subjects.to(device=device),
        pointer_anchor_handles=encoded.pointer_anchor_handles.to(device=device),
        legal_edge_bitmap=(
            encoded.legal_edge_bitmap.to(device=device)
            if encoded.legal_edge_bitmap is not None
            else None
        ),
    )

    with torch.no_grad():
        sample = backend.policy.sample_batch(
            moved,
            env_indices=env_indices,
            perspective_player_indices=perspective_player_indices,
            deterministic=deterministic,
        )

    out: list[PolicyStep] = []
    for i, (pending, layout) in enumerate(zip(pendings, sample.decoded, strict=True)):
        action = decode_decoder_action(pending, layout)
        # Sum the per-step log probs over valid (pre-PAD) tokens for the row.
        per_step = sample.log_probs[i]
        pad = layout.output_pad_mask.to(device=per_step.device)
        log_prob = (per_step * pad.to(dtype=per_step.dtype)).sum().detach().cpu()
        trace = ActionTrace(
            kind=_decision_type_to_trace_kind(int(layout.decision_type)),
        )
        out.append(
            PolicyStep(
                action=action,
                trace=trace,
                log_prob=log_prob,
                value=torch.zeros((), dtype=torch.float32),
                entropy=torch.zeros((), dtype=torch.float32),
                replay_idx=None,
            )
        )
    return out


def _decision_type_to_trace_kind(decision_type: int) -> TraceKind:
    """Map a :class:`DecisionType` to the legacy :data:`TraceKind` string.

    Trace kinds are still used by transcript / sample-game logging machinery
    that predates the decoder cutover; we keep the surface stable and route
    decoder decisions to the closest legacy bucket.
    """

    from magic_ai.text_encoder.decision_spec import DecisionType

    if decision_type < 0:
        return "priority"
    dt = DecisionType(decision_type)
    table: dict[DecisionType, TraceKind] = {
        DecisionType.PRIORITY: "priority",
        DecisionType.DECLARE_ATTACKERS: "attackers",
        DecisionType.DECLARE_BLOCKERS: "blockers",
        DecisionType.CHOOSE_TARGETS: "choice_index",
        DecisionType.MAY: "may",
        DecisionType.CHOOSE_MODE: "choice_index",
        DecisionType.CHOOSE_X: "choice_index",
    }
    return table[dt]


def _decision_type_to_trace_kind_id(decision_type: int) -> int:
    """Numeric form of :func:`_decision_type_to_trace_kind` for replay storage."""

    from magic_ai.actions import TRACE_KIND_TO_ID

    return int(TRACE_KIND_TO_ID[cast(Any, _decision_type_to_trace_kind(decision_type))])


_DECISION_TYPE_TRACE_KIND_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _decision_type_trace_kind_lut(device: torch.device) -> torch.Tensor:
    """Device-resident LUT mapping ``DecisionType`` value → trace_kind_id.

    Built once per device. ``decision_type < 0`` (priority) is handled by
    the caller; this LUT is indexed with the clamped non-negative value.
    """

    cached = _DECISION_TYPE_TRACE_KIND_LUT_CACHE.get(device)
    if cached is not None:
        return cached
    from magic_ai.text_encoder.decision_spec import DecisionType

    max_dt = max(int(d) for d in DecisionType)
    host = torch.tensor(
        [_decision_type_to_trace_kind_id(i) for i in range(max_dt + 1)],
        dtype=torch.int64,
    )
    lut = host.to(device=device)
    _DECISION_TYPE_TRACE_KIND_LUT_CACHE[device] = lut
    return lut


def _single_token_id(tokenizer: Any, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if isinstance(tid, list):
        raise TypeError(f"convert_tokens_to_ids({token!r}) returned a list")
    return int(tid)


def _render_token_ids(tokenizer: Any) -> dict[str, int | list[int]]:
    return {
        "chosen": _single_token_id(tokenizer, "<chosen>"),
        "none": _single_token_id(tokenizer, "<none>"),
        "yes": _single_token_id(tokenizer, "<yes>"),
        "no": _single_token_id(tokenizer, "<no>"),
        "mulligan": _single_token_id(tokenizer, "<mulligan>"),
        "keep": _single_token_id(tokenizer, "<keep>"),
        "self": _single_token_id(tokenizer, "<self>"),
        "opp": _single_token_id(tokenizer, "<opp>"),
        "num": [_single_token_id(tokenizer, f"<num:{k}>") for k in range(MAX_NUM)],
        "mana": [
            _single_token_id(tokenizer, f"<mana:{s}>") for s in ("W", "U", "B", "R", "G", "C")
        ],
        "card_ref": [_single_token_id(tokenizer, f"<card-ref:{k}>") for k in range(MAX_CARD_REFS)],
    }


def _truncate_text_batch(batch: Any, *, max_tokens: int) -> Any:
    seq_lengths = batch.seq_lengths.clamp(max=max_tokens)
    attention_mask = batch.attention_mask[:, :max_tokens].clone()
    for row, seq_len in enumerate(seq_lengths.tolist()):
        attention_mask[row, int(seq_len) :] = 0
    card_ref_positions = batch.card_ref_positions.clone()
    card_ref_positions[card_ref_positions >= max_tokens] = -1
    blank_positions = batch.blank_positions.clone()
    blank_positions[blank_positions >= max_tokens] = -1
    blank_legal_mask = batch.blank_legal_mask.clone()
    if blank_positions.numel() > 0:
        blank_legal_mask = blank_legal_mask & (blank_positions.unsqueeze(-1) >= 0)
    return type(batch)(
        token_ids=batch.token_ids[:, :max_tokens].contiguous(),
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lengths,
        total_tokens=int(seq_lengths.sum().item()),
        seq_lengths_host=tuple(int(v) for v in seq_lengths.tolist()),
        blank_positions=blank_positions,
        blank_kind=batch.blank_kind,
        blank_group=batch.blank_group,
        blank_group_kind=batch.blank_group_kind,
        blank_option_index=batch.blank_option_index,
        blank_legal_ids=batch.blank_legal_ids,
        blank_legal_mask=blank_legal_mask,
    )


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
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"--checkpoint path does not exist: {path}")
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


def _model_checkpoint_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not is_actor_runtime_state_key(name)
    }


def _load_model_checkpoint_state(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    result = model.load_state_dict(
        {
            name: tensor
            for name, tensor in state_dict.items()
            if not is_actor_runtime_state_key(name)
        },
        strict=False,
    )
    missing = [name for name in result.missing_keys if not is_actor_runtime_state_key(name)]
    unexpected = [name for name in result.unexpected_keys if not is_actor_runtime_state_key(name)]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint state did not match model (missing={missing}, unexpected={unexpected})"
        )


def _is_post_mlm_checkpoint(checkpoint: dict[str, Any] | None) -> bool:
    return bool(_checkpoint_metadata(checkpoint).get("post_mlm", False))


def _checkpoint_wandb_run_id(checkpoint: dict[str, Any] | None) -> str | None:
    run_id = _checkpoint_metadata(checkpoint).get("wandb_run_id")
    return str(run_id) if isinstance(run_id, str) and run_id else None


def _checkpoint_has_policy(checkpoint: dict[str, Any] | None) -> bool:
    return isinstance(checkpoint, dict) and isinstance(checkpoint.get("policy"), dict)


def _should_run_mlm_pretrain(args: argparse.Namespace, checkpoint: dict[str, Any] | None) -> bool:
    return getattr(args, "pretrain_mlm_dir", None) is not None and not _checkpoint_has_policy(
        checkpoint
    )


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
    if "state_dict" in checkpoint and "policy" not in checkpoint:
        raise ValueError(
            "--checkpoint points to a policy snapshot, not a training checkpoint. "
            "R-NaD reg_mNNN.pt files are restored automatically from the main "
            "checkpoint's saved reg_snapshot_dir; resume with the main --output "
            "checkpoint instead."
        )
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
    saved_backend = text_config.get("text_encoder_backend", "scratch")
    requested_backend = getattr(args, "text_encoder_backend", "scratch")
    if saved_backend != requested_backend:
        raise ValueError(
            f"text checkpoint text_encoder_backend={saved_backend!r} is incompatible with "
            f"--text-encoder-backend {requested_backend!r}"
        )
    common_keys = (
        "text_max_tokens",
        "hidden_layers",
        "max_options",
        "max_targets_per_option",
    )
    scratch_keys = (
        "text_d_model",
        "text_layers",
        "text_heads",
        "text_d_ff",
    )
    hf_keys = (
        "text_hf_model",
        "text_hf_revision",
        "text_hf_layers",
    )
    keys = common_keys + (hf_keys if requested_backend == "hf" else scratch_keys)
    for key in keys:
        saved = text_config.get(key)
        requested = getattr(args, key, None)
        if saved != requested:
            raise ValueError(
                f"text checkpoint {key}={saved!r} is incompatible with "
                f"--{key.replace('_', '-')} {requested!r}"
            )
    decoder_keys = (
        "decoder_layers",
        "decoder_heads",
        "decoder_d_ff",
        "decoder_max_decode_len",
    )
    for key in decoder_keys:
        saved = text_config.get(key)
        requested = getattr(args, key, None)
        if saved is not None and requested is not None and saved != requested:
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
    if checkpoint is None:
        return OpponentPool()

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
            parsed_games = snapshot_games_from_tag(tag)
            pool.add_snapshot(
                snapshot_path,
                tag,
                snapshot_games=int(parsed_games) if parsed_games is not None else 0,
            )
            known_paths.add(resolved)
    return pool


def _prune_pool_to_schedule(
    pool: OpponentPool,
    schedule: SnapshotSchedule,
    completed_games: int,
) -> None:
    """Keep only entries closest to each not-yet-completed schedule threshold.

    When a run is extended by raising ``--episodes``, the schedule's 1% / 2N%
    thresholds shift to larger absolute game counts. Old snapshots that don't
    line up with the new grid are dropped (files are left on disk; only the
    pool's in-memory entry list is shrunk). Entries without a known
    ``snapshot_games`` (e.g. legacy tags) are kept untouched so we don't
    silently delete state we can't reason about.
    """
    candidates = [e for e in pool.entries if e.snapshot_games > 0]
    if not candidates:
        return
    legacy = [e for e in pool.entries if e.snapshot_games <= 0]
    fired_thresholds = [t for t in schedule.thresholds if t <= completed_games]
    if not fired_thresholds:
        pool.entries = legacy
        return
    chosen_paths: list[Path] = []
    seen: set[Path] = set()
    for threshold in fired_thresholds:
        best = min(candidates, key=lambda e: abs(e.snapshot_games - threshold))
        key = best.path.resolve() if best.path else Path(best.tag)
        if key in seen:
            continue
        seen.add(key)
        chosen_paths.append(key)
    chosen_set = set(chosen_paths)
    kept = [e for e in candidates if (e.path.resolve() if e.path else Path(e.tag)) in chosen_set]
    kept.sort(key=lambda e: e.snapshot_games)
    pool.entries = legacy + kept


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
        _load_model_checkpoint_state(target_module, target_sd)
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

    encoder_kind = getattr(args, "encoder", "slots")

    if encoder_kind == "text":
        if text_backend is None:
            raise ValueError("text_backend is required when --encoder text")

        opponent_pool, snapshot_schedule, retrospective_schedule = _build_opponent_schedules(
            args, checkpoint_cpu
        )
        opponent_policy: Any = None
        if opponent_pool is not None:
            opponent_policy = build_text_opponent_policy(text_backend.policy, device)

        if getattr(args, "native_render_plan", False):
            if text_backend.native_encoder is None:
                raise ValueError("text backend is missing native text encoder")
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
                opponent_pool=opponent_pool,
                snapshot_schedule=snapshot_schedule,
                retrospective_schedule=retrospective_schedule,
                opponent_policy=opponent_policy,
                resume_state=_resume_state_from_checkpoint(checkpoint_cpu),
                resume_checkpoint=checkpoint_cpu,
            )
        else:
            # Slow Python text path doesn't require native rollout for training,
            # but eval does. Skip eval if native_encoder is unavailable.
            resume_state, rnad_state = train_text_envs(
                args,
                mage,
                deck_pool,
                text_backend,
                optimizer,
                opponent_pool=opponent_pool,
                snapshot_schedule=snapshot_schedule,
                retrospective_schedule=retrospective_schedule,
                opponent_policy=opponent_policy,
                resume_state=_resume_state_from_checkpoint(checkpoint_cpu),
            )
        return TrainingRunResult(
            resume_state=resume_state,
            rnad_state=rnad_state,
            opponent_pool=opponent_pool,
            snapshot_schedule=snapshot_schedule,
            retrospective_schedule=retrospective_schedule,
        )

    if slot_backend is None:
        raise ValueError("slot_backend is required when --encoder slots")
    slot_policy = slot_backend.policy
    try:
        native_rollout = ShardedNativeRolloutDriver.for_mage(
            mage, workers=slot_backend.batch_workers, pool=batch_pool
        )
    except NativeRolloutUnavailable as exc:
        raise SystemExit(f"native rollout is unavailable: {exc}") from exc

    opponent_pool, snapshot_schedule, retrospective_schedule = _build_opponent_schedules(
        args, checkpoint_cpu
    )
    opponent_policy = None
    if opponent_pool is not None:
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


def _build_opponent_schedules(
    args: argparse.Namespace,
    checkpoint_cpu: dict[str, Any] | None,
) -> tuple[OpponentPool | None, SnapshotSchedule | None, RetrospectiveLogSchedule | None]:
    if getattr(args, "disable_opponent_pool", False):
        return None, None, None
    if _is_post_mlm_checkpoint(checkpoint_cpu):
        # Resuming from a post-policy/value-pretrain checkpoint: no RL games
        # have happened yet, so start the opponent pool fresh and ignore any
        # pre-existing snapshots in args.opponent_pool_dir.
        opponent_pool = OpponentPool()
    else:
        opponent_pool = _restore_opponent_pool(checkpoint_cpu, args.opponent_pool_dir)
    snapshot_schedule = SnapshotSchedule.build(args.episodes)
    retrospective_schedule = RetrospectiveLogSchedule.build(args.episodes)
    training_state = _training_state_dict(checkpoint_cpu)
    completed_games = int(training_state.get("completed_games", 0))
    # Derive next_idx from completed_games rather than the stored index:
    # when --episodes is raised on resume, the schedule's thresholds shift,
    # so the previously-saved next_idx no longer points at the right slot.
    snapshot_schedule.next_idx = sum(
        1 for t in snapshot_schedule.thresholds if t <= completed_games
    )
    retrospective_schedule.next_idx = sum(
        1 for t in retrospective_schedule.thresholds if t <= completed_games
    )
    _prune_pool_to_schedule(opponent_pool, snapshot_schedule, completed_games)
    return opponent_pool, snapshot_schedule, retrospective_schedule


class _RLLRWarmup:
    """Linear LR warmup over the first ``warmup_updates`` RL updates.

    Sets ``param_group['lr']`` to ``base_lr * min(1, count/warmup_updates)``
    on construction and on every :meth:`step` call. ``warmup_updates == 0``
    disables warmup (no-op). Counted per RL update (one ``ppo_update`` /
    ``run_rnad_update`` call), not per minibatch step.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, base_lr: float, warmup_updates: int
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_updates = max(0, int(warmup_updates))
        self.count = 0
        self._apply()

    def _apply(self) -> None:
        if self.warmup_updates == 0:
            return
        factor = min(1.0, (self.count + 1) / self.warmup_updates)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr * factor

    def step(self) -> None:
        if self.warmup_updates == 0 or self.count >= self.warmup_updates:
            return
        self.count += 1
        self._apply()


def attach_rl_lr_warmup(
    optimizer: torch.optim.Optimizer, base_lr: float, warmup_updates: int
) -> None:
    """Install a warmup state on ``optimizer`` so call sites can call :func:`rl_lr_warmup_step`."""

    setattr(optimizer, "_rl_lr_warmup", _RLLRWarmup(optimizer, base_lr, warmup_updates))


def rl_lr_warmup_step(optimizer: torch.optim.Optimizer) -> None:
    """Advance the warmup counter if one was attached. No-op otherwise."""

    state: _RLLRWarmup | None = getattr(optimizer, "_rl_lr_warmup", None)
    if state is not None:
        state.step()


def _maybe_log_and_eval(
    *,
    sequenced: bool,
    stats: dict[str, float],
    epoch: int,
    global_step: int,
    log_now: bool,
    eval_ds: Any,
    eval_every: int,
    trainer: Any,
    eval_rng: Any,
    device: torch.device,
    amp_ctx: Any,
    args: argparse.Namespace,
) -> None:
    if log_now:
        if sequenced:
            print(
                f"[policy-value] epoch={epoch} step={global_step:6d} "
                f"loss={stats['loss']:.4f} policy={stats['policy_loss']:.4f} "
                f"value={stats['value_loss']:.4f} "
                f"value_sign={stats['value_sign_accuracy']:.3f} "
                f"v_corr={stats['v_corr']:+.3f} "
                f"v_pred={stats['v_pred_mean']:+.3f}±{stats['v_pred_std']:.3f} "
                f"v_targ={stats['v_targ_mean']:+.3f}±{stats['v_targ_std']:.3f} "
                f"v_head_gn={stats['v_head_grad_norm']:.4f} "
                f"lstm_gn={stats['lstm_grad_norm']:.3f} "
                f"n_loss={stats['n_loss_cells']} "
                f"grad_norm={stats['grad_norm']:.3f}"
            )
        else:
            print(
                f"[policy-value] epoch={epoch} step={global_step:6d} "
                f"loss={stats['loss']:.4f} policy={stats['policy_loss']:.4f} "
                f"value={stats['value_loss']:.4f} "
                f"step_acc={stats['decoder_step_accuracy']:.3f} "
                f"value_sign={stats['value_sign_accuracy']:.3f} "
                f"v_corr={stats['v_corr']:+.3f} "
                f"v_pred={stats['v_pred_mean']:+.3f}±{stats['v_pred_std']:.3f} "
                f"v_targ={stats['v_targ_mean']:+.3f}±{stats['v_targ_std']:.3f} "
                f"v_head_gn={stats['v_head_grad_norm']:.4f} "
                f"grad_norm={stats['grad_norm']:.3f}"
            )
        if not args.no_wandb and wandb.run is not None:
            wandb.log(
                {
                    **{f"pretrain_policy_value/{k}": v for k, v in stats.items()},
                    "pretrain_policy_value/step": global_step,
                }
            )
    if eval_ds is not None and eval_every > 0 and global_step > 0 and global_step % eval_every == 0:
        with amp_ctx:
            if sequenced:
                eval_stats = trainer.evaluate_sequenced(eval_ds, eval_rng, device=device)
            else:
                eval_stats = trainer.evaluate(eval_ds, eval_rng, device=device)
        train_v = stats.get("value_loss", 0.0)
        eval_v = eval_stats.get("eval_value_loss", 0.0)
        print(
            f"[policy-value] eval step={global_step} "
            f"policy={eval_stats.get('eval_policy_loss', 0.0):.4f} "
            f"value={eval_v:.4f} "
            f"value_sign={eval_stats.get('eval_value_sign_accuracy', 0.0):.3f} "
            f"train-eval value gap={train_v - eval_v:+.4f} "
            f"batches={int(eval_stats['eval_batches'])}"
        )
        if not args.no_wandb and wandb.run is not None:
            wandb.log(
                {
                    **{f"pretrain_policy_value/{k}": v for k, v in eval_stats.items()},
                    "pretrain_policy_value/train_minus_eval_value": train_v - eval_v,
                    "pretrain_policy_value/step": global_step,
                }
            )


def run_mlm_pretrain(
    args: argparse.Namespace,
    text_backend: TextTrainingBackend,
    device: torch.device,
) -> None:
    """Pretrain the text policy/value heads on extracted Forge choices.

    ``--pretrain-mlm-dir`` now points at a sharded torch, JSONL/JSONL.GZ, or
    Arrow artifact produced by ``scripts/extract_forge_choice_situations.py``
    / ``rust/forge_extract``. The historical flag name is kept so checkpoint
    and launch scripts do not need a second pretrain phase; the objective is no
    longer MLM. Each row trains the grammar-decoder target for the observed
    choice and a runtime-selected value target derived from the game result.
    """

    tokenizer = text_backend.tokenizer
    text_policy = text_backend.policy.policy
    pad_id = int(tokenizer.pad_token_id)

    cfg = ForgePolicyValueConfig(
        data_path=args.pretrain_mlm_dir,
        batch_size=args.pretrain_mlm_batch_size,
        max_tokens=args.text_max_tokens,
        eval_fraction=args.pretrain_mlm_eval_fraction,
        gamma=args.gamma,
        value_target_mode=args.pretrain_mlm_value_target,
        value_loss_weight=args.pretrain_mlm_value_loss_weight,
        policy_loss_weight=args.pretrain_mlm_policy_loss_weight,
        pad_token_id=pad_id,
        sequence_mode=args.pretrain_mlm_sequence_mode,
        games_per_batch=args.pretrain_mlm_games_per_batch,
        loss_positions_per_game=args.pretrain_mlm_loss_positions_per_game,
        max_decisions_per_game=args.pretrain_mlm_max_decisions_per_game,
    )
    train_ds = ForgeChoiceDataset(
        cfg,
        tokenizer=tokenizer,
        oracle=text_backend.oracle,
        split="train",
    )
    eval_ds: ForgeChoiceDataset | None = None
    if cfg.eval_fraction > 0:
        try:
            eval_ds = ForgeChoiceDataset(
                cfg,
                tokenizer=tokenizer,
                oracle=text_backend.oracle,
                split="eval",
            )
        except ValueError:
            eval_ds = None

    batches = batches_per_epoch(train_ds.n_examples, cfg.batch_size)
    print(
        "[policy-value] "
        f"examples={train_ds.n_examples} batches/epoch={batches} "
        f"epochs={args.pretrain_mlm_epochs} kinds={train_ds.kind_counts()} "
        f"value_target={cfg.value_target_mode}"
    )
    if batches == 0:
        raise ValueError(
            f"Forge choice corpus has {train_ds.n_examples} examples; "
            f"need at least batch_size={cfg.batch_size}"
        )
    if eval_ds is not None:
        print(
            f"[policy-value] eval_examples={eval_ds.n_examples} eval_kinds={eval_ds.kind_counts()}"
        )

    if args.torch_compile:
        _compiled_forward: Any = torch.compile(text_policy.forward, dynamic=True)
        object.__setattr__(text_policy, "forward", _compiled_forward)

    trainer = ForgePolicyValueTrainer(
        text_policy,
        cfg,
        lr=args.pretrain_mlm_lr,
        grad_clip=args.pretrain_mlm_grad_clip,
    )
    np_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)

    log_every = max(1, args.pretrain_mlm_log_every)
    eval_every = max(0, args.pretrain_mlm_eval_every)
    use_amp = bool(args.pretrain_mlm_amp) and device.type == "cuda"
    if use_amp:
        amp_ctx: Any = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_ctx = contextlib.nullcontext()
    if not args.no_wandb and wandb.run is not None:
        wandb.define_metric("pretrain_policy_value/step")
        wandb.define_metric("pretrain_policy_value/*", step_metric="pretrain_policy_value/step")
    sequenced = cfg.sequence_mode == "full"
    grad_accum = max(1, int(args.pretrain_mlm_grad_accum))
    if sequenced:
        from magic_ai.text_encoder.policy_value_pretrain import (
            _sequenced_batch_to_device,
        )
    global_step = 0
    for epoch in range(args.pretrain_mlm_epochs):
        if sequenced:
            iterator = train_ds.iter_epoch_games(
                cfg.games_per_batch, cfg.loss_positions_per_game, np_rng
            )
            accum_buf: list[Any] = []
            for host_seq_batch in iterator:
                accum_buf.append(host_seq_batch)
                if len(accum_buf) < grad_accum:
                    continue
                log_now = global_step % log_every == 0
                last_stats: dict[str, float] = {}
                for accum_idx, micro_host in enumerate(accum_buf):
                    micro = _sequenced_batch_to_device(micro_host, device)
                    with amp_ctx:
                        last_stats = trainer.sequenced_step(
                            micro,
                            compute_stats=log_now and accum_idx == grad_accum - 1,
                            accum_index=accum_idx,
                            accum_total=grad_accum,
                        )
                accum_buf = []
                _maybe_log_and_eval(
                    sequenced=True,
                    stats=last_stats,
                    epoch=epoch,
                    global_step=global_step,
                    log_now=log_now,
                    eval_ds=eval_ds,
                    eval_every=eval_every,
                    trainer=trainer,
                    eval_rng=eval_rng,
                    device=device,
                    amp_ctx=amp_ctx,
                    args=args,
                )
                global_step += 1
        else:
            for host_iid_batch in train_ds.iter_epoch(cfg.batch_size, np_rng):
                iid_batch = _batch_to_device(host_iid_batch, device)
                log_now = global_step % log_every == 0
                with amp_ctx:
                    stats = trainer.step(iid_batch, compute_stats=log_now)
                _maybe_log_and_eval(
                    sequenced=False,
                    stats=stats,
                    epoch=epoch,
                    global_step=global_step,
                    log_now=log_now,
                    eval_ds=eval_ds,
                    eval_every=eval_every,
                    trainer=trainer,
                    eval_rng=eval_rng,
                    device=device,
                    amp_ctx=amp_ctx,
                    args=args,
                )
                global_step += 1
    print(f"[policy-value] finished epochs={args.pretrain_mlm_epochs} steps={global_step}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    initialize_game_log(game_log_path(args))
    _maybe_install_sync_debug()
    trace_path = priority_trace_jsonl_path(args)
    if trace_path is not None:
        initialize_game_log(trace_path)
    if args.learning_rate is None:
        args.learning_rate = 5e-5 if args.trainer == "rnad" else 3e-4
    deck_pool = load_deck_pool(args.deck_json, args.deck_dir, args.jumpstart_dir)
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
        args.skip_text_hf_init = _checkpoint_has_policy(checkpoint_cpu)
        text_backend = build_text_backend(args, device)
        policy = text_backend.policy
        batch_pool = getattr(text_backend, "batch_pool", None)
    # Paper §199: R-NaD uses Adam with b1=0.0 (no momentum). This is
    # load-bearing for stability: nonzero b1 lets policy updates accumulate
    # directional drift across batches, which combined with NeuRD's raw-logit
    # gradient and the [-beta, beta] gate causes logit saturation and
    # frozen policies. PPO keeps the standard b1=0.9 default.
    fused_optim = device.type == "cuda"
    if args.trainer == "rnad":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(0.0, 0.999),
            eps=1e-8,
            fused=fused_optim,
        )
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, fused=fused_optim)
    checkpoint = load_training_checkpoint(args.checkpoint, map_location=device)
    if checkpoint is not None:
        _load_model_checkpoint_state(policy, checkpoint["policy"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    resumed_post_mlm = _is_post_mlm_checkpoint(checkpoint_cpu)
    resumed_policy_checkpoint = _checkpoint_has_policy(checkpoint_cpu)
    if resumed_post_mlm:
        print(
            "[policy-value] resuming from post-policy/value-pretrain checkpoint "
            f"{args.checkpoint}: skipping pretraining, opponent pool starts empty"
        )
    elif resumed_policy_checkpoint and getattr(args, "pretrain_mlm_dir", None) is not None:
        print(
            "[resume] checkpoint contains saved policy weights: skipping policy/value pretraining"
        )
    run_mlm_now = _should_run_mlm_pretrain(args, checkpoint_cpu)
    if run_mlm_now:
        if encoder_kind != "text" or text_backend is None:
            raise ValueError("--pretrain-mlm-dir requires --encoder text")
        run_mlm_pretrain(args, text_backend, device)
    if run_mlm_now or resumed_post_mlm:
        # Reset optimizer state: pretraining used a separate AdamW, so RL
        # moments must start fresh.
        if args.trainer == "rnad":
            optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=args.learning_rate,
                betas=(0.0, 0.999),
                eps=1e-8,
            )
        else:
            optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        if run_mlm_now and not args.no_post_mlm_checkpoint:
            post_mlm_path = args.post_mlm_checkpoint or (run_artifact_dir / "post_mlm.pt")
            save_checkpoint(
                post_mlm_path,
                policy,
                optimizer,
                args,
                opponent_pool=None,
                snapshot_schedule=None,
                retrospective_schedule=None,
                resume_state=None,
                wandb_run_id=active_wandb_run_id,
                run_artifact_dir=run_artifact_dir,
                rnad_state=None,
                post_mlm=True,
            )
            print(
                f"[policy-value] saved post-pretrain checkpoint -> {post_mlm_path} "
                f"(resume RL with --checkpoint {post_mlm_path})"
            )
        # Linear LR warmup over the first N RL updates to absorb the
        # pretrained-encoder × random-RL-heads mismatch. AdamW's first step
        # bias correction would otherwise land a sign-of-grad sized step that
        # saturates the LSTM/projections after supervised pretraining.
        attach_rl_lr_warmup(optimizer, args.learning_rate, args.rl_warmup_updates)

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
        default="text",
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
        "--jumpstart-dir",
        type=Path,
        default=None,
        help="directory of jumpstart-pack JSON files; each player's deck is "
        "two packs sampled from the directory and concatenated",
    )
    parser.add_argument(
        "--trainer",
        choices=("ppo", "rnad"),
        default="rnad",
        help="training algorithm: 'ppo' or 'rnad' (default, Regularized Nash "
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
        default=0.02,
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
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument(
        "--rollout-min-ready-batch",
        type=int,
        default=1,
        help="if more envs are still running, wait before policy inference until at "
        "least this many envs are ready; default 1 preserves immediate stepping",
    )
    parser.add_argument(
        "--rollout-ready-wait-ms",
        type=float,
        default=0.0,
        help="sleep this many milliseconds when waiting for --rollout-min-ready-batch; "
        "default 0 disables coalescing waits",
    )
    parser.add_argument(
        "--rollout-buffer-capacity",
        type=int,
        default=None,
        help=(
            "rows in the rollout GPU buffer; text default max(4096, "
            "3 * rollout-steps, max-steps-per-game). "
            "If a finished env's trajectory would overflow this cap, the "
            "actor coordinator defers it (keeps its staging slot occupied) "
            "until the next PPO update resets the buffer, instead of OOMing "
            "or evicting in-flight data."
        ),
    )
    parser.add_argument(
        "--num-rollout-actors",
        type=int,
        default=4,
        help=(
            "spawn this many CPU rollout actor threads sharing one GPU inference "
            "server (IMPALA-style). 1 = legacy in-line loop. >1 partitions envs "
            "across actors and dynamically batches their requests on the GPU. "
            "Default 4 fits an 8-CPU box (4 actors + 1 server + coordinator)."
        ),
    )
    parser.add_argument(
        "--inference-ready-fraction",
        type=float,
        default=0.45,
        help=(
            "actor-path inference server: launch a forward pass once this "
            "fraction of --num-envs rows is queued; tails flush explicitly"
        ),
    )
    parser.add_argument(
        "--inference-max-batch",
        type=int,
        default=1024,
        help=(
            "inference server: cap merged forward-batch row count; 0 = use --num-envs as the cap"
        ),
    )
    parser.add_argument("--max-policy-lag", type=int, default=2)
    parser.add_argument("--learner-min-rows", type=int, default=None)
    parser.add_argument("--learner-target-rows", type=int, default=None)
    parser.add_argument("--learner-max-rows", type=int, default=None)
    parser.add_argument("--replay-ring-capacity", type=int, default=None)
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help=(
            "print fixed-seed steady-state IMPALA timing windows after warmup; "
            "use with explicit --episodes/--seed for repeatable throughput runs"
        ),
    )
    parser.add_argument("--benchmark-warmup-updates", type=int, default=1)
    parser.add_argument("--benchmark-steady-updates", type=int, default=5)
    parser.add_argument(
        "--actor-watchdog-seconds",
        type=float,
        default=60.0,
        help=(
            "in actor mode, print server/queue/actor diagnostic state if no "
            "rollout progress is observed for this many seconds; helps catch "
            "hangs at startup instead of leaving them silent"
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
        default=5,
        help="parallel worker threads for Go-side engine step/encode batches "
        "(default: 5 = 4 actors + 1 dedicated snapshot-eval encoder; cgo "
        "releases the GIL so N threads run in parallel). Must be >= "
        "--num-rollout-actors + 1 when using the actor path so snapshot eval "
        "owns its own encoder/driver and never aliases an actor's scratch pool.",
    )
    parser.add_argument(
        "--shard-packed-tokens",
        action="store_true",
        help=(
            "experimentally shard native packed-token assembly across --batch-workers; "
            "off by default because current benchmarks show merge overhead dominates"
        ),
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
    parser.add_argument("--text-max-tokens", type=int, default=1536)
    parser.add_argument("--text-d-model", type=int, default=128)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--text-heads", type=int, default=4)
    parser.add_argument("--text-d-ff", type=int, default=512)
    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=None,
        help="grammar decoder transformer layers; default min(--text-layers/HF layers, 2)",
    )
    parser.add_argument(
        "--decoder-heads",
        type=int,
        default=None,
        help="grammar decoder attention heads; default largest encoder-compatible head count <= 4",
    )
    parser.add_argument(
        "--decoder-d-ff",
        type=int,
        default=None,
        help="grammar decoder feed-forward width; default matches the resolved text encoder d_ff",
    )
    parser.add_argument(
        "--decoder-max-decode-len",
        type=int,
        default=GrammarDecoderConfig.max_decode_len,
        help="maximum autoregressive decoder steps before per-decision-type shortening",
    )
    parser.add_argument(
        "--compile-decoder-mask-update",
        action="store_true",
        help=(
            "torch.compile the grammar-mask state update; faster steady-state, slower first sample"
        ),
    )
    parser.add_argument(
        "--text-encoder-backend",
        choices=("scratch", "hf"),
        default="scratch",
        help="text encoder trunk source: scratch keeps the local lightweight "
        "encoder; hf loads --text-hf-model and grows its embedding table",
    )
    parser.add_argument(
        "--text-hf-model",
        default=DEFAULT_HF_ENCODER_MODEL,
        help="Hugging Face encoder checkpoint for --text-encoder-backend=hf",
    )
    parser.add_argument(
        "--text-hf-revision",
        default=None,
        help="optional Hugging Face revision for --text-hf-model",
    )
    parser.add_argument(
        "--text-hf-layers",
        type=int,
        default=None,
        help="keep only the first N checkpoint layers; default uses all layers",
    )
    parser.add_argument(
        "--text-hf-trust-remote-code",
        action="store_true",
        help="pass trust_remote_code=True when loading the HF text encoder",
    )
    parser.add_argument(
        "--pretrain-mlm-dir",
        type=Path,
        default=None,
        help="if set, run joint policy/value pretraining over the Forge choice "
        "sharded torch snapshot directory, JSONL.GZ artifact, or Arrow corpus "
        "produced by scripts/extract_forge_choice_situations.py / rust/forge_extract "
        "before starting RL training. Historical flag name retained. Requires "
        "--encoder text.",
    )
    parser.add_argument(
        "--pretrain-mlm-epochs",
        type=int,
        default=1,
        help="number of full passes over the Forge choice corpus",
    )
    parser.add_argument(
        "--rl-warmup-updates",
        type=int,
        default=50,
        help="number of RL updates over which to linearly ramp the learning "
        "rate from 0 to --learning-rate. Active only when --pretrain-mlm-dir "
        "is set; absorbs the pretrained-encoder × random-RL-heads mismatch "
        "that otherwise overflows on the first PPO step. Set to 0 to disable.",
    )
    parser.add_argument(
        "--post-mlm-checkpoint",
        type=Path,
        default=None,
        help="path to save the post-pretrain checkpoint (policy weights + "
        "`post_mlm` metadata flag) once policy/value pretraining completes. "
        "Defaults to <run_artifact_dir>/post_mlm.pt. Pass this path back as "
        "--checkpoint to skip pretraining (opponent pool starts empty). "
        "Use --no-post-mlm-checkpoint to disable.",
    )
    parser.add_argument(
        "--no-post-mlm-checkpoint",
        action="store_true",
        help="disable the automatic post-pretrain checkpoint save.",
    )
    parser.add_argument("--pretrain-mlm-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-mlm-lr", type=float, default=2e-4)
    parser.add_argument("--pretrain-mlm-grad-clip", type=float, default=1.0)
    parser.add_argument("--pretrain-mlm-log-every", type=int, default=50)
    parser.add_argument(
        "--pretrain-mlm-eval-every",
        type=int,
        default=500,
        help="run evaluation on the held-out shard every N training steps (0 disables)",
    )
    parser.add_argument(
        "--pretrain-mlm-eval-fraction",
        type=float,
        default=0.05,
        help="fraction of games to hold out for evaluation (deterministic via "
        "hash(game_id) %% bucket); 0 disables the eval split",
    )
    parser.add_argument(
        "--pretrain-mlm-value-target",
        choices=("terminal", "gae", "vtrace"),
        default="terminal",
        help="runtime value target semantics for Forge outcomes; with one extracted "
        "choice per game, gae/vtrace use a discounted terminal sign proxy.",
    )
    parser.add_argument(
        "--pretrain-mlm-sequence-mode",
        choices=("none", "full"),
        default="full",
        help="full: replay each game through the LSTM in temporal order; loss "
        "fires on a sampled subset of cells. none: legacy IID per-record path "
        "that bypasses the LSTM.",
    )
    parser.add_argument(
        "--pretrain-mlm-games-per-batch",
        type=int,
        default=2,
        help="number of complete games per *micro*-batch in sequenced pretrain "
        "mode (one encoder forward). Effective batch is "
        "games_per_batch × pretrain_mlm_grad_accum.",
    )
    parser.add_argument(
        "--pretrain-mlm-grad-accum",
        type=int,
        default=1,
        help="number of micro-batches per optimizer step in sequenced pretrain "
        "mode. Lets you scale effective batch above what fits in GPU memory.",
    )
    parser.add_argument(
        "--pretrain-mlm-loss-positions-per-game",
        type=int,
        default=16,
        help="cells per game that count toward decoder + value losses in "
        "sequenced pretrain mode (sampled uniformly without replacement).",
    )
    parser.add_argument(
        "--pretrain-mlm-max-decisions-per-game",
        type=int,
        default=64,
        help="hard cap on decision points per game in sequenced pretrain "
        "mode; longer games are truncated to the first N decisions.",
    )
    parser.add_argument("--pretrain-mlm-value-loss-weight", type=float, default=10.0)
    parser.add_argument("--pretrain-mlm-policy-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--pretrain-mlm-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train the autoregressive grammar decoder (decoder pipeline) "
            "instead of the legacy inline-blank policy. Requires the policy "
            "to be constructed with use_grammar_decoder=True."
        ),
    )
    parser.add_argument(
        "--pretrain-mlm-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run policy/value pretraining under torch.autocast(bf16) on CUDA. No-op on CPU.",
    )
    parser.add_argument(
        "--native-text-rollout",
        "--native-render-plan",
        dest="native_render_plan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use mage-go native batched text rollouts for --encoder text "
            "(default: on; pass --no-native-text-rollout to use the older "
            "Python text rollout path). The old --native-render-plan spelling "
            "is kept as an alias."
        ),
    )
    parser.add_argument(
        "--card-body-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable card-body deduplication in the native packed-token assembler.",
    )
    parser.add_argument(
        "--text-native-assembler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the text-encoder assembler natively in mage-go via "
        "MageEncodeTokensPacked (default: on)",
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
        default=32,
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
    # Engine-declared draws are rare in MTG (true draws); step-cap timeouts
    # are now resolved by a life-total tiebreak rather than being lumped in
    # with draws, so the draw branch only fires on genuine simultaneous-loss
    # / poison / decked-out situations. Default 0.0 ⇒ engine draws contribute
    # no terminal reward to either player.
    parser.add_argument(
        "--draw-penalty",
        type=float,
        default=0.0,
        help=(
            "terminal reward magnitude applied to both players on an engine-"
            "declared draw (timeouts now use a life-total tiebreak instead)"
        ),
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument(
        "--minibatch-token-limit",
        type=int,
        default=None,
        help=(
            "maximum packed text tokens per PPO minibatch; text encoder default is "
            f"min(--minibatch-size * --text-max-tokens, {DEFAULT_TEXT_MINIBATCH_TOKEN_LIMIT}); "
            "set 0 to disable"
        ),
    )
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
    parser.add_argument(
        "--game-log-path",
        type=Path,
        default=DEFAULT_GAME_LOG_PATH,
        help="path for sampled game transcripts; rewritten at the start of training",
    )
    parser.add_argument(
        "--priority-trace-jsonl-path",
        type=Path,
        default=None,
        help=(
            "optional JSONL path for sampled priority transcript rows; suitable for "
            "scripts/inline_blank_bc_parity.py --trace-jsonl"
        ),
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
    if args.rollout_min_ready_batch < 1:
        raise ValueError("--rollout-min-ready-batch must be at least 1")
    if args.rollout_ready_wait_ms < 0.0:
        raise ValueError("--rollout-ready-wait-ms must be non-negative")
    if args.max_steps_per_game < 1:
        raise ValueError("--max-steps-per-game must be at least 1")
    if getattr(args, "num_rollout_actors", 1) < 1:
        raise ValueError("--num-rollout-actors must be at least 1")
    if not (0.0 < getattr(args, "inference_ready_fraction", 0.45) <= 1.0):
        raise ValueError("--inference-ready-fraction must be in (0, 1]")
    if getattr(args, "inference_max_batch", 0) < 0:
        raise ValueError("--inference-max-batch must be non-negative")
    if getattr(args, "max_policy_lag", 2) < 0:
        raise ValueError("--max-policy-lag must be non-negative")
    if getattr(args, "learner_min_rows", None) is None:
        args.learner_min_rows = args.rollout_steps
    if getattr(args, "learner_max_rows", None) is None:
        args.learner_max_rows = 3 * args.rollout_steps
    if getattr(args, "learner_target_rows", None) is None:
        args.learner_target_rows = min(
            args.learner_max_rows,
            max(args.learner_min_rows, 4 * args.minibatch_size),
        )
    if getattr(args, "replay_ring_capacity", None) is None:
        args.replay_ring_capacity = max(
            4096,
            6 * args.rollout_steps,
            2 * args.num_envs,
            args.max_steps_per_game,
        )
    if args.learner_min_rows < 1:
        raise ValueError("--learner-min-rows must be at least 1")
    if args.learner_target_rows < args.learner_min_rows:
        raise ValueError("--learner-target-rows must be >= --learner-min-rows")
    if args.learner_max_rows < args.learner_min_rows:
        raise ValueError("--learner-max-rows must be >= --learner-min-rows")
    if args.learner_target_rows > args.learner_max_rows:
        raise ValueError("--learner-target-rows must be <= --learner-max-rows")
    if args.replay_ring_capacity < 1:
        raise ValueError("--replay-ring-capacity must be at least 1")
    if args.replay_ring_capacity < args.learner_max_rows:
        raise ValueError(
            "--replay-ring-capacity must be at least --learner-max-rows so a claimed "
            "training window can fit in the replay ring"
        )
    args.benchmark_warmup_updates = getattr(args, "benchmark_warmup_updates", 1)
    args.benchmark_steady_updates = getattr(args, "benchmark_steady_updates", 5)
    if args.benchmark_warmup_updates < 0:
        raise ValueError("--benchmark-warmup-updates must be non-negative")
    if args.benchmark_steady_updates < 1:
        raise ValueError("--benchmark-steady-updates must be at least 1")
    if args.minibatch_size < 1:
        raise ValueError("--minibatch-size must be at least 1")
    minibatch_token_limit = getattr(args, "minibatch_token_limit", None)
    if minibatch_token_limit is not None and minibatch_token_limit < 0:
        raise ValueError("--minibatch-token-limit must be non-negative")
    if minibatch_token_limit == 0:
        args.minibatch_token_limit = None
    elif minibatch_token_limit is None and getattr(args, "encoder", "slots") == "text":
        args.minibatch_token_limit = min(
            args.minibatch_size * getattr(args, "text_max_tokens", 1),
            DEFAULT_TEXT_MINIBATCH_TOKEN_LIMIT,
        )
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
    if getattr(args, "decoder_layers", None) is not None and args.decoder_layers < 1:
        raise ValueError("--decoder-layers must be at least 1")
    if getattr(args, "decoder_heads", None) is not None and args.decoder_heads < 1:
        raise ValueError("--decoder-heads must be at least 1")
    if getattr(args, "decoder_d_ff", None) is not None and args.decoder_d_ff < 1:
        raise ValueError("--decoder-d-ff must be at least 1")
    decoder_max_decode_len = getattr(args, "decoder_max_decode_len", 1)
    if decoder_max_decode_len is not None and decoder_max_decode_len < 1:
        raise ValueError("--decoder-max-decode-len must be at least 1")
    if getattr(args, "text_hf_layers", None) is not None and args.text_hf_layers < 1:
        raise ValueError("--text-hf-layers must be at least 1")
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
            raise ValueError("--encoder text --trainer rnad requires --native-text-rollout")
        args.spr = False
    if args.torch_compile and not args.no_validate:
        raise ValueError("--torch-compile requires --no-validate")
    deck_sources = [
        name
        for name, value in (
            ("--deck-json", args.deck_json),
            ("--deck-dir", args.deck_dir),
            ("--jumpstart-dir", args.jumpstart_dir),
        )
        if value is not None
    ]
    if len(deck_sources) > 1:
        raise ValueError(f"{', '.join(deck_sources)} are mutually exclusive")
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
        if not 0.0 <= args.rnad_target_ema < 1.0:
            raise ValueError("--rnad-target-ema must be in [0, 1)")
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


def _defer_ready_batch(args: argparse.Namespace, *, ready_count: int, live_count: int) -> bool:
    if args.rollout_ready_wait_ms <= 0.0:
        return False
    if ready_count >= args.rollout_min_ready_batch:
        return False
    # If every remaining live env is ready, waiting cannot coalesce a larger
    # batch; process what we have to avoid a deadlock near rollout tail.
    return live_count > ready_count


def log_ppo_stats(
    stats: PPOStats,
    *,
    games: int,
    steps: int,
    total_rollout_steps: int,
    total_generated_rollout_steps: int,
    win_stats: WinFractionStats | None = None,
    value_metrics: dict[str, float] | None = None,
    token_metrics: dict[str, float] | None = None,
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
    if token_metrics is not None:
        payload.update(token_metrics)
    logger = wandb.log if log_fn is None else log_fn
    logger(payload)


def token_length_percentile_metrics(
    row_token_length_host: list[int],
    rows: Sequence[int],
) -> dict[str, float]:
    """Summarize encoded text state lengths for rollout rows."""
    if not rows:
        return {}
    lengths = torch.tensor(
        [row_token_length_host[int(row)] for row in rows],
        dtype=torch.float32,
        device="cpu",
    )
    if lengths.numel() == 0:
        return {}
    quantiles = torch.tensor((0.25, 0.50, 0.75), dtype=torch.float32)
    p25, p50, p75 = torch.quantile(lengths, quantiles).tolist()
    return {
        "tokens_per_encoded_state/p25": float(p25),
        "tokens_per_encoded_state/p50": float(p50),
        "tokens_per_encoded_state/p75": float(p75),
    }


def _snapshot_games_from_tag(tag: str) -> int | None:
    return snapshot_games_from_tag(tag)


def _snapshot_pct_from_tag(tag: str, total_episodes: int) -> float | None:
    games = snapshot_games_from_tag(tag)
    if games is not None:
        return 100.0 * games / max(1, total_episodes)
    match = re.search(r"_p(\d+(?:\.\d+)?)$", tag)
    if match is not None:
        return float(match.group(1))
    return None


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


def initialize_game_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def game_log_path(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "game_log_path", DEFAULT_GAME_LOG_PATH))


def priority_trace_jsonl_path(args: argparse.Namespace) -> Path | None:
    path = getattr(args, "priority_trace_jsonl_path", None)
    return None if path is None else Path(path)


def append_sample_game_log(
    path: Path,
    transcript: list[TranscriptAction],
    *,
    episode_idx: int,
    winner_idx: int,
    max_actions: int,
    encoder: str,
) -> None:
    lines = sample_game_lines(
        transcript,
        winner_idx=winner_idx,
        max_actions=max_actions,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"encoder={encoder} episode={episode_idx} "
            f"winner={winner_idx if winner_idx >= 0 else 'draw'}\n"
        )
        handle.write("\n".join(lines))
        handle.write("\n\n")


def append_priority_trace_jsonl(
    path: Path | None,
    transcript: list[TranscriptAction],
    *,
    episode_idx: int,
    winner_idx: int,
    encoder: str,
) -> None:
    """Append gate-ready priority transcript rows for inline-blank BC parity."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for action_idx, item in enumerate(transcript):
            if item.pending.get("kind") != "priority":
                continue
            handle.write(
                json.dumps(
                    {
                        "episode_idx": int(episode_idx),
                        "action_idx": int(action_idx),
                        "winner_idx": int(winner_idx),
                        "encoder": encoder,
                        "state": item.state,
                        "pending": item.pending,
                        "action": item.action,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def rollout_value_metrics(
    predicted_values: torch.Tensor,
    returns: torch.Tensor,
) -> dict[str, float]:
    """Summarize rollout return targets and sampled value predictions."""
    return_values = returns.detach().to(dtype=torch.float32)
    pv = predicted_values.detach().to(dtype=torch.float32)
    return {
        "return_mean": float(return_values.mean().item()),
        "return_std": float(return_values.std(unbiased=False).item()),
        "value_mean": float(pv.mean().item()),
        "value_std": float(pv.std(unbiased=False).item()),
    }


def rnad_value_metrics(state: RNaDTrainerState | None) -> dict[str, float]:
    if state is None or not state.last_stats:
        return {}
    rs = state.last_stats[0]
    metrics = {
        "rnad/q_clip_fraction": rs.q_clip_fraction,
        "rnad/v_hat_mean": rs.v_hat_mean,
        "rnad/transformed_reward_mean": rs.transformed_reward_mean,
        "rnad/grad_norm": rs.grad_norm,
        "rnad/outer_iteration": state.outer_iteration,
        "rnad/gradient_step": state.gradient_step,
    }
    if rs.policy_drift_diagnostics_computed:
        metrics.update(
            {
                "rnad/sampled_log_ratio_mean": rs.sampled_log_ratio_mean,
                "rnad/sampled_log_ratio_absmax": rs.sampled_log_ratio_absmax,
                "rnad/is_bias_up_mean": rs.is_bias_up_mean,
                "rnad/is_bias_down_mean": rs.is_bias_down_mean,
            }
        )
    if rs.v_target_reg_share_computed:
        metrics["rnad/v_target_reg_share"] = rs.v_target_reg_share
    return metrics


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

    pending_replay_rows: list[torch.Tensor] = []
    pending_returns: list[torch.Tensor] = []
    pending_step_count: int = 0
    # R-NaD: per-episode RolloutStep lists are still required by EpisodeBatch.
    # The PPO-only path skips this materialization to keep finish_games off
    # the host roundtrip.
    pending_episodes: list[EpisodeBatch] = []
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
                grad_clip=args.max_grad_norm,
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
    trained_completed_games = completed_games
    last_saved_games = restored_state.last_saved_games
    total_rollout_steps = restored_state.total_rollout_steps
    total_generated_rollout_steps = restored_state.total_generated_rollout_steps
    total_wandb_logs = restored_state.total_wandb_logs
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

    transcript_capture_count = max(1, args.num_envs // 256)

    def start_game(slot_idx: int, episode_idx: int) -> LiveGame:
        staging_buffer.reset_env(slot_idx)
        policy.reset_lstm_env_states([slot_idx])
        if sampling_policy is not policy:
            sampling_policy.reset_lstm_env_states([slot_idx])
        seed = args.seed + episode_idx
        deck_a, deck_b = sample_decks(
            deck_pool,
            seed,
            fixed=getattr(args, "deck_json", None) is not None,
            jumpstart=getattr(args, "jumpstart_dir", None) is not None,
        )
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
            transcript_enabled=slot_idx < transcript_capture_count,
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
        total_wandb_logs += 1
        if wandb.run is None:
            return
        wandb.log(payload)

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

    def finish_games(finished: list[tuple[LiveGame, int, bool]]) -> None:
        nonlocal completed_games, total_generated_rollout_steps, pending_step_count
        if not finished:
            return

        envs = [env for env, _, _ in finished]

        # PPO builds GAE return targets from the padded (B, T_max) staging
        # slice. R-NaD skips that target construction and only materializes the
        # per-episode rollout records required by ``EpisodeBatch``.
        device = staging_buffer.device

        # Per-row terminal reward (in p0's perspective) and zero-sum flag.
        # Engine wins/losses are zero-sum ±1, engine draws are symmetric
        # (-draw_penalty for both perspectives), step-cap timeouts get a
        # life-total tiebreak. Life totals are read here so the actor /
        # learner doesn't need a separate plumbing channel.
        terminal_p0_h: list[float] = []
        zero_sum_h: list[bool] = []
        for env, winner_idx, is_timeout in finished:
            l0, l1 = _read_life_totals(env.game) if is_timeout else (0, 0)
            tp0, zs = terminal_reward_for_finish(
                winner_idx=int(winner_idx),
                is_timeout=bool(is_timeout),
                life_p0=l0,
                life_p1=l1,
                draw_penalty=args.draw_penalty,
            )
            terminal_p0_h.append(tp0)
            zero_sum_h.append(zs)
        slot_h = torch.tensor(
            [env.slot_idx for env in envs],
            dtype=torch.long,
            pin_memory=device.type == "cuda",
        )
        slot_t = slot_h.to(device, non_blocking=True)
        step_counts = staging_buffer.step_count[slot_t]
        max_steps = int(staging_buffer.max_steps_per_trajectory)
        step_arange = torch.arange(max_steps, device=device).unsqueeze(0)
        valid_mask = step_arange < step_counts.unsqueeze(1)
        players_padded = staging_buffer.perspective_player_idx[slot_t]
        values_padded = staging_buffer.value[slot_t]

        if rnad_state is None:
            terminal_p0_t = torch.tensor(
                terminal_p0_h,
                dtype=torch.float32,
                pin_memory=device.type == "cuda",
            ).to(device, non_blocking=True)
            zero_sum_t = torch.tensor(
                zero_sum_h,
                dtype=torch.bool,
                pin_memory=device.type == "cuda",
            ).to(device, non_blocking=True)
            returns_padded = gae_returns_batched(
                values_padded,
                players_padded,
                step_counts,
                terminal_reward_p0=terminal_p0_t,
                zero_sum=zero_sum_t,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            flat_returns_dev = returns_padded[valid_mask]
            flat_replay_rows = policy.append_staged_episodes_returning_tensor(
                staging_buffer,
                [env.slot_idx for env in envs],
            )
            n_new = int(flat_replay_rows.numel())
            if n_new > 0:
                pending_replay_rows.append(flat_replay_rows)
                pending_returns.append(flat_returns_dev)
                pending_step_count += n_new
                total_generated_rollout_steps += n_new
        else:
            replay_rows_by_env = policy.append_staged_episodes_to_rollout(
                staging_buffer,
                [env.slot_idx for env in envs],
            )
            log_padded = staging_buffer.old_log_prob[slot_t]
            flat_player = players_padded[valid_mask]
            flat_log = log_padded[valid_mask]
            flat_val = values_padded[valid_mask]
            host = torch.stack(
                [flat_player.to(torch.float32), flat_log, flat_val],
                dim=0,
            ).cpu()
            host_player = host[0].long().tolist()
            host_log = host[1].tolist()
            host_val = host[2].tolist()
            step_counts_h = step_counts.tolist()
            n_new = sum(step_counts_h)
            if n_new > 0:
                pending_step_count += n_new
                total_generated_rollout_steps += n_new
            cursor = 0
            for (
                (env, winner_idx, _is_timeout),
                env_rows,
                step_count,
                tp0,
                zs,
            ) in zip(
                finished,
                replay_rows_by_env,
                step_counts_h,
                terminal_p0_h,
                zero_sum_h,
                strict=True,
            ):
                if step_count:
                    end = cursor + step_count
                    episode_steps = [
                        RolloutStep(
                            perspective_player_idx=int(p),
                            old_log_prob=float(lp),
                            value=float(v),
                            replay_idx=r,
                        )
                        for p, lp, v, r in zip(
                            host_player[cursor:end],
                            host_log[cursor:end],
                            host_val[cursor:end],
                            env_rows,
                            strict=True,
                        )
                    ]
                    cursor = end
                    pending_episodes.append(
                        EpisodeBatch(
                            steps=episode_steps,
                            terminal_reward_p0=float(tp0),
                            zero_sum=bool(zs),
                        )
                    )

        for env, winner_idx, is_timeout in finished:
            env.game.close()
            if env.slot_idx == 0:
                append_sample_game_log(
                    game_log_path(args),
                    env.transcript,
                    episode_idx=env.episode_idx,
                    winner_idx=winner_idx,
                    max_actions=args.sample_actions,
                    encoder="slots",
                )
                append_priority_trace_jsonl(
                    priority_trace_jsonl_path(args),
                    env.transcript,
                    episode_idx=env.episode_idx,
                    winner_idx=winner_idx,
                    encoder="slots",
                )
                print_sample_game(
                    env.transcript,
                    winner_idx=winner_idx,
                    max_actions=args.sample_actions,
                )
            staging_buffer.reset_env(env.slot_idx)
            free_slots.append(env.slot_idx)
            if is_timeout:
                win_stats.timeouts += 1
            elif winner_idx == 0:
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
        finished_games: list[tuple[LiveGame, int, bool]] = []
        for idx, env in enumerate(live_games):
            is_over = bool(over_l[idx])
            if is_over or env.action_count >= args.max_steps_per_game:
                finished_games.append((env, int(winner_l[idx]) if is_over else -1, not is_over))
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

        if _defer_ready_batch(args, ready_count=len(ready_envs), live_count=len(live_games)):
            time.sleep(args.rollout_ready_wait_ms / 1000.0)
            continue

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
            behavior_action_log_probs_flat = torch.tensor(
                [lp for s in policy_steps for lp in s.selected_action_log_probs],
                dtype=torch.float32,
            ).to(policy.device, non_blocking=True)
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
                behavior_action_log_probs_flat=behavior_action_log_probs_flat,
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

        if pending_step_count >= args.rollout_steps:
            rollout_step_count = pending_step_count
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
                rollout_returns = torch.cat(pending_returns)
                rollout_replay_rows = torch.cat(pending_replay_rows)
                stats = ppo_update(
                    policy,
                    optimizer,
                    rollout_replay_rows,
                    rollout_returns,
                    epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    clip_epsilon=args.clip_epsilon,
                    value_coef=args.value_coef,
                    entropy_coef=args.entropy_coef,
                    max_grad_norm=args.max_grad_norm,
                    spr_coef=args.spr_coef if args.spr else 0.0,
                    minibatch_token_limit=getattr(args, "minibatch_token_limit", None),
                    minibatch_max_tokens_per_row=getattr(args, "text_max_tokens", None)
                    if getattr(args, "encoder", "slots") == "text"
                    else None,
                )
            rl_lr_warmup_step(optimizer)
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
            if rnad_state is None:
                rollout_predicted_values = policy.rollout_buffer.value[rollout_replay_rows]
                value_metrics = rollout_value_metrics(rollout_predicted_values, rollout_returns)
            else:
                value_metrics = rnad_value_metrics(rnad_state)
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
            pending_replay_rows.clear()
            pending_returns.clear()
            pending_episodes.clear()
            pending_step_count = 0
            win_stats.reset()
            trained_completed_games = completed_games
            _record_phase("post_update", _t)

        if (
            args.save_every
            and trained_completed_games > 0
            and trained_completed_games >= last_saved_games + args.save_every
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
                    completed_games=trained_completed_games,
                    last_saved_games=trained_completed_games,
                    total_rollout_steps=total_rollout_steps,
                    total_generated_rollout_steps=total_generated_rollout_steps,
                    total_wandb_logs=total_wandb_logs,
                ),
                rnad_state=rnad_state,
            )
            last_saved_games = trained_completed_games

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
                if wandb.run is not None:
                    log_retrospective_table(
                        wandb.run,
                        horizon_pct=int(round(threshold * 100 / max(1, args.episodes))),
                        horizon_step_count=threshold,
                        ratings=retrospective_rating_rows(
                            opponent_pool,
                            total_episodes=args.episodes,
                        ),
                        log_fn=tracked_wandb_log,
                    )

        _t = time.perf_counter()
        maybe_start_games()
        _record_phase("maybe_start", _t)

    if pending_step_count > 0:
        rollout_step_count = pending_step_count
        if rnad_state is not None and pending_episodes:
            stats = run_rnad_update(
                policy,
                optimizer,
                rnad_state,
                pending_episodes,
            )
            value_metrics = rnad_value_metrics(rnad_state)
        else:
            rollout_returns = torch.cat(pending_returns)
            rollout_replay_rows = torch.cat(pending_replay_rows)
            stats = ppo_update(
                policy,
                optimizer,
                rollout_replay_rows,
                rollout_returns,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_epsilon=args.clip_epsilon,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                spr_coef=args.spr_coef if args.spr else 0.0,
                minibatch_token_limit=getattr(args, "minibatch_token_limit", None),
                minibatch_max_tokens_per_row=getattr(args, "text_max_tokens", None)
                if getattr(args, "encoder", "slots") == "text"
                else None,
            )
            rollout_predicted_values = policy.rollout_buffer.value[rollout_replay_rows]
            value_metrics = rollout_value_metrics(rollout_predicted_values, rollout_returns)
        rl_lr_warmup_step(optimizer)
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
            value_metrics=value_metrics,
            log_fn=tracked_wandb_log,
            run_active=True,
        )
        policy.reset_rollout_buffer()
        pending_episodes.clear()
        trained_completed_games = completed_games

    return (
        TrainingResumeState(
            completed_games=trained_completed_games,
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
    native_rollout: ShardedNativeRolloutDriver | None = None,
    native_encoder: ShardedNativeBatchEncoder | None = None,
    opponent_pool: OpponentPool | None = None,
    snapshot_schedule: SnapshotSchedule | None = None,
    retrospective_schedule: RetrospectiveLogSchedule | None = None,
    opponent_policy: LSTMStatefulTextPolicy | None = None,
    resume_state: TrainingResumeState | None = None,
) -> tuple[TrainingResumeState, None]:
    """Slow Python text-encoder PPO loop.

    This intentionally stays separate from the native slot loop. It provides a
    correctness path for `--encoder text`; performance work can later replace
    the single-env Python stepping with native render-plan batching.
    """

    eval_rng = random.Random(args.seed ^ 0x5EED5)
    restored_state = resume_state or TrainingResumeState()
    completed_games = restored_state.completed_games
    trained_completed_games = completed_games
    last_saved_games = restored_state.last_saved_games
    total_rollout_steps = restored_state.total_rollout_steps
    total_generated_rollout_steps = restored_state.total_generated_rollout_steps
    total_wandb_logs = restored_state.total_wandb_logs

    pending_steps: list[RolloutStep] = []
    pending_returns: list[torch.Tensor] = []
    pending_episode_rows: list[list[int]] = []
    win_stats = WinFractionStats()
    backend.replay_buffer.reset()
    backend.policy.init_lstm_env_states(1)
    last_step_time = time.monotonic()

    def cli_step_prefix() -> str:
        return f"step={total_wandb_logs}"

    def tracked_wandb_log(payload: dict[str, Any]) -> None:
        nonlocal total_wandb_logs
        total_wandb_logs += 1
        if wandb.run is None:
            return
        wandb.log(payload)

    def run_update(*, final: bool = False) -> None:
        nonlocal total_rollout_steps, last_step_time, trained_completed_games
        if not pending_steps:
            return
        rollout_returns = torch.cat(pending_returns)
        rollout_step_count = len(pending_steps)
        refresh_fn = None  # decoder pipeline does not implement LSTM-state refresh
        device = next(backend.policy.parameters()).device
        rollout_replay_rows_h = [cast(int, step.replay_idx) for step in pending_steps]
        rollout_replay_rows = torch.tensor(
            rollout_replay_rows_h,
            dtype=torch.long,
            device=device,
        )
        stats = ppo_update(
            backend.policy,
            optimizer,
            rollout_replay_rows,
            rollout_returns,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            spr_coef=args.spr_coef if args.spr else 0.0,
            between_epoch_fn=refresh_fn,
            minibatch_token_limit=getattr(args, "minibatch_token_limit", None),
            minibatch_max_tokens_per_row=getattr(args, "text_max_tokens", None),
        )
        rl_lr_warmup_step(optimizer)
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
            value_metrics=rollout_value_metrics(
                torch.tensor([step.value for step in pending_steps], dtype=torch.float32),
                rollout_returns,
            ),
            token_metrics=(
                token_length_percentile_metrics(
                    backend.replay_buffer.row_token_length_host,
                    rollout_replay_rows_h,
                )
                if rollout_replay_rows_h
                else None
            ),
            log_fn=tracked_wandb_log,
            run_active=True,
        )
        backend.replay_buffer.reset()
        pending_steps.clear()
        pending_returns.clear()
        pending_episode_rows.clear()
        win_stats.reset()
        trained_completed_games = completed_games

    for episode_idx in range(completed_games, args.episodes):
        backend.policy.reset_lstm_env_states([0])
        seed = args.seed + episode_idx
        deck_a, deck_b = sample_decks(
            deck_pool,
            seed,
            fixed=getattr(args, "deck_json", None) is not None,
            jumpstart=getattr(args, "jumpstart_dir", None) is not None,
        )
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
        transcript: list[TranscriptAction] = []
        winner_idx = -1
        is_timeout = True
        life_p0 = 0
        life_p1 = 0
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
                transcript.append(
                    TranscriptAction(
                        state=copy.deepcopy(snapshot),
                        pending=copy.deepcopy(pending),
                        action=copy.deepcopy(policy_step.action),
                    )
                )
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
            is_timeout = not game.is_over
            if game.is_over:
                winner_idx = _winner_idx_from_game(game)
            life_p0, life_p1 = _read_life_totals(game)
        finally:
            try:
                game.close()
            except Exception:
                pass

        append_sample_game_log(
            game_log_path(args),
            transcript,
            episode_idx=episode_idx,
            winner_idx=winner_idx,
            max_actions=args.sample_actions,
            encoder="text",
        )
        append_priority_trace_jsonl(
            priority_trace_jsonl_path(args),
            transcript,
            episode_idx=episode_idx,
            winner_idx=winner_idx,
            encoder="text",
        )
        completed_games += 1
        if is_timeout:
            win_stats.timeouts += 1
        elif winner_idx == 0:
            win_stats.p1_wins += 1
        elif winner_idx == 1:
            win_stats.p2_wins += 1
        else:
            win_stats.draws += 1
        if episode_steps:
            ep_rows = [s.replay_idx for s in episode_steps if s.replay_idx is not None]
            if ep_rows:
                pending_episode_rows.append(ep_rows)
            pending_steps.extend(episode_steps)
            tp0, zs = terminal_reward_for_finish(
                winner_idx=int(winner_idx),
                is_timeout=is_timeout,
                life_p0=life_p0,
                life_p1=life_p1,
                draw_penalty=args.draw_penalty,
            )
            pending_returns.append(
                gae_returns(
                    episode_steps,
                    terminal_reward_p0=tp0,
                    zero_sum=zs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
            )
            total_generated_rollout_steps += len(episode_steps)
        if len(pending_steps) >= args.rollout_steps:
            run_update()

        if (
            args.save_every
            and trained_completed_games > 0
            and trained_completed_games >= last_saved_games + args.save_every
        ):
            save_checkpoint(
                args.output,
                backend.policy,
                optimizer,
                args,
                opponent_pool=opponent_pool,
                snapshot_schedule=snapshot_schedule,
                retrospective_schedule=retrospective_schedule,
                resume_state=TrainingResumeState(
                    completed_games=trained_completed_games,
                    last_saved_games=trained_completed_games,
                    total_rollout_steps=total_rollout_steps,
                    total_generated_rollout_steps=total_generated_rollout_steps,
                    total_wandb_logs=total_wandb_logs,
                ),
            )
            last_saved_games = trained_completed_games

        if (
            opponent_pool is not None
            and snapshot_schedule is not None
            and opponent_policy is not None
            and native_rollout is not None
            and native_encoder is not None
        ):
            for threshold in snapshot_schedule.fire(completed_games):
                take_snapshot_and_eval(
                    args=args,
                    threshold=threshold,
                    policy=backend.policy,
                    opponent_policy=opponent_policy,
                    opponent_pool=opponent_pool,
                    native_encoder=native_encoder,
                    native_rollout=native_rollout,
                    mage=mage,
                    deck_pool=deck_pool,
                    rng=eval_rng,
                    step_prefix=f"step={total_wandb_logs}",
                    log_fn=tracked_wandb_log,
                )
                if wandb.run is not None:
                    log_retrospective_table(
                        wandb.run,
                        horizon_pct=int(round(threshold * 100 / max(1, args.episodes))),
                        horizon_step_count=threshold,
                        ratings=retrospective_rating_rows(
                            opponent_pool,
                            total_episodes=args.episodes,
                        ),
                        log_fn=tracked_wandb_log,
                    )

    run_update(final=True)
    return (
        TrainingResumeState(
            completed_games=trained_completed_games,
            last_saved_games=last_saved_games,
            total_rollout_steps=total_rollout_steps,
            total_generated_rollout_steps=total_generated_rollout_steps,
            total_wandb_logs=total_wandb_logs,
        ),
        None,
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
    opponent_pool: OpponentPool | None = None,
    snapshot_schedule: SnapshotSchedule | None = None,
    retrospective_schedule: RetrospectiveLogSchedule | None = None,
    opponent_policy: LSTMStatefulTextPolicy | None = None,
    resume_state: TrainingResumeState | None = None,
    resume_checkpoint: dict[str, Any] | None = None,
) -> tuple[TrainingResumeState, RNaDTrainerState | None]:
    if not native_encoder.is_available:
        raise SystemExit("native text rollout requires MageEncodeBatch")
    eval_rng = random.Random(args.seed ^ 0x5EED5)

    restored_state = resume_state or TrainingResumeState()
    completed_games = restored_state.completed_games
    trained_completed_games = completed_games
    last_saved_games = restored_state.last_saved_games
    total_rollout_steps = restored_state.total_rollout_steps
    total_generated_rollout_steps = restored_state.total_generated_rollout_steps
    total_wandb_logs = restored_state.total_wandb_logs

    pending_replay_rows: list[torch.Tensor] = []
    pending_replay_rows_host: list[int] = []
    pending_returns: list[torch.Tensor] = []
    pending_step_count: int = 0
    # Guards read-modify-write of pending_step_count between the driver thread
    # (increments on env-finish) and the learner thread (decrement on claim).
    pending_step_count_lock = threading.Lock()
    pending_episodes: list[EpisodeBatch] = []
    # PPO-only path: accumulate per-finished-batch device tensors here and
    # do a single host split at run_update time, instead of one per batch.
    pending_flat_rows_chunks: list[torch.Tensor] = []
    pending_episode_step_counts: list[torch.Tensor] = []
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
                grad_clip=args.max_grad_norm,
                q_corr_rho_bar=args.rnad_q_corr_rho_bar,
                bptt_chunk_size=args.rnad_bptt_chunk_size,
                step_minibatch_size=args.minibatch_size,
            ),
            reg_snapshot_dir=args.output.parent / "rnad",
            device=backend.policy.device,
        )
        _restore_rnad_state(rnad_state, resume_checkpoint)
        cast(LSTMStatefulTextPolicy, rnad_state.target).init_lstm_env_states(args.num_envs)
    win_stats = WinFractionStats()
    backend.replay_buffer.reset()
    staging_buffer = NativeTextTrajectoryBuffer(
        backend.replay_buffer,
        num_envs=args.num_envs,
        max_steps=args.max_steps_per_game,
        validate=not getattr(args, "no_validate", False),
    )
    backend.policy.init_lstm_env_states(args.num_envs)
    sampling_policy: LSTMStatefulTextPolicy = (
        cast(LSTMStatefulTextPolicy, rnad_state.target)
        if rnad_state is not None
        else backend.policy
    )
    next_episode_idx = completed_games
    live_games: list[LiveGame] = []
    free_slots = list(range(args.num_envs - 1, -1, -1))
    last_step_time = time.monotonic()
    transcript_warning_emitted = False
    transcript_capture_count = max(1, args.num_envs // 256)
    num_actors = int(getattr(args, "num_rollout_actors", 1))

    def cli_step_prefix() -> str:
        return f"step={total_wandb_logs}"

    def tracked_wandb_log(payload: dict[str, Any]) -> None:
        nonlocal total_wandb_logs
        total_wandb_logs += 1
        if wandb.run is None:
            return
        wandb.log(payload)

    def disable_transcript(env: LiveGame, reason: str) -> None:
        nonlocal transcript_warning_emitted
        env.transcript_enabled = False
        if not transcript_warning_emitted:
            print(
                cli_step_prefix(),
                f"warning: disabling text sample transcript capture: {reason}",
                flush=True,
            )
            transcript_warning_emitted = True

    def start_game(slot_idx: int, episode_idx: int) -> LiveGame:
        backend.policy.reset_lstm_env_states([slot_idx])
        if sampling_policy is not backend.policy:
            sampling_policy.reset_lstm_env_states([slot_idx])
        seed = args.seed + episode_idx
        deck_a, deck_b = sample_decks(
            deck_pool,
            seed,
            fixed=args.deck_json is not None,
            jumpstart=getattr(args, "jumpstart_dir", None) is not None,
        )
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
            transcript_enabled=(
                slot_idx % num_actors == 0 and (slot_idx // num_actors) < transcript_capture_count
            ),
        )

    def maybe_start_games() -> None:
        nonlocal next_episode_idx
        if pending_step_count >= args.rollout_steps:
            return
        while free_slots and next_episode_idx < args.episodes:
            live_games.append(start_game(free_slots.pop(), next_episode_idx))
            next_episode_idx += 1

    def finish_games(finished: list[tuple[LiveGame, int, bool]]) -> None:
        nonlocal completed_games, total_generated_rollout_steps, pending_step_count
        finished_with_steps = [(env, w, t) for env, w, t in finished if env.episode_steps]
        if finished_with_steps:
            device = staging_buffer.device
            # Per-row terminal reward (in p0's perspective) and zero-sum flag.
            # See the slot path's finish_games for the full breakdown of
            # cases (engine win/loss, engine draw, step-cap timeout).
            finish_part_start = time.perf_counter()
            terminal_p0_h: list[float] = []
            zero_sum_h: list[bool] = []
            for env, winner_idx, is_timeout in finished_with_steps:
                l0, l1 = _read_life_totals(env.game) if is_timeout else (0, 0)
                tp0, zs = terminal_reward_for_finish(
                    winner_idx=int(winner_idx),
                    is_timeout=bool(is_timeout),
                    life_p0=l0,
                    life_p1=l1,
                    draw_penalty=args.draw_penalty,
                )
                terminal_p0_h.append(tp0)
                zero_sum_h.append(zs)
            timing_stats = getattr(staging_buffer, "timing_stats", None)
            if timing_stats is not None:
                timing_stats.add("finish_terminal_reward", time.perf_counter() - finish_part_start)

            finish_part_start = time.perf_counter()
            needs_staging_commit = any(
                map(
                    lambda done: any(
                        map(lambda step: step.replay_idx is None, done[0].episode_steps)
                    ),
                    finished_with_steps,
                )
            )
            if needs_staging_commit:
                slot_idxs = list(map(lambda done: done[0].slot_idx, finished_with_steps))
                slot_t = torch.tensor(
                    slot_idxs,
                    dtype=torch.long,
                    pin_memory=device.type == "cuda",
                ).to(device, non_blocking=True)
                # NativeTextTrajectoryBuffer tracks step counts per env via the
                # host-side ``step_count_host`` list. ``value`` /
                # ``perspective_player_idx`` are only consulted on the PPO path
                # below and live on the per-env staged decoder rows.
                # NativeTextTrajectoryBuffer.append_envs_to_replay_returning_tensor
                # commits the reservation internally regardless of ``seal``;
                # there is no separate seal_staged_rows step to perform.
                needs_staged_seal = False
                flat_rows, counts = staging_buffer.append_envs_to_replay_returning_tensor(
                    slot_idxs,
                    backend.replay_buffer,
                    seal=True,
                )
                step_counts = counts.to(dtype=torch.long)
                counts_h = counts.detach().cpu().tolist()
                split_rows = torch.split(flat_rows.detach().cpu(), counts_h)
                per_env_rows_h = tuple(map(lambda rows: tuple(map(int, rows.tolist())), split_rows))
                tuple(
                    map(
                        lambda item: item[0].episode_steps.__setitem__(
                            slice(None),
                            list(
                                map(
                                    lambda pair: replace(pair[0], replay_idx=int(pair[1])),
                                    zip(item[0].episode_steps, item[1], strict=True),
                                )
                            ),
                        ),
                        zip(
                            map(lambda done: done[0], finished_with_steps),
                            per_env_rows_h,
                            strict=True,
                        ),
                    )
                )
                pending_replay_rows.append(flat_rows)
                pending_replay_rows_host.extend(itertools.chain.from_iterable(per_env_rows_h))
                if rnad_state is None:
                    terminal_p0_t = torch.tensor(
                        terminal_p0_h,
                        dtype=torch.float32,
                        pin_memory=device.type == "cuda",
                    ).to(device, non_blocking=True)
                    zero_sum_t = torch.tensor(
                        zero_sum_h,
                        dtype=torch.bool,
                        pin_memory=device.type == "cuda",
                    ).to(device, non_blocking=True)
                    max_steps = staging_buffer.max_steps
                    step_arange = torch.arange(max_steps, device=device).unsqueeze(0)
                    valid_mask = step_arange < step_counts.unsqueeze(1)
                    returns_padded = gae_returns_batched(
                        staging_buffer.value[slot_t],  # ty: ignore[unresolved-attribute]
                        staging_buffer.perspective_player_idx[slot_t],  # ty: ignore[unresolved-attribute]
                        step_counts,
                        terminal_reward_p0=terminal_p0_t,
                        zero_sum=zero_sum_t,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                    )
                    pending_returns.append(returns_padded[valid_mask])
                    pending_flat_rows_chunks.append(flat_rows)
                    pending_episode_step_counts.append(step_counts)
            else:
                per_env_rows_h = tuple(
                    map(
                        lambda done: tuple(
                            map(
                                lambda step: cast(int, step.replay_idx),
                                done[0].episode_steps,
                            )
                        ),
                        finished_with_steps,
                    )
                )
            flat_rows = torch.tensor(
                tuple(itertools.chain.from_iterable(per_env_rows_h)),
                dtype=torch.long,
                device=device,
            )
            n_new = int(flat_rows.numel())
            if n_new > 0:

                def _write_text_episode_metadata(item: Any) -> None:
                    if not hasattr(backend.replay_buffer, "write_episode_metadata"):
                        return
                    (env, _winner_idx, _is_timeout), rows_chunk, tp0, zs = item
                    backend.replay_buffer.write_episode_metadata(
                        rows_chunk,
                        episode_id=int(env.episode_idx),
                        terminal_reward_p0=float(tp0),
                        zero_sum=bool(zs),
                        actor_id=int(getattr(env, "actor_id", -1)),
                        behavior_policy_version=int(env.behavior_policy_version),
                        inference_policy_version=int(env.inference_policy_version),
                        target_policy_version=(
                            int(env.inference_policy_version)
                            if rnad_state is not None
                            else int(env.target_policy_version)
                        ),
                    )

                tuple(
                    map(
                        _write_text_episode_metadata,
                        zip(
                            finished_with_steps,
                            map(
                                lambda rows_h: torch.tensor(
                                    rows_h,
                                    dtype=torch.long,
                                    device=device,
                                ),
                                per_env_rows_h,
                            ),
                            terminal_p0_h,
                            zero_sum_h,
                            strict=True,
                        ),
                    )
                )
                if (
                    needs_staging_commit
                    and needs_staged_seal
                    and n_new > 0
                    and hasattr(backend.replay_buffer, "seal_staged_rows")
                ):
                    backend.replay_buffer.seal_staged_rows(flat_rows)
                with pending_step_count_lock:
                    pending_step_count += n_new
                total_generated_rollout_steps += n_new
            timing_stats = getattr(staging_buffer, "timing_stats", None)
            if timing_stats is not None:
                timing_stats.add("finish_metadata", time.perf_counter() - finish_part_start)

        finish_part_start = time.perf_counter()
        for env, winner_idx, is_timeout in finished:
            try:
                env.game.close()
            except Exception:
                pass
            if env.slot_idx == 0:
                append_sample_game_log(
                    game_log_path(args),
                    env.transcript,
                    episode_idx=env.episode_idx,
                    winner_idx=winner_idx,
                    max_actions=args.sample_actions,
                    encoder="text",
                )
                append_priority_trace_jsonl(
                    priority_trace_jsonl_path(args),
                    env.transcript,
                    episode_idx=env.episode_idx,
                    winner_idx=winner_idx,
                    encoder="text",
                )
            free_slots.append(env.slot_idx)
            if is_timeout:
                win_stats.timeouts += 1
            elif winner_idx == 0:
                win_stats.p1_wins += 1
            elif winner_idx == 1:
                win_stats.p2_wins += 1
            else:
                win_stats.draws += 1
            completed_games += 1
        timing_stats = getattr(staging_buffer, "timing_stats", None)
        if timing_stats is not None:
            timing_stats.add("finish_close_bookkeeping", time.perf_counter() - finish_part_start)

    def run_update(
        *,
        final: bool = False,
        replay_rows_chunks: list[torch.Tensor] | None = None,
        returns_chunks: list[torch.Tensor] | None = None,
        replay_rows_tensor: torch.Tensor | None = None,
        returns_tensor: torch.Tensor | None = None,
        episodes: list[EpisodeBatch] | None = None,
        flat_rows_chunks: list[torch.Tensor] | None = None,
        episode_step_counts: list[torch.Tensor] | None = None,
        step_count: int | None = None,
        replay_rows_host: Sequence[int] = (),
        completed_games_snapshot: int | None = None,
        win_stats_snapshot: WinFractionStats | None = None,
        reset_replay: bool = True,
    ) -> None:
        nonlocal total_rollout_steps, last_step_time, pending_step_count
        nonlocal trained_completed_games
        update_step_count = pending_step_count if step_count is None else int(step_count)
        if update_step_count == 0:
            return
        update_rows = pending_replay_rows if replay_rows_chunks is None else replay_rows_chunks
        update_rows_host = pending_replay_rows_host if not replay_rows_host else replay_rows_host
        update_episodes = pending_episodes if episodes is None else episodes
        update_flat_rows = (
            pending_flat_rows_chunks if flat_rows_chunks is None else flat_rows_chunks
        )
        update_episode_counts = (
            pending_episode_step_counts if episode_step_counts is None else episode_step_counts
        )
        rollout_replay_rows = (
            replay_rows_tensor.to(device=backend.replay_buffer.device, dtype=torch.long)
            if replay_rows_tensor is not None
            else torch.cat(update_rows)
        )
        rollout_step_count = update_step_count
        log_completed_games = (
            completed_games if completed_games_snapshot is None else int(completed_games_snapshot)
        )
        log_win_stats = win_stats if win_stats_snapshot is None else win_stats_snapshot
        snapshot_armed = args.cuda_memory_snapshot is not None and torch.cuda.is_available()
        if snapshot_armed:
            print(
                cli_step_prefix(),
                f"[mem] before update[{args.trainer},text,native]: "
                f"{torch.cuda.memory_allocated() / 1e9:.2f} GB live, "
                f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved",
                flush=True,
            )
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        if rnad_state is not None and replay_rows_tensor is not None and not update_episodes:
            rows_h = rollout_replay_rows.detach().cpu().tolist()
            ep_h = (
                backend.replay_buffer.episode_meta.episode_id[rollout_replay_rows]
                .detach()
                .cpu()
                .tolist()
            )
            step_h = (
                backend.replay_buffer.episode_meta.step_idx[rollout_replay_rows]
                .detach()
                .cpu()
                .tolist()
            )
            player_h = (
                backend.replay_buffer.perspective_player_idx[rollout_replay_rows]
                .detach()
                .cpu()
                .tolist()
            )
            old_lp_h = (
                backend.replay_buffer.old_log_prob[rollout_replay_rows].detach().cpu().tolist()
            )
            value_h = backend.replay_buffer.value[rollout_replay_rows].detach().cpu().tolist()
            terminal_h = (
                backend.replay_buffer.episode_meta.terminal_reward_p0[rollout_replay_rows]
                .detach()
                .cpu()
                .tolist()
            )
            zero_h = (
                backend.replay_buffer.episode_meta.zero_sum[rollout_replay_rows]
                .detach()
                .cpu()
                .tolist()
            )
            grouped_rnad: dict[int, list[tuple[int, int, int, float, float, float, bool]]] = {}
            cursor = 0
            while cursor < len(rows_h):
                grouped_rnad.setdefault(int(ep_h[cursor]), []).append(
                    (
                        int(step_h[cursor]),
                        int(rows_h[cursor]),
                        int(player_h[cursor]),
                        float(old_lp_h[cursor]),
                        float(value_h[cursor]),
                        float(terminal_h[cursor]),
                        bool(zero_h[cursor]),
                    )
                )
                cursor += 1
            update_episodes = []
            grouped_items = list(sorted(grouped_rnad.items()))
            group_cursor = 0
            while group_cursor < len(grouped_items):
                _episode_id, items = grouped_items[group_cursor]
                sorted_items = sorted(items)
                update_episodes.append(
                    EpisodeBatch(
                        steps=list(
                            map(
                                lambda item: RolloutStep(
                                    perspective_player_idx=item[2],
                                    old_log_prob=item[3],
                                    value=item[4],
                                    replay_idx=item[1],
                                ),
                                sorted_items,
                            )
                        ),
                        terminal_reward_p0=float(sorted_items[-1][5]),
                        zero_sum=bool(sorted_items[-1][6]),
                    )
                )
                group_cursor += 1
        if rnad_state is not None and update_episodes:
            try:
                stats = run_rnad_update(
                    backend.policy,
                    optimizer,
                    rnad_state,
                    update_episodes,
                )
            except torch.OutOfMemoryError:
                if snapshot_armed:
                    path = Path(args.cuda_memory_snapshot)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        torch.cuda.memory._dump_snapshot(str(path))
                        print(
                            cli_step_prefix(),
                            f"[mem] OOM in update[{args.trainer},text,native]; "
                            f"allocator history dumped to {path}. "
                            "Load at https://pytorch.org/memory_viz",
                            flush=True,
                        )
                    except Exception as dump_exc:
                        print(
                            cli_step_prefix(),
                            f"[mem] _dump_snapshot failed: {type(dump_exc).__name__}: {dump_exc}",
                            flush=True,
                        )
                raise
            finally:
                if snapshot_armed:
                    torch.cuda.memory._record_memory_history(enabled=None)
            trainer_label = "rnad,text,native"
        else:
            update_returns = pending_returns if returns_chunks is None else returns_chunks
            rollout_returns = (
                returns_tensor.to(device=backend.policy.device, dtype=torch.float32)
                if returns_tensor is not None
                else torch.cat(update_returns)
            )
            # One host sync over the rollout instead of one per finished
            # batch: cat the per-batch device tensors, transfer once, split
            # into per-episode lists for the LSTM-refresh callback.
            ep_rows_snapshot: list[list[int]] = []
            if replay_rows_tensor is not None:
                rows_h = rollout_replay_rows.detach().cpu().tolist()
                ep_h = (
                    backend.replay_buffer.episode_meta.episode_id[rollout_replay_rows]
                    .detach()
                    .cpu()
                    .tolist()
                )
                step_h = (
                    backend.replay_buffer.episode_meta.step_idx[rollout_replay_rows]
                    .detach()
                    .cpu()
                    .tolist()
                )
                grouped: dict[int, list[tuple[int, int]]] = {}
                cursor = 0
                while cursor < len(rows_h):
                    grouped.setdefault(int(ep_h[cursor]), []).append(
                        (int(step_h[cursor]), int(rows_h[cursor]))
                    )
                    cursor += 1
                ep_rows_snapshot = list(
                    map(
                        lambda item: list(map(lambda pair: pair[1], sorted(item[1]))),
                        sorted(grouped.items()),
                    )
                )
            elif update_flat_rows:
                all_rows_h = torch.cat(update_flat_rows).cpu().tolist()
                all_counts_h = torch.cat(update_episode_counts).tolist()
                cursor = 0
                counts_cursor = 0
                while counts_cursor < len(all_counts_h):
                    c = all_counts_h[counts_cursor]
                    if c > 0:
                        ep_rows_snapshot.append(list(map(int, all_rows_h[cursor : cursor + c])))
                        cursor += c
                    counts_cursor += 1
            native_refresh_fn = None  # decoder pipeline does not refresh LSTM states
            try:
                stats = ppo_update(
                    backend.policy,
                    optimizer,
                    rollout_replay_rows,
                    rollout_returns,
                    epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    clip_epsilon=args.clip_epsilon,
                    value_coef=args.value_coef,
                    entropy_coef=args.entropy_coef,
                    max_grad_norm=args.max_grad_norm,
                    spr_coef=0.0,
                    between_epoch_fn=native_refresh_fn,
                    minibatch_token_limit=getattr(args, "minibatch_token_limit", None),
                    minibatch_max_tokens_per_row=getattr(args, "text_max_tokens", None),
                )
            except torch.OutOfMemoryError:
                if snapshot_armed:
                    path = Path(args.cuda_memory_snapshot)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        torch.cuda.memory._dump_snapshot(str(path))
                        print(
                            cli_step_prefix(),
                            f"[mem] OOM in update[{args.trainer},text,native]; "
                            f"allocator history dumped to {path}. "
                            "Load at https://pytorch.org/memory_viz",
                            flush=True,
                        )
                    except Exception as dump_exc:
                        print(
                            cli_step_prefix(),
                            f"[mem] _dump_snapshot failed: {type(dump_exc).__name__}: {dump_exc}",
                            flush=True,
                        )
                raise
            finally:
                if snapshot_armed:
                    torch.cuda.memory._record_memory_history(enabled=None)
            trainer_label = "ppo,text,native"
        rl_lr_warmup_step(optimizer)
        now = time.monotonic()
        elapsed = now - last_step_time
        last_step_time = now
        fields = [
            cli_step_prefix(),
            f"final_update[{trainer_label}]" if final else f"update[{trainer_label}]",
            f"games={log_completed_games}",
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
        if rnad_state is None:
            _, rollout_predicted_values = backend.policy.gather_replay_old_log_prob_value(
                rollout_replay_rows
            )
            value_metrics = rollout_value_metrics(rollout_predicted_values, rollout_returns)
        else:
            value_metrics = rnad_value_metrics(rnad_state)
        log_ppo_stats(
            stats,
            games=log_completed_games,
            steps=rollout_step_count,
            total_rollout_steps=total_rollout_steps,
            total_generated_rollout_steps=total_generated_rollout_steps,
            win_stats=log_win_stats,
            value_metrics=value_metrics,
            token_metrics=(
                token_length_percentile_metrics(
                    backend.replay_buffer.row_token_length_host,
                    update_rows_host,
                )
                if update_rows_host
                else None
            ),
            log_fn=tracked_wandb_log,
            run_active=True,
        )
        if replay_rows_chunks is None and replay_rows_tensor is None:
            pending_replay_rows.clear()
            pending_replay_rows_host.clear()
            pending_returns.clear()
            pending_episodes.clear()
            pending_flat_rows_chunks.clear()
            pending_episode_step_counts.clear()
            with pending_step_count_lock:
                pending_step_count = 0
        if reset_replay:
            backend.replay_buffer.reset()
        if win_stats_snapshot is None:
            win_stats.reset()
        trained_completed_games = log_completed_games

    def run_text_rollouts_actor_loop() -> None:
        """IMPALA-style coordinator. Spins up N actor threads + 1 GPU server.

        The learner thread (this one) only:
          * drains the finished-env queue → finish_games(...);
          * answers refill requests by spawning new games;
          * runs PPO/R-NaD updates with the inference server paused;
          * fires periodic save / snapshot / retrospective hooks.

        All env stepping (poll, encode, step_by_choice) runs on the actor
        threads in parallel; cgo releases the GIL on each native call.
        """

        from magic_ai.native.inference_server import RolloutTimingStats, TextInferenceServer
        from magic_ai.native.rollout_actor import (
            ActorEncodeConfig,
            ActorRuntimeConfig,
            FinishedEnv,
            RefillRequest,
            RefillResponse,
            TextRolloutActor,
        )

        if not getattr(args, "text_native_assembler", True):
            raise SystemExit(
                "--num-rollout-actors > 1 requires --text-native-assembler "
                "(the actor path drives the native packed-token assembler)."
            )

        num_actors = int(getattr(args, "num_rollout_actors", 1))
        nonlocal sampling_policy
        publish_source: LSTMStatefulTextPolicy = (
            cast(LSTMStatefulTextPolicy, rnad_state.target)
            if rnad_state is not None
            else backend.policy
        )
        inference_policy = publish_source.clone_for_rnad().to(backend.policy.device)
        inference_policy.init_lstm_env_states(args.num_envs)
        sampling_policy = inference_policy
        policy_versions = PolicyVersionManager(
            online_policy=publish_source,
            inference_policy=inference_policy,
        )
        sampling_policy = cast(Any, policy_versions)

        encoders_pool = native_encoder.encoders
        drivers_pool = native_rollout.drivers
        # Carve off the last encoder/driver as a dedicated snapshot-eval pair
        # so snapshot eval never aliases an actor's scratch (the Go-side
        # scratchPool is keyed by output-buffer address; per-actor encoders
        # have distinct buffers, so a separate snapshot encoder is fully
        # lock-free against the actors).
        needed = num_actors + 1
        if len(encoders_pool) < needed:
            raise SystemExit(
                f"native_encoder has {len(encoders_pool)} workers but "
                f"--num-rollout-actors={num_actors} requires {needed} "
                f"(N actors + 1 dedicated snapshot encoder); "
                f"rebuild with --batch-workers={needed}"
            )
        if len(drivers_pool) < needed:
            raise SystemExit(
                f"native_rollout has {len(drivers_pool)} drivers but "
                f"--num-rollout-actors={num_actors} requires {needed} "
                f"(N actors + 1 dedicated snapshot driver); "
                f"rebuild with --batch-workers={needed}"
            )

        snapshot_encoder = ShardedNativeBatchEncoder(
            [encoders_pool[num_actors]], pool=None, shard_packed_tokens=False
        )
        snapshot_rollout = ShardedNativeRolloutDriver([drivers_pool[num_actors]], pool=None)

        encode_cfg = ActorEncodeConfig(
            max_tokens=int(backend.replay_buffer.max_tokens),
            max_state_tokens=int(args.text_max_tokens),
            max_options=int(args.max_options),
            max_targets_per_option=int(args.max_targets_per_option),
            max_card_refs=256,
        )
        runtime_cfg = ActorRuntimeConfig(
            max_steps_per_game=int(args.max_steps_per_game),
        )

        max_batch = int(args.inference_max_batch) or int(args.num_envs)
        min_batch_rows = max(
            1,
            min(
                max_batch,
                int(math.ceil(float(args.inference_ready_fraction) * float(args.num_envs))),
            ),
        )
        timing_stats = RolloutTimingStats()
        staging_buffer.timing_stats = timing_stats
        if rnad_state is None:
            # Compile the batched GAE graph before rollout timing starts;
            # otherwise the first large finish batch reports Inductor compile
            # time as learner_finish_games/finish_gae.
            with torch.no_grad():
                warm_b = max(1, min(int(args.num_envs), int(args.rollout_steps)))
                warm_t = int(args.max_steps_per_game)
                warm_device = staging_buffer.device
                warm_values = torch.zeros(warm_b, warm_t, dtype=torch.float32, device=warm_device)
                warm_players = torch.zeros(warm_b, warm_t, dtype=torch.long, device=warm_device)
                warm_counts = torch.ones(warm_b, dtype=torch.long, device=warm_device)
                warm_terminal = torch.zeros(warm_b, dtype=torch.float32, device=warm_device)
                warm_zero_sum = torch.ones(warm_b, dtype=torch.bool, device=warm_device)
                gae_returns_batched(
                    warm_values,
                    warm_players,
                    warm_counts,
                    terminal_reward_p0=warm_terminal,
                    zero_sum=warm_zero_sum,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
        server = TextInferenceServer(
            sampling_policy=policy_versions,
            max_batch=max_batch,
            deterministic=bool(args.deterministic_rollout),
            timing_stats=timing_stats,
            min_batch_rows=min_batch_rows,
            bucketed_inference=bool(args.torch_compile),
        )
        server.start()

        finished_q: Queue[FinishedEnv] = Queue()
        refill_req_q: Queue[RefillRequest] = Queue()
        refill_resp_qs: list[Queue[RefillResponse]] = [Queue() for _ in range(num_actors)]

        # Partition env slots across actors. We hand each actor a *static*
        # share of slot ids; refills always come from the same partition so
        # there is no cross-actor slot contention.
        all_slots = list(range(args.num_envs))
        slot_partitions: list[list[int]] = [[] for _ in range(num_actors)]
        for i, slot in enumerate(all_slots):
            slot_partitions[i % num_actors].append(slot)
        # free_slots in the closure is a single shared stack; partition it so
        # each actor has its own free pool, decoupled from the global free_slots.
        actor_free_slots: list[list[int]] = [list(reversed(p)) for p in slot_partitions]
        # Mark every slot as taken from the global free_slots view; the actor
        # path manages free slots per-actor.
        free_slots.clear()

        actor_errors: list[tuple[BaseException, str]] = []

        def _actor_error(exc: BaseException, tb: str) -> None:
            actor_errors.append((exc, tb))

        actors: list[TextRolloutActor] = []
        for actor_id in range(num_actors):
            actor = TextRolloutActor(
                actor_id=actor_id,
                encoder=encoders_pool[actor_id],
                rollout_driver=drivers_pool[actor_id],
                inference_server=server,
                staging_buffer=staging_buffer,
                replay_buffer=backend.replay_buffer,
                encode_cfg=encode_cfg,
                runtime_cfg=runtime_cfg,
                finished_queue=finished_q,
                refill_request_queue=refill_req_q,
                refill_response_queue=refill_resp_qs[actor_id],
                transcript_snapshot=_current_transcript_snapshot,
                disable_transcript=disable_transcript,
                append_transcript_action=lambda env, st, pe, ac: env.transcript.append(
                    TranscriptAction(state=st, pending=pe, action=copy.deepcopy(ac))
                ),
                record_step=lambda env, p, lp, v, r=None: env.episode_steps.append(
                    RolloutStep(
                        perspective_player_idx=int(p),
                        old_log_prob=float(lp),
                        value=float(v),
                        replay_idx=None if r is None else int(r),
                    )
                ),
                error_hook=_actor_error,
                timing_stats=timing_stats,
            )
            actors.append(actor)

        nonlocal next_episode_idx, last_saved_games

        def _spawn_initial(actor_id: int) -> list[Any]:
            # Initial games for actor `actor_id`: pull from its private free
            # stack, capped by remaining episodes.
            nonlocal next_episode_idx
            games: list[Any] = []
            while actor_free_slots[actor_id] and next_episode_idx < args.episodes:
                slot_idx = actor_free_slots[actor_id].pop()
                games.append(start_game(slot_idx, next_episode_idx))
                next_episode_idx += 1
            return games

        for actor_id, actor in enumerate(actors):
            initial = _spawn_initial(actor_id)
            actor.start(initial)

        active_actors = num_actors
        actor_done: list[bool] = [False] * num_actors
        pending_refill_slots: list[list[int]] = [[] for _ in range(num_actors)]
        deferred_finishes: list[FinishedEnv] = []
        last_progress_t = time.monotonic()
        last_progress_state = (completed_games, pending_step_count, next_episode_idx)
        watchdog_threshold_s = float(getattr(args, "actor_watchdog_seconds", 60.0))

        import os as _os

        _profile_secs = float(_os.environ.get("MAGIC_AI_PROFILE_RL_SECONDS", "0") or 0.0)
        _profile_out = _os.environ.get("MAGIC_AI_PROFILE_RL_OUTPUT", "/tmp/rl_profile.json.gz")
        _profile_state: dict[str, Any] = {"prof": None, "started_at": None, "done": False}
        _updates_seen = 0
        benchmark_samples: list[tuple[int, int, float]] = []
        benchmark_done = threading.Event()
        learner_stop = threading.Event()
        learner_wakeup = threading.Event()
        learner_force_partial = threading.Event()
        learner_errors: list[tuple[BaseException, str]] = []

        def _print_rollout_timing(stats: dict[str, float | int]) -> None:
            rows = int(stats.get("rows", 0))
            if rows <= 0:
                return
            fields = [
                cli_step_prefix(),
                "[rollout_timing]",
                f"rows={rows}",
                f"tokens/row={float(stats.get('avg_tokens_per_row', 0.0)):.1f}",
                f"max_seq={int(stats.get('max_seq', 0))}",
            ]
            for name in (
                "actor_poll",
                "actor_encode",
                "actor_pack",
                "actor_submit",
                "actor_wait_inference",
                "actor_append_replay",
                "actor_stage",
                "actor_record",
                "actor_step",
                "server_collect",
                "server_concat",
                "server_sample",
                "server_sample_move_text",
                "server_sample_lstm_state_in",
                "server_sample_forward",
                "server_sample_lstm_state_out",
                "server_sample_native_metadata_to_device",
                "server_sample_decision_init",
                "server_sample_decision_tensors_to_device",
                "server_sample_decision_full_layout",
                "server_sample_decision_priority_batch",
                "server_sample_decision_choice_batch",
                "server_sample_decision_binary_batch",
                "server_sample_decision_merge_batch",
                "server_sample_decision_inline_fast",
                "server_sample_decision_accept_check",
                "server_sample_decision_post_decision",
                "server_sample_decision_may",
                "server_sample_decision_sampling",
                "server_sample_replay_payload",
                "server_sample_append_replay",
                "server_sample_host_return",
                "server_scatter",
                "learner_finish_games",
                "finish_terminal_reward",
                "finish_metadata",
                "finish_gae",
                "finish_append_wait_lock",
                "finish_append_reset_slots",
                "finish_append_build_decoder",
                "finish_append_build_indices",
                "finish_append_payload",
                "finish_append_reserve",
                "finish_append_prepare_rows",
                "finish_append_encoded_rows",
                "finish_append_common_fields",
                "finish_append_decision_gather",
                "finish_append_lstm_gather",
                "finish_append_decoder_commit",
                "finish_append_commit",
                "finish_append_rows_tensor",
                "finish_close_bookkeeping",
                "learner_update",
            ):
                total = float(stats.get(f"{name}_total_s", 0.0))
                if total > 0.0:
                    mean = float(stats.get(f"{name}_mean_ms", 0.0))
                    count = int(stats.get(f"{name}_count", 0))
                    fields.append(f"{name}={total:.2f}s/{mean:.1f}msx{count}")
            print(*fields, flush=True)

        def _schedule_update(*, final: bool = False, force_partial: bool = False) -> bool:
            if final or force_partial:
                learner_force_partial.set()
            if pending_step_count <= 0:
                return False
            learner_wakeup.set()
            return True

        def _learner_loop() -> None:
            nonlocal _updates_seen, pending_step_count
            try:
                while not learner_stop.is_set() or pending_step_count > 0:
                    draining = learner_stop.is_set()
                    force_partial = learner_force_partial.is_set()
                    min_rows = 1 if draining or force_partial else int(args.learner_min_rows)
                    target_rows = 1 if draining or force_partial else int(args.learner_target_rows)
                    window = backend.replay_buffer.claim_train_window(
                        min_rows=min_rows,
                        max_rows=int(args.learner_max_rows),
                        target_rows=target_rows,
                        allow_partial=draining or force_partial,
                    )
                    if window is None:
                        if draining:
                            break
                        learner_wakeup.wait(timeout=0.05)
                        learner_wakeup.clear()
                        continue
                    update_start = time.perf_counter()
                    try:
                        if window.ready_events and backend.replay_buffer.device.type == "cuda":
                            stream = torch.cuda.current_stream(backend.replay_buffer.device)
                            for event in window.ready_events:
                                stream.wait_event(event)
                        steps = int(window.rows.numel())
                        if rnad_state is None:
                            returns = backend.replay_buffer.build_ppo_returns_for_rows(
                                window.rows,
                                gamma=args.gamma,
                                gae_lambda=args.gae_lambda,
                            )
                            run_update(
                                final=draining,
                                replay_rows_tensor=window.rows,
                                returns_tensor=returns,
                                step_count=steps,
                                replay_rows_host=range(window.row_start, window.row_end),
                                completed_games_snapshot=int(completed_games),
                                win_stats_snapshot=replace(win_stats),
                                reset_replay=False,
                            )
                        else:
                            run_update(
                                final=draining,
                                replay_rows_tensor=window.rows,
                                step_count=steps,
                                replay_rows_host=range(window.row_start, window.row_end),
                                completed_games_snapshot=int(completed_games),
                                win_stats_snapshot=replace(win_stats),
                                reset_replay=False,
                            )
                        backend.replay_buffer.release_train_window(window)
                        with pending_step_count_lock:
                            pending_step_count = max(0, pending_step_count - steps)
                        if force_partial:
                            learner_force_partial.clear()
                        policy_versions.publish_from_online(publish_source)
                        update_elapsed = time.perf_counter() - update_start
                        timing_stats.add("learner_update", update_elapsed)
                        if bool(getattr(args, "benchmark_mode", False)) and not draining:
                            warmup_updates = int(args.benchmark_warmup_updates)
                            steady_updates = int(args.benchmark_steady_updates)
                            if _updates_seen >= warmup_updates and steps >= target_rows:
                                benchmark_samples.append((steps, completed_games, update_elapsed))
                                if len(benchmark_samples) <= steady_updates:
                                    total_steps = sum(map(lambda item: item[0], benchmark_samples))
                                    total_s = sum(map(lambda item: item[2], benchmark_samples))
                                    print(
                                        cli_step_prefix(),
                                        "[benchmark]",
                                        f"steady_update={len(benchmark_samples)}/{steady_updates}",
                                        f"rows={steps}",
                                        f"update_s={update_elapsed:.4f}",
                                        f"steady_rows_per_s={total_steps / max(total_s, 1e-9):.1f}",
                                        flush=True,
                                    )
                                if len(benchmark_samples) >= steady_updates:
                                    benchmark_done.set()
                        _print_rollout_timing(timing_stats.snapshot_and_reset())
                        _updates_seen += 1
                    finally:
                        pass
            except BaseException as exc:  # noqa: BLE001
                import traceback as _traceback

                learner_errors.append((exc, _traceback.format_exc()))

        learner_thread = threading.Thread(
            target=_learner_loop,
            name="text-native-learner",
            daemon=True,
        )
        learner_thread.start()

        try:
            while active_actors > 0:
                active_actors = sum(
                    map(
                        lambda actor: int(actor._thread is not None and actor._thread.is_alive()),
                        actors,
                    )
                )
                if active_actors <= 0:
                    break
                if actor_errors:
                    exc, tb = actor_errors[0]
                    raise RuntimeError(f"rollout actor crashed: {tb}") from exc
                if learner_errors:
                    exc, tb = learner_errors[0]
                    raise RuntimeError(f"learner thread crashed: {tb}") from exc
                if benchmark_done.is_set():
                    break

                # Watchdog: if nothing has changed for a while, dump diagnostic
                # state. This makes startup-time hangs (e.g. server thread
                # crashed before resolving the first batch) visible instead
                # of looking like an unconditional deadlock.
                now = time.monotonic()
                cur_state = (completed_games, pending_step_count, next_episode_idx)
                if cur_state != last_progress_state:
                    last_progress_t = now
                    last_progress_state = cur_state
                elif now - last_progress_t > watchdog_threshold_s:
                    print(
                        cli_step_prefix(),
                        f"[watchdog] no rollout progress for "
                        f"{int(now - last_progress_t)}s — "
                        f"completed_games={completed_games} "
                        f"pending_steps={pending_step_count} "
                        f"next_ep={next_episode_idx} "
                        f"server_alive={server._thread.is_alive()} "
                        f"server_q={server._queue.qsize()} "
                        f"server_queued_rows={server._queue.queued_rows()} "
                        f"finished_q={finished_q.qsize()} "
                        f"deferred_finishes={len(deferred_finishes)} "
                        f"refill_req_q={refill_req_q.qsize()} "
                        f"pending_refill_slots={[len(s) for s in pending_refill_slots]} "
                        f"actor_free_slots={[len(s) for s in actor_free_slots]} "
                        f"replay_available_rows={backend.replay_buffer.available_rows} "
                        f"replay_available_tokens={backend.replay_buffer.available_tokens} "
                        f"learner_force_partial={learner_force_partial.is_set()} "
                        f"actor_done={actor_done}",
                        flush=True,
                    )
                    replay_debug = (
                        backend.replay_buffer.debug_snapshot()
                        if hasattr(backend.replay_buffer, "debug_snapshot")
                        else {}
                    )
                    if replay_debug:
                        print(
                            cli_step_prefix(),
                            f"[watchdog_replay] {replay_debug}",
                            flush=True,
                        )
                    starved_refills = [
                        aid
                        for aid, requested in enumerate(pending_refill_slots)
                        if requested and actor_free_slots[aid] and next_episode_idx < args.episodes
                    ]
                    if starved_refills:
                        print(
                            cli_step_prefix(),
                            f"[watchdog_refill_starved] actors={starved_refills}",
                            flush=True,
                        )
                    server_ident = server._thread.ident
                    if server_ident is not None:
                        frame = sys._current_frames().get(server_ident)
                        if frame is not None:
                            import traceback as _traceback

                            print(
                                cli_step_prefix(),
                                "[watchdog_server_stack]",
                                "".join(_traceback.format_stack(frame, limit=8)),
                                flush=True,
                            )
                    learner_ident = learner_thread.ident
                    if learner_ident is not None:
                        frame = sys._current_frames().get(learner_ident)
                        if frame is not None:
                            import traceback as _traceback

                            print(
                                cli_step_prefix(),
                                "[watchdog_learner_stack]",
                                "".join(_traceback.format_stack(frame, limit=12)),
                                flush=True,
                            )
                    for actor in actors:
                        thread = actor._thread
                        actor_ident = None if thread is None else thread.ident
                        if actor_ident is None:
                            continue
                        frame = sys._current_frames().get(actor_ident)
                        if frame is None:
                            continue
                        import traceback as _traceback

                        print(
                            cli_step_prefix(),
                            f"[watchdog_actor_stack actor={actor.actor_id}]",
                            "".join(_traceback.format_stack(frame, limit=8)),
                            flush=True,
                        )
                    last_progress_t = now

                # Drain a batch of finished envs (up to ~num_envs at once).
                finished_batch: list[FinishedEnv] = []
                try:
                    first = finished_q.get(timeout=0.05)
                    finished_batch.append(first)
                except Empty:
                    pass
                while True:
                    try:
                        finished_batch.append(finished_q.get_nowait())
                    except Empty:
                        break

                # Try to commit any previously-deferred finishes first, then
                # the freshly-arrived batch. Anything that won't fit in the
                # current packed replay buffer stays deferred until the next
                # PPO update frees rows.
                deferred_finishes.extend(finished_batch)
                fits: list[FinishedEnv] = []
                if deferred_finishes:
                    still_deferred: list[FinishedEnv] = []
                    candidate_slots: list[int] = []
                    for fe in deferred_finishes:
                        next_slots = [*candidate_slots, fe.slot_idx]
                        if staging_buffer.can_append_envs_to_replay(
                            next_slots,
                            backend.replay_buffer,
                        ):
                            candidate_slots = next_slots
                            fits.append(fe)
                        else:
                            still_deferred.append(fe)
                    deferred_finishes = still_deferred
                if fits:
                    finish_start = time.perf_counter()
                    tuple(
                        map(
                            lambda fe: setattr(fe.live_game, "actor_id", int(fe.actor_id)),
                            fits,
                        )
                    )
                    finish_games(
                        list(
                            map(
                                lambda fe: (fe.live_game, fe.winner_idx, fe.is_timeout),
                                fits,
                            )
                        )
                    )
                    timing_stats.add("learner_finish_games", time.perf_counter() - finish_start)
                    # Return slots to the *originating* actor's pool only
                    # for envs we actually committed.
                    per_actor_freed: list[list[int]] = list(map(lambda _idx: [], range(num_actors)))
                    tuple(
                        map(
                            lambda fe: per_actor_freed[int(fe.actor_id)].append(fe.slot_idx),
                            fits,
                        )
                    )
                    tuple(
                        map(
                            lambda item: actor_free_slots[item[0]].extend(item[1]),
                            filter(lambda item: bool(item[1]), enumerate(per_actor_freed)),
                        )
                    )
                if deferred_finishes and pending_step_count > 0:
                    _schedule_update(force_partial=True)

                if pending_step_count >= int(args.learner_min_rows):
                    _schedule_update()

                # Service refill requests.
                while True:
                    try:
                        req = refill_req_q.get_nowait()
                    except Empty:
                        break
                    pending_refill_slots[req.actor_id].extend(req.slot_indices)
                for aid, requested in enumerate(pending_refill_slots):
                    if not requested:
                        continue
                    # Slots requested *back* are slots the actor already
                    # released via FinishedEnv; they are already back in
                    # actor_free_slots[aid] above. Pull from there.
                    new_games: list[Any] = []
                    no_more = False
                    while requested and actor_free_slots[aid] and next_episode_idx < args.episodes:
                        requested.pop()
                        slot_idx = actor_free_slots[aid].pop()
                        new_games.append(start_game(slot_idx, next_episode_idx))
                        next_episode_idx += 1
                    if next_episode_idx >= args.episodes:
                        no_more = True
                        requested.clear()
                    if new_games or no_more:
                        refill_resp_qs[aid].put(
                            RefillResponse(games=new_games, no_more_episodes=no_more)
                        )
                    if no_more and not new_games and not actor_done[aid]:
                        # Actor's slice has truly drained; mark it.
                        actor_done[aid] = True
                        active_actors -= 1

                # Periodic update scheduling is non-blocking; the learner
                # thread publishes the next inference version when complete.
                if pending_step_count >= int(args.learner_min_rows):
                    _schedule_update()
                    # Start torch.profiler after the first post-compile update
                    # (one full rollout + update has triggered torch.compile
                    # for both the inference forward and the update path).
                    if (
                        _profile_secs > 0
                        and _profile_state["prof"] is None
                        and not _profile_state["done"]
                        and _updates_seen >= 1
                    ):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        prof = torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=False,
                            with_stack=False,
                            profile_memory=False,
                        )
                        prof.start()
                        _profile_state["prof"] = prof
                        _profile_state["started_at"] = time.monotonic()
                        print(
                            cli_step_prefix(),
                            f"[profile] torch.profiler started; will run for "
                            f"{_profile_secs:.0f}s, output={_profile_out}",
                            flush=True,
                        )

                _profile_started_at = _profile_state["started_at"]
                if (
                    _profile_state["prof"] is not None
                    and not _profile_state["done"]
                    and _profile_started_at is not None
                    and time.monotonic() - float(_profile_started_at) >= _profile_secs
                ):
                    prof = _profile_state["prof"]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    prof.stop()
                    print(
                        cli_step_prefix(),
                        "[profile] torch.profiler stopped; computing key_averages…",
                        flush=True,
                    )
                    skip_export = _os.environ.get("MAGIC_AI_PROFILE_RL_SKIP_EXPORT") == "1"
                    try:
                        print(
                            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40),
                            flush=True,
                        )
                        print(
                            prof.key_averages().table(sort_by="cpu_time_total", row_limit=40),
                            flush=True,
                        )
                    except Exception as _e:
                        print(f"[profile] key_averages table failed: {_e}", flush=True)
                    if not skip_export:
                        out_path = Path(_profile_out)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        prof.export_chrome_trace(str(out_path))
                        print(
                            cli_step_prefix(),
                            f"[profile] trace written to {out_path}",
                            flush=True,
                        )
                    _profile_state["done"] = True
                    _profile_state["prof"] = None
                    raise SystemExit(0)

                # Save / snapshot / retrospective (mirrors the legacy loop).
                if (
                    args.save_every
                    and trained_completed_games > 0
                    and trained_completed_games >= last_saved_games + args.save_every
                ):
                    save_checkpoint(
                        args.output,
                        backend.policy,
                        optimizer,
                        args,
                        opponent_pool=opponent_pool,
                        snapshot_schedule=snapshot_schedule,
                        retrospective_schedule=retrospective_schedule,
                        resume_state=TrainingResumeState(
                            completed_games=trained_completed_games,
                            last_saved_games=trained_completed_games,
                            total_rollout_steps=total_rollout_steps,
                            total_generated_rollout_steps=total_generated_rollout_steps,
                            total_wandb_logs=total_wandb_logs,
                        ),
                        rnad_state=rnad_state,
                    )
                    last_saved_games = trained_completed_games

                if (
                    opponent_pool is not None
                    and snapshot_schedule is not None
                    and opponent_policy is not None
                ):
                    pending_thresholds = list(snapshot_schedule.fire(completed_games))
                    if pending_thresholds:
                        # GPU is shared, so pause the inference server to
                        # serialize forward passes; actors keep running their
                        # CPU poll/encode/step loop and will block at most one
                        # batch deep on the server queue. The dedicated
                        # snapshot encoder/driver means snapshot eval cannot
                        # race actors on Go-side scratch pools either.
                        server.pause()
                        try:
                            for threshold in pending_thresholds:
                                take_snapshot_and_eval(
                                    args=args,
                                    threshold=threshold,
                                    policy=backend.policy,
                                    opponent_policy=opponent_policy,
                                    opponent_pool=opponent_pool,
                                    native_encoder=snapshot_encoder,
                                    native_rollout=snapshot_rollout,
                                    mage=mage,
                                    deck_pool=deck_pool,
                                    rng=eval_rng,
                                    step_prefix=cli_step_prefix(),
                                    log_fn=tracked_wandb_log,
                                )
                                if wandb.run is not None:
                                    log_retrospective_table(
                                        wandb.run,
                                        horizon_pct=int(
                                            round(threshold * 100 / max(1, args.episodes))
                                        ),
                                        horizon_step_count=threshold,
                                        ratings=retrospective_rating_rows(
                                            opponent_pool,
                                            total_episodes=args.episodes,
                                        ),
                                        log_fn=tracked_wandb_log,
                                    )
                        finally:
                            server.resume()
            if pending_step_count > 0 and not benchmark_done.is_set():
                while not _schedule_update(final=True):
                    time.sleep(0.01)
            learner_stop.set()
            learner_wakeup.set()
            learner_thread.join()
            if learner_errors:
                exc, tb = learner_errors[0]
                raise RuntimeError(f"learner thread crashed: {tb}") from exc
            backend.replay_buffer.reset()
        finally:
            learner_stop.set()
            # Stop the inference server first so any actor blocked in
            # ``future.result()`` wakes immediately (queued futures are
            # rejected). Then signal all actors in parallel before joining,
            # so a Ctrl-C shutdown isn't serialized across N actors.
            try:
                server.stop()
            except Exception:
                pass
            for actor in actors:
                actor.signal_stop()
            for actor in actors:
                actor.join(timeout=1.0)
            learner_thread.join(timeout=1.0)

    run_text_rollouts_actor_loop()
    return (
        TrainingResumeState(
            completed_games=trained_completed_games,
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
    policy: torch.nn.Module,
    opponent_policy: torch.nn.Module,
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
    current_entry = opponent_pool.add_snapshot(snapshot_path, tag, snapshot_games=threshold)
    current_entry.cached_policy = opponent_policy_state_dict(policy)
    print(
        step_prefix,
        f"pool: snapshot {tag} -> {snapshot_path} (pool size={len(opponent_pool.entries)})",
        flush=True,
    )

    eval_games_per_snapshot = (
        args.eval_games_per_snapshot
        if args.eval_games_per_snapshot is not None
        else max(400, args.episodes // 625)
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
    text_max_tokens = args.text_max_tokens if getattr(args, "encoder", "slots") == "text" else None
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
        text_max_tokens=text_max_tokens,
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

    own_mu = float(current_entry.rating.mu)
    own_sigma = float(current_entry.rating.sigma)
    print(
        step_prefix,
        f"eval: snapshot_tag={tag} own_rating=mu={own_mu:.2f},sigma={own_sigma:.2f}",
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


def load_deck_pool(
    deck_json: Path | None,
    deck_dir: Path | None,
    jumpstart_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if jumpstart_dir is not None:
        return load_deck_dir(jumpstart_dir)
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
    *,
    fixed: bool = False,
    jumpstart: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not deck_pool:
        raise ValueError("deck pool must contain at least one deck")
    if fixed:
        # deck_json supplies exactly [player_a, player_b] in order; don't randomize
        return deck_pool[0], deck_pool[1]
    rng = random.Random(seed)
    if jumpstart:
        if len(deck_pool) < 2:
            raise ValueError("jumpstart pool must contain at least two packs")
        a1, a2 = rng.sample(deck_pool, 2)
        b1, b2 = rng.sample(deck_pool, 2)
        return _merge_packs(a1, a2), _merge_packs(b1, b2)
    return rng.choice(deck_pool), rng.choice(deck_pool)


def _merge_packs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for pack in (a, b):
        for card in pack.get("cards", []):
            name = card["name"]
            counts[name] = counts.get(name, 0) + int(card.get("count", 1))
    return {
        "name": f"{a.get('name', '?')}+{b.get('name', '?')}",
        "cards": [{"name": n, "count": c} for n, c in counts.items()],
    }


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
    post_mlm: bool = False,
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
            "target": _model_checkpoint_state_dict(cast(PPOPolicy, rnad_state.target)),
            "reg_snapshot_dir": str(rnad_state.reg_snapshot_dir),
        }
    metadata = {
        "encoder": getattr(args, "encoder", "slots"),
        "wandb_run_id": effective_wandb_run_id,
        "run_artifact_dir": str(effective_run_artifact_dir),
    }
    if post_mlm:
        metadata["post_mlm"] = True
    if getattr(args, "encoder", "slots") == "text":
        cache_path = Path(getattr(args, "card_token_cache", ""))
        actual_text_cfg = getattr(
            getattr(getattr(policy, "policy", None), "text_policy", None), "cfg", None
        )
        actual_decoder_cfg = getattr(
            getattr(
                getattr(getattr(policy, "policy", None), "text_policy", None),
                "grammar_decoder",
                None,
            ),
            "cfg",
            None,
        )
        metadata["text_config"] = {
            "text_max_tokens": getattr(args, "text_max_tokens", None),
            "text_encoder_backend": getattr(args, "text_encoder_backend", "scratch"),
            "text_hf_model": getattr(args, "text_hf_model", None),
            "text_hf_revision": getattr(args, "text_hf_revision", None),
            "text_hf_layers": getattr(args, "text_hf_layers", None),
            "text_d_model": getattr(
                actual_text_cfg, "d_model", getattr(args, "text_d_model", None)
            ),
            "text_layers": getattr(actual_text_cfg, "n_layers", getattr(args, "text_layers", None)),
            "text_heads": getattr(actual_text_cfg, "n_heads", getattr(args, "text_heads", None)),
            "text_d_ff": getattr(actual_text_cfg, "d_ff", getattr(args, "text_d_ff", None)),
            "decoder_layers": getattr(
                actual_decoder_cfg, "n_layers", getattr(args, "decoder_layers", None)
            ),
            "decoder_heads": getattr(
                actual_decoder_cfg, "n_heads", getattr(args, "decoder_heads", None)
            ),
            "decoder_d_ff": getattr(
                actual_decoder_cfg, "d_ff", getattr(args, "decoder_d_ff", None)
            ),
            "decoder_max_decode_len": getattr(
                actual_decoder_cfg,
                "max_decode_len",
                getattr(args, "decoder_max_decode_len", None),
            ),
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
            "policy": _model_checkpoint_state_dict(policy),
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
    for line in sample_game_lines(
        transcript,
        winner_idx=winner_idx,
        max_actions=max_actions,
    ):
        print(line)
    print()


def sample_game_lines(
    transcript: list[TranscriptAction],
    *,
    winner_idx: int,
    max_actions: int,
) -> list[str]:
    lines: list[str] = []
    if not transcript:
        lines.append("(no actions)")
    else:
        condensed = condense_transcript_lines(transcript, max_actions=max_actions)
        if condensed:
            turn_width = max(len(line["turn"]) for line in condensed)
            step_width = max(len(line["step"]) for line in condensed)
            player_width = max(len(line["player"]) for line in condensed)
            life_width = max(len(line["life"]) for line in condensed)
            for line in condensed:
                lines.append(
                    f"{line['turn']:<{turn_width}}  "
                    f"{line['step']:<{step_width}}  "
                    f"{line['player']:<{player_width}}  "
                    f"{line['life']:<{life_width}}  "
                    f"{line['action']}"
                )
        remaining = len(transcript) - max_actions
        if remaining > 0:
            lines.append(f"... {remaining} more actions")
    if winner_idx >= 0:
        lines.append(f"== PLAYER {winner_idx + 1} WINS ==")
    else:
        lines.append("== DRAW ==")
    return lines


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
