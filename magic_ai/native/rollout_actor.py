"""Per-thread rollout actor for the native text-encoder path.

Each actor owns:

* a disjoint slice of env slots (and the ``LiveGame`` objects in them),
* its own ``NativeBatchEncoder`` and ``NativeRolloutDriver`` (cgo releases
  the GIL so concurrent native calls run truly in parallel),
* a handle to a shared :class:`TextInferenceServer` (the only thread that
  touches the GPU policy during rollouts).

The actor loop mirrors the inline ``run_text_rollouts`` body — poll →
partition → encode → server submit → step → record — using one per-actor
work queue for refills, poll batches, and inference completions. Completed
inference futures enqueue their continuation back onto that same queue, so
the actor only owns one scheduling stream.

The learner is the only thread that mutates global rollout state
(``pending_steps``, replay metadata,
``free_slots``/``next_episode_idx``); the actor never touches those.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

from magic_ai.actions import TRACE_KIND_VALUES, action_from_choice_accepted
from magic_ai.native.inference_server import (
    RolloutTimingStats,
    TextInferenceRequest,
    TextInferenceServer,
)

if TYPE_CHECKING:
    from magic_ai.slot_encoder.native_encoder import NativeBatchEncoder
    from magic_ai.slot_encoder.native_rollout import NativeRolloutDriver


@dataclass
class ActorEncodeConfig:
    text_max_tokens: int
    max_options: int
    max_targets_per_option: int
    max_card_refs: int = 256


@dataclass
class ActorRuntimeConfig:
    max_steps_per_game: int


@dataclass
class FinishedEnv:
    """Posted from actor → learner when an env's game ends."""

    actor_id: int
    slot_idx: int
    winner_idx: int  # -1 if step-capped without winner OR engine-declared draw
    is_timeout: bool  # True if step-capped (game not is_over); distinguishes timeouts from draws
    live_game: Any  # the original LiveGame, with .episode_steps + .transcript filled


@dataclass
class RefillRequest:
    """Posted from actor → learner asking for new games for these freed slots."""

    actor_id: int
    slot_indices: list[int]


@dataclass
class RefillResponse:
    """Learner → actor; either fresh LiveGames or an empty list = no more episodes."""

    games: list[Any]  # list of LiveGame
    no_more_episodes: bool = False


@dataclass
class _InFlightBatch:
    ready_envs: list[Any]
    ready_games: list[Any]
    ready_players: list[int]
    ready_env_indices: list[int]
    submitted_at: float
    future: Future[Any]


@dataclass
class _SubmitBatch:
    ready_envs: list[Any]
    ready_games: list[Any]
    ready_players: list[int]
    ready_env_indices: list[int]
    native_batch: Any
    packed: Any


@dataclass
class _PollBatch:
    envs: list[Any]


@dataclass
class _InferenceDone:
    batch: _InFlightBatch


@dataclass
class TextRolloutActor:
    actor_id: int
    encoder: NativeBatchEncoder
    rollout_driver: NativeRolloutDriver
    inference_server: TextInferenceServer
    staging_buffer: Any
    encode_cfg: ActorEncodeConfig
    runtime_cfg: ActorRuntimeConfig
    finished_queue: Queue[FinishedEnv]
    refill_request_queue: Queue[RefillRequest]
    refill_response_queue: Queue[Any]
    transcript_snapshot: Callable[..., tuple[Any, Any]]
    decode_text_action: Callable[..., tuple[Any, Any]]
    disable_transcript: Callable[..., None]
    append_transcript_action: Callable[..., None]
    record_step: Callable[..., None]
    error_hook: Callable[[BaseException, str], None] | None = None
    timing_stats: RolloutTimingStats | None = None
    name: str = "text-actor"
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _no_more_episodes: bool = False
    _thread: threading.Thread | None = None
    _pending_inference: int = 0
    _blocked_submits: deque[_SubmitBatch] = field(default_factory=deque)

    # ---------------------------------------------------------------- lifecycle

    def start(self, initial_games: list[Any]) -> None:
        if initial_games:
            self._no_more_episodes = False
            self.refill_response_queue.put_nowait(RefillResponse(games=list(initial_games)))
        self._thread = threading.Thread(
            target=self._run, name=f"{self.name}-{self.actor_id}", daemon=True
        )
        self._thread.start()

    def signal_stop(self) -> None:
        """Set the stop flag and wake the actor without waiting for it.

        Lets a caller signal every actor in parallel before joining any of
        them, so a Ctrl-C shutdown isn't serialized across N × join-timeout.
        """

        self._stop_event.set()
        # Unblock the actor if it's waiting on work.
        try:
            self.refill_response_queue.put_nowait(RefillResponse(games=[], no_more_episodes=True))
        except Exception:
            pass

    def join(self, timeout: float = 1.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def stop(self) -> None:
        self.signal_stop()
        self.join(timeout=1.0)

    # --------------------------------------------------------------- internal

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._blocked_submits:
                    blocked = self._blocked_submits.popleft()
                    if not self._try_submit_batch(blocked, retry=True):
                        timeout = 0.001 if self._pending_inference > 0 else 0.005
                        try:
                            item = self.refill_response_queue.get(timeout=timeout)
                        except Empty:
                            pass
                        else:
                            self._process_work_item(item)
                    continue
                try:
                    item = self.refill_response_queue.get(timeout=0.05)
                except Empty:
                    if self._no_more_episodes and self._pending_inference <= 0:
                        return
                    if self._pending_inference > 0:
                        self.inference_server.flush()
                    continue
                self._process_work_item(item)
        except BaseException as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            if self.error_hook is not None:
                try:
                    self.error_hook(exc, tb)
                except Exception:
                    pass

    def _process_work_item(self, item: Any) -> None:
        if isinstance(item, RefillResponse):
            if item.no_more_episodes:
                self._no_more_episodes = True
            if item.games:
                self.refill_response_queue.put_nowait(_PollBatch(list(item.games)))
            return
        if isinstance(item, _PollBatch):
            self._poll_and_submit(item.envs)
            return
        if isinstance(item, _InferenceDone):
            self._pending_inference = max(0, self._pending_inference - 1)
            envs = self._finish_inference(item.batch)
            if envs:
                self._poll_and_submit(envs)
            return
        raise TypeError(f"unknown actor work item: {type(item).__name__}")

    def _poll_and_submit(self, envs: list[Any]) -> None:
        if not envs:
            return
        tick_start = time.perf_counter()
        start = tick_start
        games = [env.game for env in envs]
        ready_t, over_t, player_t, winner_t = self.rollout_driver.poll(games)
        self._record_timing("actor_poll", start)
        ready_l = ready_t.tolist()
        over_l = over_t.tolist()
        player_l = player_t.tolist()
        winner_l = winner_t.tolist()

        start = time.perf_counter()
        ready_envs: list[Any] = []
        ready_players: list[int] = []
        not_ready_envs: list[Any] = []
        freed_slots: list[int] = []
        for idx, env in enumerate(envs):
            is_over = bool(over_l[idx])
            if is_over or env.action_count >= self.runtime_cfg.max_steps_per_game:
                self.finished_queue.put(
                    FinishedEnv(
                        actor_id=self.actor_id,
                        slot_idx=int(env.slot_idx),
                        winner_idx=int(winner_l[idx]) if is_over else -1,
                        is_timeout=not is_over,
                        live_game=env,
                    )
                )
                freed_slots.append(int(env.slot_idx))
                continue
            if ready_l[idx]:
                ready_envs.append(env)
                ready_players.append(int(player_l[idx]))
            else:
                not_ready_envs.append(env)

        if freed_slots:
            self.refill_request_queue.put(
                RefillRequest(actor_id=self.actor_id, slot_indices=freed_slots)
            )

        self._record_timing("actor_partition_refill", start)

        if not_ready_envs:
            self.refill_response_queue.put_nowait(_PollBatch(not_ready_envs))
        if not ready_envs:
            if self._pending_inference > 0:
                self.inference_server.flush()
            return

        ready_games = [env.game for env in ready_envs]
        ready_env_indices = [int(env.slot_idx) for env in ready_envs]

        start = time.perf_counter()
        native_batch, nat_outputs = self._encode_packed(ready_games, ready_players)
        self._record_timing("actor_encode", start)
        start = time.perf_counter()
        packed = nat_outputs.to_packed_text_batch(trim=True, derive_token_metadata=True)
        self._record_timing("actor_pack", start)

        batch = _SubmitBatch(
            ready_envs=ready_envs,
            ready_games=ready_games,
            ready_players=ready_players,
            ready_env_indices=ready_env_indices,
            native_batch=native_batch,
            packed=packed,
        )
        self._try_submit_batch(batch)
        self._record_timing("actor_tick", tick_start)

    def _try_submit_batch(self, batch: _SubmitBatch, *, retry: bool = False) -> bool:
        self._drain_completed_inference()
        start = time.perf_counter()
        request = TextInferenceRequest(
            native_batch=batch.native_batch,
            packed_batch=batch.packed,
            env_indices=batch.ready_env_indices,
            perspective_player_indices=batch.ready_players,
        )
        try_submit = getattr(self.inference_server, "try_submit", None)
        future = (
            try_submit(request) if try_submit is not None else self.inference_server.submit(request)
        )
        self._record_timing("actor_submit", start)
        if future is None:
            self.inference_server.flush()
            if retry:
                self._blocked_submits.appendleft(batch)
            else:
                self._blocked_submits.append(batch)
            return False
        in_flight = _InFlightBatch(
            ready_envs=batch.ready_envs,
            ready_games=batch.ready_games,
            ready_players=batch.ready_players,
            ready_env_indices=batch.ready_env_indices,
            submitted_at=time.perf_counter(),
            future=future,
        )
        setattr(future, "_magic_ai_actor_batch", in_flight)
        self._pending_inference += 1
        future.add_done_callback(self._enqueue_inference_done)
        return True

    def _enqueue_inference_done(self, future: Future[Any]) -> None:
        try:
            batch = getattr(future, "_magic_ai_actor_batch")
        except AttributeError:
            return
        self.refill_response_queue.put_nowait(_InferenceDone(batch=batch))

    def _finish_inference(self, batch: _InFlightBatch) -> list[Any]:
        reply = batch.future.result()
        self._record_timing("actor_wait_inference", batch.submitted_at)
        try:
            if reply.replay_rows is None:
                start = time.perf_counter()
                self.staging_buffer.stage_batch(
                    batch.ready_env_indices,
                    reply.replay_payload,
                    ready_event=getattr(reply, "ready_event", None),
                )
                self._record_timing("actor_stage", start)
        finally:
            release_item = getattr(reply, "release_item", None)
            if release_item is not None:
                release_item()
        # Build transcripts + RolloutSteps (CPU-only, actor-local) and then
        # advance the engine. Transcript work uses host-side scalars from the
        # reply, so no additional GPU sync is needed.
        start = time.perf_counter()
        cursor = 0
        for step_idx, (env, player_idx) in enumerate(
            zip(batch.ready_envs, batch.ready_players, strict=True)
        ):
            env.action_count += 1
            env.inference_policy_version = int(getattr(reply, "inference_policy_version", 0))
            env.behavior_policy_version = int(env.inference_policy_version)
            count = int(reply.decision_counts[step_idx])
            replay_idx = int(reply.replay_rows[step_idx]) if reply.replay_rows is not None else None
            selected_for_step = reply.selected_choice_cols[cursor : cursor + count]
            cursor += count
            if env.transcript_enabled:
                try:
                    state, pending = self.transcript_snapshot(env.game)
                    trace_kind = TRACE_KIND_VALUES[int(reply.trace_kind_id[step_idx])]
                    if trace_kind == "may":
                        action = action_from_choice_accepted(bool(reply.may_selected[step_idx]))
                    else:
                        _trace, action = self.decode_text_action(
                            trace_kind, pending, list(selected_for_step)
                        )
                    self.append_transcript_action(env, state, pending, action)
                except Exception as exc:  # noqa: BLE001
                    self.disable_transcript(
                        env, f"{exc} while snapshotting live game for native text action"
                    )
            self.record_step(
                env,
                int(player_idx),
                float(reply.old_log_prob[step_idx]),
                float(reply.value[step_idx]),
                replay_idx,
            )
        self._record_timing("actor_record", start)

        # CPU step_by_choice (parallel across actors thanks to GIL release in cgo).
        start = time.perf_counter()
        starts: list[int] = []
        running = 0
        for c in reply.decision_counts:
            starts.append(running)
            running += int(c)
        self.rollout_driver.step_by_choice(
            batch.ready_games,
            decision_starts=starts,
            decision_counts=list(reply.decision_counts),
            selected_choice_cols=list(reply.selected_choice_cols),
            may_selected=list(reply.may_selected),
            max_options=self.encode_cfg.max_options,
            max_targets_per_option=self.encode_cfg.max_targets_per_option,
        )
        self._record_timing("actor_step", start)
        return list(batch.ready_envs)

    def _drain_completed_inference(self) -> None:
        deferred: list[Any] = []
        completed_envs: list[Any] = []
        while True:
            try:
                item = self.refill_response_queue.get_nowait()
            except Empty:
                break
            if isinstance(item, _InferenceDone):
                self._pending_inference = max(0, self._pending_inference - 1)
                completed_envs.extend(self._finish_inference(item.batch))
            else:
                deferred.append(item)
        for item in deferred:
            self.refill_response_queue.put_nowait(item)
        if completed_envs:
            self.refill_response_queue.put_nowait(_PollBatch(completed_envs))

    def _encode_packed(self, games: list[Any], perspectives: list[int]) -> tuple[Any, Any]:
        from magic_ai.text_encoder.native_assembler import encode_tokens_packed

        return encode_tokens_packed(
            self.encoder,
            games,
            perspective_player_indices=perspectives,
            max_tokens=self.encode_cfg.text_max_tokens,
            max_options=self.encode_cfg.max_options,
            max_targets=self.encode_cfg.max_targets_per_option,
            max_card_refs=self.encode_cfg.max_card_refs,
            include_trace_kinds=False,
        )

    def _record_timing(self, name: str, start: float) -> None:
        if self.timing_stats is not None:
            self.timing_stats.add(name, time.perf_counter() - start)


__all__ = [
    "ActorEncodeConfig",
    "ActorRuntimeConfig",
    "FinishedEnv",
    "RefillRequest",
    "RefillResponse",
    "TextRolloutActor",
]
