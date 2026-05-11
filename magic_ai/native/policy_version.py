"""Thread-safe policy-version handoff for native rollout inference."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import Any, cast

import torch
from torch import nn

from magic_ai.model_state import is_actor_runtime_state_key


def _copy_model_state(dst: nn.Module, src: nn.Module, *, runtime_state: bool) -> None:
    dst_state = dst.state_dict()
    state = dict(
        filter(
            lambda item: (
                is_actor_runtime_state_key(item[0]) == runtime_state
                and item[0] in dst_state
                and tuple(item[1].shape) == tuple(dst_state[item[0]].shape)
            ),
            src.state_dict().items(),
        )
    )
    dst.load_state_dict(state, strict=False)


class PolicyVersionManager:
    """Owns a double-buffered inference policy and monotonically increasing version.

    The learner copies updated weights into the inactive buffer, waits for any
    active server forward to finish, mirrors runtime LSTM state from the active
    policy, then swaps the active pointer. The wait is short and only gates the
    pointer swap; learner gradient work still overlaps actor rollout.
    """

    def __init__(self, *, online_policy: nn.Module, inference_policy: nn.Module) -> None:
        self._online = online_policy
        self._active = inference_policy
        self._inactive = (
            cast(Any, inference_policy).clone_for_rnad().to(cast(Any, inference_policy).device)
        )
        if int(getattr(self._active, "_num_envs", 0)) > 0:
            self._inactive.init_lstm_env_states(
                int(getattr(self._active, "_num_envs")),
                players_per_env=int(getattr(self._active, "_players_per_env", 2)),
            )
        self._active.eval()
        self._inactive.eval()
        self._active.requires_grad_(False)
        self._inactive.requires_grad_(False)
        device = next(self._active.parameters()).device
        self._copy_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._publish_event: torch.cuda.Event | None = (
            torch.cuda.Event(blocking=False) if device.type == "cuda" else None
        )
        self._publish_event_ready = False
        _copy_model_state(self._active, self._online, runtime_state=False)
        _copy_model_state(self._inactive, self._online, runtime_state=False)
        self._version = 0
        self._active_readers = 0
        self._swapping = False
        self._cond = threading.Condition()

    @property
    def version(self) -> int:
        with self._cond:
            return self._version

    @property
    def device(self) -> torch.device:
        return next(self._active.parameters()).device

    @contextlib.contextmanager
    def acquire_inference_policy(self, max_lag: int | None = None) -> Iterator[tuple[Any, int]]:
        del max_lag
        with self._cond:
            while self._swapping:
                self._cond.wait()
            self._active_readers += 1
            policy = self._active
            version = self._version
            event = self._publish_event if self._publish_event_ready else None
        if event is not None:
            torch.cuda.current_stream(device=next(policy.parameters()).device).wait_event(event)
        try:
            yield policy, version
        finally:
            with self._cond:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._cond.notify_all()

    def publish_from_online(self, policy: nn.Module | None = None) -> int:
        """Publish the current online weights as the next inference version."""

        src = self._online if policy is None else policy
        if self._copy_stream is None:
            _copy_model_state(self._inactive, src, runtime_state=False)
        else:
            with torch.cuda.stream(self._copy_stream):
                _copy_model_state(self._inactive, src, runtime_state=False)
                cast(torch.cuda.Event, self._publish_event).record(self._copy_stream)
            self._publish_event_ready = True
        with self._cond:
            self._swapping = True
            while self._active_readers > 0:
                self._cond.wait()
            if self._publish_event is not None and self._publish_event_ready:
                torch.cuda.current_stream(
                    device=next(self._inactive.parameters()).device
                ).wait_event(self._publish_event)
            _copy_model_state(self._inactive, self._active, runtime_state=True)
            self._active, self._inactive = self._inactive, self._active
            self._version += 1
            version = self._version
            self._swapping = False
            self._cond.notify_all()
            return version

    def reset_lstm_env_states(self, env_indices: list[int]) -> None:
        # Don't wait for active readers: the slots being reset have just
        # been freed (game finished, slot recycled) and by construction
        # are disjoint from any in-flight inference's env_indices, so the
        # writes don't race the readers. Waiting on _active_readers > 0
        # deadlocks under slow first-call compiles (the server thread
        # holds the reader counter for the whole compile). We still wait
        # on _swapping to serialize with publish_from_online, which IS a
        # global write.
        with self._cond:
            while self._swapping:
                self._cond.wait()
            self._swapping = True
        try:
            cast(Any, self._active).reset_lstm_env_states(env_indices)
            cast(Any, self._inactive).reset_lstm_env_states(env_indices)
        finally:
            with self._cond:
                self._swapping = False
                self._cond.notify_all()


__all__ = ["PolicyVersionManager"]
