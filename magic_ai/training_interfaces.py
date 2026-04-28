"""Trainer-facing policy interfaces.

These protocols describe the surface that PPO/R-NaD trainers need from a
policy implementation. They intentionally do not model rollout collection or
state encoding; slot and text backends can satisfy these interfaces with
different buffers and encoders.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from torch import Tensor, nn

from magic_ai.buffer import RolloutBuffer


class PPOReplayPolicy(Protocol):
    """Minimal policy surface required by :func:`magic_ai.ppo.ppo_update`."""

    spr_enabled: bool

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def evaluate_replay_batch(
        self,
        replay_rows: list[int],
        *,
        return_extras: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Any | None]: ...

    def compute_spr_loss(
        self,
        step_indices: Tensor,
        *,
        extras: Any | None = None,
    ) -> Tensor: ...

    def update_spr_target(self, decay: float | None = None) -> None: ...


class RNaDReplayPolicy(Protocol):
    """Replay policy surface required by the R-NaD trajectory losses."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def evaluate_replay_batch_per_choice(
        self,
        replay_rows: list[int],
        *,
        lstm_state_override: tuple[Tensor, Tensor] | None = None,
        hidden_override: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Any]: ...

    def recompute_lstm_states_for_episode(
        self,
        replay_rows: list[int],
    ) -> tuple[Tensor, Tensor] | None: ...

    def recompute_lstm_outputs_for_episodes(
        self,
        episodes: list[list[int]],
        *,
        chunk_size: int = 200,
        compiled_lstm: Any | None = None,
    ) -> list[Tensor] | None: ...


class RNaDTrainablePolicy(RNaDReplayPolicy, Protocol):
    """Online R-NaD policy surface used by the trainer orchestration."""

    rollout_buffer: RolloutBuffer

    def evaluate_replay_batch(
        self,
        replay_rows: list[int],
    ) -> tuple[Tensor, Tensor, Tensor, Any]: ...

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]: ...

    def train(self, mode: bool = True) -> RNaDTrainablePolicy: ...

    def eval(self) -> RNaDTrainablePolicy: ...

    def to(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> RNaDTrainablePolicy: ...
