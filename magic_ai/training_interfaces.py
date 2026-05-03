"""Trainer-facing policy interfaces.

These protocols describe the surface that PPO/R-NaD trainers need from a
policy implementation. They intentionally do not model rollout collection or
state encoding; slot and text backends can satisfy these interfaces with
different buffers and encoders.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from torch import Tensor, nn


class PPOReplayPolicy(Protocol):
    """Minimal policy surface required by :func:`magic_ai.ppo.ppo_update`."""

    spr_enabled: bool

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def evaluate_replay_batch(
        self,
        replay_rows: list[int] | Tensor,
        *,
        return_extras: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Any | None]: ...

    def write_ppo_targets(
        self,
        replay_rows: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> None: ...

    def gather_ppo_targets(self, replay_rows: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...

    def gather_replay_old_log_prob_value(
        self,
        replay_rows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(old_log_prob, value)`` already on-device for the given rows.

        Used by :func:`magic_ai.ppo.ppo_update` to skip the per-step host
        roundtrip that materializing ``RolloutStep`` lists would require.
        """
        ...

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

    def evaluate_replay_batch(
        self,
        replay_rows: list[int],
    ) -> tuple[Tensor, Tensor, Tensor, Any]: ...

    def count_active_replay_steps(
        self,
        per_episode_replay_rows: Sequence[Sequence[int]],
    ) -> tuple[int, int]:
        """Return ``(cl_count, pl_count)`` totals for the given replay rows.

        ``cl_count`` is the total replay-step count (paper §164 critic-loss
        normalizer). ``pl_count`` is the may-active step count plus the total
        number of valid (decision-group, choice-column) cells across active
        decision groups (NeuRD policy-loss normalizer). Backends compute this
        from their own replay-buffer layout.
        """
        ...

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

    def clone_for_rnad(self) -> RNaDTrainablePolicy:
        """Return a clone suitable for use as a target / regularization policy.

        Implementations must:

        - Deep-copy trainable parameter state.
        - Share the rollout buffer with ``self`` (target/reg policies replay
          the same trajectories the online policy collected).
        - Reset any per-env actor-runtime caches (e.g. live LSTM hidden state)
          so the clone starts from an empty rollout state — target/reg
          policies never sample from a live env.
        """
        ...
