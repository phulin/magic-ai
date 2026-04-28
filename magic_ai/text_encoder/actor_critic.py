"""Training-facing actor-critic wrapper for the text encoder policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
    RecurrentTextPolicyOutput,
)


@dataclass(frozen=True)
class TextActorCriticStep:
    output: RecurrentTextPolicyOutput
    h_out: Tensor
    c_out: Tensor


class TextActorCritic(nn.Module):
    """Thin stateful wrapper around :class:`RecurrentTextPolicy`.

    The wrapper owns live per-env, per-player recurrent state. Replay storage
    remains external in ``TextReplayBuffer`` so PPO/R-NaD can swap algorithms
    without changing the model wrapper.
    """

    def __init__(self, cfg: RecurrentTextPolicyConfig) -> None:
        super().__init__()
        self.policy = RecurrentTextPolicy(cfg)
        self.lstm_layers = cfg.lstm_layers
        self.lstm_hidden = cfg.lstm_hidden
        self.register_buffer("live_lstm_h", torch.zeros(self.lstm_layers, 0, self.lstm_hidden))
        self.register_buffer("live_lstm_c", torch.zeros(self.lstm_layers, 0, self.lstm_hidden))
        self._num_envs = 0
        self._players_per_env = 2

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_lstm_env_states(self, num_envs: int, *, players_per_env: int = 2) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if players_per_env < 1:
            raise ValueError("players_per_env must be at least 1")
        self._num_envs = int(num_envs)
        self._players_per_env = int(players_per_env)
        slots = self._num_envs * self._players_per_env
        shape = (self.lstm_layers, slots, self.lstm_hidden)
        self.live_lstm_h = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.live_lstm_c = torch.zeros_like(self.live_lstm_h)

    def reset_lstm_env_states(self, env_indices: list[int]) -> None:
        if self._num_envs == 0:
            raise RuntimeError("LSTM env states have not been initialized")
        slots = self._state_slots(env_indices, None)
        self.live_lstm_h[:, slots] = 0
        self.live_lstm_c[:, slots] = 0

    def lstm_env_state_inputs(
        self,
        env_indices: list[int],
        perspective_player_indices: list[int],
    ) -> tuple[Tensor, Tensor]:
        slots = self._state_slots(env_indices, perspective_player_indices)
        return (
            self.live_lstm_h[:, slots].contiguous(),
            self.live_lstm_c[:, slots].contiguous(),
        )

    def forward_live(
        self,
        batch: TextEncodedBatch,
        *,
        env_indices: list[int],
        perspective_player_indices: list[int],
    ) -> TextActorCriticStep:
        h_in, c_in = self.lstm_env_state_inputs(env_indices, perspective_player_indices)
        output, (h_out, c_out) = self.policy(batch, h_in=h_in, c_in=c_in)
        self.scatter_lstm_env_states(env_indices, perspective_player_indices, h_out, c_out)
        return TextActorCriticStep(output=output, h_out=h_out, c_out=c_out)

    def scatter_lstm_env_states(
        self,
        env_indices: list[int],
        perspective_player_indices: list[int],
        h_out: Tensor,
        c_out: Tensor,
    ) -> None:
        slots = self._state_slots(env_indices, perspective_player_indices)
        expected = (self.lstm_layers, len(slots), self.lstm_hidden)
        if tuple(h_out.shape) != expected or tuple(c_out.shape) != expected:
            raise ValueError(
                f"state outputs must have shape {expected}, got {tuple(h_out.shape)} "
                f"and {tuple(c_out.shape)}"
            )
        self.live_lstm_h[:, slots] = h_out.detach()
        self.live_lstm_c[:, slots] = c_out.detach()

    def _state_slots(
        self,
        env_indices: list[int],
        perspective_player_indices: list[int] | None,
    ) -> Tensor:
        if self._num_envs == 0:
            raise RuntimeError("LSTM env states have not been initialized")
        if perspective_player_indices is not None and len(env_indices) != len(
            perspective_player_indices
        ):
            raise ValueError("env_indices and perspective_player_indices must have equal length")
        slots: list[int] = []
        if perspective_player_indices is None:
            for env_idx in env_indices:
                self._validate_env_idx(env_idx)
                start = int(env_idx) * self._players_per_env
                slots.extend(range(start, start + self._players_per_env))
        else:
            for env_idx, player_idx in zip(env_indices, perspective_player_indices, strict=True):
                self._validate_env_idx(env_idx)
                if player_idx < 0 or player_idx >= self._players_per_env:
                    raise IndexError("perspective player index out of range")
                slots.append(int(env_idx) * self._players_per_env + int(player_idx))
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def _validate_env_idx(self, env_idx: int) -> None:
        if env_idx < 0 or env_idx >= self._num_envs:
            raise IndexError("env index out of range")


__all__ = ["TextActorCritic", "TextActorCriticStep"]
