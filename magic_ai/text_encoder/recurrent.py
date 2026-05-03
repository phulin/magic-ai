"""Recurrent wrapper around :class:`TextPolicy` (history adapter v1).

Implements the v1 history strategy from ``docs/text_encoder_plan.md`` §12: a
single-layer LSTM around the encoder's pooled ``state_vector``. The LSTM's
top-layer output replaces ``state_vector`` when re-scoring the policy / target
/ value heads, so history is carried entirely through the recurrent state with
zero changes to the heads themselves and zero changes to RnaD math.

This module is self-contained: no edits to ``PPOPolicy`` / ``model.py`` /
``buffer.py``. The eventual ``policy.encoder = "text"`` integration in
``magic_ai/model.py`` is expected to construct one of these and route the
``(h_out, c_out)`` pair through the existing recurrent-state plumbing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.policy import EncodedSnapshots, TextPolicy


@dataclass
class RecurrentTextPolicyConfig:
    encoder: TextEncoderConfig
    lstm_hidden: int = 384  # match d_model by default
    lstm_layers: int = 1
    compile_forward: bool = True


@dataclass
class RecurrentTextPolicyOutput:
    policy_logits: Tensor  # [B, max_opts]
    target_logits: Tensor  # [B, max_opts, max_targets]
    values: Tensor  # [B]
    state_hidden: Tensor  # [B, lstm_hidden]
    option_vectors: Tensor  # [B, max_opts, d_model]
    option_mask: Tensor
    target_vectors: Tensor
    target_mask: Tensor
    card_vectors: Tensor
    card_mask: Tensor
    lstm_input: Tensor | None = None  # [B, lstm_hidden] in_proj(state_vector); None when bypassed


class RecurrentTextPolicy(nn.Module):
    """LSTM-around-state-vector history adapter for :class:`TextPolicy`."""

    def __init__(self, cfg: RecurrentTextPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_policy = TextPolicy(cfg.encoder)

        d_model = cfg.encoder.d_model
        self.d_model = d_model
        self.lstm_hidden = cfg.lstm_hidden
        self.lstm_layers = cfg.lstm_layers

        # Projection in / out so lstm_hidden may differ from d_model. When they
        # match these collapse to identity-shaped Linears at trivial cost; the
        # alternative ("require lstm_hidden == d_model") loses nothing in
        # behaviour and a touch of flexibility, so we keep the projections.
        self.in_proj = nn.Linear(d_model, cfg.lstm_hidden)
        self.out_proj = nn.Linear(cfg.lstm_hidden, d_model)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        self.lstm = nn.LSTM(
            input_size=cfg.lstm_hidden,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        self._compiled_forward_packed: (
            Callable[
                [PackedTextBatch, Tensor | None, Tensor | None, Tensor | None],
                tuple[RecurrentTextPolicyOutput, tuple[Tensor, Tensor]],
            ]
            | None
        ) = None
        # Compile is wired up lazily on the first CUDA forward (see
        # ``forward_packed``). flash_attn_varlen replaced the old NJT-subclass
        # attention so the historical AOT-autograd backward mismatch no longer
        # applies; CPU compile, however, hits an unrelated inductor bug, so we
        # only compile when we actually see a CUDA tensor.

    def init_state(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        shape = (self.lstm_layers, batch_size, self.lstm_hidden)
        h = torch.zeros(shape, device=device)
        c = torch.zeros(shape, device=device)
        return h, c

    def forward(
        self,
        batch: TextEncodedBatch,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
        *,
        state_hidden_override: Tensor | None = None,
    ) -> tuple[RecurrentTextPolicyOutput, tuple[Tensor, Tensor]]:
        # bf16 autocast wraps the entire forward (encoder + LSTM + heads).
        # bf16 has fp32's exponent range so no grad scaler is needed. autocast
        # is responsible for promoting any op that's not bf16-safe (cuDNN's
        # LSTM is on the eligible list); skipping the explicit fp32 round-trip
        # avoids six per-forward .to(fp32) copies of the encoded vectors.
        device_type = batch.token_ids.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            encoded = self.text_policy.encode_only(batch)
            return self._forward_encoded(
                encoded,
                h_in=h_in,
                c_in=c_in,
                state_hidden_override=state_hidden_override,
            )

    def forward_packed(
        self,
        batch: PackedTextBatch,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
        *,
        state_hidden_override: Tensor | None = None,
    ) -> tuple[RecurrentTextPolicyOutput, tuple[Tensor, Tensor]]:
        if (
            self.cfg.compile_forward
            and self._compiled_forward_packed is None
            and batch.token_ids.device.type == "cuda"
        ):
            self._compiled_forward_packed = cast(
                Callable[
                    [PackedTextBatch, Tensor | None, Tensor | None, Tensor | None],
                    tuple[RecurrentTextPolicyOutput, tuple[Tensor, Tensor]],
                ],
                torch.compile(self._forward_packed_impl, dynamic=True),
            )
        if self._compiled_forward_packed is not None:
            return self._compiled_forward_packed(batch, h_in, c_in, state_hidden_override)
        return self._forward_packed_impl(batch, h_in, c_in, state_hidden_override)

    def _forward_packed_impl(
        self,
        batch: PackedTextBatch,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
        state_hidden_override: Tensor | None = None,
    ) -> tuple[RecurrentTextPolicyOutput, tuple[Tensor, Tensor]]:
        device_type = batch.token_ids.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            encoded = self.text_policy.encode_packed_only(batch)
            return self._forward_encoded(
                encoded,
                h_in=h_in,
                c_in=c_in,
                state_hidden_override=state_hidden_override,
            )

    def _forward_encoded(
        self,
        encoded: EncodedSnapshots,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
        *,
        state_hidden_override: Tensor | None = None,
    ) -> tuple[RecurrentTextPolicyOutput, tuple[Tensor, Tensor]]:
        b = encoded.state_vector.shape[0]
        device = encoded.state_vector.device
        if h_in is None or c_in is None:
            h_in, c_in = self.init_state(b, device)

        lstm_input: Tensor | None
        if state_hidden_override is not None:
            if state_hidden_override.shape != (b, self.lstm_hidden):
                raise ValueError(
                    "state_hidden_override must have shape "
                    f"({b}, {self.lstm_hidden}); got {tuple(state_hidden_override.shape)}"
                )
            state_hidden = state_hidden_override
            h_out, c_out = h_in, c_in
            lstm_input = None
        else:
            lstm_input = self.in_proj(encoded.state_vector)
            y, (h_out, c_out) = self.lstm(lstm_input.unsqueeze(1), (h_in, c_in))
            state_hidden = y.squeeze(1)

        state_for_heads = self.out_proj(state_hidden)
        policy_logits, target_logits, values = self.text_policy.run_heads(
            encoded, state_vec=state_for_heads
        )

        out = RecurrentTextPolicyOutput(
            policy_logits=policy_logits,
            target_logits=target_logits,
            values=values,
            state_hidden=state_hidden,
            option_vectors=encoded.option_vectors,
            option_mask=encoded.option_mask,
            target_vectors=encoded.target_vectors,
            target_mask=encoded.target_mask,
            card_vectors=encoded.card_vectors,
            card_mask=encoded.card_mask,
            lstm_input=lstm_input,
        )
        return out, (h_out, c_out)

    def forward_from_encoded(
        self,
        encoded: EncodedSnapshots,
        state_hidden: Tensor,
    ) -> RecurrentTextPolicyOutput:
        """Run only the heads given precomputed encoder outputs and LSTM hidden.

        Used by R-NaD's batched-trajectory path: the encoder forward and the
        per-episode LSTM scan are run once per policy via
        ``TextActorCritic.precompute_replay_forward``; the resulting ``encoded``
        and ``state_hidden`` are then fed here directly so the per-choice scoring
        forward does not re-run the encoder. Wraps the head pass in bf16
        autocast on CUDA so callers can hand in bf16 ``encoded`` directly.
        """

        device_type = encoded.state_vector.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            state_for_heads = self.out_proj(state_hidden)
            policy_logits, target_logits, values = self.text_policy.run_heads(
                encoded, state_vec=state_for_heads
            )
        return RecurrentTextPolicyOutput(
            policy_logits=policy_logits,
            target_logits=target_logits,
            values=values,
            state_hidden=state_hidden,
            option_vectors=encoded.option_vectors,
            option_mask=encoded.option_mask,
            target_vectors=encoded.target_vectors,
            target_mask=encoded.target_mask,
            card_vectors=encoded.card_vectors,
            card_mask=encoded.card_mask,
        )


__all__ = [
    "RecurrentTextPolicy",
    "RecurrentTextPolicyConfig",
    "RecurrentTextPolicyOutput",
]
