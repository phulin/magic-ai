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

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.policy import EncodedSnapshots, TextPolicy


def _cast_encoded(encoded: EncodedSnapshots, dtype: torch.dtype) -> EncodedSnapshots:
    """Cast the float vectors of ``encoded`` to ``dtype`` (boolean masks
    are left untouched). Used to bring autocast'd encoder outputs back to
    the LSTM/head parameter dtype."""

    return EncodedSnapshots(
        card_vectors=encoded.card_vectors.to(dtype),
        card_mask=encoded.card_mask,
        option_vectors=encoded.option_vectors.to(dtype),
        option_mask=encoded.option_mask,
        target_vectors=encoded.target_vectors.to(dtype),
        target_mask=encoded.target_mask,
        state_vector=encoded.state_vector.to(dtype),
    )


@dataclass
class RecurrentTextPolicyConfig:
    encoder: TextEncoderConfig
    lstm_hidden: int = 384  # match d_model by default
    lstm_layers: int = 1
    compile_forward: bool = False


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

        if cfg.compile_forward:
            # Only compile the packed (rollout-only) path. The non-packed
            # forward runs during PPO training with .backward(), and AOT
            # autograd currently mismatches NJT subclass metadata on the
            # backward pass when compiled.
            self.forward_packed = torch.compile(self.forward_packed, dynamic=True)  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

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
        # bf16 autocast on CUDA scoped to the encoder: cuts attention memory
        # ~2x and lets SDPA dispatch to the mem-efficient kernel (it skips
        # the math-backend full [B, H, T, T] score tensor when inputs are
        # bf16/fp16). bf16 has fp32's exponent range so no grad scaler is
        # needed. We narrow the cm to the encoder block because the LSTM
        # below holds fp32 state buffers and mixing dtypes through it is a
        # pile of compatibility issues for marginal additional savings.
        device_type = batch.token_ids.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            encoded = self.text_policy.encode_only(batch)
        # Cast encoder outputs back to the LSTM's parameter dtype so the
        # downstream LSTM + heads run in fp32.
        target_dtype = self.in_proj.weight.dtype
        encoded = _cast_encoded(encoded, target_dtype)
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
        device_type = batch.token_ids.device.type
        autocast_enabled = device_type == "cuda"
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            encoded = self.text_policy.encode_packed_only(batch)
        target_dtype = self.in_proj.weight.dtype
        encoded = _cast_encoded(encoded, target_dtype)
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

        if state_hidden_override is not None:
            if state_hidden_override.shape != (b, self.lstm_hidden):
                raise ValueError(
                    "state_hidden_override must have shape "
                    f"({b}, {self.lstm_hidden}); got {tuple(state_hidden_override.shape)}"
                )
            state_hidden = state_hidden_override
            h_out, c_out = h_in, c_in
        else:
            x = self.in_proj(encoded.state_vector).unsqueeze(1)
            y, (h_out, c_out) = self.lstm(x, (h_in, c_in))
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
        forward does not re-run the encoder. Caller is responsible for casting
        ``encoded`` to the head-parameter dtype.
        """

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
