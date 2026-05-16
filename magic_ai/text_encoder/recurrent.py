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
from torch.fx._symbolic_trace import is_fx_symbolic_tracing

from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch
from magic_ai.text_encoder.decoder import GrammarDecoderConfig
from magic_ai.text_encoder.model import TextEncoderConfig
from magic_ai.text_encoder.policy import EncodedSnapshots, TextPolicy


@dataclass
class RecurrentTextPolicyConfig:
    encoder: TextEncoderConfig
    lstm_hidden: int = 384  # match d_model by default
    lstm_layers: int = 1
    compile_forward: bool = True
    grammar_decoder_cfg: GrammarDecoderConfig | None = None


@dataclass
class RecurrentTextPolicyOutput:
    values: Tensor  # [B]
    state_hidden: Tensor  # [B, lstm_hidden]
    card_vectors: Tensor
    card_mask: Tensor
    lstm_input: Tensor | None = None  # [B, lstm_hidden] in_proj(state_vector); None when bypassed


class RecurrentTextPolicy(nn.Module):
    """LSTM-around-state-vector history adapter for :class:`TextPolicy`."""

    def __init__(self, cfg: RecurrentTextPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_policy = TextPolicy(
            cfg.encoder,
            decoder_cfg=cfg.grammar_decoder_cfg,
        )

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

        # History projection: top-layer LSTM hidden -> additive embedding at
        # each document's BOS position. Zero-initialized so existing
        # checkpoints behave like the no-hist version at init time and the
        # encoder learns to use history gradually.
        self.hist_proj = nn.Linear(cfg.lstm_hidden, d_model)
        nn.init.zeros_(self.hist_proj.weight)
        nn.init.zeros_(self.hist_proj.bias)

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
        self._compiled_encode_with_history_packed: (
            Callable[
                [PackedTextBatch, Tensor | None, Tensor | None],
                tuple[EncodedSnapshots, Tensor, Tensor],
            ]
            | None
        ) = None
        # Compile is wired up lazily on the first CUDA forward (see
        # ``forward_packed``). flash_attn_varlen replaced the old NJT-subclass
        # attention so the historical AOT-autograd backward mismatch no longer
        # applies; CPU compile, however, hits an unrelated inductor bug, so we
        # only compile when we actually see a CUDA tensor.
        #
        # Inference goes through ``TextInferencePipeline``'s bucketed compile
        # (Phase C). Replay scoring can call ``encode_with_history`` directly,
        # so it gets its own lazy packed compile point when --torch-compile is
        # enabled.

    def init_state(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        shape = (self.lstm_layers, batch_size, self.lstm_hidden)
        h = torch.zeros(shape, device=device)
        c = torch.zeros(shape, device=device)
        return h, c

    def _normalize_lstm_final_state(self, state: Tensor, batch_size: int, name: str) -> Tensor:
        expected = (self.lstm_layers, batch_size, self.lstm_hidden)
        if tuple(state.shape) == expected:
            return state
        squeezed = state.squeeze(1) if state.dim() == 4 and state.shape[1] == 1 else state
        if tuple(squeezed.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}; got {tuple(state.shape)}")
        return squeezed

    def _hist_emb(
        self,
        b: int,
        device: torch.device,
        h_in: Tensor | None,
        c_in: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        """Resolve initial (h_in, c_in) and the additive hist embedding.

        Returns ``(hist_emb, h_in, c_in)`` where ``hist_emb`` is ``None`` when
        ``h_in`` was ``None`` (zero-init: hist_proj of zeros is bias only,
        which is also zero at construction; we skip the inject entirely so
        the no-history path is exactly equivalent to a pre-history encoder).
        """
        if h_in is None or c_in is None:
            h_init, c_init = self.init_state(b, device)
            # hist_proj of zeros is zero (zero-init weight + zero bias), so
            # skip the inject entirely when there's no carried state.
            return None, h_init, c_init
        # Top-layer hidden state drives the history projection.
        return self.hist_proj(h_in[-1]), h_in, c_in

    def encoder_forward_padded_with_history(
        self,
        batch: TextEncodedBatch,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the non-packed encoder.forward with hist injection + LSTM update.

        Returns ``(encoded [B, T, D], h_out, c_out)``. Used by sample-time
        paths whose downstream decoder cross-attn requires a padded
        rank-3 hidden tensor.
        """
        from magic_ai.text_encoder.model import gather_state_vector

        b = int(batch.token_ids.shape[0])
        device = batch.token_ids.device
        hist_emb, h_in_eff, c_in_eff = self._hist_emb(b, device, h_in, c_in)
        encoded = self.text_policy.encoder(batch, hist_emb=hist_emb)
        state_vec = gather_state_vector(encoded, batch)
        lstm_input = self.in_proj(state_vec)
        h_in_eff = h_in_eff.to(device=device, dtype=lstm_input.dtype)
        c_in_eff = c_in_eff.to(device=device, dtype=lstm_input.dtype)
        _, (h_out, c_out) = self.lstm(lstm_input.unsqueeze(1), (h_in_eff, c_in_eff))
        h_out = self._normalize_lstm_final_state(h_out, b, "h_out")
        c_out = self._normalize_lstm_final_state(c_out, b, "c_out")
        return encoded, h_out, c_out

    def encode_with_history(
        self,
        batch: PackedTextBatch | TextEncodedBatch,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
    ) -> tuple[EncodedSnapshots, Tensor, Tensor]:
        """Run encoder with history-conditioning + LSTM update.

        The bucketed inference path (:class:`TextInferencePipeline`)
        compiles ``_encode_with_history_impl`` directly per bucket. Replay
        scoring uses this method on packed batches, so --torch-compile
        lazily compiles that packed encode-only path on CUDA.
        """
        if (
            isinstance(batch, PackedTextBatch)
            and self.cfg.compile_forward
            and batch.token_ids.device.type == "cuda"
        ):
            if self._compiled_encode_with_history_packed is None:
                self._compiled_encode_with_history_packed = cast(
                    Callable[
                        [PackedTextBatch, Tensor | None, Tensor | None],
                        tuple[EncodedSnapshots, Tensor, Tensor],
                    ],
                    torch.compile(self._encode_with_history_impl, dynamic=True),
                )
            if not is_fx_symbolic_tracing():
                return self._compiled_encode_with_history_packed(batch, h_in, c_in)
        return self._encode_with_history_impl(batch, h_in, c_in)

    def _encode_with_history_impl(
        self,
        batch: PackedTextBatch | TextEncodedBatch,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
    ) -> tuple[EncodedSnapshots, Tensor, Tensor]:
        if isinstance(batch, PackedTextBatch):
            b = int(batch.seq_lengths.shape[0])
            device = batch.token_ids.device
        else:
            b = int(batch.token_ids.shape[0])
            device = batch.token_ids.device
        hist_emb, h_in_eff, c_in_eff = self._hist_emb(b, device, h_in, c_in)
        if isinstance(batch, PackedTextBatch):
            encoded = self.text_policy.encode_packed_only(batch, hist_emb=hist_emb)
        else:
            encoded = self.text_policy.encode_only(batch, hist_emb=hist_emb)
        lstm_input = self.in_proj(encoded.state_vector)
        h_in_eff = h_in_eff.to(device=device, dtype=lstm_input.dtype)
        c_in_eff = c_in_eff.to(device=device, dtype=lstm_input.dtype)
        _, (h_out, c_out) = self.lstm(lstm_input.unsqueeze(1), (h_in_eff, c_in_eff))
        h_out = self._normalize_lstm_final_state(h_out, b, "h_out")
        c_out = self._normalize_lstm_final_state(c_out, b, "c_out")
        return encoded, h_out, c_out

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
            return self._run(
                batch,
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
        if self._compiled_forward_packed is not None and not is_fx_symbolic_tracing():
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
            return self._run(
                batch,
                h_in=h_in,
                c_in=c_in,
                state_hidden_override=state_hidden_override,
            )

    def _run(
        self,
        batch: PackedTextBatch | TextEncodedBatch,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
        *,
        state_hidden_override: Tensor | None = None,
    ) -> tuple[RecurrentTextPolicyOutput, tuple[Tensor, Tensor]]:
        if state_hidden_override is not None:
            # Head-only path: caller provides a precomputed state_hidden and
            # the LSTM update is skipped. ``state_hidden_override`` must
            # already be the result of a real LSTM update somewhere upstream
            # (otherwise the heads see stale state — see :meth:`forward`).
            if isinstance(batch, PackedTextBatch):
                encoded = self.text_policy.encode_packed_only(batch, hist_emb=None)
            else:
                encoded = self.text_policy.encode_only(batch, hist_emb=None)
            b = encoded.state_vector.shape[0]
            if state_hidden_override.shape != (b, self.lstm_hidden):
                raise ValueError(
                    "state_hidden_override must have shape "
                    f"({b}, {self.lstm_hidden}); got {tuple(state_hidden_override.shape)}"
                )
            state_hidden = state_hidden_override
            device = encoded.state_vector.device
            if h_in is None or c_in is None:
                h_in, c_in = self.init_state(b, device)
            h_out, c_out = h_in, c_in
            lstm_input = None
        else:
            # Call the impl directly: ``_run`` itself is (lazily) wrapped by
            # ``forward_packed``'s compile, so we must not nest a second
            # compile by going through ``encode_with_history``'s wrapper.
            encoded, h_out, c_out = self._encode_with_history_impl(batch, h_in=h_in, c_in=c_in)
            lstm_input = self.in_proj(encoded.state_vector)
            # The top LSTM layer's hidden output is what feeds the heads.
            state_hidden = h_out[-1]

        state_for_heads = self.out_proj(state_hidden)
        values = self.text_policy.run_heads(encoded, state_vec=state_for_heads)

        out = RecurrentTextPolicyOutput(
            values=values,
            state_hidden=state_hidden,
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
        ``LSTMStatefulTextPolicy.precompute_replay_forward``; the resulting ``encoded``
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
            values = self.text_policy.run_heads(encoded, state_vec=state_for_heads)
        return RecurrentTextPolicyOutput(
            values=values,
            state_hidden=state_hidden,
            card_vectors=encoded.card_vectors,
            card_mask=encoded.card_mask,
        )


__all__ = [
    "RecurrentTextPolicy",
    "RecurrentTextPolicyConfig",
    "RecurrentTextPolicyOutput",
]
