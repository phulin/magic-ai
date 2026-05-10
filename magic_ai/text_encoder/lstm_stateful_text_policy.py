"""Stateful wrapper around :class:`RecurrentTextPolicy`.

Owns live per-env, per-player LSTM state buffers and exposes the
polymorphic surface the RL trainers (PPO, R-NaD) expect. Sampling and
replay scoring delegate to the module-level helpers in
:mod:`decoder_inference`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from magic_ai.text_encoder.batch import TextEncodedBatch
from magic_ai.text_encoder.decoder_batch import (
    DecoderDecisionLayout,
    NativeTextSampleBatch,
)
from magic_ai.text_encoder.decoder_inference import (
    build_replay_grammar_masks,
    decoder_sample,
    decoder_score_replay,
)
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy, RecurrentTextPolicyConfig
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer


class LSTMStatefulTextPolicy(nn.Module):
    """Owns per-env / per-player LSTM state for the recurrent text policy and
    exposes the trainer-facing sampling / replay-scoring surface.

    Phase 5 reduced this class from ~3500 LoC to a thin shell. R-NaD's
    fused per-policy forward (``evaluate_replay_batch_per_choice``,
    ``precompute_replay_forward``) used to drive a hundred lines of inline-
    blank scoring; the equivalent decoder-based wiring lives in Phase 6.
    """

    spr_enabled: bool = False

    def __init__(self, cfg: RecurrentTextPolicyConfig) -> None:
        super().__init__()
        self.policy = RecurrentTextPolicy(cfg)
        self.lstm_layers = cfg.lstm_layers
        self.lstm_hidden = cfg.lstm_hidden
        self.register_buffer("live_lstm_h", torch.zeros(self.lstm_layers, 0, self.lstm_hidden))
        self.register_buffer("live_lstm_c", torch.zeros(self.lstm_layers, 0, self.lstm_hidden))
        self._num_envs = 0
        self._players_per_env = 2
        self.rollout_buffer: TextReplayBuffer | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def clone_for_rnad(self) -> LSTMStatefulTextPolicy:
        """Deep-copy the underlying policy weights for R-NaD's target / reg copies."""
        clone = LSTMStatefulTextPolicy(self.policy.cfg)
        clone.load_state_dict(self.state_dict())
        return clone

    def init_lstm_env_states(self, num_envs: int, *, players_per_env: int = 2) -> None:
        self._num_envs = int(num_envs)
        self._players_per_env = int(players_per_env)
        total = self._num_envs * self._players_per_env
        self.live_lstm_h = torch.zeros(
            self.lstm_layers, total, self.lstm_hidden, device=self.device
        )
        self.live_lstm_c = torch.zeros(
            self.lstm_layers, total, self.lstm_hidden, device=self.device
        )

    def reset_lstm_env_states(self, env_indices: list[int]) -> None:
        if not env_indices:
            return
        for env_idx in env_indices:
            base = int(env_idx) * self._players_per_env
            self.live_lstm_h[:, base : base + self._players_per_env, :].zero_()
            self.live_lstm_c[:, base : base + self._players_per_env, :].zero_()

    def _lstm_slots(
        self,
        env_indices: Sequence[int],
        perspective_player_indices: Sequence[int],
    ) -> Tensor:
        """Build the per-row slot index tensor for live LSTM state buffers."""
        return torch.tensor(
            [
                int(e) * self._players_per_env + int(p)
                for e, p in zip(env_indices, perspective_player_indices, strict=True)
            ],
            dtype=torch.long,
            device=self.device,
        )

    def lstm_env_state_inputs(
        self,
        env_indices: Sequence[int],
        perspective_player_indices: Sequence[int],
        *,
        slots: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if slots is None:
            slots = self._lstm_slots(env_indices, perspective_player_indices)
        h_in = self.live_lstm_h.index_select(1, slots).contiguous()
        c_in = self.live_lstm_c.index_select(1, slots).contiguous()
        return h_in, c_in

    def scatter_lstm_env_states(
        self,
        env_indices: Sequence[int],
        perspective_player_indices: Sequence[int],
        h_out: Tensor,
        c_out: Tensor,
        *,
        slots: Tensor | None = None,
    ) -> None:
        if slots is None:
            slots = self._lstm_slots(env_indices, perspective_player_indices)
        self.live_lstm_h.index_copy_(1, slots, h_out.to(self.live_lstm_h.dtype))
        self.live_lstm_c.index_copy_(1, slots, c_out.to(self.live_lstm_c.dtype))

    def sample_batch(
        self,
        batch: TextEncodedBatch,
        *,
        env_indices: Sequence[int],
        perspective_player_indices: Sequence[int],
        deterministic: bool = False,
        max_decode_len: int = 32,
    ) -> NativeTextSampleBatch:
        """Sample a batch of decoder actions.

        Runs the encoder + LSTM update + decoder sampler. Returns one
        :class:`DecoderDecisionLayout` per row plus the per-step log-prob
        tensor needed for PPO importance ratios.
        """

        # Build the slot index tensor once and reuse for the gather and the
        # subsequent scatter — avoids rebuilding the same tensor twice.
        slots = self._lstm_slots(env_indices, perspective_player_indices)
        h_in, c_in = self.lstm_env_state_inputs(
            env_indices, perspective_player_indices, slots=slots
        )
        device = self.device
        moved = batch  # caller is expected to have moved the batch already
        # Single encoder forward with history injection + LSTM update;
        # decoder cross-attn requires the padded [B, T, D] hidden tensor.
        encoded, h_out, c_out = self.policy.encoder_forward_padded_with_history(
            moved, h_in=h_in, c_in=c_in
        )
        self.scatter_lstm_env_states(
            env_indices, perspective_player_indices, h_out, c_out, slots=slots
        )

        attn_mask = moved.attention_mask.to(device=device, dtype=torch.bool)
        sample = decoder_sample(
            self.policy.text_policy,
            encoded,
            attn_mask,
            moved.decision_type.to(device=device, dtype=torch.long),
            moved.pointer_anchor_positions.to(device=device, dtype=torch.long),
            moved.pointer_anchor_kinds.to(device=device, dtype=torch.long),
            moved.pointer_anchor_subjects.to(device=device, dtype=torch.long),
            moved.pointer_anchor_handles.to(device=device, dtype=torch.long),
            legal_edge_bitmap=moved.legal_edge_bitmap,
            max_decode_len=max_decode_len,
            greedy=deterministic,
        )

        decoded: list[DecoderDecisionLayout] = []
        b = int(moved.token_ids.shape[0])
        for i in range(b):
            decoded.append(
                DecoderDecisionLayout(
                    output_token_ids=sample.output_token_ids[i],
                    output_pointer_pos=sample.output_pointer_pos[i],
                    output_pointer_subjects=sample.output_pointer_subjects[i],
                    output_is_pointer=sample.output_is_pointer[i],
                    output_pad_mask=sample.output_pad_mask[i],
                    decision_type=int(sample.decision_type[i].item()),
                    pointer_anchor_handles=sample.pointer_anchor_handles[i],
                    pointer_anchor_count=int(sample.pointer_anchor_count[i].item()),
                )
            )
        return NativeTextSampleBatch(
            decoded=decoded,
            log_probs=sample.log_probs,
            replay_rows=[],
        )

    def evaluate_replay(
        self,
        encoded: Tensor,
        encoder_attention_mask: Tensor,
        target_tokens: Tensor,
        target_pointer_pos: Tensor,
        is_pointer_step: Tensor,
        pad_mask: Tensor,
        vocab_mask: Tensor,
        pointer_mask: Tensor,
    ):
        """Teacher-forced replay scoring under the grammar decoder."""

        return decoder_score_replay(
            self.policy.text_policy,
            encoded,
            encoder_attention_mask,
            target_tokens,
            target_pointer_pos,
            is_pointer_step,
            pad_mask,
            vocab_mask,
            pointer_mask,
        )

    # ------------------------------------------------------------------ #
    # Polymorphic R-NaD / PPO surface                                    #
    #                                                                    #
    # The slot policy exposes per-decision-group "per-choice" tensors    #
    # because slot-encoder steps fan out into multiple decision groups   #
    # per env step. The decoder collapses every step to a single row,    #
    # so the per-choice axis collapses too: each row contributes one     #
    # log-pi / one entropy / one value.                                  #
    # ------------------------------------------------------------------ #

    def _gather_replay_decoder(self, replay_rows: list[int] | Tensor) -> Any:
        """Return the replay buffer's gathered decoder targets for these rows."""
        if self.rollout_buffer is None:
            raise RuntimeError(
                "LSTMStatefulTextPolicy.rollout_buffer is None; cannot gather replay decoder."
            )
        if isinstance(replay_rows, Tensor):
            idx = replay_rows.to(device=self.device, dtype=torch.long)
        else:
            idx = torch.tensor(list(replay_rows), dtype=torch.long, device=self.device)
        return self.rollout_buffer.gather(idx)

    def precompute_replay_forward(
        self,
        episodes: list[list[int]],
        **_kwargs: Any,
    ) -> None:
        """Pre-encode the replay batch.

        Slot policy returns a cache that downstream per-choice scoring
        reuses; the decoder path threads its encoder forward inside
        :meth:`evaluate_replay_batch_per_choice`, so this hook returns
        ``None`` and the trainer falls back to the standard call.
        """
        del episodes
        return None

    def count_active_replay_steps(
        self,
        per_episode_replay_rows: Sequence[Sequence[int]],
    ) -> tuple[int, int]:
        """Return ``(cl_count, pl_count)`` for the given replay rows.

        Decoder semantics: every replay row is one decision step. Both
        counts equal the number of rows whose ``decision_type >= 0``
        (a row with no pending decision spec contributes zero loss).
        """
        if not per_episode_replay_rows:
            return 0, 0
        if self.rollout_buffer is None:
            raise RuntimeError(
                "LSTMStatefulTextPolicy.rollout_buffer is None; cannot count replay steps."
            )
        flat = [int(r) for ep in per_episode_replay_rows for r in ep]
        if not flat:
            return 0, 0
        idx = torch.tensor(flat, dtype=torch.long, device=self.device)
        decision_type = self.rollout_buffer.decoder.decision_type[idx]
        active = int((decision_type >= 0).sum().item())
        return active, active

    def evaluate_replay_batch(
        self,
        replay_rows: list[int] | Tensor,
        *,
        return_extras: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Any | None]:
        """Per-row ``(log_pi, entropy, value, extras)`` for these replay rows.

        Used by PPO. ``extras`` is reserved for SPR; the decoder path does
        not currently emit SPR features so this returns ``None``.
        """
        del return_extras
        batch = self._gather_replay_decoder(replay_rows)
        # Run encoder with the per-row recurrent state recorded at rollout
        # time so train-time scoring matches sample-time exactly. Replay
        # storage layout is [B, layers, hidden]; LSTM wants [layers, B, hidden].
        h_in = (
            batch.lstm_h_in.permute(1, 0, 2).contiguous() if batch.lstm_h_in is not None else None
        )
        c_in = (
            batch.lstm_c_in.permute(1, 0, 2).contiguous() if batch.lstm_c_in is not None else None
        )
        encoded_snaps, _h_out, _c_out = self.policy.encode_with_history(
            batch.encoded, h_in=h_in, c_in=c_in
        )
        encoded = encoded_snaps.encoded
        # Build the [B, T_enc] attention mask from packed seq lengths.
        b = int(batch.encoded.seq_lengths.shape[0])
        t_enc = int(encoded.shape[1])
        seq_lengths = batch.encoded.seq_lengths.to(device=encoded.device, dtype=torch.long)
        positions = torch.arange(t_enc, device=encoded.device).unsqueeze(0).expand(b, -1)
        attn_mask = positions < seq_lengths.unsqueeze(-1)

        decoder = batch.decoder
        target_tokens = decoder.output_token_ids.to(dtype=torch.long).clamp_min(0)
        target_pointer_pos = decoder.output_pointer_pos.to(dtype=torch.long).clamp_min(0)
        is_pointer_step = decoder.output_is_pointer.to(dtype=torch.bool)
        pad_mask = decoder.output_pad_mask.to(dtype=torch.bool)

        # Reconstruct per-step subject indices from stored encoder positions via
        # the per-row pointer-anchor metadata. Vocab steps get -1. Built via
        # a sync-free scatter (see :func:`gpu_grammar` for the same pattern):
        # invalid (b, j) anchor entries scatter into a trash slot and are
        # sliced off, so we never need ``valid.any().item()``.
        anchor_pos_dev = batch.decoder.pointer_anchor_positions.to(
            device=encoded.device, dtype=torch.long
        )
        anchor_subj_dev = batch.decoder.pointer_anchor_subjects.to(
            device=encoded.device, dtype=torch.long
        )
        valid_anchor = (anchor_pos_dev >= 0) & (anchor_pos_dev < t_enc) & (anchor_subj_dev >= 0)
        pos_to_subject_full = torch.full(
            (b, t_enc + 1), -1, dtype=torch.long, device=encoded.device
        )
        trash_pos = torch.full_like(anchor_pos_dev, t_enc)
        safe_pos = torch.where(valid_anchor, anchor_pos_dev, trash_pos)
        pos_to_subject_full.scatter_(1, safe_pos, anchor_subj_dev)
        pos_to_subject_map = pos_to_subject_full[:, :t_enc].contiguous()
        target_pointer_subjects = torch.where(
            is_pointer_step,
            pos_to_subject_map.gather(1, target_pointer_pos.clamp_min(0)),
            torch.full_like(target_pointer_pos, -1),
        )

        # Reconstruct grammar masks from stored anchor metadata so the
        # softmax-renormalization matches the sampler — silent all-True
        # masks would compute log-π over the wrong support and corrupt
        # PPO importance ratios. Output is already on ``encoded.device``.
        vocab_mask, pointer_mask = build_replay_grammar_masks(
            decision_type=batch.decoder.decision_type,
            pointer_anchor_kinds=batch.decoder.pointer_anchor_kinds,
            pointer_anchor_subjects=batch.decoder.pointer_anchor_subjects,
            pointer_anchor_positions=batch.decoder.pointer_anchor_positions,
            pointer_anchor_handles=batch.decoder.pointer_anchor_handles,
            target_tokens=decoder.output_token_ids,
            target_pointer_subjects=target_pointer_subjects,
            target_is_pointer=is_pointer_step,
            target_pad_mask=pad_mask,
            encoded_seq_len=t_enc,
            legal_edge_bitmap=getattr(batch.decoder, "legal_edge_bitmap", None),
        )
        scores = decoder_score_replay(
            self.policy.text_policy,
            encoded,
            attn_mask,
            target_tokens,
            target_pointer_pos,
            is_pointer_step,
            pad_mask,
            vocab_mask,
            pointer_mask,
        )
        values = self.policy.text_policy.run_heads(encoded_snaps)
        return scores.per_row_log_pi, scores.per_row_entropy, values.squeeze(-1), None

    def evaluate_replay_batch_per_choice(
        self,
        replay_rows: list[int],
        *,
        lstm_state_override: tuple[Tensor, Tensor] | None = None,
        hidden_override: Tensor | None = None,
        cached: Any | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Any]:
        """R-NaD per-choice scoring is not implemented for the decoder pipeline.

        Slot policy's NeuRD update consumes per-decision-group flat-action
        logits over a fixed action space. The grammar decoder factors a
        decision into a variable-length token sequence, so the slot-shaped
        ``ReplayPerChoice`` payload doesn't have a coherent translation
        without a redesign of the NeuRD assembly. Until that lands, R-NaD
        is unsupported on the decoder path; use ``--trainer ppo``.
        """
        del replay_rows, lstm_state_override, hidden_override, cached
        raise NotImplementedError(
            "R-NaD evaluate_replay_batch_per_choice is not wired for the decoder "
            "pipeline. Use --trainer ppo, or design a per-step NeuRD assembly "
            "for variable-length decoded sequences before calling this."
        )

    def write_ppo_targets(
        self,
        replay_rows: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> None:
        if self.rollout_buffer is None:
            raise RuntimeError(
                "LSTMStatefulTextPolicy.rollout_buffer is None; cannot write PPO targets."
            )
        self.rollout_buffer.write_ppo_targets(replay_rows, old_log_probs, returns, advantages)

    def gather_ppo_targets(self, replay_rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.rollout_buffer is None:
            raise RuntimeError(
                "LSTMStatefulTextPolicy.rollout_buffer is None; cannot gather PPO targets."
            )
        return self.rollout_buffer.gather_ppo_targets(replay_rows)

    def gather_replay_old_log_prob_value(
        self,
        replay_rows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.rollout_buffer is None:
            raise RuntimeError(
                "LSTMStatefulTextPolicy.rollout_buffer is None; cannot gather replay rows."
            )
        idx = replay_rows.to(device=self.rollout_buffer.device, dtype=torch.long)
        return self.rollout_buffer.core.old_log_prob[idx], self.rollout_buffer.core.value[idx]

    def compute_spr_loss(
        self,
        step_indices: Tensor,
        *,
        extras: Any | None = None,
    ) -> Tensor:
        """SPR is not used by the decoder pipeline; only invoked when
        ``spr_enabled`` is True, which it never is here."""

        del step_indices, extras
        raise RuntimeError("LSTMStatefulTextPolicy does not implement SPR; spr_enabled stays False")

    def update_spr_target(self, decay: float | None = None) -> None:
        del decay
        raise RuntimeError("LSTMStatefulTextPolicy does not implement SPR; spr_enabled stays False")

    def recompute_lstm_states_for_episode(
        self,
        replay_rows: list[int],
    ) -> tuple[Tensor, Tensor] | None:
        """Decoder pipeline does not currently recompute per-episode LSTM
        input states for R-NaD; returns ``None`` so the trainer skips the
        override path.
        """
        del replay_rows
        return None

    def recompute_lstm_outputs_for_episodes(
        self,
        episodes: list[list[int]],
        *,
        chunk_size: int = 200,
        compiled_lstm: Any | None = None,
    ) -> list[Tensor] | None:
        del episodes, chunk_size, compiled_lstm
        return None


__all__ = ["LSTMStatefulTextPolicy"]
