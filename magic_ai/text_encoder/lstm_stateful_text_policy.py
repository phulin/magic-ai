"""Stateful wrapper around :class:`RecurrentTextPolicy`.

Owns live per-env, per-player LSTM state buffers and exposes the
polymorphic surface the RL trainers (PPO, R-NaD) expect. Sampling and
replay scoring delegate to the module-level helpers in
:mod:`decoder_inference`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from magic_ai.replay_decisions import ReplayPerChoice
from magic_ai.text_encoder.batch import TextEncodedBatch, scatter_packed_to_padded
from magic_ai.text_encoder.decoder_batch import (
    DecoderDecisionLayout,
    DecoderReplayScores,
    NativeTextSampleBatch,
)
from magic_ai.text_encoder.decoder_inference import (
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
        self.register_buffer(
            "live_lstm_h",
            torch.zeros(self.lstm_layers, 0, self.lstm_hidden),
            persistent=False,
        )
        self.register_buffer(
            "live_lstm_c",
            torch.zeros(self.lstm_layers, 0, self.lstm_hidden),
            persistent=False,
        )
        self._num_envs = 0
        self._players_per_env = 2
        self.rollout_buffer: TextReplayBuffer | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def clone_for_rnad(self) -> LSTMStatefulTextPolicy:
        """Deep-copy the underlying policy weights for R-NaD's target / reg copies.

        The clone shares ``rollout_buffer`` with ``self`` (target/reg policies
        replay the same trajectories the online policy collected) and starts
        with empty live LSTM state buffers — clones never sample from a live
        env, so per-env caches don't apply. See
        :class:`magic_ai.training_interfaces.RNaDTrainablePolicy.clone_for_rnad`.
        """
        # Skip HF warm-init: load_state_dict below overwrites the encoder
        # weights, so re-downloading the Ettin checkpoint would just be thrown
        # away (and prints a misleading second "Loading weights" report).
        cfg = self.policy.cfg
        if cfg.encoder.hf_model_name is not None:
            cfg = replace(cfg, encoder=replace(cfg.encoder, hf_model_name=None))
        clone = LSTMStatefulTextPolicy(cfg)
        clone.load_state_dict(self.state_dict())
        clone.to(self.device)
        clone.rollout_buffer = self.rollout_buffer
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
            # ``TextReplayBuffer.gather`` needs host row ids for its per-row
            # token-length lookups; round-trip a non-CPU tensor explicitly.
            host_rows = (
                [int(x) for x in replay_rows.tolist()]
                if replay_rows.device.type == "cpu"
                else [int(x) for x in replay_rows.cpu().tolist()]
            )
        else:
            host_rows = [int(x) for x in replay_rows]
        # Replay storage is host-side; bring the gathered batch to the
        # policy device for the forward pass.
        return self.rollout_buffer.gather(host_rows).to(self.device)

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
        # Replay storage may be host-side; index on the buffer's device.
        idx = torch.tensor(flat, dtype=torch.long, device=self.rollout_buffer.device)
        decision_type = self.rollout_buffer.decoder.decision_type[idx]
        active = int((decision_type >= 0).sum().item())
        return active, active

    def _score_replay_rows(
        self, replay_rows: list[int] | Tensor
    ) -> tuple[DecoderReplayScores, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Any]:
        """Shared core for ``evaluate_replay_batch`` and the per-choice variant.

        Returns the decoder replay scores plus the per-step targets, masks,
        and value head outputs — everything both PPO and R-NaD downstream
        paths need. ``batch`` is the gathered replay batch so the per-choice
        path can also lift stored mu log-probs.
        """

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
        # ``encode_with_history`` on a PackedTextBatch returns a packed
        # ``[T_packed, D]`` hidden tensor; the decoder cross-attn wants the
        # padded ``[B, T_max, D]`` shape with an explicit attention mask.
        encoded, attn_mask = scatter_packed_to_padded(encoded_snaps.encoded, batch.encoded)
        b = int(encoded.shape[0])
        t_enc = int(encoded.shape[1])

        decoder = batch.decoder
        target_tokens = decoder.output_token_ids.to(dtype=torch.long).clamp_min(0)
        target_pointer_pos = decoder.output_pointer_pos.to(dtype=torch.long).clamp_min(0)
        is_pointer_step = decoder.output_is_pointer.to(dtype=torch.bool)
        pad_mask = decoder.output_pad_mask.to(dtype=torch.bool)

        # Per-step grammar masks were saved by the live sampler at rollout
        # time; carry them straight to the score function. The replay
        # buffer stores ``pointer_mask`` at the buffer's ``max_tokens``
        # column width — truncate to the current encoder padding.
        # (Stored cells past ``T_enc_sample`` are False by construction,
        # so truncation never drops a True cell.)
        vocab_mask = decoder.vocab_mask.to(device=encoded.device, dtype=torch.bool)
        pointer_mask_full = decoder.pointer_mask.to(device=encoded.device, dtype=torch.bool)
        if pointer_mask_full.shape[2] >= t_enc:
            pointer_mask = pointer_mask_full[:, :, :t_enc].contiguous()
        else:
            # Replay-time encoder is wider than what the buffer stored —
            # zero-pad the tail. Anchor positions never exceed sample-time
            # T_enc, so the padded tail is False anyway.
            pad = torch.zeros(
                (b, pointer_mask_full.shape[1], t_enc - pointer_mask_full.shape[2]),
                dtype=torch.bool,
                device=encoded.device,
            )
            pointer_mask = torch.cat([pointer_mask_full, pad], dim=2)
        scores = decoder_score_replay(
            self.policy.text_policy,
            encoded,
            attn_mask,
            target_tokens,
            pad_mask,
            vocab_mask,
            batch.cells.to(device=encoded.device),
        )
        values = self.policy.text_policy.run_heads(encoded_snaps)
        return (
            scores,
            values.squeeze(-1),
            target_tokens,
            target_pointer_pos,
            is_pointer_step,
            pad_mask,
            vocab_mask,
            (batch, pointer_mask),
        )

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
        scores, values, *_ = self._score_replay_rows(replay_rows)
        return scores.per_row_log_pi, scores.per_row_entropy, values, None

    def evaluate_replay_batch_per_choice(
        self,
        replay_rows: list[int],
        *,
        lstm_state_override: tuple[Tensor, Tensor] | None = None,
        hidden_override: Tensor | None = None,
        cached: Any | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, ReplayPerChoice]:
        """R-NaD per-choice scoring over the grammar decoder.

        Each active decoder step (``pad_mask``-True position in a replay
        row) becomes one decision group. Its legal choices are the
        ``True`` cells of the on-device grammar mask: vocab indices for
        vocab steps, encoder positions for pointer steps. ``flat_logits``
        and ``flat_log_probs`` concatenate every (group, legal-choice)
        pair across the batch in the same order R-NaD's NeuRD update
        expects from :class:`magic_ai.replay_decisions.ReplayPerChoice`.

        ``lstm_state_override`` / ``hidden_override`` / ``cached`` are
        accepted for protocol parity with the slot policy. The decoder
        pipeline always rescans the LSTM from the per-row state stored
        in the replay buffer (see :meth:`_score_replay_rows`), so all
        three are ignored.
        """
        del lstm_state_override, hidden_override, cached
        (
            scores,
            values,
            target_tokens,
            target_pointer_pos,
            is_pointer_step,
            pad_mask,
            vocab_mask,
            extras,
        ) = self._score_replay_rows(replay_rows)
        batch, pointer_mask = extras
        del pointer_mask, target_tokens, target_pointer_pos, is_pointer_step, vocab_mask, pad_mask
        device = values.device
        cells = batch.cells

        # ----- Per-choice from packed cells -----
        # ``cells`` is a sync-free packed view built CPU-side at gather:
        # vocab and pointer cells live in separated arenas, each with
        # its own legal-choice arena. The per-choice path is now a
        # straight gather — no nonzero, no cumsum, no [B, L, V] /
        # [B, L, T_enc] mask materialization.
        v_cell_b = cells.v_cell_b.to(dtype=torch.long)
        v_cell_t = cells.v_cell_t.to(dtype=torch.long)
        p_cell_b = cells.p_cell_b.to(dtype=torch.long)
        # ``p_cell_t`` isn't needed in the consumer anymore — pointer
        # logits are already per-cell from the decoder forward, so the
        # only per-cell ``(b, t)`` we need is for the dense-vocab gather
        # and the per-cell ``b`` for ``step_for_decision_group``.
        v_legal_cell_id = cells.v_legal_cell_id.to(dtype=torch.long)
        p_legal_cell_id = cells.p_legal_cell_id.to(dtype=torch.long)
        # Per-legal (b, t, choice): each legal entry inherits (b, t) of
        # its cell.
        v_b = v_cell_b[v_legal_cell_id]
        v_t = v_cell_t[v_legal_cell_id]
        v_c = cells.v_legal_choice.to(dtype=torch.long)
        p_b = p_cell_b[p_legal_cell_id]
        p_c = cells.p_legal_choice.to(dtype=torch.long)

        # Vocab side is still dense (``V`` is small); pointer side comes
        # straight off the decoder's per-cell head — no [B, L, T_enc]
        # indexing left.
        v_logits = scores.vocab_logits[v_b, v_t, v_c]
        v_logp = scores.vocab_log_softmax[v_b, v_t, v_c]
        p_logits = scores.p_legal_logits
        p_logp = scores.p_legal_log_softmax

        # Group ordering: vocab cells get ids [0, N_v_cells), pointer
        # cells get ids [N_v_cells, N_v_cells + N_p_cells). R-NaD's
        # downstream operations on ``decision_group_id_flat`` are
        # commutative (scatter_add) and only read consistent mappings
        # within a single per_choice output, so the choice of ordering
        # doesn't affect downstream results.
        n_v_cells = int(cells.v_cell_b.shape[0])
        v_group = v_legal_cell_id
        p_group = p_legal_cell_id + n_v_cells

        flat_logits = torch.cat([v_logits, p_logits], dim=0)
        flat_log_probs = torch.cat([v_logp, p_logp], dim=0)
        decision_group_id_flat = torch.cat([v_group, p_group], dim=0)
        is_sampled_flat = torch.cat([cells.v_legal_is_chosen, cells.p_legal_is_chosen], dim=0)
        choice_cols = torch.cat([v_c, p_c], dim=0)
        group_idx = torch.cat([v_b, p_b], dim=0)
        step_for_decision_group = torch.cat([v_cell_b, p_cell_b], dim=0)
        behavior_lp_per_group = torch.cat(
            [cells.v_cell_behavior_log_prob, cells.p_cell_behavior_log_prob], dim=0
        ).to(dtype=values.dtype)

        # Decoder has no "may" Bernoulli head; zero everything so
        # may_neurd_loss masks itself out via ``may_is_active=False``.
        b = int(batch.encoded.seq_lengths.shape[0])
        may_is_active = torch.zeros(b, dtype=torch.bool, device=device)
        may_logits_per_step = torch.zeros(b, dtype=values.dtype, device=device)
        may_selected_per_step = torch.zeros(b, dtype=values.dtype, device=device)

        per_choice = ReplayPerChoice(
            flat_logits=flat_logits,
            flat_log_probs=flat_log_probs,
            group_idx=group_idx,
            choice_cols=choice_cols,
            is_sampled_flat=is_sampled_flat,
            may_is_active=may_is_active,
            may_logits_per_step=may_logits_per_step,
            may_selected_per_step=may_selected_per_step,
            decision_group_id_flat=decision_group_id_flat,
            step_for_decision_group=step_for_decision_group,
            behavior_action_log_prob_per_decision_group=behavior_lp_per_group,
        )
        return scores.per_row_log_pi, scores.per_row_entropy, values, per_choice

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
        # Replay buffer is host-side; bring the targets onto the policy device.
        log_p, ret, adv = self.rollout_buffer.gather_ppo_targets(replay_rows)
        return (
            log_p.to(self.device, non_blocking=True),
            ret.to(self.device, non_blocking=True),
            adv.to(self.device, non_blocking=True),
        )

    def gather_replay_old_log_prob_value(
        self,
        replay_rows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.rollout_buffer is None:
            raise RuntimeError(
                "LSTMStatefulTextPolicy.rollout_buffer is None; cannot gather replay rows."
            )
        idx = replay_rows.to(device=self.rollout_buffer.device, dtype=torch.long)
        return (
            self.rollout_buffer.core.old_log_prob[idx].to(self.device, non_blocking=True),
            self.rollout_buffer.core.value[idx].to(self.device, non_blocking=True),
        )

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
