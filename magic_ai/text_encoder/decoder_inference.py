"""Pure functions that run the grammar decoder against an encoded batch.

- :func:`decoder_sample` — autoregressive sampling under the grammar mask.
  Fully on-device: the grammar state machine in :mod:`gpu_grammar` builds
  the next-step mask each iteration without leaving the GPU. Returns
  per-step log probabilities under the (vocab|pointer) mask of the chosen
  action so PPO/R-NaD can compute importance ratios.
- :func:`decoder_score_replay` — teacher-forced per-row log p of stored
  decoder targets, plus per-step entropy.
- :func:`build_replay_grammar_masks` — roll the on-device grammar state
  machine forward over stored target sequences to recover per-step
  ``(vocab_mask, pointer_mask)`` for replay scoring.
"""

from __future__ import annotations

import torch
from torch import Tensor

from magic_ai.text_encoder.decoder import DecoderState, combined_sample
from magic_ai.text_encoder.decoder_batch import DecoderReplayScores, DecoderSampleOutput
from magic_ai.text_encoder.gpu_grammar import GrammarMaskState
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, GrammarVocab
from magic_ai.text_encoder.policy import TextPolicy


def decoder_sample(
    text_policy: TextPolicy,
    encoded: Tensor,
    encoder_attention_mask: Tensor,
    decision_type: Tensor,
    pointer_anchor_positions: Tensor,
    pointer_anchor_kinds: Tensor,
    pointer_anchor_subjects: Tensor,
    pointer_anchor_handles: Tensor,
    *,
    legal_edge_bitmap: Tensor | None = None,
    spec_max_value: Tensor | None = None,
    max_decode_len: int = 32,
    greedy: bool = False,
    temperature: float = 1.0,
) -> DecoderSampleOutput:
    """Autoregressive sampling under the grammar mask.

    Fully on-device: :class:`gpu_grammar.GrammarMaskState` mirrors the
    per-decision-type FSAs from :mod:`grammar` as tensor updates so the
    inner loop has no host syncs. The loop always runs ``max_decode_len``
    iterations — eliminating the per-step ``ended.all()`` check trades a
    handful of cheap masked decoder steps for ``L`` saved sync points.
    """

    grammar_decoder = text_policy.grammar_decoder
    if grammar_decoder is None:
        raise RuntimeError("text_policy.grammar_decoder must be configured")
    device = encoded.device
    b, t_enc, _ = encoded.shape
    L = int(max_decode_len)

    decision_type_dev = decision_type.to(device=device, dtype=torch.long)
    anchor_pos_dev = pointer_anchor_positions.to(device=device, dtype=torch.long)
    anchor_kinds_dev = pointer_anchor_kinds.to(device=device, dtype=torch.long)
    anchor_subj_dev = pointer_anchor_subjects.to(device=device, dtype=torch.long)
    anchor_handles_dev = pointer_anchor_handles.to(device=device, dtype=torch.long)

    state = GrammarMaskState(
        decision_type=decision_type_dev,
        pointer_anchor_kinds=anchor_kinds_dev,
        pointer_anchor_subjects=anchor_subj_dev,
        pointer_anchor_positions=anchor_pos_dev,
        encoded_seq_len=t_enc,
        legal_edge_bitmap=legal_edge_bitmap,
        max_value=spec_max_value,
    )

    out_tokens = torch.zeros((b, L), dtype=torch.long, device=device)
    out_pointer_pos = torch.full((b, L), -1, dtype=torch.long, device=device)
    out_pointer_subjects = torch.full((b, L), -1, dtype=torch.long, device=device)
    out_is_pointer = torch.zeros((b, L), dtype=torch.bool, device=device)
    out_pad_mask = torch.zeros((b, L), dtype=torch.bool, device=device)
    out_log_probs = torch.zeros((b, L), device=device)

    pos_to_subject_map = state.pos_to_subj  # [B, T_enc] long, -1 fill

    decoder_state: DecoderState | None = None
    prev_token = torch.full((b,), int(GrammarVocab.PAD), dtype=torch.long, device=device)
    prev_pointer_pos = torch.full((b,), -1, dtype=torch.long, device=device)

    for step in range(L):
        valid_step = ~state.ended
        vocab_mask, pointer_mask = state.next_mask()

        vocab_logits, pointer_logits, decoder_state = grammar_decoder.step(
            prev_token, prev_pointer_pos, encoded, encoder_attention_mask, decoder_state
        )

        is_pointer_step = ~vocab_mask.any(dim=-1)

        sampled_vocab, sampled_pointer = combined_sample(
            vocab_logits,
            pointer_logits,
            vocab_mask,
            pointer_mask,
            is_pointer_step,
            greedy=greedy,
            temperature=temperature,
        )

        neg_inf = torch.finfo(vocab_logits.dtype).min
        v_logp = torch.log_softmax(vocab_logits.masked_fill(~vocab_mask, neg_inf), dim=-1)
        p_logp = torch.log_softmax(pointer_logits.masked_fill(~pointer_mask, neg_inf), dim=-1)
        v_chosen = v_logp.gather(-1, sampled_vocab.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        p_chosen = p_logp.gather(-1, sampled_pointer.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        step_log_prob = torch.where(is_pointer_step, p_chosen, v_chosen)

        out_tokens[:, step] = sampled_vocab
        out_pointer_pos[:, step] = sampled_pointer
        # Translate encoder position → subject_index via the per-row dense map.
        # Vocab steps gather a value too but we mask them to -1 so the wire
        # shape carries clean -1 fill.
        safe_pointer = sampled_pointer.clamp_min(0)
        gathered_subj = pos_to_subject_map.gather(1, safe_pointer.unsqueeze(-1)).squeeze(-1)
        out_pointer_subjects[:, step] = torch.where(
            is_pointer_step & valid_step,
            gathered_subj,
            torch.full_like(gathered_subj, -1),
        )
        out_is_pointer[:, step] = is_pointer_step
        out_pad_mask[:, step] = valid_step
        out_log_probs[:, step] = step_log_prob.where(valid_step, torch.zeros_like(step_log_prob))

        # Advance the grammar state, then update the decoder's prev-token
        # inputs. Pointer steps feed PAD as the structural token (matches
        # teacher-forced training, where the decoder embeds the structural
        # token, not pointer info).
        state.update(sampled_vocab, sampled_pointer, is_pointer_step)
        prev_token = torch.where(
            is_pointer_step,
            torch.full_like(sampled_vocab, int(GrammarVocab.PAD)),
            sampled_vocab,
        )
        prev_pointer_pos = sampled_pointer

    pointer_anchor_count = (anchor_subj_dev >= 0).sum(dim=-1).to(dtype=torch.long)

    return DecoderSampleOutput(
        output_token_ids=out_tokens,
        output_pointer_pos=out_pointer_pos,
        output_pointer_subjects=out_pointer_subjects,
        output_is_pointer=out_is_pointer,
        output_pad_mask=out_pad_mask,
        log_probs=out_log_probs,
        decision_type=decision_type_dev,
        pointer_anchor_handles=anchor_handles_dev,
        pointer_anchor_count=pointer_anchor_count,
    )


def decoder_score_replay(
    text_policy: TextPolicy,
    encoded: Tensor,
    encoder_attention_mask: Tensor,
    target_tokens: Tensor,
    target_pointer_pos: Tensor,
    is_pointer_step: Tensor,
    pad_mask: Tensor,
    vocab_mask: Tensor,
    pointer_mask: Tensor,
) -> DecoderReplayScores:
    """Teacher-forced log-prob of stored decoder targets, per row.

    Mirrors :func:`policy_value_pretrain.decoder_cross_entropy_loss` but
    returns per-row log-π and per-row entropy instead of a scalar loss, so
    PPO / R-NaD can plug in importance ratios and entropy bonuses.
    """

    grammar_decoder = text_policy.grammar_decoder
    if grammar_decoder is None:
        raise RuntimeError("text_policy.grammar_decoder must be configured")
    vocab_logits, pointer_logits = grammar_decoder.forward_teacher_forced(
        target_tokens.to(dtype=torch.long), encoded, encoder_attention_mask
    )
    neg_inf = torch.finfo(vocab_logits.dtype).min
    v_logp = torch.log_softmax(vocab_logits.masked_fill(~vocab_mask, neg_inf), dim=-1)
    p_logp = torch.log_softmax(pointer_logits.masked_fill(~pointer_mask, neg_inf), dim=-1)
    target_tok = target_tokens.to(dtype=torch.long).clamp_min(0)
    target_ptr = target_pointer_pos.to(dtype=torch.long).clamp_min(0)
    v_chosen = v_logp.gather(-1, target_tok.unsqueeze(-1)).squeeze(-1)
    p_chosen = p_logp.gather(-1, target_ptr.unsqueeze(-1)).squeeze(-1)
    step_logp = torch.where(is_pointer_step, p_chosen, v_chosen)
    step_logp = step_logp.where(pad_mask, torch.zeros_like(step_logp))

    # Entropy under the same mask.
    v_p = v_logp.exp()
    p_p = p_logp.exp()
    v_ent = -(v_p * v_logp.where(v_p > 0, torch.zeros_like(v_logp))).sum(dim=-1)
    p_ent = -(p_p * p_logp.where(p_p > 0, torch.zeros_like(p_logp))).sum(dim=-1)
    step_ent = torch.where(is_pointer_step, p_ent, v_ent)
    step_ent = step_ent.where(pad_mask, torch.zeros_like(step_ent))

    return DecoderReplayScores(
        per_row_log_pi=step_logp.sum(dim=-1),
        per_row_entropy=step_ent.sum(dim=-1),
        per_step_log_pi=step_logp,
        vocab_logits=vocab_logits,
        pointer_logits=pointer_logits,
        vocab_log_softmax=v_logp,
        pointer_log_softmax=p_logp,
    )


def build_replay_grammar_masks(
    decision_type: Tensor,
    pointer_anchor_kinds: Tensor,
    pointer_anchor_subjects: Tensor,
    pointer_anchor_positions: Tensor,
    pointer_anchor_handles: Tensor,
    target_tokens: Tensor,
    target_pointer_subjects: Tensor,
    target_is_pointer: Tensor,
    target_pad_mask: Tensor,
    *,
    encoded_seq_len: int,
    legal_edge_bitmap: Tensor | None = None,
    spec_max_value: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Reconstruct per-step ``(vocab_mask, pointer_mask)`` for stored decoder targets.

    Rolls the on-device grammar state machine forward over the stored
    prefix and reads off the legal mask at each step. Output shapes:
    ``vocab_mask`` ``[B, L, GRAMMAR_VOCAB_SIZE]``, ``pointer_mask``
    ``[B, L, encoded_seq_len]``, both bool, on the input device.

    Mirrors the mask-build path used by the sampler so replay scoring
    softmaxes over the same support — silent all-True masks would compute
    log-π over the wrong support and corrupt PPO importance ratios.
    """
    del pointer_anchor_handles  # accepted for API symmetry; not needed here

    device = decision_type.device
    b = int(decision_type.shape[0])
    L = int(target_tokens.shape[1])
    t_enc = int(encoded_seq_len)

    state = GrammarMaskState(
        decision_type=decision_type.to(device=device, dtype=torch.long),
        pointer_anchor_kinds=pointer_anchor_kinds.to(device=device, dtype=torch.long),
        pointer_anchor_subjects=pointer_anchor_subjects.to(device=device, dtype=torch.long),
        pointer_anchor_positions=pointer_anchor_positions.to(device=device, dtype=torch.long),
        encoded_seq_len=t_enc,
        legal_edge_bitmap=legal_edge_bitmap,
        max_value=spec_max_value,
    )

    target_tokens_dev = target_tokens.to(device=device, dtype=torch.long)
    is_ptr_dev = target_is_pointer.to(device=device, dtype=torch.bool)
    pad_dev = target_pad_mask.to(device=device, dtype=torch.bool)
    # The grammar state needs encoder-position pointers, not subject indices.
    # Map subject_index → encoder position via the per-row anchor lookup.
    target_subj_dev = target_pointer_subjects.to(device=device, dtype=torch.long)
    pos_for_subj = _build_subject_to_position_map(state.pos_to_subj, b, t_enc, device)
    safe_subj = target_subj_dev.clamp(min=0, max=max(pos_for_subj.shape[1] - 1, 0))
    target_pos = pos_for_subj.gather(1, safe_subj)
    target_pos = torch.where(target_subj_dev >= 0, target_pos, torch.full_like(target_pos, -1))

    vocab_out = torch.zeros((b, L, GRAMMAR_VOCAB_SIZE), dtype=torch.bool, device=device)
    ptr_out = torch.zeros((b, L, t_enc), dtype=torch.bool, device=device)

    for step in range(L):
        v_mask, p_mask = state.next_mask()
        vocab_out[:, step, :] = v_mask
        ptr_out[:, step, :] = p_mask
        # Advance only rows whose stored target has a step here.
        active_step = pad_dev[:, step]
        sampled_vocab = torch.where(
            active_step,
            target_tokens_dev[:, step],
            torch.full_like(target_tokens_dev[:, step], int(GrammarVocab.PAD)),
        )
        sampled_pointer = torch.where(
            active_step,
            target_pos[:, step],
            torch.full_like(target_pos[:, step], -1),
        )
        is_ptr = is_ptr_dev[:, step] & active_step
        state.update(sampled_vocab, sampled_pointer, is_ptr)

    return vocab_out, ptr_out


def _build_subject_to_position_map(
    pos_to_subj: Tensor,  # [B, T_enc] long
    b: int,
    t_enc: int,
    device: torch.device,
) -> Tensor:
    """Invert ``pos_to_subj`` into a per-row ``[B, T_enc]`` map.

    Stored decoder targets carry ``subject_index`` (anchor's local ordinal)
    as the pointer label; the GPU grammar state advances on encoder
    *positions*. ``T_enc`` is a safe upper bound for ``subject_index`` since
    every subject occupies at most one encoder position.
    """
    width = max(t_enc, 1)
    # Allocate one trash column past ``width`` so invalid (b, t) entries
    # scatter into it and then get sliced off — keeps the scatter sync-free.
    out = torch.full((b, width + 1), -1, dtype=torch.long, device=device)
    valid = pos_to_subj >= 0
    pos_arange = torch.arange(t_enc, device=device).unsqueeze(0).expand(b, t_enc)
    safe_subj = torch.where(valid, pos_to_subj, torch.full_like(pos_to_subj, width))
    out.scatter_(1, safe_subj, pos_arange)
    return out[:, :width]


__all__ = [
    "build_replay_grammar_masks",
    "decoder_sample",
    "decoder_score_replay",
]
