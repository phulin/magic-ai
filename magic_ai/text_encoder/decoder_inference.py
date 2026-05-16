"""Pure functions that run the grammar decoder against an encoded batch.

- :func:`decoder_sample` — autoregressive sampling under the grammar mask.
  Fully on-device: the grammar state machine in :mod:`gpu_grammar` builds
  the next-step mask each iteration without leaving the GPU. Returns
  per-step log probabilities under the (vocab|pointer) mask of the chosen
  action so PPO/R-NaD can compute importance ratios.
- :func:`decoder_score_replay` — teacher-forced per-row log p of stored
  decoder targets, plus per-step entropy. Replay scoring reuses the
  per-step grammar masks captured at sample time (carried through the
  replay buffer on :class:`DecoderSampleOutput`); no FSA rebuild here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from magic_ai.text_encoder.decoder import DecoderState, combined_sample
from magic_ai.text_encoder.decoder_batch import DecoderReplayScores, DecoderSampleOutput
from magic_ai.text_encoder.gpu_grammar import GrammarMaskState
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, GrammarVocab
from magic_ai.text_encoder.policy import TextPolicy


def _decoder_step_body(
    grammar_decoder: Any,
    state: GrammarMaskState,
    decoder_state: DecoderState,
    prev_token: Tensor,
    prev_pointer_pos: Tensor,
    encoded: Tensor,
    encoder_attention_mask: Tensor,
    pointer_key_states: Tensor,
    greedy: bool,
    temperature: float,
) -> tuple[
    Tensor,  # vocab_mask
    Tensor,  # pointer_mask
    Tensor,  # structural_tok (PAD on pointer steps, sampled vocab otherwise)
    Tensor,  # sampled_pointer anchor index
    Tensor,  # sampled_vocab (raw, kept for state.update)
    Tensor,  # gathered_subj
    Tensor,  # is_pointer_step
    Tensor,  # valid_step
    Tensor,  # step_log_prob
    DecoderState,  # next decoder_state (always non-None on return)
]:
    """One step of ``decoder_sample``'s loop.

    Factored out so :class:`TextInferencePipeline` can ``torch.compile`` a
    small graph (per-step ops) instead of the 32-iteration unroll, which
    cuts inductor compile time roughly L× at the cost of losing
    inter-step fusion. The grammar state's ``update`` and the
    write-into-output-tensors live in the outer Python loop.
    """
    valid_step = ~state.ended
    vocab_mask, pointer_mask = state.next_mask()
    vocab_logits, pointer_logits, decoder_state = grammar_decoder.step(
        prev_token,
        prev_pointer_pos,
        encoded,
        encoder_attention_mask,
        decoder_state,
        pointer_keys=pointer_key_states,
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
    structural_tok = torch.where(
        is_pointer_step,
        torch.full_like(sampled_vocab, int(GrammarVocab.PAD)),
        sampled_vocab,
    )
    safe_pointer = sampled_pointer.clamp(min=0, max=max(pointer_key_states.shape[1] - 1, 0))
    gathered_subj = state.pointer_anchor_subjects.gather(1, safe_pointer.unsqueeze(-1)).squeeze(-1)
    return (
        vocab_mask,
        pointer_mask,
        structural_tok,
        sampled_pointer,
        sampled_vocab,
        gathered_subj,
        is_pointer_step,
        valid_step,
        step_log_prob,
        decoder_state,
    )


def _decoder_step_body_single_batch(*args: Any, **kwargs: Any) -> Any:
    """Distinct compile target for ``B == 1`` decoder-step calls."""

    return _decoder_step_body(*args, **kwargs)


def _decoder_step_body_multi_batch(*args: Any, **kwargs: Any) -> Any:
    """Distinct compile target for ``B > 1`` decoder-step calls."""

    return _decoder_step_body(*args, **kwargs)


def _decoder_step_callable(text_policy: TextPolicy) -> Callable[..., Any]:
    """Return a closure binding ``grammar_decoder`` to a step function.

    The returned callable is what ``decoder_sample`` calls per iteration.
    :class:`TextInferencePipeline` can monkey-patch ``decoder_inference._decoder_step_callable``
    to replace it with a ``torch.compile``-wrapped variant (see
    Phase F).
    """

    grammar_decoder = text_policy.grammar_decoder
    if grammar_decoder is None:
        raise RuntimeError("text_policy.grammar_decoder must be configured")

    def step(*args: Any, **kwargs: Any) -> Any:
        return _decoder_step_body(grammar_decoder, *args, **kwargs)

    return step


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
    anchor_pos_in = pointer_anchor_positions.to(device=device, dtype=torch.long)
    anchor_kinds_in = pointer_anchor_kinds.to(device=device, dtype=torch.long)
    anchor_subj_in = pointer_anchor_subjects.to(device=device, dtype=torch.long)
    anchor_handles_dev = pointer_anchor_handles.to(device=device, dtype=torch.long)
    n_anchor_in = int(anchor_pos_in.shape[1])
    if n_anchor_in == 0:
        anchor_pos_dev = torch.full((b, 1), -1, dtype=torch.long, device=device)
        anchor_kinds_dev = torch.full((b, 1), -1, dtype=torch.long, device=device)
        anchor_subj_dev = torch.full((b, 1), -1, dtype=torch.long, device=device)
    else:
        anchor_pos_dev = anchor_pos_in
        anchor_kinds_dev = anchor_kinds_in
        anchor_subj_dev = anchor_subj_in

    state = GrammarMaskState(
        decision_type=decision_type_dev,
        pointer_anchor_kinds=anchor_kinds_dev,
        pointer_anchor_subjects=anchor_subj_dev,
        pointer_anchor_positions=anchor_pos_dev,
        encoded_seq_len=t_enc,
        legal_edge_bitmap=legal_edge_bitmap,
        max_value=spec_max_value,
        compile_update=bool(getattr(grammar_decoder.cfg, "compile_mask_update", False)),
    )

    out_tokens = torch.zeros((b, L), dtype=torch.long, device=device)
    out_pointer_pos = torch.full((b, L), -1, dtype=torch.long, device=device)
    out_pointer_subjects = torch.full((b, L), -1, dtype=torch.long, device=device)
    out_is_pointer = torch.zeros((b, L), dtype=torch.bool, device=device)
    out_pad_mask = torch.zeros((b, L), dtype=torch.bool, device=device)
    out_log_probs = torch.zeros((b, L), device=device)
    # Capture the live mask each step so replay scoring softmaxes over
    # the exact same support. The FSA-reconstruction path the buffer
    # used to take has been removed.
    out_vocab_mask = torch.zeros((b, L, GRAMMAR_VOCAB_SIZE), dtype=torch.bool, device=device)
    n_anchor = int(anchor_pos_dev.shape[1])
    out_pointer_mask = torch.zeros((b, L, n_anchor), dtype=torch.bool, device=device)

    safe_anchor_pos = anchor_pos_dev.clamp(min=0, max=max(t_enc - 1, 0))
    anchor_valid = (anchor_pos_dev >= 0) & (anchor_pos_dev < t_enc)
    pointer_key_states = encoded.gather(
        1,
        safe_anchor_pos.unsqueeze(-1).expand(b, n_anchor, int(encoded.shape[-1])),
    )
    pointer_key_states = pointer_key_states.masked_fill(~anchor_valid.unsqueeze(-1), 0.0)

    # Pre-allocate the KV cache to the full decode length so the compiled
    # step body sees constant shapes every iteration. init_state also does
    # the cross-KV projection so the step body never branches on
    # ``state is None``.
    decoder_state = grammar_decoder.init_state(encoded, max_decode_len=L)
    prev_token = torch.full((b,), int(GrammarVocab.PAD), dtype=torch.long, device=device)
    prev_pointer_pos = torch.full((b,), -1, dtype=torch.long, device=device)

    step_fn = _decoder_step_callable(text_policy)
    for step in range(L):
        (
            vocab_mask,
            pointer_mask,
            structural_tok,
            sampled_pointer,
            sampled_vocab,
            gathered_subj,
            is_pointer_step,
            valid_step,
            step_log_prob,
            decoder_state,
        ) = step_fn(
            state,
            decoder_state,
            prev_token,
            prev_pointer_pos,
            encoded,
            encoder_attention_mask,
            pointer_key_states,
            greedy,
            temperature,
        )
        safe_anchor = sampled_pointer.clamp(min=0, max=max(n_anchor - 1, 0))
        sampled_pointer_pos = anchor_pos_dev.gather(1, safe_anchor.unsqueeze(-1)).squeeze(-1)
        out_vocab_mask[:, step, :] = vocab_mask
        out_pointer_mask[:, step, :] = pointer_mask
        out_tokens[:, step] = structural_tok
        out_pointer_pos[:, step] = torch.where(
            is_pointer_step & valid_step,
            sampled_pointer_pos,
            torch.full_like(sampled_pointer_pos, -1),
        )
        out_pointer_subjects[:, step] = torch.where(
            is_pointer_step & valid_step,
            gathered_subj,
            torch.full_like(gathered_subj, -1),
        )
        out_is_pointer[:, step] = is_pointer_step
        out_pad_mask[:, step] = valid_step
        out_log_probs[:, step] = step_log_prob.where(valid_step, torch.zeros_like(step_log_prob))

        # Advance grammar state on the host side of the loop (outside the
        # compiled step) so the compiled body has no in-place side effects
        # — those are awkward for Dynamo and add guards.
        state.update(sampled_vocab, sampled_pointer, is_pointer_step)
        prev_token = structural_tok
        prev_pointer_pos = sampled_pointer_pos

    pointer_anchor_count = (
        ((anchor_pos_in >= 0) & (anchor_pos_in < t_enc) & (anchor_kinds_in >= 0))
        .sum(dim=-1)
        .to(dtype=torch.long)
    )

    return DecoderSampleOutput(
        output_token_ids=out_tokens,
        output_pointer_pos=out_pointer_pos,
        output_pointer_subjects=out_pointer_subjects,
        output_is_pointer=out_is_pointer,
        output_pad_mask=out_pad_mask,
        log_probs=out_log_probs,
        vocab_mask=out_vocab_mask,
        pointer_mask=out_pointer_mask,
        decision_type=decision_type_dev,
        pointer_anchor_handles=anchor_handles_dev,
        pointer_anchor_count=pointer_anchor_count,
    )


def decoder_score_replay(
    text_policy: TextPolicy,
    encoded: Tensor,
    encoder_attention_mask: Tensor,
    target_tokens: Tensor,
    pad_mask: Tensor,
    vocab_mask: Tensor,
    cells: Any,
) -> DecoderReplayScores:
    """Teacher-forced log-prob of stored decoder targets, per row.

    Mirrors :func:`policy_value_pretrain.decoder_cross_entropy_loss` but
    returns per-row log-π and per-row entropy instead of a scalar loss, so
    PPO / R-NaD can plug in importance ratios and entropy bonuses.

    Vocab side: dense ``[B, L, V]`` (``V`` is small for the grammar
    decoder, so the dense head is the right shape). Pointer side: the
    decoder pointer head runs per-legal-cell via
    :meth:`GrammarDecoder.forward_teacher_forced_with_cells`, avoiding
    the dense ``[B, L, T_enc]`` materialization. Per-step pointer log-π
    and entropy come from a segment softmax over each pointer cell's
    legal entries.
    """

    grammar_decoder = text_policy.grammar_decoder
    if grammar_decoder is None:
        raise RuntimeError("text_policy.grammar_decoder must be configured")
    v_cell_b = cells.v_cell_b.to(dtype=torch.long)
    v_cell_t = cells.v_cell_t.to(dtype=torch.long)
    p_cell_b = cells.p_cell_b.to(dtype=torch.long)
    p_cell_t = cells.p_cell_t.to(dtype=torch.long)
    p_legal_cell_id = cells.p_legal_cell_id.to(dtype=torch.long)
    p_legal_choice = cells.p_legal_choice.to(dtype=torch.long)
    vocab_logits, p_legal_logits = grammar_decoder.forward_teacher_forced_with_cells(
        target_tokens.to(dtype=torch.long),
        encoded,
        encoder_attention_mask,
        p_cell_b=p_cell_b,
        p_cell_t=p_cell_t,
        p_legal_cell_id=p_legal_cell_id,
        p_legal_choice=p_legal_choice,
    )
    neg_inf = torch.finfo(vocab_logits.dtype).min
    v_logp = torch.log_softmax(vocab_logits.masked_fill(~vocab_mask, neg_inf), dim=-1)

    # Per-cell segment log_softmax over the pointer cells' legal entries.
    n_p_cells = int(p_cell_b.shape[0])
    if p_legal_logits.numel() > 0:
        max_per_p_cell = torch.full(
            (n_p_cells,),
            neg_inf,
            dtype=p_legal_logits.dtype,
            device=p_legal_logits.device,
        )
        max_per_p_cell.scatter_reduce_(
            0, p_legal_cell_id, p_legal_logits, reduce="amax", include_self=True
        )
        shifted = p_legal_logits - max_per_p_cell[p_legal_cell_id]
        exp_shifted = shifted.exp()
        sum_per_p_cell = torch.zeros(
            n_p_cells, dtype=p_legal_logits.dtype, device=p_legal_logits.device
        )
        sum_per_p_cell.scatter_add_(0, p_legal_cell_id, exp_shifted)
        logsumexp_per_p_cell = sum_per_p_cell.log() + max_per_p_cell
        p_legal_log_softmax = p_legal_logits - logsumexp_per_p_cell[p_legal_cell_id]
    else:
        p_legal_log_softmax = p_legal_logits.new_empty((0,))

    # ``target_tokens`` is already PAD-substituted at sample time (see
    # ``decoder_sample``), so no clamp needed.
    target_tok = target_tokens.to(dtype=torch.long)
    v_chosen = v_logp.gather(-1, target_tok.unsqueeze(-1)).squeeze(-1)

    # Pointer chosen log-π per cell: exactly one ``is_chosen`` legal
    # entry per cell (invariant from sampling); masking preserves cell
    # order since legal entries are stored contiguously per cell.
    p_chosen_per_cell = p_legal_log_softmax[cells.p_legal_is_chosen]  # [N_p_cells]

    # Assemble per-step log-π. Vocab cells use the dense gather; pointer
    # cells use the per-cell scalar — scatter back to [B, L] so the rest
    # of the contract (per_step_log_pi shape, per_row sum) is unchanged.
    step_logp = torch.zeros_like(v_chosen)
    if int(v_cell_b.shape[0]) > 0:
        step_logp[v_cell_b, v_cell_t] = v_chosen[v_cell_b, v_cell_t]
    if n_p_cells > 0:
        step_logp[p_cell_b, p_cell_t] = p_chosen_per_cell
    # Inactive cells contribute zero.
    step_logp = step_logp.where(pad_mask, torch.zeros_like(step_logp))

    # Entropy.
    v_p = v_logp.exp()
    v_ent_dense = -(v_p * v_logp.where(v_p > 0, torch.zeros_like(v_logp))).sum(dim=-1)
    # Per-pointer-cell entropy from the segment-softmax distribution.
    if p_legal_logits.numel() > 0:
        p_p_per_legal = p_legal_log_softmax.exp()
        p_ent_per_legal = -(p_p_per_legal * p_legal_log_softmax)
        p_ent_per_cell = torch.zeros(
            n_p_cells, dtype=p_legal_logits.dtype, device=p_legal_logits.device
        )
        p_ent_per_cell.scatter_add_(0, p_legal_cell_id, p_ent_per_legal)
    else:
        p_ent_per_cell = p_legal_logits.new_empty((0,))
    step_ent = torch.zeros_like(v_ent_dense)
    if int(v_cell_b.shape[0]) > 0:
        step_ent[v_cell_b, v_cell_t] = v_ent_dense[v_cell_b, v_cell_t]
    if n_p_cells > 0:
        step_ent[p_cell_b, p_cell_t] = p_ent_per_cell
    step_ent = step_ent.where(pad_mask, torch.zeros_like(step_ent))

    return DecoderReplayScores(
        per_row_log_pi=step_logp.sum(dim=-1),
        per_row_entropy=step_ent.sum(dim=-1),
        per_step_log_pi=step_logp,
        vocab_logits=vocab_logits,
        vocab_log_softmax=v_logp,
        p_legal_logits=p_legal_logits,
        p_legal_log_softmax=p_legal_log_softmax,
    )


__all__ = [
    "decoder_sample",
    "decoder_score_replay",
]
