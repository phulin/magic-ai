"""Pure functions that run the grammar decoder against an encoded batch.

- :func:`decoder_sample` — autoregressive sampling under the grammar mask.
  One host sync per decoder step (for the prefix → next-mask call). Returns
  per-step log probabilities under the (vocab|pointer) mask of the chosen
  action so PPO/R-NaD can compute importance ratios.
- :func:`decoder_score_replay` — teacher-forced per-row log p of stored
  decoder targets, plus per-step entropy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from magic_ai.text_encoder.decision_spec import (
    AnchorKind,
    DecisionSpec,
    DecisionType,
    PointerAnchor,
)
from magic_ai.text_encoder.decoder import DecoderState, combined_sample
from magic_ai.text_encoder.decoder_batch import DecoderReplayScores, DecoderSampleOutput
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, GrammarVocab, batch_next_mask
from magic_ai.text_encoder.policy import TextPolicy


def _build_batched_specs(
    decision_type: Tensor,
    pointer_anchor_kinds: Tensor,
    pointer_anchor_subjects: Tensor,
    pointer_anchor_positions: Tensor,
    pointer_anchor_handles: Tensor,
    legal_edge_bitmap: Tensor | None,
) -> list[DecisionSpec]:
    """Reconstruct per-row :class:`DecisionSpec` objects from batched anchor tensors.

    The Python grammar mask consumes ``DecisionSpec`` instances. The encoder
    batch already carries the anchor metadata; we rebuild lightweight specs
    so :func:`grammar.batch_next_mask` can run.
    """

    dt_cpu = decision_type.detach().to("cpu", dtype=torch.int64).tolist()
    kinds_cpu = pointer_anchor_kinds.detach().to("cpu", dtype=torch.int64).numpy()
    subjects_cpu = pointer_anchor_subjects.detach().to("cpu", dtype=torch.int64).numpy()
    positions_cpu = pointer_anchor_positions.detach().to("cpu", dtype=torch.int64).numpy()
    handles_cpu = pointer_anchor_handles.detach().to("cpu", dtype=torch.int64).numpy()
    legal_edge_np: np.ndarray | None = None
    if legal_edge_bitmap is not None:
        legal_edge_np = legal_edge_bitmap.detach().to("cpu", dtype=torch.bool).numpy()

    specs: list[DecisionSpec] = []
    b = len(dt_cpu)
    for i in range(b):
        dt_i = int(dt_cpu[i])
        if dt_i < 0:
            specs.append(DecisionSpec(decision_type=DecisionType.PRIORITY))
            continue
        anchors: list[PointerAnchor] = []
        for j in range(int(kinds_cpu.shape[1])):
            kind = int(kinds_cpu[i, j])
            if kind < 0:
                continue
            anchors.append(
                PointerAnchor(
                    kind=AnchorKind(kind),
                    token_position=int(positions_cpu[i, j]),
                    subject_index=int(subjects_cpu[i, j]),
                    handle=int(handles_cpu[i, j]),
                )
            )
        spec = DecisionSpec(
            decision_type=DecisionType(dt_i),
            anchors=anchors,
            legal_edge_bitmap=legal_edge_np[i] if legal_edge_np is not None else None,
        )
        specs.append(spec)
    return specs


def _pointer_position_mask_from_anchors(
    specs: Sequence[DecisionSpec],
    anchor_subject_mask: np.ndarray,
    expected_kind_per_row: list[AnchorKind | None],
    encoded_seq_len: int,
) -> np.ndarray:
    """Map a per-anchor-index legal mask to a per-encoder-position mask.

    ``anchor_subject_mask`` is ``[B, N_max]`` from ``batch_next_mask``: for
    each row whose next step is a pointer step, it's True for legal anchors of
    the *expected anchor kind* indexed by ``subject_index`` order. We need a
    mask over absolute encoder positions ``[B, T_enc]`` for the pointer head.
    """

    b = len(specs)
    out = np.zeros((b, encoded_seq_len), dtype=bool)
    for i, spec in enumerate(specs):
        kind = expected_kind_per_row[i]
        if kind is None:
            continue
        anchors_of_kind = spec.anchors_of_kind(kind)
        for j, anchor in enumerate(anchors_of_kind):
            if j >= anchor_subject_mask.shape[1]:
                break
            if not bool(anchor_subject_mask[i, j]):
                continue
            pos = int(anchor.token_position)
            if 0 <= pos < encoded_seq_len:
                out[i, pos] = True
    return out


def _expected_anchor_kind_for_step(
    spec: DecisionSpec,
    prefix_tokens: Sequence[int],
) -> AnchorKind | None:
    """Best-effort: which AnchorKind is the next pointer step ranging over?

    Uses prefix structure to disambiguate DECLARE_ATTACKERS (alternates
    LEGAL_ATTACKER / DEFENDER) and DECLARE_BLOCKERS (alternates LEGAL_BLOCKER
    / LEGAL_ATTACKER). Returns ``None`` for vocab-only next steps.
    """

    dt = spec.decision_type
    if dt == DecisionType.PRIORITY:
        return AnchorKind.LEGAL_ACTION if len(prefix_tokens) == 1 else None
    if dt == DecisionType.CHOOSE_TARGETS:
        return AnchorKind.LEGAL_TARGET if len(prefix_tokens) == 1 else None
    if dt == DecisionType.DECLARE_ATTACKERS:
        # body = prefix[1:]; pointer steps are at body indices 1, 3 (mod 4).
        body_len = len(prefix_tokens) - 1
        if body_len < 0:
            return None
        if body_len % 4 == 1:
            return AnchorKind.LEGAL_ATTACKER
        if body_len % 4 == 3:
            return AnchorKind.DEFENDER
        return None
    if dt == DecisionType.DECLARE_BLOCKERS:
        body_len = len(prefix_tokens) - 1
        if body_len < 0:
            return None
        if body_len % 4 == 1:
            return AnchorKind.LEGAL_BLOCKER
        if body_len % 4 == 3:
            return AnchorKind.LEGAL_ATTACKER
        return None
    return None


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
    max_decode_len: int = 32,
    greedy: bool = False,
    temperature: float = 1.0,
) -> DecoderSampleOutput:
    """Autoregressive sampling under the grammar mask.

    One host sync per decoder step: after sampling, we copy the chosen
    token + pointer back to CPU buffers that feed the next-step mask. The
    rest of the loop (attention, sampling, log-prob gather) stays on
    device.

    The next-step legal mask is computed via :func:`grammar.batch_next_mask`
    on CPU. The native batch-handle callback (``mage.decision_mask_next``) is
    a future replacement; the Python path is kept here so rollout / training
    can run before that lands.
    """

    grammar_decoder = text_policy.grammar_decoder
    if grammar_decoder is None:
        raise RuntimeError("text_policy.grammar_decoder must be configured")
    device = encoded.device
    b, t_enc, _ = encoded.shape
    L = int(max_decode_len)

    specs = _build_batched_specs(
        decision_type,
        pointer_anchor_kinds,
        pointer_anchor_subjects,
        pointer_anchor_positions,
        pointer_anchor_handles,
        legal_edge_bitmap,
    )

    out_tokens = torch.zeros((b, L), dtype=torch.long, device=device)
    out_pointer_pos = torch.full((b, L), -1, dtype=torch.long, device=device)
    out_pointer_subjects = torch.full((b, L), -1, dtype=torch.long, device=device)
    out_is_pointer = torch.zeros((b, L), dtype=torch.bool, device=device)
    out_pad_mask = torch.zeros((b, L), dtype=torch.bool, device=device)
    out_log_probs = torch.zeros((b, L), device=device)

    # Per-row [encoder_position -> subject_index] dense map (-1 fill) so we can
    # vectorize the encoder-position → subject translation each step. This is
    # built once from the pointer-anchor metadata (same data the prefix loop
    # uses below).
    pos_to_subject_map = torch.full((b, t_enc), -1, dtype=torch.long, device=device)
    anchor_pos_dev = pointer_anchor_positions.to(device=device, dtype=torch.long)
    anchor_subj_dev = pointer_anchor_subjects.to(device=device, dtype=torch.long)
    valid_anchor = (anchor_pos_dev >= 0) & (anchor_pos_dev < t_enc) & (anchor_subj_dev >= 0)
    if int(valid_anchor.any().item()):
        n_anchors = int(anchor_pos_dev.shape[1])
        flat_row = torch.arange(b, device=device).view(-1, 1).expand(b, n_anchors)[valid_anchor]
        flat_pos = anchor_pos_dev[valid_anchor]
        flat_subj = anchor_subj_dev[valid_anchor]
        pos_to_subject_map[flat_row, flat_pos] = flat_subj

    # Prefix buffers consumed by the next-mask call.
    prefix_tokens_np = np.zeros((b, L), dtype=np.int64)
    prefix_pointers_np = np.full((b, L), -1, dtype=np.int64)
    prefix_lens_np = np.zeros((b,), dtype=np.int64)
    # Map each pointer-step pointer choice (encoder position) back to the
    # anchor's subject_index for the grammar mask.
    pos_to_subject: list[dict[int, int]] = []
    for spec in specs:
        m = {int(a.token_position): int(a.subject_index) for a in spec.anchors}
        pos_to_subject.append(m)

    state: DecoderState | None = None
    prev_token = torch.full((b,), int(GrammarVocab.PAD), dtype=torch.long, device=device)
    prev_pointer_pos = torch.full((b,), -1, dtype=torch.long, device=device)
    ended = torch.zeros((b,), dtype=torch.bool, device=device)
    ended_host = np.zeros((b,), dtype=bool)

    for step in range(L):
        # Per-row "what kind does the next pointer step need?" from the
        # current prefix, before computing the mask.
        expected_kind: list[AnchorKind | None] = []
        for i, spec in enumerate(specs):
            ln = int(prefix_lens_np[i])
            expected_kind.append(
                _expected_anchor_kind_for_step(spec, prefix_tokens_np[i, :ln].tolist())
            )

        vocab_mask_np, anchor_mask_np = batch_next_mask(
            specs, prefix_tokens_np, prefix_pointers_np, prefix_lens_np
        )
        pointer_mask_np = _pointer_position_mask_from_anchors(
            specs, anchor_mask_np, expected_kind, t_enc
        )

        vocab_mask = torch.from_numpy(vocab_mask_np).to(device=device)
        pointer_mask = torch.from_numpy(pointer_mask_np).to(device=device)

        vocab_logits, pointer_logits, state = grammar_decoder.step(
            prev_token, prev_pointer_pos, encoded, encoder_attention_mask, state
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

        valid_step = ~ended
        out_tokens[:, step] = sampled_vocab
        out_pointer_pos[:, step] = sampled_pointer
        # Translate encoder position → subject_index via per-row dense map.
        # Pointer steps that didn't fire (vocab steps) still gather a value;
        # we mask them to -1 below so the wire shape carries clean -1 fill.
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

        # Single host sync to fold the chosen action into the prefix
        # buffer for the next mask call.
        sampled_vocab_host = sampled_vocab.detach().to("cpu", dtype=torch.int64).numpy()
        sampled_pointer_host = sampled_pointer.detach().to("cpu", dtype=torch.int64).numpy()
        is_pointer_host = is_pointer_step.detach().to("cpu").numpy()
        valid_step_host = ~ended_host
        for i in range(b):
            if not valid_step_host[i]:
                continue
            ln = int(prefix_lens_np[i])
            if is_pointer_host[i]:
                pos = int(sampled_pointer_host[i])
                subj = pos_to_subject[i].get(pos, -1)
                # Pointer steps store PAD as the grammar token; the subject
                # index lives in the parallel pointers array.
                prefix_tokens_np[i, ln] = int(GrammarVocab.PAD)
                prefix_pointers_np[i, ln] = subj
            else:
                prefix_tokens_np[i, ln] = int(sampled_vocab_host[i])
                prefix_pointers_np[i, ln] = -1
                if int(sampled_vocab_host[i]) == int(GrammarVocab.END):
                    ended_host[i] = True
            prefix_lens_np[i] = ln + 1

        ended = torch.from_numpy(ended_host).to(device=device)
        # On pointer steps, prev_token is PAD (matches teacher-forced
        # decoder, which embeds the structural token, not pointer info).
        prev_token = torch.where(
            is_pointer_step,
            torch.full_like(sampled_vocab, int(GrammarVocab.PAD)),
            sampled_vocab,
        )
        prev_pointer_pos = sampled_pointer

        if bool(ended_host.all()):
            break

    # Per-row anchor count = number of anchors with subject_index >= 0.
    pointer_anchor_count = (anchor_subj_dev >= 0).sum(dim=-1).to(dtype=torch.long)

    return DecoderSampleOutput(
        output_token_ids=out_tokens,
        output_pointer_pos=out_pointer_pos,
        output_pointer_subjects=out_pointer_subjects,
        output_is_pointer=out_is_pointer,
        output_pad_mask=out_pad_mask,
        log_probs=out_log_probs,
        decision_type=decision_type.to(dtype=torch.long),
        pointer_anchor_handles=pointer_anchor_handles.to(dtype=torch.long),
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
) -> tuple[Tensor, Tensor]:
    """Reconstruct per-step ``(vocab_mask, pointer_mask)`` for stored decoder targets.

    Rolls the grammar state machine forward over the stored prefix and asks
    :func:`grammar.batch_next_mask` for the legal mask at each step. Output
    shapes: ``vocab_mask`` ``[B, L, GRAMMAR_VOCAB_SIZE]``, ``pointer_mask``
    ``[B, L, encoded_seq_len]``, both bool, on CPU. Caller is responsible for
    moving them to the encoder device.

    Mirrors the mask-build path in :mod:`policy_value_pretrain` so replay
    scoring softmaxes over the same support the sampler did.
    """

    specs = _build_batched_specs(
        decision_type,
        pointer_anchor_kinds,
        pointer_anchor_subjects,
        pointer_anchor_positions,
        pointer_anchor_handles,
        legal_edge_bitmap=None,
    )

    b, L = int(target_tokens.shape[0]), int(target_tokens.shape[1])
    tokens_np = target_tokens.detach().to("cpu", dtype=torch.int64).numpy()
    subjects_np = target_pointer_subjects.detach().to("cpu", dtype=torch.int64).numpy()
    is_ptr_np = target_is_pointer.detach().to("cpu", dtype=torch.bool).numpy()
    pad_np = target_pad_mask.detach().to("cpu", dtype=torch.bool).numpy()

    # Prefix at step `s` is the first `s` stored tokens. Pointer steps store
    # PAD as the structural token; the subject_index lives in the parallel
    # subjects array — same convention as the sampler's prefix buffers.
    prefix_tokens_np = np.zeros((b, L), dtype=np.int64)
    prefix_subjects_np = np.full((b, L), -1, dtype=np.int64)
    for i in range(b):
        for s in range(L):
            if not pad_np[i, s]:
                break
            if is_ptr_np[i, s]:
                prefix_tokens_np[i, s] = int(GrammarVocab.PAD)
                prefix_subjects_np[i, s] = int(subjects_np[i, s])
            else:
                prefix_tokens_np[i, s] = int(tokens_np[i, s])
                prefix_subjects_np[i, s] = -1

    vocab_mask = torch.zeros((b, L, GRAMMAR_VOCAB_SIZE), dtype=torch.bool)
    pointer_mask = torch.zeros((b, L, encoded_seq_len), dtype=torch.bool)
    prefix_lens_np = np.zeros((b,), dtype=np.int64)

    for step in range(L):
        expected_kind = [
            _expected_anchor_kind_for_step(
                spec, prefix_tokens_np[i, : int(prefix_lens_np[i])].tolist()
            )
            for i, spec in enumerate(specs)
        ]
        v_mask_np, anchor_mask_np = batch_next_mask(
            specs, prefix_tokens_np, prefix_subjects_np, prefix_lens_np
        )
        ptr_mask_np = _pointer_position_mask_from_anchors(
            specs, anchor_mask_np, expected_kind, encoded_seq_len
        )
        vocab_mask[:, step, :] = torch.from_numpy(v_mask_np)
        pointer_mask[:, step, :] = torch.from_numpy(ptr_mask_np)
        # Advance the prefix only for rows whose stored target has another step.
        for i in range(b):
            if step < L and pad_np[i, step]:
                prefix_lens_np[i] = step + 1

    return vocab_mask, pointer_mask


__all__ = [
    "build_replay_grammar_masks",
    "decoder_sample",
    "decoder_score_replay",
]
