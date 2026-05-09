"""Training-facing actor-critic wrapper for the text encoder policy.

Phase 5 of the inline-blank cutover replaced the inline-blank sampling
machinery with the autoregressive grammar decoder. The decoder is now the
*only* sampling and replay-scoring path; this module exposes:

- :func:`decoder_sample` — autoregressive sampling on a padded encoder batch.
  One host sync per decoder step (for the prefix → next-mask call). Returns
  per-step log probabilities under the (vocab|pointer) mask of the chosen
  action so PPO/R-NaD can compute importance ratios.
- :func:`decoder_score_replay` — teacher-forced per-row log p of stored
  decoder targets, plus per-step entropy.
- :func:`decode_decoder_action` — translate a single decoded token sequence
  into an :class:`ActionRequest` for the engine.
- :class:`TextActorCritic` — thin wrapper over :class:`RecurrentTextPolicy`
  that owns live per-env / per-player LSTM state and delegates sampling and
  scoring to the helpers above.

Other phases of the cutover (Phase 6 wiring R-NaD / PPO trainers to call
``decoder_score_replay`` end-to-end, Phase 7 native batch-handle plumbing
into ``decoder_sample``) build on top of this file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from magic_ai.actions import (
    ActionRequest,
    BlockerAssignState,
    action_from_attackers,
    action_from_choice_accepted,
    action_from_choice_index,
    action_from_priority_candidate,
    build_priority_candidates,
)
from magic_ai.game_state import PendingState
from magic_ai.text_encoder.batch import PackedTextBatch, TextEncodedBatch
from magic_ai.text_encoder.decision_spec import (
    AnchorKind,
    DecisionSpec,
    DecisionType,
    PointerAnchor,
)
from magic_ai.text_encoder.decoder import DecoderState, combined_sample
from magic_ai.text_encoder.grammar import (
    DIGIT_0_ID,
    DIGIT_9_ID,
    GRAMMAR_VOCAB_SIZE,
    GrammarVocab,
    batch_next_mask,
)
from magic_ai.text_encoder.policy import TextPolicy
from magic_ai.text_encoder.recurrent import (
    RecurrentTextPolicy,
    RecurrentTextPolicyConfig,
)
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer

# --------------------------------------------------------------------------- #
# Public dataclasses                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DecoderSampleOutput:
    """Result of one batched ``decoder_sample`` call.

    Lengths are right-padded to ``max_decode_len``. ``output_pad_mask``
    is False for steps after END (and for rows that ended early).
    """

    output_token_ids: Tensor  # [B, L] int64 (PAD = 0)
    output_pointer_pos: Tensor  # [B, L] int64 (-1 fill, encoder position)
    output_pointer_subjects: Tensor  # [B, L] int64 (-1 fill, anchor subject_index)
    output_is_pointer: Tensor  # [B, L] bool
    output_pad_mask: Tensor  # [B, L] bool (True = valid, False = post-END/pad)
    log_probs: Tensor  # [B, L] float — per-step log p of chosen action
    decision_type: Tensor  # [B] int64 (passed through)
    pointer_anchor_handles: Tensor  # [B, N_max] int64 (passed through)
    pointer_anchor_count: Tensor  # [B] int64 (passed through)


@dataclass(frozen=True)
class DecoderReplayScores:
    """Result of teacher-forced replay scoring for the grammar decoder."""

    per_row_log_pi: Tensor  # [B] sum of per-step log p of stored target
    per_row_entropy: Tensor  # [B] sum of per-step entropy of the stored decision
    per_step_log_pi: Tensor  # [B, L] per-step log p (zeroed at pad positions)


@dataclass(frozen=True)
class DecoderDecisionLayout:
    """Per-row decoded action returned by the sampler (replaces inline-blank
    ``TextDecisionLayout``).
    """

    output_token_ids: Tensor  # [L] int64
    output_pointer_pos: Tensor  # [L] int64 (encoder position, -1 fill)
    output_pointer_subjects: Tensor  # [L] int64 (anchor subject_index, -1 fill)
    output_is_pointer: Tensor  # [L] bool
    output_pad_mask: Tensor  # [L] bool
    decision_type: int
    pointer_anchor_handles: Tensor  # [N_max] int64
    pointer_anchor_count: int


@dataclass(frozen=True)
class NativeTextDecoderBatch:
    """Wire shape for the IMPALA inference-server reply / actor step input.

    Mirrors the Go-side ``MageDecoderStepRequest`` struct: a flat int
    rectangle that the Go entrypoint slices and dispatches per-decision-
    type. Plus the IMPALA-side bookkeeping (log_probs, value, pad_mask)
    used by replay scoring and importance ratios.
    """

    decision_type: Tensor  # [B] int32
    output_token_ids: Tensor  # [B, L_max] int32 (PAD = 0)
    output_pointer_subjects: Tensor  # [B, L_max] int32 (anchor subject_index, -1 fill)
    output_is_pointer: Tensor  # [B, L_max] bool
    output_lens: Tensor  # [B] int32 (number of valid steps per row)
    pointer_anchor_handles: Tensor  # [B, N_max] int32
    pointer_anchor_count: Tensor  # [B] int32
    log_probs: Tensor  # [B, L_max] float (zero at pad)
    value: Tensor  # [B] float
    output_pad_mask: Tensor  # [B, L_max] bool

    def __len__(self) -> int:
        return int(self.decision_type.shape[0])

    def __getitem__(self, key: slice | int) -> NativeTextDecoderBatch:
        """Row slice. Integer keys are wrapped to a length-1 batch."""
        if isinstance(key, int):
            key = slice(key, key + 1)
        return NativeTextDecoderBatch(
            decision_type=self.decision_type[key],
            output_token_ids=self.output_token_ids[key],
            output_pointer_subjects=self.output_pointer_subjects[key],
            output_is_pointer=self.output_is_pointer[key],
            output_lens=self.output_lens[key],
            pointer_anchor_handles=self.pointer_anchor_handles[key],
            pointer_anchor_count=self.pointer_anchor_count[key],
            log_probs=self.log_probs[key],
            value=self.value[key],
            output_pad_mask=self.output_pad_mask[key],
        )

    @classmethod
    def concat(cls, parts: Sequence[NativeTextDecoderBatch]) -> NativeTextDecoderBatch:
        """Row-axis concatenation; left-pads ragged L_max / N_max with fill."""
        if len(parts) == 1:
            return parts[0]
        l_max = max(int(p.output_token_ids.shape[1]) for p in parts)
        n_max = max(int(p.pointer_anchor_handles.shape[1]) for p in parts)

        def _pad2d(t: Tensor, width: int, *, fill: int | float) -> Tensor:
            cur = int(t.shape[1])
            if cur == width:
                return t
            rows = int(t.shape[0])
            out = torch.full((rows, width), fill, dtype=t.dtype, device=t.device)
            out[:, :cur] = t
            return out

        return cls(
            decision_type=torch.cat([p.decision_type for p in parts], dim=0),
            output_token_ids=torch.cat(
                [_pad2d(p.output_token_ids, l_max, fill=0) for p in parts], dim=0
            ),
            output_pointer_subjects=torch.cat(
                [_pad2d(p.output_pointer_subjects, l_max, fill=-1) for p in parts], dim=0
            ),
            output_is_pointer=torch.cat(
                [_pad2d(p.output_is_pointer, l_max, fill=0) for p in parts], dim=0
            ),
            output_lens=torch.cat([p.output_lens for p in parts], dim=0),
            pointer_anchor_handles=torch.cat(
                [_pad2d(p.pointer_anchor_handles, n_max, fill=0) for p in parts], dim=0
            ),
            pointer_anchor_count=torch.cat([p.pointer_anchor_count for p in parts], dim=0),
            log_probs=torch.cat([_pad2d(p.log_probs, l_max, fill=0.0) for p in parts], dim=0),
            value=torch.cat([p.value for p in parts], dim=0),
            output_pad_mask=torch.cat(
                [_pad2d(p.output_pad_mask, l_max, fill=0) for p in parts], dim=0
            ),
        )


# Kept for backwards-compatible imports while the rest of the cutover lands.
@dataclass(frozen=True)
class NativeTextReplayPayload:
    """Placeholder for the legacy native replay payload struct.

    Phase 5 stripped the inline-blank training path. The native replay
    pipeline will be re-wired around :class:`DecoderDecisionPayload` in
    a later phase. Symbol kept so unrelated import sites don't blow up
    mid-cutover; constructing one will fail on access.
    """

    encoded: PackedTextBatch | None = None


@dataclass(frozen=True)
class NativeTextSampleBatch:
    """Result of ``TextActorCritic.sample_batch`` — a list of decoded actions.

    ``replay_payload`` will be wired up in a later phase; for now the field
    is kept ``None`` so call-sites that pass it through don't crash.
    """

    decoded: list[DecoderDecisionLayout]
    log_probs: Tensor  # [B, L]
    replay_rows: list[int]
    replay_payload: NativeTextReplayPayload | None = None


# --------------------------------------------------------------------------- #
# Sampling / scoring                                                          #
# --------------------------------------------------------------------------- #


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


def native_decoder_batch_from_sample(
    sample: DecoderSampleOutput,
    *,
    value: Tensor,
) -> NativeTextDecoderBatch:
    """Convert a :class:`DecoderSampleOutput` (+ per-row value head outputs)
    into the IMPALA wire shape consumed by the inference-server reply and
    the actor-side ``mage.batch_step_by_decoder_action`` call.
    """

    output_lens = sample.output_pad_mask.sum(dim=-1).to(dtype=torch.int32)
    return NativeTextDecoderBatch(
        decision_type=sample.decision_type.to(dtype=torch.int32),
        output_token_ids=sample.output_token_ids.to(dtype=torch.int32),
        output_pointer_subjects=sample.output_pointer_subjects.to(dtype=torch.int32),
        output_is_pointer=sample.output_is_pointer.to(dtype=torch.bool),
        output_lens=output_lens,
        pointer_anchor_handles=sample.pointer_anchor_handles.to(dtype=torch.int32),
        pointer_anchor_count=sample.pointer_anchor_count.to(dtype=torch.int32),
        log_probs=sample.log_probs.to(dtype=torch.float32),
        value=value.to(dtype=torch.float32),
        output_pad_mask=sample.output_pad_mask.to(dtype=torch.bool),
    )


# --------------------------------------------------------------------------- #
# Engine action translation                                                   #
# --------------------------------------------------------------------------- #


def _digit_seq_to_int(token_ids: Sequence[int]) -> int:
    value = 0
    saw_digit = False
    for tok in token_ids:
        t = int(tok)
        if not (DIGIT_0_ID <= t <= DIGIT_9_ID):
            continue
        value = value * 10 + (t - DIGIT_0_ID)
        saw_digit = True
    return value if saw_digit else 0


def decode_decoder_action(
    pending: PendingState,
    layout: DecoderDecisionLayout,
) -> ActionRequest:
    """Translate a decoded token sequence into an engine ``ActionRequest``.

    The :class:`DecoderDecisionLayout` carries pointer-anchor handles for
    each pointer step; the handle is the engine option index (PRIORITY) or
    object id (combat / targets / mode). Decoding routes per
    :class:`DecisionType` and falls back to a safe default if the sequence
    is malformed.
    """

    dt_val = int(layout.decision_type)
    if dt_val < 0:
        # No pending decision spec was attached to this row; fall back to a
        # pass / no-op action that the engine will accept for the current pending.
        return action_from_choice_index(0)
    dt = DecisionType(dt_val)
    tokens = layout.output_token_ids.detach().to("cpu", dtype=torch.long).tolist()
    is_ptr = layout.output_is_pointer.detach().to("cpu").tolist()
    pad = layout.output_pad_mask.detach().to("cpu").tolist()
    handles = layout.pointer_anchor_handles.detach().to("cpu", dtype=torch.long).tolist()

    # Walk pointer steps in order; the i-th valid pointer step maps to
    # handles[i] (anchor list ordering). This requires the caller to pass
    # the anchor handles in subject_index order — which is how the renderer
    # builds them.
    ptr_step_indices: list[int] = []
    for i in range(len(tokens)):
        if not bool(pad[i]):
            break
        if bool(is_ptr[i]):
            ptr_step_indices.append(i)

    # Look up handles by anchor subject_index. ``pointer_pos`` here holds
    # the encoder absolute position chosen by the pointer head; we need the
    # subject index to index into ``handles``. Without the spec we don't
    # have the position→subject map at decode time, but the layout above
    # was built with handles in subject_index order and ``pointer_pos[i]``
    # points at one specific anchor's encoder position. We assume the caller
    # tracks the subject ordering (renderer guarantees it) and resolves
    # handles by the anchor at the chosen position. Fallback below uses
    # ptr_step ordinal as the pointer-step index into handles (matches the
    # PRIORITY case where there is exactly one pointer step).
    def _handle_for_pointer(step_idx: int, ordinal: int) -> int:
        del step_idx
        if ordinal < len(handles):
            return int(handles[ordinal])
        return 0

    if dt == DecisionType.PRIORITY:
        # Single pointer step → engine option index.
        if not ptr_step_indices:
            candidates = build_priority_candidates(pending)
            return action_from_priority_candidate(candidates[0]) if candidates else {"kind": "pass"}
        opt_idx = _handle_for_pointer(ptr_step_indices[0], 0)
        candidates = build_priority_candidates(pending)
        for cand in candidates:
            if cand.option_index == opt_idx:
                return action_from_priority_candidate(cand)
        if candidates:
            return action_from_priority_candidate(candidates[0])
        return {"kind": "pass"}

    if dt == DecisionType.DECLARE_ATTACKERS:
        # Pairs of (LEGAL_ATTACKER ptr, DEFENDER ptr). Build a binary attacker
        # selection by attacker handle.
        options = pending.get("options", []) or []
        n = len(options)
        selected = [False] * n
        for ord_i in range(0, len(ptr_step_indices), 2):
            attacker_handle = _handle_for_pointer(ptr_step_indices[ord_i], ord_i)
            if 0 <= attacker_handle < n:
                selected[attacker_handle] = True
        return action_from_attackers(pending, selected)

    if dt == DecisionType.DECLARE_BLOCKERS:
        # Pairs of (LEGAL_BLOCKER ptr, ATTACKER ptr). Engine wants
        # ``{blockers: [{blocker, attacker}, ...]}`` keyed by id.
        options = pending.get("options", []) or []
        assignments: list[BlockerAssignState] = []
        for ord_i in range(0, len(ptr_step_indices) - 1, 2):
            blk_h = _handle_for_pointer(ptr_step_indices[ord_i], ord_i)
            atk_h = _handle_for_pointer(ptr_step_indices[ord_i + 1], ord_i + 1)
            if not (0 <= blk_h < len(options)):
                continue
            opt = options[blk_h]
            blocker_id = str(opt.get("permanent_id", "") or "")
            targets = opt.get("valid_targets", []) or []
            if not (0 <= atk_h < len(targets)):
                continue
            attacker_id = str(targets[atk_h].get("id", "") or "")
            if blocker_id and attacker_id:
                assignments.append(BlockerAssignState(blocker=blocker_id, attacker=attacker_id))
        return ActionRequest(blockers=assignments)

    if dt == DecisionType.CHOOSE_TARGETS:
        if not ptr_step_indices:
            return action_from_choice_index(0)
        h = _handle_for_pointer(ptr_step_indices[0], 0)
        return action_from_choice_index(int(h))

    if dt == DecisionType.MAY:
        # Look for YES / NO grammar tokens.
        for i, t in enumerate(tokens):
            if not bool(pad[i]):
                break
            if int(t) == int(GrammarVocab.YES):
                return action_from_choice_accepted(True)
            if int(t) == int(GrammarVocab.NO):
                return action_from_choice_accepted(False)
        return action_from_choice_accepted(False)

    if dt in (DecisionType.CHOOSE_MODE, DecisionType.CHOOSE_X):
        digits = [tokens[i] for i in range(len(tokens)) if bool(pad[i])]
        return action_from_choice_index(_digit_seq_to_int(digits))

    return action_from_choice_index(0)


# --------------------------------------------------------------------------- #
# TextActorCritic wrapper                                                     #
# --------------------------------------------------------------------------- #


class TextActorCritic(nn.Module):
    """Stateful wrapper around :class:`RecurrentTextPolicy`.

    Owns live per-env, per-player LSTM state. Sampling and replay scoring
    delegate to the module-level ``decoder_*`` helpers above.

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

    def clone_for_rnad(self) -> TextActorCritic:
        """Deep-copy the underlying policy weights for R-NaD's target / reg copies."""
        clone = TextActorCritic(self.policy.cfg)
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

    def lstm_env_state_inputs(
        self,
        env_indices: Sequence[int],
        perspective_player_indices: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        slots = torch.tensor(
            [
                int(e) * self._players_per_env + int(p)
                for e, p in zip(env_indices, perspective_player_indices, strict=True)
            ],
            dtype=torch.long,
            device=self.device,
        )
        h_in = self.live_lstm_h.index_select(1, slots).contiguous()
        c_in = self.live_lstm_c.index_select(1, slots).contiguous()
        return h_in, c_in

    def scatter_lstm_env_states(
        self,
        env_indices: Sequence[int],
        perspective_player_indices: Sequence[int],
        h_out: Tensor,
        c_out: Tensor,
    ) -> None:
        slots = torch.tensor(
            [
                int(e) * self._players_per_env + int(p)
                for e, p in zip(env_indices, perspective_player_indices, strict=True)
            ],
            dtype=torch.long,
            device=self.device,
        )
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

        h_in, c_in = self.lstm_env_state_inputs(env_indices, perspective_player_indices)
        device = self.device
        moved = batch  # caller is expected to have moved the batch already
        # Single encoder forward with history injection + LSTM update;
        # decoder cross-attn requires the padded [B, T, D] hidden tensor.
        encoded, h_out, c_out = self.policy.encoder_forward_padded_with_history(
            moved, h_in=h_in, c_in=c_in
        )
        self.scatter_lstm_env_states(env_indices, perspective_player_indices, h_out, c_out)

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
    ) -> DecoderReplayScores:
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
                "TextActorCritic.rollout_buffer is None; cannot gather replay decoder."
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
            raise RuntimeError("TextActorCritic.rollout_buffer is None; cannot count replay steps.")
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
        attn_mask = batch.encoded.cu_seqlens.new_zeros(0)  # placeholder; pack_batch path
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
        L = int(target_tokens.shape[1])
        vocab_mask = torch.ones((b, L, GRAMMAR_VOCAB_SIZE), dtype=torch.bool, device=encoded.device)
        pointer_mask = torch.ones((b, L, t_enc), dtype=torch.bool, device=encoded.device)
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
        """Decoder analog of slot's per-choice scoring.

        Each replay row is exactly one decision; the per-choice flat
        tensors collapse to one entry per row. The returned
        :class:`ReplayPerChoice` packs the row-level logits / log-probs
        so R-NaD's NeuRD assembly can run on the decoder pipeline with
        the same downstream code.
        """
        del lstm_state_override, hidden_override, cached  # decoder path is stateless across calls
        from magic_ai.replay_decisions import ReplayPerChoice

        log_pi, entropy, values, _ = self.evaluate_replay_batch(replay_rows)
        device = log_pi.device
        n = int(log_pi.shape[0])
        zeros_b = torch.zeros(n, dtype=log_pi.dtype, device=device)
        zeros_long_b = torch.zeros(n, dtype=torch.long, device=device)
        # Decoder rows have no may-bit; expose an inactive may mask.
        per_choice = ReplayPerChoice(
            flat_logits=log_pi,  # one logit per row (the row's log p, in lieu of per-choice)
            flat_log_probs=log_pi,
            group_idx=torch.arange(n, dtype=torch.long, device=device),
            choice_cols=zeros_long_b,
            is_sampled_flat=torch.ones(n, dtype=torch.bool, device=device),
            decision_group_id_flat=torch.arange(n, dtype=torch.long, device=device),
            step_for_decision_group=torch.arange(n, dtype=torch.long, device=device),
            may_is_active=torch.zeros(n, dtype=torch.bool, device=device),
            may_logits_per_step=zeros_b,
            may_selected_per_step=zeros_b,
            behavior_action_log_prob_per_decision_group=zeros_b,
        )
        return log_pi, entropy, values, per_choice

    def write_ppo_targets(
        self,
        replay_rows: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> None:
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer is None; cannot write PPO targets.")
        self.rollout_buffer.write_ppo_targets(replay_rows, old_log_probs, returns, advantages)

    def gather_ppo_targets(self, replay_rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer is None; cannot gather PPO targets.")
        return self.rollout_buffer.gather_ppo_targets(replay_rows)

    def gather_replay_old_log_prob_value(
        self,
        replay_rows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.rollout_buffer is None:
            raise RuntimeError("TextActorCritic.rollout_buffer is None; cannot gather replay rows.")
        idx = replay_rows.to(device=self.rollout_buffer.device, dtype=torch.long)
        return self.rollout_buffer.core.old_log_prob[idx], self.rollout_buffer.core.value[idx]

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


__all__ = [
    "DecoderDecisionLayout",
    "DecoderReplayScores",
    "DecoderSampleOutput",
    "NativeTextDecoderBatch",
    "NativeTextReplayPayload",
    "NativeTextSampleBatch",
    "TextActorCritic",
    "decode_decoder_action",
    "decoder_sample",
    "decoder_score_replay",
    "native_decoder_batch_from_sample",
    "GRAMMAR_VOCAB_SIZE",
]
