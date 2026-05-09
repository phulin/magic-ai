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
    output_pointer_pos: Tensor  # [B, L] int64 (-1 fill)
    output_is_pointer: Tensor  # [B, L] bool
    output_pad_mask: Tensor  # [B, L] bool (True = valid, False = post-END/pad)
    log_probs: Tensor  # [B, L] float — per-step log p of chosen action
    decision_type: Tensor  # [B] int64 (passed through)
    pointer_anchor_handles: Tensor  # [B, N_max] int64 (passed through)


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
    output_pointer_pos: Tensor  # [L] int64
    output_is_pointer: Tensor  # [L] bool
    output_pad_mask: Tensor  # [L] bool
    decision_type: int
    pointer_anchor_handles: Tensor  # [N_max] int64


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
    out_is_pointer = torch.zeros((b, L), dtype=torch.bool, device=device)
    out_pad_mask = torch.zeros((b, L), dtype=torch.bool, device=device)
    out_log_probs = torch.zeros((b, L), device=device)

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

    return DecoderSampleOutput(
        output_token_ids=out_tokens,
        output_pointer_pos=out_pointer_pos,
        output_is_pointer=out_is_pointer,
        output_pad_mask=out_pad_mask,
        log_probs=out_log_probs,
        decision_type=decision_type.to(dtype=torch.long),
        pointer_anchor_handles=pointer_anchor_handles.to(dtype=torch.long),
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
        encoded = self.policy.text_policy.encoder(moved)
        # Update recurrent state through the standard policy forward.
        _, (h_out, c_out) = self.policy(moved, h_in=h_in, c_in=c_in)
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
                    output_is_pointer=sample.output_is_pointer[i],
                    output_pad_mask=sample.output_pad_mask[i],
                    decision_type=int(sample.decision_type[i].item()),
                    pointer_anchor_handles=sample.pointer_anchor_handles[i],
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


__all__ = [
    "DecoderDecisionLayout",
    "DecoderReplayScores",
    "DecoderSampleOutput",
    "NativeTextReplayPayload",
    "NativeTextSampleBatch",
    "TextActorCritic",
    "decode_decoder_action",
    "decoder_sample",
    "decoder_score_replay",
    "GRAMMAR_VOCAB_SIZE",
]
