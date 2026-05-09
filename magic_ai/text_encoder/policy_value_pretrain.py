"""Policy + value pretraining on extracted Forge choice situations.

Consumes the sharded torch or gzip JSONL artifact produced by
``scripts/extract_forge_choice_situations.py``. Each record stores a
pre-choice snapshot, the action text Forge actually took, and the
terminal outcome.

The pipeline trains the autoregressive grammar decoder: it renders the
decision spec via :class:`DecisionSpecRenderer`, translates the observed
event into a flat decoder target via
:mod:`magic_ai.text_encoder.forge_target_encoding`, and runs the
:func:`decoder_cross_entropy_loss` loss together with the value head.

The on-disk record schema is the V2 extractor format which persists
``DecoderTarget`` and ``PendingState`` directly; older V1 records are no
longer supported (the cutover from inline-blank training dropped the
synthetic-pending path).
"""

from __future__ import annotations

import gzip
import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import orjson
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.game_state import GameStateSnapshot, PendingState
from magic_ai.text_encoder.batch import (
    TextEncodedBatch,
    collate,
    tokenize_snapshot,
)
from magic_ai.text_encoder.decision_spec import AnchorKind, DecisionSpec, DecisionType
from magic_ai.text_encoder.forge_target_encoding import (
    DecoderTarget,
    pending_decision_type,
)
from magic_ai.text_encoder.forge_target_encoding import (
    translate as translate_observed_to_target,
)
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE, batch_next_mask
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy
from magic_ai.text_encoder.render import OracleEntry, RenderError, render_snapshot
from magic_ai.text_encoder.render_spec import DecisionSpecRenderer
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS
from magic_ai.text_encoder.training import (
    decoder_cross_entropy_loss,
    decoder_per_step_accuracy,
    value_loss,
)

ValueTargetMode = Literal["terminal", "gae", "vtrace"]


@dataclass(frozen=True)
class ForgePolicyValueConfig:
    data_path: Path
    batch_size: int
    max_tokens: int | None = None
    eval_fraction: float = 0.05
    gamma: float = 1.0
    value_target_mode: ValueTargetMode = "terminal"
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    pad_token_id: int = 0


@dataclass(frozen=True)
class ForgeDecoderBatch:
    """Decoder pipeline batch.

    ``encoded`` carries the combined ``[state, spec]`` token stream and
    pointer-anchor metadata produced by :func:`magic_ai.text_encoder.batch.collate`.
    """

    encoded: TextEncodedBatch
    output_token_ids: Tensor
    output_pointer_pos: Tensor
    output_is_pointer: Tensor
    output_pad_mask: Tensor
    vocab_mask: Tensor
    pointer_mask: Tensor
    decision_type_per_row: Tensor
    value_targets: Tensor


@dataclass(frozen=True)
class _PreparedDecoderExample:
    """Per-row payload for the decoder pipeline."""

    encoded: Any  # TextEncodedExample
    spec: DecisionSpec
    target: DecoderTarget
    value_target: float
    # subject_index → encoder_position, by anchor kind, for fast pointer lookup.
    anchor_pos_by_kind: dict[int, list[int]] = field(default_factory=dict)


def _stable_eval_bucket(game_id: str, bucket: int) -> int:
    digest = hashlib.blake2b(game_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % bucket


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    paths = (
        sorted([*path.rglob("*.pt"), *path.rglob("*.pth"), *path.rglob("*.jsonl.gz")])
        if path.is_dir()
        else [path]
    )
    for item in paths:
        if item.suffix in (".pt", ".pth"):
            payload = torch.load(item, map_location="cpu", weights_only=False)
            records = payload.get("records") if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise ValueError(f"torch choice artifact {item} does not contain a records list")
            yield from cast(list[dict[str, Any]], records)
            continue
        opener = gzip.open if item.suffix == ".gz" else open
        with opener(item, "rt", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    yield cast(dict[str, Any], orjson.loads(stripped))


def _value_target(record: dict[str, Any], cfg: ForgePolicyValueConfig) -> float:
    sign = float((record.get("outcome") or {}).get("terminal_sign") or 0.0)
    if cfg.value_target_mode == "terminal":
        return sign
    choice = record.get("choice") or {}
    remaining = max(
        0, int(choice.get("candidate_count") or 1) - int(choice.get("candidate_index") or 0) - 1
    )
    return float(sign * (cfg.gamma**remaining))


class ForgeChoiceDataset:
    def __init__(
        self,
        cfg: ForgePolicyValueConfig,
        *,
        tokenizer: PreTrainedTokenizerFast,
        oracle: dict[str, OracleEntry],
        split: str = "train",
    ) -> None:
        if split not in ("train", "eval", "all"):
            raise ValueError(f"split must be train/eval/all, got {split!r}")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.oracle = oracle
        self._spec_renderer = DecisionSpecRenderer(tokenizer)
        bucket = 0 if cfg.eval_fraction <= 0 else max(2, int(round(1.0 / cfg.eval_fraction)))
        self.records: list[dict[str, Any]] = []
        for record in _iter_records(cfg.data_path):
            game_id = str(record.get("game_id") or "")
            if bucket > 0 and split != "all":
                in_eval = _stable_eval_bucket(game_id, bucket) == 0
                if split == "train" and in_eval:
                    continue
                if split == "eval" and not in_eval:
                    continue
            self.records.append(record)
        if not self.records:
            raise ValueError(f"no Forge choice records loaded from {cfg.data_path} split={split}")

    @property
    def n_examples(self) -> int:
        return len(self.records)

    def kind_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for record in self.records:
            kind = str((record.get("choice") or {}).get("kind") or "unknown")
            out[kind] = out.get(kind, 0) + 1
        return out

    def _prepare_decoder(self, record: dict[str, Any]) -> _PreparedDecoderExample | None:
        choice = record.get("choice") or {}
        observed = cast(dict[str, Any], choice.get("observed") or {})
        snapshot = cast(dict[str, Any], (record.get("state") or {}).get("snapshot") or {})
        pending_raw = snapshot.get("pending")
        if pending_raw is None:
            return None
        pending = cast(PendingState, pending_raw)
        decision_type = pending_decision_type(pending)
        if decision_type is None:
            return None
        target = translate_observed_to_target(pending, observed)
        if target is None or not target.output_token_ids:
            return None

        try:
            rendered = render_snapshot(
                cast(GameStateSnapshot, snapshot),
                oracle=self.oracle,
                max_card_refs=MAX_CARD_REFS,
            )
            encoded = tokenize_snapshot(rendered, self.tokenizer)
            spec = self._spec_renderer.render(
                cast(GameStateSnapshot, snapshot), card_refs=rendered.card_refs
            )
        except RenderError, RuntimeError, KeyError, TypeError, ValueError, NotImplementedError:
            return None

        # Build per-kind subject_index → encoder_position lookup; the
        # spec anchors are positioned relative to the spec section start,
        # so add the row's state-token length here (the same offset the
        # collator applies).
        state_len = len(encoded.token_ids)
        anchor_pos_by_kind: dict[int, list[int]] = {}
        for anchor in spec.anchors:
            arr = anchor_pos_by_kind.setdefault(int(anchor.kind), [])
            while len(arr) <= anchor.subject_index:
                arr.append(-1)
            arr[anchor.subject_index] = int(anchor.token_position) + state_len

        return _PreparedDecoderExample(
            encoded=encoded,
            spec=spec,
            target=target,
            value_target=_value_target(record, self.cfg),
            anchor_pos_by_kind=anchor_pos_by_kind,
        )

    def _batch_from_indices(self, indices: Sequence[int]) -> ForgeDecoderBatch:
        prepared: list[_PreparedDecoderExample] = []
        cursor = 0
        while len(prepared) < len(indices) and cursor < len(indices) * 4:
            item = self._prepare_decoder(self.records[int(indices[cursor % len(indices)])])
            cursor += 1
            if item is not None:
                prepared.append(item)
        if not prepared:
            raise ValueError("no renderable Forge decoder examples in selected batch")

        encoded = collate(
            [p.encoded for p in prepared],
            [p.spec for p in prepared],
            pad_id=self.cfg.pad_token_id,
        )

        batch_size = len(prepared)
        L = max(len(p.target.output_token_ids) for p in prepared)
        T_enc = int(encoded.token_ids.shape[1])

        out_tokens = torch.zeros((batch_size, L), dtype=torch.long)
        out_pointer_pos = torch.full((batch_size, L), 0, dtype=torch.long)
        out_is_pointer = torch.zeros((batch_size, L), dtype=torch.bool)
        out_pad_mask = torch.zeros((batch_size, L), dtype=torch.bool)
        decision_type_per_row = torch.empty((batch_size,), dtype=torch.long)
        values = torch.empty((batch_size,), dtype=torch.float32)

        vocab_mask = torch.zeros((batch_size, L, GRAMMAR_VOCAB_SIZE), dtype=torch.bool)
        pointer_mask = torch.zeros((batch_size, L, T_enc), dtype=torch.bool)

        for b, item in enumerate(prepared):
            decision_type_per_row[b] = int(item.target.decision_type)
            values[b] = float(item.value_target)
            tokens = item.target.output_token_ids
            subjects = item.target.output_pointer_subjects
            is_ptrs = item.target.output_is_pointer
            n = len(tokens)
            out_pad_mask[b, :n] = True
            for i, (tok, subj, is_ptr) in enumerate(zip(tokens, subjects, is_ptrs, strict=True)):
                out_tokens[b, i] = int(tok)
                out_is_pointer[b, i] = bool(is_ptr)
                if is_ptr:
                    kind = _expected_pointer_kind(item.spec, item.target, i)
                    positions = item.anchor_pos_by_kind.get(int(kind), [])
                    if 0 <= subj < len(positions) and positions[subj] >= 0:
                        out_pointer_pos[b, i] = positions[subj]
                    else:
                        out_pad_mask[b, i] = False

        prefix_tokens_np = np.zeros((batch_size, L), dtype=np.int64)
        prefix_subjects_np = np.full((batch_size, L), -1, dtype=np.int64)
        prefix_lens_np = np.zeros((batch_size,), dtype=np.int64)
        for b, item in enumerate(prepared):
            for i, (tok, subj) in enumerate(
                zip(item.target.output_token_ids, item.target.output_pointer_subjects, strict=True)
            ):
                prefix_tokens_np[b, i] = int(tok)
                prefix_subjects_np[b, i] = int(subj)

        specs = [p.spec for p in prepared]
        per_row_max_prefix = np.array(
            [max(0, len(p.target.output_token_ids) - 1) for p in prepared], dtype=np.int64
        )
        for step in range(L):
            safe_prefix_lens = np.minimum(prefix_lens_np, per_row_max_prefix)
            v_mask, _ptr_mask_subj = batch_next_mask(
                specs,
                prefix_tokens_np,
                prefix_subjects_np,
                safe_prefix_lens,
            )
            vocab_mask[:, step, :] = torch.from_numpy(v_mask)
            for b, item in enumerate(prepared):
                if step >= len(item.target.output_token_ids):
                    continue
                if not item.target.output_is_pointer[step]:
                    continue
                kind = _expected_pointer_kind(item.spec, item.target, step)
                positions = item.anchor_pos_by_kind.get(int(kind), [])
                for pos in positions:
                    if 0 <= pos < T_enc:
                        pointer_mask[b, step, pos] = True
            for b, item in enumerate(prepared):
                if step < len(item.target.output_token_ids):
                    prefix_lens_np[b] = step + 1

        if (
            self.cfg.max_tokens is not None
            and int(encoded.token_ids.shape[1]) > self.cfg.max_tokens
        ):
            cap = int(self.cfg.max_tokens)
            encoded = _truncate_encoded_batch(encoded, max_tokens=cap)
            beyond = out_pointer_pos >= cap
            out_pad_mask = out_pad_mask & ~(out_is_pointer & beyond)
            pointer_mask = pointer_mask[:, :, :cap]

        return ForgeDecoderBatch(
            encoded=encoded,
            output_token_ids=out_tokens,
            output_pointer_pos=out_pointer_pos,
            output_is_pointer=out_is_pointer,
            output_pad_mask=out_pad_mask,
            vocab_mask=vocab_mask,
            pointer_mask=pointer_mask,
            decision_type_per_row=decision_type_per_row,
            value_targets=values,
        )

    def iter_epoch(self, batch_size: int, rng: np.random.Generator) -> Iterator[ForgeDecoderBatch]:
        order = rng.permutation(len(self.records))
        end = (len(order) // batch_size) * batch_size
        for off in range(0, end, batch_size):
            yield self._batch_from_indices([int(i) for i in order[off : off + batch_size]])

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> ForgeDecoderBatch:
        indices = rng.integers(0, len(self.records), size=batch_size)
        return self._batch_from_indices([int(i) for i in indices])


def _expected_pointer_kind(
    spec: DecisionSpec, target: DecoderTarget, step_index: int
) -> AnchorKind:
    """Anchor kind expected at ``target.output[step_index]`` (a pointer step)."""

    dt = DecisionType(target.decision_type)
    if dt is DecisionType.PRIORITY:
        return AnchorKind.LEGAL_ACTION
    if dt is DecisionType.CHOOSE_TARGETS:
        return AnchorKind.LEGAL_TARGET
    if dt is DecisionType.DECLARE_ATTACKERS:
        body_off = step_index - 1
        return AnchorKind.LEGAL_ATTACKER if body_off % 4 == 1 else AnchorKind.DEFENDER
    if dt is DecisionType.DECLARE_BLOCKERS:
        body_off = step_index - 1
        return AnchorKind.LEGAL_BLOCKER if body_off % 4 == 1 else AnchorKind.LEGAL_ATTACKER
    raise ValueError(f"decision type {dt} has no pointer steps")


class ForgePolicyValueTrainer:
    def __init__(
        self,
        policy: RecurrentTextPolicy,
        cfg: ForgePolicyValueConfig,
        *,
        lr: float,
        grad_clip: float | None,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)

    def step(self, batch: ForgeDecoderBatch, *, compute_stats: bool = True) -> dict[str, float]:
        return self._decoder_step(batch, compute_stats=compute_stats)

    def _decoder_step(self, batch: ForgeDecoderBatch, *, compute_stats: bool) -> dict[str, float]:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        text_policy = self.policy.text_policy
        if text_policy.grammar_decoder is None:
            raise RuntimeError("decoder pipeline requires TextPolicy with a grammar_decoder")
        device = next(text_policy.parameters()).device
        encoded = batch.encoded
        target_tokens = batch.output_token_ids.to(device)
        vocab_logits, pointer_logits = text_policy.forward_decoder_teacher_forced(
            encoded, target_tokens
        )
        out, _ = self.policy(encoded, h_in=None, c_in=None)

        decoder_loss = decoder_cross_entropy_loss(
            vocab_logits,
            pointer_logits,
            target_tokens,
            batch.output_pointer_pos.to(device),
            batch.output_is_pointer.to(device),
            batch.vocab_mask.to(device),
            batch.pointer_mask.to(device),
            batch.output_pad_mask.to(device),
        )
        v_loss = value_loss(out.values.float(), batch.value_targets.to(out.values.device))
        loss = self.cfg.policy_loss_weight * decoder_loss + self.cfg.value_loss_weight * v_loss
        loss.backward()
        if self.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)
        self.optimizer.step()
        if not compute_stats:
            return {}
        with torch.no_grad():
            per_step = decoder_per_step_accuracy(
                vocab_logits,
                pointer_logits,
                target_tokens,
                batch.output_pointer_pos.to(device),
                batch.output_is_pointer.to(device),
                batch.vocab_mask.to(device),
                batch.pointer_mask.to(device),
                batch.output_pad_mask.to(device),
            )
            neg_inf = torch.finfo(vocab_logits.dtype).min
            v_pred = vocab_logits.masked_fill(~batch.vocab_mask.to(device), neg_inf).argmax(-1)
            p_pred = pointer_logits.masked_fill(~batch.pointer_mask.to(device), neg_inf).argmax(-1)
            correct_per_step = torch.where(
                batch.output_is_pointer.to(device),
                p_pred == batch.output_pointer_pos.to(device),
                v_pred == target_tokens,
            )
            row_correct = ((~batch.output_pad_mask.to(device)) | correct_per_step).all(dim=-1)
            combat_kinds = (
                (batch.decision_type_per_row == int(DecisionType.DECLARE_ATTACKERS))
                | (batch.decision_type_per_row == int(DecisionType.DECLARE_BLOCKERS))
            ).to(device)
            combat_total = int(combat_kinds.sum().item())
            combat_exact = (
                float((row_correct & combat_kinds).sum().item()) / combat_total
                if combat_total > 0
                else 0.0
            )
            pred_sign = torch.sign(out.values.float())
            target = batch.value_targets.to(out.values.device)
            non_draw = target.abs() > 1e-6
            sign_acc = (
                (pred_sign[non_draw] == torch.sign(target[non_draw])).float().mean()
                if non_draw.any()
                else torch.tensor(0.0, device=loss.device)
            )
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(decoder_loss.detach()),
            "value_loss": float(v_loss.detach()),
            "decoder_step_accuracy": float(per_step["accuracy"]),
            "decoder_combat_exact_match": float(combat_exact),
            "value_sign_accuracy": float(sign_acc.detach()),
            "grad_norm": float(grad_norm.detach()),
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataset: ForgeChoiceDataset,
        rng: np.random.Generator,
        *,
        batches: int = 8,
        device: torch.device,
    ) -> dict[str, float]:
        self.policy.eval()
        totals: dict[str, float] = {}
        count = 0
        for _ in range(batches):
            batch = dataset.sample_batch(self.cfg.batch_size, rng)
            batch = _batch_to_device(batch, device)
            text_policy = self.policy.text_policy
            if text_policy.grammar_decoder is None:
                continue
            vocab_logits, pointer_logits = text_policy.forward_decoder_teacher_forced(
                batch.encoded, batch.output_token_ids
            )
            out, _ = self.policy(batch.encoded, h_in=None, c_in=None)
            policy_l = decoder_cross_entropy_loss(
                vocab_logits,
                pointer_logits,
                batch.output_token_ids,
                batch.output_pointer_pos,
                batch.output_is_pointer,
                batch.vocab_mask,
                batch.pointer_mask,
                batch.output_pad_mask,
            )
            v_loss = value_loss(out.values.float(), batch.value_targets)
            stats = {
                "eval_policy_loss": float(policy_l.detach()),
                "eval_value_loss": float(v_loss.detach()),
            }
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        denom = max(1, count)
        return {key: value / denom for key, value in totals.items()} | {
            "eval_batches": float(count)
        }


def _encoded_to_device(batch: TextEncodedBatch, device: torch.device) -> TextEncodedBatch:
    return TextEncodedBatch(
        token_ids=batch.token_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        card_ref_positions=batch.card_ref_positions.to(device),
        seq_lengths=batch.seq_lengths.to(device),
        total_tokens=batch.total_tokens,
        seq_lengths_host=batch.seq_lengths_host,
        spec_tokens=batch.spec_tokens.to(device),
        spec_lens=batch.spec_lens.to(device),
        decision_type=batch.decision_type.to(device),
        pointer_anchor_positions=batch.pointer_anchor_positions.to(device),
        pointer_anchor_kinds=batch.pointer_anchor_kinds.to(device),
        pointer_anchor_subjects=batch.pointer_anchor_subjects.to(device),
        pointer_anchor_handles=batch.pointer_anchor_handles.to(device),
        legal_edge_bitmap=(
            batch.legal_edge_bitmap.to(device) if batch.legal_edge_bitmap is not None else None
        ),
    )


def _truncate_encoded_batch(batch: TextEncodedBatch, *, max_tokens: int) -> TextEncodedBatch:
    seq_lengths = batch.seq_lengths.clamp(max=max_tokens)
    attention_mask = batch.attention_mask[:, :max_tokens].clone()
    for row, seq_len in enumerate(seq_lengths.tolist()):
        attention_mask[row, int(seq_len) :] = 0
    card_ref_positions = batch.card_ref_positions.clone()
    card_ref_positions[card_ref_positions >= max_tokens] = -1
    return TextEncodedBatch(
        token_ids=batch.token_ids[:, :max_tokens].contiguous(),
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lengths,
        total_tokens=int(seq_lengths.sum().item()),
        seq_lengths_host=tuple(int(v) for v in seq_lengths.tolist()),
        spec_tokens=batch.spec_tokens,
        spec_lens=batch.spec_lens,
        decision_type=batch.decision_type,
        pointer_anchor_positions=batch.pointer_anchor_positions,
        pointer_anchor_kinds=batch.pointer_anchor_kinds,
        pointer_anchor_subjects=batch.pointer_anchor_subjects,
        pointer_anchor_handles=batch.pointer_anchor_handles,
        legal_edge_bitmap=batch.legal_edge_bitmap,
    )


def _batch_to_device(batch: ForgeDecoderBatch, device: torch.device) -> ForgeDecoderBatch:
    return ForgeDecoderBatch(
        encoded=_encoded_to_device(batch.encoded, device),
        output_token_ids=batch.output_token_ids.to(device),
        output_pointer_pos=batch.output_pointer_pos.to(device),
        output_is_pointer=batch.output_is_pointer.to(device),
        output_pad_mask=batch.output_pad_mask.to(device),
        vocab_mask=batch.vocab_mask.to(device),
        pointer_mask=batch.pointer_mask.to(device),
        decision_type_per_row=batch.decision_type_per_row.to(device),
        value_targets=batch.value_targets.to(device),
    )


def batches_per_epoch(n_examples: int, batch_size: int) -> int:
    return int(math.floor(n_examples / batch_size))


__all__ = [
    "ForgeChoiceDataset",
    "ForgeDecoderBatch",
    "ForgePolicyValueConfig",
    "ForgePolicyValueTrainer",
    "ValueTargetMode",
    "batches_per_epoch",
    "_batch_to_device",
]
