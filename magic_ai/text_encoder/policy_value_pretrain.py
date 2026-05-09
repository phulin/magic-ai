"""Policy + value pretraining on extracted Forge choice situations.

Consumes the sharded torch or gzip JSONL artifact produced by
``scripts/extract_forge_choice_situations.py``. Each record stores a
pre-choice snapshot, the action text Forge actually took, and the
terminal outcome.

Two policy training modes are supported:

* **Inline-blank** (default, ``ForgePolicyValueConfig.decoder=False``).
  Reconstructs a conservative inline-blank legal set from the visible
  state, maps the observed choice onto the corresponding blank target,
  and trains :class:`InlineBlankPolicy` per
  ``docs/text_encoder_inline_blanks_plan.md``.
* **Decoder** (``ForgePolicyValueConfig.decoder=True``).
  Renders the decision spec via :func:`render_decision_spec`, translates
  the observed event into a flat decoder target sequence via
  :mod:`magic_ai.text_encoder.forge_target_encoding`, and trains the
  autoregressive grammar decoder with
  :func:`decoder_cross_entropy_loss` (see
  ``docs/decoder_grammar_plan.md`` step 10).

The on-disk record schema is currently V1 (no persisted PendingState);
the decoder loader synthesizes the pending state at load time using the
same conservative reconstruction as the inline-blank path. A future
extractor change will bump ``FORMAT_VERSION`` to 2 and persist the
PendingState + DecoderTarget directly; the loader detects the version
and skips synthesis when V2 records are available.
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

from magic_ai.game_state import GameStateSnapshot, PendingOptionState, PendingState
from magic_ai.text_encoder.batch import (
    TextEncodedBatch,
    collate,
    collate_with_specs,
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
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS, MAX_NUM
from magic_ai.text_encoder.training import (
    decoder_cross_entropy_loss,
    decoder_per_step_accuracy,
    inline_blank_per_blank_accuracy,
    inline_blank_per_blank_loss,
    inline_blank_priority_accuracy,
    inline_blank_priority_loss,
    value_loss,
)

ChoiceKind = Literal["priority", "attack", "block", "may", "choose"]
ValueTargetMode = Literal["terminal", "gae", "vtrace"]

# Pending-kind dispatch for the decoder pipeline. Mirrors the keys the
# extractor emits (`choice.kind`) onto pending-state kinds the renderer
# understands.
_CHOICE_KIND_TO_PENDING_KIND: dict[str, str] = {
    "priority": "priority",
    "attack": "attackers",
    "block": "blockers",
    "may": "may",
    "choose": "permanent",
}


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
    # When True, train the autoregressive grammar decoder (decoder pipeline).
    # When False, train the legacy inline-blank policy.
    decoder: bool = False


@dataclass(frozen=True)
class ForgeChoiceBatch:
    """Inline-blank (V1) batch shape; populated when ``cfg.decoder=False``."""

    encoded: TextEncodedBatch
    priority_target_blank: Tensor
    per_blank_target_legal: Tensor
    value_targets: Tensor


@dataclass(frozen=True)
class ForgeDecoderBatch:
    """Decoder pipeline batch.

    ``encoded`` carries the combined ``[state, spec]`` token stream and
    pointer-anchor metadata produced by
    :func:`magic_ai.text_encoder.batch.collate_with_specs`.

    Decoder-target tensors:

    * ``output_token_ids [B, L]`` int64 — grammar token id per step
      (PAD on pointer steps; the supervised target is the pointer
      position, not the vocab id).
    * ``output_pointer_pos [B, L]`` int64 — absolute encoder position
      of the chosen anchor on pointer steps (-1 elsewhere).
    * ``output_is_pointer [B, L]`` bool — True at pointer steps.
    * ``output_pad_mask [B, L]`` bool — True at valid steps, False at
      sequence padding.
    * ``vocab_mask [B, L, V_grammar]`` / ``pointer_mask [B, L, T_enc]``
      bool — per-step legality masks consulted by the loss.
    * ``decision_type_per_row [B]`` int64 — for combat exact-match metric.
    * ``value_targets [B]`` float — same as the inline-blank batch.
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
class _PreparedExample:
    encoded: Any
    priority_target_blank: int
    per_blank_target_legal: list[int]
    value_target: float


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


def _card_name(card: dict[str, Any]) -> str:
    return str(card.get("Name") or card.get("name") or "")


def _card_id(card: dict[str, Any]) -> str:
    return str(card.get("ID") or card.get("id") or "")


def _is_creature(card: dict[str, Any], oracle: dict[str, OracleEntry]) -> bool:
    name = _card_name(card)
    entry = oracle.get(name)
    if entry is None:
        return False
    return "creature" in str(entry.get("type_line") or "").lower()


def _is_land(card: dict[str, Any], oracle: dict[str, OracleEntry]) -> bool:
    name = _card_name(card)
    entry = oracle.get(name)
    if entry is None:
        return False
    return "land" in str(entry.get("type_line") or "").lower()


def _untapped(card: dict[str, Any]) -> bool:
    return card.get("Tapped") is not True and card.get("tapped") is not True


def _self_player(snapshot: dict[str, Any]) -> dict[str, Any]:
    players = snapshot.get("players") or []
    return cast(dict[str, Any], players[0]) if players else {}


def _opp_player(snapshot: dict[str, Any]) -> dict[str, Any]:
    players = snapshot.get("players") or []
    return cast(dict[str, Any], players[1]) if len(players) > 1 else {}


def _battlefield(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], c) for c in player.get("Battlefield") or []]


def _hand(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], c) for c in player.get("Hand") or []]


def _contains_card_name(text: str, card: dict[str, Any]) -> bool:
    name = _card_name(card)
    return bool(name and name in text)


def _priority_actions_and_target(
    snapshot: dict[str, Any],
    observed: dict[str, Any],
    oracle: dict[str, OracleEntry],
) -> tuple[list[PendingOptionState], int]:
    player = _self_player(snapshot)
    raw = str(observed.get("raw") or "")
    actions: list[PendingOptionState] = []
    selected = -1
    lower = raw.lower()
    for card in _hand(player):
        cid = _card_id(card)
        name = _card_name(card)
        if not cid or not name:
            continue
        kind = "play" if _is_land(card, oracle) else "cast"
        idx = len(actions)
        actions.append(
            cast(
                PendingOptionState,
                {
                    "id": f"{kind}:{cid}",
                    "kind": kind,
                    "card_id": cid,
                    "card_name": name,
                },
            )
        )
        if (
            selected < 0
            and name in raw
            and ((" played " in lower and kind == "play") or (" cast " in lower and kind == "cast"))
        ):
            selected = idx
    for card in _battlefield(player):
        cid = _card_id(card)
        name = _card_name(card)
        if not cid or not name or _is_land(card, oracle):
            continue
        idx = len(actions)
        actions.append(
            cast(
                PendingOptionState,
                {
                    "id": f"activate:{cid}",
                    "kind": "activate",
                    "permanent_id": cid,
                    "ability_index": 0,
                    "card_name": name,
                },
            )
        )
        if selected < 0 and " activated " in lower and name in raw:
            selected = idx
    pass_idx = len(actions)
    actions.append(cast(PendingOptionState, {"id": "pass", "kind": "pass"}))
    if selected < 0 and "pass" in lower:
        selected = pass_idx
    return actions, selected


def _attack_actions_and_targets(
    snapshot: dict[str, Any],
    observed: dict[str, Any],
    oracle: dict[str, OracleEntry],
) -> tuple[list[PendingOptionState], dict[int, int]]:
    attackers_text = str(observed.get("attackers_text") or "")
    actions: list[PendingOptionState] = []
    targets: dict[int, int] = {}
    for card in _battlefield(_self_player(snapshot)):
        if not _untapped(card) or not _is_creature(card, oracle):
            continue
        cid = _card_id(card)
        if not cid:
            continue
        option_index = len(actions)
        name = _card_name(card)
        actions.append(
            cast(
                PendingOptionState,
                {
                    "id": f"attack:{cid}",
                    "kind": "attacker",
                    "permanent_id": cid,
                    "card_id": cid,
                    "card_name": name,
                },
            )
        )
        targets[option_index] = 1 if _contains_card_name(attackers_text, card) else 0
    return actions, targets


def _block_actions_and_targets(
    snapshot: dict[str, Any],
    observed: dict[str, Any],
    oracle: dict[str, OracleEntry],
) -> tuple[list[PendingOptionState], dict[int, int]]:
    assignments = list(observed.get("assignments") or [])
    attackers = [
        card
        for card in _battlefield(_opp_player(snapshot))
        if _is_creature(card, oracle) and _card_id(card)
    ]
    actions: list[PendingOptionState] = []
    targets: dict[int, int] = {}
    for blocker in _battlefield(_self_player(snapshot)):
        if not _untapped(blocker) or not _is_creature(blocker, oracle):
            continue
        blocker_id = _card_id(blocker)
        if not blocker_id:
            continue
        valid_targets = [
            {"id": _card_id(attacker), "label": _card_name(attacker)} for attacker in attackers
        ]
        option_index = len(actions)
        name = _card_name(blocker)
        actions.append(
            cast(
                PendingOptionState,
                {
                    "id": f"block:{blocker_id}",
                    "kind": "block",
                    "permanent_id": blocker_id,
                    "card_id": blocker_id,
                    "card_name": name,
                    "valid_targets": valid_targets,
                },
            )
        )
        chosen_attacker_text = ""
        for assignment in assignments:
            if assignment.get("kind") != "block":
                continue
            if _contains_card_name(str(assignment.get("blockers_text") or ""), blocker):
                chosen_attacker_text = str(assignment.get("attacker_text") or "")
                break
        target_slot = 0
        for i, attacker in enumerate(attackers, start=1):
            if _contains_card_name(chosen_attacker_text, attacker):
                target_slot = i
                break
        targets[option_index] = target_slot
    return actions, targets


def _may_actions_and_targets(
    observed: dict[str, Any],
) -> tuple[list[PendingOptionState], dict[int, int]]:
    return [], {-1: 1 if bool(observed.get("accepted")) else 0}


def _choose_actions_and_targets(
    observed: dict[str, Any],
) -> tuple[list[PendingOptionState], dict[int, int]]:
    raw = str(observed.get("raw") or "observed")
    return [
        cast(
            PendingOptionState,
            {"id": raw, "kind": "permanent", "label": raw, "card_name": raw},
        )
    ], {-1: 0}


def _actions_and_targets(
    record: dict[str, Any],
    oracle: dict[str, OracleEntry],
) -> tuple[list[PendingOptionState], int, dict[int, int]]:
    choice = record.get("choice") or {}
    kind = str(choice.get("kind") or "")
    observed = cast(dict[str, Any], choice.get("observed") or {})
    snapshot = cast(dict[str, Any], (record.get("state") or {}).get("snapshot") or {})
    if kind == "priority":
        actions, selected = _priority_actions_and_target(snapshot, observed, oracle)
        return actions, selected, {}
    if kind == "attack":
        actions, targets = _attack_actions_and_targets(snapshot, observed, oracle)
        return actions, -1, targets
    if kind == "block":
        actions, targets = _block_actions_and_targets(snapshot, observed, oracle)
        return actions, -1, targets
    if kind == "may":
        actions, targets = _may_actions_and_targets(observed)
        return actions, -1, targets
    if kind == "choose":
        actions, targets = _choose_actions_and_targets(observed)
        return actions, -1, targets
    return [], -1, {}


def _with_pending(
    snapshot: dict[str, Any],
    kind: str,
    actions: list[PendingOptionState],
) -> dict[str, Any]:
    out = dict(snapshot)
    pending_kind = _CHOICE_KIND_TO_PENDING_KIND.get(kind, kind)
    out["pending"] = cast(
        PendingState,
        {
            "kind": pending_kind,
            "player_idx": 0,
            "options": actions,
        },
    )
    return out


def _render_token_ids(tokenizer: PreTrainedTokenizerFast) -> dict[str, int]:
    names = {
        "<chosen>",
        "<none>",
        "<yes>",
        "<no>",
        "<mulligan>",
        "<keep>",
        "<self>",
        "<opp>",
        *(f"<num:{i}>" for i in range(MAX_NUM)),
        *(f"<card-ref:{i}>" for i in range(MAX_CARD_REFS)),
    }
    out = {}
    for name in names:
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, list):
            raise TypeError(f"token {name!r} resolved to multiple ids")
        out[name] = int(tid)
    mana = ("{W}", "{U}", "{B}", "{R}", "{G}", "{C}")
    for token in mana:
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, list):
            raise TypeError(f"token {token!r} resolved to multiple ids")
        out[token] = int(tid)
    return out


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
        self.token_ids = _render_token_ids(tokenizer)
        self._spec_renderer: DecisionSpecRenderer | None = (
            DecisionSpecRenderer(tokenizer) if cfg.decoder else None
        )
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

    # ----------------------------------------------------------------- inline
    def _prepare(self, record: dict[str, Any]) -> _PreparedExample | None:
        actions, priority_option_index, per_blank_by_option = _actions_and_targets(
            record, self.oracle
        )
        choice = record.get("choice") or {}
        kind = str(choice.get("kind") or "")
        snapshot = cast(dict[str, Any], (record.get("state") or {}).get("snapshot") or {})
        snapshot = _with_pending(snapshot, kind, actions)
        try:
            rendered = render_snapshot(
                cast(GameStateSnapshot, snapshot),
                actions,
                oracle=self.oracle,
                chosen_token_id=self.token_ids["<chosen>"],
                none_token_id=self.token_ids["<none>"],
                yes_token_id=self.token_ids["<yes>"],
                no_token_id=self.token_ids["<no>"],
                mulligan_token_id=self.token_ids["<mulligan>"],
                keep_token_id=self.token_ids["<keep>"],
                self_token_id=self.token_ids["<self>"],
                opp_token_id=self.token_ids["<opp>"],
                num_token_ids=[self.token_ids[f"<num:{i}>"] for i in range(MAX_NUM)],
                mana_token_ids=[
                    self.token_ids[t] for t in ("{W}", "{U}", "{B}", "{R}", "{G}", "{C}")
                ],
                card_ref_token_ids=[
                    self.token_ids[f"<card-ref:{i}>"] for i in range(MAX_CARD_REFS)
                ],
            )
            encoded = tokenize_snapshot(rendered, self.tokenizer)
        except RenderError, RuntimeError, KeyError, TypeError, ValueError:
            return None

        priority_target_blank = -1
        per_blank_targets = [-1 for _ in rendered.blank_anchors]
        for anchor in rendered.blank_anchors:
            if priority_option_index >= 0 and anchor.option_index == priority_option_index:
                priority_target_blank = int(anchor.blank_index)
            per_blank_target = per_blank_by_option.get(int(anchor.option_index))
            if per_blank_target is not None and 0 <= int(anchor.blank_index) < len(
                per_blank_targets
            ):
                per_blank_targets[int(anchor.blank_index)] = int(per_blank_target)
        if priority_option_index >= 0 and priority_target_blank < 0:
            return None
        return _PreparedExample(
            encoded=encoded,
            priority_target_blank=priority_target_blank,
            per_blank_target_legal=per_blank_targets,
            value_target=_value_target(record, self.cfg),
        )

    # ----------------------------------------------------------------- decoder
    def _prepare_decoder(self, record: dict[str, Any]) -> _PreparedDecoderExample | None:
        if self._spec_renderer is None:
            raise RuntimeError("decoder pipeline requires DecisionSpecRenderer")
        actions, _priority_option_index, _ = _actions_and_targets(record, self.oracle)
        choice = record.get("choice") or {}
        kind = str(choice.get("kind") or "")
        observed = cast(dict[str, Any], choice.get("observed") or {})
        snapshot = cast(dict[str, Any], (record.get("state") or {}).get("snapshot") or {})
        snapshot = _with_pending(snapshot, kind, actions)

        pending = cast(PendingState, snapshot["pending"])
        decision_type = pending_decision_type(pending)
        if decision_type is None:
            return None
        target = translate_observed_to_target(pending, observed)
        if target is None or not target.output_token_ids:
            return None

        try:
            rendered = render_snapshot(
                cast(GameStateSnapshot, snapshot),
                actions,
                oracle=self.oracle,
                chosen_token_id=self.token_ids["<chosen>"],
                none_token_id=self.token_ids["<none>"],
                yes_token_id=self.token_ids["<yes>"],
                no_token_id=self.token_ids["<no>"],
                mulligan_token_id=self.token_ids["<mulligan>"],
                keep_token_id=self.token_ids["<keep>"],
                self_token_id=self.token_ids["<self>"],
                opp_token_id=self.token_ids["<opp>"],
                num_token_ids=[self.token_ids[f"<num:{i}>"] for i in range(MAX_NUM)],
                mana_token_ids=[
                    self.token_ids[t] for t in ("{W}", "{U}", "{B}", "{R}", "{G}", "{C}")
                ],
                card_ref_token_ids=[
                    self.token_ids[f"<card-ref:{i}>"] for i in range(MAX_CARD_REFS)
                ],
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
        # collator applies). One small dict per row; B is small.
        state_len = len(encoded.token_ids)
        anchor_pos_by_kind: dict[int, list[int]] = {}
        for anchor in spec.anchors:
            arr = anchor_pos_by_kind.setdefault(int(anchor.kind), [])
            # anchors of one kind appear in subject_index order in render_spec.
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

    # ----------------------------------------------------------------- batching
    def _batch_from_indices(self, indices: Sequence[int]) -> ForgeChoiceBatch | ForgeDecoderBatch:
        if self.cfg.decoder:
            return self._decoder_batch_from_indices(indices)
        return self._inline_batch_from_indices(indices)

    def _inline_batch_from_indices(self, indices: Sequence[int]) -> ForgeChoiceBatch:
        prepared: list[_PreparedExample] = []
        cursor = 0
        while len(prepared) < len(indices) and cursor < len(indices) * 4:
            item = self._prepare(self.records[int(indices[cursor % len(indices)])])
            cursor += 1
            if item is not None:
                prepared.append(item)
        if not prepared:
            raise ValueError("no renderable Forge choice examples in selected batch")
        encoded = collate([p.encoded for p in prepared], self.cfg.pad_token_id)
        if (
            self.cfg.max_tokens is not None
            and int(encoded.token_ids.shape[1]) > self.cfg.max_tokens
        ):
            encoded = _truncate_encoded_batch(encoded, max_tokens=self.cfg.max_tokens)
        batch_size = len(prepared)
        max_blanks = int(encoded.blank_positions.shape[1])
        per_blank = torch.full((batch_size, max_blanks), -1, dtype=torch.long)
        priority = torch.full((batch_size,), -1, dtype=torch.long)
        values = torch.empty((batch_size,), dtype=torch.float32)
        for row, item in enumerate(prepared):
            priority[row] = int(item.priority_target_blank)
            values[row] = float(item.value_target)
            n = min(max_blanks, len(item.per_blank_target_legal))
            if n:
                per_blank[row, :n] = torch.tensor(item.per_blank_target_legal[:n], dtype=torch.long)
        return ForgeChoiceBatch(
            encoded=encoded,
            priority_target_blank=priority,
            per_blank_target_legal=per_blank,
            value_targets=values,
        )

    def _decoder_batch_from_indices(self, indices: Sequence[int]) -> ForgeDecoderBatch:
        prepared: list[_PreparedDecoderExample] = []
        cursor = 0
        while len(prepared) < len(indices) and cursor < len(indices) * 4:
            item = self._prepare_decoder(self.records[int(indices[cursor % len(indices)])])
            cursor += 1
            if item is not None:
                prepared.append(item)
        if not prepared:
            raise ValueError("no renderable Forge decoder examples in selected batch")

        # Combined-stream collate (state + spec).
        encoded = collate_with_specs(
            [p.encoded for p in prepared],
            [p.spec for p in prepared],
            pad_id=self.cfg.pad_token_id,
        )

        batch_size = len(prepared)
        # Output sequences are typically short (<= ~16 tokens). Pad to
        # the per-batch max.
        L = max(len(p.target.output_token_ids) for p in prepared)
        T_enc = int(encoded.token_ids.shape[1])

        out_tokens = torch.zeros((batch_size, L), dtype=torch.long)
        out_pointer_pos = torch.full((batch_size, L), 0, dtype=torch.long)
        out_is_pointer = torch.zeros((batch_size, L), dtype=torch.bool)
        out_pad_mask = torch.zeros((batch_size, L), dtype=torch.bool)
        decision_type_per_row = torch.empty((batch_size,), dtype=torch.long)
        values = torch.empty((batch_size,), dtype=torch.float32)

        # vocab/pointer masks built per row by walking the grammar.
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
                    # Map subject_index to encoder position via the
                    # appropriate AnchorKind for this grammar step.
                    kind = _expected_pointer_kind(item.spec, item.target, i)
                    positions = item.anchor_pos_by_kind.get(int(kind), [])
                    if 0 <= subj < len(positions) and positions[subj] >= 0:
                        out_pointer_pos[b, i] = positions[subj]
                    else:
                        # Defensive: anchor missing. Mark step invalid.
                        out_pad_mask[b, i] = False

        # Build per-step legality masks by walking the grammar once per row.
        # batch_next_mask wants padded prefix arrays — we call it L times.
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
        for step in range(L):
            v_mask, _ptr_mask_subj = batch_next_mask(
                specs,
                prefix_tokens_np,
                prefix_subjects_np,
                prefix_lens_np,
            )
            vocab_mask[:, step, :] = torch.from_numpy(v_mask)
            # Translate the per-anchor pointer mask into encoder-position
            # space via each row's anchor_pos_by_kind. The expected anchor
            # kind at this step is implied by the grammar.
            for b, item in enumerate(prepared):
                if step >= len(item.target.output_token_ids):
                    continue
                if not item.target.output_is_pointer[step]:
                    continue
                kind = _expected_pointer_kind(item.spec, item.target, step)
                positions = item.anchor_pos_by_kind.get(int(kind), [])
                # Spec-side anchors of this kind are all legal at the
                # subject level; cross-subject constraints (uniqueness,
                # block legal-edge) live in the grammar mask itself,
                # which we mirror by walking the grammar above. For the
                # loss-side mask in encoder-position space we simply
                # allow all anchors of the expected kind whose position
                # is set; the actual uniqueness/edge constraint is
                # already encoded in the supervision target.
                for pos in positions:
                    if 0 <= pos < T_enc:
                        pointer_mask[b, step, pos] = True
            # advance the prefix lengths.
            for b, item in enumerate(prepared):
                if step < len(item.target.output_token_ids):
                    prefix_lens_np[b] = step + 1

        # Truncate the combined stream if needed.
        if (
            self.cfg.max_tokens is not None
            and int(encoded.token_ids.shape[1]) > self.cfg.max_tokens
        ):
            # Truncation invalidates pointer positions beyond the cap;
            # mark those steps as padding so they don't contribute loss.
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

    def iter_epoch(
        self, batch_size: int, rng: np.random.Generator
    ) -> Iterator[ForgeChoiceBatch | ForgeDecoderBatch]:
        order = rng.permutation(len(self.records))
        end = (len(order) // batch_size) * batch_size
        for off in range(0, end, batch_size):
            yield self._batch_from_indices([int(i) for i in order[off : off + batch_size]])

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> ForgeChoiceBatch | ForgeDecoderBatch:
        indices = rng.integers(0, len(self.records), size=batch_size)
        return self._batch_from_indices([int(i) for i in indices])


def _expected_pointer_kind(
    spec: DecisionSpec, target: DecoderTarget, step_index: int
) -> AnchorKind:
    """Anchor kind expected at ``target.output[step_index]`` (a pointer step).

    Mirrors the per-decision-type grammar in
    :mod:`magic_ai.text_encoder.grammar`. Used by collate to look up the
    encoder position for the supervised pointer target.
    """

    dt = DecisionType(target.decision_type)
    if dt is DecisionType.PRIORITY:
        return AnchorKind.LEGAL_ACTION
    if dt is DecisionType.CHOOSE_TARGETS:
        return AnchorKind.LEGAL_TARGET
    if dt is DecisionType.DECLARE_ATTACKERS:
        # Pattern: OPEN [ATTACK ptr-attacker DEFENDER ptr-defender]+ END
        # body offset = step_index - 1; mod 4 → 1 = attacker, 3 = defender.
        body_off = step_index - 1
        return AnchorKind.LEGAL_ATTACKER if body_off % 4 == 1 else AnchorKind.DEFENDER
    if dt is DecisionType.DECLARE_BLOCKERS:
        # Pattern: OPEN [BLOCK ptr-blocker ATTACKER ptr-attacker]+ END
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

    # ------------------------------------------------------------------ inline
    def step(
        self, batch: ForgeChoiceBatch | ForgeDecoderBatch, *, compute_stats: bool = True
    ) -> dict[str, float]:
        if isinstance(batch, ForgeDecoderBatch):
            return self._decoder_step(batch, compute_stats=compute_stats)
        return self._inline_step(batch, compute_stats=compute_stats)

    def _inline_step(self, batch: ForgeChoiceBatch, *, compute_stats: bool) -> dict[str, float]:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        out, _ = self.policy(batch.encoded, h_in=None, c_in=None)
        if out.blank_logits is None:
            raise RuntimeError("policy/value pretrain requires inline blank logits")
        priority_loss = inline_blank_priority_loss(
            out.blank_logits,
            batch.encoded.blank_group.to(out.blank_logits.device),
            batch.encoded.blank_group_kind.to(out.blank_logits.device),
            batch.encoded.blank_legal_mask.to(out.blank_logits.device),
            batch.priority_target_blank.to(out.blank_logits.device),
        )
        per_blank_loss = inline_blank_per_blank_loss(
            out.blank_logits,
            batch.encoded.blank_group_kind.to(out.blank_logits.device),
            batch.encoded.blank_legal_mask.to(out.blank_logits.device),
            batch.per_blank_target_legal.to(out.blank_logits.device),
        )
        policy_loss = priority_loss + per_blank_loss
        v_loss = value_loss(out.values.float(), batch.value_targets.to(out.values.device))
        loss = self.cfg.policy_loss_weight * policy_loss + self.cfg.value_loss_weight * v_loss
        loss.backward()
        if self.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)
        self.optimizer.step()
        if not compute_stats:
            return {}
        with torch.no_grad():
            pri_acc = inline_blank_priority_accuracy(
                out.blank_logits,
                batch.encoded.blank_group.to(out.blank_logits.device),
                batch.encoded.blank_group_kind.to(out.blank_logits.device),
                batch.encoded.blank_legal_mask.to(out.blank_logits.device),
                batch.priority_target_blank.to(out.blank_logits.device),
            )
            per_acc = inline_blank_per_blank_accuracy(
                out.blank_logits,
                batch.encoded.blank_group_kind.to(out.blank_logits.device),
                batch.encoded.blank_legal_mask.to(out.blank_logits.device),
                batch.per_blank_target_legal.to(out.blank_logits.device),
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
            "policy_loss": float(policy_loss.detach()),
            "priority_loss": float(priority_loss.detach()),
            "per_blank_loss": float(per_blank_loss.detach()),
            "value_loss": float(v_loss.detach()),
            "priority_accuracy": float(pri_acc["accuracy"]),
            "per_blank_accuracy": float(per_acc["accuracy"]),
            "value_sign_accuracy": float(sign_acc.detach()),
            "grad_norm": float(grad_norm.detach()),
        }

    def _decoder_step(self, batch: ForgeDecoderBatch, *, compute_stats: bool) -> dict[str, float]:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Run the encoder through the recurrent policy to get value head;
        # then run the grammar decoder teacher-forced via the underlying
        # TextPolicy.
        text_policy = self.policy.text_policy
        if text_policy.grammar_decoder is None:
            raise RuntimeError("decoder pipeline requires TextPolicy(use_grammar_decoder=True)")
        device = next(text_policy.parameters()).device
        encoded = batch.encoded
        target_tokens = batch.output_token_ids.to(device)
        vocab_logits, pointer_logits = text_policy.forward_decoder_teacher_forced(
            encoded, target_tokens
        )
        # Run the value head off the encoder CLS pool from the recurrent
        # policy too, keeping the value training path stable. The
        # recurrent policy's forward does pack_batch internally; reuse it
        # by calling its forward directly.
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
            # Combat full-sequence exact-match: a row is correct if every
            # supervised step is correct.
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
            if isinstance(batch, ForgeDecoderBatch):
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
            else:
                out, _ = self.policy(batch.encoded, h_in=None, c_in=None)
                if out.blank_logits is None:
                    continue
                priority_loss = inline_blank_priority_loss(
                    out.blank_logits,
                    batch.encoded.blank_group,
                    batch.encoded.blank_group_kind,
                    batch.encoded.blank_legal_mask,
                    batch.priority_target_blank,
                )
                per_blank_loss = inline_blank_per_blank_loss(
                    out.blank_logits,
                    batch.encoded.blank_group_kind,
                    batch.encoded.blank_legal_mask,
                    batch.per_blank_target_legal,
                )
                v_loss = value_loss(out.values.float(), batch.value_targets)
                stats = {
                    "eval_policy_loss": float((priority_loss + per_blank_loss).detach()),
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
        blank_positions=batch.blank_positions.to(device),
        blank_kind=batch.blank_kind.to(device),
        blank_group=batch.blank_group.to(device),
        blank_group_kind=batch.blank_group_kind.to(device),
        blank_option_index=batch.blank_option_index.to(device),
        blank_legal_ids=batch.blank_legal_ids.to(device),
        blank_legal_mask=batch.blank_legal_mask.to(device),
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
    blank_positions = batch.blank_positions.clone()
    blank_positions[blank_positions >= max_tokens] = -1
    blank_legal_mask = batch.blank_legal_mask.clone()
    if blank_positions.numel() > 0:
        blank_legal_mask = blank_legal_mask & (blank_positions.unsqueeze(-1) >= 0)
    return TextEncodedBatch(
        token_ids=batch.token_ids[:, :max_tokens].contiguous(),
        attention_mask=attention_mask,
        card_ref_positions=card_ref_positions,
        seq_lengths=seq_lengths,
        total_tokens=int(seq_lengths.sum().item()),
        seq_lengths_host=tuple(int(v) for v in seq_lengths.tolist()),
        blank_positions=blank_positions,
        blank_kind=batch.blank_kind,
        blank_group=batch.blank_group,
        blank_group_kind=batch.blank_group_kind,
        blank_option_index=batch.blank_option_index,
        blank_legal_ids=batch.blank_legal_ids,
        blank_legal_mask=blank_legal_mask,
        spec_tokens=batch.spec_tokens,
        spec_lens=batch.spec_lens,
        decision_type=batch.decision_type,
        pointer_anchor_positions=batch.pointer_anchor_positions,
        pointer_anchor_kinds=batch.pointer_anchor_kinds,
        pointer_anchor_subjects=batch.pointer_anchor_subjects,
        pointer_anchor_handles=batch.pointer_anchor_handles,
        legal_edge_bitmap=batch.legal_edge_bitmap,
    )


def _batch_to_device(
    batch: ForgeChoiceBatch | ForgeDecoderBatch, device: torch.device
) -> ForgeChoiceBatch | ForgeDecoderBatch:
    if isinstance(batch, ForgeDecoderBatch):
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
    return ForgeChoiceBatch(
        encoded=_encoded_to_device(batch.encoded, device),
        priority_target_blank=batch.priority_target_blank.to(device),
        per_blank_target_legal=batch.per_blank_target_legal.to(device),
        value_targets=batch.value_targets.to(device),
    )


def batches_per_epoch(n_examples: int, batch_size: int) -> int:
    return int(math.floor(n_examples / batch_size))


__all__ = [
    "ForgeChoiceBatch",
    "ForgeChoiceDataset",
    "ForgeDecoderBatch",
    "ForgePolicyValueConfig",
    "ForgePolicyValueTrainer",
    "ValueTargetMode",
    "batches_per_epoch",
    "_batch_to_device",
]
