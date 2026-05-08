"""Policy + value pretraining on extracted Forge choice situations.

Consumes the gzip JSONL artifact produced by
``scripts/extract_forge_choice_situations.py``. Each record stores a pre-choice
snapshot, the action text Forge actually took, and the terminal outcome. This
loader reconstructs a conservative inline-blank legal set from the visible
state, maps the observed choice onto the corresponding blank target, and
trains both the inline policy scorer and value head.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from magic_ai.game_state import GameStateSnapshot, PendingOptionState, PendingState
from magic_ai.text_encoder.batch import TextEncodedBatch, collate, tokenize_snapshot
from magic_ai.text_encoder.recurrent import RecurrentTextPolicy
from magic_ai.text_encoder.render import OracleEntry, RenderError, render_snapshot
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS, MAX_NUM
from magic_ai.text_encoder.training import (
    inline_blank_per_blank_accuracy,
    inline_blank_per_blank_loss,
    inline_blank_priority_accuracy,
    inline_blank_priority_loss,
    value_loss,
)

ChoiceKind = Literal["priority", "attack", "block", "may", "choose"]
ValueTargetMode = Literal["terminal", "gae", "vtrace"]


@dataclass(frozen=True)
class ForgePolicyValueConfig:
    data_path: Path
    batch_size: int
    eval_fraction: float = 0.05
    gamma: float = 1.0
    value_target_mode: ValueTargetMode = "terminal"
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    pad_token_id: int = 0


@dataclass(frozen=True)
class ForgeChoiceBatch:
    encoded: TextEncodedBatch
    priority_target_blank: Tensor
    per_blank_target_legal: Tensor
    value_targets: Tensor


@dataclass(frozen=True)
class _PreparedExample:
    encoded: Any
    priority_target_blank: int
    per_blank_target_legal: list[int]
    value_target: float


def _stable_eval_bucket(game_id: str, bucket: int) -> int:
    digest = hashlib.blake2b(game_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % bucket


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    paths = sorted(path.rglob("*.jsonl.gz")) if path.is_dir() else [path]
    for item in paths:
        opener = gzip.open if item.suffix == ".gz" else open
        with opener(item, "rt", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)


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
        actions.append(
            cast(
                PendingOptionState,
                {
                    "id": f"attack:{cid}",
                    "kind": "attacker",
                    "permanent_id": cid,
                    "card_id": cid,
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
        actions.append(
            cast(
                PendingOptionState,
                {
                    "id": f"block:{blocker_id}",
                    "kind": "block",
                    "permanent_id": blocker_id,
                    "card_id": blocker_id,
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
    # The Forge logs do not expose the legal alternatives for generic choices.
    # Keep a single observed placeholder so value training still uses the row
    # and policy loss has a well-defined, zero-entropy supervised target.
    raw = str(observed.get("raw") or "observed")
    return [cast(PendingOptionState, {"id": raw, "kind": "choice", "label": raw})], {-1: 0}


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
    pending_kind = {
        "priority": "priority",
        "attack": "attack",
        "block": "block",
        "may": "may",
        "choose": "mode",
    }.get(kind, kind)
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
    # With one extracted row per game and no behavior-policy trace, both GAE
    # and v-trace reduce to a discounted Monte Carlo terminal sign. The runtime
    # flag exists so callers can preserve the intended target semantics.
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

    def _batch_from_indices(self, indices: Sequence[int]) -> ForgeChoiceBatch:
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

    def iter_epoch(self, batch_size: int, rng: np.random.Generator) -> Iterator[ForgeChoiceBatch]:
        order = rng.permutation(len(self.records))
        end = (len(order) // batch_size) * batch_size
        for off in range(0, end, batch_size):
            yield self._batch_from_indices([int(i) for i in order[off : off + batch_size]])

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> ForgeChoiceBatch:
        indices = rng.integers(0, len(self.records), size=batch_size)
        return self._batch_from_indices([int(i) for i in indices])


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

    def step(self, batch: ForgeChoiceBatch, *, compute_stats: bool = True) -> dict[str, float]:
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
    )


def _batch_to_device(batch: ForgeChoiceBatch, device: torch.device) -> ForgeChoiceBatch:
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
    "ForgePolicyValueConfig",
    "ForgePolicyValueTrainer",
    "ValueTargetMode",
    "batches_per_epoch",
    "_batch_to_device",
]
