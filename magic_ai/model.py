"""Actor-critic model for mage-go legal action spaces."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli
from torch.nn import functional as F

from magic_ai.actions import (
    COLORS,
    OPTION_SCALAR_DIM,
    TARGET_SCALAR_DIM,
    ActionOptionsEncoder,
    ActionRequest,
    LegalActionCandidate,
    ParsedActionBatch,
    ParsedActionInputs,
    PendingState,
    action_from_attackers,
    action_from_blockers,
    action_from_choice_accepted,
    action_from_choice_color,
    action_from_choice_ids,
    action_from_choice_index,
    action_from_priority_candidate,
    selected_option_id,
)
from magic_ai.buffer import NativeTrajectoryBuffer, RolloutBuffer
from magic_ai.game_state import (
    GAME_INFO_DIM,
    ZONE_SLOT_COUNT,
    GameStateEncoder,
    GameStateSnapshot,
    ParsedGameState,
    ParsedGameStateBatch,
)
from magic_ai.lstm_recompute import (
    LstmRecomputeStrategy,
    lstm_recompute_per_step_h_out,
    lstm_recompute_per_step_states,
)
from magic_ai.native_encoder import NativeEncodedBatch
from magic_ai.replay_decisions import (
    ReplayPerChoice,
    ReplayScoringForward,
    decision_logits_reference,
    flat_decision_distribution_from_forward,
    flat_decision_distribution_impl,
    score_may_decisions_from_forward,
    validate_decision_indices,
    validate_flat_scored_indices,
)

TraceKind = Literal[
    "priority",
    "attackers",
    "blockers",
    "choice_index",
    "choice_ids",
    "choice_color",
    "may",
]

TRACE_KIND_VALUES: tuple[TraceKind, ...] = (
    "priority",
    "attackers",
    "blockers",
    "choice_index",
    "choice_ids",
    "choice_color",
    "may",
)
TRACE_KIND_TO_ID: dict[TraceKind, int] = {
    trace_kind: idx for idx, trace_kind in enumerate(TRACE_KIND_VALUES)
}


@dataclass(frozen=True)
class ActionTrace:
    """Enough information to recompute a sampled action's log-probability."""

    kind: TraceKind
    indices: tuple[int, ...] = ()
    binary: tuple[float, ...] = ()


@dataclass(frozen=True)
class ParsedStep:
    """Pure-Python parsed policy inputs for one step.

    Holds the parsed game-state / action-options plus the decision-row layout.
    No tensors; ``RolloutBuffer.ingest_batch`` does the bulk CPU→GPU copy.
    """

    parsed_state: ParsedGameState
    parsed_action: ParsedActionInputs
    trace_kind: TraceKind
    trace_kind_id: int
    decision_option_idx: list[list[int]]  # [G, C]
    decision_target_idx: list[list[int]]  # [G, C]
    decision_mask: list[list[bool]]  # [G, C]
    uses_none_head: list[bool]  # [G]
    pending: PendingState


@dataclass(frozen=True)
class ParsedBatch:
    """Batched parsed policy inputs for one actor forward."""

    parsed_state: ParsedGameStateBatch
    parsed_action: ParsedActionBatch
    trace_kinds: list[TraceKind]
    trace_kind_ids: Tensor  # [N]
    pendings: list[PendingState]
    decision_option_idx: Tensor  # [total_groups, max_cached_choices]
    decision_target_idx: Tensor  # [total_groups, max_cached_choices]
    decision_mask: Tensor  # [total_groups, max_cached_choices]
    uses_none_head: Tensor  # [total_groups]
    decision_starts: list[int]
    decision_counts: list[int]


class PolicyStep(NamedTuple):
    # NamedTuple instead of @dataclass: ~3-5x faster to construct, which
    # matters because the rollout sampling path builds ~80 of these per
    # poll (~5k per profiled iter).
    action: ActionRequest
    trace: ActionTrace
    log_prob: Tensor
    value: Tensor
    entropy: Tensor
    replay_idx: int | None = None
    selected_choice_cols: tuple[int, ...] = ()
    may_selected: int = 0


class PPOPolicy(nn.Module):
    """Actor-critic network that scores legal mage-go action options."""

    def __init__(
        self,
        game_state_encoder: GameStateEncoder,
        *,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
        max_options: int = 64,
        max_targets_per_option: int = 4,
        rollout_capacity: int = 4096,
        decision_capacity: int | None = None,
        use_lstm: bool = False,
        spr_enabled: bool = False,
        spr_action_dim: int = 32,
        spr_ema_decay: float = 0.99,
        spr_k: int = 5,
        spr_proj_dim: int = 256,
        validate: bool = True,
        compile_forward: bool = False,
    ) -> None:
        super().__init__()
        self.game_state_encoder = game_state_encoder
        self.validate = validate
        self.compile_forward = compile_forward
        self.action_encoder = ActionOptionsEncoder(
            game_state_encoder,
            max_options=max_options,
            max_targets_per_option=max_targets_per_option,
        )
        self.max_options = max_options
        self.max_targets_per_option = max_targets_per_option
        self.max_cached_choices = max(max_options, max_options * max(1, max_targets_per_option))

        input_dim = game_state_encoder.output_dim + game_state_encoder.d_model * 2
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        self.use_lstm = use_lstm
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        if use_lstm:
            self.feature_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=hidden_layers,
                batch_first=True,
            )
            self.register_buffer(
                "live_lstm_h",
                torch.zeros(hidden_layers, 0, hidden_dim),
                persistent=False,
            )
            self.register_buffer(
                "live_lstm_c",
                torch.zeros(hidden_layers, 0, hidden_dim),
                persistent=False,
            )
        else:
            trunk_layers: list[nn.Module] = []
            in_dim = input_dim
            for _ in range(hidden_layers):
                trunk_layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU()))
                in_dim = hidden_dim
            self.trunk = nn.Sequential(*trunk_layers)
        self.action_query = nn.Linear(hidden_dim, game_state_encoder.d_model)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.none_blocker_head = nn.Linear(hidden_dim, 1)
        self.may_head = nn.Linear(hidden_dim, 1)

        self.spr_enabled = spr_enabled
        self.spr_ema_decay = spr_ema_decay
        self.spr_k = spr_k
        self.spr_proj_dim = spr_proj_dim
        if spr_enabled:
            if not use_lstm:
                raise ValueError("SPR auxiliary loss currently requires use_lstm=True")
            if spr_k < 1:
                raise ValueError("spr_k must be >= 1")
            self.target_game_state_encoder = copy.deepcopy(game_state_encoder)
            self.target_action_encoder = copy.deepcopy(self.action_encoder)
            self.target_feature_projection = copy.deepcopy(self.feature_projection)
            self.target_lstm = copy.deepcopy(self.lstm)
            for module in (
                self.target_game_state_encoder,
                self.target_action_encoder,
                self.target_feature_projection,
                self.target_lstm,
            ):
                for p in module.parameters():
                    p.requires_grad_(False)
            self.spr_action_embedding = nn.Embedding(len(TRACE_KIND_VALUES), spr_action_dim)
            d_model = game_state_encoder.d_model
            self.spr_action_projector = nn.Sequential(
                nn.Linear(2 * d_model + 1, spr_action_dim),
                nn.GELU(),
            )
            # Transition model h: (z_t, action_emb_t) -> z_{t+1}
            self.spr_transition = nn.Sequential(
                nn.Linear(hidden_dim + 2 * spr_action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # BYOL-style projection heads g_o (online) and g_m (EMA target)
            self.spr_g_online = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, spr_proj_dim),
            )
            self.spr_g_target = copy.deepcopy(self.spr_g_online)
            for p in self.spr_g_target.parameters():
                p.requires_grad_(False)
            # Predictor q on top of online projection
            self.spr_q = nn.Linear(spr_proj_dim, spr_proj_dim)

        if decision_capacity is None:
            decision_capacity = rollout_capacity * 8

        self.rollout_buffer = RolloutBuffer(
            capacity=rollout_capacity,
            decision_capacity=decision_capacity,
            max_options=max_options,
            max_targets_per_option=max_targets_per_option,
            max_cached_choices=self.max_cached_choices,
            zone_slot_count=ZONE_SLOT_COUNT,
            game_info_dim=GAME_INFO_DIM,
            option_scalar_dim=OPTION_SCALAR_DIM,
            target_scalar_dim=TARGET_SCALAR_DIM,
            recurrent_layers=hidden_layers if use_lstm else 0,
            recurrent_hidden_dim=hidden_dim if use_lstm else 0,
        )
        self._compiled_compute_forward_impl: Callable[..., Any] | None = None
        self._compiled_compute_hidden_target_impl: Callable[..., Any] | None = None
        self._compiled_flat_decision_distribution_impl: Callable[..., Any] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_rollout_buffer(self) -> None:
        self.rollout_buffer.reset()

    def init_lstm_env_states(self, num_envs: int) -> None:
        if not self.use_lstm:
            return
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        device = self.device
        self.live_lstm_h = torch.zeros(
            self.hidden_layers,
            num_envs,
            self.hidden_dim,
            dtype=torch.float32,
            device=device,
        )
        self.live_lstm_c = torch.zeros_like(self.live_lstm_h)

    def reset_lstm_env_states(self, env_indices: list[int]) -> None:
        if not self.use_lstm:
            return
        if self.live_lstm_h.shape[1] == 0:
            raise RuntimeError("LSTM env states have not been initialized")
        idx_t = torch.tensor(env_indices, dtype=torch.long, device=self.device)
        self.live_lstm_h[:, idx_t] = 0
        self.live_lstm_c[:, idx_t] = 0

    def lstm_env_state_inputs(self, env_indices: list[int]) -> tuple[Tensor, Tensor] | None:
        if not self.use_lstm:
            return None
        if self.live_lstm_h.shape[1] == 0:
            raise RuntimeError("LSTM env states have not been initialized")
        idx_t = torch.tensor(env_indices, dtype=torch.long, device=self.device)
        return (
            self.live_lstm_h[:, idx_t].permute(1, 0, 2).contiguous().detach(),
            self.live_lstm_c[:, idx_t].permute(1, 0, 2).contiguous().detach(),
        )

    def release_replay_rows(self, replay_rows: list[int]) -> None:
        return

    def append_staged_episode_to_rollout(
        self,
        staging: NativeTrajectoryBuffer,
        env_idx: int,
    ) -> list[int]:
        return self.append_staged_episodes_to_rollout(staging, [env_idx])[0]

    def append_staged_episodes_to_rollout(
        self,
        staging: NativeTrajectoryBuffer,
        env_indices: list[int],
    ) -> list[list[int]]:
        active_envs = [env_idx for env_idx in env_indices if staging.active_step_count(env_idx) > 0]
        if not active_envs:
            return [[] for _ in env_indices]

        step_counts = [staging.active_step_count(env_idx) for env_idx in active_envs]
        write = self.rollout_buffer.ingest_staged_episodes(staging, active_envs)
        flat_rows = [int(row) for row in write.step_indices.detach().cpu().tolist()]

        grouped_rows: dict[int, list[int]] = {}
        cursor = 0
        for env_idx, count in zip(active_envs, step_counts, strict=True):
            grouped_rows[env_idx] = flat_rows[cursor : cursor + count]
            cursor += count
        return [grouped_rows.get(env_idx, []) for env_idx in env_indices]

    def parse_inputs(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None = None,
    ) -> ParsedStep:
        """Parse a state + pending request into pure-Python index lists.

        No GPU allocations — ``act_batch`` does the bulk ingest into the
        rollout buffer in one pass per field.
        """

        parsed_state = self.game_state_encoder.parse_state(state, perspective_player_idx)
        player_idx = self.game_state_encoder._resolve_perspective_player_idx(
            state, perspective_player_idx
        )
        parsed_action = self.action_encoder.parse_pending(
            state,
            pending,
            perspective_player_idx=player_idx,
            card_id_to_slot=parsed_state.card_id_to_slot,
        )
        trace_kind = _trace_kind_for_pending(pending)
        option_idx, target_idx, mask, uses_none = self._build_decision_layout(
            trace_kind, pending, parsed_action
        )
        return ParsedStep(
            parsed_state=parsed_state,
            parsed_action=parsed_action,
            trace_kind=trace_kind,
            trace_kind_id=TRACE_KIND_TO_ID[trace_kind],
            decision_option_idx=option_idx,
            decision_target_idx=target_idx,
            decision_mask=mask,
            uses_none_head=uses_none,
            pending=pending,
        )

    def parse_inputs_batch(
        self,
        states: list[GameStateSnapshot],
        pendings: list[PendingState],
        *,
        perspective_player_indices: list[int | None],
    ) -> ParsedBatch:
        parsed_state = self.game_state_encoder.parse_state_batch(
            states,
            perspective_player_indices,
        )
        resolved_player_indices = [
            self.game_state_encoder._resolve_perspective_player_idx(state, perspective_player_idx)
            for state, perspective_player_idx in zip(
                states, perspective_player_indices, strict=True
            )
        ]
        parsed_action = self.action_encoder.parse_pending_batch(
            states,
            pendings,
            perspective_player_indices=resolved_player_indices,
            card_id_to_slots=parsed_state.card_id_to_slots,
        )

        trace_kinds = [_trace_kind_for_pending(pending) for pending in pendings]
        trace_kind_ids = torch.tensor(
            [TRACE_KIND_TO_ID[trace_kind] for trace_kind in trace_kinds],
            dtype=torch.long,
        )

        flat_option_idx: list[list[int]] = []
        flat_target_idx: list[list[int]] = []
        flat_mask: list[list[bool]] = []
        flat_uses_none: list[bool] = []
        decision_starts: list[int] = []
        decision_counts: list[int] = []
        cursor = 0

        for step_idx, (trace_kind, pending) in enumerate(zip(trace_kinds, pendings, strict=True)):
            option_idx, target_idx, mask, uses_none = self._build_decision_layout_batch_step(
                trace_kind,
                pending,
                num_present_options=int(parsed_action.num_present_options[step_idx].item()),
                target_mask=parsed_action.target_mask[step_idx],
                priority_candidates=parsed_action.priority_candidates[step_idx],
            )
            decision_starts.append(cursor)
            decision_counts.append(len(option_idx))
            cursor += len(option_idx)
            flat_option_idx.extend(option_idx)
            flat_target_idx.extend(target_idx)
            flat_mask.extend(mask)
            flat_uses_none.extend(uses_none)

        if flat_option_idx:
            decision_option_idx = torch.tensor(flat_option_idx, dtype=torch.long)
            decision_target_idx = torch.tensor(flat_target_idx, dtype=torch.long)
            decision_mask = torch.tensor(flat_mask, dtype=torch.bool)
            uses_none_head = torch.tensor(flat_uses_none, dtype=torch.bool)
        else:
            decision_option_idx = torch.zeros((0, self.max_cached_choices), dtype=torch.long)
            decision_target_idx = torch.zeros((0, self.max_cached_choices), dtype=torch.long)
            decision_mask = torch.zeros((0, self.max_cached_choices), dtype=torch.bool)
            uses_none_head = torch.zeros((0,), dtype=torch.bool)

        return ParsedBatch(
            parsed_state=parsed_state,
            parsed_action=parsed_action,
            trace_kinds=trace_kinds,
            trace_kind_ids=trace_kind_ids,
            pendings=pendings,
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )

    def _build_decision_layout_batch_step(
        self,
        trace_kind: TraceKind,
        pending: PendingState,
        *,
        num_present_options: int,
        target_mask: Tensor,
        priority_candidates: list[LegalActionCandidate],
    ) -> tuple[list[list[int]], list[list[int]], list[list[bool]], list[bool]]:
        choices = self.max_cached_choices

        if trace_kind == "may":
            return [], [], [], []

        if trace_kind == "priority":
            candidates = priority_candidates[:choices]
            if not candidates:
                return [], [], [], []
            option_row = [-1] * choices
            target_row = [-1] * choices
            mask_row = [False] * choices
            for col, cand in enumerate(candidates):
                option_row[col] = cand.option_index
                if cand.target_index is not None:
                    target_row[col] = cand.target_index
                mask_row[col] = True
            return [option_row], [target_row], [mask_row], [False]

        option_count = min(num_present_options, self.max_options)

        if trace_kind == "attackers":
            if option_count == 0:
                return [], [], [], []
            option_idx_l: list[list[int]] = []
            target_idx_l: list[list[int]] = []
            mask_l: list[list[bool]] = []
            for i in range(option_count):
                option_row = [-1] * choices
                option_row[1] = i
                target_row = [-1] * choices
                mask_row = [False] * choices
                mask_row[0] = True
                mask_row[1] = True
                option_idx_l.append(option_row)
                target_idx_l.append(target_row)
                mask_l.append(mask_row)
            return option_idx_l, target_idx_l, mask_l, [True] * option_count

        if trace_kind == "blockers":
            if option_count == 0:
                return [], [], [], []
            option_idx_l = []
            target_idx_l = []
            mask_l = []
            target_counts = target_mask[:option_count].count_nonzero(dim=-1).detach().cpu().tolist()
            for i, target_count in enumerate(target_counts):
                option_row = [-1] * choices
                target_row = [-1] * choices
                mask_row = [False] * choices
                mask_row[0] = True
                for t in range(int(target_count)):
                    col = t + 1
                    option_row[col] = i
                    target_row[col] = t
                    mask_row[col] = True
                option_idx_l.append(option_row)
                target_idx_l.append(target_row)
                mask_l.append(mask_row)
            return option_idx_l, target_idx_l, mask_l, [True] * option_count

        if option_count == 0:
            return [], [], [], []
        option_row = [-1] * choices
        target_row = [-1] * choices
        mask_row = [False] * choices
        for i in range(option_count):
            option_row[i] = i
            mask_row[i] = True
        return [option_row], [target_row], [mask_row], [False]

    def act(
        self,
        state: GameStateSnapshot,
        pending: PendingState,
        *,
        perspective_player_idx: int | None = None,
        deterministic: bool = False,
    ) -> PolicyStep:
        parsed = self.parse_inputs(
            state,
            pending,
            perspective_player_idx=perspective_player_idx,
        )
        return self.act_batch([parsed], deterministic=deterministic)[0]

    def act_parsed_batch(
        self,
        parsed_batch: ParsedBatch | NativeEncodedBatch,
        *,
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        return self.act_batch(parsed_batch, deterministic=deterministic)

    def sample_native_batch(
        self,
        native_batch: NativeEncodedBatch,
        *,
        env_indices: list[int] | None = None,
        deterministic: bool = False,
        finetune_eps: float = 0.0,
        finetune_n_disc: int = 0,
    ) -> list[PolicyStep]:
        n = int(native_batch.trace_kind_id.shape[0])
        if n == 0:
            return []

        device = self.device
        forward = self._forward_native_batch(native_batch, env_indices=env_indices)
        # native_batch fields are CPU-side from the encoder; .tolist() is the
        # single host transfer (detach()/cpu() are no-ops for grad-free CPU
        # tensors and were costing ~5% of this function in line_profiler).
        may_positions = native_batch.may_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        decision_starts = native_batch.decision_start.tolist()
        decision_counts = native_batch.decision_count.tolist()
        decision_option_idx = native_batch.decision_option_idx.to(device)
        decision_target_idx = native_batch.decision_target_idx.to(device)
        # PyTorch indexing / nonzero / & accept uint8 masks the same way as
        # bool, so skip the dtype conversion (was 1.8% of sample_native_batch
        # by itself). Move to device only.
        decision_mask = native_batch.decision_mask.to(device)
        uses_none_head = native_batch.uses_none_head.to(device)
        group_step_positions: list[int] = []
        group_decision_indices: list[int] = []
        trace_kind_ids = native_batch.trace_kind_id.tolist()
        for step_idx, trace_kind_id in enumerate(trace_kind_ids):
            if trace_kind_id == TRACE_KIND_TO_ID["may"]:
                continue
            count = decision_counts[step_idx]
            start = decision_starts[step_idx]
            for k in range(count):
                group_step_positions.append(step_idx)
                group_decision_indices.append(start + k)

        may_log_probs: Tensor | None = None
        may_entropies: Tensor | None = None
        may_selected_cpu: list[float] = []
        if may_positions:
            may_pos_t = torch.tensor(may_positions, dtype=torch.long, device=device)
            may_logits = forward.may_logits[may_pos_t]
            may_dist = Bernoulli(logits=may_logits)
            if deterministic:
                may_sel = (may_logits >= 0).to(dtype=forward.values.dtype)
            else:
                may_sel = may_dist.sample()
            may_log_probs = may_dist.log_prob(may_sel)
            may_entropies = may_dist.entropy()
            may_selected_cpu = may_sel.detach().cpu().tolist()

        per_step_log_prob_sum: Tensor | None = None
        per_step_entropy_sum: Tensor | None = None
        decision_selected_cpu: list[int] = []
        if group_step_positions:
            pos_t = torch.tensor(group_step_positions, dtype=torch.long, device=device)
            idx_t = torch.tensor(group_decision_indices, dtype=torch.long, device=device)
            option_idx = decision_option_idx[idx_t]
            target_idx = decision_target_idx[idx_t]
            masks = decision_mask[idx_t]
            uses_none = uses_none_head[idx_t]
            group_idx, choice_cols, flat_logits, flat_log_probs, group_entropies = (
                self._flat_decision_distribution(
                    step_positions=pos_t,
                    option_idx=option_idx,
                    target_idx=target_idx,
                    masks=masks,
                    uses_none=uses_none,
                    option_vectors=forward.option_vectors,
                    target_vectors=forward.target_vectors,
                    query=cast(Tensor, forward.query),
                    none_logits=forward.none_logits,
                )
            )
            decision_selected, decision_log_probs = self._sample_flat_decisions(
                group_idx=group_idx,
                choice_cols=choice_cols,
                flat_logits=flat_logits,
                flat_log_probs=flat_log_probs,
                deterministic=deterministic,
                finetune_eps=finetune_eps,
                finetune_n_disc=finetune_n_disc,
            )
            per_step_log_prob_sum = torch.zeros(n, dtype=forward.values.dtype, device=device)
            per_step_entropy_sum = torch.zeros(n, dtype=forward.values.dtype, device=device)
            per_step_log_prob_sum.scatter_add_(0, pos_t, decision_log_probs)
            per_step_entropy_sum.scatter_add_(0, pos_t, group_entropies)
            decision_selected_cpu = decision_selected.detach().cpu().tolist()

        # Unbind once into Python lists so the per-env loop below indexes into
        # plain lists instead of issuing N aten::select calls per tensor field.
        values_list = list(forward.values.unbind(0))
        may_log_probs_list = list(may_log_probs.unbind(0)) if may_log_probs is not None else []
        may_entropies_list = list(may_entropies.unbind(0)) if may_entropies is not None else []
        per_step_log_prob_sum_list = (
            list(per_step_log_prob_sum.unbind(0)) if per_step_log_prob_sum is not None else []
        )
        per_step_entropy_sum_list = (
            list(per_step_entropy_sum.unbind(0)) if per_step_entropy_sum is not None else []
        )

        may_lookup = {step_idx: pos for pos, step_idx in enumerate(may_positions)}
        results: list[PolicyStep] = []
        offset = 0
        for step_idx, trace_kind_id in enumerate(trace_kind_ids):
            trace_kind = TRACE_KIND_VALUES[trace_kind_id]
            value = values_list[step_idx]
            decision_count = decision_counts[step_idx]

            if trace_kind == "may":
                pos = may_lookup[step_idx]
                sel_scalar = may_selected_cpu[pos]
                results.append(
                    PolicyStep(
                        action=action_from_choice_accepted(bool(sel_scalar >= 0.5)),
                        trace=ActionTrace("may", binary=(float(sel_scalar),)),
                        log_prob=may_log_probs_list[pos],
                        value=value,
                        entropy=may_entropies_list[pos],
                        replay_idx=None,
                        may_selected=int(sel_scalar >= 0.5),
                    )
                )
                continue

            if decision_count == 0:
                zero = torch.zeros((), device=device)
                results.append(
                    PolicyStep(
                        action=cast(ActionRequest, {}),
                        trace=ActionTrace(trace_kind, indices=(0,)),
                        log_prob=zero,
                        value=value,
                        entropy=zero,
                        replay_idx=None,
                    )
                )
                continue

            step_selected = decision_selected_cpu[offset : offset + decision_count]
            offset += decision_count
            trace, action = self._trace_action_without_pending(trace_kind, step_selected)
            results.append(
                PolicyStep(
                    action=action,
                    trace=trace,
                    log_prob=per_step_log_prob_sum_list[step_idx],
                    value=value,
                    entropy=per_step_entropy_sum_list[step_idx],
                    replay_idx=None,
                    selected_choice_cols=tuple(step_selected),
                )
            )
        return results

    def act_batch(
        self,
        parsed_steps: list[ParsedStep] | ParsedBatch | NativeEncodedBatch,
        *,
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        """Ingest parsed steps, run one forward, sample actions, write back.

        Sampling is batched: one ``Bernoulli`` over may-head logits and one
        ``Categorical`` over concatenated decision groups. Sampled values are
        written back to the buffer so ``evaluate_cached_batch`` can replay log
        probs from integer handles alone.
        """

        if isinstance(parsed_steps, NativeEncodedBatch):
            parsed_batch = self._parsed_batch_from_native(parsed_steps)
        elif isinstance(parsed_steps, ParsedBatch):
            parsed_batch = parsed_steps
        else:
            if not parsed_steps:
                return []
            parsed_batch = self._parsed_batch_from_steps(parsed_steps)

        n = int(parsed_batch.trace_kind_ids.shape[0])
        if n == 0:
            return []

        device = self.device
        rb = self.rollout_buffer
        write = rb.ingest_parsed_batch(parsed_batch)

        forward = self._forward_batch(write.step_indices)

        may_positions: list[int] = []
        group_step_positions: list[int] = []
        group_decision_indices: list[int] = []
        for step_idx, trace_kind in enumerate(parsed_batch.trace_kinds):
            if trace_kind == "may":
                may_positions.append(step_idx)
                continue
            count = write.decision_counts[step_idx]
            start = write.decision_starts[step_idx]
            for k in range(count):
                group_step_positions.append(step_idx)
                group_decision_indices.append(start + k)

        may_log_probs: Tensor | None = None
        may_entropies: Tensor | None = None
        may_selected_cpu: list[float] = []
        if may_positions:
            may_pos_t = torch.tensor(may_positions, dtype=torch.long, device=device)
            may_logits = forward.may_logits[may_pos_t]
            may_dist = Bernoulli(logits=may_logits)
            if deterministic:
                may_sel = (may_logits >= 0).to(dtype=forward.values.dtype)
            else:
                may_sel = may_dist.sample()
            may_log_probs = may_dist.log_prob(may_sel)
            may_entropies = may_dist.entropy()
            may_selected_cpu = may_sel.detach().cpu().tolist()
            may_buf = write.step_indices[may_pos_t]
            rb.may_selected[may_buf] = may_sel.to(dtype=rb.may_selected.dtype)

        per_step_log_prob_sum: Tensor | None = None
        per_step_entropy_sum: Tensor | None = None
        decision_selected_cpu: list[int] = []
        if group_step_positions:
            pos_t = torch.tensor(group_step_positions, dtype=torch.long, device=device)
            idx_t = torch.tensor(group_decision_indices, dtype=torch.long, device=device)
            option_idx = rb.decision_option_idx[idx_t]
            target_idx = rb.decision_target_idx[idx_t]
            masks = rb.decision_mask[idx_t]
            uses_none = rb.uses_none_head[idx_t]

            group_idx, choice_cols, flat_logits, flat_log_probs, group_entropies = (
                self._flat_decision_distribution(
                    step_positions=pos_t,
                    option_idx=option_idx,
                    target_idx=target_idx,
                    masks=masks,
                    uses_none=uses_none,
                    option_vectors=forward.option_vectors,
                    target_vectors=forward.target_vectors,
                    query=cast(Tensor, forward.query),
                    none_logits=forward.none_logits,
                )
            )
            decision_selected, decision_log_probs = self._sample_flat_decisions(
                group_idx=group_idx,
                choice_cols=choice_cols,
                flat_logits=flat_logits,
                flat_log_probs=flat_log_probs,
                deterministic=deterministic,
            )
            per_step_log_prob_sum = torch.zeros(n, dtype=forward.values.dtype, device=device)
            per_step_entropy_sum = torch.zeros(n, dtype=forward.values.dtype, device=device)
            per_step_log_prob_sum.scatter_add_(0, pos_t, decision_log_probs)
            per_step_entropy_sum.scatter_add_(0, pos_t, group_entropies)
            decision_selected_cpu = decision_selected.detach().cpu().tolist()
            rb.selected_indices[idx_t] = decision_selected

        may_lookup = {step_idx: pos for pos, step_idx in enumerate(may_positions)}
        results: list[PolicyStep] = []
        offset = 0
        for step_idx, trace_kind in enumerate(parsed_batch.trace_kinds):
            pending = (
                parsed_batch.pendings[step_idx]
                if step_idx < len(parsed_batch.pendings)
                else cast(PendingState, {})
            )
            value = forward.values[step_idx]
            replay_idx = int(write.step_indices[step_idx])
            decision_count = write.decision_counts[step_idx]

            if trace_kind == "may":
                pos = may_lookup[step_idx]
                assert may_log_probs is not None and may_entropies is not None
                sel_scalar = may_selected_cpu[pos]
                trace = ActionTrace("may", binary=(float(sel_scalar),))
                results.append(
                    PolicyStep(
                        action=action_from_choice_accepted(bool(sel_scalar >= 0.5)),
                        trace=trace,
                        log_prob=may_log_probs[pos],
                        value=value,
                        entropy=may_entropies[pos],
                        replay_idx=replay_idx,
                        may_selected=int(sel_scalar >= 0.5),
                    )
                )
                continue

            if decision_count == 0:
                zero = torch.zeros((), device=device)
                results.append(
                    PolicyStep(
                        action=cast(ActionRequest, {"kind": "pass"}),
                        trace=ActionTrace("priority", indices=(0,)),
                        log_prob=zero,
                        value=value,
                        entropy=zero,
                        replay_idx=replay_idx,
                        selected_choice_cols=(),
                    )
                )
                continue

            assert per_step_log_prob_sum is not None and per_step_entropy_sum is not None
            step_selected = decision_selected_cpu[offset : offset + decision_count]
            offset += decision_count
            if pending:
                trace, action = self._decode_action(trace_kind, pending, step_selected)
            else:
                trace, action = self._trace_action_without_pending(trace_kind, step_selected)
            results.append(
                PolicyStep(
                    action=action,
                    trace=trace,
                    log_prob=per_step_log_prob_sum[step_idx],
                    value=value,
                    entropy=per_step_entropy_sum[step_idx],
                    replay_idx=replay_idx,
                    selected_choice_cols=tuple(step_selected),
                )
            )
        return results

    def _parsed_batch_from_steps(self, parsed_steps: list[ParsedStep]) -> ParsedBatch:
        parsed_state = ParsedGameStateBatch(
            slot_card_rows=torch.tensor(
                [parsed.parsed_state.slot_card_rows for parsed in parsed_steps],
                dtype=torch.long,
            ),
            slot_occupied=torch.tensor(
                [parsed.parsed_state.slot_occupied for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            slot_tapped=torch.tensor(
                [parsed.parsed_state.slot_tapped for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            game_info=torch.tensor(
                [parsed.parsed_state.game_info for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            card_id_to_slots=[parsed.parsed_state.card_id_to_slot for parsed in parsed_steps],
        )
        parsed_action = ParsedActionBatch(
            pending_kind_id=torch.tensor(
                [parsed.parsed_action.pending_kind_id for parsed in parsed_steps],
                dtype=torch.long,
            ),
            num_present_options=torch.tensor(
                [parsed.parsed_action.num_present_options for parsed in parsed_steps],
                dtype=torch.long,
            ),
            option_kind_ids=torch.tensor(
                [parsed.parsed_action.option_kind_ids for parsed in parsed_steps],
                dtype=torch.long,
            ),
            option_scalars=torch.tensor(
                [parsed.parsed_action.option_scalars for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            option_mask=torch.tensor(
                [parsed.parsed_action.option_mask for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            option_ref_slot_idx=torch.tensor(
                [parsed.parsed_action.option_ref_slot_idx for parsed in parsed_steps],
                dtype=torch.long,
            ),
            option_ref_card_row=torch.tensor(
                [parsed.parsed_action.option_ref_card_row for parsed in parsed_steps],
                dtype=torch.long,
            ),
            target_mask=torch.tensor(
                [parsed.parsed_action.target_mask for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            target_type_ids=torch.tensor(
                [parsed.parsed_action.target_type_ids for parsed in parsed_steps],
                dtype=torch.long,
            ),
            target_scalars=torch.tensor(
                [parsed.parsed_action.target_scalars for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            target_overflow=torch.tensor(
                [parsed.parsed_action.target_overflow for parsed in parsed_steps],
                dtype=torch.float32,
            ),
            target_ref_slot_idx=torch.tensor(
                [parsed.parsed_action.target_ref_slot_idx for parsed in parsed_steps],
                dtype=torch.long,
            ),
            target_ref_is_player=torch.tensor(
                [parsed.parsed_action.target_ref_is_player for parsed in parsed_steps],
                dtype=torch.bool,
            ),
            target_ref_is_self=torch.tensor(
                [parsed.parsed_action.target_ref_is_self for parsed in parsed_steps],
                dtype=torch.bool,
            ),
            priority_candidates=[
                parsed.parsed_action.priority_candidates for parsed in parsed_steps
            ],
        )
        flat_option_idx: list[list[int]] = []
        flat_target_idx: list[list[int]] = []
        flat_mask: list[list[bool]] = []
        flat_uses_none: list[bool] = []
        decision_starts: list[int] = []
        decision_counts: list[int] = []
        cursor = 0
        for parsed in parsed_steps:
            decision_starts.append(cursor)
            count = len(parsed.decision_option_idx)
            decision_counts.append(count)
            cursor += count
            flat_option_idx.extend(parsed.decision_option_idx)
            flat_target_idx.extend(parsed.decision_target_idx)
            flat_mask.extend(parsed.decision_mask)
            flat_uses_none.extend(parsed.uses_none_head)
        if flat_option_idx:
            decision_option_idx = torch.tensor(flat_option_idx, dtype=torch.long)
            decision_target_idx = torch.tensor(flat_target_idx, dtype=torch.long)
            decision_mask = torch.tensor(flat_mask, dtype=torch.bool)
            uses_none_head = torch.tensor(flat_uses_none, dtype=torch.bool)
        else:
            decision_option_idx = torch.zeros((0, self.max_cached_choices), dtype=torch.long)
            decision_target_idx = torch.zeros((0, self.max_cached_choices), dtype=torch.long)
            decision_mask = torch.zeros((0, self.max_cached_choices), dtype=torch.bool)
            uses_none_head = torch.zeros((0,), dtype=torch.bool)
        return ParsedBatch(
            parsed_state=parsed_state,
            parsed_action=parsed_action,
            trace_kinds=[parsed.trace_kind for parsed in parsed_steps],
            trace_kind_ids=torch.tensor(
                [parsed.trace_kind_id for parsed in parsed_steps],
                dtype=torch.long,
            ),
            pendings=[parsed.pending for parsed in parsed_steps],
            decision_option_idx=decision_option_idx,
            decision_target_idx=decision_target_idx,
            decision_mask=decision_mask,
            uses_none_head=uses_none_head,
            decision_starts=decision_starts,
            decision_counts=decision_counts,
        )

    def _parsed_batch_from_native(self, native_batch: NativeEncodedBatch) -> ParsedBatch:
        return ParsedBatch(
            parsed_state=ParsedGameStateBatch(
                slot_card_rows=native_batch.slot_card_rows,
                slot_occupied=native_batch.slot_occupied,
                slot_tapped=native_batch.slot_tapped,
                game_info=native_batch.game_info,
                card_id_to_slots=[{} for _ in native_batch.pendings],
            ),
            parsed_action=ParsedActionBatch(
                pending_kind_id=native_batch.pending_kind_id,
                num_present_options=native_batch.num_present_options,
                option_kind_ids=native_batch.option_kind_ids,
                option_scalars=native_batch.option_scalars,
                option_mask=native_batch.option_mask,
                option_ref_slot_idx=native_batch.option_ref_slot_idx,
                option_ref_card_row=native_batch.option_ref_card_row,
                target_mask=native_batch.target_mask,
                target_type_ids=native_batch.target_type_ids,
                target_scalars=native_batch.target_scalars,
                target_overflow=native_batch.target_overflow,
                target_ref_slot_idx=native_batch.target_ref_slot_idx,
                target_ref_is_player=native_batch.target_ref_is_player,
                target_ref_is_self=native_batch.target_ref_is_self,
                priority_candidates=[[] for _ in native_batch.pendings],
            ),
            trace_kinds=[cast(TraceKind, trace_kind) for trace_kind in native_batch.trace_kinds],
            trace_kind_ids=native_batch.trace_kind_id,
            pendings=native_batch.pendings,
            decision_option_idx=native_batch.decision_option_idx,
            decision_target_idx=native_batch.decision_target_idx,
            decision_mask=native_batch.decision_mask,
            uses_none_head=native_batch.uses_none_head,
            decision_starts=native_batch.decision_start.detach().cpu().tolist(),
            decision_counts=native_batch.decision_count.detach().cpu().tolist(),
        )

    def evaluate_replay_batch(
        self,
        replay_rows: list[int],
        *,
        return_extras: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, _ReplayBatchExtras | None]:
        """Replay log-probs/entropies/values from replay rows with gradients.

        When ``return_extras`` is true, also returns the forward batch plus the
        decision-row expansion tensors so callers (e.g. the SPR auxiliary loss)
        can reuse them without recomputing.
        """

        if not replay_rows:
            raise ValueError("replay_rows must not be empty")

        device = self.device
        rb = self.rollout_buffer
        step_indices = torch.tensor(replay_rows, dtype=torch.long, device=device)
        n = int(step_indices.numel())
        forward = self._forward_batch(step_indices)
        log_probs = torch.zeros(n, dtype=forward.values.dtype, device=device)
        entropies = torch.zeros(n, dtype=forward.values.dtype, device=device)

        trace_kind_ids = rb.trace_kind_id[step_indices]
        may_mask = trace_kind_ids == TRACE_KIND_TO_ID["may"]
        may_log_probs, may_entropies, _may_logits_per_step, _may_selected_per_step = (
            score_may_decisions_from_forward(
                forward,
                may_selected=rb.may_selected[step_indices],
                may_mask=may_mask,
            )
        )
        log_probs = log_probs + may_log_probs
        entropies = entropies + may_entropies

        decision_starts = rb.decision_start[step_indices]
        decision_counts = rb.decision_count[step_indices]
        active_mask = decision_counts > 0
        group_pos_t = torch.empty(0, dtype=torch.long, device=device)
        group_idx_t = torch.empty(0, dtype=torch.long, device=device)
        group_selected = torch.empty(0, dtype=torch.long, device=device)
        group_uses_none = torch.empty(0, dtype=torch.bool, device=device)
        if active_mask.any():
            pos_t = active_mask.nonzero(as_tuple=False).squeeze(-1)
            active_counts = decision_counts[pos_t]
            # Static upper bound (decision-cache width); valid_offsets masks
            # unused positions. Avoids a per-call sync on active_counts.max().
            max_count = int(rb.decision_option_idx.shape[1])
            offsets = torch.arange(max_count, dtype=torch.long, device=device).unsqueeze(0)
            expanded_offsets = offsets.expand(pos_t.shape[0], -1)
            valid_offsets = expanded_offsets < active_counts.unsqueeze(1)
            idx_t = (decision_starts[pos_t].unsqueeze(1) + expanded_offsets)[valid_offsets]
            pos_t = torch.repeat_interleave(pos_t, active_counts)
            option_idx = rb.decision_option_idx[idx_t]
            target_idx = rb.decision_target_idx[idx_t]
            masks = rb.decision_mask[idx_t]
            uses_none = rb.uses_none_head[idx_t]
            selected = rb.selected_indices[idx_t]

            group_idx, choice_cols, _flat_logits, flat_log_probs, group_entropies = (
                self._flat_decision_distribution(
                    step_positions=pos_t,
                    option_idx=option_idx,
                    target_idx=target_idx,
                    masks=masks,
                    uses_none=uses_none,
                    option_vectors=forward.option_vectors,
                    target_vectors=forward.target_vectors,
                    query=cast(Tensor, forward.query),
                    none_logits=forward.none_logits,
                )
            )
            selected_mask = choice_cols == selected[group_idx]
            selected_flat_log_probs = torch.where(
                selected_mask,
                flat_log_probs,
                torch.zeros_like(flat_log_probs),
            )
            group_log_probs = torch.zeros_like(group_entropies)
            group_log_probs.scatter_add_(0, group_idx, selected_flat_log_probs)
            log_probs.scatter_add_(0, pos_t, group_log_probs)
            entropies.scatter_add_(0, pos_t, group_entropies)

            group_pos_t = pos_t
            group_idx_t = idx_t
            group_selected = selected
            group_uses_none = uses_none

        extras: _ReplayBatchExtras | None = None
        if return_extras:
            extras = _ReplayBatchExtras(
                forward=forward,
                step_indices=step_indices,
                group_pos_t=group_pos_t,
                group_idx_t=group_idx_t,
                group_selected=group_selected,
                group_uses_none=group_uses_none,
            )
        return log_probs, entropies, forward.values, extras

    @torch.no_grad()
    def recompute_lstm_states_for_episode(
        self,
        replay_rows: list[int],
    ) -> tuple[Tensor, Tensor] | None:
        """Re-run the LSTM scan from h=c=0 over an episode using *this* policy.

        Returns per-step (h_in, c_in) of shape ``(num_layers, T, hidden)`` so
        the caller can override the rollout buffer's stored hidden states
        (which were written by the *behavior* policy at rollout time) when
        calling :meth:`evaluate_replay_batch_per_choice` on this policy.

        ``replay_rows`` must list a single episode in forward order — h starts
        at zero and threads through, so cross-episode rows would silently
        carry state across the wrong boundary. Returns ``None`` if the policy
        is not recurrent.

        Issue 2: each R-NaD policy (online/target/reg_cur/reg_prev) has its
        own parameters and thus its own hidden-state trajectory through a
        replayed episode. Evaluating target or reg from the behavior-policy
        hidden state (the buffer's stored ``lstm_h_in``) would silently mix
        policies. Each policy must own its replay-time hidden states.
        """

        if not self.use_lstm:
            return None
        if not replay_rows:
            raise ValueError("replay_rows must be non-empty")
        device = self.device
        step_indices = torch.tensor(replay_rows, dtype=torch.long, device=device)
        n = int(step_indices.numel())
        inputs, _stale_lstm_state = self._gather_from_rollout(step_indices)
        features, _option_vectors, _target_vectors = self._embed_forward_inputs(inputs)
        dtype = self._compute_hidden_dtype()
        h = torch.zeros(self.hidden_layers, 1, self.hidden_dim, dtype=dtype, device=device)
        c = torch.zeros_like(h)
        h_list: list[Tensor] = []
        c_list: list[Tensor] = []
        for t in range(n):
            h_list.append(h)
            c_list.append(c)
            projected = self.feature_projection(features[t : t + 1]).unsqueeze(1)
            _output, (h, c) = self.lstm(projected, (h.contiguous(), c.contiguous()))
        h_stack = torch.cat(h_list, dim=1).contiguous()  # (num_layers, T, hidden)
        c_stack = torch.cat(c_list, dim=1).contiguous()
        return h_stack, c_stack

    @torch.no_grad()
    def recompute_lstm_states_for_episodes(
        self,
        episodes: list[list[int]],
        *,
        strategy: LstmRecomputeStrategy = "legacy",
    ) -> list[tuple[Tensor, Tensor]] | None:
        """Per-episode LSTM input-state recompute for a *batch* of episodes.

        Returns ``None`` for non-recurrent policies. Otherwise, a list with
        one ``(h_in, c_in)`` per episode of shape
        ``(num_layers, T_i, hidden)`` -- matching
        :meth:`recompute_lstm_states_for_episode`. ``strategy`` selects
        between the per-episode reference loop and three batched variants;
        see :mod:`magic_ai.lstm_recompute` for details. All strategies are
        mathematically equivalent under fp32.
        """

        if not self.use_lstm:
            return None
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if any(len(ep) == 0 for ep in episodes):
            raise ValueError("each episode must contain at least one row")
        device = self.device
        dtype = self._compute_hidden_dtype()
        flat_rows: list[int] = [r for ep in episodes for r in ep]
        step_indices = torch.tensor(flat_rows, dtype=torch.long, device=device)
        inputs, _stale_lstm_state = self._gather_from_rollout(step_indices)
        features_flat, _option_vectors, _target_vectors = self._embed_forward_inputs(inputs)
        proj_flat = self.feature_projection(features_flat).to(dtype=dtype)
        n = len(episodes)
        lengths = [len(ep) for ep in episodes]
        t_max = max(lengths)
        projected = torch.zeros(t_max, n, self.hidden_dim, dtype=dtype, device=device)
        offset = 0
        for i, t_i in enumerate(lengths):
            projected[:t_i, i] = proj_flat[offset : offset + t_i]
            offset += t_i
        return lstm_recompute_per_step_states(self.lstm, projected, lengths, strategy=strategy)

    def recompute_lstm_outputs_for_episodes(
        self,
        episodes: list[list[int]],
        *,
        chunk_size: int = 200,
        compiled_lstm: Callable[..., Any] | None = None,
    ) -> list[Tensor] | None:
        """Fused per-step ``h_out`` recompute for the override-interface path.

        Returns ``None`` for non-recurrent policies. Otherwise a list of
        ``(T_i, hidden)`` tensors -- the top-layer LSTM hidden output at each
        replay step -- produced by a single fused ``nn.LSTM`` call. The
        consumer should pass each per-episode tensor as
        ``hidden_override`` to :meth:`evaluate_replay_batch_per_choice``,
        which skips the per-step LSTM cell.

        Not ``@torch.no_grad`` -- gradient flows through the LSTM (full BPTT
        for the online policy). Callers that don't need gradients (target /
        regularizer policies) should wrap in ``torch.no_grad()`` themselves.
        """

        if not self.use_lstm:
            return None
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if any(len(ep) == 0 for ep in episodes):
            raise ValueError("each episode must contain at least one row")
        device = self.device
        dtype = self._compute_hidden_dtype()
        flat_rows: list[int] = [r for ep in episodes for r in ep]
        step_indices = torch.tensor(flat_rows, dtype=torch.long, device=device)
        inputs, _stale_lstm_state = self._gather_from_rollout(step_indices)
        features_flat, _option_vectors, _target_vectors = self._embed_forward_inputs(inputs)
        proj_flat = self.feature_projection(features_flat).to(dtype=dtype)
        lengths = [len(ep) for ep in episodes]
        proj_per_episode: list[Tensor] = []
        offset = 0
        for t_i in lengths:
            proj_per_episode.append(proj_flat[offset : offset + t_i])
            offset += t_i
        return lstm_recompute_per_step_h_out(
            self.lstm, proj_per_episode, chunk_size=chunk_size, compiled_lstm=compiled_lstm
        )

    def evaluate_replay_batch_per_choice(
        self,
        replay_rows: list[int],
        *,
        lstm_state_override: tuple[Tensor, Tensor] | None = None,
        hidden_override: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, ReplayPerChoice]:
        """Like :meth:`evaluate_replay_batch` but also returns per-choice tensors.

        Used by the R-NaD full per-action NeuRD trainer path: the returned
        :class:`ReplayPerChoice` carries the flat logits and log-probs for
        every legal decision-group choice across the batch, plus the
        mapping back to batch-step indices and the sampled column per step.
        ``may``-kind steps have no decision group and get a ``-1`` sentinel
        in ``is_sampled_flat`` (a boolean mask over the flat axis,
        ``True`` at exactly the sampled cell within each decision
        group; this preserves per-group sampled identity for
        multi-decision-group steps).
        """

        if not replay_rows:
            raise ValueError("replay_rows must not be empty")

        device = self.device
        rb = self.rollout_buffer
        step_indices = torch.tensor(replay_rows, dtype=torch.long, device=device)
        n = int(step_indices.numel())
        if hidden_override is not None and lstm_state_override is not None:
            raise ValueError("pass at most one of hidden_override / lstm_state_override")
        if hidden_override is not None:
            forward = self._forward_batch_with_hidden_override(step_indices, hidden_override)
        elif lstm_state_override is not None:
            forward = self._forward_batch_with_lstm_override(step_indices, lstm_state_override)
        else:
            forward = self._forward_batch(step_indices)
        log_probs = torch.zeros(n, dtype=forward.values.dtype, device=device)
        entropies = torch.zeros(n, dtype=forward.values.dtype, device=device)

        trace_kind_ids = rb.trace_kind_id[step_indices]
        may_mask = trace_kind_ids == TRACE_KIND_TO_ID["may"]
        may_log_probs, may_entropies, may_logits_per_step, may_selected_per_step = (
            score_may_decisions_from_forward(
                forward,
                may_selected=rb.may_selected[step_indices],
                may_mask=may_mask,
            )
        )
        log_probs = log_probs + may_log_probs
        entropies = entropies + may_entropies

        decision_starts = rb.decision_start[step_indices]
        decision_counts = rb.decision_count[step_indices]
        active_mask = decision_counts > 0

        flat_logits = forward.values.new_zeros(0)
        flat_log_probs = forward.values.new_zeros(0)
        group_idx_out = torch.zeros(0, dtype=torch.long, device=device)
        choice_cols_out = torch.zeros(0, dtype=torch.long, device=device)
        is_sampled_flat_out = torch.zeros(0, dtype=torch.bool, device=device)
        decision_group_id_flat_out = torch.zeros(0, dtype=torch.long, device=device)
        step_for_decision_group_out = torch.zeros(0, dtype=torch.long, device=device)

        if active_mask.any():
            pos_t = active_mask.nonzero(as_tuple=False).squeeze(-1)
            active_counts = decision_counts[pos_t]
            # Static upper bound (decision-cache width); valid_offsets masks
            # unused positions. Avoids a per-call sync on active_counts.max().
            max_count = int(rb.decision_option_idx.shape[1])
            offsets = torch.arange(max_count, dtype=torch.long, device=device).unsqueeze(0)
            expanded_offsets = offsets.expand(pos_t.shape[0], -1)
            valid_offsets = expanded_offsets < active_counts.unsqueeze(1)
            idx_t = (decision_starts[pos_t].unsqueeze(1) + expanded_offsets)[valid_offsets]
            pos_t_flat = torch.repeat_interleave(pos_t, active_counts)
            option_idx = rb.decision_option_idx[idx_t]
            target_idx = rb.decision_target_idx[idx_t]
            masks = rb.decision_mask[idx_t]
            uses_none = rb.uses_none_head[idx_t]
            selected = rb.selected_indices[idx_t]

            group_idx, choice_cols, flat_logits_all, flat_log_probs_all, group_entropies = (
                self._flat_decision_distribution(
                    step_positions=pos_t_flat,
                    option_idx=option_idx,
                    target_idx=target_idx,
                    masks=masks,
                    uses_none=uses_none,
                    option_vectors=forward.option_vectors,
                    target_vectors=forward.target_vectors,
                    query=cast(Tensor, forward.query),
                    none_logits=forward.none_logits,
                )
            )
            # group_idx maps each flat entry to its POSITION within pos_t_flat
            # (0..len(idx_t)-1); we want the batch-step index in 0..n-1.
            step_for_flat = pos_t_flat[group_idx]
            selected_mask = choice_cols == selected[group_idx]
            selected_flat_log_probs = torch.where(
                selected_mask,
                flat_log_probs_all,
                torch.zeros_like(flat_log_probs_all),
            )
            group_log_probs = torch.zeros_like(group_entropies)
            group_log_probs.scatter_add_(0, group_idx, selected_flat_log_probs)
            log_probs.scatter_add_(0, pos_t_flat, group_log_probs)
            entropies.scatter_add_(0, pos_t_flat, group_entropies)

            flat_logits = flat_logits_all
            flat_log_probs = flat_log_probs_all
            group_idx_out = step_for_flat
            choice_cols_out = choice_cols
            is_sampled_flat_out = selected_mask
            decision_group_id_flat_out = group_idx
            step_for_decision_group_out = pos_t_flat

        per_choice = ReplayPerChoice(
            flat_logits=flat_logits,
            flat_log_probs=flat_log_probs,
            group_idx=group_idx_out,
            choice_cols=choice_cols_out,
            is_sampled_flat=is_sampled_flat_out,
            decision_group_id_flat=decision_group_id_flat_out,
            step_for_decision_group=step_for_decision_group_out,
            may_is_active=may_mask,
            may_logits_per_step=may_logits_per_step,
            may_selected_per_step=may_selected_per_step,
        )
        return log_probs, entropies, forward.values, per_choice

    def _encode_latent(
        self,
        step_indices: Tensor,
        *,
        use_target: bool,
    ) -> Tensor:
        """Compute the post-LSTM feature for the given replay rows.

        Mirrors the feature path of ``_forward_batch`` but returns only the
        recurrent hidden vector, using either the online or the frozen target
        copy of the encoder stack.
        """

        inputs, lstm_state = self._gather_from_rollout(step_indices)
        if lstm_state is None:
            raise RuntimeError("SPR latent encoding requires LSTM state inputs")
        if use_target:
            hidden, _next_state = self._call_compute_hidden_target(
                inputs,
                lstm_state[0],
                lstm_state[1],
            )
        else:
            hidden, _next_state = self._compute_hidden(
                inputs,
                h_in=lstm_state[0],
                c_in=lstm_state[1],
                use_target=False,
            )
        return hidden

    def _selected_action_vectors_from_groups(
        self,
        *,
        batch_size: int,
        decision_counts: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        group_pos_t: Tensor,
        group_idx_t: Tensor,
        group_selected: Tensor,
        group_uses_none: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Mean of selected option/target vectors per step, over decision groups.

        Uses pre-expanded decision-row tensors (one entry per decision group)
        so no per-minibatch ``.item()`` sync is needed.
        """

        rb = self.rollout_buffer
        d_model = int(option_vectors.shape[-1])
        device = option_vectors.device
        dtype = option_vectors.dtype
        zero = torch.zeros(batch_size, d_model, device=device, dtype=dtype)
        if group_pos_t.numel() == 0:
            return zero, zero.clone()

        is_none = group_uses_none & (group_selected == 0)
        opt_idx = rb.decision_option_idx[group_idx_t, group_selected]
        tgt_idx = rb.decision_target_idx[group_idx_t, group_selected]
        opt_idx_c = opt_idx.clamp_min(0).clamp_max(option_vectors.shape[1] - 1)
        tgt_idx_c = tgt_idx.clamp_min(0).clamp_max(target_vectors.shape[2] - 1)

        opt_vec = option_vectors[group_pos_t, opt_idx_c]
        opt_vec = torch.where(is_none.unsqueeze(-1), torch.zeros_like(opt_vec), opt_vec)

        tgt_has = (~is_none) & (tgt_idx >= 0)
        tgt_vec = target_vectors[group_pos_t, opt_idx_c, tgt_idx_c]
        tgt_vec = torch.where(tgt_has.unsqueeze(-1), tgt_vec, torch.zeros_like(tgt_vec))

        scatter_idx = group_pos_t.unsqueeze(-1).expand(-1, d_model)
        opt_sum = torch.zeros_like(zero).scatter_add(0, scatter_idx, opt_vec)
        tgt_sum = torch.zeros_like(zero).scatter_add(0, scatter_idx, tgt_vec)
        denom = decision_counts.clamp_min(1).to(dtype).unsqueeze(-1)
        return opt_sum / denom, tgt_sum / denom

    def compute_spr_loss(
        self,
        step_indices: Tensor,
        *,
        extras: _ReplayBatchExtras | None = None,
    ) -> Tensor:
        """Self-predictive (SPR) auxiliary loss on the given replay rows.

        For each row t with a valid next row t+1 in the same episode, predict
        the target network's post-LSTM latent at t+1 from the online latent at
        t plus an embedding of the full action taken at t (trace kind + mean
        of selected option/target vectors + may bit), and penalize their
        (normalized) mean squared error.

        If ``extras`` is provided (from ``evaluate_replay_batch(..., return_extras=True)``)
        the cached online forward and decision expansion are reused, avoiding
        a duplicate forward pass and a ``.item()`` sync per minibatch.
        """

        if not self.spr_enabled:
            raise RuntimeError("SPR is not enabled on this policy")

        rb = self.rollout_buffer

        if extras is not None:
            forward = extras.forward
            group_pos_t = extras.group_pos_t
            group_idx_t = extras.group_idx_t
            group_selected = extras.group_selected
            group_uses_none = extras.group_uses_none
        else:
            forward = self._forward_batch(step_indices)
            decision_starts = rb.decision_start[step_indices]
            decision_counts_all = rb.decision_count[step_indices]
            active_mask = decision_counts_all > 0
            device = step_indices.device
            if active_mask.any():
                pos_t = active_mask.nonzero(as_tuple=False).squeeze(-1)
                active_counts = decision_counts_all[pos_t]
                # Static upper bound (decision-cache width); valid_offsets
                # masks unused positions. Avoids a per-call sync on max().
                max_count = int(rb.decision_option_idx.shape[1])
                offsets = torch.arange(max_count, dtype=torch.long, device=device).unsqueeze(0)
                expanded_offsets = offsets.expand(pos_t.shape[0], -1)
                valid_offsets = expanded_offsets < active_counts.unsqueeze(1)
                idx_t = (decision_starts[pos_t].unsqueeze(1) + expanded_offsets)[valid_offsets]
                pos_t = torch.repeat_interleave(pos_t, active_counts)
                group_pos_t = pos_t
                group_idx_t = idx_t
                group_selected = rb.selected_indices[idx_t]
                group_uses_none = rb.uses_none_head[idx_t]
            else:
                group_pos_t = torch.empty(0, dtype=torch.long, device=device)
                group_idx_t = torch.empty(0, dtype=torch.long, device=device)
                group_selected = torch.empty(0, dtype=torch.long, device=device)
                group_uses_none = torch.empty(0, dtype=torch.bool, device=device)

        batch_size = int(step_indices.shape[0])
        decision_counts = rb.decision_count[step_indices]
        opt_mean, tgt_mean = self._selected_action_vectors_from_groups(
            batch_size=batch_size,
            decision_counts=decision_counts,
            option_vectors=forward.option_vectors,
            target_vectors=forward.target_vectors,
            group_pos_t=group_pos_t,
            group_idx_t=group_idx_t,
            group_selected=group_selected,
            group_uses_none=group_uses_none,
        )
        z_online = forward.hidden
        action_emb_0 = self._spr_action_embedding_full(
            step_indices, opt_mean=opt_mean, tgt_mean=tgt_mean, dtype=z_online.dtype
        )

        z_hat = z_online
        cur_idx = step_indices
        action_emb = action_emb_0
        cumulative_has_next = torch.ones(
            step_indices.shape[0], dtype=z_online.dtype, device=z_online.device
        )
        loss_terms: list[Tensor] = []
        for k in range(1, self.spr_k + 1):
            step_has_next = rb.has_next_same_perspective[cur_idx]
            cumulative_has_next = cumulative_has_next * step_has_next

            pred_in = torch.cat([z_hat, action_emb], dim=-1)
            z_hat = self.spr_transition(pred_in)

            next_idx = rb.next_same_perspective_step_idx[cur_idx]
            with torch.no_grad():
                z_target_next = self._encode_latent(next_idx, use_target=True)
                y_target = self.spr_g_target(z_target_next)
            y_pred = self.spr_q(self.spr_g_online(z_hat))

            cos = F.cosine_similarity(y_pred, y_target, dim=-1)
            denom_k = cumulative_has_next.sum().clamp_min(1.0)
            mean_cos_k = (cos * cumulative_has_next).sum() / denom_k
            loss_terms.append(mean_cos_k)

            if k < self.spr_k:
                cur_idx = next_idx
                action_emb = self._spr_action_embedding_simple(cur_idx, dtype=z_online.dtype)

        return 1.0 - torch.stack(loss_terms).sum() / self.spr_k

    def _spr_action_embedding_full(
        self,
        step_indices: Tensor,
        *,
        opt_mean: Tensor,
        tgt_mean: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        rb = self.rollout_buffer
        may_bit = rb.may_selected[step_indices].unsqueeze(-1).to(dtype)
        action_raw = torch.cat([opt_mean, tgt_mean, may_bit], dim=-1)
        action_proj = self.spr_action_projector(action_raw)
        trace_emb = self.spr_action_embedding(rb.trace_kind_id[step_indices])
        return torch.cat([trace_emb, action_proj], dim=-1)

    def _spr_action_embedding_simple(
        self,
        step_indices: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        """Cheap action embedding for chained SPR rollout steps.

        Skips a fresh forward pass by zeroing the option/target mean components
        and keeping only the trace-kind embedding plus the may bit.
        """

        rb = self.rollout_buffer
        d_model = self.game_state_encoder.d_model
        n = int(step_indices.shape[0])
        device = step_indices.device
        zeros = torch.zeros(n, 2 * d_model, device=device, dtype=dtype)
        may_bit = rb.may_selected[step_indices].unsqueeze(-1).to(dtype)
        action_raw = torch.cat([zeros, may_bit], dim=-1)
        action_proj = self.spr_action_projector(action_raw)
        trace_emb = self.spr_action_embedding(rb.trace_kind_id[step_indices])
        return torch.cat([trace_emb, action_proj], dim=-1)

    @torch.no_grad()
    def update_spr_target(self, decay: float | None = None) -> None:
        """EMA update of the frozen target encoder stack toward the online one."""

        if not self.spr_enabled:
            return
        tau = self.spr_ema_decay if decay is None else decay
        pairs = (
            (self.game_state_encoder, self.target_game_state_encoder),
            (self.action_encoder, self.target_action_encoder),
            (self.feature_projection, self.target_feature_projection),
            (self.lstm, self.target_lstm),
            (self.spr_g_online, self.spr_g_target),
        )
        for online, target in pairs:
            for op, tp in zip(online.parameters(), target.parameters(), strict=True):
                tp.mul_(tau).add_(op.detach(), alpha=1.0 - tau)
            for ob, tb in zip(online.buffers(), target.buffers(), strict=True):
                tb.copy_(ob)

    def _compute_hidden_dtype(self) -> torch.dtype:
        if self.use_lstm:
            return next(self.feature_projection.parameters()).dtype
        return next(self.trunk.parameters()).dtype

    def _gather_from_rollout(
        self,
        step_indices: Tensor,
    ) -> tuple[_ForwardInputs, tuple[Tensor, Tensor] | None]:
        rb = self.rollout_buffer
        inputs = _ForwardInputs(
            slot_card_rows=rb.slot_card_rows[step_indices],
            slot_occupied=rb.slot_occupied[step_indices],
            slot_tapped=rb.slot_tapped[step_indices],
            game_info=rb.game_info[step_indices],
            pending_kind_id=rb.pending_kind_id[step_indices],
            option_kind_ids=rb.option_kind_ids[step_indices],
            option_scalars=rb.option_scalars[step_indices],
            option_mask=rb.option_mask[step_indices],
            option_ref_slot_idx=rb.option_ref_slot_idx[step_indices],
            option_ref_card_row=rb.option_ref_card_row[step_indices],
            target_mask=rb.target_mask[step_indices],
            target_type_ids=rb.target_type_ids[step_indices],
            target_scalars=rb.target_scalars[step_indices],
            target_ref_slot_idx=rb.target_ref_slot_idx[step_indices],
            target_ref_is_player=rb.target_ref_is_player[step_indices],
            target_ref_is_self=rb.target_ref_is_self[step_indices],
        )
        if self.validate:
            self._validate_slot_card_rows(
                inputs.slot_card_rows,
                self.game_state_encoder.card_embedding_table.shape[0],
            )

        if not self.use_lstm:
            return inputs, None

        dtype = self._compute_hidden_dtype()
        return inputs, (
            rb.lstm_h_in[step_indices].permute(1, 0, 2).contiguous().to(dtype=dtype),
            rb.lstm_c_in[step_indices].permute(1, 0, 2).contiguous().to(dtype=dtype),
        )

    def _gather_from_native(
        self,
        native_batch: NativeEncodedBatch,
        *,
        env_indices: list[int] | None = None,
    ) -> tuple[_ForwardInputs, tuple[Tensor, Tensor] | None]:
        device = self.device
        inputs = _ForwardInputs(
            slot_card_rows=native_batch.slot_card_rows.to(device),
            slot_occupied=native_batch.slot_occupied.to(device),
            slot_tapped=native_batch.slot_tapped.to(device),
            game_info=native_batch.game_info.to(device),
            pending_kind_id=native_batch.pending_kind_id.to(device),
            option_kind_ids=native_batch.option_kind_ids.to(device),
            option_scalars=native_batch.option_scalars.to(device),
            option_mask=native_batch.option_mask.to(device),
            option_ref_slot_idx=native_batch.option_ref_slot_idx.to(device),
            option_ref_card_row=native_batch.option_ref_card_row.to(device),
            target_mask=native_batch.target_mask.to(device),
            target_type_ids=native_batch.target_type_ids.to(device),
            target_scalars=native_batch.target_scalars.to(device),
            target_ref_slot_idx=native_batch.target_ref_slot_idx.to(device),
            target_ref_is_player=native_batch.target_ref_is_player.to(device),
            target_ref_is_self=native_batch.target_ref_is_self.to(device),
        )
        if self.validate:
            self._validate_slot_card_rows(
                inputs.slot_card_rows,
                self.game_state_encoder.card_embedding_table.shape[0],
            )

        if not self.use_lstm:
            return inputs, None
        if env_indices is None:
            raise ValueError("env_indices are required for LSTM native rollout")
        batch_size = int(inputs.slot_card_rows.shape[0])
        if len(env_indices) != batch_size:
            raise ValueError("env_indices length must match native batch length")
        if self.live_lstm_h.shape[1] == 0:
            raise RuntimeError("LSTM env states have not been initialized")

        env_idx_t = torch.tensor(env_indices, dtype=torch.long, device=device)
        dtype = self._compute_hidden_dtype()
        return inputs, (
            self.live_lstm_h[:, env_idx_t].to(dtype=dtype),
            self.live_lstm_c[:, env_idx_t].to(dtype=dtype),
        )

    def _scatter_lstm_state(
        self,
        env_indices: list[int],
        h_next: Tensor,
        c_next: Tensor,
    ) -> None:
        env_idx_t = torch.tensor(env_indices, dtype=torch.long, device=h_next.device)
        self.live_lstm_h[:, env_idx_t] = h_next.detach()
        self.live_lstm_c[:, env_idx_t] = c_next.detach()

    def _embed_forward_inputs(
        self,
        inputs: _ForwardInputs,
        *,
        use_target: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if use_target:
            game_state_encoder = self.target_game_state_encoder
            action_encoder = self.target_action_encoder
        else:
            game_state_encoder = self.game_state_encoder
            action_encoder = self.action_encoder

        slot_vectors = game_state_encoder.embed_slot_vectors(
            inputs.slot_card_rows,
            inputs.slot_occupied,
            inputs.slot_tapped,
        )
        state_vector = game_state_encoder.state_vector_from_slots(slot_vectors, inputs.game_info)
        pending_vector, option_vectors, target_vectors = action_encoder.embed_from_parsed(
            slot_vectors=slot_vectors,
            pending_kind_id=inputs.pending_kind_id,
            option_kind_ids=inputs.option_kind_ids,
            option_scalars=inputs.option_scalars,
            option_mask=inputs.option_mask,
            option_ref_slot_idx=inputs.option_ref_slot_idx,
            option_ref_card_row=inputs.option_ref_card_row,
            target_mask=inputs.target_mask,
            target_type_ids=inputs.target_type_ids,
            target_scalars=inputs.target_scalars,
            target_ref_slot_idx=inputs.target_ref_slot_idx,
            target_ref_is_player=inputs.target_ref_is_player,
            target_ref_is_self=inputs.target_ref_is_self,
        )
        option_mask_f = inputs.option_mask.unsqueeze(-1)
        option_count = inputs.option_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (option_vectors * option_mask_f).sum(dim=1) / option_count
        features = torch.cat([state_vector, pending_vector, pooled], dim=-1)
        return features, option_vectors, target_vectors

    def _apply_trunk(self, features: Tensor) -> Tensor:
        return self.trunk(features)

    def _apply_lstm_cell(
        self,
        features: Tensor,
        h_in: Tensor,
        c_in: Tensor,
        *,
        use_target: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        feature_projection = (
            self.target_feature_projection if use_target else self.feature_projection
        )
        lstm = self.target_lstm if use_target else self.lstm
        projected = feature_projection(features).unsqueeze(1)
        output, (h_next, c_next) = lstm(projected, (h_in.contiguous(), c_in.contiguous()))
        return output[:, 0, :], h_next, c_next

    def _compute_hidden(
        self,
        inputs: _ForwardInputs,
        *,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
        use_target: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        features, _option_vectors, _target_vectors = self._embed_forward_inputs(
            inputs,
            use_target=use_target,
        )
        if not self.use_lstm:
            return self._apply_trunk(features), None
        if h_in is None or c_in is None:
            raise ValueError("h_in and c_in are required when use_lstm=True")
        hidden, h_next, c_next = self._apply_lstm_cell(
            features,
            h_in,
            c_in,
            use_target=use_target,
        )
        return hidden, (h_next, c_next)

    def _compute_hidden_target_impl(
        self,
        inputs: _ForwardInputs,
        h_in: Tensor,
        c_in: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        return self._compute_hidden(
            inputs,
            h_in=h_in,
            c_in=c_in,
            use_target=True,
        )

    def _compute_forward_impl(
        self,
        inputs: _ForwardInputs,
        *,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, tuple[Tensor, Tensor] | None
    ]:
        features, option_vectors, target_vectors = self._embed_forward_inputs(inputs)
        if self.use_lstm:
            if h_in is None or c_in is None:
                raise ValueError("h_in and c_in are required when use_lstm=True")
            hidden, h_next, c_next = self._apply_lstm_cell(
                features,
                h_in,
                c_in,
            )
            next_state: tuple[Tensor, Tensor] | None = (h_next, c_next)
        else:
            hidden = self._apply_trunk(features)
            next_state = None

        query = self.action_query(hidden) / math.sqrt(float(self.game_state_encoder.d_model))
        values = self.value_head(hidden).squeeze(-1)
        none_logits = self.none_blocker_head(hidden).squeeze(-1)
        may_logits = self.may_head(hidden).squeeze(-1)
        return (
            query,
            values,
            none_logits,
            may_logits,
            option_vectors,
            target_vectors,
            hidden,
            next_state,
        )

    def _maybe_init_compiled_functions(self) -> None:
        if not self.compile_forward:
            return
        if self._compiled_compute_forward_impl is not None:
            return
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")

        self._compiled_compute_forward_impl = torch.compile(
            self._compute_forward_impl,
            dynamic=True,
        )
        if self.use_lstm and self.spr_enabled:
            self._compiled_compute_hidden_target_impl = torch.compile(
                self._compute_hidden_target_impl,
                dynamic=True,
            )
        self._compiled_flat_decision_distribution_impl = torch.compile(
            flat_decision_distribution_impl,
            dynamic=True,
        )

    def _call_compute_hidden_target(
        self,
        inputs: _ForwardInputs,
        h_in: Tensor,
        c_in: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        self._maybe_init_compiled_functions()
        fn = self._compiled_compute_hidden_target_impl
        if fn is not None:
            return fn(inputs, h_in, c_in)
        return self._compute_hidden_target_impl(inputs, h_in, c_in)

    def _compute_forward(
        self,
        inputs: _ForwardInputs,
        *,
        h_in: Tensor | None = None,
        c_in: Tensor | None = None,
    ) -> tuple[_ForwardBatch, tuple[Tensor, Tensor] | None]:
        self._maybe_init_compiled_functions()
        fn = self._compiled_compute_forward_impl or self._compute_forward_impl
        (
            query,
            values,
            none_logits,
            may_logits,
            option_vectors,
            target_vectors,
            hidden,
            next_state,
        ) = fn(inputs, h_in=h_in, c_in=c_in)
        return (
            _ForwardBatch(
                query=query,
                values=values,
                none_logits=none_logits,
                may_logits=may_logits,
                option_vectors=option_vectors,
                target_vectors=target_vectors,
                hidden=hidden,
            ),
            next_state,
        )

    def _forward_batch(self, step_indices: Tensor) -> _ForwardBatch:
        inputs, lstm_state = self._gather_from_rollout(step_indices)
        h_in: Tensor | None = None
        c_in: Tensor | None = None
        if lstm_state is not None:
            h_in, c_in = lstm_state
        forward, _next_state = self._compute_forward(inputs, h_in=h_in, c_in=c_in)
        return forward

    def _forward_batch_with_hidden_override(
        self,
        step_indices: Tensor,
        hidden: Tensor,
    ) -> _ForwardBatch:
        """Forward path that takes precomputed top-layer ``hidden`` per step.

        Override-interface variant for issue 2: the recompute pass produces
        ``h_out[t]`` (top-layer hidden) per step via a single fused
        ``nn.LSTM`` call, and the action heads consume it directly --
        skipping the per-step LSTM cell entirely. This drops the per-step
        recompute launch count from O(T_max) to O(1) per policy.
        """

        inputs, _stale_lstm_state = self._gather_from_rollout(step_indices)
        dtype = self._compute_hidden_dtype()
        hidden = hidden.to(dtype=dtype).contiguous()
        _features, option_vectors, target_vectors = self._embed_forward_inputs(inputs)
        query = self.action_query(hidden) / math.sqrt(float(self.game_state_encoder.d_model))
        values = self.value_head(hidden).squeeze(-1)
        none_logits = self.none_blocker_head(hidden).squeeze(-1)
        may_logits = self.may_head(hidden).squeeze(-1)
        return _ForwardBatch(
            query=query,
            values=values,
            none_logits=none_logits,
            may_logits=may_logits,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
            hidden=hidden,
        )

    def _forward_batch_with_lstm_override(
        self,
        step_indices: Tensor,
        lstm_state: tuple[Tensor, Tensor],
    ) -> _ForwardBatch:
        """Run the batched forward but with caller-supplied per-step LSTM state.

        Supports issue 2 (per-policy recurrent recompute): the caller's
        ``lstm_state`` was generated by re-running the LSTM scan from h=0
        through the episode under *this* policy's parameters, and overrides
        the rollout buffer's stored ``lstm_h_in/c_in`` (which are the
        behavior policy's hidden states).
        """

        inputs, _stale_lstm_state = self._gather_from_rollout(step_indices)
        if not self.use_lstm:
            forward, _next_state = self._compute_forward(inputs)
            return forward
        dtype = self._compute_hidden_dtype()
        h_in, c_in = lstm_state
        h_in = h_in.to(dtype=dtype).contiguous()
        c_in = c_in.to(dtype=dtype).contiguous()
        forward, _next_state = self._compute_forward(inputs, h_in=h_in, c_in=c_in)
        return forward

    def _forward_native_batch(
        self,
        native_batch: NativeEncodedBatch,
        *,
        env_indices: list[int] | None = None,
    ) -> _ForwardBatch:
        inputs, lstm_state = self._gather_from_native(native_batch, env_indices=env_indices)
        h_in: Tensor | None = None
        c_in: Tensor | None = None
        if lstm_state is not None:
            h_in, c_in = lstm_state
        forward, next_state = self._compute_forward(inputs, h_in=h_in, c_in=c_in)
        if next_state is not None:
            if env_indices is None:
                raise ValueError("env_indices are required when scattering LSTM state")
            self._scatter_lstm_state(env_indices, next_state[0], next_state[1])
        return forward

    @staticmethod
    def _validate_slot_card_rows(slot_card_rows: Tensor, max_card_rows: int) -> None:
        bad = (slot_card_rows < 0) | (slot_card_rows >= max_card_rows)
        if not bad.any():
            return
        bad_pos = bad.nonzero(as_tuple=False)[0]
        batch = int(bad_pos[0].item())
        slot = int(bad_pos[1].item())
        row = int(slot_card_rows[batch, slot].item())
        raise ValueError(
            f"invalid slot card row: batch={batch} slot={slot} row={row} "
            f"bounds=[0, {max_card_rows})"
        )

    def _decision_logits(
        self,
        *,
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        query: Tensor,
        none_logits: Tensor,
    ) -> Tensor:
        return decision_logits_reference(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
            query=query,
            none_logits=none_logits,
            validate=self.validate,
        )

    def _decision_logits_reference(
        self,
        *,
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        query: Tensor,
        none_logits: Tensor,
    ) -> Tensor:
        return decision_logits_reference(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            option_vectors=option_vectors,
            target_vectors=target_vectors,
            query=query,
            none_logits=none_logits,
            validate=True,
        )

    @staticmethod
    def _flat_decision_distribution_impl(
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        query: Tensor,
        none_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return flat_decision_distribution_impl(
            step_positions,
            option_idx,
            target_idx,
            masks,
            uses_none,
            option_vectors,
            target_vectors,
            query,
            none_logits,
        )

    def _flat_decision_distribution(
        self,
        *,
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        option_vectors: Tensor,
        target_vectors: Tensor,
        query: Tensor,
        none_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return flat valid-choice logits plus grouped log-probs/entropies."""

        self._maybe_init_compiled_functions()
        return flat_decision_distribution_from_forward(
            ReplayScoringForward(
                values=query.new_zeros(query.shape[0]),
                option_vectors=option_vectors,
                target_vectors=target_vectors,
                none_logits=none_logits,
                may_logits=none_logits.new_zeros(none_logits.shape),
                hidden=query,
                query=query,
            ),
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            validate=self.validate,
            compiled_fn=self._compiled_flat_decision_distribution_impl,
        )

    @staticmethod
    def _validate_flat_scored_indices(
        *,
        scored_groups: Tensor,
        scored_cols: Tensor,
        scored_steps: Tensor,
        scored_option_idx: Tensor,
        scored_target_idx: Tensor,
        max_steps: int,
        max_options: int,
        max_targets: int,
    ) -> None:
        validate_flat_scored_indices(
            scored_groups=scored_groups,
            scored_cols=scored_cols,
            scored_steps=scored_steps,
            scored_option_idx=scored_option_idx,
            scored_target_idx=scored_target_idx,
            max_steps=max_steps,
            max_options=max_options,
            max_targets=max_targets,
        )

    @staticmethod
    def _validate_decision_indices(
        *,
        step_positions: Tensor,
        option_idx: Tensor,
        target_idx: Tensor,
        masks: Tensor,
        uses_none: Tensor,
        max_steps: int,
        max_options: int,
        max_targets: int,
    ) -> None:
        validate_decision_indices(
            step_positions=step_positions,
            option_idx=option_idx,
            target_idx=target_idx,
            masks=masks,
            uses_none=uses_none,
            max_steps=max_steps,
            max_options=max_options,
            max_targets=max_targets,
        )

    def _sample_flat_decisions(
        self,
        *,
        group_idx: Tensor,
        choice_cols: Tensor,
        flat_logits: Tensor,
        flat_log_probs: Tensor,
        deterministic: bool,
        finetune_eps: float = 0.0,
        finetune_n_disc: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Sample one valid choice per decision group.

        When ``finetune_eps > 0`` or ``finetune_n_disc > 0``, applies the
        R-NaD fine-tune / test-time projection (paper §197 / §279-282) to
        the per-group distribution before sampling. ``finetune_eps`` zeros
        entries below the threshold and renormalizes the survivors;
        ``finetune_n_disc`` quantizes survivor probabilities to multiples
        of ``1 / finetune_n_disc`` (rounded up, highest-probability first)
        and truncates once the running sum reaches 1. A group with no
        survivors falls back to the raw distribution.
        """

        device = flat_logits.device
        group_count = int(group_idx.max().item()) + 1
        choice_count = flat_logits.shape[0]
        flat_positions = torch.arange(choice_count, dtype=torch.long, device=device)

        counts = torch.bincount(group_idx, minlength=group_count)
        group_offsets = torch.zeros(group_count, dtype=torch.long, device=device)
        if group_count > 1:
            group_offsets[1:] = counts.cumsum(dim=0)[:-1]
        group_last = group_offsets + counts - 1

        finetune_active = (finetune_eps > 0.0 or finetune_n_disc > 0) and not deterministic
        if finetune_active:
            probs = flat_log_probs.exp()
            survivors = (
                probs >= finetune_eps
                if finetune_eps > 0.0
                else torch.ones_like(probs, dtype=torch.bool)
            )
            kept = torch.where(survivors, probs, torch.zeros_like(probs))
            group_sums = torch.zeros(group_count, dtype=probs.dtype, device=device)
            group_sums.scatter_add_(0, group_idx, kept)
            degenerate = group_sums <= 0.0
            per_choice_sum = group_sums[group_idx]
            safe_sum = torch.where(
                per_choice_sum > 0.0, per_choice_sum, torch.ones_like(per_choice_sum)
            )
            projected_probs = kept / safe_sum
            projected_probs = torch.where(degenerate[group_idx], probs, projected_probs)

            if finetune_n_disc > 0:
                # Quantize per group: highest-probability-first, round up to
                # the nearest 1/n, truncate any entry after cumulative >= 1.
                quantum = 1.0 / float(finetune_n_disc)
                # Sort within groups by sorting the flat tensor with a key
                # that groups first, then rank descending by prob within group.
                sort_key = group_idx.to(dtype=projected_probs.dtype) * 2.0 - projected_probs
                order = torch.argsort(sort_key, stable=True)
                sorted_probs = projected_probs[order]
                sorted_group = group_idx[order]
                # Ceil-quantize.
                rounded = torch.ceil(sorted_probs / quantum) * quantum
                # Per-group running cumsum.
                cumsum = torch.zeros_like(rounded)
                accum = torch.zeros(group_count, dtype=rounded.dtype, device=device)
                # Iterative cumsum per group — group sizes are small
                # (<= max_cached_choices), so this stays fast.
                for pos in range(rounded.shape[0]):
                    g = int(sorted_group[pos].item())
                    accum[g] = accum[g] + rounded[pos]
                    cumsum[pos] = accum[g]
                # Clamp the first entry that crosses 1.0 to fill the remainder,
                # zero subsequent entries.
                prev = cumsum - rounded
                crossed = cumsum > 1.0
                first_over = crossed & ~torch.roll(crossed, 1)
                first_over[0] = first_over[0] | (crossed[0] and True)
                rounded = torch.where(first_over, torch.clamp(1.0 - prev, min=0.0), rounded)
                rounded = torch.where(crossed & ~first_over, torch.zeros_like(rounded), rounded)
                # Restore to original order.
                restored = torch.zeros_like(projected_probs)
                restored[order] = rounded
                # Renormalize each group to absorb rounding drift.
                final_sums = torch.zeros(group_count, dtype=restored.dtype, device=device)
                final_sums.scatter_add_(0, group_idx, restored)
                per_group_sum = final_sums[group_idx]
                safe_group_sum = torch.where(
                    per_group_sum > 0.0, per_group_sum, torch.ones_like(per_group_sum)
                )
                projected_probs = torch.where(
                    per_group_sum > 0.0,
                    restored / safe_group_sum,
                    projected_probs,
                )

            flat_log_probs = torch.where(
                projected_probs > 0.0,
                projected_probs.log(),
                torch.full_like(projected_probs, float("-inf")),
            )

        if deterministic:
            group_max = torch.full(
                (group_count,),
                -torch.inf,
                dtype=flat_logits.dtype,
                device=device,
            )
            group_max.scatter_reduce_(0, group_idx, flat_logits, reduce="amax", include_self=True)
            sentinel = torch.full((choice_count,), choice_count, dtype=torch.long, device=device)
            first_best = torch.where(flat_logits == group_max[group_idx], flat_positions, sentinel)
            selected_flat = torch.full(
                (group_count,), choice_count, dtype=torch.long, device=device
            )
            selected_flat.scatter_reduce_(
                0,
                group_idx,
                first_best,
                reduce="amin",
                include_self=True,
            )
        else:
            probs = flat_log_probs.exp()
            flat_cumsum = probs.cumsum(dim=0)
            group_cumsum_offsets = torch.zeros(group_count, dtype=flat_logits.dtype, device=device)
            if group_count > 1:
                group_cumsum_offsets[1:] = flat_cumsum[group_last[:-1]]
            local_cumsum = flat_cumsum - group_cumsum_offsets[group_idx]
            thresholds = torch.rand(group_count, dtype=flat_logits.dtype, device=device)
            sentinel = torch.full((choice_count,), choice_count, dtype=torch.long, device=device)
            first_over = torch.where(
                local_cumsum >= thresholds[group_idx], flat_positions, sentinel
            )
            selected_flat = torch.full(
                (group_count,), choice_count, dtype=torch.long, device=device
            )
            selected_flat.scatter_reduce_(
                0,
                group_idx,
                first_over,
                reduce="amin",
                include_self=True,
            )
            selected_flat = torch.where(selected_flat == choice_count, group_last, selected_flat)

        selected_cols = choice_cols[selected_flat]
        selected_log_probs = flat_log_probs[selected_flat]
        return selected_cols, selected_log_probs

    def _decode_action(
        self,
        trace_kind: TraceKind,
        pending: PendingState,
        selected: list[int],
    ) -> tuple[ActionTrace, ActionRequest]:
        if trace_kind == "priority":
            selected_idx = selected[0]
            from magic_ai.actions import build_priority_candidates

            candidates_list = build_priority_candidates(
                pending, max_targets_per_option=self.max_targets_per_option
            )
            if not candidates_list:
                return ActionTrace("priority", indices=(0,)), cast(ActionRequest, {"kind": "pass"})
            selected_idx = min(selected_idx, len(candidates_list) - 1)
            return (
                ActionTrace("priority", indices=(selected_idx,)),
                action_from_priority_candidate(candidates_list[selected_idx]),
            )
        if trace_kind == "attackers":
            binary = tuple(float(v == 1) for v in selected)
            return (
                ActionTrace("attackers", binary=binary),
                action_from_attackers(pending, [value == 1.0 for value in binary]),
            )
        if trace_kind == "blockers":
            indices = tuple(v - 1 for v in selected)
            return (
                ActionTrace("blockers", indices=indices),
                action_from_blockers(pending, list(indices)),
            )
        if trace_kind == "choice_ids":
            selected_idx = selected[0]
            target_id = selected_option_id(pending, selected_idx)
            return (
                ActionTrace("choice_ids", indices=(selected_idx,)),
                action_from_choice_ids([target_id] if target_id else []),
            )
        if trace_kind == "choice_color":
            selected_idx = selected[0]
            options = pending.get("options", [])
            if 0 <= selected_idx < len(options):
                option = options[selected_idx]
                color = option.get("color", option.get("id", COLORS[selected_idx % len(COLORS)]))
            else:
                color = COLORS[selected_idx % len(COLORS)]
            return (
                ActionTrace("choice_color", indices=(selected_idx,)),
                action_from_choice_color(color),
            )
        selected_idx = selected[0]
        return (
            ActionTrace("choice_index", indices=(selected_idx,)),
            action_from_choice_index(selected_idx),
        )

    def _trace_action_without_pending(
        self,
        trace_kind: TraceKind,
        selected: list[int],
    ) -> tuple[ActionTrace, ActionRequest]:
        if trace_kind == "attackers":
            binary = tuple(float(v == 1) for v in selected)
            return ActionTrace("attackers", binary=binary), cast(ActionRequest, {})
        if trace_kind == "blockers":
            indices = tuple(v - 1 for v in selected)
            return ActionTrace("blockers", indices=indices), cast(ActionRequest, {})
        if trace_kind == "choice_color":
            selected_idx = selected[0]
            return ActionTrace("choice_color", indices=(selected_idx,)), cast(ActionRequest, {})
        if trace_kind == "choice_ids":
            selected_idx = selected[0]
            return ActionTrace("choice_ids", indices=(selected_idx,)), cast(ActionRequest, {})
        selected_idx = selected[0] if selected else 0
        return ActionTrace("priority", indices=(selected_idx,)), cast(ActionRequest, {})

    def _build_decision_layout(
        self,
        trace_kind: TraceKind,
        pending: PendingState,
        parsed: ParsedActionInputs,
    ) -> tuple[list[list[int]], list[list[int]], list[list[bool]], list[bool]]:
        """Return Python-side decision layout rows (padded to max_cached_choices).

        ``RolloutBuffer.ingest_batch`` flattens these into the buffer's
        decision storage in one torch.tensor call per field.
        """

        choices = self.max_cached_choices

        if trace_kind == "may":
            return [], [], [], []

        if trace_kind == "priority":
            candidates = parsed.priority_candidates[:choices]
            if not candidates:
                return [], [], [], []
            option_row = [-1] * choices
            target_row = [-1] * choices
            mask_row = [False] * choices
            for col, cand in enumerate(candidates):
                option_row[col] = cand.option_index
                if cand.target_index is not None:
                    target_row[col] = cand.target_index
                mask_row[col] = True
            return [option_row], [target_row], [mask_row], [False]

        option_count = min(parsed.num_present_options, self.max_options)

        if trace_kind == "attackers":
            if option_count == 0:
                return [], [], [], []
            option_idx_l: list[list[int]] = []
            target_idx_l: list[list[int]] = []
            mask_l: list[list[bool]] = []
            for i in range(option_count):
                option_row = [-1] * choices
                option_row[1] = i
                target_row = [-1] * choices
                mask_row = [False] * choices
                mask_row[0] = True
                mask_row[1] = True
                option_idx_l.append(option_row)
                target_idx_l.append(target_row)
                mask_l.append(mask_row)
            return option_idx_l, target_idx_l, mask_l, [True] * option_count

        if trace_kind == "blockers":
            options = pending.get("options", [])[: self.max_options]
            if not options:
                return [], [], [], []
            option_idx_l = []
            target_idx_l = []
            mask_l = []
            for i, option in enumerate(options):
                target_count = min(
                    len(option.get("valid_targets", [])), self.max_targets_per_option
                )
                option_row = [-1] * choices
                target_row = [-1] * choices
                mask_row = [False] * choices
                mask_row[0] = True  # col 0 will use none-blocker head
                for t in range(target_count):
                    col = t + 1
                    option_row[col] = i
                    target_row[col] = t
                    mask_row[col] = True
                option_idx_l.append(option_row)
                target_idx_l.append(target_row)
                mask_l.append(mask_row)
            return option_idx_l, target_idx_l, mask_l, [True] * len(options)

        # choice_index / choice_ids / choice_color
        if option_count == 0:
            return [], [], [], []
        option_row = [-1] * choices
        target_row = [-1] * choices
        mask_row = [False] * choices
        for i in range(option_count):
            option_row[i] = i
            mask_row[i] = True
        return [option_row], [target_row], [mask_row], [False]


@dataclass(frozen=True)
class _ForwardInputs:
    """Device-resident tensors consumed by the pure compute core."""

    slot_card_rows: Tensor
    slot_occupied: Tensor
    slot_tapped: Tensor
    game_info: Tensor
    pending_kind_id: Tensor
    option_kind_ids: Tensor
    option_scalars: Tensor
    option_mask: Tensor
    option_ref_slot_idx: Tensor
    option_ref_card_row: Tensor
    target_mask: Tensor
    target_type_ids: Tensor
    target_scalars: Tensor
    target_ref_slot_idx: Tensor
    target_ref_is_player: Tensor
    target_ref_is_self: Tensor


class _ForwardBatch(ReplayScoringForward):
    pass


@dataclass(frozen=True)
class _ReplayBatchExtras:
    """Cached intermediates from ``evaluate_replay_batch`` for reuse by SPR."""

    forward: _ForwardBatch
    step_indices: Tensor
    group_pos_t: Tensor
    group_idx_t: Tensor
    group_selected: Tensor
    group_uses_none: Tensor


def _trace_kind_for_pending(pending: PendingState) -> TraceKind:
    pending_kind = pending.get("kind", "")
    if pending_kind == "priority":
        return "priority"
    if pending_kind == "attackers":
        return "attackers"
    if pending_kind == "blockers":
        return "blockers"
    if pending_kind == "may":
        return "may"
    if pending_kind == "mana_color":
        return "choice_color"
    if pending_kind in {"cards_from_hand", "card_from_library", "permanent"}:
        return "choice_ids"
    return "choice_index"
