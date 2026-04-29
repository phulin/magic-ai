"""Shard NativeBatchEncoder and NativeRolloutDriver calls across Python threads.

Cgo releases the GIL on entry, so N Python threads each calling MageEncodeBatch
or MageBatchStepByChoice on disjoint handles run in parallel native OS threads.
This module wraps the single-threaded drivers with a ThreadPoolExecutor-backed
fan-out, using one driver/encoder instance per worker so scratch buffers stay
thread-local.
"""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, cast

import torch

from magic_ai.native_encoder import NativeBatchEncoder, NativeEncodedBatch
from magic_ai.native_rollout import NativeRolloutDriver, NativeRolloutUnavailable


def _shard_ranges(n: int, workers: int) -> list[tuple[int, int]]:
    if workers <= 1 or n <= 1:
        return [(0, n)]
    workers = min(workers, n)
    base, rem = divmod(n, workers)
    out: list[tuple[int, int]] = []
    cursor = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        out.append((cursor, cursor + size))
        cursor += size
    return out


def _collect(futures: list[Any]) -> None:
    wait(futures)
    for f in futures:
        exc = f.exception()
        if exc is not None:
            raise exc


def _concat_encoded_batches(results: list[NativeEncodedBatch]) -> NativeEncodedBatch:
    decision_offsets: list[int] = [0]
    for r in results[:-1]:
        decision_offsets.append(decision_offsets[-1] + r.decision_rows_written)

    decision_start = torch.cat(
        [r.decision_start + off for r, off in zip(results, decision_offsets, strict=True)],
        dim=0,
    )

    def cat(name: str) -> torch.Tensor:
        return torch.cat([getattr(r, name) for r in results], dim=0)

    render_plan = None
    render_plan_lengths = None
    render_plan_overflow = None
    if all(r.render_plan is not None for r in results):
        render_plan = torch.cat([cast(torch.Tensor, r.render_plan) for r in results], dim=0)
        render_plan_lengths = torch.cat(
            [cast(torch.Tensor, r.render_plan_lengths) for r in results], dim=0
        )
        render_plan_overflow = torch.cat(
            [cast(torch.Tensor, r.render_plan_overflow) for r in results], dim=0
        )

    return NativeEncodedBatch(
        trace_kind_id=cat("trace_kind_id"),
        slot_card_rows=cat("slot_card_rows"),
        slot_occupied=cat("slot_occupied"),
        slot_tapped=cat("slot_tapped"),
        game_info=cat("game_info"),
        pending_kind_id=cat("pending_kind_id"),
        num_present_options=cat("num_present_options"),
        option_kind_ids=cat("option_kind_ids"),
        option_scalars=cat("option_scalars"),
        option_mask=cat("option_mask"),
        option_ref_slot_idx=cat("option_ref_slot_idx"),
        option_ref_card_row=cat("option_ref_card_row"),
        target_mask=cat("target_mask"),
        target_type_ids=cat("target_type_ids"),
        target_scalars=cat("target_scalars"),
        target_overflow=cat("target_overflow"),
        target_ref_slot_idx=cat("target_ref_slot_idx"),
        target_ref_is_player=cat("target_ref_is_player"),
        target_ref_is_self=cat("target_ref_is_self"),
        may_mask=cat("may_mask"),
        decision_start=decision_start,
        decision_count=cat("decision_count"),
        decision_option_idx=cat("decision_option_idx"),
        decision_target_idx=cat("decision_target_idx"),
        decision_mask=cat("decision_mask"),
        uses_none_head=cat("uses_none_head"),
        decision_rows_written=sum(r.decision_rows_written for r in results),
        pendings=[p for r in results for p in r.pendings],
        trace_kinds=[k for r in results for k in r.trace_kinds],
        render_plan=render_plan,
        render_plan_lengths=render_plan_lengths,
        render_plan_overflow=render_plan_overflow,
    )


class ShardedNativeBatchEncoder:
    """Fan `encode_handles` calls across a thread pool of NativeBatchEncoders."""

    def __init__(
        self,
        encoders: list[NativeBatchEncoder],
        pool: ThreadPoolExecutor | None,
    ) -> None:
        if not encoders:
            raise ValueError("ShardedNativeBatchEncoder requires at least one encoder")
        self._encoders = encoders
        self._pool = pool
        self.is_available = encoders[0].is_available

    @classmethod
    def for_policy(
        cls,
        policy: Any,
        *,
        workers: int,
        pool: ThreadPoolExecutor | None,
    ) -> ShardedNativeBatchEncoder:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        try:
            mage = importlib.import_module("mage")
            mage_any = cast(Any, mage)
            if mage_any._lib is None or mage_any._ffi is None:
                mage_any.load()
            lib = mage_any._lib
            ffi = mage_any._ffi
        except Exception:
            return cls([NativeBatchEncoder.for_policy(policy)], pool=None)

        encoders: list[NativeBatchEncoder] = []
        for i in range(workers):
            encoders.append(
                NativeBatchEncoder(
                    max_options=policy.max_options,
                    max_targets_per_option=policy.max_targets_per_option,
                    max_cached_choices=policy.max_cached_choices,
                    zone_slot_count=int(policy.rollout_buffer.slot_card_rows.shape[1]),
                    game_info_dim=int(policy.rollout_buffer.game_info.shape[1]),
                    option_scalar_dim=int(
                        policy.action_encoder.option_scalar_projection.in_features
                    ),
                    target_scalar_dim=int(
                        policy.action_encoder.target_scalar_projection.in_features
                    ),
                    lib=lib,
                    ffi=ffi,
                    # Card-row registration is process-global on the Go side;
                    # only the first worker needs to install it.
                    card_name_to_row=(
                        policy.game_state_encoder._card_name_to_row if i == 0 else None
                    ),
                    validate=policy.validate,
                )
            )
        return cls(encoders, pool=pool if workers > 1 else None)

    @classmethod
    def for_text(
        cls,
        *,
        max_options: int,
        max_targets_per_option: int,
        max_cached_choices: int,
        zone_slot_count: int,
        game_info_dim: int,
        option_scalar_dim: int,
        target_scalar_dim: int,
        card_name_to_row: dict[str, int],
        emit_render_plan: bool,
        render_plan_capacity: int,
        validate: bool,
        workers: int,
        pool: ThreadPoolExecutor | None,
        dedup_card_bodies: bool = False,
    ) -> ShardedNativeBatchEncoder:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        try:
            mage = importlib.import_module("mage")
            mage_any = cast(Any, mage)
            if mage_any._lib is None or mage_any._ffi is None:
                mage_any.load()
            lib = mage_any._lib
            ffi = mage_any._ffi
        except Exception:
            return cls(
                [
                    NativeBatchEncoder(
                        max_options=max_options,
                        max_targets_per_option=max_targets_per_option,
                        max_cached_choices=max_cached_choices,
                    )
                ],
                pool=None,
            )

        encoders: list[NativeBatchEncoder] = []
        for i in range(workers):
            encoders.append(
                NativeBatchEncoder(
                    max_options=max_options,
                    max_targets_per_option=max_targets_per_option,
                    max_cached_choices=max_cached_choices,
                    zone_slot_count=zone_slot_count,
                    game_info_dim=game_info_dim,
                    option_scalar_dim=option_scalar_dim,
                    target_scalar_dim=target_scalar_dim,
                    lib=lib,
                    ffi=ffi,
                    card_name_to_row=card_name_to_row if i == 0 else None,
                    validate=validate,
                    emit_render_plan=emit_render_plan,
                    render_plan_capacity=render_plan_capacity,
                    dedup_card_bodies=dedup_card_bodies,
                )
            )
        return cls(encoders, pool=pool if workers > 1 else None)

    def encode_tokens(
        self,
        games: list[Any],
        *,
        perspective_player_indices: list[int],
        max_tokens: int,
        max_options: int,
        max_targets: int,
        max_card_refs: int,
    ) -> tuple[NativeEncodedBatch, Any]:
        """Run the native text-encoder assembler (Phase 5 cutover).

        Single-shard path: delegates to the first encoder. Multi-shard
        sharding for ``encode_tokens`` is a follow-up; for now the rollout
        hot path stages a single encode-tokens call per ready batch.
        """

        from magic_ai.text_encoder.native_assembler import encode_tokens

        return encode_tokens(
            self._encoders[0],
            games,
            perspective_player_indices=perspective_player_indices,
            max_tokens=max_tokens,
            max_options=max_options,
            max_targets=max_targets,
            max_card_refs=max_card_refs,
        )

    def encode_handles(
        self,
        games: list[Any],
        *,
        perspective_player_indices: list[int],
    ) -> NativeEncodedBatch:
        n = len(games)
        if self._pool is None or n <= 1 or len(self._encoders) == 1:
            return self._encoders[0].encode_handles(
                games, perspective_player_indices=perspective_player_indices
            )

        shards = _shard_ranges(n, len(self._encoders))
        if len(shards) == 1:
            return self._encoders[0].encode_handles(
                games, perspective_player_indices=perspective_player_indices
            )

        results: list[NativeEncodedBatch | None] = [None] * len(shards)

        def run(idx: int, a: int, b: int) -> None:
            results[idx] = self._encoders[idx].encode_handles(
                games[a:b],
                perspective_player_indices=perspective_player_indices[a:b],
            )

        futures = [self._pool.submit(run, i, a, b) for i, (a, b) in enumerate(shards)]
        _collect(futures)
        return _concat_encoded_batches([cast(NativeEncodedBatch, r) for r in results])


class ShardedNativeRolloutDriver:
    """Fan `step_by_choice` calls across a thread pool of NativeRolloutDrivers."""

    def __init__(
        self,
        drivers: list[NativeRolloutDriver],
        pool: ThreadPoolExecutor | None,
    ) -> None:
        if not drivers:
            raise ValueError("ShardedNativeRolloutDriver requires at least one driver")
        self._drivers = drivers
        self._pool = pool

    @classmethod
    def for_mage(
        cls,
        mage: Any,
        *,
        workers: int,
        pool: ThreadPoolExecutor | None,
    ) -> ShardedNativeRolloutDriver:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        try:
            if mage._lib is None or mage._ffi is None:
                mage.load()
            lib = mage._lib
            ffi = mage._ffi
        except Exception as exc:
            raise NativeRolloutUnavailable("failed to load mage native library") from exc
        drivers = [NativeRolloutDriver(lib=lib, ffi=ffi) for _ in range(workers)]
        return cls(drivers, pool=pool if workers > 1 else None)

    def poll(
        self, games: list[Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # poll is ~0.4% of rollout; keep it single-threaded on the first driver.
        return self._drivers[0].poll(games)

    def step_by_choice(
        self,
        games: list[Any],
        *,
        decision_starts: list[int],
        decision_counts: list[int],
        selected_choice_cols: list[int],
        may_selected: list[int],
        max_options: int,
        max_targets_per_option: int,
    ) -> None:
        n = len(games)
        if self._pool is None or n <= 1 or len(self._drivers) == 1:
            self._drivers[0].step_by_choice(
                games,
                decision_starts=decision_starts,
                decision_counts=decision_counts,
                selected_choice_cols=selected_choice_cols,
                may_selected=may_selected,
                max_options=max_options,
                max_targets_per_option=max_targets_per_option,
            )
            return

        shards = _shard_ranges(n, len(self._drivers))
        if len(shards) == 1:
            self._drivers[0].step_by_choice(
                games,
                decision_starts=decision_starts,
                decision_counts=decision_counts,
                selected_choice_cols=selected_choice_cols,
                may_selected=may_selected,
                max_options=max_options,
                max_targets_per_option=max_targets_per_option,
            )
            return

        # Precompute the flat-selected offset for each shard start in one pass.
        cumulative_cols: list[int] = [0]
        for c in decision_counts:
            cumulative_cols.append(cumulative_cols[-1] + c)

        def run(idx: int, a: int, b: int) -> None:
            shard_counts = decision_counts[a:b]
            shard_starts: list[int] = []
            cursor = 0
            for c in shard_counts:
                shard_starts.append(cursor)
                cursor += c
            cols_start = cumulative_cols[a]
            cols_end = cumulative_cols[b]
            self._drivers[idx].step_by_choice(
                games[a:b],
                decision_starts=shard_starts,
                decision_counts=shard_counts,
                selected_choice_cols=selected_choice_cols[cols_start:cols_end],
                may_selected=may_selected[a:b],
                max_options=max_options,
                max_targets_per_option=max_targets_per_option,
            )

        futures = [self._pool.submit(run, i, a, b) for i, (a, b) in enumerate(shards)]
        _collect(futures)
