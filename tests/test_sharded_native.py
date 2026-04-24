from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, cast

import torch
from magic_ai.native_encoder import NativeEncodedBatch
from magic_ai.native_rollout import NativeRolloutDriver
from magic_ai.sharded_native import (
    ShardedNativeRolloutDriver,
    _concat_encoded_batches,
    _shard_ranges,
)


def _make_batch(batch_size: int, decision_rows: int, *, start_offset: int) -> NativeEncodedBatch:
    max_options = 3
    max_targets = 2
    option_scalar_dim = 4
    target_scalar_dim = 2
    slots = 5
    game_info_dim = 6
    max_cached = 2

    def zeros(*shape: int, dtype: torch.dtype = torch.int64) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    # decision_start values mimic what the encoder emits: per-batch-row offsets
    # into the shard's local decision rows, before concat shifts them.
    decision_start = torch.arange(batch_size, dtype=torch.int64)
    decision_count = torch.ones((batch_size,), dtype=torch.int64)
    # Ensure decision_rows_written >= sum of counts
    assert decision_rows >= int(decision_count.sum().item())

    return NativeEncodedBatch(
        trace_kind_id=zeros(batch_size) + start_offset,
        slot_card_rows=zeros(batch_size, slots),
        slot_occupied=zeros(batch_size, slots, dtype=torch.float32),
        slot_tapped=zeros(batch_size, slots, dtype=torch.float32),
        game_info=zeros(batch_size, game_info_dim, dtype=torch.float32),
        pending_kind_id=zeros(batch_size),
        num_present_options=zeros(batch_size),
        option_kind_ids=zeros(batch_size, max_options),
        option_scalars=zeros(batch_size, max_options, option_scalar_dim, dtype=torch.float32),
        option_mask=zeros(batch_size, max_options, dtype=torch.float32),
        option_ref_slot_idx=zeros(batch_size, max_options),
        option_ref_card_row=zeros(batch_size, max_options),
        target_mask=zeros(batch_size, max_options, max_targets, dtype=torch.float32),
        target_type_ids=zeros(batch_size, max_options, max_targets),
        target_scalars=zeros(
            batch_size, max_options, max_targets, target_scalar_dim, dtype=torch.float32
        ),
        target_overflow=zeros(batch_size, max_options, dtype=torch.float32),
        target_ref_slot_idx=zeros(batch_size, max_options, max_targets),
        target_ref_is_player=zeros(batch_size, max_options, max_targets, dtype=torch.uint8),
        target_ref_is_self=zeros(batch_size, max_options, max_targets, dtype=torch.uint8),
        may_mask=zeros(batch_size, dtype=torch.uint8),
        decision_start=decision_start,
        decision_count=decision_count,
        decision_option_idx=zeros(decision_rows, max_cached) + start_offset,
        decision_target_idx=zeros(decision_rows, max_cached),
        decision_mask=zeros(decision_rows, max_cached, dtype=torch.uint8),
        uses_none_head=zeros(decision_rows, dtype=torch.uint8),
        decision_rows_written=decision_rows,
        pendings=[],
        trace_kinds=[f"kind{start_offset}"] * batch_size,
    )


@dataclass
class _RecordedStep:
    games: list[Any]
    decision_starts: list[int]
    decision_counts: list[int]
    selected_choice_cols: list[int]
    may_selected: list[int]


class _RecordingDriver:
    def __init__(self) -> None:
        self.calls: list[_RecordedStep] = []

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
        self.calls.append(
            _RecordedStep(
                games=list(games),
                decision_starts=list(decision_starts),
                decision_counts=list(decision_counts),
                selected_choice_cols=list(selected_choice_cols),
                may_selected=list(may_selected),
            )
        )


class ShardRangesTests(unittest.TestCase):
    def test_degenerate(self) -> None:
        self.assertEqual(_shard_ranges(0, 4), [(0, 0)])
        self.assertEqual(_shard_ranges(1, 4), [(0, 1)])
        self.assertEqual(_shard_ranges(5, 1), [(0, 5)])

    def test_even_split(self) -> None:
        self.assertEqual(_shard_ranges(8, 4), [(0, 2), (2, 4), (4, 6), (6, 8)])

    def test_uneven_split_front_loads_remainder(self) -> None:
        self.assertEqual(_shard_ranges(7, 3), [(0, 3), (3, 5), (5, 7)])

    def test_more_workers_than_items(self) -> None:
        self.assertEqual(_shard_ranges(3, 8), [(0, 1), (1, 2), (2, 3)])


class ConcatEncodedBatchesTests(unittest.TestCase):
    def test_concat_offsets_decision_start_across_shards(self) -> None:
        a = _make_batch(batch_size=2, decision_rows=2, start_offset=1)
        b = _make_batch(batch_size=3, decision_rows=3, start_offset=2)

        merged = _concat_encoded_batches([a, b])

        self.assertEqual(merged.trace_kind_id.shape[0], 5)
        self.assertEqual(merged.decision_rows_written, 5)
        self.assertEqual(merged.decision_option_idx.shape[0], 5)
        # Shard A's decision_start is untouched; shard B's values shift by 2.
        torch.testing.assert_close(
            merged.decision_start,
            torch.tensor([0, 1, 0 + 2, 1 + 2, 2 + 2], dtype=torch.int64),
        )
        # Per-batch tensors retain per-shard values.
        torch.testing.assert_close(
            merged.trace_kind_id,
            torch.tensor([1, 1, 2, 2, 2], dtype=torch.int64),
        )
        # Per-decision tensors concatenate in order.
        torch.testing.assert_close(
            merged.decision_option_idx[:, 0],
            torch.tensor([1, 1, 2, 2, 2], dtype=torch.int64),
        )
        self.assertEqual(merged.trace_kinds, ["kind1", "kind1", "kind2", "kind2", "kind2"])


class StepByChoiceShardArgSplittingTests(unittest.TestCase):
    def test_shards_reslice_selected_cols_and_reset_starts(self) -> None:
        drivers = [_RecordingDriver(), _RecordingDriver()]
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=2)
        try:
            sharded = ShardedNativeRolloutDriver(
                drivers=cast(list[NativeRolloutDriver], drivers), pool=pool
            )
            games = ["g0", "g1", "g2", "g3"]
            # Flat selected layout: g0 uses cols [10], g1 [20,21], g2 [30], g3 [40,41,42]
            decision_counts = [1, 2, 1, 3]
            decision_starts = [0, 1, 3, 4]
            selected_choice_cols = [10, 20, 21, 30, 40, 41, 42]
            may_selected = [0, 1, 0, 1]

            sharded.step_by_choice(
                games,
                decision_starts=decision_starts,
                decision_counts=decision_counts,
                selected_choice_cols=selected_choice_cols,
                may_selected=may_selected,
                max_options=5,
                max_targets_per_option=3,
            )
        finally:
            pool.shutdown(wait=True)

        # Shard 0: games[0:2] → counts [1,2], starts [0,1], cols [10,20,21]
        self.assertEqual(drivers[0].calls[0].games, ["g0", "g1"])
        self.assertEqual(drivers[0].calls[0].decision_counts, [1, 2])
        self.assertEqual(drivers[0].calls[0].decision_starts, [0, 1])
        self.assertEqual(drivers[0].calls[0].selected_choice_cols, [10, 20, 21])
        self.assertEqual(drivers[0].calls[0].may_selected, [0, 1])

        # Shard 1: games[2:4] → counts [1,3], starts [0,1], cols [30,40,41,42]
        self.assertEqual(drivers[1].calls[0].games, ["g2", "g3"])
        self.assertEqual(drivers[1].calls[0].decision_counts, [1, 3])
        self.assertEqual(drivers[1].calls[0].decision_starts, [0, 1])
        self.assertEqual(drivers[1].calls[0].selected_choice_cols, [30, 40, 41, 42])
        self.assertEqual(drivers[1].calls[0].may_selected, [0, 1])

    def test_single_worker_path_passes_through(self) -> None:
        driver = _RecordingDriver()
        sharded = ShardedNativeRolloutDriver(
            drivers=cast(list[NativeRolloutDriver], [driver]), pool=None
        )
        sharded.step_by_choice(
            ["g0", "g1"],
            decision_starts=[0, 1],
            decision_counts=[1, 1],
            selected_choice_cols=[7, 8],
            may_selected=[0, 0],
            max_options=4,
            max_targets_per_option=2,
        )
        self.assertEqual(len(driver.calls), 1)
        self.assertEqual(driver.calls[0].games, ["g0", "g1"])
        self.assertEqual(driver.calls[0].decision_starts, [0, 1])


if __name__ == "__main__":
    unittest.main()
