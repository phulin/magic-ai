"""Tests for ``scripts.train.NativeTextTrajectoryBuffer`` (decoder shape).

The buffer stages per-env decoder targets + encoded snapshots from inference
replies and commits whole episodes into the shared :class:`TextReplayBuffer`
via :meth:`TextReplayBuffer.commit_decoder_decision`. We exercise:

* ``stage_batch`` accumulates decoder rows per env without crossing wires;
* ``append_envs_to_replay_returning_tensor`` produces matching replay rows
  whose decoder targets equal what was staged.
"""

from __future__ import annotations

import unittest

import torch
from magic_ai.text_encoder.batch import PackedTextBatch
from magic_ai.text_encoder.decoder_batch import (
    NativeTextDecoderBatch,
    native_decoder_batch_from_sample,
)
from magic_ai.text_encoder.replay_buffer import TextReplayBuffer
from magic_ai.text_encoder.tokenizer import MAX_CARD_REFS
from scripts.train import NativeTextTrajectoryBuffer


def _make_replay_buffer(*, capacity: int = 16) -> TextReplayBuffer:
    return TextReplayBuffer(
        capacity=capacity,
        max_tokens=8,
        max_options=2,
        max_targets_per_option=1,
        max_decision_groups=1,
        max_cached_choices=1,
        max_decoder_len=4,
        max_anchors=2,
        max_blockers=0,
        max_attackers=0,
        device="cpu",
        use_triton_append=False,
        use_triton_gather=False,
    )


def _make_decoder_batch(b: int, *, l_max: int = 4, n_max: int = 2) -> NativeTextDecoderBatch:
    from magic_ai.text_encoder.decoder_batch import DecoderSampleOutput

    sample = DecoderSampleOutput(
        output_token_ids=torch.arange(b * l_max, dtype=torch.long).view(b, l_max) % 5,
        output_pointer_pos=torch.full((b, l_max), -1, dtype=torch.long),
        output_pointer_subjects=torch.zeros((b, l_max), dtype=torch.long),
        output_is_pointer=torch.zeros((b, l_max), dtype=torch.bool),
        output_pad_mask=torch.ones((b, l_max), dtype=torch.bool),
        log_probs=torch.full((b, l_max), -0.1),
        decision_type=torch.zeros((b,), dtype=torch.long),  # PRIORITY
        pointer_anchor_handles=torch.arange(b * n_max, dtype=torch.long).view(b, n_max),
        pointer_anchor_count=torch.full((b,), n_max, dtype=torch.long),
    )
    return native_decoder_batch_from_sample(sample, value=torch.zeros(b))


def _make_packed_row(token_count: int = 3) -> PackedTextBatch:
    n = 1
    rows_t = torch.tensor([token_count], dtype=torch.int32)
    cu = torch.tensor([0, token_count], dtype=torch.int32)
    state_pos = cu[:-1].clone()
    seq_id = torch.zeros(token_count, dtype=torch.int32)
    pos_in_seq = torch.arange(token_count, dtype=torch.int32)
    token_ids = torch.arange(token_count, dtype=torch.int32) + 1
    card_ref = torch.full((n, MAX_CARD_REFS), -1, dtype=torch.int32)
    pointer_anchor_positions = torch.full((n, 2), -1, dtype=torch.int32)
    pointer_anchor_kinds = torch.full((n, 2), -1, dtype=torch.int32)
    pointer_anchor_subjects = torch.full((n, 2), -1, dtype=torch.int32)
    pointer_anchor_handles = torch.full((n, 2), -1, dtype=torch.int32)
    return PackedTextBatch(
        token_ids=token_ids,
        seq_id=seq_id,
        pos_in_seq=pos_in_seq,
        cu_seqlens=cu,
        seq_lengths=rows_t,
        state_positions=state_pos,
        card_ref_positions=card_ref,
        seq_lengths_host=(token_count,),
        max_seqlen=token_count,
        spec_lens=torch.zeros(n, dtype=torch.int32),
        decision_type=torch.zeros(n, dtype=torch.int32),
        pointer_anchor_positions=pointer_anchor_positions,
        pointer_anchor_kinds=pointer_anchor_kinds,
        pointer_anchor_subjects=pointer_anchor_subjects,
        pointer_anchor_handles=pointer_anchor_handles,
        legal_edge_bitmap=None,
    )


class NativeTextTrajectoryBufferTest(unittest.TestCase):
    def test_stage_then_commit_writes_decoder_rows(self) -> None:
        replay = _make_replay_buffer()
        buf = NativeTextTrajectoryBuffer(replay, num_envs=2, max_steps=4)

        # Stage two consecutive env-steps for env 0.
        for _ in range(2):
            decoder = _make_decoder_batch(b=1)
            buf.stage_batch(
                env_indices=[0],
                decoder_batch=decoder,
                packed_rows=[_make_packed_row(token_count=3)],
                perspective_player_indices=[0],
            )
        # Stage one env-step for env 1.
        decoder = _make_decoder_batch(b=1)
        buf.stage_batch(
            env_indices=[1],
            decoder_batch=decoder,
            packed_rows=[_make_packed_row(token_count=2)],
            perspective_player_indices=[1],
        )
        self.assertEqual(buf.active_step_count(0), 2)
        self.assertEqual(buf.active_step_count(1), 1)

        rows, counts = buf.append_envs_to_replay_returning_tensor([0, 1], replay)
        self.assertEqual(int(rows.numel()), 3)
        self.assertEqual(counts.tolist(), [2, 1])
        # Buffer cleared.
        self.assertEqual(buf.active_step_count(0), 0)
        self.assertEqual(buf.active_step_count(1), 0)

        # Decoder targets stored at the returned rows match what we staged.
        # All staged rows had decision_type=0 (PRIORITY).
        rows_h = rows.tolist()
        for r in rows_h:
            self.assertEqual(int(replay.decoder.decision_type[r].item()), 0)
        # pointer_anchor_count should be 2 for each row.
        for r in rows_h:
            self.assertEqual(int(replay.decoder.pointer_anchor_count[r].item()), 2)

    def test_reset_env_drops_staged_rows(self) -> None:
        replay = _make_replay_buffer()
        buf = NativeTextTrajectoryBuffer(replay, num_envs=2, max_steps=4)
        decoder = _make_decoder_batch(b=1)
        buf.stage_batch(
            env_indices=[0],
            decoder_batch=decoder,
            packed_rows=[_make_packed_row()],
            perspective_player_indices=[0],
        )
        self.assertEqual(buf.active_step_count(0), 1)
        buf.reset_env(0)
        self.assertEqual(buf.active_step_count(0), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
