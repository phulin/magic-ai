"""Parity check: the host-side count formulation in
``TextReplayBuffer.gather`` matches the True-cell count that the original
``mask.nonzero()`` would have produced, and ``torch.nonzero_static`` with
that count returns the same indices as ``mask.nonzero(as_tuple=False)``.

This is what lets ``LSTMStatefulTextPolicy.evaluate_replay_batch_per_choice``
swap ``.nonzero()`` for ``.nonzero_static(size=N)`` and avoid three
data-dependent shape syncs per call.
"""

import unittest

import torch


def _counts_for(
    pad_mask: torch.Tensor,
    is_pointer_step: torch.Tensor,
    vocab_mask: torch.Tensor,
    pointer_mask: torch.Tensor,
) -> tuple[int, int, int]:
    """Mirror of the count computation in ``TextReplayBuffer.gather``.

    Sums the choice axis first so the [B, L, V] / [B, L, T_enc] AND
    intermediate is never materialized.
    """
    vocab_active_step = pad_mask & ~is_pointer_step  # [B, L]
    ptr_active_step = pad_mask & is_pointer_step  # [B, L]
    vocab_legal_per_step = vocab_mask.sum(dim=-1)  # [B, L]
    ptr_legal_per_step = pointer_mask.sum(dim=-1)
    n_active = int(pad_mask.sum().item())
    n_vocab = int((vocab_active_step * vocab_legal_per_step).sum().item())
    n_ptr = int((ptr_active_step * ptr_legal_per_step).sum().item())
    return n_active, n_vocab, n_ptr


class PerChoiceSyncFreeTests(unittest.TestCase):
    def test_counts_match_nonzero(self) -> None:
        rng = torch.Generator().manual_seed(0xCAFE)
        b, l_max, v_vocab, t_enc = 4, 6, 9, 11
        pad_mask = torch.rand(b, l_max, generator=rng) > 0.3
        is_pointer_step = torch.rand(b, l_max, generator=rng) > 0.5
        vocab_mask = torch.rand(b, l_max, v_vocab, generator=rng) > 0.7
        pointer_mask = torch.rand(b, l_max, t_enc, generator=rng) > 0.7

        n_active, n_vocab, n_ptr = _counts_for(pad_mask, is_pointer_step, vocab_mask, pointer_mask)

        # Reference: what the original ``.nonzero()`` paths would have
        # produced (these are the very masks the per-choice path builds).
        group_active = pad_mask
        vocab_cell_active = pad_mask.unsqueeze(-1) & (~is_pointer_step).unsqueeze(-1) & vocab_mask
        ptr_cell_active = pad_mask.unsqueeze(-1) & is_pointer_step.unsqueeze(-1) & pointer_mask

        self.assertEqual(n_active, int(group_active.nonzero(as_tuple=False).shape[0]))
        self.assertEqual(n_vocab, int(vocab_cell_active.nonzero(as_tuple=False).shape[0]))
        self.assertEqual(n_ptr, int(ptr_cell_active.nonzero(as_tuple=False).shape[0]))

    def test_nonzero_static_matches_nonzero(self) -> None:
        rng = torch.Generator().manual_seed(0xBEEF)
        b, l_max, v = 5, 7, 12
        mask = torch.rand(b, l_max, v, generator=rng) > 0.6
        n = int(mask.sum().item())
        expected = mask.nonzero(as_tuple=False)
        got = torch.nonzero_static(mask, size=n)
        self.assertEqual(got.shape, expected.shape)
        # Order is deterministic and identical for both ops.
        torch.testing.assert_close(got, expected)

    def test_counts_edge_zero(self) -> None:
        # All-False masks: every count should be zero, no error.
        b, l_max, v_vocab, t_enc = 3, 4, 5, 6
        pad_mask = torch.zeros(b, l_max, dtype=torch.bool)
        is_pointer_step = torch.zeros(b, l_max, dtype=torch.bool)
        vocab_mask = torch.zeros(b, l_max, v_vocab, dtype=torch.bool)
        pointer_mask = torch.zeros(b, l_max, t_enc, dtype=torch.bool)
        self.assertEqual(
            _counts_for(pad_mask, is_pointer_step, vocab_mask, pointer_mask),
            (0, 0, 0),
        )
        # ``nonzero_static(..., size=0)`` returns [0, ndim].
        empty = torch.nonzero_static(pad_mask, size=0)
        self.assertEqual(tuple(empty.shape), (0, 2))


class DecoderCellsParityTests(unittest.TestCase):
    """Validate that the packed-cell builder produces a representation
    equivalent to the dense-mask + nonzero_static path."""

    def test_cells_match_dense_nonzero(self) -> None:
        from magic_ai.text_encoder.replay_buffer import _build_decoder_cells

        rng = torch.Generator().manual_seed(0xFACE)
        b, l_max, v_vocab, t_enc = 3, 5, 8, 10
        pad_mask = torch.rand(b, l_max, generator=rng) > 0.3
        is_pointer_step = torch.rand(b, l_max, generator=rng) > 0.5
        vocab_mask = torch.rand(b, l_max, v_vocab, generator=rng) > 0.6
        pointer_mask = torch.rand(b, l_max, t_enc, generator=rng) > 0.6
        target_tokens = torch.randint(0, v_vocab, (b, l_max), generator=rng)
        target_pointer_pos = torch.randint(0, t_enc, (b, l_max), generator=rng)
        output_log_prob = torch.randn(b, l_max, generator=rng)

        cells = _build_decoder_cells(
            pad_mask=pad_mask,
            is_pointer_step=is_pointer_step,
            vocab_mask=vocab_mask,
            pointer_mask=pointer_mask,
            target_tokens=target_tokens,
            target_pointer_pos=target_pointer_pos,
            output_log_prob=output_log_prob,
        )

        # Reference: the (b, t, choice) tuples from the dense paths.
        v_active = pad_mask & ~is_pointer_step
        p_active = pad_mask & is_pointer_step
        v_dense = (v_active.unsqueeze(-1) & vocab_mask).nonzero(as_tuple=False)
        p_dense = (p_active.unsqueeze(-1) & pointer_mask).nonzero(as_tuple=False)
        # Cells' (b, t, choice) tuples must match (same nonzero order).
        v_cell_b_per_legal = cells.v_cell_b[cells.v_legal_cell_id.long()]
        v_cell_t_per_legal = cells.v_cell_t[cells.v_legal_cell_id.long()]
        v_got = torch.stack(
            [v_cell_b_per_legal.long(), v_cell_t_per_legal.long(), cells.v_legal_choice.long()],
            dim=1,
        )
        p_cell_b_per_legal = cells.p_cell_b[cells.p_legal_cell_id.long()]
        p_cell_t_per_legal = cells.p_cell_t[cells.p_legal_cell_id.long()]
        p_got = torch.stack(
            [p_cell_b_per_legal.long(), p_cell_t_per_legal.long(), cells.p_legal_choice.long()],
            dim=1,
        )
        torch.testing.assert_close(v_got, v_dense)
        torch.testing.assert_close(p_got, p_dense)

        # ``is_chosen`` must match elementwise the dense (choice == target)
        # condition at each cell.
        chosen_v_per_legal = target_tokens[v_cell_b_per_legal.long(), v_cell_t_per_legal.long()]
        self.assertTrue(
            torch.equal(cells.v_legal_is_chosen, cells.v_legal_choice == chosen_v_per_legal.int())
        )
        chosen_p_per_legal = target_pointer_pos[
            p_cell_b_per_legal.long(), p_cell_t_per_legal.long()
        ]
        self.assertTrue(
            torch.equal(cells.p_legal_is_chosen, cells.p_legal_choice == chosen_p_per_legal.int())
        )

        # Per-cell behavior log prob = output_log_prob[cell_b, cell_t].
        torch.testing.assert_close(
            cells.v_cell_behavior_log_prob,
            output_log_prob[cells.v_cell_b.long(), cells.v_cell_t.long()].float(),
        )
        torch.testing.assert_close(
            cells.p_cell_behavior_log_prob,
            output_log_prob[cells.p_cell_b.long(), cells.p_cell_t.long()].float(),
        )

    def test_cells_zero_when_no_active(self) -> None:
        from magic_ai.text_encoder.replay_buffer import _build_decoder_cells

        b, l_max, v_vocab, t_enc = 2, 3, 5, 6
        pad_mask = torch.zeros(b, l_max, dtype=torch.bool)
        is_pointer_step = torch.zeros(b, l_max, dtype=torch.bool)
        vocab_mask = torch.zeros(b, l_max, v_vocab, dtype=torch.bool)
        pointer_mask = torch.zeros(b, l_max, t_enc, dtype=torch.bool)
        target_tokens = torch.zeros(b, l_max, dtype=torch.long)
        target_pointer_pos = torch.zeros(b, l_max, dtype=torch.long)
        output_log_prob = torch.zeros(b, l_max, dtype=torch.float32)
        cells = _build_decoder_cells(
            pad_mask=pad_mask,
            is_pointer_step=is_pointer_step,
            vocab_mask=vocab_mask,
            pointer_mask=pointer_mask,
            target_tokens=target_tokens,
            target_pointer_pos=target_pointer_pos,
            output_log_prob=output_log_prob,
        )
        self.assertEqual(tuple(cells.v_cell_b.shape), (0,))
        self.assertEqual(tuple(cells.v_legal_choice.shape), (0,))
        self.assertEqual(tuple(cells.p_cell_b.shape), (0,))
        self.assertEqual(tuple(cells.p_legal_choice.shape), (0,))


if __name__ == "__main__":
    unittest.main()
