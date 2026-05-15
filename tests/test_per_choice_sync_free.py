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


if __name__ == "__main__":
    unittest.main()
