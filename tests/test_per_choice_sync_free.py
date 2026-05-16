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


class DecoderPerCellPointerParityTests(unittest.TestCase):
    """The per-cell pointer head must agree with the dense
    `[B, L, T_enc]` pointer head on per_row_log_pi / per_row_entropy /
    per_step_log_pi (within float tolerance). Verifies the segment
    log-softmax reduction matches the dense log-softmax over the same
    masked positions."""

    def test_per_cell_matches_dense_pointer_head(self) -> None:
        from typing import cast

        from magic_ai.text_encoder.decoder import GrammarDecoder, GrammarDecoderConfig
        from magic_ai.text_encoder.decoder_inference import decoder_score_replay
        from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE
        from magic_ai.text_encoder.policy import TextPolicy
        from magic_ai.text_encoder.replay_buffer import _build_decoder_cells

        torch.manual_seed(0xDEAD)
        b, t_enc, l_max, d_model = 2, 6, 4, 16
        cfg = GrammarDecoderConfig(
            d_model=d_model,
            n_layers=1,
            n_heads=2,
            max_decode_len=l_max,
            pointer_temperature=1.0,
            grammar_vocab_size=GRAMMAR_VOCAB_SIZE,
        )
        decoder = GrammarDecoder(cfg).eval()

        class _StubPolicy:
            def __init__(self, decoder):
                self.grammar_decoder = decoder

        encoded = torch.randn(b, t_enc, d_model)
        attn = torch.ones((b, t_enc), dtype=torch.bool)
        target_tokens = torch.randint(0, GRAMMAR_VOCAB_SIZE, (b, l_max))
        target_pointer_pos = torch.randint(0, t_enc, (b, l_max))
        is_pointer_step = torch.tensor([[False, True, True, False]] * b, dtype=torch.bool)
        pad_mask = torch.tensor([[True, True, True, False]] * b, dtype=torch.bool)
        vocab_mask = torch.ones((b, l_max, GRAMMAR_VOCAB_SIZE), dtype=torch.bool)
        # Force pointer_mask to be non-trivial so the per-cell segment
        # softmax has multiple legal entries per cell.
        pointer_mask = torch.zeros((b, l_max, t_enc), dtype=torch.bool)
        pointer_mask[:, 1, :4] = True
        pointer_mask[:, 2, 1:5] = True
        cells = _build_decoder_cells(
            pad_mask=pad_mask,
            is_pointer_step=is_pointer_step,
            vocab_mask=vocab_mask,
            pointer_mask=pointer_mask,
            target_tokens=target_tokens,
            target_pointer_pos=target_pointer_pos,
            output_log_prob=torch.zeros((b, l_max), dtype=torch.float32),
        )

        text_policy = _StubPolicy(decoder)

        with torch.no_grad():
            new_scores = decoder_score_replay(
                cast(TextPolicy, text_policy),
                encoded,
                attn,
                target_tokens,
                pad_mask,
                vocab_mask,
                cells,
            )

            # Reference: dense computation. Mirrors what
            # ``decoder_score_replay`` did before the per-cell rewrite.
            vocab_logits, pointer_logits = decoder.forward_teacher_forced(
                target_tokens.to(dtype=torch.long), encoded, attn
            )
            neg_inf = torch.finfo(vocab_logits.dtype).min
            v_logp = torch.log_softmax(vocab_logits.masked_fill(~vocab_mask, neg_inf), dim=-1)
            p_logp_dense = torch.log_softmax(
                pointer_logits.masked_fill(~pointer_mask, neg_inf), dim=-1
            )
            p_max = p_logp_dense.shape[-1] - 1
            target_tok = target_tokens.to(dtype=torch.long)
            target_ptr = target_pointer_pos.to(dtype=torch.long).clamp(min=0, max=max(p_max, 0))
            v_chosen = v_logp.gather(-1, target_tok.unsqueeze(-1)).squeeze(-1)
            p_chosen = p_logp_dense.gather(-1, target_ptr.unsqueeze(-1)).squeeze(-1)
            ref_step_logp = torch.where(is_pointer_step, p_chosen, v_chosen).where(
                pad_mask, torch.zeros_like(v_chosen)
            )
            ref_per_row_log_pi = ref_step_logp.sum(dim=-1)

            v_p = v_logp.exp()
            p_p = p_logp_dense.exp()
            v_ent = -(v_p * v_logp.where(v_p > 0, torch.zeros_like(v_logp))).sum(dim=-1)
            p_ent = -(p_p * p_logp_dense.where(p_p > 0, torch.zeros_like(p_logp_dense))).sum(dim=-1)
            ref_step_ent = torch.where(is_pointer_step, p_ent, v_ent).where(
                pad_mask, torch.zeros_like(v_ent)
            )
            ref_per_row_entropy = ref_step_ent.sum(dim=-1)

        torch.testing.assert_close(new_scores.per_step_log_pi, ref_step_logp, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            new_scores.per_row_log_pi, ref_per_row_log_pi, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            new_scores.per_row_entropy, ref_per_row_entropy, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
