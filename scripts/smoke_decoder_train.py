"""End-to-end smoke for the decoder pipeline.

Builds a small GrammarDecoder, runs autoregressive sampling on a synthetic
PRIORITY-decision batch, decodes one of the sampled action layouts into an
:class:`ActionRequest` and asserts it picks a legal option index, then runs
one teacher-forced replay-scoring forward and an optimizer step to confirm
gradients flow through the decoder.

This script intentionally avoids the ModernBERT trunk (which downloads HF
weights) and the native engine; the goal is to prove the post-cutover
decoder sampling / scoring / action-decoding API runs end-to-end.

Run with::

    uv run scripts/smoke_decoder_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from magic_ai.game_state import PendingState  # noqa: E402
from magic_ai.text_encoder.decision_spec import AnchorKind, DecisionType  # noqa: E402
from magic_ai.text_encoder.decoder import GrammarDecoder, GrammarDecoderConfig  # noqa: E402
from magic_ai.text_encoder.decoder_action import decode_decoder_action  # noqa: E402
from magic_ai.text_encoder.decoder_batch import DecoderDecisionLayout  # noqa: E402
from magic_ai.text_encoder.decoder_inference import (  # noqa: E402
    decoder_sample,
    decoder_score_replay,
)
from magic_ai.text_encoder.grammar import GRAMMAR_VOCAB_SIZE  # noqa: E402
from magic_ai.text_encoder.policy import TextPolicy  # noqa: E402


class _StubTextPolicy(torch.nn.Module):
    """Bare wrapper exposing ``grammar_decoder`` so the sampler/scorer run."""

    def __init__(self, decoder: GrammarDecoder) -> None:
        super().__init__()
        self.grammar_decoder = decoder


def _build_priority_batch(
    b: int,
    t_enc: int,
    n_actions: int,
) -> tuple[torch.Tensor, ...]:
    decision_type = torch.full((b,), int(DecisionType.PRIORITY), dtype=torch.long)
    n_max = n_actions
    pos = torch.full((b, n_max), -1, dtype=torch.long)
    kinds = torch.full((b, n_max), -1, dtype=torch.long)
    subj = torch.full((b, n_max), -1, dtype=torch.long)
    handles = torch.full((b, n_max), -1, dtype=torch.long)
    for i in range(b):
        for j in range(n_actions):
            pos[i, j] = j + 1
            kinds[i, j] = int(AnchorKind.LEGAL_ACTION)
            subj[i, j] = j
            handles[i, j] = j  # handle == option index (engine option_index)
    attn_mask = torch.ones((b, t_enc), dtype=torch.bool)
    return decision_type, pos, kinds, subj, handles, attn_mask


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    cfg = GrammarDecoderConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        max_decode_len=16,
    )
    decoder = GrammarDecoder(cfg).to(device)
    text_policy = _StubTextPolicy(decoder).to(device)

    b, t_enc, n_actions = 2, 6, 3
    encoded = torch.randn(b, t_enc, cfg.d_model, device=device)
    dt, pos, kinds, subj, handles, attn_mask = _build_priority_batch(b, t_enc, n_actions)
    dt = dt.to(device)
    pos = pos.to(device)
    kinds = kinds.to(device)
    subj = subj.to(device)
    handles = handles.to(device)
    attn_mask = attn_mask.to(device)

    # ---- 1) Sample one decoder rollout per row.
    decoder.eval()
    with torch.no_grad():
        sample = decoder_sample(
            cast(TextPolicy, text_policy),
            encoded,
            attn_mask,
            dt,
            pos,
            kinds,
            subj,
            handles,
            max_decode_len=16,
            greedy=False,
        )
    assert sample.output_token_ids.shape == (b, 16)
    assert sample.output_pad_mask[:, 0].all(), "first decoder step should always be valid"

    # ---- 2) Translate the row 0 layout into an engine ActionRequest.
    options = [{"kind": "pass"}] + [
        {"kind": "play_land", "card_id": f"c{i}"} for i in range(n_actions - 1)
    ]
    pending = cast("PendingState", {"kind": "priority", "player_idx": 0, "options": options})
    layout = DecoderDecisionLayout(
        output_token_ids=sample.output_token_ids[0],
        output_pointer_pos=sample.output_pointer_pos[0],
        output_pointer_subjects=sample.output_pointer_subjects[0],
        output_is_pointer=sample.output_is_pointer[0],
        output_pad_mask=sample.output_pad_mask[0],
        decision_type=int(sample.decision_type[0].item()),
        pointer_anchor_handles=sample.pointer_anchor_handles[0],
        pointer_anchor_count=int(sample.pointer_anchor_count[0].item()),
    )
    action = decode_decoder_action(pending, layout)  # type: ignore[arg-type]
    chosen = int(action.get("option_index", 0))  # type: ignore[union-attr]
    legal_indices = list(range(len(options)))
    assert chosen in legal_indices, (
        f"decoder picked illegal option {chosen!r} (legal={legal_indices})"
    )
    print(
        f"sampled action: {action} (chose option {chosen}/{len(options) - 1})",
        flush=True,
    )

    # ---- 3) Teacher-forced replay scoring + optimizer step.
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    # Use the sampled output as the supervised target so the loss is well-defined.
    # Pointer-step rows store -1 in the vocab slot; the decoder's token embedding
    # expects ids in [0, V). Replace with PAD (0) for the embedding lookup; the
    # actual chosen pointer index lives in target_pointer_pos.
    target_tokens = sample.output_token_ids.clamp_min(0)
    target_pointer_pos = sample.output_pointer_pos.clamp_min(0)
    is_pointer_step = sample.output_is_pointer
    pad_mask = sample.output_pad_mask
    L = target_tokens.shape[1]
    vocab_mask = torch.ones((b, L, GRAMMAR_VOCAB_SIZE), dtype=torch.bool, device=device)
    pointer_mask = torch.ones((b, L, pos.shape[1]), dtype=torch.bool, device=device)
    # ``decoder_score_replay`` now consumes packed cells; build them on
    # host from the (fully True) grammar masks for the smoke run.
    from magic_ai.text_encoder.replay_buffer import _build_decoder_cells  # noqa: PLC0415

    cells = _build_decoder_cells(
        pad_mask=pad_mask.cpu(),
        is_pointer_step=is_pointer_step.cpu(),
        vocab_mask=vocab_mask.cpu(),
        pointer_mask=pointer_mask.cpu(),
        pointer_anchor_positions=pos.cpu(),
        target_tokens=target_tokens.cpu(),
        target_pointer_pos=target_pointer_pos.cpu(),
        output_log_prob=torch.zeros((b, L), dtype=torch.float32),
    ).to(device)

    scores = decoder_score_replay(
        cast(TextPolicy, text_policy),
        encoded,
        attn_mask,
        target_tokens,
        pad_mask,
        vocab_mask,
        cells,
    )
    loss = -scores.per_row_log_pi.mean()
    optimizer.zero_grad()
    loss.backward()
    grad_present = any(
        p.grad is not None and torch.any(p.grad.abs() > 0) for p in decoder.parameters()
    )
    assert grad_present, "no decoder parameter received a non-zero gradient"
    optimizer.step()
    print(
        f"teacher-forced loss={float(loss.item()):.4f}; gradient flow OK; device={device}",
        flush=True,
    )
    print("smoke OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
