#!/usr/bin/env python3
"""Build per-card embeddings by pooling ``<card-ref:0>`` hidden states from the
text encoder on a minimal single-card snapshot.

This is the eval bridge for the text encoder, per
``docs/text_encoder_plan.md`` §7 step 6: it reuses the existing
``scripts/eval_card_embeddings.py`` triplet/synonym/cluster harness on
per-card hidden states.

Usage
-----
    uv run scripts/build_text_card_embeddings.py [--checkpoint PATH] \
        [--oracle data/card_oracle_embeddings.json] \
        [--output data/text_encoder_card_embeddings.json] \
        [--device cpu|cuda] [--batch-size 32] [--smoke]

If ``--checkpoint`` is omitted the script uses a freshly random-initialized
``TextStateEncoder`` — useful as a sanity check that the pipeline plumbs end
to end. With ``--smoke`` it processes only the first 10 cards and asserts
each yields a finite vector of the expected dim before writing the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magic_ai.game_state import GameStateSnapshot, PlayerState  # noqa: E402
from magic_ai.text_encoder.batch import collate, tokenize_snapshot  # noqa: E402
from magic_ai.text_encoder.model import (  # noqa: E402
    TextEncoderConfig,
    TextStateEncoder,
    gather_card_vectors,
)
from magic_ai.text_encoder.render import load_oracle_text, render_snapshot  # noqa: E402
from magic_ai.text_encoder.tokenizer import load_tokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = REPO_ROOT / "data" / "card_oracle_embeddings.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "text_encoder_card_embeddings.json"


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------


def make_single_card_snapshot(card_name: str, *, card_id: str = "c0") -> GameStateSnapshot:
    """Build a minimal snapshot with ``card_name`` alone on Self.Battlefield.

    Both players at 20 life, turn=1, step="Precombat Main", no actions, empty
    library/hand/graveyard/exile. This is the canonical input for per-card
    pooled embeddings: only one ``<card-ref:0>`` is emitted.
    """

    self_player: PlayerState = {
        "ID": "p0",
        "Name": "Self",
        "Life": 20,
        "Hand": [],
        "Graveyard": [],
        "Battlefield": [{"ID": card_id, "Name": card_name, "Tapped": False}],
        "LibraryCount": 60,
        "HandCount": 0,
        "GraveyardCount": 0,
    }
    opp_player: PlayerState = {
        "ID": "p1",
        "Name": "Opp",
        "Life": 20,
        "Hand": [],
        "Graveyard": [],
        "Battlefield": [],
        "LibraryCount": 60,
        "HandCount": 0,
        "GraveyardCount": 0,
    }
    snapshot: GameStateSnapshot = {
        "turn": 1,
        "active_player": "Self",
        "step": "Precombat Main",
        "players": [self_player, opp_player],
        "stack": [],
    }
    return snapshot


# ---------------------------------------------------------------------------
# Batched forward
# ---------------------------------------------------------------------------


def encode_card_batch(
    card_names: list[str],
    *,
    encoder: TextStateEncoder,
    tokenizer: Any,
    oracle: dict[str, Any],
    pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode ``card_names`` into a ``[N, d_model]`` tensor of pooled vectors."""

    examples = []
    for name in card_names:
        snap = make_single_card_snapshot(name)
        rendered = render_snapshot(snap, actions=[], oracle=oracle)
        examples.append(tokenize_snapshot(rendered, tokenizer))
    batch = collate(examples, pad_id=pad_id)

    batch_on_device = type(batch)(
        token_ids=batch.token_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        card_ref_positions=batch.card_ref_positions.to(device),
        seq_lengths=batch.seq_lengths.to(device),
    )

    with torch.no_grad():
        hidden = encoder(batch_on_device)  # [N, T, D]
        card_vecs, mask = gather_card_vectors(hidden, batch_on_device)  # [N, MAX_REFS, D]

    # Take K=0 (the only ref). Sanity-check the mask says it's present.
    if not bool(mask[:, 0].all().item()):
        missing = [card_names[i] for i in range(len(card_names)) if not bool(mask[i, 0].item())]
        raise RuntimeError(f"<card-ref:0> missing in tokenized batch for: {missing}")
    return card_vecs[:, 0, :].detach().cpu().float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="process only the first 10 cards and assert finiteness",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"loading tokenizer + oracle ({args.oracle})")
    tokenizer = load_tokenizer()
    oracle = load_oracle_text(args.oracle)
    pad_id_raw = tokenizer.convert_tokens_to_ids("<pad>")
    if isinstance(pad_id_raw, list):
        raise TypeError("convert_tokens_to_ids('<pad>') returned a list")
    pad_id = int(pad_id_raw)

    cfg = TextEncoderConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        pad_id=pad_id,
    )
    encoder = TextStateEncoder(cfg)
    if args.checkpoint is not None:
        print(f"loading checkpoint {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "encoder" in state:
            state = state["encoder"]
        encoder.load_state_dict(state, strict=False)
    else:
        print("no --checkpoint; using random init (sanity-check mode)")
    encoder.to(device).eval()

    payload = json.loads(args.oracle.read_text())
    source_cards: list[dict[str, Any]] = list(payload.get("cards", []))
    if args.smoke:
        source_cards = source_cards[:10]
    print(f"encoding {len(source_cards)} cards (batch={args.batch_size}, device={device})")

    out_cards: list[dict[str, Any]] = []
    total_cards = 0
    total_time = 0.0
    for batch_start in range(0, len(source_cards), args.batch_size):
        chunk = source_cards[batch_start : batch_start + args.batch_size]
        names = [c["name"] for c in chunk]
        t0 = time.perf_counter()
        vectors = encode_card_batch(
            names,
            encoder=encoder,
            tokenizer=tokenizer,
            oracle=cast(dict[str, Any], oracle),
            pad_id=pad_id,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        per_card = elapsed / max(1, len(chunk))
        total_time += elapsed
        total_cards += len(chunk)
        print(
            f"  [{batch_start:>4d}..{batch_start + len(chunk):>4d}] "
            f"{len(chunk)} cards in {elapsed * 1000:.1f} ms "
            f"({per_card * 1000:.1f} ms/card)"
        )

        for card_record, vec in zip(chunk, vectors, strict=True):
            if args.smoke:
                if vec.shape[0] != cfg.d_model:
                    raise AssertionError(f"unexpected dim {vec.shape[0]} != {cfg.d_model}")
                if not bool(torch.isfinite(vec).all().item()):
                    raise AssertionError(f"non-finite vector for {card_record['name']!r}")
            out_record = {
                "name": card_record["name"],
                "type_line": card_record.get("type_line"),
                "mana_cost": card_record.get("mana_cost"),
                "oracle_text": card_record.get("oracle_text"),
                "power_toughness": card_record.get("power_toughness"),
                "colors": card_record.get("colors"),
                "embedding": vec.tolist(),
            }
            out_cards.append(out_record)

    print(
        f"encoded {total_cards} cards in {total_time:.2f}s "
        f"({total_time / max(1, total_cards) * 1000:.1f} ms/card avg); "
        f"vector dim={cfg.d_model}"
    )

    output = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_oracle": str(args.oracle),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "encoder_config": {
                "vocab_size": cfg.vocab_size,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "d_ff": cfg.d_ff,
                "max_seq_len": cfg.max_seq_len,
            },
            "smoke": args.smoke,
            "card_count": len(out_cards),
        },
        "cards": out_cards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output) + "\n")
    print(f"wrote -> {args.output}")


if __name__ == "__main__":
    main()
