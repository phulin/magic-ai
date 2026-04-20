#!/usr/bin/env python3
"""
Download Magic card oracle text from Scryfall and build local embeddings.

The fixed card list is intentionally kept in this file so it can be versioned
with the experiment that consumes the embeddings.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRYFALL_API = "https://api.scryfall.com"
DEFAULT_SET_CODE = "jmp"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_EMBEDDING_DIMENSIONS = 4096
DEFAULT_MAX_LENGTH = 8192
DEFAULT_USER_AGENT = "magic-ai/0.1.0"
EMBEDDING_TEXT_FORMATS = ("json", "tagged", "plain")

# Edit this list for the project-local baseline cards that should always be
# included unless --no-fixed-list is passed.
FIXED_CARD_NAMES: list[str] = [
    "Black Lotus",
    "Counterspell",
    "Giant Growth",
    "Llanowar Elves",
    "Serra Angel",
]


class DownloadError(RuntimeError):
    """Raised when an upstream API request fails."""


def main() -> None:
    args = parse_args()
    payload = build_card_embedding_file(
        output=args.output,
        include_fixed_list=not args.no_fixed_list,
        include_jumpstart=args.include_jumpstart,
        additional_card_names=args.card,
        set_code=args.set_code,
        include_embeddings=not args.no_embeddings,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        user_agent=args.user_agent,
        embedding_text_format=args.embedding_text_format,
    )
    print(f"saved {payload['metadata']['card_count']} cards -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MTG oracle text and create local card embeddings."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/card_oracle_embeddings.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--include-jumpstart",
        action="store_true",
        help="include every print from original Jumpstart via Scryfall set code jmp",
    )
    parser.add_argument(
        "--set-code",
        default=DEFAULT_SET_CODE,
        help="Scryfall set code used by --include-jumpstart",
    )
    parser.add_argument(
        "--card",
        action="append",
        default=[],
        help='additional exact card name; repeat for multiple cards, e.g. --card Mountain --card "Lightning Bolt"',
    )
    parser.add_argument(
        "--no-fixed-list",
        action="store_true",
        help="skip the card names hard-coded in FIXED_CARD_NAMES",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="only download oracle text; skip local embedding generation",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="local Hugging Face embedding model",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        help="embedding vector length; Qwen3 supports 32 through 4096",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="number of cards per local embedding batch",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for local embeddings: auto, cuda, mps, or cpu",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="torch dtype for local embeddings: auto, bfloat16, float16, or float32",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="maximum tokenizer length for oracle text",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent sent to Scryfall",
    )
    parser.add_argument(
        "--embedding-text-format",
        choices=EMBEDDING_TEXT_FORMATS,
        default="json",
        help="format for the text passed to the embedding model",
    )
    return parser.parse_args()


def build_card_embedding_file(
    *,
    output: Path,
    include_fixed_list: bool,
    include_jumpstart: bool,
    additional_card_names: list[str],
    set_code: str,
    include_embeddings: bool,
    embedding_model: str,
    embedding_dimensions: int | None,
    batch_size: int,
    device: str,
    dtype: str,
    max_length: int,
    user_agent: str,
    embedding_text_format: str,
) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    source_urls: list[str] = []

    fixed_names = FIXED_CARD_NAMES if include_fixed_list else []
    requested_names = normalize_card_names([*fixed_names, *additional_card_names])
    for name in requested_names:
        cards.append(fetch_named_card(name, user_agent=user_agent))
        time.sleep(0.12)

    if include_jumpstart:
        jumpstart_cards = fetch_set_cards(set_code, user_agent=user_agent)
        cards.extend(jumpstart_cards)
        source_urls.append(scryfall_search_url(set_code))

    if not cards:
        raise ValueError(
            "no cards requested; use --include-jumpstart, --card, or populate FIXED_CARD_NAMES"
        )

    records = [
        card_to_record(card, embedding_text_format=embedding_text_format)
        for card in dedupe_cards(cards)
    ]
    if include_embeddings:
        add_transformers_embeddings(
            records,
            model_name=embedding_model,
            dimensions=embedding_dimensions,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
    else:
        for record in records:
            record["embedding"] = None

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "Scryfall API",
            "source_urls": source_urls,
            "fixed_card_names": FIXED_CARD_NAMES if include_fixed_list else [],
            "additional_card_names": additional_card_names,
            "jumpstart_included": include_jumpstart,
            "jumpstart_set_code": set_code if include_jumpstart else None,
            "card_count": len(records),
            "embeddings_included": include_embeddings,
            "embedding_backend": "transformers" if include_embeddings else None,
            "embedding_model": embedding_model if include_embeddings else None,
            "embedding_dimensions": embedding_dimensions if include_embeddings else None,
            "embedding_normalized": include_embeddings,
            "embedding_max_length": max_length if include_embeddings else None,
            "embedding_device": device if include_embeddings else None,
            "embedding_dtype": dtype if include_embeddings else None,
            "embedding_text_format": embedding_text_format,
        },
        "cards": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def normalize_card_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_name in names:
        name = " ".join(raw_name.split())
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)
    return normalized


def fetch_named_card(name: str, *, user_agent: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"exact": name})
    url = f"{SCRYFALL_API}/cards/named?{query}"
    return get_json(url, user_agent=user_agent)


def fetch_set_cards(set_code: str, *, user_agent: str) -> list[dict[str, Any]]:
    url = scryfall_search_url(set_code)
    cards: list[dict[str, Any]] = []

    while url:
        page = get_json(url, user_agent=user_agent)
        cards.extend(page.get("data", []))
        url = page.get("next_page") if page.get("has_more") else None
        if url:
            # Scryfall asks clients to stay under 10 requests per second.
            time.sleep(0.12)

    cards.sort(key=lambda card: _collector_sort_key(card.get("collector_number", "")))
    return cards


def scryfall_search_url(set_code: str) -> str:
    query = urllib.parse.urlencode(
        {
            "q": f"e:{set_code}",
            "unique": "prints",
            "order": "set",
        }
    )
    return f"{SCRYFALL_API}/cards/search?{query}"


def dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in cards:
        key = card.get("oracle_id") or card.get("id") or card.get("name", "").casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped


def card_to_record(
    card: dict[str, Any],
    *,
    embedding_text_format: str,
) -> dict[str, Any]:
    oracle_text = extract_oracle_text(card)
    name = card.get("name", "")
    type_line = card.get("type_line", "")
    mana_cost = card.get("mana_cost", "")
    power_toughness = extract_power_toughness(card)
    embedding_text = build_embedding_text(
        name=name,
        type_line=type_line,
        mana_cost=mana_cost,
        oracle_text=oracle_text,
        power_toughness=power_toughness,
        text_format=embedding_text_format,
    )

    return {
        "name": name,
        "scryfall_id": card.get("id"),
        "oracle_id": card.get("oracle_id"),
        "set": card.get("set"),
        "collector_number": card.get("collector_number"),
        "rarity": card.get("rarity"),
        "mana_cost": mana_cost,
        "type_line": type_line,
        "power_toughness": power_toughness,
        "colors": card.get("colors", []),
        "color_identity": card.get("color_identity", []),
        "oracle_text": oracle_text,
        "embedding_text": embedding_text,
    }


def build_embedding_text(
    *,
    name: str,
    type_line: str,
    mana_cost: str,
    oracle_text: str,
    power_toughness: str | None,
    text_format: str,
) -> str:
    masked_oracle_text = mask_card_name(oracle_text, name)
    embedding_oracle_text = strip_parenthetical_text(masked_oracle_text)
    embedding_type_line = format_type_line_for_embedding(type_line)
    if text_format in {"tagged", "plain"}:
        return build_plain_embedding_text(
            type_line=embedding_type_line,
            mana_cost=mana_cost,
            oracle_text=embedding_oracle_text,
            power_toughness=power_toughness,
            wrap_in_tag=text_format == "tagged",
            include_cardname_line=text_format == "plain",
        )
    if text_format != "json":
        raise ValueError(f"unknown embedding text format: {text_format}")
    return json.dumps(
        {
            "object": "Magic: the Gathering card",
            "type_line": embedding_type_line,
            "mana_cost": mana_cost,
            "oracle_text": embedding_oracle_text,
            "power_toughness": power_toughness,
        },
        ensure_ascii=False,
        sort_keys=False,
    )


def format_type_line_for_embedding(type_line: str) -> str:
    left, separator, right = type_line.partition(" — ")
    if not separator:
        left, separator, right = type_line.partition(" - ")
    if not separator:
        return type_line

    subtypes = " ".join(right.split())
    if not subtypes:
        return left
    return f"{left} (subtype {subtypes})"


def build_plain_embedding_text(
    *,
    type_line: str,
    mana_cost: str,
    oracle_text: str,
    power_toughness: str | None,
    wrap_in_tag: bool,
    include_cardname_line: bool,
) -> str:
    header = " ".join(part for part in (type_line, mana_cost) if part)
    body_parts = [part for part in (oracle_text, power_toughness) if part]
    body = "\n".join(body_parts)
    if header and body:
        content = f"{header}\n{body}"
    else:
        content = header or body
    if include_cardname_line:
        content = f"CARDNAME\n{content}" if content else "CARDNAME"
    if not wrap_in_tag:
        return content
    return f"<magic-the-gathering-card>\n{content}\n</magic-the-gathering-card>"


def mask_card_name(text: str, name: str) -> str:
    names = [part.strip() for part in name.split("//")]
    names.append(name)
    for candidate in sorted(set(names), key=len, reverse=True):
        if candidate:
            text = text.replace(candidate, "CARDNAME")
    return text


def strip_parenthetical_text(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\s*\([^()]*\)", "", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_power_toughness(card: dict[str, Any]) -> str | None:
    power = card.get("power")
    toughness = card.get("toughness")
    if power is not None and toughness is not None:
        return f"{power}/{toughness}"

    faces = card.get("card_faces") or []
    face_values: list[str] = []
    for face in faces:
        face_power = face.get("power")
        face_toughness = face.get("toughness")
        if face_power is None or face_toughness is None:
            continue
        face_name = face.get("name")
        value = f"{face_power}/{face_toughness}"
        face_values.append(f"{face_name}: {value}" if face_name else value)

    return "; ".join(face_values) if face_values else None


def extract_oracle_text(card: dict[str, Any]) -> str:
    oracle_text = card.get("oracle_text")
    if oracle_text:
        return oracle_text

    faces = card.get("card_faces") or []
    face_texts: list[str] = []
    for face in faces:
        face_oracle = face.get("oracle_text") or ""
        if not face_oracle:
            continue
        face_name = face.get("name") or ""
        face_texts.append(f"{face_name}\n{face_oracle}" if face_name else face_oracle)
    return "\n\n".join(face_texts)


def add_transformers_embeddings(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    dimensions: int | None,
    batch_size: int,
    device: str,
    dtype: str,
    max_length: int,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if dimensions is not None and not 32 <= dimensions <= 4096:
        raise ValueError("embedding_dimensions must be between 32 and 4096")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require transformers and torch. "
            "Install the project dependencies, then rerun this command."
        ) from exc

    resolved_device = resolve_torch_device(device, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model_kwargs: dict[str, Any] = {"torch_dtype": resolve_torch_dtype(dtype, torch)}
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.to(resolved_device)
    model.eval()

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        inputs = [record["embedding_text"] for record in batch]
        encoded = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            if dimensions is not None:
                embeddings = embeddings[:, :dimensions]
                embeddings = F.normalize(embeddings, p=2, dim=1)
        for record, embedding in zip(batch, embeddings.cpu().tolist()):
            record["embedding"] = embedding


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


def resolve_torch_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(dtype: str, torch: Any) -> Any:
    if dtype == "auto":
        return "auto"
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype: {dtype}") from exc


def get_json(url: str, *, user_agent: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json;q=0.9,*/*;q=0.8",
            "User-Agent": user_agent,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DownloadError(f"{request.full_url} failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"{request.full_url} failed: {exc.reason}") from exc


def _collector_sort_key(value: str) -> tuple[int, str]:
    digits = ""
    suffix = ""
    for char in value:
        if char.isdigit() and not suffix:
            digits += char
        else:
            suffix += char
    return (int(digits) if digits else 0, suffix)


if __name__ == "__main__":
    main()
