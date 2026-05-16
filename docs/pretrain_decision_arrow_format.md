# Pretraining Decision Arrow Format

This document specifies the Arrow-based corpus format for Forge policy/value
pretraining decisions. It replaces monolithic `.jsonl.gz` interchange files with
typed, compressed, game-atomic Arrow IPC shards.

## Goals

- Keep extraction output compressed while avoiding gzip's sequential-only access
  pattern.
- Move high-cardinality metadata into typed columns instead of per-row JSON
  objects.
- Preserve game ordering for sequenced LSTM pretraining.
- Allow a second pass to convert extracted decisions into model-ready token
  arrays without reparsing JSONL.

## Layout

An Arrow corpus is a directory:

```text
data/forge_decisions_arrow/
  manifest.json
  game_index.arrow
  part-000000.arrow
  part-000001.arrow
  ...
```

Each `part-*.arrow` file is an Arrow IPC file. Writers should use Arrow IPC
buffer compression with Zstd when available. Shards must be game-atomic: all
records for a game are written to the same shard, and records for that game
remain in source order.

`game_index.arrow` is an Arrow IPC file with one row per written game. It lets
sequenced pretraining load game spans without scanning all data shards at
startup. Consumers may build and persist it on first use when reading an older
corpus, but new writers should emit it directly.

The Rust extractor currently writes the `extracted` schema below. A tokenizer
pass should read that schema and write the `tokenized` schema once state/spec
rendering and decoder target construction are materialized.

## Manifest

`manifest.json` is UTF-8 JSON with this shape:

```json
{
  "format": "forge_pretrain_decision_arrow",
  "format_version": 1,
  "stage": "extracted",
  "compression": "zstd",
  "game_index": "game_index.arrow",
  "shards": 12,
  "records_written": 123456,
  "shard_target_rows": 262144,
  "stats": {
    "games_seen": 1000,
    "games_written": 998,
    "candidates_seen": 57123,
    "written_priority": 50000,
    "written_attack": 1000,
    "written_block": 900,
    "written_may": 4000,
    "written_choose": 1223
  }
}
```

Consumers should reject unknown `format` values and unsupported
`format_version` values.

## Game Index Schema

`game_index.arrow` has schema metadata:

```json
{
  "format": "forge_pretrain_game_index_arrow",
  "format_version": "1"
}
```

| Column | Arrow type | Nullable | Description |
| --- | --- | --- | --- |
| `game_id` | `utf8` | no | Forge game id. |
| `shard_idx` | `uint32` | no | Zero-based `part-*.arrow` shard number. |
| `batch_idx` | `uint32` | no | Record batch number inside the shard; the Rust extractor currently writes one batch per shard. |
| `row_start` | `uint32` | no | First row for the game within that record batch. |
| `row_count` | `uint32` | no | Number of decision rows for the game. |

## Extracted Schema

The `extracted` schema is the normalized decision record emitted by
`rust/forge_extract`. It is intentionally close to the existing JSONL schema,
but the hot metadata fields are typed columns.

| Column | Arrow type | Nullable | Description |
| --- | --- | --- | --- |
| `format_version` | `uint16` | no | Extracted record schema version. Currently `2`, matching the historical JSON record. |
| `game_id` | `utf8` | no | Forge game id. |
| `archive_member` | `utf8` | no | Source member inside the Forge ZIP. |
| `kind_id` | `uint8` | no | Choice kind id: `0=priority`, `1=attack`, `2=block`, `3=may`, `4=choose`. |
| `candidate_index` | `uint32` | no | Candidate index within the source game. |
| `candidate_count` | `uint32` | no | Number of extractable candidates in the source game/window. |
| `source_seq` | `int64` | no | Source pre-choice event sequence number. |
| `target_seq` | `int64` | no | Target observed post-choice event sequence number. |
| `perspective_id` | `utf8` | no | Player id for the decision perspective. |
| `perspective_name` | `utf8` | no | Player display name for the decision perspective. |
| `winner_id` | `utf8` | yes | Winning player id, if known. |
| `winner_name` | `utf8` | yes | Winning player display name, if known. |
| `terminal_sign` | `float32` | no | `+1`, `-1`, or `0` from the perspective player. |
| `snapshot_json` | `large_utf8` | no | Normalized pre-choice snapshot JSON, including `pending`. |
| `observed_json` | `large_utf8` | no | Observed Forge choice/event JSON. |
| `outcome_players_json` | `large_utf8` | no | JSON array of final player summaries. |
| `outcome_extras_json` | `large_utf8` | no | JSON object for extra terminal metadata. |

The JSON columns are the remaining compatibility bridge. They keep the Rust
extractor parser-only while making the costly outer record parse columnar. A
tokenizer pass should treat these as source material and should not copy them
into the final model-ready corpus unless debugging requires it.

## Tokenized Schema

The `tokenized` schema is the intended direct model input. It is not emitted by
the Rust extractor yet.

| Column | Arrow type | Nullable | Description |
| --- | --- | --- | --- |
| `game_id_hash` | `uint64` | no | Stable hash of `game_id` for split assignment and grouping. |
| `game_index` | `uint32` | no | Dense game index within the corpus. |
| `source_seq` | `uint32` | no | Source sequence number, narrowed after validation. |
| `kind_id` | `uint8` | no | Same kind id as the extracted schema. |
| `decision_type` | `uint8` | no | Decoder decision type id. |
| `terminal_sign` | `float32` | no | Value target before optional discounting. |
| `state_tokens` | `large_list<uint16>` | no | Rendered state token ids. Use `uint32` if the tokenizer vocab can exceed `65535`. |
| `spec_tokens` | `large_list<uint16>` | no | Rendered decision-spec token ids appended to the encoder input. |
| `decoder_tokens` | `list<uint16>` | no | Teacher-forced grammar decoder token ids. |
| `decoder_pointer_subjects` | `list<int16>` | no | Pointer subject ids, `-1` for vocab steps. |
| `decoder_is_pointer` | `list<bool>` | no | True when the decoder step is a pointer step. |
| `card_ref_positions` | `fixed_size_list<int16, 64>` | no | State-token positions of card refs, `-1` when absent. |
| `anchor_kinds` | `list<uint8>` | no | Pointer-anchor kind per anchor. |
| `anchor_subjects` | `list<int16>` | no | Pointer-anchor subject index per anchor. |
| `anchor_positions` | `list<int16>` | no | Encoder token position per anchor. |

The tokenized schema should be written in game-atomic shards as well. Loaders
can then batch by complete games for sequenced pretraining or sample rows IID
for non-sequenced debugging.

## Compression And Sharding

Default writer settings:

- Arrow IPC file format, not stream format.
- Zstd IPC buffer compression.
- Target shard size: `262144` rows.
- Flush only between games, so shards may exceed the target by up to one game.
- Refuse to write into a directory that already contains `manifest.json` or
  `part-*.arrow`, to avoid mixing stale shards with a new corpus.

For very large corpora, prefer more medium-sized shards over a few huge files.
This keeps evaluation splits, retries, and distributed reads manageable.
