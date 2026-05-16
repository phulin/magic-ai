# rust/

Rust utilities and benchmarks that support high-throughput data extraction and serialization experiments.

## Subpackages

- `forge_extract/` — Rust port of the Forge choice-situation parser that streams game logs into extracted JSONL records.
- `tensor_pickle_bench/` — Benchmark `pickled` vs `serde-pickle` when serializing tensor-shaped payloads with large byte buffers.
