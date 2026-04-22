# mage-go -> PyTorch Marshaling Redesign

## Goal

Replace the current hot path:

`Go engine -> JSON string -> cffi -> Python dict/list -> Python parsing -> torch.tensor(...)`

with a batched native feature export path:

`Go engine -> flat native buffers -> Python-owned torch CPU tensors -> optional async H2D copy -> model forward`

The profiling result already tells us the bottleneck is marshaling, not model math. This design therefore focuses on removing JSON serialization, Python object materialization, and repeated Python-side tensor construction.

## Current State

Today the integration boundary is:

- `../mage-go/cmd/pylib/main.go`
  - exports `MageNewGame`, `MageState`, `MageLegal`, `MageStep`
  - all state/legal responses are JSON strings
- `../mage-go/cmd/pylib/mage/__init__.py`
  - loads `libmage.so` / `.dylib` / `.dll` via `cffi`
  - calls `json.loads(...)` for every state/legal response
- `magic_ai/game_state.py`
  - walks nested Python dicts and populates tensors
- `magic_ai/actions.py`
  - does the same for legal actions / pending state
- `magic_ai/model.py`
  - still builds decision layouts in Python

This pays for the same information several times:

1. Go allocates and serializes strings.
2. Python decodes bytes and allocates dict/list/string objects.
3. Python reinterprets those objects into numeric feature arrays.
4. PyTorch allocates tensors from those arrays.

## Design Principles

1. Python should own tensor memory.
2. Native code should fill preallocated numeric buffers in place.
3. The exported ABI must be plain C, not C++.
4. The first redesign should keep the Go engine intact.
5. The design should support a later torch C++ extension without changing the Go-side feature ABI.

## Non-Goals

- Rewriting the game engine in C++
- Making Go directly create `torch.Tensor` objects
- Changing PPO/model semantics in the first phase
- Eliminating all Python from the rollout loop

## Proposed Architecture

### Phase 1: Native Feature Export ABI

Keep the Go shared library, but add a second API family beside the JSON API:

- game lifecycle and stepping stay in Go
- feature extraction becomes a native write-into-buffer operation
- Python allocates flat tensors and passes raw pointers to the shared library

The new path is:

1. Python maintains a batch of active game handles.
2. Python allocates reusable CPU tensors for encoded state/action features.
3. Python calls `MageEncodeBatch(...)` with:
   - game handles
   - perspective player indices
   - raw output pointers
   - capacity/shape metadata
4. Go writes integer/float/bool features directly into those buffers.
5. Python uses those tensors immediately, or copies them to GPU in one shot.

### Phase 2: Native Decision Layout Export

Move decision layout generation out of Python too:

- `trace_kind_id`
- `decision_start`
- `decision_count`
- `decision_option_idx`
- `decision_target_idx`
- `decision_mask`
- `uses_none_head`

This removes the remaining Python loop-heavy layout code in `PPOPolicy.parse_inputs_batch`.

### Phase 3: Optional Torch C++ Extension

If CPU->GPU transfer or Python call overhead is still meaningful after phases 1 and 2:

- add a small PyTorch C++ extension
- let it allocate/fill CPU tensors or pinned tensors
- optionally call into the Go `.so` directly through the same C ABI

This is explicitly a follow-on step, not the first migration.

## Memory Ownership Model

Python owns all output tensor memory.

That means:

- Python allocates reusable `torch.Tensor` objects on CPU
- Python passes raw data pointers into the native call
- Go writes into that memory only for the duration of the call
- Go does not retain those pointers after return

This avoids:

- Go-managed heap ownership crossing into Python
- extra native allocations per batch
- `from_blob` lifetime traps in early phases
- DLPack complexity before it is needed

## Tensor / Buffer Layout

The native encoder should target the same logical fields already used by the model code, so the model side changes are minimal.

### Step-major buffers

These are `[N, ...]`, where `N` is the number of envs in the batch.

- `trace_kind_id`: `int64[N]`
- `slot_card_rows`: `int64[N, ZONE_SLOT_COUNT]`
- `slot_occupied`: `float32[N, ZONE_SLOT_COUNT]`
- `slot_tapped`: `float32[N, ZONE_SLOT_COUNT]`
- `game_info`: `float32[N, GAME_INFO_DIM]`
- `pending_kind_id`: `int64[N]`
- `num_present_options`: `int64[N]`
- `option_kind_ids`: `int64[N, MAX_OPTIONS]`
- `option_scalars`: `float32[N, MAX_OPTIONS, OPTION_SCALAR_DIM]`
- `option_mask`: `float32[N, MAX_OPTIONS]`
- `option_ref_slot_idx`: `int64[N, MAX_OPTIONS]`
- `option_ref_card_row`: `int64[N, MAX_OPTIONS]`
- `target_mask`: `float32[N, MAX_OPTIONS, MAX_TARGETS]`
- `target_type_ids`: `int64[N, MAX_OPTIONS, MAX_TARGETS]`
- `target_scalars`: `float32[N, MAX_OPTIONS, MAX_TARGETS, TARGET_SCALAR_DIM]`
- `target_overflow`: `float32[N, MAX_OPTIONS]`
- `target_ref_slot_idx`: `int64[N, MAX_OPTIONS, MAX_TARGETS]`
- `target_ref_is_player`: `bool[N, MAX_OPTIONS, MAX_TARGETS]`
- `target_ref_is_self`: `bool[N, MAX_OPTIONS, MAX_TARGETS]`
- `may_mask`: `bool[N]`

### Decision buffers

These are flattened across the batch to avoid ragged nested structures.

- `decision_start`: `int64[N]`
- `decision_count`: `int64[N]`
- `decision_option_idx`: `int64[TOTAL_DECISION_ROWS, MAX_CACHED_CHOICES]`
- `decision_target_idx`: `int64[TOTAL_DECISION_ROWS, MAX_CACHED_CHOICES]`
- `decision_mask`: `bool[TOTAL_DECISION_ROWS, MAX_CACHED_CHOICES]`
- `uses_none_head`: `bool[TOTAL_DECISION_ROWS]`

The caller provides `TOTAL_DECISION_ROWS` capacity for the batch. The encoder returns the actual number written.

## ABI Surface

The shared library should export a new C ABI family. The JSON API can stay temporarily for debugging and fallback.

### Core structs

```c
typedef struct {
    int64_t n;  // number of game handles in this batch
    const int64_t* handles;
    const int64_t* perspective_player_idx;  // length n, -1 means infer current pending player
} MageBatchRequest;

typedef struct {
    int64_t max_options;
    int64_t max_targets_per_option;
    int64_t max_cached_choices;
    int64_t zone_slot_count;
    int64_t game_info_dim;
    int64_t option_scalar_dim;
    int64_t target_scalar_dim;
    int64_t decision_capacity;  // total rows available in flattened decision buffers
} MageEncodeConfig;

typedef struct {
    int64_t* trace_kind_id;
    int64_t* slot_card_rows;
    float* slot_occupied;
    float* slot_tapped;
    float* game_info;
    int64_t* pending_kind_id;
    int64_t* num_present_options;
    int64_t* option_kind_ids;
    float* option_scalars;
    float* option_mask;
    int64_t* option_ref_slot_idx;
    int64_t* option_ref_card_row;
    float* target_mask;
    int64_t* target_type_ids;
    float* target_scalars;
    float* target_overflow;
    int64_t* target_ref_slot_idx;
    uint8_t* target_ref_is_player;
    uint8_t* target_ref_is_self;
    uint8_t* may_mask;
    int64_t* decision_start;
    int64_t* decision_count;
    int64_t* decision_option_idx;
    int64_t* decision_target_idx;
    uint8_t* decision_mask;
    uint8_t* uses_none_head;
} MageEncodeOutputs;

typedef struct {
    int64_t decision_rows_written;
    int64_t error_code;
    char* error_message;  // optional, free with MageFreeString
} MageEncodeResult;
```

Use `uint8_t` for boolean arrays at the ABI boundary. Python can view them as `torch.bool` or cast if needed. Do not expose C `_Bool` in the ABI.

### Exported functions

```c
MageEncodeResult MageEncodeBatch(
    const MageBatchRequest* req,
    const MageEncodeConfig* cfg,
    const MageEncodeOutputs* out
);
```

Optional helpers:

```c
int64_t MagePendingPlayer(int64_t handle);
int64_t MageIsOver(int64_t handle);
char* MageWinner(int64_t handle);  // debug / episode completion only
```

The existing `MageStep` entrypoint can remain JSON initially. A later optimization can replace it with a compact action ABI too, but that is secondary if rollout state encoding is the measured bottleneck.

## Python API Layer

Replace the current dict-centric parse path with a tensor-centric encoder wrapper.

Suggested new Python module:

- `magic_ai/native_encoder.py`

Responsibilities:

- load the shared library
- declare the C ABI once
- own reusable CPU tensors for each batch size
- validate shape/dtype/contiguity
- pass raw pointers into `MageEncodeBatch`
- return a `NativeEncodedBatch` object holding tensor views

### Proposed Python object

```python
@dataclass
class NativeEncodedBatch:
    trace_kind_id: Tensor
    slot_card_rows: Tensor
    slot_occupied: Tensor
    slot_tapped: Tensor
    game_info: Tensor
    pending_kind_id: Tensor
    num_present_options: Tensor
    option_kind_ids: Tensor
    option_scalars: Tensor
    option_mask: Tensor
    option_ref_slot_idx: Tensor
    option_ref_card_row: Tensor
    target_mask: Tensor
    target_type_ids: Tensor
    target_scalars: Tensor
    target_overflow: Tensor
    target_ref_slot_idx: Tensor
    target_ref_is_player: Tensor
    target_ref_is_self: Tensor
    may_mask: Tensor
    decision_start: Tensor
    decision_count: Tensor
    decision_option_idx: Tensor
    decision_target_idx: Tensor
    decision_mask: Tensor
    uses_none_head: Tensor
    decision_rows_written: int
```

`PPOPolicy.act_batch(...)` should then consume this object directly instead of `ParsedBatch`.

## Dynamic Library Strategy

### Recommendation

Keep a single Go shared library for the engine and native feature encoder:

- Linux: `libmage.so`
- macOS: `libmage.dylib`
- Windows: `mage.dll`

Do not add a torch dependency to the Go library in phase 1.

That keeps the runtime link story simple:

- one existing native artifact
- one extra exported ABI family
- Python continues to load the same library path

### Python-side loading

The current loader in `../mage-go/cmd/pylib/mage/__init__.py` already searches:

- `MAGE_LIB`
- shared library next to the Python package
- platform-specific extension names

Keep that behavior and extend it rather than inventing a second loader path.

For the redesign:

1. `mage.__init__` continues to locate the shared library.
2. `magic_ai.native_encoder` imports `mage`, asks it for the resolved path, and opens the same library through:
   - `ctypes`
   - or a tiny CPython extension
   - or `cffi`

Recommendation: use `ctypes` for the new encoder wrapper unless there is a strong reason to keep `cffi`. The new ABI is fixed-size pointer passing, which `ctypes` handles cleanly and avoids mixing JSON- and tensor-centric logic in the same wrapper.

### Required `mage` package change

Expose the resolved shared library path:

```python
def resolved_library_path() -> str:
    _ensure_loaded()
    return os.path.abspath(_lib_path_used)
```

This prevents duplicate path resolution logic in `magic-ai`.

### Runtime search paths

#### Linux

Preferred:

- place `libmage.so` adjacent to the Python package as today
- open by absolute path from Python

Avoid relying on:

- global `LD_LIBRARY_PATH`
- `/usr/lib`
- system install locations

Absolute-path loading from Python is deterministic and adequate.

#### macOS

Preferred:

- place `libmage.dylib` adjacent to the Python package
- load by absolute path

If later adding a torch C++ extension, use:

- `@loader_path`-relative install names for any secondary dylibs
- `install_name_tool` only if packaging requires it

#### Windows

Preferred:

- co-locate `mage.dll` with the Python package or extension module
- use absolute path loading

If a torch extension is added later, ensure dependent DLL directories are registered with:

- `os.add_dll_directory(...)`

before loading the extension.

## Build and Packaging Plan

### Phase 1: Go shared library only

The existing `../mage-go/setup.py` already builds the shared library during packaging. Extend that build so the exported encoder ABI is compiled into the same artifact.

No new packaging unit is required for phase 1.

### Development workflow

For local development:

```bash
cd ../mage-go
go build -buildmode=c-shared -o cmd/pylib/mage/libmage.so ./cmd/pylib
```

or platform equivalent.

`magic-ai` continues to depend on the editable `mage-go` package via:

- `pyproject.toml`
- `mage-go = { path = "../mage-go" }`

### Optional Phase 3: torch extension

If a torch C++ extension is added later, it should live in `magic-ai`, not `mage-go`.

Reason:

- `magic-ai` owns the tensor contract
- `mage-go` should remain engine-focused
- linking libtorch into the Go package makes the engine package much harder to build and distribute

Suggested artifact split in phase 3:

- `libmage.so` / `.dylib` / `.dll`: Go engine and encoder ABI
- `_mage_torch_encoder.so` / `.pyd`: PyTorch extension in `magic-ai`

The torch extension should dynamically load or link against `libmage` through the stable C ABI.

## Why Go Should Not Construct `torch.Tensor` Directly

There is no practical or safe direct path from Go to native `torch.Tensor` construction for this codebase.

The problems are:

- PyTorch’s C++ tensor API is not a Go API
- Go FFI into C++ is awkward and brittle
- crossing Go GC ownership with `at::Tensor` lifetime rules is high-risk
- libtorch linking would complicate `mage-go` builds substantially

The correct separation is:

- Go exports plain C ABI functions over raw pointers and integer handles
- Python or a C++ extension constructs/owns tensors

## Decision Layout Encoding

The current Python layout builder uses:

- pending kind
- number of options
- target masks
- priority candidate expansion

That logic is deterministic and should move native with the rest of the encoder.

Recommendation:

- implement the exact existing semantics in Go first
- preserve current row ordering so PPO behavior is unchanged
- add regression tests that compare native decision buffers against current Python-generated buffers over a corpus of real pending states

## Stepping API

The current `MageStep` still takes JSON:

```c
char *MageStep(int64_t id, char *actionJSON);
```

That can stay initially. Most action payloads are much smaller than full state/legal snapshots, so it is reasonable to defer.

If needed later, add a compact action ABI:

```c
typedef struct {
    int64_t handle;
    int64_t action_kind;
    int64_t option_index;
    int64_t target_index;
    int64_t aux_index0;
    int64_t aux_index1;
    uint8_t accepted;
    int64_t x_value;
} MageAction;

int64_t MageStepEncoded(const MageAction* action, char** error_message);
```

But this should be phase 2b or later, not part of the initial redesign.

## Error Handling

Native encoding must not partially succeed silently.

Rules:

1. Any invalid handle returns non-zero `error_code`.
2. Any capacity overflow returns non-zero `error_code`.
3. `decision_capacity` overflow must be a hard error.
4. `error_message` is optional and freed with `MageFreeString`.

Suggested error codes:

- `1`: invalid handle
- `2`: game over / no pending decision
- `3`: output pointer null
- `4`: capacity mismatch
- `5`: decision capacity exceeded
- `6`: internal encode failure

## Synchronization and Threading

Assume the encoder call is synchronous.

Rules:

- one `MageEncodeBatch` call owns its output buffers exclusively during the call
- Go must not retain or write to Python-owned pointers after return
- no background goroutines may mutate those output buffers
- existing per-game locking remains in the handle layer

This keeps the memory model simple and compatible with both Python and future C++ callers.

## GPU Transfer Strategy

The first implementation should use CPU tensors and one batched copy to device.

Recommended progression:

1. contiguous CPU tensors
2. reusable pinned CPU tensors
3. `to(device, non_blocking=True)` on the whole batch

Do not start by trying to write directly into CUDA tensors from Go. That would require:

- CUDA-aware native code at the ABI boundary
- stream semantics
- much tighter integration with PyTorch

That is not justified before the CPU-native marshaling path is finished and measured.

## Python Code Changes

### Remove or bypass

- `GameStateEncoder.parse_state_batch`
- `ActionOptionsEncoder.parse_pending_batch`
- `PPOPolicy.parse_inputs_batch`
- Python-side decision layout generation in `PPOPolicy`

These can remain temporarily for fallback/testing, but the rollout path should stop using them.

### Add

- `magic_ai/native_encoder.py`
- native-batch-aware dataclasses or tensor containers
- differential tests comparing old Python encoding vs new native encoding

### Training loop changes

In `scripts/train_ppo.py`, replace:

- `state = merge_pending_into_state(...)`
- `policy.parse_inputs_batch(...)`

with:

- collect active game handles
- call `native_encoder.encode_batch(handles, perspective_player_indices)`
- pass returned tensors directly into `policy.act_native_batch(...)`

The transcript/debug path can continue to request JSON snapshots if needed.

## Migration Plan

### Step 1

Add the native feature ABI to `mage-go` while keeping all JSON APIs intact.

### Step 2

Implement a Python wrapper that allocates reusable tensors and calls `MageEncodeBatch`.

### Step 3

Teach `PPOPolicy` to consume the native encoded batch directly.

### Step 4

Add parity tests:

- sample a large corpus of real game states/pending states
- compare every exported tensor field against the current Python encoder
- compare decision layout buffers exactly

### Step 5

Flip training to use the native path by default, keep Python path behind a debug flag.

### Step 6

After parity and rollout correctness are stable, remove or de-emphasize the Python parse path.

### Step 7

Only if still justified by profiling, add a torch C++ extension for tighter tensor integration.

## Testing Strategy

### Parity tests

For each sampled env batch:

- compare tensor values field-by-field
- compare `decision_rows_written`
- compare replay behavior by sampling actions from both paths and verifying equivalent logits for the same model weights

### Stress tests

- max options
- max targets
- empty pending/action states
- terminal games
- large batch sizes
- `decision_capacity` near limit

### Failure tests

- invalid handle
- mismatched dimensions
- null pointers from Python wrapper bugs
- buffer too small

## Recommended Implementation Choice

The recommended first implementation is:

1. Keep `libmage` as the only native artifact.
2. Add a flat pointer-based batch encoder ABI.
3. Allocate reusable CPU tensors in Python.
4. Fill them from Go synchronously.
5. Copy to GPU in one batch.

This gives the largest marshaling win with the smallest linking/build risk.

## Open Questions

1. Should `MageStep` remain JSON for the first migration?
   Recommendation: yes.

2. Should the native encoder infer perspective player or require it explicitly?
   Recommendation: accept `-1` for infer, but keep explicit indices in batch mode where possible.

3. Should priority candidate expansion stay separate from general action encoding?
   Recommendation: no, export the exact decision layout buffers directly so policy code stays simple.

4. Should native outputs be CPU-only in phase 1?
   Recommendation: yes.

## Summary

The correct redesign is not “make Go produce PyTorch tensors.” It is:

- keep the Go engine
- add a stable C ABI for batched feature encoding
- let Python own and reuse tensor memory
- remove JSON and Python object materialization from the rollout hot path
- preserve a later option to insert a torch C++ extension without changing the engine ABI

That approach minimizes build/linking complexity, preserves engine ownership boundaries, and directly attacks the measured bottleneck.
