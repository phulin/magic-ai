# Text-encoder render-plan ABI (PR 13-B, Go side)

Audience: someone fluent in `magic-ai` and `mage-go`. This document specifies the
cross-repo Go-side change for PR 13-B from `docs/text_encoder_plan.md` §13. No
Go code is included; the implementation PR lands in `../mage-go`.

## 1. Goal & scope

§13 of the encoder plan ("Hot-path prompt assembly") replaces per-step Python
rendering with a cached-card-token memcpy driven by a structured **render-plan**
stream that the Go engine emits alongside today's slot output. PR 13-B is the
Go/ABI half of that stream: extend `MageEncodeConfig` with an opt-in flag,
extend `MageEncodeOutputs` with a flat `int32` opcode stream and length /
overflow side tables, surface the missing per-permanent status fields that
`stateCard` does not currently carry, and emit the plan from a new code path
that reuses the data extracted by `collectSlotCards` / `fillStateEncoding` /
`fillActionEncoding` (`../mage-go/cmd/pylib/encoder.go`).

**In scope (this PR):**
- `MageEncodeConfig` / `MageEncodeOutputs` ABI additions.
- Render-plan emission code path in `cmd/pylib`.
- Plumbing missing status flags off `interactive.PermanentState` into the new
  emitter.
- Golden-file unit tests for the emitted byte stream.

**Out of scope (lands in PR 13-C, Python side):**
- The numpy / Cython assembler that walks the plan into a `[batch, max_tokens]`
  buffer.
- The card-token cache (`data/text_encoder_card_tokens.pt`) — already covered
  by PR 13-A.
- Any change to the existing slot-encoder outputs.
- A Go-side assembler (Stage B in §13) — deferred until profile evidence.

## 2. Compatibility constraint

The slot encoder must keep shipping unchanged. Concretely:

- When `cfg.emit_render_plan == 0` the new code path is not entered. All
  existing `MageEncodeOutputs` fields are populated exactly as before; the new
  pointers (`render_plan`, `render_plan_lengths`, `render_plan_overflow`) may
  be `NULL`.
- The new `int32` `render_plan` stream is an *additional* output; the existing
  scalar slot tensors (`slot_card_rows`, `slot_tapped`, `game_info`, …) are
  produced unchanged regardless of the flag. PR 13-D / a future deprecation
  decides whether to drop the scalar tensors when the text encoder ships; that
  is not this PR's call.
- The new fields in `MageEncodeConfig` are appended after `decision_capacity`
  so older Python builds that initialize the existing fields and zero the rest
  continue to behave correctly (`emit_render_plan == 0`, no allocation).
- `MageEncodeBatch` returns `MageEncodeResult` unchanged. New error code
  values reuse the existing `mageEncodeErr*` constants
  (`mageEncodeErrBuffer` for capacity overflow when `render_plan_capacity` is
  too small to fit even the structural prologue; `mageEncodeErrArg` for
  inconsistent config).

## 3. `MageEncodeConfig` extension

Append two fields:

```c
typedef struct {
    int64_t max_options;
    int64_t max_targets_per_option;
    int64_t max_cached_choices;
    int64_t zone_slot_count;
    int64_t game_info_dim;
    int64_t option_scalar_dim;
    int64_t target_scalar_dim;
    int64_t decision_capacity;
    /* added in PR 13-B */
    int64_t emit_render_plan;     /* 0 = legacy; 1 = emit render_plan stream */
    int64_t render_plan_capacity; /* max int32 elements per env, must fit in int32 */
} MageEncodeConfig;
```

- Default for both is `0`. With `emit_render_plan == 0` the second field is
  ignored (no allocation expected on the Python side).
- `render_plan_capacity` is the **per-env** budget, matching how
  `decision_capacity * max_cached_choices` already partitions decision storage.
  Total `render_plan` allocation is `n * render_plan_capacity * sizeof(int32)`.
- Validation (extend `validateEncodeConfig`):
  - `emit_render_plan != 0 && render_plan_capacity <= 0` → `mageEncodeErrArg`.
  - `render_plan_capacity > INT32_MAX` → `mageEncodeErrArg` (int32 cursor).
- Overflow behavior: if the emitter would write past
  `render_plan_capacity` for env *i*, it stops writing additional **opcodes**
  (it does not split a multi-int opcode), sets
  `render_plan_overflow[i] = 1`, and continues with the next env. No error
  result; the Python side checks the overflow flag and either falls back to
  the slow renderer or skips that env. This matches today's
  `target_overflow` precedent of a soft-truncation flag.

## 4. `MageEncodeOutputs` extension

Append three pointers:

```c
typedef struct {
    /* … existing fields, unchanged order … */
    uint8_t* uses_none_head;
    /* added in PR 13-B */
    int32_t* render_plan;          /* length n * cfg.render_plan_capacity */
    int64_t* render_plan_lengths;  /* length n; int32 elements written for env i */
    int64_t* render_plan_overflow; /* length n; 1 if env i was truncated */
} MageEncodeOutputs;
```

- Alignment: `int32` natural alignment for `render_plan`; `int64` for the side
  arrays. Python preallocates as `numpy.zeros((n, render_plan_capacity), dtype=np.int32)`
  / `(n,)` int64; Go fills via `unsafe.Slice` cast (mirrors the existing
  `requiredI64`/`requiredF32` helpers in `makeOutputViewsC`,
  `cmd/pylib/main.go:225`).
- Ownership: Python owns the buffers; Go writes into them. Lifetime ends with
  the `MageEncodeBatch` call.
- When `cfg.emit_render_plan == 0` Go must not dereference these pointers.
  `makeOutputViewsC` should skip the requirement check on them in that case.
- Pointers may be `NULL` only when `emit_render_plan == 0`; otherwise they
  must be non-NULL and `mageEncodeErrArg` is returned.

## 5. Opcode encoding

**Header convention (chosen).** Each opcode begins with a single int32 header
`opcode_id` (no length-encoded prefix); operand counts are **fixed per opcode
and table-driven on the Python assembler side**. Justification: opcodes are
few (~14), most have payloads of 1–5 ints, and the assembler already needs an
opcode-dispatch table to know which structural tokens to emit. Encoding the
length redundantly in the header buys nothing and costs an int32 per opcode
(~30 % of stream size on small examples). The fixed-length table also lets
the assembler validate the stream length deterministically: `sum(1 +
arity[op]) == render_plan_lengths[i]` is a unit-test invariant.

The Python side stores the arity table as a constant in
`magic_ai/text_encoder/assembler.py`; PR 13-C ships it.

**Opcode IDs.** Defined as a Go `const` block (suggested file:
`cmd/pylib/render_plan.go`) and re-exported in C for documentation; the
authoritative list lives in Go.

| Opcode | ID | Arity | Payload (int32 fields, in order) | Notes |
|---|---:|---:|---|---|
| `OP_OPEN_STATE`        | 1  | 0 | — | First opcode in every plan. |
| `OP_CLOSE_STATE`       | 2  | 0 | — | Last opcode. |
| `OP_TURN`              | 3  | 2 | `turn`, `step_id` | `step_id` per §7. `turn` clipped to int32. |
| `OP_LIFE`              | 4  | 2 | `player_id` (0=self,1=opp), `life` | One per player. |
| `OP_MANA`              | 5  | 3 | `player_id`, `color_id` (§7), `amount` | Emitted only for nonzero pool entries to save bytes. |
| `OP_OPEN_PLAYER`       | 6  | 1 | `owner_id` (0=self,1=opp) | Wraps player-scoped zone opcodes. |
| `OP_CLOSE_PLAYER`      | 7  | 0 | — | |
| `OP_OPEN_ZONE`         | 8  | 2 | `zone_id` (§7), `owner_id` | |
| `OP_CLOSE_ZONE`        | 9  | 0 | — | |
| `OP_PLACE_CARD`        | 10 | 4 | `slot_idx`, `card_row_id`, `status_bits`, `uuid_idx` | `slot_idx` matches the slot encoder's index. `uuid_idx` is the per-snapshot card-ref index = `<card-ref:K>` K. |
| `OP_COUNTER`           | 11 | 2 | `kind_id` (§7), `count` | Emitted *after* the `OP_PLACE_CARD` for the same permanent, before the next `OP_PLACE_CARD` / `OP_CLOSE_ZONE`. |
| `OP_ATTACHED_TO`       | 12 | 1 | `target_uuid_idx` | Same containment rule as `OP_COUNTER`. `-1` if attachment target is not in the current snapshot's UUID table (rare; emit `-1` rather than dropping). |
| `OP_OPEN_ACTIONS`      | 13 | 0 | — | |
| `OP_CLOSE_ACTIONS`     | 14 | 0 | — | |
| `OP_OPTION`            | 15 | 5 | `kind_id` (§7), `source_card_row`, `source_uuid_idx`, `mana_cost_id`, `ability_idx` | `source_uuid_idx == -1` for options whose source isn't on the battlefield (e.g. casting from hand still has a UUID, but a generic "pass" has none). `mana_cost_id` indexes a startup-time interned mana-cost table; `-1` when not applicable. |
| `OP_TARGET`            | 16 | 3 | `target_card_row`, `target_uuid_idx`, `target_kind` (§7) | Repeated 0..N times after the parent `OP_OPTION`, before the next `OP_OPTION` / `OP_CLOSE_ACTIONS`. |

All payload fields are `int32`. Indices that have no value use `-1` as the
sentinel (matching the existing encoder's `optionRefSlotIdx = -1` convention,
`encoder.go:199`).

**Stream order (deterministic, see §11):**

```
OP_OPEN_STATE
OP_TURN
OP_LIFE(0,L0) OP_LIFE(1,L1)
OP_MANA(0,...)*  OP_MANA(1,...)*
for owner in (self, opp):
    OP_OPEN_PLAYER(owner)
    for zone in zoneSpecs(owner):
        OP_OPEN_ZONE(zone, owner)
        for slot, perm in collectSlotCards(...):  # battlefield slots; nil-skipped
            OP_PLACE_CARD(slot, row, status, uuid_idx)
            OP_COUNTER(...)*
            OP_ATTACHED_TO(...)?
        OP_CLOSE_ZONE
    OP_CLOSE_PLAYER
OP_OPEN_ACTIONS
for opt in options (= numPresent, capped by maxOptions):
    OP_OPTION(...)
    OP_TARGET(...)*  # capped by maxTargetsPerOption
OP_CLOSE_ACTIONS
OP_CLOSE_STATE
```

The assembler maps each opcode to the §3 token rendering exactly (e.g.
`OP_OPEN_ZONE` with `zone_id == battlefield, owner_id == self` writes
`<self><battlefield>`; `OP_PLACE_CARD` writes `<card-ref:K>` then memcpys the
cached body; `status_bits` decodes into one or more flag tokens).

## 6. `status_bits` layout

`int32` bitfield. Bits 0..15 defined; bits 16..31 reserved (must be zero).

| Bit | Mask  | Name        | Source field (see §8) |
|---:|------:|-------------|---|
| 0  | 0x0001 | `TAPPED`    | `PermanentState.Tapped` |
| 1  | 0x0002 | `SICK`      | `PermanentState.SummonSick` |
| 2  | 0x0004 | `ATTACKING` | `PermanentState.Attacking` |
| 3  | 0x0008 | `BLOCKING`  | derived: `PermanentState.Blocking != uuid.Nil` |
| 4  | 0x0010 | `MONSTROUS` | **not surfaced today** (open question §13) |
| 5  | 0x0020 | `FLIPPED`   | **not surfaced today** (open question §13) |
| 6  | 0x0040 | `FACEDOWN`  | `Permanent.FaceDown` (`pkg/mage/card.go:471`) — needs to be added to `PermanentState` |
| 7  | 0x0080 | `PHASED_OUT` | `Permanent.PhasedOut` (`pkg/mage/card.go:457`) — needs to be added to `PermanentState` |
| 8  | 0x0100 | `IS_CREATURE` | `PermanentState.IsCreature` |
| 9  | 0x0200 | `IS_LAND`     | `PermanentState.IsLand` |
| 10 | 0x0400 | `IS_ARTIFACT` | `PermanentState.IsArtifact` |
| 11 | 0x0800 | `IS_ATTACHED` | derived: `PermanentState.AttachedTo != uuid.Nil`. Distinguishes "no attachment ref emitted because there is none" from "attachment ref pointed outside the snapshot". |
| 12..15 | reserved | | for keyword surfacing (Flying / Trample / etc.) in a follow-up PR. |

Type-identity bits (8..10) are technically duplicative with the cached
body's type line, but exposing them explicitly lets the assembler emit
status-style tokens (`<creature>` etc.) without re-parsing card bodies and
matches the existing slot encoder's per-bit type signal. They are cheap.

## 7. Zone IDs and step IDs

To stay in sync with the existing slot encoder, IDs reuse / parallel the
constants in `cmd/pylib/encoder.go`:

**`zone_id`** — extension of `zoneSpecs` (`encoder.go:49`). The slot encoder's
`zoneSpec` is keyed by `(zone, owner)` jointly; for the render plan we split
them so opcodes can carry just the zone:

| ID | Name |
|---:|---|
| 0 | hand |
| 1 | battlefield |
| 2 | graveyard |
| 3 | exile |
| 4 | library |
| 5 | stack |
| 6 | command |

`owner_id`: `0 = self`, `1 = opp`.

**`step_id`** — exactly the indices of `stepNames` (`encoder.go:45`):

| ID | Step |
|---:|---|
| 0 | Untap |
| 1 | Upkeep |
| 2 | Draw |
| 3 | Precombat Main |
| 4 | Begin Combat |
| 5 | Declare Attackers |
| 6 | Declare Blockers |
| 7 | Combat Damage |
| 8 | End Combat |
| 9 | Postcombat Main |
| 10 | End |
| 11 | Cleanup |
| 12 | Unknown |

**`color_id`** — index into `manaSymbols` (`encoder.go:44`):

| ID | Color |
|---:|---|
| 0 | W |
| 1 | U |
| 2 | B |
| 3 | R |
| 4 | G |
| 5 | C |

**`kind_id` for `OP_OPTION`** — index into `actionKinds` (`encoder.go:47`):
`0 pass, 1 play_land, 2 cast_spell, 3 activate_ability, 4 attacker, 5 blocker,
6 choice, 7 unknown`.

**`target_kind` for `OP_TARGET`** — parallels the existing
`targetTypeIDs` enum used by `fillActionEncoding` (`encoder.go:455`):
`0 = player, 1 = permanent, 2 = card_in_zone, 3 = unknown`. The existing
`unknownTargetID = 3` constant (`encoder.go:27`) must be reused as-is.

**`kind_id` for `OP_COUNTER`** — index into `core.CounterType`
(`pkg/mage/core/counter.go`). Treat the enum's underlying ordering as the
on-the-wire ID so `core.NumCounters` is the upper bound. The Go emitter
should re-export the enum values to a Python-visible header (suggested
`cmd/pylib/render_plan.go` includes a generator comment) so PR 13-C can
hard-code the same map. `core.NumCounters` is currently 33 (counted from
`counter.go`); fits trivially in int32.

All IDs are **0-indexed** and **stable** — adding a new step / color / counter
must append to the end and bump a version constant in `render_plan.go`.

## 8. Status-field plumbing audit

| Bit | Source-of-truth field | File | Surfaced in `interactive.PermanentState`? | Plumbing required |
|---|---|---|:---:|---|
| TAPPED   | `Permanent.Tapped` | `pkg/mage/card.go:456` | Yes (`PermanentState.Tapped`, `interactive/types.go:150`) | None. Already snapshotted at `interactive/snapshot.go:146`. |
| SICK     | `Permanent.HasAttr(core.AttrSummonSick)` | `pkg/mage/card.go:528`, `pkg/mage/core/attr.go:14` | Yes (`PermanentState.SummonSick`, `interactive/types.go:151`) | None. Snapshotted at `interactive/snapshot.go:147`. |
| ATTACKING | `Combat.IsAttacking(permID)` via `Game.GetCombat()` | `pkg/mage/combat.go` (interface), `pkg/mage/game_mutator.go:88` | Yes (`PermanentState.Attacking`, `interactive/types.go:155`) | None. Snapshotted at `interactive/snapshot.go:151`. |
| BLOCKING | `CombatGroup.BlockerIDs` traversal yielding the attacker UUID | `pkg/mage/combat.go` (search) | Partial — `PermanentState.Blocking` carries the *attacker UUID*, not a bool (`interactive/types.go:156`). | Emitter derives the bit from `permState.Blocking != uuid.Nil`. No engine change. |
| MONSTROUS | **No dedicated field.** Some monstrosity-style cards in the catalog flag a stored bool but there is no canonical `Permanent.Monstrous` or `AttrMonstrous`. | n/a | No | **Open question §13.** Either drop the bit from v1 or define `AttrMonstrous` and seed it from the relevant card abilities. v1 recommendation: leave bit 4 reserved-zero, defer until a card actually needs it. |
| FLIPPED  | **No dedicated field.** Permanents have `FaceDown` but not a flipped/transformed bit; transform/flip mechanics are not yet implemented in the engine (no references in `pkg/mage/`). | n/a | No | **Open question §13.** Same disposition as MONSTROUS. |
| FACEDOWN | `Permanent.FaceDown` | `pkg/mage/card.go:471` | **No** — `PermanentState` does not currently carry this. | Add `FaceDown bool` to `PermanentState` and copy in `snapshotPlayer` (`interactive/snapshot.go:141`). One-line change. |
| PHASED_OUT | `Permanent.PhasedOut` | `pkg/mage/card.go:457` | **No** — same. | Add `PhasedOut bool` to `PermanentState` and copy in `snapshotPlayer`. One-line change. |
| IS_CREATURE / IS_LAND / IS_ARTIFACT | `PermanentState.IsCreature/IsLand/IsArtifact` | `interactive/types.go:152-154` | Yes | None. |
| IS_ATTACHED | derived from `PermanentState.AttachedTo` | `interactive/types.go:163` | Yes | None. The opcode `OP_ATTACHED_TO` uses the same field. |

**Counters** (`OP_COUNTER`): `PermanentState.Counters` is currently a
`map[string]int` keyed by `core.CounterType.String()` (`interactive/types.go:157`,
populated at `interactive/snapshot.go:167`). The emitter needs the
underlying enum value, not the string. Two options:

1. Change `PermanentState.Counters` to `[NumCounters]uint8` (matching the
   raw `Permanent.Counters` storage) — fastest, but breaks any existing
   TUI consumer.
2. Add a parallel `RawCounters [NumCounters]uint8` field next to the
   existing string map — additive, doesn't break anything.

Recommendation: option 2 for this PR; the string map can be deleted in a
follow-up once nothing reads it. Cost is 33 bytes per permanent, negligible.

**Attachment ID** (`OP_ATTACHED_TO`): `PermanentState.AttachedTo`
(`interactive/types.go:163`, populated at `interactive/snapshot.go:178`)
already exposes the UUID; emitter looks it up in the per-snapshot UUID
table built during render-plan emission.

## 9. Custom cards / oracle text gap

`../mage-go/cards/custom/` registers two cards via `Register(...)`:

| Name | File | In `data/card_oracle_embeddings.json`? |
|---|---|:---:|
| `Modal Test Artifact`     | `modal.go:15`        | No |
| `Wraithbloom Cultivator`  | `wraithbloom.go:13`  | No |

Verified by loading `card_oracle_embeddings.json` (150 entries, neither name
present). Both are pure custom cards with no Scryfall counterpart, so they
will *never* appear in the Scryfall-derived oracle JSON.

**Recommendation.** Add an `OracleText string` (and matching `TypeLine`,
`PowerToughness`) field to whatever surfaces card metadata to the
card-cache builder — most cleanly an `interactive.CardOracle{Name, TypeLine,
ManaCost, OracleText, PowerToughness}` struct returned alongside
`MageRegisteredCards()`, populated for custom cards from a hand-written
table colocated with the `Register(...)` call (e.g. add a
`WithOracleText("…")` card option, or a parallel `RegisterOracleText(name,
text)` registry). For Scryfall-backed names, either leave the field empty
(card-cache prefers the JSON when both exist) or backfill from the JSON at
build time.

The lazy alternative — fail loud at cache-build time and require a
hand-written entry per missing card — works for n=2 but won't scale as
custom cards grow. The catalog/registry route is small and one-off.

This PR ships only the ABI; the registry / oracle field decision is called
out as an explicit ask in §13.

## 10. Capacity sizing

Worst-case board (drawn from §3 / §13's "busy mid-game" phrasing):

- Battlefield: 12 permanents/side × 2 sides = 24 permanents.
  - Each: `OP_PLACE_CARD` (5 ints inc. header) + up to 3 `OP_COUNTER` (3
    ints each, inc. header) + 1 `OP_ATTACHED_TO` (2 ints).
  - Per permanent: 5 + 9 + 2 = **16 ints worst case**, ~6 typical.
- Hand: 7/side × 2 = 14 cards × 5 ints = 70 ints.
- Graveyard: 30/side × 2 = 60 cards × 5 ints = 300 ints (large cap; usually
  much smaller).
- Library / exile / stack / command: small constant; budget 50 ints total.
- Zone wrappers: 7 zones × 2 owners × (open=3 + close=1) = 56 ints.
- Player wrappers + state header: 2 × (`OP_OPEN_PLAYER`+`OP_CLOSE_PLAYER`)
  + life/mana/turn = ~30 ints.
- Actions: 50 options × (`OP_OPTION` 6 ints + 3 targets × 4 ints) = 50 ×
  (6 + 12) = 900 ints.
- `OP_OPEN_ACTIONS` / `OP_CLOSE_ACTIONS` / `OP_OPEN_STATE` /
  `OP_CLOSE_STATE`: 4 ints.

**Sum:** 24×16 + 70 + 300 + 50 + 56 + 30 + 900 + 4 ≈ **1794 ints** ≈ 7.2 KB
per env worst case.

**Recommended default `render_plan_capacity`: 4096 int32** (16 KB/env).
Comfortable 2× headroom over the worst case above. At a typical 256-env
batch this is 4 MB — acceptable; memory dwarfed by the slot tensors today.
Capacity can be tuned per call if the Python side is willing to fall back
on overflow.

## 11. Determinism guarantees

The render plan is a pure function of `(snapshot, perspective)` — no
goroutine scheduling, no map-iteration order, no time-based input.
Specific ordering rules:

1. **Player order**: self before opponent. `perspective_player_idx` is
   resolved exactly as `resolvePerspectivePlayerIndex` does today
   (`encoder.go:217`).
2. **Zone order**: extension of `zoneSpecs` (`encoder.go:49`). Per owner:
   battlefield, hand, graveyard, exile, library (count only), stack,
   command. Stack and command are emitted once at the top level (not per
   owner) since they aren't owned.
3. **Card order within a zone**: matches `zoneCards` (`encoder.go:292`):
   battlefield in `player.Battlefield` slice order, hand in `player.Hand`
   slice order, graveyard in `player.Graveyard` slice order. Engine
   guarantees these are append-on-event and never re-sorted.
4. **Status flags within a `PLACE_CARD`**: `OP_COUNTER` opcodes emitted in
   `core.CounterType` enum order (`P1P1` first, then `M1M1`, …),
   skipping zeros. `OP_ATTACHED_TO` after all counters.
5. **`uuid_idx` assignment**: incremented once per card in the traversal
   above, capped at `MAX_CARD_REFS` (currently 64, matching
   `magic_ai/text_encoder/render.py`'s `MAX_CARD_REFS`). Stack objects
   reuse the index of the underlying card if it has one already; if not,
   they get a fresh index after all zone cards.
6. **Action order**: matches today's `pending.Options` slice order, capped
   by `cfg.maxOptions` (mirrors `numPresent` at `encoder.go:412`).
7. **Target order**: matches `option.ValidTargets` slice order, capped by
   `cfg.maxTargetsPerOption` (mirrors `encoder.go:449`).

A parity test should hash the emitted plan and assert byte-for-byte
equality across two consecutive `MageEncodeBatch` calls on the same handle
without state mutation in between.

## 12. Test plan (Go side)

Recommended test files: `cmd/pylib/render_plan_test.go` (and a fixture
directory `cmd/pylib/testdata/render_plan/`).

1. **Validation tests** (cheap):
   - `emit_render_plan = 0` with NULL render-plan pointers → success, no
     writes.
   - `emit_render_plan = 1` with NULL render-plan pointers →
     `mageEncodeErrArg`.
   - `render_plan_capacity = 0` with `emit_render_plan = 1` →
     `mageEncodeErrArg`.
   - `render_plan_capacity` smaller than the prologue (e.g. 4 ints) →
     overflow flag set, `render_plan_lengths[i]` ≤ capacity.

2. **Opcode-level invariants**:
   - Every plan starts with `OP_OPEN_STATE` and ends with
     `OP_CLOSE_STATE`.
   - `OP_OPEN_ZONE` / `OP_CLOSE_ZONE` and `OP_OPEN_PLAYER` /
     `OP_CLOSE_PLAYER` and `OP_OPEN_ACTIONS` / `OP_CLOSE_ACTIONS` are
     balanced.
   - `OP_PLACE_CARD.uuid_idx` is unique within a plan and densely packed
     `[0, K)`.
   - `sum(1 + arity[op]) == render_plan_lengths[i]`.

3. **Determinism**: emit the plan twice on the same handle, assert
   byte-equal.

4. **Golden-file test (the headline)**: capture the render-plan bytes for
   ~10 fixture snapshots covering:
   - Opening hand mulligan choice.
   - Untap/upkeep with summoning-sick creatures.
   - Combat: declare attackers, then declare blockers (different
     `pending.Kind`).
   - Cluttered mid-game with counters and an attached aura.
   - Empty graveyard / empty exile (zone delimiters still emitted).
   - X-spell on the stack with multiple targets.
   - Activated ability with no targets.
   - Modal spell (`Modal Test Artifact`) — exercises ability-index
     plumbing.
   - Custom card without Scryfall oracle text — exercises §9.
   - Game-over snapshot (should still produce a well-formed plan; or
     return `mageEncodeErrOver` consistently — pick and assert).
   Persist each as `testdata/render_plan/<name>.bin` plus a `.json`
   metadata file; the test reads the snapshot, runs the encoder, and
   compares bytes with a clear regenerate-on-purpose path
   (`go test -update` or a `RENDER_PLAN_REGEN=1` env knob).

5. **Slot-encoder parity**: with `emit_render_plan = 1`, the slot tensors
   must equal those emitted with `emit_render_plan = 0`. This guards §2
   compatibility.

## 13. Open questions for review (Go-side asks)

1. **MONSTROUS / FLIPPED** — neither has a source-of-truth field in
   `pkg/mage/`. Recommend leaving bits 4 and 5 reserved-zero in v1 and
   revisiting when an actual card needs them. The Go-side implementer
   should confirm there is no `AttrMonstrous` / `AttrFlipped` I missed.
2. **`PermanentState` extension for `FaceDown` and `PhasedOut`** — needs a
   one-line addition in `interactive/types.go` and a copy in
   `interactive/snapshot.go:141`. Confirm this is in scope for PR 13-B
   (it seems minimal enough that bundling is cleaner than a separate PR).
3. **`PermanentState.Counters` representation** — option 2 (add
   `RawCounters [NumCounters]uint8`) recommended. Confirm no other
   consumer depends on the field staying a `map[string]int`. A quick
   grep for `permState.Counters` outside `interactive/` would close this.
4. **Custom-card oracle text** — green-light the `OracleText` registry
   route in §9, or push back with an alternative. Worst case the cache
   builder fails loud at startup for the two known custom cards; that's
   acceptable as a stop-gap if the registry route is too invasive for
   this PR.
5. **Multi-color / hybrid mana costs** — the existing scalar pipeline
   parses costs at emit time (`fillManaCostFeatures`, `encoder.go:494`)
   and discards the per-symbol order. The render plan currently routes
   the cost through `mana_cost_id` (interned int) so the assembler can
   emit the original string. Confirm that the source string available
   inside `apiOption.ManaCost` (e.g. as currently used at `encoder.go:488`)
   is the canonical Scryfall form (`{2}{W/U}{R}` etc.). If it has been
   pre-lowercased or stripped of `/`, the cache key will diverge from the
   tokenizer's vocab.
6. **`OP_COUNTER` arity for very high stacks** — `Permanent.Counters` is
   `uint8`, max 255. We're emitting `int32 count`; fine. Confirm no
   downstream consumer assumes a smaller width.
7. **Stack object UUID re-use** — when a card on the stack also has a
   battlefield reflection (e.g. a copy of a permanent), should the stack
   `OP_PLACE_CARD` reuse the existing `uuid_idx` or get a fresh one? §11
   recommends reuse-if-present; confirm with the engine team that
   `apiPending` / stack-object IDs are stable across snapshots.
8. **`MageBatchPoll` symmetry** — should `MageBatchPoll` also accept the
   render-plan flag, or is the plan only emitted from `MageEncodeBatch`?
   v1 recommendation: `MageEncodeBatch` only — `MageBatchPoll` is the
   readiness probe and shouldn't grow more outputs. Confirm.
9. **Header/length convention** — chose fixed-arity-by-opcode (§5).
   Confirm preference; the alternative `(opcode << 24 | arity)` header
   was considered and rejected as redundant.
10. **`render_plan_capacity` upper bound** — int32 cursor implies
    `< 2^31`. Should we additionally cap at e.g. `1 << 20` to catch
    pathological misconfiguration? Low priority.
