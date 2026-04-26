# Plan: Replacing mage-go with XMage (Java) under PyTorch

## What we're preserving

The existing Python side already assumes a fast, flat-buffer interface — `NativeEncodedBatch` in `magic_ai/native_encoder.py:20-110` is a pack of fixed-shape `int64`/`float32` arrays (`slot_card_rows[N,50]`, `option_kind_ids[N,64]`, `option_scalars[N,64,14]`, target fields, etc.) read via `ctypes`. `NativeRolloutDriver` (`magic_ai/native_rollout.py:76`) steps N envs in one call. **Our job is to make Java produce those exact buffers** so the training loop, encoder, PPO rollout driver, and model stay untouched.

## Architecture

```
PyTorch trainer (unchanged)
    ↕ ctypes / numpy (zero-copy views over DirectByteBuffer memory)
libmage_bridge.so  ← thin C shim, JNI invocation API
    ↕ JNI
[single JVM]
    ├── BatchCoordinator  (owns N games, gates them at decisions)
    ├── Game 1 … Game N   (each on its own thread)
    └── TorchPlayer extends ComputerPlayer  (intercepts priority/choose)
```

One JVM, N games, each on a dedicated thread. Games run full-speed until they hit a decision; `TorchPlayer` parks the game thread on a per-game `CountDownLatch` / `SynchronousQueue`. When all ready games are parked, Python is unblocked and reads the batch.

## The fast path — serialization

**Non-goal:** JSON, Gson, protobuf, gRPC, Py4J, JPype's object-level bridge. All have per-field overhead that dominates at ~10⁵ decisions/sec.

**Strategy: pre-allocated direct buffers, identical layout to the Go side.**

1. On JVM init, allocate one `java.nio.ByteBuffer.allocateDirect(...)` per field in `NativeEncodedBatch` (slot_card_rows, slot_occupied, option_kind_ids, option_scalars, target_mask, …). These live in off-heap memory with a stable C pointer.
2. Expose their addresses once via JNI (`GetDirectBufferAddress`). Python wraps each with `np.frombuffer(ctypes.cast(ptr, …), …)` — same zero-copy trick the Go cffi path uses.
3. At each decision, `TorchPlayer` writes scalars directly into its slot (`buf.putInt(offset, value)`, `buf.putFloat(...)`). No allocation, no boxing, no reflection. This is the Java equivalent of the Go "native encoder" and the work of porting it is mostly translating `magic_ai/game_state.py` + the Go encoder into Java.
4. Action/return channel is symmetric: Python writes chosen `option_idx`/`target_idx` into an action buffer; Java reads and dispatches.

Numbers to beat: current Go→Python step is on the order of microseconds per env per decision. DirectByteBuffer + JNI is in the same ballpark; `JNIEnv->GetDirectBufferAddress` is O(1) and we call it *once*.

## Synchronization

Avoid per-call JNI hops. Use a **ring-buffer gate**:

- Java `BatchCoordinator` keeps an int[N] "ready" flag in a direct buffer.
- Python calls one JNI entrypoint: `step_batch(action_ptr) → ready_mask`. It:
  1. Writes actions into the action buffer (CPU memcpy).
  2. Releases N latches.
  3. Waits on a single `Phaser` / semaphore until ≥ M games have re-parked.
  4. Returns.
- Each game thread runs: park → write state → signal → await action → apply → resume. No JNI in the hot path on the Java side.

This is one JNI round-trip per batched step, not per game.

## Card DB / JVM startup

- Fat-jar Mage core + `Mage.Player.AI` + all `Mage.Sets.*` via `maven-shade-plugin`. One uber-JAR on the classpath.
- H2 card DB loads once per JVM (~hundreds of ms). Amortized across training run.
- Deck parsing + `Game.loadCards` on each reset is the hot allocation; pool `Game` objects and use `GameImpl.copy()` (it's `Copyable<Game>`) to snapshot a starting position instead of re-initing.

## Action enumeration

`TorchPlayer.priority(game)` needs to produce the same 64-wide option slate the Go encoder does. `Player.getPlayable(game, true)` + `getPlayableObjects(...)` gives abilities; map each ability to `(kind_id, scalars[14])` using the same schema `magic_ai/actions.py:19-51` defines. Target enumeration uses the existing `Target.possibleTargets(...)` API on each ability.

## Migration phases

1. **Scaffolding** — Maven module `mage-torch-bridge` with JNI entrypoints, uber-jar, `libmage_bridge.so`, single-env smoke test that runs a game with `TorchPlayer` making random choices via a Python callback. Proves JVM-in-Python and the buffer path.
2. **Encoder parity** — port the Go native encoder's field population to Java `TorchPlayer`. Validate by playing identical scripted games in Go and Java and diffing buffer contents byte-for-byte.
3. **Batch coordinator** — N-game parking, single-JNI step. Benchmark steps/sec vs. mage-go; target within 2× before optimizing.
4. **Swap** — point `NativeRolloutDriver` at `libmage_bridge.so`. `scripts/train_ppo.py` should need zero changes.

## The main risks

### 1. Reflection-heavy init forces a real JVM
XMage dynamically loads every card's implementation class by name (one class per printed card, thousands of them) and uses reflection throughout the rules engine. This has two consequences:
- **GraalVM native-image is effectively off the table.** Native-image needs closed-world reachability; Mage's reflection + `Class.forName` card loading would require exhaustive reachability metadata for every card and ability. The maintenance burden is larger than the project.
- **We're stuck with a full HotSpot JVM in-process.** That means GC pauses land inside training steps. Under default G1, a young-gen pause of 20–50 ms will stall all N game threads at once and show up as tail latency in PPO step time. Mitigations: `-XX:+UseG1GC -XX:MaxGCPauseMillis=10`, generous heap (`-Xmx8g+`) to keep young-gen collections infrequent, and pool/reuse `Game` objects so the allocation rate per decision is low. If jitter is still bad, ZGC or Shenandoah give sub-ms pauses at some throughput cost.

### 2. Thread-per-game caps N
XMage's game loop is deeply synchronous — `GameImpl.play()` calls into `priority()` which blocks the calling thread until the player returns a decision. There is no coroutine/continuation seam to suspend mid-rules-resolution. So "N envs" = "N live OS threads parked on a latch."
- Fine up to maybe 256–512 envs on a beefy box. Past that, context-switch overhead and scheduler noise start eating the batch-gate's wait-for-M-ready window.
- If we want 1000+ envs per JVM, we'd need to bump Mage's Java version and port the game loop onto Project Loom virtual threads. That's a real port — not trivial, because some XMage code uses thread-local state and synchronized blocks that interact badly with virtual-thread pinning.
- Alternative: run multiple JVMs (one per CPU socket) and let the Python side multiplex. Simpler but pays the card-DB load cost per JVM.

### 3. `GameImpl.copy()` is deep and expensive
`GameImpl` implements `Copyable<Game>` by deep-copying the entire state — players, zones, stack, continuous effects, triggered-ability watchers, the lot. It's what the existing MCTS player uses for simulations, so we know it *works*, but:
- Cost is proportional to game complexity (permanents on battlefield, effect layers). A cluttered midgame state can be single-digit milliseconds per copy.
- If we try to use `copy()` as a cheap "reset to start" to avoid re-running deck construction, we'll find that the starting state is small and copy is fast there — good. But if we try to use it for step-level rollouts (lookahead inside the policy), costs add up fast. Measure before designing MCTS-style features around it.
- Copying is also a silent source of allocation pressure feeding back into risk #1.

### 4. Action/option enumeration parity with mage-go
The Go encoder produces a precise `(kind_id, scalars[14], target slate)` layout that the model is trained against. Reproducing this exactly from XMage's `getPlayable()` / `Target.possibleTargets()` is subtle:
- XMage's ability graph is richer (modal spells, X costs, split/adventure/MDFC cards, replacement effects changing legal targets). The Go engine may implement a simplified subset.
- Any mismatch in option ordering, kind assignment, or target filtering silently poisons training — the policy head learns on indices that don't mean what the runtime thinks they mean. Symptoms look like "model plays legally but badly"; very hard to debug.
- Mitigation: the Phase 2 "byte-for-byte diff vs. Go" gate is the single most important correctness check in the project. Script a corpus of seeded games in Go and replay the exact same actions in Java, diff the emitted buffers. Do not skip this.

### 5. Card coverage gap
XMage supports ~every printed Magic card; mage-go almost certainly implements a tiny subset. Moving to XMage broadens the action/observation distribution enormously. Consequences:
- Embedding table (`card_embedding_table`, 128-dim per card) needs rows for the new vocabulary. If `embeddings.json` was built only over mage-go's card set, we need a retraining or bootstrap strategy for the new cards (random init + train, or pretrain from card-text encoders).
- Existing PPO checkpoints may not transfer cleanly. Budget for a cold-start training run.

### 6. JNI failure modes are nasty
Getting the bridge wrong fails in ways that crash the Python process, not raise exceptions:
- Forgetting `AttachCurrentThread` on a Java-spawned callback thread → segfault.
- Holding `GetDirectBufferAddress` pointers across a GC that moves the buffer → silent corruption (direct buffers *don't* move, but if someone swaps in a heap buffer by accident, boom).
- JVM `abort()` on uncaught errors takes Python down with it.
- Mitigation: keep the JNI shim tiny (≤ a few hundred lines), one code path, reviewed carefully. Do not let it grow feature creep — push logic up into Java or down into Python.

### 7. Determinism and reproducibility
Mage's RNG is centralized (`RandomUtil`), but threading + HashMap iteration order + reflection class-load order can leak nondeterminism into game execution. For reproducible training runs we'll want:
- A seeded `RandomUtil` per game.
- `-Djava.util.Map` replaced with `LinkedHashMap` where iteration order touches gameplay (audit needed).
- Fixed thread-to-game assignment so scheduling noise doesn't change outcomes.

This is solvable but it's a chunk of audit work that mage-go probably got for free.


## One-line recommendation

Mirror the mage-go interface exactly: **one `.so`, ctypes, flat DirectByteBuffers with identical layout to `NativeEncodedBatch`, one JNI call per batched step**. Everything above Python's `NativeRolloutDriver` stays unchanged; the Java work is (a) an uber-jar, (b) a `TorchPlayer`, (c) a `BatchCoordinator`, (d) a ~200-line JNI shim.
