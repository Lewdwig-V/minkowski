# Minkowski Skills & Example Rewrites Design

**Goal:** Comprehensive Claude Code skills that make Claude a Minkowski expert for users who've cloned the repo, plus example rewrites that dogfood the reducer API as the idiomatic mutation path.

**Architecture:** One auto-triggering skill for passive expertise, eight slash commands for deliberate decision-making, seven example rewrites demonstrating reducers throughout.

---

## Auto-Triggering Skill

### `minkowski-guide.md`

**Trigger:** Fires when Claude sees Minkowski imports, `World::new()`, ECS-related questions, or component/entity/query patterns in user code. Description targets: Minkowski ECS usage, data modeling, query patterns, reducer selection, concurrency model, persistence, spatial indexing, performance optimization.

**Audience:** Developers who've cloned the repo and are building on Minkowski. They can read Rust. They don't know Minkowski's idioms.

**Tone:** Adaptive — prescriptive for common patterns ("use QueryWriter for durable bulk updates"), Socratic for workload-dependent decisions ("what's your read/write ratio?").

**Contents (tiered, approximately 300-400 lines):**

1. **Quick reference** — one-paragraph summary of each major subsystem (World, queries, transactions, reducers, persistence, spatial indexing). Enough for Claude to give correct guidance in passing.

2. **Decision flowcharts** — if/then trees for the 8 decision areas:
   - "Spawning inside a query loop? -> CommandBuffer or Spawner reducer"
   - "Need to undo mutations? -> EnumChangeSet"
   - "Single-threaded, no persistence? -> Sequential, direct World methods for setup"
   - "Multiple systems touching overlapping components? -> Optimistic + ReducerRegistry"
   - "Bulk iteration with buffered writes? -> QueryWriter"
   - "Access depends on runtime state? -> Dynamic reducer"

3. **Strong defaults** — prescriptive recommendations:
   - "Start with tuple queries, graduate to Table when schema is fixed"
   - "Start with Sequential, add Optimistic when concurrency is needed"
   - "Use QueryWriter over QueryMut when the reducer needs to be durable"
   - "Prefer `for_each_chunk` over `for_each` for numeric data (enables SIMD)"
   - "Every mutation path should go through a reducer — direct World methods are for setup and debugging"
   - "Register all systems in a ReducerRegistry, even single-threaded — it enables conflict detection for free"

4. **Pitfall alerts** — patterns Claude should flag proactively:
   - `world.insert()` in a hot loop -> archetype migration cost warning
   - `Changed<T>` without understanding tick semantics -> "since last query, not since last frame"
   - `query_raw()` where `query()` was intended -> explain cache/tick tradeoff
   - Missing `is_alive()` check on stale entity references
   - `&mut T` in a transaction query -> must use `ReadOnlyWorldQuery` through `&World`
   - `QueryMut` for a durable reducer -> should be `QueryWriter` (buffers writes, compatible with WAL)
   - Dynamic reducer accessing undeclared components -> runtime panic in all builds

5. **Example pointers** — "for a concrete example of X, see `examples/examples/Y.rs`":
   - Data modeling + Table: `life.rs`
   - Spatial indexing: `boids.rs`, `nbody.rs`
   - Transaction strategies: `transaction.rs`
   - Reducer patterns: `reducer.rs`
   - Durable persistence: `persist.rs`
   - Concurrent systems: `battle.rs`
   - Access conflict detection: `scheduler.rs`

**Does NOT duplicate CLAUDE.md.** References it: "See CLAUDE.md Architecture section for storage model details." The skill adds the opinionated layer — what to use when, and what to avoid.

---

## Slash Commands

Eight commands in `.claude/commands/minkowski/`. Each follows the same structure:

1. **Assess** — examine user's current code context (imports, components, existing patterns)
2. **Recommend** — clear recommendation with reasoning
3. **Implement** — help write the code or point to relevant example

### `/minkowski:model` — Data Modeling

**Strong defaults:**
- One concept = one component. Don't nest structs as component fields.
- Components are `'static + Send + Sync + Copy` when possible (enables zero-copy query).
- Use `#[derive(Table)]` when you have a fixed schema queried together (e.g., physics: Pos+Vel+Accel).
- Use sparse components (`register_sparse`) for rare optional data (e.g., DebugLabel).
- Bundle tuples for spawn: `world.spawn((Pos(0.0), Vel(0.0), Health(100)))`.

**Socratic when:**
- "Is this data queried independently or always together?" -> separate components vs Table
- "How many entities will have this component?" -> archetype vs sparse

### `/minkowski:query` — Query Design

**Strong defaults:**
- `&T` for reads, `&mut T` for writes, `Entity` to get the ID.
- `Option<&T>` for components that may or may not be present (doesn't filter archetype).
- `for_each_chunk` for numeric tight loops (SIMD-friendly contiguous slices).
- `par_for_each` for CPU-heavy per-entity work.

**Socratic when:**
- "How often does this data change?" -> Changed<T> filter
- "Do you need the entity ID alongside component data?" -> include Entity in tuple

### `/minkowski:mutate` — Mutation Strategy

**Strong defaults:**
- Inside a query loop needing structural changes: `CommandBuffer`.
- For reversible/serializable mutations: `EnumChangeSet`.
- For transaction-buffered writes: handled automatically by reducer handles.
- Direct `world.insert/remove()` only for setup and debugging.

**Socratic when:**
- "Do you need rollback?" -> EnumChangeSet with reverse
- "Is this part of a transaction?" -> reducer handle (EntityMut, QueryWriter)

### `/minkowski:concurrency` — Concurrency Model

**Strong defaults:**
- Single-threaded: `Sequential` (zero overhead, all ops delegate to World).
- First step into concurrency: `Optimistic` (cheap reads, tick-based validation).
- Write-heavy with expensive retries: `Pessimistic` (lock guarantee, backoff).
- Need crash safety: `Durable<S, W>` wrapping any strategy.

**Socratic when:**
- "What's your read/write ratio?"
- "How expensive is a retry vs a lock?"
- "Do you need crash recovery?"

### `/minkowski:reducer` — Reducer Selection

**Strong defaults:**
- Entity-scoped mutation: `EntityMut<C>` (register_entity)
- Bulk read-only iteration: `QueryRef<Q>` (register_query_ref)
- Bulk read-write, direct: `QueryMut<Q>` (register_query) — scheduled, compile-time safety
- Bulk read-write, buffered: `QueryWriter<Q>` (register_query_writer) — transactional, durable-compatible
- Entity creation: `Spawner<B>` (register_spawner)
- Runtime-conditional access: `DynamicCtx` (registry.dynamic())
- Structural mutations (remove, despawn): EntityMut with register_entity_despawn, or Dynamic with can_remove/can_despawn

**Socratic when:**
- "Does the access pattern depend on runtime state?" -> Dynamic
- "Does this need to be durable?" -> QueryWriter over QueryMut
- "Do you need to despawn or remove components?" -> register_entity_despawn or can_despawn

### `/minkowski:persist` — Persistence

**Strong defaults:**
- WAL for crash safety (Durable wraps any strategy).
- Snapshots for fast restore (periodic, not every frame).
- Register codecs for every component type that needs persistence.
- `sync_reserved()` after snapshot restore (prevents entity ID overlap).

**Socratic when:**
- "What's your recovery time budget?" -> snapshot frequency
- "Which components need to survive restarts?" -> codec registration scope

### `/minkowski:index` — Spatial Indexing

**Strong defaults:**
- Only if you do spatial neighbor queries (not for all games/sims).
- Uniform density: grid. Clustered: quadtree/BVH.
- Implement `SpatialIndex` trait — `rebuild` required, `update` optional for incremental.
- Always check `world.is_alive(entity)` on query results (stale entries from despawns).

**Socratic when:**
- "How often do entities move?" -> rebuild frequency, incremental vs full
- "What's the spatial distribution?" -> grid cell size, tree branching factor

### `/minkowski:optimize` — Performance

**Strong defaults:**
- `#[repr(align(16))]` or `[f32; 4]` for SIMD-friendly components.
- `for_each_chunk` yields typed slices — LLVM auto-vectorizes.
- Build with `-C target-cpu=native` (already in `.cargo/config.toml`).
- Minimize archetype migrations: batch `insert()`/`remove()` calls.
- Query cache is automatic — repeat queries skip archetype scans.

**Socratic when:**
- "Where's the bottleneck? Iteration, migration, or matching?"
- "How many archetypes do you have?" -> migration cost vs query cache miss rate

---

## Example Rewrites

Every example uses reducers as the primary mutation path. Direct World methods only for initial setup and final reads.

### `boids.rs`

**Current:** Direct `world.query()` + `world.get_mut()` in a loop.

**Rewrite:** Three query reducers registered in a `ReducerRegistry`:
- `separation` — `QueryMut<(&mut Vel, &Pos)>` reads neighbor positions from spatial grid, adjusts velocity
- `alignment` — `QueryMut<(&mut Vel,)>` averages neighbor velocities
- `cohesion` — `QueryMut<(&mut Vel, &Pos)>` steers toward neighbor center of mass
- `integrate` — `QueryMut<(&mut Pos, &Vel)>` applies velocity to position
- Spatial grid `rebuild` called between frames (stays external, not a reducer)

Setup spawns via direct `world.spawn()`. Frame loop dispatches reducers via `registry.run()`.

### `life.rs`

**Current:** `EnumChangeSet` manually built per generation by reading grid.

**Rewrite:** Step function as a `QueryWriter` reducer:
- Reads cell state via `WritableRef<Cell>`, counts neighbors
- Buffers birth/death writes through `modify()`
- Undo/redo still works via changeset reversal (QueryWriter produces changesets)
- Table derive stays as-is (demonstrates schema + reducer together)

### `nbody.rs`

**Current:** Direct query + `get_mut` for force accumulation.

**Rewrite:**
- `rebuild_tree` — not a reducer (external SpatialIndex, called between frames)
- `calculate_forces` — `QueryMut<(&mut Accel, &Pos, &Mass)>` traverses Barnes-Hut tree, accumulates forces
- `integrate` — `QueryMut<(&mut Pos, &mut Vel, &Accel)>` Verlet integration step

### `scheduler.rs`

**Current:** Manual `Access::of::<Q>()` for six systems.

**Rewrite:** Register six systems as reducers in a `ReducerRegistry`. Use `reducer_access()` / `query_reducer_access()` instead of manual Access construction. The conflict detection and graph coloring logic stays the same but operates on registry-provided access metadata.

### `transaction.rs`

**Current:** Raw `Tx` API demonstrating three strategies.

**Rewrite:** Two parts:
1. Brief raw `Tx` demo (keep — teaches the building blocks)
2. Same logic rewritten with `EntityMut` reducers dispatched via `registry.call()` across all three strategies. Shows how reducers abstract away the transaction boilerplate.

### `battle.rs`

**Current:** Rayon + Optimistic, direct `tx.query()`/`tx.write()`.

**Rewrite:** Attack and heal as `EntityMut` reducers registered in a `ReducerRegistry`. Rayon dispatches `registry.call()` in parallel. Conflict detection and retry handled by the registry + Optimistic strategy. The tunable conflict rate parameter stays.

### `persist.rs`

**Current:** Durable transactions with raw `Tx` API.

**Rewrite:** `QueryWriter` reducer for the mutation logic (the motivating use case — buffered writes compatible with WAL). `Durable<Optimistic, Wal>` strategy. Snapshot save/restore stays. Shows the full persistence story: WAL-logged reducer dispatch + periodic snapshots.

---

## What Doesn't Ship

- **ECS tutorial** — the skill guides decisions, doesn't teach ECS from scratch
- **Benchmarking methodology** — existing `cargo bench` + criterion is sufficient
- **Scheduler implementation guidance** — Minkowski provides Access metadata, not scheduling policy
- **Plugin packaging** — in-repo only; refactor to plugin if/when published to crates.io
- **CLAUDE.md duplication** — skill references it, doesn't repeat architecture docs

---

## Files Touched

| File | Change |
|---|---|
| `.claude/skills/minkowski-guide.md` | Create: auto-triggering comprehensive skill |
| `.claude/commands/minkowski/model.md` | Create: data modeling command |
| `.claude/commands/minkowski/query.md` | Create: query design command |
| `.claude/commands/minkowski/mutate.md` | Create: mutation strategy command |
| `.claude/commands/minkowski/concurrency.md` | Create: concurrency model command |
| `.claude/commands/minkowski/reducer.md` | Create: reducer selection command |
| `.claude/commands/minkowski/persist.md` | Create: persistence command |
| `.claude/commands/minkowski/index.md` | Create: spatial indexing command |
| `.claude/commands/minkowski/optimize.md` | Create: performance command |
| `examples/examples/boids.rs` | Rewrite: reducer-based separation/alignment/cohesion/integrate |
| `examples/examples/life.rs` | Rewrite: QueryWriter step function |
| `examples/examples/nbody.rs` | Rewrite: reducer-based force calculation + integration |
| `examples/examples/scheduler.rs` | Rewrite: registry-based access instead of manual Access::of |
| `examples/examples/transaction.rs` | Rewrite: add reducer-based section alongside raw Tx demo |
| `examples/examples/battle.rs` | Rewrite: EntityMut reducers dispatched via rayon |
| `examples/examples/persist.rs` | Rewrite: QueryWriter + Durable |
| `CLAUDE.md` | Update example descriptions, test count |
| `.claude/commands/soundness-audit.md` | Already created (this session) |
| `.claude/commands/validate-macro.md` | Already created (this session) |

## Testing

- All examples compile and run: `cargo run -p minkowski-examples --example <name> --release`
- Existing unit tests still pass: `cargo test -p minkowski --lib`
- Clippy clean: `cargo clippy --workspace --all-targets -- -D warnings`
- Skills/commands are markdown — validated by using them in a Claude Code session
