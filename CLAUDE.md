# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo test -p minkowski --lib          # Unit tests (387 tests, fast)
cargo test -p minkowski                # All tests including doc tests
cargo test -p minkowski -- entity      # Run tests matching a filter

cargo clippy --workspace --all-targets -- -D warnings   # Lint (strict, warnings are errors)
cargo fmt --all                                          # Format

cargo bench -p minkowski-bench                      # All standardized benchmarks (simple_insert, simple_iter, fragmented_iter, heavy_compute, add_remove, schedule, serialize, reducer)
cargo bench -p minkowski-bench -- simple_iter       # Single scenario
cargo bench -p minkowski-bench -- simple_iter/par   # Sub-benchmark filter
cargo bench -p minkowski-persist       # Persistence benchmarks (snapshot save/load/zero-copy, WAL append)

cargo run -p minkowski-examples --example boids --release   # Boids flocking with query reducers + spatial grid (5K entities, 1K frames)
cargo run -p minkowski-examples --example life --release    # Game of Life with QueryMut reducer, Table (64x64 grid, 500 gens)
cargo run -p minkowski-examples --example nbody --release   # Barnes-Hut N-body with query reducers (2K entities, 1K frames)
cargo run -p minkowski-examples --example scheduler --release   # ReducerRegistry-based conflict detection + batch scheduling (6 systems, 10 frames)
cargo run -p minkowski-examples --example transaction --release   # Transaction strategies: raw Tx + reducer comparison (3 strategies, 100 entities)
cargo run -p minkowski-examples --example battle --release   # Multi-threaded EntityMut reducers with tunable conflict (500 entities, 100 frames)
cargo run -p minkowski-examples --example persist --release   # Durable QueryWriter reducer: WAL + rkyv snapshots + zero-copy load (100 entities, 3 archetypes, 10 frames)
cargo run -p minkowski-examples --example replicate --release   # Pull-based WAL replication: cursor + batch + apply to replica (20 source + 10 WAL, convergence check)
cargo run -p minkowski-examples --example reducer --release   # Typed reducer system: entity/query/spawner/query-writer/dynamic handles + structural mutations + conflict detection
cargo run -p minkowski-examples --example index --release   # B-tree range queries + hash exact lookups (200 entities)
cargo run -p minkowski-examples --example flatworm --release   # Flatworm (planarian) simulator: chemotaxis, fission, starvation, spatial grid (200 worms, 1K frames)
cargo run -p minkowski-examples --example circuit --release   # Analog circuit sim: 555 astable → LCR bandpass → 741 follower, symplectic Euler (200K steps, ASCII waveform)
cargo run -p minkowski-examples --example tactical --release   # Multi-operator tactical map: sparse components, par_for_each, Optimistic Conflict, entity bit packing, HashIndex stale validation, EnumChangeSet/MutationRef replication (100 units, 10 ticks, 2 threads)
cargo run -p minkowski-examples --example observe --release   # Observability: MetricsSnapshot capture, diff, entity churn (100 entities, 2 archetypes)

MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- --skip par_for_each  # UB check (strict)
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski --lib par_for_each  # rayon tests

RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p minkowski --lib --tests -Z build-std --target x86_64-unknown-linux-gnu -- --skip par_for_each  # TSan (data race detection)
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p minkowski --lib --tests -Z build-std --target x86_64-unknown-linux-gnu par_for_each  # TSan rayon tests

RUSTFLAGS="--cfg loom" cargo test -p minkowski --lib --features loom -- loom_tests  # loom: exhaustive concurrency verification

cargo +nightly fuzz run fuzz_world_ops -- -max_total_time=60     # fuzz: random World operations
cargo +nightly fuzz run fuzz_reducers -- -max_total_time=60      # fuzz: query iteration paths
cargo +nightly fuzz run fuzz_snapshot_load -- -max_total_time=60 -max_len=65536  # fuzz: snapshot deserialization
cargo +nightly fuzz run fuzz_wal_replay -- -max_total_time=60 -max_len=65536    # fuzz: WAL replay
```

Miri flags: `-Zmiri-tree-borrows` because crossbeam-epoch (rayon dep) violates Stacked Borrows; `-Zmiri-ignore-leaks` only for the two `par_for_each` tests because rayon's thread pool intentionally outlives main. All other tests run without leak suppression to catch real leaks.

Pre-commit hooks run `cargo fmt` and `cargo clippy -D warnings` on commit, `cargo test` on push.

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every PR and push to main:

| Job | Toolchain | Command | `needs` |
|---|---|---|---|
| fmt | stable | `cargo fmt --all -- --check` | — |
| clippy | stable | `cargo clippy --workspace --all-targets -- -D warnings` | — |
| test | stable | `cargo test -p minkowski` | fmt, clippy |
| miri | nightly | Full Miri run including rayon (push to main only) | test |
| tsan | nightly | Two-step TSan run with `-Z build-std` (see Build & Test Commands) | test |
| loom | stable | Exhaustive concurrency verification (see Build & Test Commands) | test |

Format and clippy run in parallel. Test runs after both pass. TSan and Loom run in parallel after test on every PR. Miri runs only on push to main (full test suite including rayon). A `ci-pass` aggregator job (runs with `if: always()`) is the single required status check for branch protection — it verifies all required jobs succeeded and allows Miri to be skipped on PRs without failing the gate.

## Architecture

Minkowski is a **column-oriented archetype ECS**. Five crates: `minkowski` (core), `minkowski-derive` (`#[derive(Table)]` proc macro), `minkowski-persist` (WAL, snapshots, durable transactions), `minkowski-observe` (metrics capture and display), and `minkowski-examples` (examples as external API consumers).

### Storage Model

Each unique set of component types gets an **Archetype** — a struct containing parallel `BlobVec` columns (type-erased `Vec<T>` storing raw bytes via `Layout`) plus a `Vec<Entity>` for row-to-entity mapping. A `FixedBitSet` on each archetype tracks which `ComponentId`s it contains, enabling fast query matching via bitwise subset checks.

**Entity** = u64 bit-packed: low 32 bits = index, high 32 bits = generation. `EntityAllocator` maintains a generation array + free list. `entity_locations: Vec<Option<EntityLocation>>` maps entity index → (archetype_id, row) for O(1) lookup.

**Sparse components** (`PagedSparseSet` with 4096-entry pages mapped to a dense `BlobVec`) are opt-in via `insert_sparse`. Not stored in archetypes.

### Data Flow

`world.spawn((Pos, Vel))` → `Bundle::component_ids` registers types → `Archetypes::get_or_create` finds/creates archetype by sorted component ID set → `Bundle::put` writes each component into BlobVec columns via raw pointer copy → `EntityLocation` recorded.

`world.query::<(&mut Pos, &Vel)>()` → looks up `QueryCacheEntry` by `TypeId` → if archetype count unchanged, reuses cached `matched_ids`; otherwise incrementally scans only new archetypes via `required.is_subset(&arch.component_ids)` → `init_fetch` grabs raw column pointers per cached archetype → `QueryIter` yields items via pointer arithmetic (`ptr.add(row)`).

### Archetype Migration

`world.insert(entity, NewComponent)` moves an entity from archetype A to A∪{new}: copies each column via `BlobVec::push` + `swap_remove_no_drop`, writes new component, updates all `EntityLocation`s (including the entity swapped into the vacated row). `get_pair_mut` uses `split_at_mut` for safe double-mutable archetype access.

### Key Traits

- **Component**: marker, blanket impl for `'static + Send + Sync`
- **Bundle** (unsafe): tuple impls 1-12 via `impl_bundle!` macro. `component_ids()` registers + sorts + deduplicates. `put()` yields component pointers via `ManuallyDrop`.
- **WorldQuery** (unsafe): tuple impls 1-12 via `impl_world_query_tuple!` macro. `Fetch` type holds `ThinSlicePtr<T>` (raw pointer wrapped for Send+Sync). Impls for `&T`, `&mut T`, `Entity`, `Option<&T>`, `Changed<T>`. Methods: `required_ids`, `init_fetch`, `fetch`, `as_slice`, `mutable_ids`, `matches_filters`.
- **Table** (unsafe): generated by `#[derive(Table)]`. Named struct = schema declaration + data container. Associated types `Ref<'w>` / `Mut<'w>` for typed row access. `TableDescriptor` caches archetype_id + field-to-column index mapping. `query_table`/`query_table_mut` skip archetype matching entirely.
- **TableRow** (unsafe): constructs typed row references from raw column pointers. Generated for `FooRef<'w>` and `FooMut<'w>` by the derive macro.
- **Transact**: closure-based transaction API. Primary method `transact(&self, world, access, closure)` runs the closure in a retry loop. Building-block methods `begin()` and `try_commit()` enable advanced patterns. Impls: `Optimistic` (tick-based validation), `Pessimistic` (cooperative column locks). `Durable<S>` (in persist crate) wraps any `Transact` to add WAL logging.

### Query Caching

`World::query()` maintains a `HashMap<TypeId, QueryCacheEntry>` that caches matched archetype IDs per query type. On repeat calls with no new archetypes, the archetype scan is skipped entirely. When new archetypes are created (spawn/insert/remove can trigger this), only the new ones are scanned incrementally. Empty archetypes are filtered at iteration time, not cache time. `query()` takes `&mut self` for cache mutation.

### Column Alignment & Vectorization

BlobVec columns are allocated with 64-byte alignment (cache line). `QueryIter::for_each_chunk` yields typed `&[T]` / `&mut [T]` slices per archetype — LLVM can auto-vectorize loops over these slices.

Component types that are 16-byte-aligned (e.g., `#[repr(align(16))]` or naturally `[f32; 4]`) vectorize better than odd-sized ones. The engine guarantees 64-byte column alignment; component layout determines whether LLVM can pack operations.

Build with `-C target-cpu=native` (configured in `.cargo/config.toml`) to enable platform-specific SIMD instructions.

### Change Detection

Each BlobVec column stores a `changed_tick: Tick` — the tick at which it was last mutably accessed. The tick is a monotonic u64 counter that auto-advances on every mutation and query — there is no user-facing `World::tick()`. Every mutable access path (spawn, get_mut, insert, query `&mut T`, query_table_mut, query_table_raw, changeset apply) advances the tick and marks affected columns. Marking is pessimistic (on mutable access, not actual write) but zero-cost at the write site.

`Changed<T>` is a `WorldQuery` filter that skips entire archetypes whose column tick is older than the query's `last_read_tick` (stored per query type in `QueryCacheEntry`). `Changed<T>` means "since the last time this query observed this column" — it has no concept of frames or simulation time. `Tick` is `pub(crate)` — not exposed to users.

### Deferred Mutation

`CommandBuffer` stores `Vec<Box<dyn FnOnce(&mut World) + Send>>`. Used during query iteration when structural changes (spawn/despawn/insert/remove) must be deferred. Applied via `cmds.apply(&mut world)`.

`EnumChangeSet` is the data-driven alternative: mutations are recorded as a `Vec<Mutation>` enum with component bytes in a contiguous `Arena`. `apply()` consumes the changeset and returns `Result<(), ApplyError>`. Useful for persistence (WAL serialization) and transactions. Typed safe helpers (`insert<T>`, `remove<T>`, `spawn_bundle<B>`) wrap the raw `record_*` methods — they auto-register component types and handle `ManuallyDrop` internally. The raw methods remain for power users who already have a `ComponentId`.

### Secondary Indexes

`SpatialIndex` is a lifecycle trait for user-owned spatial data structures (grids, quadtrees, BVH, k-d trees). Indexes are fully external to World — they compose from existing query primitives. The trait has two methods: `rebuild` (required, full reconstruction) and `update` (optional, defaults to rebuild — override for incremental updates via `Changed<T>`). Despawned entities are handled via generational validation: stale entries are skipped at query time when `world.is_alive()` returns false, and cleaned up on the next rebuild.

### System Scheduling Primitives

`Access` extracts component-level read/write metadata from any `WorldQuery` type. `Access::of::<(&mut Pos, &Vel)>(world)` returns a struct with two `FixedBitSet`s: reads (Vel) and writes (Pos), plus a `despawns: bool` flag. `conflicts_with()` detects whether two accesses violate the read-write lock rule — two bitwise ANDs over the component bitsets, plus despawn-vs-any-access blanket conflict. `has_any_access()` returns true if the access touches any component.

This is a building block for framework-level schedulers. Minkowski provides the access metadata; scheduling policy (dependency graphs, topological sort, parallel execution) is the framework's responsibility.

### Transaction Semantics

`Transact` is a closure-based trait. The primary entry point is `transact(&self, world, access, |tx, world| { ... })` which runs the closure in a retry loop, handling begin/commit/abort automatically. Building-block methods `begin()`, `try_commit()`, and `max_retries()` are also available for advanced patterns (e.g. parallel execute phases where multiple transactions read concurrently before committing sequentially).

`Tx` is the unified transaction handle. It does NOT hold `&mut World` — methods take `world` as a parameter. This split-phase design enables concurrent reads: `tx.query(&world)` uses `World::query_raw(&self)` (shared-ref, no ticks/cache, requires `ReadOnlyWorldQuery`), while writes go through `tx.write()`, `tx.remove()`, `tx.spawn()` which buffer into an internal `EnumChangeSet`. On successful commit, the changeset is applied atomically to World.

Three built-in strategies: `Sequential` (zero-cost passthrough — all ops delegate directly to World, commit always succeeds), `Optimistic` (live reads via `query_raw`, buffered writes into `EnumChangeSet`, tick-based validation at commit — `Err(Conflict)` if any accessed column was modified since begin, default 3 retries), `Pessimistic` (cooperative per-column locks acquired at begin, buffered writes, commit always succeeds — locks released on drop, default 64 retries with spin+yield backoff). `Optimistic` and `Pessimistic` are constructed with `::new(&world)` to capture a shared orphan queue handle.

Three tiers of mutation:

| Tier | API | Durability | Use case |
|---|---|---|---|
| Direct | `world.spawn()`, `world.insert()`, etc. | None | Single-threaded, no conflict detection |
| Transactional | `strategy.transact(world, access, \|tx, world\| { ... })` | In-memory | Concurrent access with conflict detection/retry |
| Durable | `Durable::new(strategy, wal, codecs).transact(...)` | WAL-backed | Crash-safe persistence via `minkowski-persist` crate |

`Durable<S>` (in `minkowski-persist`) wraps any `Transact` strategy. On successful commit, the forward changeset is written to the WAL before being applied to World. Failed attempts (retries) are not logged. WAL write failure panics — the durability invariant is non-negotiable.

Lock granularity is per-column `(ArchetypeId, ComponentId)`. `ColumnLockTable` is owned by `Pessimistic` strategy (not World — it's concurrency policy, not storage). Not MVCC — no version chains. Optimistic uses existing `changed_tick` infrastructure for validation. `World::query_raw(&self)` is the shared-ref read path — scans archetypes without touching cache or ticks.

**Entity ID lifecycle invariant**: entity IDs allocated during a transaction (`tx.spawn`) are tracked. On successful commit they become placed entities. On abort (drop without commit or conflict), the IDs are pushed to a shared `OrphanQueue` owned by World. World drains this queue automatically at the top of every `&mut self` method — bumping generations and recycling indices. No entity ID ever leaks, regardless of how the transaction ends. No manual drain step required.

### Reducer System

Typed reducers narrow what a closure *can* touch so that conflict freedom is provable from the type signature. Three execution models:

| Model | Handle types | Isolation | Conflict detection |
|---|---|---|---|
| Transactional | `EntityMut<C>`, `Spawner<B>`, `QueryWriter<Q>` | Buffered writes via EnumChangeSet | Runtime (optimistic ticks or pessimistic locks) |
| Scheduled | `QueryMut<Q>`, `QueryRef<Q>` | Direct `&mut World` (hidden) | Compile-time (Access bitsets) |
| Dynamic | `DynamicCtx` | Buffered writes via EnumChangeSet | Conservative (builder-declared upper bounds) |

**ComponentSet** declares a set of component types with pre-resolved IDs. **Contains<T, INDEX>** uses a const generic index to avoid coherence conflicts with generic tuple impls — the compiler infers INDEX at call sites. Both are macro-generated for tuples 1–12.

**Typed handles** hide World behind a facade exposing exactly the declared operations: `EntityRef<C>` (read-only), `EntityMut<C>` (read + buffered write + remove + optional despawn), `Spawner<B>` (entity creation via `reserve()`), `QueryRef<Q>` (read-only iteration), `QueryMut<Q>` (read-write iteration), `QueryWriter<Q>` (buffered query iteration via `WritableRef<T>`). EntityMut and Spawner hold `&mut EnumChangeSet` (not `&mut Tx`) for clean borrow splitting — Tx retains lifecycle ownership, handles borrow disjoint fields. `EntityMut::remove()` is bounded by `Contains<T, IDX>`. `EntityMut::despawn()` requires `register_entity_despawn` (sets despawn flag on Access).

**QueryWriter** iterates like a query but buffers writes through `ChangeSet` instead of mutating directly. `&T` items pass through unchanged; `&mut T` items become `WritableRef<T>` handles with `get`/`set`/`modify` methods. Uses manual archetype scanning (not `world.query()`) to avoid marking mutable columns as changed, which would cause self-conflict with optimistic validation. A separate `WriterQuery` trait (not on `WorldQuery`) defines the `&mut T` → `WritableRef<T>` mapping. Per-reducer `AtomicU64` tick state enables `Changed<T>` filter support. Compatible with `Durable` for WAL logging — the motivating use case.

**Dynamic reducers** trade compile-time precision for runtime flexibility. `DynamicReducerBuilder` (via `registry.dynamic(name, &mut world)`) declares upper-bound access with `can_read::<T>()`, `can_write::<T>()`, `can_spawn::<B>()`, `can_remove::<T>()`, `can_despawn()`. `DynamicCtx` provides `read`/`try_read`/`write`/`try_write`/`spawn`/`remove`/`try_remove`/`despawn`/`for_each`/`for_each_chunk` — accessing undeclared types, writing to read-only components, removing undeclared components, despawning without declaration, and iterating undeclared components all panic in all builds. `for_each::<Q>()` takes a `ReadOnlyWorldQuery` type parameter — the iteration is fully typed, only the access validation is dynamic. Supports `Changed<T>` via per-reducer `Arc<AtomicU64>` tick state updated post-commit. Component IDs pre-resolved at registration; O(1) HashMap lookup by `TypeId` at runtime via `DynamicResolved`.

**ReducerRegistry** is external to World (same composition pattern as SpatialIndex). Registration type-erases closures with Access metadata and pre-resolved ComponentIds. Dispatch: `call()` for transactional reducers (entity, spawner, query writer — runs through `strategy.transact()`, entity is part of args), `run()` for scheduled query reducers (direct `&mut World`), `dynamic_call()` for dynamic reducers (routes through `strategy.transact()`). `id_by_name()` / `dynamic_id_by_name()` enable network dispatch. `access()` / `dynamic_access()` enable scheduler conflict analysis.

**EntityAllocator::reserve(&self)** provides lock-free entity ID allocation via `AtomicU32` — enables `Spawner` to work inside transactional closures where only `&World` is available.

## Key Conventions

- `pub` for user-facing API (`World`, `Entity`, `CommandBuffer`, `Bundle`, `WorldQuery`, `Table`, `EnumChangeSet`, `Changed`, `ChangeTick`, `ComponentId`, `SpatialIndex`, `Access`, `BTreeIndex`, `HashIndex`, `Transact`, `Tx`, `Sequential`, `SequentialTx`, `Optimistic`, `Pessimistic`, `Conflict`, `ReducerRegistry`, `ReducerId`, `QueryReducerId`, `DynamicReducerId`, `DynamicReducerBuilder`, `DynamicCtx`, `ComponentSet`, `Contains`, `EntityRef`, `EntityMut`, `QueryRef`, `QueryMut`, `QueryWriter`, `WritableRef`, `WriterQuery`, `Spawner`, `WorldStats`). `pub(crate)` for internals (`BlobVec`, `Archetype`, `EntityAllocator`, `QueryCacheEntry`, `Tick`, `ColumnLockTable`, `OrphanQueue`, `TxCleanup`, `ResolvedComponents`, `DynamicResolved`). `ComponentRegistry` is `#[doc(hidden)] pub` — exposed only for derive macro codegen, not user code.
- `extern crate self as minkowski;` at crate root — allows `#[derive(Table)]` generated code (which references `::minkowski::*`) to resolve when used inside this crate's own tests.
- `#![allow(private_interfaces)]` at crate root — pub traits reference pub(crate) types in signatures. Intentional; fix when building public API facade.
- Every module has `#[cfg(test)] mod tests` with inline tests.
- `#[allow(dead_code)]` on fields/methods reserved for future phases.
- **Change detection invariant**: every path that hands out a mutable pointer to column data must either use `BlobVec::get_ptr_mut(row, tick)` (marks the column changed) or mark via the entry-point method (`World::query` for `&mut T`, `query_table_mut`, `query_table_raw`, `get_batch_mut`). `BlobVec::get_ptr` is the read path — writing through it silently bypasses `Changed<T>`. If you add a new mutable accessor, it must go through one of these two mechanisms.
- **Semantic review checklist**: every new primitive that touches concurrency, entity lifecycle, or cross-system state gets a "what can go wrong" review before implementation. The type system catches syntax; these catch design:
  1. Can this be called with the wrong World?
  2. Can Drop observe inconsistent state?
  3. Can two threads reach this through `&self`?
  4. Does dedup/merge/collapse preserve the strongest invariant?
  5. What happens if this is abandoned halfway through?
  6. Can a type bound be violated by a legal generic instantiation?
  7. Does the API surface of this handle permit any operation not covered by the Access bitset?
- **Transaction safety invariants**: any query path reachable from `&World` must be bounded by `ReadOnlyWorldQuery`. Any shared structure between World and a strategy uses `Arc` with a `WorldId` check at every entry point. Lock privilege in a `ColumnLockSet` can only escalate, never downgrade. Drop is the abort path — if a transaction can allocate engine resources (entity IDs, locks), it must be able to release them from Drop without `&mut World`, which means those resources route through interior-mutable shared handles (`OrphanQueue`, `Mutex<ColumnLockTable>`).
- **Assert boundary rule**: if violating a check makes the scheduler's Access bitset disagree with reality, it's `assert!`. The scheduler runs in release builds; the checks that protect its invariants must also run in release builds. `debug_assert!` is for checks within an already-correct access boundary — like verifying that a read through `can_write` still works (over-declared but sound). Crossing the boundary from read declaration to write operation is unsound, full stop.
- **Verify-before-design rule**: before designing features that depend on existing APIs, `grep` for the actual current methods/types. Confirm which ones exist vs which need to be created. Past bugs: assuming `EntityAllocator::reserve()` had atomics (it didn't exist), proposing `&mut World` in reducer APIs (unsound). The type system doesn't catch "assumed this method exists" — verification does.
- **Derive macro visibility rule**: after implementing or modifying derive macros or code generation, always test the generated code from an external example/binary target (`minkowski-examples`) to verify `pub` vs `pub(crate)` visibility is correct. In-crate tests won't catch `pub(crate)` leaking into generated code.
- **Change detection audit rule**: when implementing change detection or mutation tracking, enumerate ALL mutation paths (spawn, get_mut, insert, remove, query `&mut T`, query_table_mut, query_table_raw, changeset apply, any new accessor) and verify each path triggers detection. Consider same-tick interleaving edge cases where two mutations in the same tick must both be visible.
- **Bypass-path invariant rule**: when adding a new code path that skips the normal access/mutation pipeline (e.g., `query_table_mut` bypassing `world.query()`, manual archetype scanning in QueryWriter, `query_raw` skipping cache/ticks), verify that ALL existing invariants are maintained through the bypass: change detection ticks, query cache invalidation, Access bitset accuracy, entity lifecycle tracking. The normal pipeline enforces these automatically; bypass paths must enforce them manually or explicitly document which invariants they intentionally skip and why.
- **Bug fix workflow rule**: when a specific bug is reported, address it directly with targeted investigation — don't route through brainstorming or design exploration workflows. Brainstorming is for new features and design decisions, not concrete bugs with reproduction steps.
- **Reducer determinism rule**: reducers must be pure functions of their handle state and args. No RNG, system time, HashMap iteration, I/O, or global mutable state. When writing example reducers, use deterministic alternatives (BTreeMap, args-provided seeds, pre-computed values). See `docs/reducer-correctness.md`.
- **Drop cleanup rule**: if Drop needs to clean up engine state, the cleanup path must be reachable from `&self`. This constrains where state can live. If it's on World, Drop can't reach it. If it's behind `Arc<Mutex<_>>` shared between World and the transaction, Drop can. Every future resource that transactions can allocate — entity IDs, archetype slots, reserved capacity — must follow this pattern or it will leak on abort.

# Unifying principle behind every ID type in the system:

| Type | Proof at construction | Residue after erasure |
|---|---|---|
| `ComponentId` | `T: Component`, layout, drop fn | `usize` |
| `ArchetypeId` | Sorted component set, column allocation | `usize` |
| `Entity` | Generational uniqueness, archetype placement | `u64` |
| `ReducerId` | Typed closure, access set, resolved components, args type | `usize` |
| `QueryReducerId` | `WorldQuery` bound, column access pattern | `usize` |
| `WorldId` | Unique allocator identity | `u64` |
| `TxId` | Transaction lifecycle, cleanup registration | `u64` |
| `ChangeTick` | Monotonic ordering, causal relationship | `u64` |

Every one of these is a small integer that stands in for a proof the type system already verified. The runtime never re-checks the proof — it trusts the integer because the only way to obtain it was through a typed construction path that enforced the invariants.

The corollary is that forging an ID — constructing one from a raw integer without going through the typed path — is the one thing that can violate the system's guarantees. Which is why all the constructors are `pub(crate)` or behind registration APIs. The user can hold IDs, copy them, store them, send them. They can't fabricate them. The type system did its work at the gate and the ID is the stamp that says "verified."

## Dependencies

| Crate | Purpose |
|---|---|
| `fixedbitset` | Archetype component bitmasks for query matching |
| `parking_lot` | Mutex for `ColumnLockTable`, `OrphanQueue`, `Durable` WAL lock |
| `rayon` | `par_for_each` parallel iteration |
| `minkowski-derive` | `#[derive(Table)]` proc macro (syn/quote/proc-macro2) |
| `minkowski-persist` | WAL, snapshots, codec registry, `Durable<S>` wrapper |
| `minkowski-observe` | Observability companion: metrics capture, diff, display |
| `rkyv` (persist) | Zero-copy serialization for WAL records and snapshots |
| `memmap2` (persist) | Memory-mapped file I/O for zero-copy snapshot loading |
| `thiserror` (persist) | Derive macros for `std::error::Error` impls |
| `criterion` (dev) | Benchmark harness |
| `tempfile` (dev, bench) | Temporary directories for serialize benchmarks |
| `fastrand` (examples) | Example RNG |
