# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo test -p minkowski --lib          # Unit tests (118 tests, fast)
cargo test -p minkowski                # All tests including doc tests
cargo test -p minkowski -- entity      # Run tests matching a filter

cargo clippy --workspace --all-targets -- -D warnings   # Lint (strict, warnings are errors)
cargo fmt --all                                          # Format

cargo bench -p minkowski               # All criterion benchmarks
cargo bench -p minkowski -- spawn      # Single benchmark

cargo run -p minkowski-examples --example boids --release   # Boids simulation (5K entities, 1K frames)
cargo run -p minkowski-examples --example life --release    # Game of Life with undo + derive(Table) (64x64 grid, 500 gens)
cargo run -p minkowski-examples --example nbody --release   # Barnes-Hut N-body (2K entities, 1K frames)
cargo run -p minkowski-examples --example scheduler --release   # Access conflict detection demo (6 systems, 10 frames)
cargo run -p minkowski-examples --example transaction --release   # Transaction strategies demo (3 strategies, 100 entities)
cargo run -p minkowski-examples --example battle --release   # Multi-threaded battle with tunable conflict rates (500 entities, 100 frames)

MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- --skip par_for_each  # UB check (strict)
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski --lib par_for_each  # rayon tests
```

Miri flags: `-Zmiri-tree-borrows` because crossbeam-epoch (rayon dep) violates Stacked Borrows; `-Zmiri-ignore-leaks` only for the two `par_for_each` tests because rayon's thread pool intentionally outlives main. All other tests run without leak suppression to catch real leaks.

Pre-commit hooks run `cargo fmt` and `cargo clippy -D warnings` on commit, `cargo test` on push.

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every PR and push to main:

| Job | Toolchain | Command | `needs` |
|---|---|---|---|
| fmt | stable | `cargo fmt --all -- --check` | — |
| clippy | stable | `cargo clippy --workspace --all-targets -- -D warnings` | fmt |
| test | stable | `cargo test -p minkowski` | clippy |
| miri | nightly | Two-step Miri run (see Build & Test Commands) | test |

Sequential chain: fmt failure skips all downstream jobs. A `ci-pass` aggregator job (runs with `if: always()`) is the single required status check for branch protection — it explicitly verifies all four jobs succeeded, avoiding GitHub's "skipped = passed" loophole with chained `needs:`.

## Architecture

Minkowski is a **column-oriented archetype ECS**. Three crates: `minkowski` (core), `minkowski-derive` (`#[derive(Table)]` proc macro), and `minkowski-examples` (examples as external API consumers).

### Storage Model

Each unique set of component types gets an **Archetype** — a struct containing parallel `BlobVec` columns (type-erased `Vec<T>` storing raw bytes via `Layout`) plus a `Vec<Entity>` for row-to-entity mapping. A `FixedBitSet` on each archetype tracks which `ComponentId`s it contains, enabling fast query matching via bitwise subset checks.

**Entity** = u64 bit-packed: low 32 bits = index, high 32 bits = generation. `EntityAllocator` maintains a generation array + free list. `entity_locations: Vec<Option<EntityLocation>>` maps entity index → (archetype_id, row) for O(1) lookup.

**Sparse components** (`HashMap<Entity, T>` per component behind `Box<dyn Any>`) are opt-in via `register_sparse`. Not stored in archetypes.

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

`EnumChangeSet` is the data-driven alternative: mutations are recorded as a `Vec<Mutation>` enum with component bytes in a contiguous `Arena`. `apply()` returns a reverse `EnumChangeSet` for rollback — applying the reverse undoes the original changes. Useful for persistence (WAL serialization) and transactions. Typed safe helpers (`insert<T>`, `remove<T>`, `spawn_bundle<B>`) wrap the raw `record_*` methods — they auto-register component types and handle `ManuallyDrop` internally. The raw methods remain for power users who already have a `ComponentId`.

### Secondary Indexes

`SpatialIndex` is a lifecycle trait for user-owned spatial data structures (grids, quadtrees, BVH, k-d trees). Indexes are fully external to World — they compose from existing query primitives. The trait has two methods: `rebuild` (required, full reconstruction) and `update` (optional, defaults to rebuild — override for incremental updates via `Changed<T>`). Despawned entities are handled via generational validation: stale entries are skipped at query time when `world.is_alive()` returns false, and cleaned up on the next rebuild.

### System Scheduling Primitives

`Access` extracts component-level read/write metadata from any `WorldQuery` type. `Access::of::<(&mut Pos, &Vel)>(world)` returns a struct with two `FixedBitSet`s: reads (Vel) and writes (Pos). `conflicts_with()` detects whether two accesses violate the read-write lock rule — two bitwise ANDs over the component bitsets.

This is a building block for framework-level schedulers. Minkowski provides the access metadata; scheduling policy (dependency graphs, topological sort, parallel execution) is the framework's responsibility.

### Transaction Semantics

`TransactionStrategy` is a trait with one method: `begin(&self, world, access) -> Tx`. The Tx does NOT hold `&mut World` — methods take `world` as a parameter. This split-phase design enables concurrent reads: `tx.query(&world)` uses `World::query_raw(&self)` (shared-ref, no ticks/cache), while `tx.commit(&mut world)` validates and applies atomically.

Three built-in strategies: `Sequential` (zero-cost passthrough — all ops delegate directly to World, commit always succeeds), `Optimistic` (live reads via `query_raw`, buffered writes into `EnumChangeSet`, tick-based validation at commit — `Err(Conflict)` if any accessed column was modified since begin), `Pessimistic` (cooperative per-column locks acquired at begin, buffered writes, commit always succeeds — locks released on drop). `Optimistic` and `Pessimistic` are constructed with `::new(&world)` to capture a shared orphan queue handle.

Lock granularity is per-column `(ArchetypeId, ComponentId)`. `ColumnLockTable` is owned by `Pessimistic` strategy (not World — it's concurrency policy, not storage). Not MVCC — no version chains. Optimistic uses existing `changed_tick` infrastructure for validation. `World::query_raw(&self)` is the shared-ref read path — scans archetypes without touching cache or ticks.

**Entity ID lifecycle invariant**: entity IDs allocated during a transaction (`tx.spawn`) are tracked. On successful commit they become placed entities. On abort (drop without commit or conflict), the IDs are pushed to a shared `OrphanQueue` owned by World. World drains this queue automatically at the top of every `&mut self` method — bumping generations and recycling indices. No entity ID ever leaks, regardless of how the transaction ends. No manual drain step required.

## Key Conventions

- `pub` for user-facing API (`World`, `Entity`, `CommandBuffer`, `Bundle`, `WorldQuery`, `Table`, `EnumChangeSet`, `Changed`, `ComponentId`, `SpatialIndex`, `Access`, `TransactionStrategy`, `Sequential`, `Optimistic`, `Pessimistic`, `Conflict`). `pub(crate)` for internals (`BlobVec`, `Archetype`, `EntityAllocator`, `QueryCacheEntry`, `Tick`, `ColumnLockTable`, `OrphanQueue`). `ComponentRegistry` is `#[doc(hidden)] pub` — exposed only for derive macro codegen, not user code.
- `extern crate self as minkowski;` at crate root — allows `#[derive(Table)]` generated code (which references `::minkowski::*`) to resolve when used inside this crate's own tests.
- `#![allow(private_interfaces)]` at crate root — pub traits reference pub(crate) types in signatures. Intentional; fix when building public API facade.
- Every module has `#[cfg(test)] mod tests` with inline tests.
- `#[allow(dead_code)]` on fields/methods reserved for future phases.
- **Change detection invariant**: every path that hands out a mutable pointer to column data must either use `BlobVec::get_ptr_mut(row, tick)` (marks the column changed) or mark via the entry-point method (`World::query` for `&mut T`, `query_table_mut`, `query_table_raw`). `BlobVec::get_ptr` is the read path — writing through it silently bypasses `Changed<T>`. If you add a new mutable accessor, it must go through one of these two mechanisms.
- **Semantic review checklist**: every new primitive that touches concurrency, entity lifecycle, or cross-system state gets a "what can go wrong" review before implementation. The type system catches syntax; these catch design:
  1. Can this be called with the wrong World?
  2. Can Drop observe inconsistent state?
  3. Can two threads reach this through `&self`?
  4. Does dedup/merge/collapse preserve the strongest invariant?
  5. What happens if this is abandoned halfway through?
  6. Can a type bound be violated by a legal generic instantiation?
- **Transaction safety invariants**: any query path reachable from `&World` must be bounded by `ReadOnlyWorldQuery`. Any shared structure between World and a strategy uses `Arc` with a `WorldId` check at every entry point. Lock privilege in a `ColumnLockSet` can only escalate, never downgrade. Drop is the abort path — if a transaction can allocate engine resources (entity IDs, locks), it must be able to release them from Drop without `&mut World`, which means those resources route through interior-mutable shared handles (`OrphanQueue`, `Mutex<ColumnLockTable>`).
- **Drop cleanup rule**: if Drop needs to clean up engine state, the cleanup path must be reachable from `&self`. This constrains where state can live. If it's on World, Drop can't reach it. If it's behind `Arc<Mutex<_>>` shared between World and the transaction, Drop can. Every future resource that transactions can allocate — entity IDs, archetype slots, reserved capacity — must follow this pattern or it will leak on abort.

## Dependencies

| Crate | Purpose |
|---|---|
| `fixedbitset` | Archetype component bitmasks for query matching |
| `parking_lot` | Mutex for `ColumnLockTable`, `OrphanQueue` |
| `rayon` | `par_for_each` parallel iteration |
| `minkowski-derive` | `#[derive(Table)]` proc macro (syn/quote/proc-macro2) |
| `criterion` (dev) | Benchmark harness |
| `hecs` (dev) | Benchmark comparison target |
| `fastrand` (examples) | Example RNG |
