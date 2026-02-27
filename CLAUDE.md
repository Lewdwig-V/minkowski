# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo test -p minkowski --lib          # Unit tests (60 tests, fast)
cargo test -p minkowski                # All tests including doc tests
cargo test -p minkowski -- entity      # Run tests matching a filter

cargo clippy --workspace --all-targets -- -D warnings   # Lint (strict, warnings are errors)
cargo fmt --all                                          # Format

cargo bench -p minkowski               # All criterion benchmarks
cargo bench -p minkowski -- spawn      # Single benchmark

cargo run -p minkowski --example boids --release   # Boids simulation (5K entities, 1K frames)

MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski --lib  # UB check
```

Miri flags: `-Zmiri-tree-borrows` because crossbeam-epoch (rayon dep) violates Stacked Borrows; `-Zmiri-ignore-leaks` because rayon's thread pool intentionally outlives main.

Pre-commit hooks run `cargo fmt` and `cargo clippy -D warnings` on commit, `cargo test` on push.

## Architecture

Minkowski is a **column-oriented archetype ECS**. Two crates: `minkowski` (core) and `minkowski-derive` (proc-macro placeholder for Phase 2).

### Storage Model

Each unique set of component types gets an **Archetype** — a struct containing parallel `BlobVec` columns (type-erased `Vec<T>` storing raw bytes via `Layout`) plus a `Vec<Entity>` for row-to-entity mapping. A `FixedBitSet` on each archetype tracks which `ComponentId`s it contains, enabling fast query matching via bitwise subset checks.

**Entity** = u64 bit-packed: low 32 bits = index, high 32 bits = generation. `EntityAllocator` maintains a generation array + free list. `entity_locations: Vec<Option<EntityLocation>>` maps entity index → (archetype_id, row) for O(1) lookup.

**Sparse components** (`HashMap<Entity, T>` per component behind `Box<dyn Any>`) are opt-in via `register_sparse`. Not stored in archetypes.

### Data Flow

`world.spawn((Pos, Vel))` → `Bundle::component_ids` registers types → `Archetypes::get_or_create` finds/creates archetype by sorted component ID set → `Bundle::put` writes each component into BlobVec columns via raw pointer copy → `EntityLocation` recorded.

`world.query::<(&mut Pos, &Vel)>()` → `WorldQuery::required_ids` builds FixedBitSet → filter archetypes by `required.is_subset(&arch.component_ids)` → `init_fetch` grabs raw column pointers → `QueryIter` yields items via pointer arithmetic (`ptr.add(row)`).

### Archetype Migration

`world.insert(entity, NewComponent)` moves an entity from archetype A to A∪{new}: copies each column via `BlobVec::push` + `swap_remove_no_drop`, writes new component, updates all `EntityLocation`s (including the entity swapped into the vacated row). `get_pair_mut` uses `split_at_mut` for safe double-mutable archetype access.

### Key Traits

- **Component**: marker, blanket impl for `'static + Send + Sync`
- **Bundle** (unsafe): tuple impls 1-12 via `impl_bundle!` macro. `component_ids()` registers + sorts + deduplicates. `put()` yields component pointers via `ManuallyDrop`.
- **WorldQuery** (unsafe): tuple impls 1-12 via `impl_world_query_tuple!` macro. `Fetch` type holds `ThinSlicePtr<T>` (raw pointer wrapped for Send+Sync). Impls for `&T`, `&mut T`, `Entity`, `Option<&T>`.

### Deferred Mutation

`CommandBuffer` stores `Vec<Box<dyn FnOnce(&mut World) + Send>>`. Used during query iteration when structural changes (spawn/despawn/insert/remove) must be deferred. Applied via `cmds.apply(&mut world)`.

## Key Conventions

- `pub` for user-facing API (`World`, `Entity`, `CommandBuffer`, `Bundle`, `WorldQuery`). `pub(crate)` for internals (`BlobVec`, `Archetype`, `ComponentRegistry`, `EntityAllocator`).
- `#![allow(private_interfaces)]` at crate root — pub traits reference pub(crate) types in signatures. Intentional; fix when building public API facade.
- Every module has `#[cfg(test)] mod tests` with inline tests.
- `#[allow(dead_code)]` on fields/methods reserved for future phases.

## Dependencies

| Crate | Purpose |
|---|---|
| `fixedbitset` | Archetype component bitmasks for query matching |
| `rayon` | `par_for_each` parallel iteration |
| `criterion` (dev) | Benchmark harness |
| `hecs` (dev) | Benchmark comparison target |
| `fastrand` (dev) | Boids example RNG |
