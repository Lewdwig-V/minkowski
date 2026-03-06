# Documentation Uplift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite README as hackathon showcase, create 10 ADRs for delivered decisions, add tiered rustdocs to the public API.

**Architecture:** Three independent workstreams (README, ADRs, rustdocs) that touch completely different files. Can be parallelized via worktree-isolated subagents. Single PR at the end.

**Tech Stack:** Markdown (README, ADRs), Rust doc comments (rustdocs)

---

### Task 1: README.md Rewrite

**Files:**
- Rewrite: `README.md`

**Context:** The current README is 409 lines of architecture documentation with an outdated roadmap, Phase 5 promises (B-tree, Volcano, rkyv, replication), and a build roadmap with checkboxes. It doesn't mention the reducer system or AI skills. It needs to become a hackathon showcase.

**Step 1: Write the new README**

Replace `README.md` entirely with the following structure. Read the existing README first, then overwrite.

**Structure (in order):**

1. **Title + tagline** — `# Minkowski` then one-line: "A column-oriented database engine built from scratch by one human and one AI."

2. **Hero paragraph** — 3-4 sentences covering: what it is (archetype ECS that's also a transactional database), what's novel (typed reducer system proving conflict freedom from signatures, split-phase transactions enabling safe parallelism, WAL persistence composing with any strategy), scale (295 tests, 8 examples, Miri clean, built in one week).

3. **What makes this interesting** — 5 bullets:
   - Column-oriented ECS that's also a transactional database engine
   - Typed reducers: closures that prove conflict freedom from their type signature
   - Split-phase transactions: Tx doesn't hold `&mut World`, enabling safe concurrent reads
   - AI-powered developer tooling: skills and commands that teach the paradigm
   - Built from scratch in one week — 26 PRs, 295 tests, Miri verified

4. **Quick start** — Compact code block (~35 lines) showing:
   ```rust
   use minkowski::{World, ReducerRegistry, QueryMut, QueryRef};

   #[derive(Clone, Copy)]
   struct Pos { x: f32, y: f32 }
   #[derive(Clone, Copy)]
   struct Vel { dx: f32, dy: f32 }

   let mut world = World::new();
   let mut registry = ReducerRegistry::new();

   // Spawn entities
   for i in 0..1000 {
       world.spawn((Pos { x: i as f32, y: 0.0 }, Vel { dx: 1.0, dy: 0.5 }));
   }

   // Register a query reducer — type signature declares access
   let move_id = registry.register_query::<(&mut Pos, &Vel), (), _>(
       &mut world,
       "movement",
       |mut query: QueryMut<'_, (&mut Pos, &Vel)>, ()| {
           query.for_each(|(pos, vel)| {
               pos.x += vel.dx;
               pos.y += vel.dy;
           });
       },
   );

   // Dispatch — scheduler can prove this is conflict-free
   registry.run(&mut world, move_id, ());
   ```

5. **Features** — Subsections with `##` headers. Each: 2-4 sentences + optional 3-5 line code snippet.

   - **Column-Oriented Storage** — BlobVec columns in archetypes, 64-byte aligned, SIMD-friendly. Entity = generational u64.
   - **Query Engine** — Bitset matching, incremental cache, `for_each_chunk` for auto-vectorization, `par_for_each` for rayon, `Changed<T>` for archetype-level skip.
   - **Typed Reducers** — Three execution models (transactional/scheduled/dynamic). Six handle types. ReducerRegistry for conflict detection + dispatch. Brief table of the 6 handles with one-line descriptions.
   - **Transactions** — Sequential/Optimistic/Pessimistic. Split-phase design. Code snippet showing `strategy.transact(...)`.
   - **Persistence** — WAL + snapshots via `minkowski-persist`. Durable wrapper composes with any strategy. Code snippet showing `Durable::new(strategy, wal, codecs)`.
   - **Schema & Mutation** — `#[derive(Table)]` for compile-time schemas. `EnumChangeSet` for data-driven mutations with automatic undo. `CommandBuffer` for deferred structural changes.
   - **Spatial Indexing** — `SpatialIndex` trait, external composition pattern. Two implementations in examples (grid, quadtree).

6. **Examples** — Table format:

   | Example | What it exercises | Run |
   |---|---|---|
   | `boids` | Query reducers, SpatialGrid, parallel force computation | `cargo run -p minkowski-examples --example boids --release` |
   | `life` | QueryMut reducer, Table, EnumChangeSet undo/redo, Changed<T> | `...` |
   | `nbody` | Query reducer, Barnes-Hut quadtree, SpatialIndex trait | `...` |
   | `scheduler` | ReducerRegistry conflict detection, greedy batch scheduling | `...` |
   | `transaction` | Sequential/Optimistic/Pessimistic strategies, query reducers | `...` |
   | `battle` | EntityMut reducers, rayon parallel snapshots, tunable conflict | `...` |
   | `persist` | QueryWriter reducer, Durable WAL, snapshot save/load/recovery | `...` |
   | `reducer` | All 6 handle types, structural mutations, dynamic reducers | `...` |

7. **AI-Assisted Development** — Section explaining:
   - Built with Claude Code from first commit to last
   - Auto-triggering skill (`.claude/skills/minkowski-guide.md`) provides passive expertise as you code
   - 8 slash commands (`/minkowski:model`, `/minkowski:query`, `/minkowski:reducer`, etc.) for guided decision-making
   - Skills reference the architecture but teach the *paradigm* — which reducer to use, which concurrency model, how to optimize

8. **Architecture Decision Records** — One paragraph: "Design decisions are documented as ADRs in `docs/adr/`. Each records what was decided, what alternatives were considered, and what trade-offs were accepted." Link to the directory.

9. **Building & Testing** — Compact:
   ```
   cargo test -p minkowski          # 295 tests
   cargo clippy --workspace -- -D warnings
   cargo bench -p minkowski         # criterion benchmarks vs hecs
   ```

10. **Roadmap** — Table of stretch goals:

    | Feature | Rationale |
    |---|---|
    | Query planning (Volcano model) | Optimize complex queries across indexes |
    | B-tree / hash indexes | O(log n) lookups by column value |
    | rkyv zero-copy snapshots | Zero-copy deserialization matching BlobVec layout |
    | Replication & sync | Filtered WAL replay for read replicas and client mirrors |

11. **License** — MPL 2.0, same as current.

**Step 2: Verify**

Run: `cargo doc -p minkowski --no-deps 2>&1 | grep -c warning` — should not increase (README doesn't affect this, but verify nothing broke).

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as hackathon showcase"
```

---

### Task 2: ADRs

**Files:**
- Create: `docs/adr/001-column-oriented-archetype-storage.md` through `docs/adr/010-ai-assisted-developer-tooling.md`
- Delete: all 43 files in `docs/plans/` EXCEPT `2026-03-05-documentation-uplift-design.md` and `2026-03-06-documentation-uplift-impl.md` (current plan)

**Context:** Each ADR documents a delivered feature cluster. Keep them concise — ~40-60 lines each. Use the format from the design doc. Dates come from the PR that delivered each feature.

**Step 1: Create `docs/adr/` directory and write all 10 ADRs**

**ADR-001: Column-Oriented Archetype Storage** (Date: 2026-02-28, PR #1-#7)
- Context: Need storage that combines ECS runtime flexibility with database performance characteristics
- Decision: Type-erased BlobVec columns grouped into archetypes, 64-byte aligned, with generational entity IDs and optional sparse components
- Alternatives: Row-oriented storage (poor cache utilization), pure column store without archetypes (loses entity identity), SoA with fixed schemas only (loses runtime flexibility)
- Consequences: Cache-friendly iteration, SIMD-compatible column layout, O(1) entity lookup, archetype migration cost on structural changes

**ADR-002: Table Derive Macro** (Date: 2026-03-01, PR #16)
- Context: Dynamic archetype queries require bitset matching per query — overhead for known schemas
- Decision: `#[derive(Table)]` proc macro pre-registers archetype and caches column offsets. `query_table`/`query_table_mut` bypass archetype matching entirely
- Alternatives: Manual schema registration (verbose), query specialization (compiler-dependent), cached-only dynamic queries (still pays matching cost)
- Consequences: Zero-overhead typed row access for known schemas, dynamic queries still available for ad-hoc access, derive macro generates `pub` code that must be visible from external crates

**ADR-003: Change Detection via Tick Tracking** (Date: 2026-02-28, PR #7)
- Context: Need to skip unchanged data during iteration without per-entity tracking overhead
- Decision: Per-column `changed_tick: Tick` (monotonic u64), auto-advancing on every mutable access. `Changed<T>` filter skips entire archetypes whose column tick is older than the query's last read tick
- Alternatives: Per-entity dirty flags (too fine-grained, high overhead), frame-based detection (requires manual tick management), observer pattern (allocation per change)
- Consequences: Zero-cost at write site (pessimistic marking), archetype-granularity skip, no user-facing tick management. Every mutable access path must mark — invariant enforced by convention

**ADR-004: Mutation Strategies** (Date: 2026-02-28, PR #8-#9)
- Context: Need deferred structural mutation during iteration AND data-driven mutations for persistence/undo
- Decision: Two complementary systems: `CommandBuffer` (closure-based deferred ops) and `EnumChangeSet` (data-driven mutations with automatic reverse generation)
- Alternatives: Single unified mutation system (over-constrains either use case), immediate mutation with borrow-checker workarounds (unsound), event sourcing only (high overhead for simple cases)
- Consequences: CommandBuffer for iteration-time structural changes, EnumChangeSet for WAL serialization and undo/redo, two systems to learn but each optimized for its use case

**ADR-005: Spatial Index Trait** (Date: 2026-03-01, PR #14)
- Context: Need spatial queries (neighbor search, range queries) without coupling index implementations to World
- Decision: Two-method trait (`rebuild` + optional `update`), fully external to World, composes from existing query primitives. No generic query method, no component type parameters, no stored World reference, no World registration
- Alternatives: Built-in grid (too specific), generic query method on trait (can't express grid vs quadtree vs BVH query shapes), World-integrated indexes (grows World API surface)
- Consequences: Structurally different algorithms (grid, quadtree) fit without friction. Stale entity handles handled via generational validation. Users define their own query methods per concrete type

**ADR-006: Query Conflict Detection** (Date: 2026-03-01, PR #17)
- Context: Framework-level schedulers need to detect data races between systems at registration time
- Decision: `Access` struct with per-component read/write bitsets + despawn flag. `conflicts_with()` applies standard read-write lock rule via two bitwise ANDs. Minkowski provides metadata; scheduling policy is the framework's responsibility
- Alternatives: Built-in scheduler (framework-level concern, not storage-level), runtime-only detection (too late), per-entity tracking (too fine-grained for system-level analysis)
- Consequences: O(1) conflict detection via bitset ops. Framework authors build schedulers on top. Composable with reducer system for type-level conflict proofs

**ADR-007: Split-Phase Transactions** (Date: 2026-03-02, PR #18-#20)
- Context: Need concurrent transaction execution without sacrificing soundness
- Decision: Tx doesn't hold `&mut World`. Methods take world as parameter. Three phases: begin (`&mut World`) → execute (`&World`, parallel) → commit (`&mut World`). `tx.query()` requires `ReadOnlyWorldQuery` to prevent aliased `&mut T`. Three strategies: Sequential (zero-cost), Optimistic (tick validation), Pessimistic (cooperative locks). Entity lifecycle closed through shared `OrphanQueue`
- Alternatives: MVCC (version chains too complex for ECS), lock-per-entity (too fine-grained), `&mut World` in Tx (prevents parallel reads — the fundamental soundness issue)
- Consequences: Safe parallel transaction execution, zero-cost sequential path, entity IDs never leak regardless of commit/abort. WorldId prevents cross-world corruption

**ADR-008: WAL + Snapshot Persistence** (Date: 2026-03-02, PR #19-#20)
- Context: Need crash-safe persistence without serializing the entire world on every write
- Decision: `Durable<S, W>` wraps any `Transact` strategy, adding WAL logging on successful commit. `Snapshot` captures full world state at a point-in-time. Recovery = load snapshot + replay WAL from snapshot's sequence number. `CodecRegistry` maps `ComponentId` → serde codec
- Alternatives: Full-world flush per frame (too expensive), mmap (crash consistency + TLB pressure), custom binary format (maintenance burden)
- Consequences: Composable — any transaction strategy gets durability by wrapping with `Durable`. WAL write failure panics (durability invariant is non-negotiable). Schema changes require migration or snapshot wipe

**ADR-009: Typed Reducer System** (Date: 2026-03-04, PR #21-#25)
- Context: Need to narrow what a closure can touch so conflict freedom is provable from the type signature, not just detectable at runtime
- Decision: Three execution models — Transactional (EntityMut, Spawner, QueryWriter — buffered writes via EnumChangeSet), Scheduled (QueryMut, QueryRef — direct `&mut World`), Dynamic (DynamicCtx — runtime-validated access). Six typed handles hide World behind a facade. ReducerRegistry type-erases closures with Access metadata for conflict analysis and dispatch
- Alternatives: Raw `&mut World` in closures (unsound for concurrent use), capability tokens without handles (verbose, error-prone), single execution model (over-constrains different use cases)
- Consequences: Type signatures prove what a reducer touches. Registry enables both compile-time scheduling and runtime dispatch by name. Three models cover the spectrum from maximum safety to maximum flexibility. Contains<T, INDEX> const generic solves coherence for component set tuples

**ADR-010: AI-Assisted Developer Tooling** (Date: 2026-03-05, PR #22, #26)
- Context: Minkowski is an unusual database — users won't intuitively know which reducer to use, which concurrency model fits, how to model data, or how to optimize queries
- Decision: One auto-triggering skill (`.claude/skills/minkowski-guide.md`) provides passive expertise. Eight slash commands (`/minkowski:model`, `/minkowski:query`, `/minkowski:reducer`, `/minkowski:concurrency`, `/minkowski:mutate`, `/minkowski:persist`, `/minkowski:index`, `/minkowski:optimize`) guide specific decisions. Skills reference CLAUDE.md architecture but teach the paradigm, not the internals
- Alternatives: Traditional documentation only (doesn't meet users where they are), built-in help system (not interactive), tutorial-style guides (static, can't adapt to user's code)
- Consequences: Claude Code users get contextual guidance as they code. Skills auto-trigger on relevant keywords. Commands provide Socratic decision-making for key choices. Skill content must stay in sync with the evolving API

**Step 2: Delete old plan files**

```bash
# Delete all plan files except the current uplift design and impl
find docs/plans/ -name "*.md" ! -name "2026-03-05-documentation-uplift*" ! -name "2026-03-06-documentation-uplift*" -delete
```

**Step 3: Verify**

Confirm 10 ADR files exist and plan directory is clean:
```bash
ls docs/adr/ | wc -l   # should be 10
ls docs/plans/ | wc -l  # should be 2 (design + impl)
```

**Step 4: Commit**

```bash
git add docs/adr/ && git add -u docs/plans/
git commit -m "docs: 10 ADRs for delivered decisions, remove planning docs"
```

---

### Task 3: Rustdocs — Crate-Level and Module-Level

**Files:**
- Modify: `crates/minkowski/src/lib.rs` (add crate-level `//!` docs)
- Modify: `crates/minkowski/src/world.rs` (module not `pub mod` — add docs on World struct)
- Modify: `crates/minkowski/src/entity.rs`
- Modify: `crates/minkowski/src/access.rs`
- Modify: `crates/minkowski/src/changeset.rs`
- Modify: `crates/minkowski/src/command.rs`
- Modify: `crates/minkowski/src/component.rs`
- Modify: `crates/minkowski/src/index.rs`
- Modify: `crates/minkowski/src/table.rs`
- Modify: `crates/minkowski/src/bundle.rs`
- Modify: `crates/minkowski/src/query/mod.rs`
- Modify: `crates/minkowski/src/query/fetch.rs` (fix HTML warning)
- Modify: `crates/minkowski/src/query/iter.rs`
- Modify: `crates/minkowski/src/reducer.rs`
- Modify: `crates/minkowski/src/transaction.rs` (fix broken links)
- Modify: `crates/minkowski/src/storage/mod.rs`

**Context:** Currently `lib.rs` has no `//!` docs at all. Module files have sparse or no module-level docs. The goal is rich crate-level docs explaining the mental model, plus module-level docs on each `pub mod`.

**Step 1: Add crate-level docs to `lib.rs`**

Add `//!` block at the top of `lib.rs` (before the `#![allow]` line). Content:

```rust
//! # Minkowski
//!
//! A column-oriented database engine built on an archetype Entity-Component System.
//!
//! Minkowski combines the runtime flexibility of entity-component systems with
//! the performance characteristics of analytical databases: cache-friendly
//! columnar storage, SIMD-friendly iteration, and compile-time schema
//! declarations via [`Table`].
//!
//! ## Core concepts
//!
//! - **[`World`]** — the central store. Holds all entities, components, and
//!   archetype metadata. Most operations start here.
//! - **[`Entity`]** — a lightweight handle (generational u64) that identifies
//!   a row across archetypes.
//! - **Components** — any `'static + Send + Sync` type. Stored in contiguous
//!   columns within archetypes for cache-friendly iteration.
//! - **Queries** — [`world.query::<(&mut Pos, &Vel)>()`](World::query)
//!   iterates matching archetypes via bitset matching with incremental caching.
//!
//! ## Typed reducers
//!
//! The [`ReducerRegistry`] provides typed closures whose signatures declare
//! exactly which components they read and write. This enables:
//!
//! - **Compile-time conflict proofs** — the scheduler can verify two reducers
//!   touch disjoint data without running them.
//! - **Six handle types** — [`EntityRef`], [`EntityMut`], [`QueryRef`],
//!   [`QueryMut`], [`QueryWriter`], [`Spawner`] — each exposing only the
//!   operations the reducer declared.
//! - **Three execution models** — transactional (buffered writes),
//!   scheduled (direct `&mut World`), and dynamic (runtime-validated access
//!   via [`DynamicCtx`]).
//!
//! ## Transactions
//!
//! The [`Transact`] trait provides closure-based transactions with three
//! built-in strategies:
//!
//! - [`Sequential`] — zero-cost passthrough for single-threaded use.
//! - [`Optimistic`] — tick-based validation, retries on conflict.
//! - [`Pessimistic`] — cooperative per-column locks, guaranteed commit.
//!
//! All strategies use split-phase execution: [`Tx`] doesn't hold `&mut World`,
//! enabling concurrent reads via `tx.query(&world)` (bounded by
//! [`ReadOnlyWorldQuery`] to prevent aliased `&mut T`).
//!
//! ## Mutation
//!
//! - [`CommandBuffer`] — deferred structural changes during iteration.
//! - [`EnumChangeSet`] — data-driven mutations with automatic reverse
//!   generation for rollback. The serialization boundary for WAL persistence.
//!
//! ## Where to start
//!
//! 1. Create a [`World`] and spawn entities with component tuples.
//! 2. Query with [`world.query::<Q>()`](World::query) or register reducers
//!    on a [`ReducerRegistry`].
//! 3. For concurrency, wrap dispatch in a [`Transact`] strategy.
//! 4. For persistence, use [`minkowski_persist::Durable`] around any strategy.
//!
//! See the `examples/` directory for complete programs exercising every feature.
```

**Step 2: Fix the 4 existing rustdoc warnings**

In `crates/minkowski/src/transaction.rs`:
- Line 60: change `[`WorldId`]: crate::world::WorldId` to `[`WorldId`]: self::WorldId` or remove the link (WorldId is pub(crate))
- Line 184: change `[`TxCleanup`]` to backtick-only `` `TxCleanup` `` (it's pub(crate))

In `crates/minkowski/src/query/fetch.rs`:
- Line 78: change `Changed<T>` to `` `Changed<T>` `` (backtick to prevent HTML parsing)

**Step 3: Add module-level `//!` docs**

For each `pub mod` in lib.rs, add a brief `//!` doc at the top of the module file if one doesn't already exist. Modules that already have good docs (like `transaction.rs`) just need minor fixes.

Module docs to add/update:

- `world.rs`: No `//!` needed (it's `pub mod world` in lib.rs). Add doc on `World` struct itself (see Task 4).
- `query/mod.rs`: Add `//! Query engine — bitset matching, incremental caching, parallel iteration.`
- `query/iter.rs`: Add `//! Query iterators — sequential, parallel (rayon), and chunk-based iteration.`
- `storage/mod.rs`: Add `//! Storage internals — archetypes, BlobVec columns, sparse components.`
- `component.rs`: No `//!` needed. Add doc on `ComponentId` struct.
- `bundle.rs`: Add `//! Bundle trait — tuple-based entity construction.`

**Step 4: Verify**

```bash
cargo doc -p minkowski --no-deps 2>&1 | grep warning
# Expected: zero warnings
cargo test -p minkowski --doc 2>&1 | tail -5
# Expected: doc tests pass (or no doc tests yet)
```

**Step 5: Commit**

```bash
git add crates/minkowski/src/
git commit -m "docs: crate-level + module-level rustdocs, fix 4 warnings"
```

---

### Task 4: Rustdocs — Tier 1 Types (Full Treatment)

**Files:**
- Modify: `crates/minkowski/src/world.rs` — `World` struct
- Modify: `crates/minkowski/src/entity.rs` — `Entity` struct
- Modify: `crates/minkowski/src/reducer.rs` — `ReducerRegistry`, `EntityRef`, `EntityMut`, `QueryRef`, `QueryMut`, `QueryWriter`, `Spawner`, `DynamicCtx`, `DynamicReducerBuilder`
- Modify: `crates/minkowski/src/transaction.rs` — `Transact`, `Tx`, `Sequential`, `Optimistic`, `Pessimistic`
- Modify: `crates/minkowski/src/changeset.rs` — `EnumChangeSet`
- Modify: `crates/minkowski/src/index.rs` — `SpatialIndex` (already well-documented, verify)
- Modify: `crates/minkowski/src/command.rs` — `CommandBuffer`

**Context:** These are the types users interact with most. Each needs a doc comment explaining what it is, when to use it, and linking to related types. Code examples where they clarify usage. Read each file first to see what docs already exist.

**Guidelines for Tier 1 docs:**

- Every struct/trait gets `///` with: one-line summary, blank line, 2-4 sentence explanation, `# Example` section with compilable code where practical.
- For reducer handles: explain what the handle exposes and what it hides, with a usage snippet inside a `register_*` closure.
- For transaction types: explain the strategy's trade-offs and when to choose it.
- Cross-link related types with `[`TypeName`]` syntax.
- Do NOT add docs to methods that are already well-documented.
- Do NOT add doc examples that would require complex setup (world + spawn + register) — keep examples focused on the type itself.
- If a type already has good docs (like `SpatialIndex`, `Access`), verify and move on.

**Step 1: Read each file, add/update docs on Tier 1 types**

Work through each file. Add docs where missing, improve where sparse. Skip types that are already well-documented.

**Step 2: Verify**

```bash
cargo doc -p minkowski --no-deps 2>&1 | grep warning  # zero warnings
cargo test -p minkowski --doc  # doc tests pass
cargo test -p minkowski --lib  # unit tests still pass
cargo clippy --workspace --all-targets -- -D warnings  # no regressions
```

**Step 3: Commit**

```bash
git add crates/minkowski/src/
git commit -m "docs: tier-1 rustdocs — World, Entity, reducers, transactions, mutations"
```

---

### Task 5: Rustdocs — Tier 2 Types (One-Liners)

**Files:**
- Modify: `crates/minkowski/src/access.rs` — `Access` (already has example, verify)
- Modify: `crates/minkowski/src/component.rs` — `ComponentId`
- Modify: `crates/minkowski/src/query/fetch.rs` — `Changed`, `ReadOnlyWorldQuery`
- Modify: `crates/minkowski/src/reducer.rs` — `WritableRef`, `WriterQuery`, `ComponentSet`, `Contains`, `ReducerId`, `QueryReducerId`, `DynamicReducerId`
- Modify: `crates/minkowski/src/transaction.rs` — `Conflict`, `SequentialTx`

**Context:** These are supporting types users encounter but don't need deep documentation on. Each gets a one-line `///` summary + a link to the primary type they support.

**Guidelines:**
- One-line summary, then `///` blank line, then "See [`PrimaryType`] for usage."
- No code examples needed.
- If the type already has adequate docs, skip it.

**Step 1: Read each file, add one-liners where missing**

**Step 2: Verify**

```bash
cargo doc -p minkowski --no-deps 2>&1 | grep warning  # zero warnings
cargo test -p minkowski --lib  # still passes
```

**Step 3: Commit**

```bash
git add crates/minkowski/src/
git commit -m "docs: tier-2 rustdocs — supporting types with cross-links"
```

---

## Execution Notes

**Parallelization:** Tasks 1, 2, and 3+4+5 are independent workstreams:
- Agent A: Task 1 (README)
- Agent B: Task 2 (ADRs)
- Agent C: Tasks 3+4+5 (Rustdocs — must be sequential within this agent since they touch the same files)

All three agents can run in parallel worktrees. Cherry-pick all commits into a single branch for the PR.

**Final verification after cherry-pick:**
```bash
cargo test -p minkowski --lib
cargo test -p minkowski --doc
cargo doc -p minkowski --no-deps 2>&1 | grep warning
cargo clippy --workspace --all-targets -- -D warnings
```

**PR title:** `docs: comprehensive documentation uplift — README, ADRs, rustdocs`
