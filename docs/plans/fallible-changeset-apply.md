# Fallible Changeset Apply Design (#294)

## Problem

Everything that grows the world's pool through a public fallible API now returns `PoolExhausted` with the world unchanged: `try_spawn`, `try_insert`, `try_insert_sparse`, `import_page`, `try_restore_allocator_state`. `EnumChangeSet::apply` is the exception. Its spawn, insert, migration and sparse paths grow columns and bookkeeping through aborting calls, and its overwrites `expect` a scratch block. That leaves three user-visible failure modes:

1. **WAL tail replay** (`wal.rs:516`, `apply_replay`) aborts the process when a budgeted recovery (`recover_world_with_builder`, #291) fits the LSM baseline but not the tail.
2. **Transactions** commit, then apply: `forward.apply(world).expect("changeset apply after successful commit")` (`transaction.rs:742, 857`). Under `Durable` the WAL record is written in `try_commit` (`durable.rs:87`) *before* that apply. If apply could fail on the pool, the WAL would hold a record the world never applied, so a fallible apply on its own would turn an abort into divergence.
3. **Direct `apply`** on a budgeted world aborts instead of returning an error.

A related correctness gap found while verifying: **`despawn` allocates.** `BlobVec::begin_extract_swap` (`blob_vec.rs:343`) takes one pool scratch block per column whose component has a destructor (panic-safety from #235), and `World::despawn` / `despawn_batch` `panic!` if it fails (`world.rs:684, 774`). Changeset overwrites use the same scratch (`replace_protected`). #286's docs (`memory-management.md`, CHANGELOG, `pool-entity-bookkeeping.md` B3) state that despawn never allocates; that is true only for components without `Drop`.

## Current State

Verified on `main` (`c0e16d6`).

| Path in `apply_mutations` (`changeset.rs`) | Pool growth | On exhaustion today |
|---|---|---|
| Fast-lane batches (overwrite) | scratch per droppable overwrite | `expect` panic |
| `Insert` overwrite batch (`flush_insert_batch`) | scratch per droppable overwrite | `expect` panic |
| `Insert` migration (`changeset_insert_raw`) | target columns, target entity list; source extract scratch | abort / `expect` |
| `Remove` (`changeset_remove_raw`) | target columns, target entity list (may create the empty archetype) | abort |
| `Spawn` | `materialize_reserved` (generations, free list), `adopt` on replay, `entity_locations`, target columns, entity list | abort |
| `Despawn` (`World::despawn`) | extract scratch per droppable column | `panic!` |
| `SparseInsert` (`SparseStorage::insert_raw`) | sparse page, dense entities, dense values | abort |
| `SparseRemove` | none (swap-remove) | n/a |

`ApplyError` is `#[non_exhaustive]`; `TransactError` and `WalError::Apply` wrap it. `TransactError` is **not** `#[non_exhaustive]`.

## Proposed Design

Two parts: a resident scratch slot per droppable column, which makes overwrites and despawn allocation-free, and a reservation phase that reserves a whole changeset's growth before any mutation.

### Part 1: resident scratch slot

Each `BlobVec` with a `drop_fn` and non-zero size keeps one pool block of its item layout (`resident_scratch: Option<NonNull<u8>>`). It is allocated together with the column's first buffer (so it is covered by the column's own growth preflight) and freed in `Drop`. `replace_protected` and `begin_extract_swap` use it instead of allocating:

- Overwrite: copy the old value into the resident slot, write the new value into the row, drop the old value from the slot. The slot's bytes are dead afterwards and reusable.
- Extract (despawn): the same, with the drop deferred to `ExtractedSlot::drop_value`. `ExtractedSlot` borrows the column's slot instead of owning a pool block. `despawn` extracts one row per column at a time, so one slot per column suffices.

A panicking `Drop` still leaks at most the half-dropped value: the slot's bytes are never read again. `despawn` and every overwrite become allocation-free, which makes B3's "despawn never allocates" true for every component type.

Cost: one item-sized block per droppable column per archetype (for example 24 B for a `String` column). A slot is never shared between columns or archetypes, so nothing can re-enter it: component destructors have no world access.

### Part 2: reservation phase

New crate-internal method:

```rust
impl EnumChangeSet {
    /// Reserve every pool slot `apply` will need. On Err only spare capacity
    /// grew (and empty archetypes may have been created, as try_spawn does).
    pub(crate) fn try_reserve(&self, world: &mut World, adopt: bool) -> Result<(), PoolExhausted>;
}
```

It simulates the mutation sequence without touching data:

- **Per archetype:** a simulated length starting at `len()`. Spawns and migrations into it add one; despawns and migrations out subtract one. The reservation is the peak simulated length, because columns never shrink their capacity mid-apply. Reserving the peak up front therefore needs no more pool than sequential growth to that peak. Target archetypes are resolved with `get_or_create`, as `try_spawn` does.
- **Per touched entity:** a simulated location (`HashMap<Entity, Option<ArchetypeId>>`, filled lazily from `entity_locations`), only for entities that spawn, despawn or migrate. Overwrites need no entry.
- **Entity bookkeeping:** generations and free list (B3) up to `max(reserved, max adopted index + 1)`; `entity_locations` up to the highest spawned index + 1.
- **Sparse:** per component, the peak simulated dense length, plus one 4 KiB page for each page index not yet present.

A mutation that would fail validation (dead or unplaced entity) is simulated as a no-op. Apply stops at that index anyway, so the only effect is over-reservation on a changeset that fails regardless.

`apply` becomes `try_reserve`, then the existing mutation loop, which no longer allocates in the pool. The loop's push sites switch to debug-asserted, non-growing pushes.

### API Surface

| Change | Kind |
|---|---|
| `ApplyError::PoolExhausted(PoolExhausted)`: returned before any mutation, world unchanged | additive (`ApplyError` is `#[non_exhaustive]`) |
| `TransactError::PoolExhausted(PoolExhausted)`: `try_commit` calls `forward.try_reserve` before returning `Ok`, and `Durable::try_commit` does so before the WAL append | **breaking** (`TransactError` is exhaustive) |
| `WalError::Apply(ApplyError::PoolExhausted)` from replay, surfacing as `RecoverError::Wal` | no signature change |
| `recover_world_with_builder` docs: drop the "leave headroom for the WAL tail" caveat | docs |

With reservation done in `try_commit`, the post-commit `apply(...).expect(...)` cannot fail on the pool. The reservation holds because the world is exclusively borrowed between `try_commit` and `apply`.

### Invariant Matrix

| Invariant | Holds by | Tested by (new) |
|---|---|---|
| **A1** `apply` returning `ApplyError::PoolExhausted` left the world unchanged (no entity, row, location, sparse entry or tick change) | reservation precedes the tick and every mutation | `apply_pool_exhaustion_leaves_world_unchanged` (spawn / migrate / sparse variants, every size class exhausted) |
| **A2** after `try_reserve` succeeds, the mutation loop never allocates in the pool | simulation covers every growth site in the table above | `reserved_changeset_applies_with_exhausted_pool`, `reserved_replay_applies_with_exhausted_pool` (reserve, exhaust every size class, then apply; each reservation step is mutation-checked) |
| **A3** a transaction never writes a WAL record it cannot apply | strategies' `try_commit` reserve, and `Durable` reserves again right before `append` (covers custom strategies) | `durable_commit_pool_exhausted_writes_no_wal_record`, `optimistic_commit_reports_pool_exhaustion` |
| **A4** overwrites and despawn never allocate (B3 for all component types) | resident scratch | `despawn_droppable_with_exhausted_pool`, `overwrite_droppable_with_exhausted_pool` |
| **A5** a panicking `Drop` during overwrite or despawn leaves a consistent column (#180) | slot reuse keeps the structural commit before the drop | existing #180 tests, re-run against resident slots |
| **A6** reservation never needs more pool than sequential apply would have grown | peak-length reservation; `PoolVec` reserves exactly when that exceeds doubling | `reservation_matches_sequential_growth` (capacity after a reserved apply ≤ one-by-one spawning) |

### Convention Sheet

- Fallible = preflight capacity, then infallible commit, as in `try_spawn` and `import_page`. Reservations only grow spare capacity.
- Infallible paths that can still exhaust (plain `spawn` / `insert`) keep aborting through `handle_alloc_error`, unchanged.
- No new `unsafe` beyond the resident-slot pointer; every block keeps its pool and layout pairing.

## Alternatives Considered

- **Per-mutation preflight** (each mutation reserves its own needs; failure mid-changeset becomes `PartialApply { source: PoolExhausted }`). Cheaper, but it fixes only case 1. Transactions would still hit the post-commit `expect`, and under `Durable` the WAL would run ahead of the world (A3 fails). Rejected.
- **Reserve a scratch block per overwrite in the reservation** instead of resident slots. Peak memory is one block per droppable overwrite in the changeset, which is unbounded. Rejected.
- **Full atomic apply** (also pre-validate dead and unplaced entities, so every `ApplyError` leaves the world unchanged). The same simulation could carry this, and it would close the remaining half of #186 R2. But it has to mirror `adopt` and placement semantics exactly. Deferred to phase 2, below.

## Phase 2 (out of scope here)

- Pre-validate entity liveness and placement in the same simulation, making `apply` fully atomic and retiring `PartialApply` (#186 R2).
- Reuse the simulation for an exact `EnumChangeSet::pool_demand()` estimate, for admission control.

## Semantic Review

1. **Wrong world?** `try_reserve` takes the world `apply` will run on. Transactions already check `WorldMismatch` before commit.
2. **Stale reservation?** Nothing can run between `try_commit` and `apply`: both hold `&mut World`, and the `Durable` WAL lock is not held across the apply. A changeset applied without reservation reserves inside `apply` itself.
3. **Fast-lane batches?** Overwrite-only, with no row growth, so they are covered by resident scratch (A4).
4. **Replay adopt?** The simulation uses the same `max index + 1` growth that `adopt` performs (generations resize, free-list extend, B3).
5. **Cost on the commit path?** One pass over the mutations. Hash lookups happen only for spawn, despawn and migration entities, never for overwrites. Gate: `simple_insert/changeset`, `reducer/*`, `add_remove` benchmarks within noise.

## Risks

- **Simulation drift:** a new growth site added to `apply` without a matching simulation entry silently breaks A2. Mitigation: the reserve-exhaust-apply tests (A2) abort on any unreserved growth, and `fuzz_changeset_budget` drives random changesets against a small budget with a shadow model (run with `-max_len=8192` so batches reach the limit).
- **Breaking `TransactError`:** downstream exhaustive matches need a new arm. Noted in the CHANGELOG.

## Implementation Plan

1. ~~Correct the false "despawn never allocates" claims~~ — superseded: step 2 makes the claim true, and the docs record the gap instead.
2. Resident scratch slot in `BlobVec` (overwrite and extract paths); A4 and A5 tests. **Done (first PR).**
3. `EnumChangeSet::try_reserve` with simulation; A1, A2 and A6 tests. **Done (second PR).**
4. `apply` / `apply_replay` call `try_reserve` first; `ApplyError::PoolExhausted`. **Done.**
5. `TransactError::PoolExhausted` (and `ReducerError::PoolExhausted`); reserve in `Optimistic` / `Pessimistic` `try_commit`, which `Durable` calls before its WAL append; A3 test. **Done.**
6. Docs: `memory-management.md`, the `recover_world_with_builder` caveat, CHANGELOG; close #294. **Done.**
7. Gates: fmt and clippy (1.97, 1.99), workspace tests, loom, TSan, Miri on `blob_vec` and `changeset`, the new fuzz target plus `fuzz_world_ops` 60 s, benchmarks against main.

Steps 1–2 can ship as their own PR (they fix a real despawn panic independently); steps 3–6 as a second.
