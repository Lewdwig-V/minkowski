# Reducer Correctness

Reducers are user-supplied closures that mutate world state. The type system constrains *what* they can access (via typed handles and Access bitsets), but it cannot constrain *how* they compute. Four behavioral properties matter for correctness — all beyond Rust's type system today, so they are enforced by convention.

## Determinism

A deterministic reducer's output is a pure function of its handle state and args. Same world state + same args = same mutations, every time.

### What breaks determinism

- **Random number generators**: `fastrand`, `rand::thread_rng()`, any unseeded RNG.
- **System time**: `std::time::SystemTime::now()`, `Instant::now()`.
- **HashMap iteration order**: non-deterministic across runs (randomized hashing). Use `BTreeMap`/`BTreeSet` instead.
- **File and network I/O**: results depend on external state.
- **Global mutable state**: `static mut`, thread-locals, `lazy_static` with interior mutability.
- **Pointer-derived values**: `std::ptr::addr_of!`, `as *const _ as usize`. Addresses vary across runs due to ASLR.
- **Thread identity**: `std::thread::current().id()`.

### What's already safe

- All handle operations: `EntityMut::get`, `EntityMut::set`, `QueryWriter::for_each`, `Spawner::reserve`, `DynamicCtx::read`, `DynamicCtx::write`, etc.
- `BTreeMap` / `BTreeSet` iteration (deterministic ordering).
- Args passed into the reducer (caller controls determinism of inputs).
- Entity IDs returned by `Spawner::reserve` (deterministic given the same allocator state).

### Iteration order

Query iteration currently visits archetypes in creation order and rows within each archetype in insertion order. This is deterministic given identical world state. However, this is **not a stability guarantee** — do not rely on specific ordering across engine versions.

### Replication implication

Non-deterministic reducers break WAL replay on replicas. If using `Durable<S>`, the WAL records the changeset produced by each reducer invocation. A replica replaying the WAL gets the same mutations regardless of reducer determinism. But if a reducer is used on both the source and replica (e.g., for validation or live queries), non-determinism causes divergence. Keep reducers deterministic if your architecture involves replicated execution.

## No Unwinding

### Transactional reducers (EntityMut, Spawner, QueryWriter, DynamicCtx)

Panic-safe by design. If the closure panics:

- `Tx::Drop` reclaims entity IDs via `OrphanQueue` — no entity ID leak.
- The buffered changeset is discarded — no partial writes reach the world.
- The panic propagates to the caller of `registry.call()`.

### Scheduled reducers (QueryMut, QueryRef)

Run with direct `&mut World` access. If the closure panics:

- World remains consistent (no partial column writes — iteration yields references, not owned data).
- The panic propagates to the caller of `registry.run()`.
- No automatic rollback — any mutations applied before the panic persist.

**Guidance**: prefer transactional reducers when panic risk is non-trivial. The buffered write model means a panic discards all changes atomically.

## Termination

Reducers should complete in bounded time.

- Under `Pessimistic`, the reducer holds column locks for its entire execution. A non-terminating reducer deadlocks the system — no other reducer can acquire conflicting locks.
- Under `Optimistic`, a long-running reducer increases the conflict window. Other transactions that commit while this one runs will cause a conflict on validation, leading to retries and wasted work.
- Scheduled reducers (`QueryMut`, `QueryRef`) run with `&mut World`, blocking all other world access.

**Guidance**: avoid unbounded loops inside reducers. Use the handle's `for_each` for iteration — it's bounded by entity count. If you need to process items until convergence, cap the iteration count and check convergence outside the reducer.

## No Host APIs

This overlaps with determinism — I/O, network, and system clock break both properties. The additional concern is performance: blocking I/O inside a reducer holds locks or transactions open, starving other reducers and the scheduler.

**Guidance**: perform I/O outside reducers. Read files before dispatch, pass data in via args. Write results to a buffer, flush after commit. If a reducer needs external data, the caller should fetch it and pass it as args.
