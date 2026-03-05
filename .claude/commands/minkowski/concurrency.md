---
description: Help with Minkowski ECS concurrency model — Sequential, Optimistic, Pessimistic, Durable strategies
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user choose and configure the right concurrency strategy for their Minkowski ECS project.

## Step 1: Assess

Search the user's codebase for existing concurrency patterns:

- `Sequential` / `Optimistic` / `Pessimistic` imports and construction
- `Durable` wrapper usage (persistence + concurrency)
- `rayon` usage (`par_for_each`, `par_iter`, thread pool)
- `thread::spawn` or other threading primitives
- `strategy.transact()` / `strategy.begin()` / `try_commit()` calls
- `ReducerRegistry` dispatch via `call()` (transactional) vs `run()` (scheduled)
- Number of systems/reducers and their access patterns (read/write overlap)

## Step 2: Recommend

**Strong defaults:**
- **Single-threaded, no persistence needed**: `Sequential` — zero overhead, all ops delegate directly to World. Commit always succeeds. Use `registry.run()` for query reducers.
- **First step into concurrency**: `Optimistic` — live reads via `query_raw(&self)`, buffered writes into `EnumChangeSet`, tick-based validation at commit. Cheap when conflicts are rare. Default 3 retries.
- **Write-heavy with expensive retries**: `Pessimistic` — cooperative per-column locks acquired at begin, buffered writes, commit always succeeds. Higher lock overhead but no wasted work. Default 64 retries with spin+yield backoff.
- **Need crash safety**: `Durable<S, W>` wrapping any strategy. On successful commit, forward changeset is written to WAL before being applied to World. WAL write failure panics (durability invariant is non-negotiable).

**Strategy selection guide:**
1. Start with `Sequential`. It is the right choice until proven otherwise.
2. If you have multiple systems that touch overlapping components and want to run them concurrently, switch to `Optimistic`.
3. If optimistic retries waste too much work (write-heavy workloads, expensive computations), switch to `Pessimistic`.
4. If you need crash recovery, wrap whichever strategy you chose with `Durable`.

**Ask if unclear:**
- "What's your read/write ratio?" — High reads, low writes: `Optimistic`. High writes with overlap: `Pessimistic`.
- "How expensive is a retry vs a lock?" — If the computation inside a transaction is cheap, optimistic retries are fine. If expensive, pessimistic locks avoid wasted work.
- "Do you need crash recovery?" — `Durable` wraps any strategy. Requires `CodecRegistry` for component serialization.
- "Are you using rayon for parallel dispatch?" — `registry.call()` is safe to call from multiple rayon threads. The strategy handles retry/locking internally.

**Key design principle:** `Tx` does NOT hold `&mut World`. Methods take world as a parameter. This split-phase design enables: begin (sequential, `&mut World`) -> execute (parallel, `&World`) -> commit (sequential, `&mut World`). The `ReadOnlyWorldQuery` bound on `tx.query(&world)` prevents `&mut T` through a shared reference.

## Step 3: Implement

Help write concurrency code. Point to relevant examples:

- **Sequential (simplest)**: See `examples/examples/transaction.rs` — zero-overhead passthrough
- **Optimistic with reducers**: See `examples/examples/battle.rs` — rayon parallel dispatch with automatic retry
- **Pessimistic**: See `examples/examples/battle.rs` — same API, different strategy, lock guarantee
- **Durable**: See `examples/examples/persist.rs` — `Durable<Optimistic, Wal>` wrapping QueryWriter reducer
- **Conflict detection**: See `examples/examples/scheduler.rs` — `registry.query_reducer_access(id)` for Access metadata, `conflicts_with()` for NxN conflict matrix

**Pitfall alerts:**
- `&mut T` in transaction query: `tx.query(&world)` requires `ReadOnlyWorldQuery`. Use `tx.write()` for mutations.
- Cross-world corruption: strategies capture `WorldId` at construction. Using a strategy from world A with world B will panic.
- Entity ID lifecycle: IDs allocated during a transaction are tracked. On abort, they go to `OrphanQueue` and are recycled automatically. No manual cleanup needed.

For architecture details, see CLAUDE.md § "Transaction Semantics" and § "Key Conventions" (transaction safety invariants).
