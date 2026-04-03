# Minkowski Scaling Roadmap: Seven Stages

*Inspired by TigerBeetle's "Seven Stages of Database Scaling."*
*Status: Minkowski is at Stage 2. This document is the roadmap to Stage 7.*

---

## Table of Contents

1. [Stage 1: The Log](#stage-1-the-log) ✅
2. [Stage 2: Snapshots](#stage-2-snapshots) ✅ (current)
3. [Stage 2.5: SlabPool as the Only Allocator](#stage-25-slabpool-as-the-only-allocator)
4. [Stage 3: LSM Tree Storage](#stage-3-lsm-tree-storage)
5. [Stage 4: Replicated State Machine](#stage-4-replicated-state-machine)
6. [Stage 5: Horizontal Scaling (Sharding)](#stage-5-horizontal-scaling-sharding)
7. [Stage 6: Separate Storage and Compute](#stage-6-separate-storage-and-compute)
8. [Stage 7: Diagonal Scaling](#stage-7-diagonal-scaling)

---

## Stage 1: The Log

**Status: Complete.** (`minkowski-persist` crate, WAL subsystem)

The log is the foundation of durability. Every committed transaction is written
to the WAL before being applied to the in-memory World. On crash, replay the
log to recover state. The log is also the unit of replication (Stage 4).

### What exists

- `Wal`: segmented write-ahead log with CRC32 frame integrity, rollover, crash recovery
- `Durable<S>`: wraps any `Transact` strategy to add WAL logging on commit
- `WalCursor`: filesystem-specific log reader for pull-based replication
- `ReplicationBatch`: transport-agnostic wire format + `apply_batch`
- `CodecRegistry`: component name → serializer mapping, stable names for schema evolution
- `CrcProof`: zero-sized proof token — gates the raw memcpy fast path (only
  CRC-validated frames bypass rkyv bytecheck)

### The in-memory foundation

The log sits atop a column-oriented archetype ECS:

- `World`: entity allocator, archetype storage, sparse storage, query cache
- `BlobVec`: type-erased column storage with 64-byte alignment for SIMD
- `SlabPool`: mmap-backed slab allocator with thread-local cache
- `QueryPlanner`: cost-based plans, joins, aggregates, index-driven lookups
- `ReducerRegistry`: typed mutation handles with compile-time conflict detection
- `EnumChangeSet`: ordered mutation log with arena allocation
- `Transact` trait: Optimistic (tick-based), Pessimistic (lock-based). `Sequential`: zero-cost non-transactional path with its own API

### Key properties carried forward

These properties were established at Stage 1 and are load-bearing for every
subsequent stage:

| Property | Why it matters later |
|---|---|
| **Deterministic reducers** | RSM (Stage 4) requires identical replay across replicas |
| **Ordered mutation log** (`EnumChangeSet`) | Unit of WAL write (Stage 1), replication (Stage 4), and compaction (Stage 3) |
| **Generational entity IDs** | Stale references detected without coordination (Stage 5 cross-shard) |
| **External composition** (indexes, registry, planner outside World) | Clean separation for compute/storage split (Stage 6) |
| **Access bitsets** | Conflict detection generalizes to cross-shard conflict detection (Stage 5) |

---

## Stage 2: Snapshots

**Status: Complete.** (`minkowski-persist` crate, snapshot + checkpoint subsystem)

The log alone means crash recovery replays *every* mutation since the beginning
of time. Snapshots provide a baseline: checkpoint the world state to disk, then
recovery only needs to replay the WAL tail from the last checkpoint.

### What exists

- `Snapshot`: full-world serialization via rkyv (v2 format with CRC32 envelope)
- `CheckpointHandler` / `AutoCheckpoint`: snapshot trigger when WAL exceeds threshold
- `Durable<S>` integration: checkpoint handler fires automatically when accumulated WAL bytes exceed the configured threshold

### What makes this a good Stage 2

The WAL and snapshots are properly integrated, not bolted on separately:

1. `Durable<S>` writes the WAL on every commit (Stage 1)
2. `AutoCheckpoint` triggers a snapshot when WAL size exceeds a threshold
3. On recovery: load snapshot → replay WAL from checkpoint sequence number
4. `WalCursor` can read from the checkpoint point for replication
5. `CrcProof` validates both snapshot and WAL frame integrity

### What's missing (prerequisites for Stage 3)

- **WAL compaction**: old segments are deleted after checkpoint but never merged.
  Stage 3 needs a log-structured compaction pipeline.
- **Incremental snapshots**: `Snapshot::save` serializes the entire world every
  time. Large worlds make this prohibitively expensive. Stage 3 replaces this
  with incremental (delta) snapshots via LSM compaction.
- **Page-level change tracking**: `Changed<T>` is column-granular within each
  archetype (a single row mutation marks that column's `changed_tick`).
  Incremental snapshots need finer-grained page-level dirty tracking.

---

## Stage 2.5: SlabPool as the Only Allocator

**Status: Not started. Prerequisite for Stage 3.**

**Goal**: Eliminate `SystemAllocator` — all BlobVec columns must be mmap-backed
via `SlabPool`. This is a hard prerequisite for every subsequent stage.

### Why this is necessary

Every stage from 3 onwards assumes mmap-backed, page-aligned memory:

| Stage | Assumption |
|---|---|
| 3 (LSM) | Dirty page tracking requires knowing page boundaries — `malloc`'d memory has no page structure |
| 4 (RSM) | io_uring + O_DIRECT requires page-aligned buffers with `RLIMIT_MEMLOCK` |
| 5 (Sharding) | Entity migration between shards copies page-aligned column slices |
| 6 (Storage/Compute) | Demand-paged working set maps directly onto mmap pages |
| 7 (Diagonal) | Object storage sorted runs are page-aligned for zero-copy load |

Today, `WorldBuilder` defaults to `SystemAllocator` unless `memory_budget()` is
called. This means most worlds in tests, examples, and user code use `malloc`'d
memory. The `SlabPool` path is opt-in, not default.

### What changes

1. **Remove `SystemAllocator`** as a backing option for `BlobVec` columns.
   All column storage goes through `SlabPool`.

2. **`World::new()` allocates a default `SlabPool`** with a sensible default
   budget (e.g., 256 MB). `WorldBuilder::memory_budget()` becomes a way to
   tune the budget, not to opt into a different allocator.

3. **Tests and examples** that currently use `World::new()` get `SlabPool`
   automatically. Tests that need small worlds can use a small budget.

4. **Remove the allocator abstraction** — one code path, one allocation model,
   one set of invariants. This is a simplification, not a complication.

### Migration risk

- **Breaking change**: any user code that constructs `World::new()` will now
  require `mmap` to succeed. Environments that restrict `mmap` (some containers,
  WASM) need a plan. Options: (a) WASM gets a `malloc`-backed `SlabPool` shim,
  (b) WASM is explicitly unsupported for persistent workloads.
- **Default budget sizing**: too small and users hit `PoolExhausted` unexpectedly;
  too large and the process reserves address space it never uses (virtual memory
  is cheap, but `RLIMIT_MEMLOCK` is not).
- **Miri compatibility**: Miri doesn't support `mmap` syscalls. The Miri test
  suite (860 tests) will need a mock allocator or `cfg(miri)` fallback.

---

## Stage 3: LSM Tree Storage

**Status: Not started. Prerequisites: Stage 2.5 (SlabPool-only allocation).**

**Goal**: Incremental persistence. Only write what changed, not the entire world.

### The problem

Full-world snapshots are O(world size) regardless of how much changed. A world
with 10M entities and 1KB average row size is ~10 GB. Snapshotting 10 GB to disk
every checkpoint is unsustainable. We need persistence cost proportional to the
*mutation rate*, not the *world size*.

This is also the **128 TiB ceiling** that plagues cloud databases. Many claim
"unlimited storage" but monolithic snapshot save/restore at that scale takes
5–6 hours. The ceiling isn't storage capacity — it's recovery time. If your
recovery procedure is O(total data), your architecture has a hard upper bound
no matter how much disk you provision. LSM breaks this ceiling by making
recovery O(delta since last checkpoint): load the baseline from sorted runs,
replay the WAL tail. Recovery time scales with *mutation rate*, not *world size*.

The ambition here — following TigerBeetle's lead — is a system with no
theoretical upper bound on storage. Petabytes of state, if you can afford the
RAM and NVMe to hold the working set, with object storage for the rest.

### Architecture: LSM over archetypes

The LSM tree operates on **archetype pages**, not individual rows. This preserves
column-oriented access patterns and SIMD alignment.

```
 ┌─────────────────────────────────────────────────┐
 │  L0: MemTable (current World archetypes)        │
 │  - Hot data, fully mutable, in-memory only      │
 │  - This IS the existing World storage            │
 └────────────────────┬────────────────────────────┘
                      │ flush (dirty pages only)
 ┌────────────────────▼────────────────────────────┐
 │  L1: Immutable sorted runs (memory-mapped)      │
 │  - Recent flushes, still in main memory         │
 │  - Sorted by (archetype_id, page_id)            │
 │  - Read via mmap, zero-copy access              │
 └────────────────────┬────────────────────────────┘
                      │ compaction (merge sorted runs)
 ┌────────────────────▼────────────────────────────┐
 │  L2: Compacted sorted runs (NVMe)               │
 │  - Merged from L1, larger files, fewer seeks    │
 │  - Page-aligned for O_DIRECT / io_uring         │
 └────────────────────┬────────────────────────────┘
                      │ major compaction
 ┌────────────────────▼────────────────────────────┐
 │  L3: Archive (NVMe / object storage)            │
 │  - Full compacted baseline                      │
 │  - Input for snapshot restore                   │
 └─────────────────────────────────────────────────┘
```

### Key design decisions

**Unit of flush: the dirty page, not the row.**
A "page" in the LSM is a contiguous range of rows within one archetype column.
Dirty tracking is a bitset per page. Flush writes only dirty pages to L1.

Stage 2.5 ensures all columns are mmap-backed via `SlabPool`, providing the
page-aligned memory required for dirty-page tracking and O_DIRECT I/O.

**WAL remains the source of truth for crash recovery.**
The LSM is not a replacement for the WAL. The WAL captures the logical mutations
(insert, update, remove, spawn, despawn). The LSM captures the physical state
(page images). On crash, replay the WAL from the last L1 flush point. On clean
shutdown, flush all dirty pages to L1 and discard the WAL.

**Compaction is archetype-aware.**
L1→L2 compaction merges sorted runs for the same archetype. Cross-archetype
compaction is never needed because archetypes are independent column groups.
This means compaction parallelizes trivially across archetypes.

**Reads never touch L1/L2/L3 in normal operation.**
L0 (the live World) always has the latest data. LSM levels are only read during
recovery (snapshot restore) and cold-start (load world from L2/L3). This is a
critical difference from a traditional LSM database where reads must merge
across levels. In Minkowski, the in-memory World IS the merged view.

### New components

| Component | Purpose |
|---|---|
| `DirtyPageTracker` | Per-column bitset tracking which pages were mutated since last flush (**done** — `storage::dirty_pages`) |
| `BlockedBloomFilter` | Cache-line-blocked bloom filter for recovery-time level lookups (see design below) |
| `SortedRun` | Immutable file containing sorted archetype page images, with a sparse index for page lookup |
| `FlushWriter` | Writes dirty pages from L0 → L1 as a new sorted run |
| `Compactor` | Merges L1 sorted runs → L2, L2 → L3. Archetype-parallel. |
| `LsmManifest` | Tracks which sorted runs exist at each level, their archetype coverage, and sequence ranges |
| `LsmRecovery` | Restores World from L2/L3 baseline + L1 delta + WAL tail |

### Dependencies on existing infrastructure

- `SlabPool` (mandatory after Stage 2.5) → mmap-backed, page-aligned
- `DirtyPageTracker` (**done**) → per-column page-level dirty bitset (256 rows/page), wired into all BlobVec mutation paths
- `CrcProof` → gates the zero-copy read path from mmap'd sorted runs
- `CodecRegistry` → sorted runs use the same serialization as WAL/snapshots

### Open questions

1. **Page size**: 4 KB (NVMe-aligned) or 64 KB (fewer metadata entries)?
   TigerBeetle uses 64 KB. Larger pages reduce compaction metadata but waste
   space for sparse archetypes.

2. **Compaction scheduling**: background thread vs. cooperative (compact during
   idle frames)? Game loops want deterministic frame times; background compaction
   can spike I/O. Cooperative compaction bounds I/O per frame but extends total
   compaction time.

3. **Archetype migration during compaction**: if an entity moves between
   archetypes (insert/remove component), the old archetype page is dirty and the
   new one is dirty. Compaction must handle tombstones. Does the sorted run
   store tombstones explicitly, or is "page not present" sufficient?

4. **Bloom filters**: ~~traditional LSMs use bloom filters to avoid reading levels
   that don't contain a key. Our "reads never touch L1+" property may make this
   unnecessary for normal operation, but recovery-time reads would benefit.~~
   **Resolved**: `BlockedBloomFilter` promoted to a Stage 3 component. Recovery
   reads (cold-start, replica catch-up) must probe L1/L2/L3 sorted runs to find
   which level holds a given page — without a filter this is one I/O per level.
   See design section below.

### BlockedBloomFilter design

A **Blocked Bloom Filter** (also called a Cache-line Bloom Filter) partitions the
bit array into 64-byte blocks aligned to cache lines. Each key hashes to exactly
one block, then sets/checks *k* bits within that block. This guarantees **one
cache miss per probe** regardless of *k*, vs. up to *k* misses in a standard
Bloom filter.

**Block layout**: `#[repr(C, align(64))] struct Block { words: [u64; 8] }` — one
512-bit block per cache line. 8 hash functions, one per u64 word (bit position
0..63 within each word). SIMD-friendly: the compiler auto-vectorizes the 8-word
AND+compare via AVX2/NEON with `target-cpu=native`.

**Hash scheme** (enhanced double-hashing with per-word entropy):
- `h1 = splitmix64(key ^ seed)` → block index (`h1 % num_blocks`)
- `h2 = splitmix64(h1)` → bit positions for words 0–7
- `h3 = splitmix64(h2)` → independent perturbation for per-word mixing
- Bit position for word `i`: `bit_i = ((h2 >> (i*6)) ^ (h3 >> ((7-i)*6))) & 0x3F`

Each word's bit position is derived from two independent 6-bit slices — one
from `h2` at offset `i*6` and one from `h3` at the mirror offset `(7-i)*6`.
XOR-combining them doubles the effective entropy per position (12 bits reduced
to 6) and ensures every word can reach all 64 bit positions uniformly. This
avoids the pitfall of standard enhanced double hashing (`g_i = h1 + i·h2 +
i·(i+1)/2 mod m`) which, when applied mod 64 per word, collapses all entropy
to `h2 % 64` and restricts reachable positions for words with even multipliers.

**Sizing**: ~10 bits per expected key → ~1% false-positive rate. For *N* keys:
`num_blocks = ceil(10 * N / 512)`. With 8 hashes in a 512-bit block and ~51
keys per block on average, theoretical FPR ≈ 0.84%.

**Allocation**: statically allocated from `SlabPool` at filter construction.
No runtime growth — the filter is sized once for the expected key count. Fits
the pool's fixed-budget model and avoids allocation failures during hot paths.

**SIMD optimization**: the `contains` hot path is a tight loop over 8 u64
words — `block.words[i] & mask[i] == mask[i]` for all *i*. With
`#[repr(align(64))]` and `target-cpu=native`, LLVM emits two 256-bit
`vpand`+`vpcmpeq` pairs on AVX2 or equivalent NEON instructions on aarch64.
Short-circuit on first mismatch in the scalar fallback.

**Integration points**:
- One filter per sorted run (attached to `SortedRun`)
- Filter keys are `(ArchetypeId, page_index)` pairs packed as u64
- `LsmRecovery` probes filters top-down (L1 → L2 → L3) to skip levels
- Filters are serialized inline in the sorted run footer (small: ~1.25 bytes/key)
- `Compactor` rebuilds filters when merging runs (bulk insert, no incremental)

---

## Stage 4: Replicated State Machine (RSM)

**Status: Not started. Prerequisites: Stage 3 (efficient state transfer for new replicas).**

**Goal**: Fault tolerance. The world survives machine failure.

### The insight

Minkowski already has the two hardest prerequisites for RSM:

1. **Deterministic execution**: the reducer determinism rule (no RNG, no
   HashMap iteration, no system time, no I/O) means two replicas processing
   the same mutation log produce identical state.

2. **Ordered mutation log**: `EnumChangeSet` is an ordered mutation log.
   `Durable<S>` already writes this log to a WAL. RSM extends this from
   "write to local disk" to "replicate to quorum of machines."

### Architecture

```
 Client
   │
   ▼
 ┌─────────┐    log replication    ┌─────────┐    ┌─────────┐
 │ Leader  │ ──────────────────▶   │Follower │    │Follower │
 │         │                       │  (hot   │    │  (hot   │
 │ World   │                       │ standby)│    │ standby)│
 │ WAL     │                       │ World   │    │ World   │
 │ LSM     │                       │ WAL     │    │ WAL     │
 └─────────┘                       └─────────┘    └─────────┘
```

All mutations go through the leader. The leader appends to its local WAL and
replicates the log entry to followers. Once a quorum acknowledges, the mutation
is committed. Followers apply mutations in log order — deterministic execution
guarantees convergence.

### Consensus protocol

**Viewstamped Replication (VR)** over Raft. Rationale:

- VR is conceptually simpler (no log compaction, no snapshots-as-part-of-consensus)
- TigerBeetle chose VR for the same reasons
- The Minkowski WAL already has sequence numbers — these become VR operation numbers
- Leader election uses view numbers, not term+index pairs

### Key design decisions

**Log entry = `ReplicationBatch`.**
The existing `ReplicationBatch` is already the transport-agnostic wire format.
RSM replication sends the same bytes that `WalCursor` reads. No new
serialization format needed.

**State transfer uses LSM sorted runs, not full snapshots.**
When a new replica joins or a follower falls too far behind, it needs the full
world state. Instead of a monolithic snapshot (Stage 2), send the LSM L2/L3
sorted runs + L1 delta + WAL tail. This is incremental — if the follower has
a recent baseline, only the delta needs to transfer.

**Follower reads with bounded staleness.**
The internal `query_raw(&self)` method (used by the transaction system) requires
no mutation — execution is side-effect-free. This means followers *can* serve
reads, but without a read lease or leader round-trip to confirm the follower's
commit index, they provide **bounded-staleness reads, not linearizable reads**.
Linearizable reads require either routing to the leader or a read lease protocol.
The staleness bound is configurable per query.

**Unified I/O via io_uring (TigerBeetle pattern).**
Following TigerBeetle's architecture, storage and network I/O should be
unified into a single io_uring instance. A single `io_uring_enter` syscall
can: write WAL frames to disk AND send replication packets to followers AND
poll for client requests. This is not "use async I/O" — it's a fundamentally
different architecture where storage writes and network sends are siblings in
the same completion ring.

Key properties:
- **O_DIRECT + pre-allocated buffers**: bypass the kernel page cache, copy
  data directly from network buffer → WAL file → state machine. Stage 2.5
  ensures all columns are `SlabPool`/mmap-backed, but **pinning pages requires
  `mlock`**, which is currently best-effort (`try_mlockall` falls back to
  manual page-touch on failure). For io_uring O_DIRECT, pinned memory is
  mandatory — Stage 4 must make `mlock` a hard requirement, not opt-in.
  `RLIMIT_MEMLOCK` is only the kernel quota for how much can be locked; it
  does not pin pages on its own.
- **Zero-copy WAL write** (aspirational): today's `ReplicationBatch` is a
  heap-backed `Vec<WalRecord>` that must be serialized via `to_bytes()` and
  deserialized via `from_bytes()` — `apply_batch` accepts a parsed
  `&ReplicationBatch`, not raw bytes. True zero-copy (network ring buffer →
  WAL file → state machine with no decode step) requires a **new flat wire
  format** where the on-wire representation IS the on-disk representation.
  This is a Stage 4 deliverable, not an existing capability.
- **Fixed-size, cache-aligned structures**: Minkowski already has this —
  `BlobVec` is fixed-layout per component `Layout`, entity IDs are fixed
  8 bytes, `SlabPool` pre-allocates all memory.

**View changes do NOT drain the I/O ring.**
This is a critical design constraint learned from TigerBeetle. When a view
change (leader election) begins:

1. The replica transitions state (e.g., `status=normal` → `status=view_change`)
2. It **stops submitting new storage requests** for client transactions
3. It **continues pumping the io_uring event loop** — processing completions
   and submitting consensus protocol messages (StartViewChange, DoViewChange)
4. Pending storage writes from the previous view remain in-flight within the
   kernel/hardware

When late completions arrive after the view change:
- Check the completion's view ID against the current view
- If the write is still valid in the new view, mark it as durable
- If the view change truncated the log, safely discard the stale completion

**Critical caveat: fencing stale DMA writes.**
"Discarding the stale completion" only discards the *acknowledgement* — the
bytes may have already hit the disk via DMA before the CQE arrived. If the
new leader reuses the same WAL offset for a new log entry, the old-view write
races with the new-view write at the storage level. TigerBeetle solves this
with **generation-isolated WAL slots**: each slot's header contains the view
number, and the new leader explicitly overwrites stale slots before reusing
them. Minkowski's WAL must adopt the same pattern — each WAL frame header
needs a view field, and recovery must validate the view of every frame against
the known view history to detect torn writes from view transitions.

The analogy to generational entity IDs still holds: a stale WAL frame is like
a stale entity handle — the generation (view number in the frame header)
catches it. But unlike entity handles (which are checked at read time), WAL
frames are checked at *recovery* time, so the cost of a missed check is
silent data corruption, not a runtime error.

**Why this matters**: if view changes blocked on draining the ring, failover
latency would be bounded by the slowest in-flight I/O. A "gray failure" — a
disk that hasn't crashed but responds in 30 seconds — would hang the election
for 30 seconds. By decoupling view changes from storage completions, failover
is bounded by *network* latency (microseconds), not *storage* tail latency.

### New components

| Component | Purpose |
|---|---|
| `IoRing` | Unified io_uring event loop for storage + network I/O (with kqueue/epoll fallback) |
| `ReplicaGroup` | Manages the set of replicas, their states, and the consensus protocol |
| `LogReplicator` | Sends `ReplicationBatch` entries from leader to followers via `IoRing` |
| `ConsensusState` | VR protocol state: view number, op number, commit number, log |
| `StateTransfer` | Sends LSM sorted runs to new/lagging replicas |
| `LeaderElection` | VR view change protocol (non-blocking — does not drain I/O ring) |

### Dependencies on existing infrastructure

- `ReplicationBatch` → already the wire format
- `WalCursor` → followers use cursor to track replication position
- `apply_batch` → followers apply mutations identically to leader
- `CodecRegistry` → all replicas must agree on component schemas (version negotiation)
- Reducer determinism → guarantees replicas converge

### Open questions

1. **io_uring bootstrapping**: io_uring requires Linux 5.6+. What's the
   fallback for macOS/Windows development? Options: epoll fallback (TigerBeetle
   uses kqueue on macOS), or require Linux for production with a simulated I/O
   layer for dev/test.

2. **Client interaction model**: do clients send mutation requests to the leader
   (command pattern) or send `ReplicationBatch` directly? The command pattern is
   simpler but adds a serialization hop.

3. **Read staleness**: how stale can follower reads be? Options: linearizable
   (forward reads to leader), bounded staleness (follower serves if within N
   ops), eventual (any follower, any time).

4. **Multi-world**: can one replica group host multiple Worlds? Useful for
   game servers with multiple rooms/instances. Each World is an independent
   state machine sharing the same consensus group.

---

## Stage 5: Horizontal Scaling (Sharding)

**Status: Not started. Prerequisites: Stage 4 (each shard is an RSM).**

**Goal**: Scale beyond one machine's memory and CPU.

### The problem

A single RSM group handles ~1 machine's worth of data and compute. When the
world grows beyond that, we need to partition entities across machines. Each
partition (shard) is its own RSM group with its own leader and followers.

### The ECS advantage

Entities are naturally partitionable:

- Entity IDs are opaque 64-bit handles with no inherent ordering or locality
- Components are per-entity — no cross-entity referential integrity
- Archetypes are independent column groups — an archetype can live entirely on one shard

This is structurally simpler than sharding a relational database where foreign
keys create cross-shard dependencies.

### Sharding strategy

**Phase 1: Hash-based sharding by entity ID.**

```
shard_id = entity.index() % num_shards
```

Simple, uniform distribution, no hotspots. Each shard owns a disjoint subset of
entity indices. Archetype structure is replicated across shards (schema is
global, data is partitioned).

**Phase 2: Range-based sharding by archetype.**

Entities with the same archetype tend to be queried together. Group entities by
archetype and assign archetype ranges to shards. This optimizes query locality —
a query that scans one archetype hits one shard, not all shards.

**Phase 3: Adaptive sharding.**

Monitor query patterns and rebalance shards to minimize cross-shard queries.
Entity migration between shards uses the same archetype migration machinery
(spawn on target shard, despawn on source shard, update routing table).

### Cross-shard operations

This is where Amdahl's law bites.

| Operation | Cross-shard cost |
|---|---|
| `get(entity)` | Route to owning shard — O(1) lookup in routing table |
| `query::<(&Pos, &Vel)>()` | Fan out to all shards, merge results — O(shards) |
| `join()` | Collect entity sets from all shards, intersect — expensive |
| `spatial_query(within)` | Fan out to shards overlapping the spatial region |
| `transact` (single entity) | Route to owning shard — no cross-shard coordination |
| `transact` (multi-entity) | 2PC across owning shards — serialization point |

**The serial fraction**: cross-shard transactions require coordination (2PC or
deterministic scheduling). As cross-shard transaction rate increases, Amdahl's
law limits speedup from adding shards. The break-even point depends on the
workload's locality.

### Key design decisions

**Routing table is a first-class component.**
A `ShardRouter` maps entity index ranges to shard IDs. All shards hold a copy
(replicated via the consensus protocol). Entity migration updates the routing
table atomically.

**Queries are push-down, not pull-up.**
A scatter-gather query sends the query plan to each shard. Each shard executes
locally and returns results. The coordinator merges. This avoids shipping raw
data across the network — only results travel.

**Deterministic scheduling over 2PC.**
Instead of two-phase commit for cross-shard transactions, use deterministic
scheduling: a **global sequencer** (shared log) assigns a total order to all
cross-shard transactions. Each shard reads this shared log and independently
executes its portion in the agreed order. No coordination at execution time.
Calvin/BOHM-style.

This is critical: per-shard consensus logs do NOT give a total order across
shards. The global sequencer is the missing piece that makes deterministic
cross-shard ordering possible. It can be implemented as a dedicated RSM group
(a lightweight consensus group that only sequences transaction IDs, not data)
or as a shared log service (e.g., Corfu/DELOS-style).

This eliminates the serial coordination bottleneck of 2PC at the cost of
requiring all shards to process the full log (even entries they don't own,
to maintain order). TigerBeetle uses this approach.

### New components

| Component | Purpose |
|---|---|
| `ShardRouter` | Entity → shard mapping with migration support |
| `ShardCoordinator` | Scatter-gather query execution across shards |
| `CrossShardPlanner` | Extends `QueryPlanner` with shard-aware cost model |
| `EntityMigrator` | Move entities between shards (spawn+despawn+route update) |
| `GlobalSequencer` | Shared log assigning total order to cross-shard transactions |
| `DeterministicScheduler` | Per-shard executor consuming the global sequence |

### Open questions

1. **Shard-local indexes**: each shard maintains its own `BTreeIndex`/`HashIndex`
   for local data. Cross-shard index queries require fan-out. Should there be a
   global index tier?

2. **Archetype schema agreement**: all shards must agree on component types and
   archetype structure. How is schema evolution coordinated? (CodecRegistry
   already has stable names — extend with version negotiation.)

3. **Entity migration granularity**: migrate one entity at a time, or batch
   migrate entire archetypes? Archetype-level migration is more efficient but
   less flexible.

4. **Spatial sharding**: for spatially-organized worlds (games, simulations),
   shard by spatial region rather than entity ID. This optimizes spatial queries
   but creates hotspots when entities cluster.

---

## Stage 6: Separate Storage and Compute

**Status: Not started. Prerequisites: Stage 5 (sharding provides the need).**

**Goal**: Scale storage and compute independently.

### The problem

In Stages 1–5, each node stores its data locally. Storage capacity is limited
by local disk. Compute capacity is limited by local CPU. They scale together,
but workloads rarely need both to scale equally. A read-heavy analytical
workload needs more compute but not more storage. A large-world simulation
needs more storage but not more compute per entity.

### Architecture

```
 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │   Compute    │  │   Compute    │  │   Compute    │
 │   Node 0     │  │   Node 1     │  │   Node 2     │
 │              │  │              │  │              │
 │  World (L0)  │  │  World (L0)  │  │  World (L0)  │
 │  LSM L1      │  │  LSM L1      │  │  LSM L1      │
 │  (working    │  │  (working    │  │  (working    │
 │   set cache) │  │   set cache) │  │   set cache) │
 └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                 │                 │
        └────────┬────────┴────────┬────────┘
                 │                 │
        ┌────────▼─────────────────▼────────┐
        │        Object Storage             │
        │        (S3, MinIO, Tigris)        │
        │                                   │
        │  LSM L2/L3 sorted runs            │
        │  WAL segments (archived)          │
        │  No practical capacity limits     │
        └───────────────────────────────────┘
```

### The catch

Object storage has no practical limits on capacity. But you can't process
transactions from object storage — latency is milliseconds per GET, and there's
no random write. Transactions need microsecond-latency random access to the
working set.

This is the fundamental tension that Stage 7 resolves.

### What changes from Stage 5

**Compute nodes become stateless (almost).**
A compute node holds:
- L0: the live World (hot working set, in memory)
- L1: recent dirty pages (local NVMe, acts as a write buffer)

Everything else lives in object storage. If a compute node crashes, another
node can resume by loading the working set from object storage + replaying the
WAL tail.

**LSM L2/L3 move to object storage.**
Compacted sorted runs are uploaded to S3/MinIO. Compute nodes pull them on
demand (cold start, entity migration, recovery). Once uploaded, local copies
can be evicted.

**WAL segments are archived to object storage.**
After a WAL segment is fully replicated and its data is flushed to L1, the
segment is uploaded to object storage for long-term retention. Local WAL
retains only the tail needed for crash recovery.

### Key design decisions

**Object storage is append-only.**
Sorted runs and WAL segments are immutable once written. Updates create new
sorted runs; compaction merges old runs into new ones and deletes the old.
This maps naturally to object storage's PUT/GET/DELETE semantics.

**Compute node working set is demand-paged.**
On cold start, a compute node loads the LSM manifest from object storage,
then lazily fetches sorted run pages as queries touch them. Hot pages are
cached in L1 (local NVMe). This is essentially a cooperative page cache
with object storage as the backing store.

**Schema and routing metadata live in a coordination service.**
Component schemas, shard routing tables, and LSM manifests are small and
rarely change. Store them in etcd/FoundationDB/the consensus log. Compute
nodes subscribe to changes.

### New components

| Component | Purpose |
|---|---|
| `ObjectStore` | Abstraction over S3/MinIO/local-fs for sorted run storage |
| `RemoteLsmManifest` | LSM manifest stored in object storage + coordination service |
| `PageCache` | Demand-paged cache of sorted run pages from object storage |
| `WalArchiver` | Uploads sealed WAL segments to object storage |
| `ComputeNodeBootstrap` | Cold-start sequence: fetch manifest → load working set → replay WAL tail |

### Open questions

1. **Object storage latency hiding**: S3 GET latency is 50–200ms. Can we
   prefetch pages speculatively based on query patterns? The query planner
   knows which archetypes a query will touch — could it issue prefetch
   requests during plan compilation?

2. **Cost model**: object storage charges per GET/PUT. The compaction scheduler
   must be aware of cost — too-frequent compaction wastes PUT budget, too-rare
   compaction wastes GET budget (more levels to read through on recovery).

3. **Multi-tenancy**: with compute separated from storage, multiple tenants
   can share compute nodes with isolated storage namespaces. How does this
   interact with the shard routing table?

---

## Stage 7: Diagonal Scaling

**Status: Not started. Prerequisites: Stages 3–6.**

**Goal**: Scale everything simultaneously with no architectural ceiling.

### The synthesis

Each previous stage solved one bottleneck but created another:

| Stage | Solved | Created |
|---|---|---|
| 3 (LSM) | Snapshot cost proportional to mutations, not world size | Compaction I/O competes with transaction processing |
| 4 (RSM) | Fault tolerance, durable state | Single-machine throughput ceiling |
| 5 (Sharding) | Horizontal throughput scaling | Cross-shard coordination (Amdahl) |
| 6 (Storage/Compute) | Independent scaling, unlimited storage | Object storage latency for working set |

Stage 7 combines all four into a single architecture where:

- **Object storage** provides unlimited capacity (Stage 6)
- **LSM** provides fast local access to the working set (Stage 3)
- **RSM** provides consistency and fault tolerance (Stage 4)
- **Sharding** provides horizontal throughput (Stage 5)

### Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     Coordination Layer                          │
 │  Shard routing │ Schema registry │ LSM manifests │ Consensus   │
 └───────┬─────────────────┬─────────────────┬────────────────────┘
         │                 │                 │
 ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
 │  Shard 0 RSM  │ │  Shard 1 RSM  │ │  Shard N RSM  │
 │               │ │               │ │               │
 │  Leader       │ │  Leader       │ │  Leader       │
 │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │
 │  │World(L0)│  │ │  │World(L0)│  │ │  │World(L0)│  │
 │  │LSM L1   │  │ │  │LSM L1   │  │ │  │LSM L1   │  │
 │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │
 │       │       │ │       │       │ │       │       │
 │  Follower(s)  │ │  Follower(s)  │ │  Follower(s)  │
 └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  │                 │
         ┌────────▼─────────────────▼────────┐
         │        Object Storage             │
         │  L2/L3 sorted runs (per shard)    │
         │  Archived WAL segments            │
         │  LSM manifests (versioned)        │
         └───────────────────────────────────┘
```

### How it works

1. **A mutation arrives** at the shard's leader (routed via `ShardRouter`).

2. **Leader replicates the log entry** to its followers via VR consensus.
   Once a quorum acknowledges, the mutation is committed.

3. **Leader and followers apply the mutation** to their in-memory World (L0).
   Deterministic execution guarantees all replicas converge.

4. **Dirty pages accumulate in L0.** When the dirty page budget is exceeded,
   the flush writer writes dirty pages to L1 (local NVMe).

5. **Compaction merges L1 → L2** (local NVMe → local NVMe). When L2 reaches
   its size budget, **L2 → L3** compaction uploads sorted runs to object storage.

6. **Old sorted runs in object storage are garbage-collected** after major
   compaction produces a new baseline.

7. **Queries within a shard** execute entirely from L0 (in-memory) — no disk I/O.
   LSM levels are only read during recovery.

8. **Cross-shard queries** scatter to shard leaders, execute locally, and
   gather results at the coordinator. The query planner pushes predicates
   and projections to each shard.

9. **Shard rebalancing** migrates entity ranges between shards. The source
   shard writes migrating entities to its log; the target shard's leader
   applies them. The routing table updates atomically via the coordination
   layer.

10. **Node failure** triggers leader election within the RSM group. The new
    leader has all committed log entries (VR guarantee). It rebuilds L0 from
    its local L1 + WAL tail. If L1 is lost (disk failure), it pulls L2/L3
    from object storage.

### The diagonal

"Diagonal scaling" means scaling both vertically (bigger machines, more memory,
faster NVMe) and horizontally (more shards, more replicas) simultaneously,
with object storage as the unbounded backing store. There is no architectural
ceiling — each axis scales independently:

- **More memory** → larger L0 working set → fewer L1 flushes
- **Faster NVMe** → faster compaction → lower L1 read amplification
- **More shards** → higher transaction throughput (up to Amdahl's limit)
- **More replicas per shard** → higher read throughput, better fault tolerance
- **Object storage** → unlimited total capacity, pay per GB

### What Minkowski uniquely brings

An ECS engine at Stage 7 has properties that a general-purpose database does not:

1. **Column-oriented storage is native.** LSM sorted runs are already columnar
   (BlobVec pages). No row-to-column conversion needed.

2. **Entities are partition-friendly.** No foreign keys, no referential integrity
   across entities. Sharding is structurally clean.

3. **Query plans are push-down ready.** The existing `QueryPlanner` generates
   structured execution plans (`PlanNode` tree) that could be made serializable
   for remote shard dispatch. Plan serialization does not exist today — it is
   a Stage 5 requirement.

4. **Deterministic execution is already enforced.** The reducer determinism rule
   is a cultural and tooling convention, not a runtime check. RSM requires this
   to be an invariant, not a guideline — Stage 4 must add runtime verification
   (e.g., hash the mutation output and compare across replicas).

5. **External composition means clean boundaries.** `ReducerRegistry`,
   `QueryPlanner`, `SpatialIndex` are all external to World. This maps naturally
   to separate compute-layer services that compose over a shared storage layer.

---

## Sequencing and Dependencies

```
 Stage 1 ──▶ Stage 2 ──▶ Stage 2.5 ──▶ Stage 3 ──▶ Stage 4 ──▶ Stage 5 ──▶ Stage 6 ──▶ Stage 7
 The Log    Snapshots   SlabPool-     LSM         RSM       Sharding    Storage/     Diagonal
  (done)     (done)      only                                           Compute
                                       │                        │
                                       │            ┌───────────┘
                                       │            │
                                       ▼            ▼
                                 Stage 3 is    Stage 5 is
                                 prerequisite  prerequisite
                                 for Stage 4   for Stage 6
                                 (state        (sharding
                                 transfer)     creates the
                                               need to
                                               separate)
```

**Stage 2.5 (SlabPool-only) is the immediate next step.** It's a prerequisite
for Stage 3 — LSM dirty-page tracking requires mmap-backed, page-aligned
columns. This is a breaking change that removes `SystemAllocator`, so do it
first while the user base is small.

**Stage 3 (LSM) is the critical path.** It blocks Stage 4 (efficient state
transfer for new replicas) and is the foundation for Stage 6 (object storage
as LSM backing store).

**Stages 4 and 5 can partially overlap.** The consensus protocol (Stage 4) can
be developed and tested with a single shard. Sharding (Stage 5) adds the
routing and coordination layer on top. But the cross-shard deterministic
scheduling design should be done upfront — it constrains the consensus protocol.

**Stage 6 is an incremental change to Stage 5.** Once shards have LSM storage,
moving L2/L3 to object storage is a storage backend swap. The hard part is the
demand-paged working set and cold-start sequence.

**Stage 7 is not a separate implementation.** It's the emergent result of Stages
3–6 working together. If each stage is designed with the full architecture in
mind, Stage 7 is "just" integration and tuning.

---

## Effort Estimates

These are order-of-magnitude estimates, not commitments.

| Stage | Scope | Key risk |
|---|---|---|
| Stage 3 | New crate (`minkowski-lsm`), ~5-8 KLOC | Page-level dirty tracking in BlobVec without breaking SIMD alignment |
| Stage 4 | New crate (`minkowski-consensus`), ~3-5 KLOC | VR protocol correctness (use TLA+ spec or Loom verification) |
| Stage 5 | New crate (`minkowski-shard`), ~4-6 KLOC | Deterministic cross-shard scheduling without 2PC |
| Stage 6 | Extend `minkowski-lsm` + new `minkowski-objstore`, ~3-4 KLOC | Object storage latency hiding for cold start |
| Stage 7 | Integration + tuning, ~2-3 KLOC | Operational complexity (many moving parts) |

---

## References

- [TigerBeetle: The Secret to Scaling Databases (7 Stages)](https://tigerbeetle.com/) — the framework
- [Viewstamped Replication Revisited](https://pmg.csail.mit.edu/papers/vr-revisited.pdf) — consensus protocol
- [Calvin: Fast Distributed Transactions for Partitioned Database Systems](http://cs.yale.edu/homes/thomson/publications/calvin-sigmod12.pdf) — deterministic scheduling
- [LSM-based Storage Techniques: A Survey](https://arxiv.org/abs/1812.07527) — LSM tree design space
- [The Design and Implementation of a Log-Structured Merge-Tree (LSM-Tree)](https://www.cs.umb.edu/~poneil/lsmtree.pdf) — original LSM paper
