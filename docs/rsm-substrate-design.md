# RSM Substrate Design (Stage 4.0)

Date: 2026-08-30. Status: approved direction. This document specifies the Stage 4.0 substrate for the Replicated State Machine (RSM). The phase plan is in section 4.

## 1. Context

Stage 4 (RSM) was to precede Stage 3.75 (the query language). The reason: RSM can change internals that the query language hooks. A design review against the code found a different picture.

RSM is external composition. It does not change the internals of `World`. The seams that RSM needs already exist in `minkowski-persist`:

| RSM requirement | Shipped piece |
|---|---|
| Ordered mutation log | `EnumChangeSet` — deterministic, ordered, applies atomically |
| Wire format | `ReplicationBatch` with `to_bytes` / `from_bytes` (rkyv, transport-agnostic) |
| Follower apply | `apply_batch` — applies records one by one. Section 2.7 gives the failure rules. |
| Log-to-local id translation | `build_remap` in WAL replay (keyed by stable type name, refusable) |
| State transfer | `recover_world` with `LsmRecovery` and manifest sequence bounds (Stage 3) |
| Durability tier | `Durable<S>` wraps any `Transact` |

TigerBeetle separates consensus state machines from data structures. The `Replicated<S>` type follows that lesson. It lands beside `Durable<S>` in `minkowski-persist`, not inside `World`.

The review narrowed "RSM changes internals" to four small invariants:

1. INV-1, commit = tick. Each committed record advances the world tick exactly once. `tick_after` rides the record, and replay compares against it.
2. INV-2, view fencing. The WAL frame header carries a `view` field. Stale-view frames die by generation mismatch, like stale `Entity` references.
3. INV-3, replica mode. `World` gains a read-only flag for replicas. This is the only change inside `World`.
4. INV-4, `read_at(applied_seq)`. This one function is the whole integration point for any future read surface, including the query language.

Everything else in the Stage 4 design moves to Stages 4.1 and later. Section 5 lists it.

## 2. Design

### 2.1 Component layout

```
minkowski-persist
├── Durable<S>            (existing — WAL durability)
├── Replicated<S>         (new — wraps Durable<S>, Transact delegates)
│     ├── outbound pump: committed changesets → batch frames → peers
│     └── Views: ViewId minted only by install(quorum cert)
├── Follower              (new — owns applied_seq: AtomicU64)
│     ├── advance(batch) → apply_batch
│     └── read_at(seq, |world| ..) → gate then query_raw
└── WAL frame header      (additive: view field; record: tick_after)
```

### 2.2 INV-1 — commit = tick

- `tick_after: u64` is a field of each committed mutation record. It is not a field of the `ReplicationBatch` transport envelope. A committed record advances the world tick exactly once, so the tick follows from log position.
- Granularity trap: a `ReplicationBatch` carries many committed records from `WalCursor::next_batch(limit)`. A batch-level tick would make `Changed<T>` semantics depend on the cursor batch size. The tick moves with the commit.
- Replay sets the world tick from the record. If the record and the world tick disagree, replay stops with an error.
- `apply()` marks columns with the replayed record tick. As a result, column `changed_tick`s rebuild correctly.

TigerBeetle lesson: "the log is all the state". This invariant extends the rule to state that today lives only in RAM.

### 2.3 INV-2 — view fencing

`ViewId` is a `Copy + Ord` newtype over u64. Replicas can hold, copy, compare, and stamp a `ViewId`. Only `Views::install` can mint one, and it requires a quorum certificate. This is the typed-construction rule from the ID principle table. Forging is prevented structurally.

- `ViewId` is stamped into WAL frame headers, record headers, IO completions, and client acks.
- The runtime check is one comparison at the receive boundary. `hdr.view < views.current()` means drop. No I/O ring quiesce. In-flight writes of a deposed leader die at the fence.
- Replay validates each frame view against the durable view history. Frames from unknown or superseded views are discarded, like orphan drains.

The main constraint: the view record must be durable before the first prepare that carries the view. The view history must never be truncated. If a restarted replica re-mints an old view, the fence forges. Entity generations do not face this problem because the allocator is in-process. This is a durability-ordering requirement, not a type-system requirement.

Decision deferred to 4.1: pack `ViewId` like `Entity` (low bits = replica slot, high bits = view generation). This adds a row to the ID principle table: `ViewId | quorum certificate | u64`.

### 2.4 INV-3 — replica mode

`World` gains a read-only flag for replicas. Every `&mut self` mutation entry point reads the flag. When the flag is set, the entry point returns a typed error. The only mutation path on a replica is `Follower::advance`. `apply_batch` bypasses the flag internally. This bypass is the documented exception (bypass-path rule in AGENTS.md).

Why this is necessary: convergence holds only if every state change flows through the log. Today `&mut World` hands out paths that bypass the WAL. A leader can hold unreplicated state with no error. The flag turns silent divergence into a typed error at the call site.

### 2.5 INV-4 — read_at(applied_seq)

```rust
impl Follower {
    pub fn read_at<R>(&self, seq: u64, f: impl FnOnce(&World) -> R) -> Result<R, Stale>
}
```

`read_at` compares `applied_seq` with `seq`. If `applied_seq` is behind, `read_at` returns `Stale`. Otherwise it runs the closure with a shared reference. Reads use `query_raw`. Ticks do not advance. This is the only surface that the query language (3.75) can hook for replicated reads. Plans from 3.75 run through `execute_*_raw` inside `read_at`. The planner stays RSM-unaware and gains snapshot reads.

If a 3.75 feature needs to bypass `read_at` on replicated state, we reject the feature or expand the surface. This matches the 3.75 typestate rule.

`read_at` gives bounded-staleness reads, consistent at a logged prefix. These are not linearizable reads. Linearizable reads go to the leader (4.1 and later, read lease protocol).

### 2.6 State that this substrate does not replicate

- Entity generations and orphan drains. Orphan drains fire at arbitrary local times on the primary, so generations do not follow from the log today. Stage 4.1 picks between two options: (a) log `GenerationBump(index)` records, or (b) move drains to apply time so generations follow from the log. Option (b) removes a record type but changes drain timing that live transactions observe. Until then, `apply_batch` explicit-id semantics define the replica generation state. The convergence test pins the behavior.
- Per-reader `Changed<T>` watermarks. These are process-local by design. Failover contract: the first `Changed<T>` query after failover can see everything or nothing as changed. Readers re-read once. The tick is a lattice, not a scalar. Per-query watermarks need read-lease machinery (4.1 and later).

### 2.7 Follower failure rules

`apply_batch` applies records one by one with no rollback. A failure at record k leaves records 0..k applied. `AlreadyPlaced` is a refusable `ApplyError`, not an idempotent-success signal. The substrate does not depend on atomicity:

- Idempotency by position. The follower never applies an applied prefix again. `applied_seq` is the dedup boundary.
- Pre-flight where possible. `build_remap` runs before any mutation. Id-translation failures cost zero mutation.
- Mid-apply failure = poison and rejoin. A failure inside the apply loop means the replica state and the log diverged. The follower stops. It does not continue past the failed record. It does not retry in place, because a retry keeps the applied prefix and diverges. It does not add rollback, because rollback duplicates the transaction engine. Instead `Follower` poisons itself: `advance` and `read_at` return `Poisoned` after that point. The replica rejoins with state transfer (`recover_world` from a peer, Stage 3 machinery). Poisoning is one `AtomicBool`. The rejoin path exists.
- Leader side. The leader commits only through `Durable` (WAL first, then apply). The log is always the recovery position. The leader does not need cross-record rollback either.

## 3. The convergence test

The acceptance criterion of the substrate is a test. It turns the implicit correctness of the `replicate` example into a proven invariant:

> 100 mixed transactions (spawn, insert, remove, despawn, heap and POD components) run on a leader through `Replicated<S>`. The batches move to a fresh replica world as transport bytes. The test asserts full equality through a world fingerprint: order-independent over archetype creation order, hashed by stable `type_name` (never per-world `ComponentId`), entities by `(index, generation)`.

The fingerprint is reusable. Stage 4.1 uses it for replay-equals-transfer proofs and for divergence refusal. It keys identities the way recovery does, by type and not by numeric id.

## 4. Phase plan

- 4.0-a, log headers and convergence test. `view: u64` in WAL frame headers, with a default for old frames. `tick_after: u64` on each committed record (section 2.2). Set-and-compare on replay. World fingerprint. The 100-transaction convergence test. Follower poison-and-rejoin rules (section 2.7) with the poison test. Views start at 0. `Views` exists as a monotonic counter. Quorum certificates come with 4.1.
- 4.0-b, `Replicated<S>`, `Follower`, `read_at`. Promotion of the `replication.rs` primitives to a component. The outbound pump sits behind a `Transport` trait. Tests use an in-memory channel. TCP comes later. `read_at` returns a `Stale` error.
- 4.0-c, replica mode. The `World` write-mode flag, refusal errors, and the documented `apply_batch` bypass. The failover contract goes into the `Changed<T>` docs.
- 4.0-d, stretch, follower view fencing. A durable view record and replay validation against view history. Real quorum machinery stays in 4.1.

Sequencing decision. After 4.0-a, Stage 3.75 can proceed in parallel. The query language hooks `read_at` (INV-4) and nothing else. 4.0 alone pays the RSM-first insurance. Stage 4.1 (VR core, io_uring, transfer wire protocol, client sessions) is decoupled from 3.75.

## 5. Out of scope

These stay in Stages 4.1 and later, unchanged from the roadmap:

- The VR consensus core: prepare, commit, quorum.
- Leader election.
- io_uring unified I/O with O_DIRECT and pinned buffers.
- Zero-copy receive-into-WAL buffer identity.
- Generation-isolated slot overwrites with real deposed leaders.
- Client session tables.
- Reducer-intent replication for thin clients.
- Linearizable read leases.

## 6. Test strategy

- Convergence test. Section 3 gives the definition. This is the substrate invariant.
- Fence test. Append frames at view 0. Bump the view. A late view-0 frame is dropped by replay.
- Tick test. A replayed log with a wrong record `tick_after` stops with an error.
- Replica-mode test. Every public `&mut` entry point on a replica world returns a typed refusal. `advance` is the only mutation path.
- Poison test. A follower fed a record that fails mid-batch refuses `advance` and `read_at` with `Poisoned`. State equals the applied prefix exactly.
- Loom. `applied_seq` advance and read races run under the loom suite.
- Fuzz. `fuzz_wal_replay` gets view-stamped frames with stale views in the corpus.
- Failpoint hooks. The outbound pump and the `advance` path get failpoint hooks now. The 4.1 deterministic simulator needs them. Retrofitting is harder.
