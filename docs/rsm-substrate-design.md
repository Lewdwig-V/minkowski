# RSM Substrate Design (Stage 4.0)

Date: 2026-08-30. Status: **4.0-a delivered** (PR #251, #252). This document records the substrate architecture as it exists in the code. The remaining 4.0 phases are listed in section 6.

## 1. Architecture summary

RSM is external composition. It does not change the internals of `World`. The replication machinery lives in `minkowski-persist`:

```
minkowski-persist
├── Durable<S>              wraps any Transact; WAL-first commit; stamps tick_after
├── Follower                replica apply state: high-water, poison, read_at gate
├── world_fingerprint       deterministic, order-independent state hash
└── WAL (wal.rs)            MKW3 frames [len][crc][view]; records carry tick_after
```

Four invariants carry the substrate. Three are delivered; one (replica mode) is pending.

| Invariant | Statement | Status |
|---|---|---|
| INV-1, commit = tick | each committed record replays at its commit-boundary tick | delivered |
| INV-2, view fencing | stale-view frames are fenced at every consumer of the log | delivered |
| INV-3, replica mode | `World` refuses `&mut` mutation on a replica | pending, 4.0-c |
| INV-4, `read_at` | one gate function for all replicated reads | delivered |

## 2. INV-1 — commit = tick

- `tick_after: u64` is a field of `WalRecord`. The leader stamps it at WAL-write time with the pre-apply world tick (`Durable` calls `world.current_tick()`).
- `apply_record` owns the tick semantics: it rejects a record whose `tick_after` is below the world's current tick (`WalError::TickRegression`), then sets the world tick to `record.tick_after`, then applies the mutations. Every apply path routes through `apply_record` — there is no bypass.
- Mark precision: `apply_record` sets the world tick to `tick_after` (the pre-apply value the leader stamped). `EnumChangeSet::apply` then advances the tick once and marks columns at the advanced value. The leader runs the identical sequence, so leader and replica column marks match exactly — one tick above the record's stamped boundary.
- Consequence for `Changed<T>`: the replica's tick tracks the leader's post-apply tick when no interleaved leader mutations exist between commits.
- The tick is a lattice, not a scalar: per-reader `Changed<T>` watermarks are process-local and do not survive failover. The documented failover contract: the first `Changed<T>` query after failover can see everything or nothing as changed. Readers re-read once. See section 4.1 of the test strategy in the docs for `QueryCacheEntry` details.

Convention sheet (INV-1 and INV-2 position semantics):

| Name | Meaning | Pinned by |
|---|---|---|
| `high_water` (Follower) | next expected sequence; all records below it are applied. 0 = nothing applied | `follower_baseline_seeds_high_water` |
| WAL sequences | 0-based, one per committed record, contiguous per leader | `convergence_100_transactions_leader_replica` |
| replay floor (`from_seq`) | `replay_from(from_seq)` applies records with `seq >= from_seq` — inclusive | `recover.rs` replay-floor invariant |
| baseline coverage | a restored baseline covers records strictly below `baseline_seq` | `follower_baseline_seeds_high_water` |

## 3. INV-2 — view fencing

WAL frame layout: `[len: u32][crc32: u32][view: u64][payload]` (16-byte header). Segment magic is `MKW3`; segments written with the previous 8-byte header fail loudly at magic validation.

`Views` is a monotonic counter on `Wal`, starting at 0. `bump()` moves it forward. Quorum-certified view installation (`Views::install`) is a 4.1 deliverable.

The fence rule, one sentence: **a consumer that has seen view V must not act on a frame with view < V.** The enforcement surface is a matrix — every consumer × every frame kind:

| Consumer | Schema frame | Mutations frame | Checkpoint frame |
|---|---|---|---|
| `replay_from` (recovery) | processed, raises the fence — a stale schema still builds the remap for its segment's records | fenced: dropped, never applied | fenced: skipped |
| `WalCursor::open` seek loop | raises the fence | raises the fence (no apply — records below `from_seq` are skipped by sequence) | raises the fence |
| `WalCursor::next_batch` | raises the fence | fenced: dropped, not shipped | raises the fence |
| `try_advance_segment` (cursor) | raises the fence from the new segment's preamble | n/a | n/a |
| `Wal::open` (crash recovery) | preamble stamped with the resumed view (views restore precedes a torn-header rewrite) | stale tail truncated at the frame offset — overwritten, so new-leader records reuse the sequence | counted by the header scan |

Fence maintenance:

- `Wal::open` resumes `views` from `scan_max_view` — a header-only scan (the view lives in the header; payloads are skipped via the length field) across **all** segments, active and sealed. It runs before a torn-header rewrite, so a rewritten preamble stamps the recovered view.
- `scan_active_segment` seeds its fence from the sealed segments' max view. A stale frame in the active segment truncates the file at that offset: the deposed leader's bytes are overwritten, and the next leader append reuses the stale record's sequence.
- Cursors seed `max_view_seen` from all segments before the resume point (`scan_max_view(&segments[..seg_idx])`), then from each frame they pass, including segment preambles and checkpoints.

Known fenced-out case with no special handling: a sealed segment can still hold a stale frame (sealed segments are immutable). Both replay and cursors fence it at read time.

Durability constraint for 4.1: the view record must be durable before the first prepare that carries the view, and the view history must never be truncated. A restarted replica that re-mints an old view forges the fence. This is a durability-ordering requirement, not a type-system requirement.

## 4. Follower

```rust
pub struct Follower { high_water: AtomicU64, poisoned: AtomicBool }
impl Follower {
    pub fn new() -> Self;                        // high_water = 0
    pub fn with_baseline(baseline_seq: u64);     // resuming onto an LSM-restored world
    pub fn advance(&self, batch: &ReplicationBatch, world: &mut World, codecs: &CodecRegistry)
        -> Result<u64, FollowerError>;
    pub fn read_at<R>(&self, seq: u64, world: &World, f: impl FnOnce(&World) -> R)
        -> Result<R, FollowerError>;
}
```

`advance` applies records one by one through `apply_record` (the same chokepoint recovery uses). Per record:

1. Sequence check. `seq < high_water` skips — the record is inside the applied prefix. `seq > high_water` is a gap: poison with `FollowerError::Gap`. Transport ordering violations are fatal, not skippable.
2. Tick. `apply_record` rejects a record tick below the world tick (poison) and sets the world tick to the record's boundary tick before mutations.
3. Apply. Any error poisons the follower. `AlreadyPlaced` is a refusable `ApplyError`, not idempotent success — idempotency is by position only. No retry, no rollback. Recovery from poison is rejoin via state transfer (`recover_world` from a peer). Poisoning is one `AtomicBool`; the rejoin path already exists.

`read_at(seq, world, f)` runs `f` only when `seq < high_water` (INV-4). It is the only surface the query language (3.75) may hook for replicated reads. Reads are bounded-staleness, consistent at a logged prefix — not linearizable. Linearizable reads go to the leader (4.1, read lease).

Pre-flight: `build_remap` runs before any mutation. Id-translation failures cost zero mutation.

## 5. The convergence test and the fingerprint

The substrate invariant is a test (`convergence_100_transactions_leader_replica`): 100 mixed transactions (spawn, insert, remove, despawn; POD + heap components) go through the leader's real WAL byte path to a fresh replica. The test asserts:

- `world_fingerprint(leader) == world_fingerprint(replica)`.
- Replica tick == leader tick (the leader runs no interleaved mutations).
- `applied_seq` == high-water == last shipped sequence + 1, follower not poisoned.

`world_fingerprint` properties:

- Archetypes are keyed by their sorted set of stable component names, resolved by `TypeId` (`CodecRegistry::stable_name_by_type`) — never by per-world `ComponentId`. Recovered worlds re-register types and may compact ids.
- Entities are keyed by packed `(index, generation)` bits.
- Component values hash through the codec path (`serialize_by_type`), so heap components hash by content, not by pointer bytes.
- Sparse components are not fingerprinted (dense-only for 4.0-a).
- The hash is deterministic within one binary; cross-version comparison is not a goal.

## 6. Remaining 4.0 phases

- 4.0-b, `Replicated<S>` + `Transport`: the leader-side component wrapping `Durable<S>`, with an outbound pump behind a `Transport` trait (in-memory channel for tests, TCP later). Promotion of `replication.rs` primitives.
- 4.0-c, replica mode (INV-3): the `World` write-mode flag. Every `&mut self` mutation entry point returns a typed refusal on a replica; `Follower::advance` is the only mutation path; `apply_record`'s internal bypass is the documented exception.
- 4.0-d, durable view record: view history persisted and validated at replay; real quorum installation stays in 4.1.

Deferred to 4.1 and later, unchanged: VR consensus core, leader election, io_uring unified I/O, zero-copy buffer identity, client session tables, reducer-intent replication, linearizable read leases.

## 7. Test inventory

| Test | Pins |
|---|---|
| `convergence_100_transactions_leader_replica` | fingerprint equality, tick equality, applied_seq |
| `follower_skips_applied_prefix` | position idempotency |
| `follower_poisons_on_mid_batch_failure` | poison semantics, applied-prefix state |
| `follower_gap_poisons` | gap rejection |
| `follower_tick_regression_poisons` | record tick below world tick |
| `follower_baseline_seeds_high_water` | baseline coverage convention |
| `read_at_bounds` | INV-4 gate |
| `stale_view_frames_dropped_by_replay` | replay fence |
| `cursor_seeks_seed_view_state` | seek-loop fence seeding |
| `cursor_seeds_fence_from_earlier_segments` | cross-segment fence seeding |
| `cursor_checkpoint_frames_raise_view_fence` | checkpoint frames count toward the fence |
| `open_resumes_view_across_segments` | view recovery across all segments |
| `open_truncates_stale_view_tail_in_active_segment` | overwrite-stale-slots at open |
| `frame_header_size_is_sixteen` | 16-byte header format pin |
| `rollover_many_appends_does_not_collide` | segment rollover under the new header |

Pending, listed so the gaps stay visible: loom coverage for `Follower` high-water/poison races, `fuzz_lsm_recovery` corpus entries with view-stamped frames (stale views included), and failpoint hooks in the outbound pump for the 4.1 simulator.
