# RSM Substrate Design (Stage 4.0)

Date: 2026-08-30. Updated: 2026-09-05. Status: 4.0-a delivered (PR #251, #252).

Phase 4.0-b delivers application prerequisites, private recovery/raw-range plans and the raw divergent-world gate (section 7.7), plus the local durable raw-range reader (section 7.2). Journal-backed raw follower ingestion and restart from an empty baseline are also delivered (section 7.4). Nonempty state-transfer baselines, terminal dispositions, journal compaction, and transport remain planned. Sections 1–5 describe delivered behavior except the explicitly marked 4.0-b amendments. Sections 6–8 distinguish those changes from the remaining planned work. Proposed APIs and reserved test names are not implementation claims.

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
| `high_water` / `applied_seq()` (Follower) | 4.0-a: next expected sequence; all records below it are applied. 4.0-b: exclusive **settled prefix** — every slot below it is covered by the baseline, successfully applied, or authoritatively resolved as a terminal no-op (§4). 0 = no slots settled | `follower_baseline_seeds_high_water`; planned `follower_ingest_fenced_slot_advances_prefix` |
| `consumed_seq` (4.0-b protocol) | the same settled boundary exported by ingest for fetch and retention; not a second independently advancing cursor, a mutation count, or evidence that a fenced command committed | planned `follower_ingest_fenced_slot_advances_prefix` |
| `WalFrameRange.from_seq`, `next_seq`, `WalRangeLimits.max_records` (4.0-b) | mutation slots are exactly `[from_seq, next_seq)`; positive limits cap slots, bytes, and controls separately. Schema/checkpoint context consumes no slots | `records_from_range_boundary_edges` |
| `durable_next_seq` (4.0-b, internal published tail) | exclusive post-fsync mutation boundary; accompanying durable byte bounds also gate control frames (§7.2) | `records_from_excludes_unsynced_frames` |
| WAL sequences | 0-based, one per committed record, contiguous per leader | `convergence_100_transactions_leader_replica` |
| replay floor (`from_seq`) | `replay_from(from_seq)` applies records with `seq >= from_seq` — inclusive | `recover.rs` replay-floor invariant |
| baseline coverage | a restored baseline covers records strictly below `baseline_seq`; a 4.0-b resume also restores the corresponding fence and stream identity. If recovery replays a tail, seed from its reconstructed settled boundary, not the pre-replay floor | `follower_baseline_seeds_high_water`; planned `follower_ingest_restart_reconstructs_progress` |

## 3. INV-2 — view fencing

WAL frame layout: `[len: u32][crc32: u32][view: u64][payload]` (16-byte header). Segment magic is `MKW3`; segments written with the previous 8-byte header fail loudly at magic validation.

`Views` is a monotonic counter on `Wal`, starting at 0. `bump()` moves it forward. Quorum-certified view installation (`Views::install`) is a 4.1 deliverable.

The delivered 4.0 storage fence rule is: **after seeing view V, a consumer must not apply mutation effects from a frame with view < V.** Schema metadata remains usable for remapping; validating or transporting stale bytes does not authorize their effects. This rule describes the single-source storage paths below. It is not the 4.1 rule for selecting a committed history across views; committed older-view operations must be preserved (§6). The enforcement surface is a matrix — every consumer × every frame kind:

| Consumer | Schema frame | Mutations frame | Checkpoint frame |
|---|---|---|---|
| `replay_from` (recovery) | processed, raises the fence — a stale schema still builds the remap for its segment's records | fenced: dropped, never applied | fenced: skipped |
| `WalCursor::open` seek loop | raises the fence | raises the fence (no apply — records below `from_seq` are skipped by sequence) | raises the fence |
| `WalCursor::next_batch` | raises the fence | fenced: dropped, not shipped | raises the fence |
| `try_advance_segment` (cursor) | raises the fence from the new segment's preamble | n/a | n/a |
| `Wal::open` (crash recovery) | preamble stamped with the resumed view (views restore precedes a torn-header rewrite) | stale tail truncated at the frame offset — overwritten, so new-leader records reuse the sequence | counted by the header scan |
| `records_from` (4.0-b raw range reader) | enforced: include each run's schema, even stale when sealed; refuse stale active schemas; `records_from_mid_segment_includes_schema_context`, `records_from_resume_preserves_fence_context` | exempt from filtering: preserve every returned slot; refuse stale mutations until durable terminal dispositions exist; never publish a rewritable stale active tail; `records_from_refuses_unresolved_history` | enforced: preserve durable control order and prefix fence context; refuse stale active controls; `records_from_resume_preserves_fence_context`, `records_from_refuses_unresolved_history` |
| `Follower::ingest_frames` (4.0-b, shipped ranges) | enforced: validate/build the run's remap even when stale; raise fence with `max`, never lower it; `follower_ingest_processes_stale_schema_frame` | enforced: reject stale effects before `apply_record`; settle an expected stale slot only with a terminal disposition (§4), otherwise refuse; `follower_ingest_drops_stale_mutation_frame`, `follower_ingest_fenced_slot_advances_prefix`, `follower_ingest_refuses_unresolved_fenced_slot` | enforced: raise fence with `max`; ignore `flush_seq` for world/progress/baseline coverage; `follower_ingest_checkpoint_raises_fence` |

Fence maintenance:

- `observe_view` owns the fence arithmetic for header scans, recovery, cursors, and raw range reads. Consumers decide whether a stale frame can supply schema context, must be filtered, or requires refusal/truncation.
- `Wal::open` resumes `views` from `scan_max_view` — a header-only scan (the view lives in the header; payloads are skipped via the length field) across **all** segments, active and sealed. It runs before a torn-header rewrite, so a rewritten preamble stamps the recovered view.
- `scan_active_segment` seeds its fence from the sealed segments' max view. A stale frame in the active segment truncates the file at that offset: the deposed leader's bytes are overwritten, and the next leader append reuses the stale record's sequence.
- Cursors seed `max_view_seen` from all segments before the resume point (`scan_max_view(&segments[..seg_idx])`), then from each frame they pass, including segment preambles and checkpoints.

A sealed segment can still hold a stale frame (sealed segments are immutable). Recovery and the existing decoded cursor filter its effects/records. The 4.0-b raw range reader preserves its slot for ingest to settle only when the source supplies a durable terminal no-op disposition; otherwise the range requires repair/rejoin. It must not use `WalCursor::next_batch`'s filtered result as a complete sequence range.

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

**4.0-b amendment — settle slots, apply eligible mutations.** Raw ingest validates framing, CRCs, range continuity, and every schema/remap needed by the batch before touching the world. It uses one shared frame/fence interpreter with recovery; `apply_record` remains the tick/remap/apply chokepoint and does not itself inspect frame views. `advance` and raw ingest share positional/poison publication logic; the decoded `advance` path cannot stand in for raw frame fencing.

At `seq == consumed_seq`, a valid current mutation advances the boundary to `seq + 1` only after successful `apply_record`; a fenced mutation with an authoritative terminal no-op disposition advances it to `seq + 1` without calling `apply_record` or changing the world/tick. A lower sequence is a positional duplicate; a higher sequence is a real gap and poisons. Schema/checkpoint frames update context, never sequence progress. A response's claimed `next_seq` cannot advance follower state. Malformed input, remap failure, tick regression, or apply failure is never a fenced no-op: stop and poison, with no publication beyond the last settled slot and no successful progress return. Reads refuse poisoned state; failed apply is not promised to roll back.

An older frame view or a sealed file is not, by itself, authority to replace a logical operation with a no-op. Ingest requires the source's durable terminal disposition for that slot; without it, refuse the ambiguous range and request repair/rejoin without advancing past the slot. This is a condition on the fixed-source 4.0-b history, not a substitute for the 4.1 consensus decision. A committed operation from an older view remains committed.

Example: start at 40; a checkpoint raises the fence to 3; mutation `(40, view 2)`, durably resolved by the authoritative source as a terminal no-op, settles as a no-op, moving the boundary to 41; valid mutation `(41, view 3)` applies and moves it to 42. A batch containing only the fenced slot still moves the next fetch to 41. `read_at` keeps its exclusive check but, in 4.0-b, means the world's projection of this settled prefix. It does not certify that a rejected stale command succeeded. Keeping a separate fetch cursor while freezing the read boundary would leave reads permanently stuck; the convention sheet therefore explicitly changes the prefix definition.

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

**Phase boundary:** 4.0-b is asynchronous replication of an authoritative source's locally committed, finalized history. It does not provide quorum commit or safe automatic failover. 4.1 changes the commit protocol: durable prepare receipt, quorum commitment, and state-machine application are distinct boundaries. A prepare acknowledgment must not depend on application, or waiting to apply until commit creates a cycle. Push is the intended normal-delivery shape for a VSR-style protocol; pull remains useful for catch-up/repair. The absence of synchronous network I/O in the local storage call can remain, but client commit completion must depend on the required replication quorum.

- 4.0-b, `Replicated<S>` + follower pump + pull transport. Deliverables: durable raw range reader; differential round-trip gate before wire code; shared ingest/fence/position handling and restart coverage; `transport.rs` with `Fetch`, `RecordingFetch`, loopback and caller-driven `ReplicationPump::pump_once`; leader retention registry and seeded chaos fixture. Section 7 specifies the ownership and ordering.
- 4.0-c, replica mode (INV-3): the `World` write-mode flag. Every `&mut self` mutation entry point returns a typed refusal on a replica; follower ingestion is the only replicated mutation path; `apply_record`'s internal bypass is the documented exception.
- 4.0-d, durable view record: view history persisted and validated at replay; real quorum installation stays in 4.1.

Deferred to 4.1 and later, unchanged: VR consensus core, leader election, io_uring unified I/O, zero-copy buffer identity, client session tables, reducer-intent replication, linearizable read leases.

## 7. 4.0-b design: follower pull and leader retention (decision record)

**The WAL is the outbound queue.** The commit path never calls transport or hands off a batch. The follower requests its next unsettled slot; the leader retains durable bytes and a progress registry. Pull has no separate acknowledgment message, but it does have progress reporting and leader-side retention state. Push remains 4.1.

### 7.1 Components and ownership

| Owner | Owns | Responsibility |
|---|---|---|
| Leader `Replicated<S>` | `Durable<S>`, WAL read access, source disposition journal, registry keyed by follower identity and session generation | Delegate transactions unchanged; serve ranges; track reported settled boundaries for retention |
| Follower `ReplicationPump` | fetch client bound to that identity/session, `Follower`, ingest journal | Caller invokes `pump_once`; fetch, ingest, then use ingest's returned boundary for the next request; world/codecs are supplied to apply |
| Each fetch response | detached owned buffer with a self-describing `BatchRef` | No WAL lock or segment pin crosses network handoff |
| 4.1 VSR normal delivery | leader prepare pipeline, send transport, durable-receipt acknowledgment | Persist prepares and acknowledge receipt before application; apply only after quorum commitment (§7.5) |

There are no leader-owned per-follower transports, persistent shipping cursors, or pump threads in 4.0-b. The read operation resolves and pins its source segments long enough to copy a consistent response, then releases them. Follower lag affects retention, not a commit-to-transport dependency.

The registry stores the last valid `consumed_seq` reported by each registered session. Registration/rejoin binds a verified baseline and log identity to a new session generation; old-session requests cannot update its entry. Updates are monotonic within a session, cannot exceed the published source boundary, and never use the end of a response as evidence of follower progress. A duplicate or delayed request can conservatively lag the registry; it cannot move it backward or invent progress. Restart blocks retention until the configured membership and conservative progress floors are restored/re-established. Disconnect does not silently remove a member; removal is an explicit policy action.

Retention's candidate cutoff is `min(leader_replay_floor, min(registered consumed_seq))`. `leader_replay_floor` is the recoverable LSM baseline's replay floor, not merely `last_checkpoint_seq`; without a baseline it is 0. With no registered followers, only the leader floor applies. Delete only whole sealed segments whose exclusive end is at or below the cutoff, after existing read pins release. Preserve the fence context needed at the retained floor: until a durable fence-at-floor summary exists, keep the segments needed to reconstruct it. Never erase view history required by §3. A follower below the retained floor gets a typed rejoin response; it must install state, not skip forward.

### 7.2 The durable range boundary

The local reader is `Wal::records_from(from_seq, WalRangeLimits)`.
Limits bound mutation slots, returned bytes, and returned control frames independently; zero limits refuse.
The detached `WalFrameRange` contains exact schema, mutation, and checkpoint frames in segment runs.
It is a local read result, not yet a versioned transport envelope or proof of follower application.
It carries no terminal no-op decisions until the source disposition journal exists.

The WAL publishes a sequence boundary and per-segment byte endpoints only after successful file synchronization.
Creation also synchronizes directory ancestry; rollover synchronizes the new segment and its directory before publication.
Reopen synchronizes recovered files and directory entries before publishing their retained endpoints.
The reader holds a shared WAL borrow while copying; append and deletion require exclusive access.
Deletion updates the published segment set, and a missing prefix requires rejoin until durable fence-at-floor metadata exists.

| Reader input or transition | Required behavior | Check |
|---|---|---|
| Mid-segment start and rollover | Return exact original schemas and frames in separate runs. | `records_from_mid_segment_includes_schema_context` |
| Schema and checkpoint prefix | Seed only from the prefix before the first returned mutation; tail requests include durable control context. | `records_from_resume_preserves_fence_context` |
| Unsynced writes | Exclude sequence slots and control bytes beyond published endpoints, including after failed fsync. | `records_from_excludes_unsynced_frames` |
| Create, rollover, and reopen | Publish only synchronized files and directory entries. | `records_from_survives_reopen_and_rollover` |
| Failed rollover | Remove the uncommitted segment and synchronize deletion before later writes or retention; failed cleanup blocks both until cleanup succeeds. | `records_from_survives_failed_rollover_sync`, `failed_rollover_cleanup_blocks_writes` |
| Stale or missing history | Refuse unresolved slots and a stale active suffix; never infer terminal no-ops. | `records_from_refuses_unresolved_history` |
| Corrupt published bytes | Refuse bad magic, CRCs, missing schemas, and torn frames. | `records_from_rejects_corrupt_published_frames` |
| Request and size bounds | Refuse zero limits and past-tail requests; return a bounded contiguous prefix or a size refusal. | `records_from_range_boundary_edges` |
| Retention | Refuse below-floor requests and missing prefix fence context after deletion or reopen. | `records_from_requires_retained_fence_context` |

`wal.rs::Wal::append` advances allocated `next_seq` before fsync. `sync_active_prefix` publishes `durable_next_seq` and the active byte endpoint only after successful fsync; `acknowledge_flush` now uses the same helper. The scalar tail alone cannot authorize shipping a later unsynced checkpoint or using its view to seed a range. Schema and checkpoint frames consume no sequence slots, so byte endpoints also gate control-only progress.

The reader currently scans and validates the retained prefix to reconstruct the exact seed fence and detect unresolved history. Returned bytes and controls are bounded; prefix scan work is proportional to history before the requested range. Add a durable seek/fence index before using repeated small reads for large-log catch-up. After retention removes the prefix, raw reads require rejoin until a durable fence-at-floor summary exists.

For `records_from(seq, limits)`, positive `max_records` caps mutation slots; byte and control limits can shorten that prefix further. If the first mutation and its schema cannot fit, return `RangeLimitTooSmall`. Requests at the published tail return no mutation slots; requests beyond it return `RangeAhead`; sequence arithmetic is bounded by the published tail. A sequence is not durable merely because it was allocated. Terminal no-op slots remain unsupported until the disposition journal below exists.

The published prefix must also be stable under source recovery. Sealed stale bytes are immutable, but that storage fact alone does not establish the logical slot's outcome. A stale slot may be transported as a terminal no-op only when the authoritative source has durably finalized that disposition. Without a disposition, return unresolved-history/rejoin-required; never manufacture one from the frame view or from file sealing. Under 4.1, view-change/commit rules must establish the selected history, preserving committed older-view operations. A stale active suffix is different: `scan_active_segment` can truncate it and reuse its sequence numbers. Stop range publication before such a rewritable suffix and require source recovery/rejoin; do not report caught-up while unresolved slots remain. No slot already advertised as settled may later be silently reused in the same stream/session.

**Source disposition lifecycle (planned):** Source recovery/finalization under `Replicated<S>` is the sole creator of terminal no-op decisions and owns their source-side journal; `records_from` only reads its published durable snapshot. Creating a decision requires an explicit final outcome from the configured source's recovery protocol that excludes any committed operation at that slot. Neither a higher view nor sealed bytes supply that authority, and a previously published mutation or committed older-view operation cannot be reclassified. If source recovery cannot establish this outcome, refuse the unresolved slot and require repair/rejoin; 4.0-b does not add a view-change or quorum finalization algorithm.

Before publishing a no-op, durably create/append a checksummed journal record binding source-history identity, sequence and original frame identity to that immutable decision, then fsync it. Source restart restores and validates this journal before enabling recovery-dependent range publication; source recovery and range reads use the same decision map, and finalized slots cannot be reused. An unflushed or incomplete record cannot authorize a no-op; a crash after fsync must restore the same decision even if no response was sent. Reject conflicting decisions or identities; missing/corrupt required disposition state requires repair/rejoin, never inference from stale bytes. Retain this journal for the lifetime of the source history in 4.0-b, independently of WAL segment retention; journal compaction is deferred. This is new persistence work, covered by `terminal_disposition_survives_source_restart`.

### 7.3 Self-describing raw ranges and ingest

Use a versioned envelope identifying MKW3 frames:

```text
BatchRef { from_seq, next_seq, seed_view, runs }
SegmentRun { segment_start_seq, schema_frame, frames, terminal_noops }
```

These are proposed fields, not existing types. `schema_frame` is the exact original schema frame, including its header/CRC/view; `frames` contains original mutation and checkpoint frames in source order, without segment magic. The envelope identifies run boundaries; source-side reads validate segment magic. `terminal_noops` identifies source-resolved no-op slots by sequence and original frame identity, bound to the registered source history; it must come from durable source dispositions, not be inferred by the range reader from age/sealing. It is persisted with the ingest journal. 4.0-b trusts the configured authoritative source; this field is not a quorum certificate. Mutation payloads are never reserialized per follower.

Every nonempty range includes the active schema at its start, even for a mid-segment resume or a fresh fetch client. Each crossed segment starts a new run with its own schema. Schema context is outside the mutation interval and does not count toward `limit`, advance `consumed_seq`, or represent reapplication of an older record. Remaps are scoped to their runs and rebuilt from stable names; there is no identity-ID fallback or dependence on a previous response's remap. Validate all required remaps before mutation.

`seed_view` is the maximum durable view established by the source prefix before the first returned mutation, including skipped frames and earlier segments, plus any retained fence-at-floor context. It is not the newest view from the end of the log: later views must not retroactively fence earlier valid mutations. Ingest combines it with its restored fence using `max`, then processes run schemas and frame views in order. For an empty range it describes only the durable control prefix reached at that position. A leader checkpoint's `flush_seq` never claims follower baseline coverage.

Mutation frames cover exactly `[from_seq, next_seq)`, once per slot and in order, including stale frames with explicit terminal dispositions. A stale frame without such a disposition is an unresolved logical slot, not an automatically consumable no-op. No server-side filtering creates invisible holes. A response may contain no mutations and still provide durable context; repeated context is harmless. Reject missing/incorrect schema, bad CRCs, format errors, unaccounted gaps, and inconsistent envelope bounds before applying.

The proposed raw ingestion chokepoint is delivered as `JournaledFollower::ingest_frames(history, range)` for the empty-baseline subset described in section 7.4; the wrapper owns its world and codecs. After preflight, persist the complete replayable range and its schema/fence context in a follower ingest journal and fsync it before application or advertising progress. Repeated ranges remain positionally deduplicated, including during recovery. This is new raw-frame persistence work: `Wal::append` currently serializes changesets and cannot be assumed to append received frames. Treat a journal as replayable range envelopes; blindly concatenating response buffers into a local leader WAL is not the format contract.

Apply and publish per the shared §4 state machine. Only its successful return may supply the next pull position or a future applied-progress Ack for finalized-range push (§7.5); a failed journal write/apply produces no successful progress result. On restart, restore a consistent world baseline, its fence and stream identity, then replay the journal through the same interpreter/position rules (without re-journaling) before reconstructing progress. Never restore a saved number onto an older world. `replay_from` alone and `recover_world` alone do not reconstruct follower progress today. Volatile-only acknowledgment is outside this 4.0-b contract.

**First semantic gate, before wire code:** `wal_frames_round_trip_divergent_live_follower` now starts from equivalent leader/follower baseline state, with pre-existing entities and divergent component registration/allocation history. It passes exact ranged frame bytes through the private plan, including a mid-segment start and rollover, and verifies fingerprints, ticks, the executed range end, and subsequent allocation. This proves raw application equivalence, not durable follower progress: journal-backed ingestion separately establishes the applied boundary for its empty-baseline subset. Merely decoding an entire segment into a fresh world does not prove this contract.

### 7.4 Pull contract and failure behavior

The delivered journal slice adds `JournaledFollower`, owning its world, codecs, legacy `Follower` position/poison state, source-history identity, restored view, and ingest file.
It starts from an empty world at sequence zero; opening reconstructs that world from the retained journal before returning any progress.
Nonempty state-transfer baselines and journal compaction remain deferred. No journal prefix can be deleted in this slice.
The caller supplies a stable 16-byte source-history identity and must change it when replacing the source history.
`ingest_frames(history, range)` checks that identity, preflights the complete range, appends and fsyncs its envelope, then executes only the unapplied suffix.
The owned world is accessible only through the existing poisoned/read-at gate, so progress cannot be paired with another world or bypassed by local writes.
Decoded and journaled application share per-record position checking, poison handling, and publication after successful apply.

Journal format: `MKI1`, 16-byte history identity, CRC32 of those header bytes; then 16-byte length/CRC/complemented-length headers and rkyv envelopes containing history plus the original `WalFrameRange`. The length guard is checked before allocating or interpreting a partial tail, so a damaged length cannot silently discard later committed entries.
Creation fsyncs the file and directory ancestry before returning. Reopen fsyncs recovered bytes before replay; only an incomplete final frame may be truncated, followed by fsync. Complete checksum/format errors refuse.
Empty ranges that raise the fence are journaled and fsynced even though they consume no sequence slots. Complete duplicates that add no context need no extra journal entry.
Overlapping responses retain their full original bytes in the journal; sequence deduplication and restored fences select only the new suffix on ingestion and restart.
Stale mutation slots still refuse until authoritative terminal dispositions exist.

| Ingest/restart input or boundary | Required behavior | Check |
|---|---|---|
| Valid raw range | Fsync before effects; publish each slot only after successful apply. | `journaled_follower_round_trip_and_restart` |
| Duplicate/overlap and empty controls | Skip applied effects; retain the restored fence; persist new control-only context. | `journaled_follower_deduplicates_overlap_and_controls` |
| Bad schema/frame, gaps, bounds, or history | Poison without new effects or successful progress; reads refuse. | `journaled_follower_refuses_invalid_input` |
| Invalid component value with valid CRC | Checked decoding refuses on ingest and restart, including raw-copy-certified types. | `journaled_follower_checks_component_bytes_despite_valid_crc` |
| Write/fsync failure and late apply failure | No successful result; preserve only applied prefix, poison reads, retain replayable evidence. | `journaled_follower_failure_boundaries` |
| Restart and partial tail | Reconstruct world, ticks, fence, and sequence from the same history; truncate only incomplete tail. | `journaled_follower_restart_reconstructs_progress` |
| Corrupt or mismatched journal | Refuse recovery; never restore a number onto an unrelated world. | `journaled_follower_rejects_corrupt_history` |
| Concurrent journal owner | Acquire a nonblocking exclusive file lock before reading or writing; refuse a second owner. | `journaled_follower_rejects_corrupt_history` |

The failure-boundary tests distinguish process restart (page-cache bytes may survive) from explicitly discarding unflushed tail bytes. Copying a failed journal plus the matching codec setup reproduces application failure from the fixed empty baseline; broader state-transfer failure-capture manifests remain planned.

The fetch client is bound to one registered follower/session. Its synchronous wire operation remains:

```rust
fn fetch(&self, seq: u64, limit: usize) -> Result<BatchRef, TransportError>
```

The pump obtains `seq` only from the restored or successfully ingested `consumed_seq` (§2), never from a received range's end. The request reports the previous completed prefix for retention and asks for the next slots. The leader may record that report even if the response is lost. At catch-up, a subsequent fetch reports the final progress even when it returns no mutation frames. There is no separate Ack message. A raw integer on the wire is not proof of application: correctness rests on this session-bound pump/ingest discipline, so remove the claim that premature acknowledgment is impossible merely because the topology is pull.

4.0-b permits one outstanding fetch per follower. `Lost` retries the same position; `Down` backs off and reconnects. Neither changes progress. `CursorBehind`/rejoin-required and source-history changes require state transfer; corruption, remap failure, sequence gaps and tick/apply failures are not transport loss. Duplicate complete responses deduplicate by position. Frame reordering/gaps are rejected; chaos reordering does not license applying out of sequence.

Fixtures are `RecordingFetch`, loopback, and a seeded chaos fetch with pinned seeds (loss, duplicate responses/requests, delay, and invalid order). A blocked fetch holds its detached response buffer, not a leader WAL lock.

### 7.5 4.1 consensus and push

A push optimization for shipping already-finalized ranges can reuse this ingest contract, with an opaque applied-progress Ack. A VSR prepare path has a different acknowledgment: durable receipt before application, followed by quorum commit notification and then application. Do not reuse the applied-progress Ack as `prepare_ok`. Likewise, a locally fsynced `durable_tail` does not identify the quorum-committed prefix. Separate these boundaries when designing 4.1; adding `send_batch` alone does not implement consensus. `RecordingTransport` and leader shipping cursors belong to that phase; the raw range reader remains useful for repair/catch-up.

### 7.6 Fence matrix and implementation boundary

Section 3 adds **consumer rows** for raw range production and follower ingestion, with Schema/Mutations/Checkpoint cells and planned test names. Reuse shared frame validation/fence interpretation and positional publication; new ingress paths must not clone those rules. `apply_record` remains the convergence point for eligible mutation effects. Section 8 reserves the tests before implementation.

### 7.7 Selected implementation work

Decision: 2026-09-05. Adopt the three proposals below. Start with the private execution plan and its application prerequisites.
These proposals preserve the ownership, durability, and sequence contracts in sections 7.1–7.6.

#### Private execution plan

`ValidatedRange` is a private, temporary execution plan for one raw range.
It holds the original bytes, frame offsets, schema bindings, and component mappings.
The shared frame interpreter constructs the plan before any world mutation.
It makes sure that every frame, sequence, schema, component reference, and terminal disposition satisfies the range contract.

Build and consume the plan during one ingestion call with exclusive access to the same world.
Keep codec identifiers separate from destination component identifiers.
Descriptors contain offsets and identifiers instead of references into world storage.
Bound response bytes and control-frame counts separately from the mutation-slot limit.

Persist and fsync the original range envelope before execution.
Execute eligible mutations through `apply_record` and the shared settlement rules.
State-dependent application failures still poison the follower and can leave partial effects.
On restart, rebuild the plan from the journal and skip the journal-write stage.
Do not persist the plan or reuse its numeric mappings in another world.

The first implementation slice exercises the existing decoded follower path against equivalent, nonempty worlds with different component registrations and allocator histories.
It establishes component mapping and logged-entity allocation before raw-range execution starts.
`build_apply_remap` resolves each source component to a codec and a destination component by type.
The schema can declare unused components that the destination does not register.
Each record must resolve its referenced components before it changes the world or tick.
WAL frames and incoming decoded batches use aligned storage before archive validation.
Transport buffers and ordinary byte vectors do not guarantee the archive's required alignment.
`EnumChangeSet::apply_replay` uses the existing mutation loop and claims logged entity slots when each spawn executes.
Logged spawns must claim their specified slot without consuming an unrelated free slot.
Recycled-slot lookup must avoid repeated scans and preserve the order of untouched free slots.
The allocator keeps the ordinary tail pop and lazily indexes non-tail removals.
Removed entries use internal tombstones, with amortized compaction and a dense snapshot view.

| Free-list consumer | Ordering and membership | Check |
|---|---|---|
| Replay adoption | Remove the specified free slot; preserve the order of other slots. | `replay_recycled_slots_preserves_order_and_snapshot` |
| Local allocation and despawn | Pop the last live slot and append returned slots; update any existing index. | `replay_recycled_slots_preserves_order_and_snapshot` |
| Snapshot read and restore | Export the original dense format; invalidate cached views on mutation and rebuild indexes after restore. | `replay_recycled_slots_preserves_order_and_snapshot` |
| Repeated recycled spawns | Tail pops remain constant time; indexed removals and compaction are amortized constant time. | `replay_recycled_slots_scaling` (release timing, tail/head/permuted orders) |

An occupied slot or an older generation requires a typed refusal.
Process entity allocation in mutation order so a despawn and subsequent spawn can reuse one slot within a record.
This prerequisite test does not replace `wal_frames_round_trip_divergent_live_follower`, which also requires exact raw ranges, schema runs, and durable bounds.

The application prerequisites use this invariant matrix:

| Consumer | Component identifiers | Entity allocation | Tick and failure behavior |
|---|---|---|---|
| `apply_record` through decoded follower or WAL recovery | Resolve source → codec → destination type. `follower_round_trip_divergent_live_world` | Claim the logged slot in mutation order. `follower_replay_reuses_logged_entity_slot_in_mutation_order` | Keep the existing per-record tick and poison rules. `follower_tick_regression_poisons` |
| Private raw `ValidatedRange` execution | Resolve all runs before effects. `raw_plan_rejects_invalid_range_before_effects` | Use the same `apply_record` path. `wal_frames_round_trip_divergent_live_follower` | Journal-before-effects and shared per-record publication. `journaled_follower_failure_boundaries` |
| Authorized terminal no-op, planned | Validate its run schema. `follower_ingest_processes_stale_schema_frame` | Exempt: no entity allocation occurs. `follower_ingest_drops_stale_mutation_frame` | Advance only the settled prefix. `follower_ingest_fenced_slot_advances_prefix` |

Local WAL recovery now uses the private `ValidatedRange` raw-frame plan.
It retains selected original frames and resolves every referenced component before execution.
The plan exclusively borrows its destination world until execution finishes.
Execution decodes the retained mutation frames again and calls `apply_record`.
Preflight and execution share `validate_record_tick`.
Each selected record must start at or after the preceding record's predicted post-apply tick, including the increment for an empty changeset.
Tick overflow refuses before application.
Component payload decoding and state-dependent application can still fail after earlier records have applied.
Local recovery keeps its existing stale-frame and torn-tail rules.
This caller does not establish contiguous finalized ranges or authorize follower progress.
The durable range reader (section 7.2) and journal-backed raw ingestion (section 7.4) are available. Unresolved stale slots still refuse pending terminal dispositions.

`wal/plan.rs::ValidatedRange::from_frames_after` extends the private plan to detached `WalFrameRange` inputs and passes the raw divergent-world gate.
It validates one complete range against the destination world and an explicitly supplied restored fence.
It validates overlapping deliveries and selects only the unapplied suffix. `JournaledFollower` journals bytes before execution, poisons on failure, and uses the shared `Follower::apply_next` boundary to publish progress.
Execution reports the local recovery last-sequence convention and the final fence for the internal gate. Local recovery keeps its inclusive return and empty-range fallback; this result is not a follower acknowledgment.
The public journal owner calls this private constructor on both live ingestion and restart. Received component payloads use checked decoding even when their frame CRC is valid; local WAL recovery retains its existing certified raw-copy path.
Schema remapping rejects duplicate source IDs and duplicate names at the shared `build_apply_remap` boundary, including decoded replication and recovery callers.

| Raw plan input | Before effects | Check |
|---|---|---|
| Mid-segment range crossing rollover | Validate exact original schema/frame buffers; resolve source, codec, and destination IDs independently; adopt logged slots. | `wal_frames_round_trip_divergent_live_follower` |
| Schema and checkpoint | Require one exact schema per ordered run; raise the restored/source fence with the shared interpreter; checkpoint sequence is not baseline coverage. | `raw_plan_preserves_run_remaps_and_fences` |
| Mutation | Require exact interval coverage, current view, valid component references and predicted ticks before any effects. | `raw_plan_rejects_invalid_range_before_effects` |
| Malformed or oversized input | Check interval, byte and control limits, full frames, CRCs, and run boundaries before execution; truncated buffers must not allocate their claimed payload size. | `raw_plan_rejects_invalid_range_before_effects`, `raw_plan_enforces_limits`, `raw_frame_buffer_checks_lengths_and_alignment` |
| State-dependent failure | Preserve the existing partial-application error; do not return a successful completion boundary. | `raw_plan_propagates_partial_apply_failure` |

| Recovery plan input | Before effects | Execution |
|---|---|---|
| Schema | Validate codec layout and build destination mappings, including stale schemas. `replay_plan_preserves_schema_and_fence_across_resume` | Exempt: schema does not mutate the world. |
| Eligible mutation | Validate every component reference and the predicted post-apply tick, including overflow. `replay_preflight_rejects_late_component_reference`, `replay_preflight_validates_post_apply_ticks` | Use the same tick guard in `apply_record`. `follower_tick_regression_poisons`, `replay_preflight_validates_post_apply_ticks` |
| Stale mutation | Validate framing, then retain the existing local recovery exclusion. `stale_view_frames_dropped_by_replay` | Exempt: no application or progress publication. |
| Checkpoint | Raise the fence in source order. `replay_plan_preserves_schema_and_fence_across_resume` | Exempt: checkpoint does not mutate the world. |
| Corrupt frame or partial tail | Corruption refuses before effects; partial tails keep the existing recovery behavior. `replay_plan_preserves_schema_and_fence_across_resume`, `torn_entry_truncated_on_replay` | Exempt: no corrupt or partial frame executes. |

#### Replayable failure captures

A failure capture is a diagnostic fixture that reproduces one ingestion failure.
Its manifest references the verified baseline, preceding durable journal, and original failing range.
It records the source history, frame identity, interpreter stage, settled boundary, and poison state as reproduction assertions.
The fixture must preserve the allocator and component-registration context needed to reproduce the destination state.

The replay harness restores these dependencies and uses the normal interpreter.
It rebuilds component mappings from the destination world.
Capture is best effort and cannot change ingestion results, recovery authority, or progress.
Start with an in-memory fixture before defining an export format.
Later fixtures can include fetch transcripts and reduced failure cases that retain their required schema and fence context.

#### Deterministic crash sequencer

A crash sequencer interrupts a repeatable execution trace at named persistence boundaries.
Cover source writes, disposition-journal fsync, publication, follower-journal fsync, application, and the next progress report.
After each interruption, compare the reconstructed world, tick, fence, and settled prefix.
Include a successful apply followed by a lost progress report and an empty fetch after restart.

Separate process-restart tests from simulated power-loss tests.
A file reopen does not discard bytes that remain in the operating system's cache.
The power-loss model must include segment creation and rollover durability before those cases support durability claims.
Keep the model limited to the WAL and replication journals.
A durable disposition record stores an authoritative decision but cannot establish finality by itself.

Delivery order:

1. Establish the decoded application prerequisites and their regression tests.
2. Build the durable range reader and the shared interpreter for `ValidatedRange`.
3. Pass the raw divergent-world gate before transport work.
4. Add journals, restart reconstruction, crash tests, and failure captures.
5. Add the pull pump, session registry, retention, and seeded transport faults.

## 8. Test inventory

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
| `follower_round_trip_divergent_component_ids` | Source, codec registry, and destination use different component identifiers. Values reach the correct destination columns. |
| `follower_round_trip_divergent_live_world` | Equivalent nonempty worlds have different component registrations and free-list orders. Heap values, tick, progress, retries, and subsequent allocation agree. |
| `follower_replay_reuses_logged_entity_slot_in_mutation_order` | A despawn, replacement spawn, and insert reuse the logged slot within one record. |
| `follower_replay_refuses_occupied_or_older_entity_generation` | Replay refuses occupied slots and older generations without changing allocator state or publishing progress. |
| `follower_replay_claims_logged_slots_beyond_allocator_tail` | Logged spawns claim their exact indices and generations. Later local allocations cannot reuse those occupied slots. |
| `follower_remap_refuses_missing_world_type_before_mutation` | A referenced component must exist in the destination world. Refusal leaves values, tick, allocator, and progress unchanged. |
| `batch_round_trip_from_unaligned_bytes` | A valid batch decodes from a transport payload at an unaligned offset. |
| `frame_round_trip_preserves_payload_and_view` | The frame reader validates and decodes file bytes while preserving the payload, view, and next offset. Also runs under Miri. |
| `replay_preflight_rejects_late_component_reference` | An invalid component reference in a later record refuses before earlier records change the world or tick. |
| `replay_preflight_validates_post_apply_ticks` | Equal boundary ticks and tick overflow refuse before effects; adjacent valid ticks replay successfully. |
| `replay_plan_preserves_schema_and_fence_across_resume` | Mid-segment recovery preserves checkpoint fences and changed mappings across segments. A corrupt later frame refuses before effects. |
| `records_from_mid_segment_includes_schema_context` | A mid-segment read includes exact original schema and mutation bytes. |
| `records_from_resume_preserves_fence_context` | Prefix-only seed, durable control-only progress, and sealed versus active stale schemas. |
| `records_from_survives_reopen_and_rollover` | Exact segment runs, per-run control limits, nested directory creation, and equivalent results after reopen. |
| `records_from_refuses_unresolved_history` | Gaps, duplicates, stale mutations (active or sealed), and stale active controls cannot advance a returned prefix. |
| `records_from_rejects_corrupt_published_frames` | Bad magic, CRC mismatch, missing schema, and torn published bytes refuse. |
| `records_from_range_boundary_edges` | Sequence zero, positive limits, byte/control caps, short prefixes, tail-empty, past-tail, and maximum integer requests. |
| `records_from_excludes_unsynced_frames` | Pending mutations and controls stay outside published endpoints, including after a failed fsync. |
| `records_from_requires_retained_fence_context` | Below-floor requests and retained-floor reads refuse after deletion and reopen until prefix fence metadata exists. |
| `records_from_survives_failed_rollover_sync` | File-sync and directory-sync failures leave no orphan segment; later appends and rollover still produce identical ranges after reopen. |
| `failed_rollover_cleanup_blocks_writes` | Failed cleanup blocks appends, checkpoint writes, and retention before changes; retry resumes after cleanup becomes possible. |
| `wal_frames_round_trip_divergent_live_follower` | Private raw execution from equivalent nonempty baselines with divergent IDs/allocation history, heap values, mid-segment resume, and rollover; matching fingerprints, ticks, range end, and later allocation. |
| `raw_plan_preserves_run_remaps_and_fences` | Per-run ID changes, stale schema context, restored fence, ordered checkpoints, and empty control progress without tick changes. |
| `raw_plan_rejects_invalid_range_before_effects` | Malformed ranges, corrupt frames, duplicate schema IDs/names, gaps, stale mutations, bad component references, and invalid ticks leave the world and allocator unchanged. |
| `raw_plan_enforces_limits` | Exact/over-limit records, bytes, and controls; reject truncated claimed payloads before allocation. |
| `raw_frame_buffer_checks_lengths_and_alignment` | Decode from a byte buffer starting at offset one; reject every truncated prefix and oversized claimed payload. Runs under Miri without a world/pool fixture. |
| `raw_plan_propagates_partial_apply_failure` | A late state-dependent failure propagates the shared partial-application error without a successful completion result. |

**Planned 4.0-b tests (names reserved; not yet implemented):**

| Test | Pins |
|---|---|
| `records_from_preserves_fenced_slots` | A sealed stale slot with a durable source no-op disposition is returned and counted; a rewritable stale active suffix produces a refusal instead of progress or false caught-up. |
| `terminal_disposition_survives_source_restart` | Crashes before/after disposition fsync and before/after range publication never advertise an undurable decision or lose a durable one; source recovery and resumed ranges restore the same settled slot after WAL retention; conflicting identity, committed-operation reclassification and missing/corrupt required journal state refuse. |
| `follower_ingest_processes_stale_schema_frame` | Stale schema still builds the correct remap for later current mutations; neither progress nor fence regresses. |
| `follower_ingest_drops_stale_mutation_frame` | A fenced mutation performs no world, tick, allocator, or column-mark effects; its terminal slot is accounted for. |
| `follower_ingest_checkpoint_raises_fence` | A higher-view checkpoint fences later stale mutations; schema/checkpoints and flush_seq never advance the sequence or claim follower baseline coverage. |
| `follower_ingest_fenced_slot_advances_prefix` | One stream: stale schema, higher-view checkpoint, expected stale mutation with an explicit terminal disposition, next current mutation. Boundary 40→41→42; all-stale batch advances; retransmission is a no-op. |
| `follower_ingest_refuses_unresolved_fenced_slot` | An older view, sealed bytes or a missing record without an authoritative terminal disposition cannot advance logical progress; refuse and repair/rejoin. |
| `follower_ingest_preflight_rejects_invalid_range` | Missing/wrong schema, unmapped ID, bad CRC, inconsistent bounds and a real gap/reorder refuse before world mutation; no fabricated skip. |
| `follower_ingest_retry_and_failure_preserve_prefix` | Duplicate spawn bytes never reapply; journal/fsync failure advertises nothing; mid-range apply/tick failure preserves only the settled boundary, poisons reads and emits no successful progress. |
| `follower_ingest_restart_reconstructs_progress` | Crash before fsync, after durable receipt but before apply, and after apply but before next fetch; restore baseline plus journal including fenced slots, fence and IDs; never restore progress ahead of world. |
| `read_at_after_fenced_slot_uses_settled_prefix` | After slot 40 is authoritatively resolved as a no-op, read_at(40) observes the no-op projection and read_at(41) refuses; after current slot 41 applies, read_at(41) succeeds; poison always refuses. |
| `pull_pump_reports_only_ingested_progress` | Fetch runs on follower; stalled/lost response cannot advance registry or touch leader commit path; next request, including empty catch-up, reports only successful ingest. |
| `retention_respects_followers_and_recovery_floor` | Two follower positions, lagging/disconnected member, no followers and absent baseline; cutoff cannot cross leader LSM replay floor or use checkpoint flush_seq in its place. |
| `retention_rejects_old_session_progress` | Old-session/duplicate/delayed requests cannot inflate new-session progress; rejoin seeds only a verified baseline; leader restart blocks deletion until conservative registry restoration. |
| `retention_preserves_resume_fence_and_pinned_reads` | Deletion cannot race an in-flight copy or erase required fence context; retained-floor resume uses the right schema/fence, older requests require rejoin. |
| `pull_fetch_chaos_converges_pinned_seeds` | Loss/duplicates/delay converge after recovery of the link; injected invalid frame ordering fails closed; retries do not falsely report completion. |

Existing Loom tests `loom_follower_advance_vs_read_at` and `loom_follower_concurrent_advance` cover the decoded follower path. Pending, listed so the gaps stay visible: Loom coverage for the new raw-ingest state, `fuzz_wal_replay` corpus entries with view-stamped frames (stale views included — its mode 0 already feeds raw bytes at replay), and failpoint hooks in the outbound pump for the 4.1 simulator.
