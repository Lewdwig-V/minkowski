# RSM Substrate Design (Stage 4.0)

Date: 2026-08-30. Updated: 2026-09-05. Status: 4.0-a delivered (PR #251, #252).

Phase 4.0-b starts with application prerequisites and a private recovery plan in section 7.7. Sections 1–5 describe delivered behavior except the explicitly marked 4.0-b amendments. Sections 6–8 distinguish those changes from the remaining planned work. Proposed APIs and reserved test names are not implementation claims.

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
| `BatchRef.from_seq`, `next_seq`, `limit` (4.0-b) | mutation slots are exactly `[from_seq, next_seq)`; `limit > 0` caps slot count including fenced slots. Schema/checkpoint context consumes no slots | planned `records_from_range_boundary_edges` |
| `durable_tail` (4.0-b) | exclusive post-fsync mutation boundary; accompanying durable byte bounds also gate control frames (§7.2) | planned `records_from_excludes_unsynced_frames` |
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
| `records_from` (4.0-b raw range reader) | enforced: include each run's schema, even stale; `records_from_mid_segment_includes_schema_context` | exempt from filtering: preserve every returned slot; stale bytes require a durable source terminal disposition, otherwise refuse; never publish a rewritable stale active tail; `records_from_preserves_fenced_slots` | enforced: preserve durable control order and prefix fence context; `records_from_resume_preserves_fence_context` |
| `Follower::ingest_frames` (4.0-b, shipped ranges) | enforced: validate/build the run's remap even when stale; raise fence with `max`, never lower it; `follower_ingest_processes_stale_schema_frame` | enforced: reject stale effects before `apply_record`; settle an expected stale slot only with a terminal disposition (§4), otherwise refuse; `follower_ingest_drops_stale_mutation_frame`, `follower_ingest_fenced_slot_advances_prefix`, `follower_ingest_refuses_unresolved_fenced_slot` | enforced: raise fence with `max`; ignore `flush_seq` for world/progress/baseline coverage; `follower_ingest_checkpoint_raises_fence` |

Fence maintenance:

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

Verified in `wal.rs::Wal::append`: `next_seq` advances before `sync_all`. Publish `durable_tail` only after successful fsync, using a consistent read snapshot. Also capture durable frame/byte endpoints: schema and checkpoint frames have no mutation sequence, and `acknowledge_flush` currently does not fsync. The scalar tail alone cannot authorize shipping a later unsynced checkpoint or using its view to seed a range. Publish control-only progress only after its bytes are durable.

For `records_from(seq, limit)`, `limit > 0` caps mutation slots, including fenced slots; use checked bounds. Requests at the published tail return no mutation slots; requests beyond it are typed refusals; a large limit yields a shorter range. A sequence is not durable merely because it was allocated.

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

`Follower::ingest_frames(&mut self, batch, world, codecs)` is the raw ingestion chokepoint. After preflight, persist the complete replayable range and its schema/fence context in a follower ingest journal and fsync it before application or advertising progress. Repeated ranges remain positionally deduplicated, including during recovery. This is new raw-frame persistence work: `Wal::append` currently serializes changesets and cannot be assumed to append received frames. Treat a journal as replayable range envelopes; blindly concatenating response buffers into a local leader WAL is not the format contract.

Apply and publish per the shared §4 state machine. Only its successful return may supply the next pull position or a future applied-progress Ack for finalized-range push (§7.5); a failed journal write/apply produces no successful progress result. On restart, restore a consistent world baseline, its fence and stream identity, then replay the journal through the same interpreter/position rules (without re-journaling) before reconstructing progress. Never restore a saved number onto an older world. `replay_from` alone and `recover_world` alone do not reconstruct follower progress today. Volatile-only acknowledgment is outside this 4.0-b contract.

**First semantic gate, before wire code:** `wal_frames_round_trip_divergent_live_follower` starts from equivalent leader/follower baseline state, with pre-existing entities and divergent component registration/allocation history. Ship exact ranged frame bytes, including a mid-segment start and rollover; assert fingerprint/tick equality and the exclusive settled boundary. Merely decoding an entire segment into a fresh world does not prove this contract. Failure identifies required remap or allocator-adoption work; do not assert that existing recovery already guarantees it.

### 7.4 Pull contract and failure behavior

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
An occupied slot or an older generation requires a typed refusal.
Process entity allocation in mutation order so a despawn and subsequent spawn can reuse one slot within a record.
This prerequisite test does not replace `wal_frames_round_trip_divergent_live_follower`, which also requires exact raw ranges, schema runs, and durable bounds.

The application prerequisites use this invariant matrix:

| Consumer | Component identifiers | Entity allocation | Tick and failure behavior |
|---|---|---|---|
| `apply_record` through decoded follower or WAL recovery | Resolve source → codec → destination type. `follower_round_trip_divergent_live_world` | Claim the logged slot in mutation order. `follower_replay_reuses_logged_entity_slot_in_mutation_order` | Keep the existing per-record tick and poison rules. `follower_tick_regression_poisons` |
| Raw `ValidatedRange` execution, planned | Resolve all runs before effects. `follower_ingest_preflight_rejects_invalid_range` | Use the same `apply_record` path. `wal_frames_round_trip_divergent_live_follower` | Journal before effects. `follower_ingest_retry_and_failure_preserve_prefix` |
| Authorized terminal no-op, planned | Validate its run schema. `follower_ingest_processes_stale_schema_frame` | Exempt: no entity allocation occurs. `follower_ingest_drops_stale_mutation_frame` | Advance only the settled prefix. `follower_ingest_fenced_slot_advances_prefix` |

Local WAL recovery now uses the private `ValidatedRange` raw-frame plan.
It retains selected original frames and resolves every referenced component before execution.
The plan exclusively borrows its destination world until execution finishes.
Execution decodes the retained mutation frames again and calls `apply_record`.
Component payload decoding and state-dependent application can still fail after earlier records have applied.
Local recovery keeps its existing stale-frame and torn-tail rules.
This caller does not establish contiguous finalized ranges or authorize follower progress.
The durable range reader, terminal dispositions, and ingest journal remain required before exposing raw follower ingestion.

| Recovery plan input | Before effects | Execution |
|---|---|---|
| Schema | Validate codec layout and build destination mappings, including stale schemas. `replay_plan_preserves_schema_and_fence_across_resume` | Exempt: schema does not mutate the world. |
| Eligible mutation | Validate every referenced component across the selected range. `replay_preflight_rejects_late_component_reference` | Use `apply_record` for ticks, entity allocation, and values. `follower_round_trip_divergent_component_ids` |
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
| `replay_plan_preserves_schema_and_fence_across_resume` | Mid-segment recovery preserves checkpoint fences and changed mappings across segments. A corrupt later frame refuses before effects. |

**Planned 4.0-b tests (names reserved; not yet implemented):**

| Test | Pins |
|---|---|
| `records_from_mid_segment_includes_schema_context` | Mid-segment fetch and reconnect carry the exact active schema; rollover starts a new remap scope; schema consumes no slots. |
| `records_from_resume_preserves_fence_context` | A checkpoint or schema before the resume point, including an earlier segment, seeds the first mutation's fence; a later view cannot retroactively fence the prefix. |
| `records_from_preserves_fenced_slots` | A sealed stale slot with a durable source no-op disposition is returned and counted; a rewritable stale active suffix produces a refusal instead of progress or false caught-up. |
| `terminal_disposition_survives_source_restart` | Crashes before/after disposition fsync and before/after range publication never advertise an undurable decision or lose a durable one; source recovery and resumed ranges restore the same settled slot after WAL retention; conflicting identity, committed-operation reclassification and missing/corrupt required journal state refuse. |
| `records_from_range_boundary_edges` | 0, baseline floor, limit=0 refusal, limit including stale slots, short reads, tail-empty, past-tail refusal, overflow and retained-floor rejoin. |
| `records_from_excludes_unsynced_frames` | Pause before fsync: neither mutations nor later schema/checkpoint bytes or their views escape the durable snapshot. |
| `wal_frames_round_trip_divergent_live_follower` | Before wire code: equivalent nonempty baseline, divergent IDs/allocation history, POD+heap components, mid-segment/rollover raw bytes; fingerprint, tick, settled boundary. |
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
