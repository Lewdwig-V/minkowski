//! Transport-agnostic replication primitives.
//!
//! [`ReplicationBatch`] is a self-describing mutation payload that can be
//! serialized to bytes (`to_bytes`) and deserialized (`from_bytes`) for
//! transport over any medium — network, channels, files, shared memory.
//! [`apply_batch`] consumes a batch and applies it to a target [`World`].
//!
//! How you produce and transport batches is up to you. For local-filesystem
//! scenarios, [`WalCursor`](crate::WalCursor) reads batches directly from
//! WAL segment files.

#[cfg(loom)]
use crate::sync::{Arc, AtomicBool, AtomicU64};
#[cfg(not(loom))]
use std::sync::Arc;
use std::sync::atomic::Ordering;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicU64};

use minkowski::World;

use crate::record::ReplicationBatch;
use crate::wal::{WalError, apply_record, build_apply_remap};
use minkowski_lsm::codec::{CodecError, CodecRegistry};

/// Errors from transport-agnostic replication operations.
///
/// Deliberately independent of [`WalError`](crate::WalError) — a replica
/// server that only deserializes and applies batches should not need to
/// know about WAL file I/O.
#[derive(Debug, thiserror::Error)]
pub enum ReplicationError {
    #[error("replication format error: {0}")]
    Format(String),
    #[error("replication codec error: {0}")]
    Codec(#[from] CodecError),
}

impl From<WalError> for ReplicationError {
    fn from(e: WalError) -> Self {
        match e {
            WalError::Codec(c) => ReplicationError::Codec(c),
            other => ReplicationError::Format(other.to_string()),
        }
    }
}

impl ReplicationBatch {
    /// Serialize to bytes via rkyv.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReplicationError> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map(rkyv::util::AlignedVec::into_vec)
            .map_err(|e| ReplicationError::Format(e.to_string()))
    }

    /// Deserialize from bytes via rkyv.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplicationError> {
        // Transport buffers do not guarantee the archive's alignment.
        let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
        aligned.extend_from_slice(bytes);
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(&aligned)
            .map_err(|e| ReplicationError::Format(e.to_string()))
    }
}

/// Apply a replication batch record-by-record in log order.
///
/// Stateless primitive: no position tracking, no poison, no retry. Each
/// record replays at its own commit-boundary tick (INV-1); a record tick
/// below the world's current tick errors with
/// [`WalError::TickRegression`]. For replica ingestion use
/// [`Follower::advance`], which adds position-based idempotency, gap
/// rejection, and poison-on-failure.
pub fn apply_batch(
    batch: &ReplicationBatch,
    world: &mut World,
    codecs: &CodecRegistry,
) -> Result<Option<u64>, ReplicationError> {
    let remap = if batch.schema.components.is_empty() {
        None
    } else {
        Some(build_apply_remap(
            Some(&batch.schema.components),
            world,
            codecs,
        )?)
    };

    let mut last_seq = None;
    for record in &batch.records {
        apply_record(record, world, codecs, remap.as_ref(), None)?;
        last_seq = Some(record.seq);
    }

    Ok(last_seq)
}

/// Follower-side apply state for the stage 4.0 substrate.
///
/// Owns the replica's applied-sequence boundary and poison state. The world
/// itself is passed per call — a `Follower` holds no borrow (same split-phase
/// rule as `Tx`).
///
/// Failure semantics (spec §2.7): idempotency is by position — records at or
/// below `applied_seq` are skipped, never re-applied. Any mid-batch apply
/// failure poisons the follower: `advance` and `read_at` return
/// [`FollowerError::Poisoned`] forever after, and the replica must rejoin via
/// state transfer (`recover_world` from a peer). No retry, no rollback.
pub struct Follower {
    /// Next sequence the follower expects: all records below this are
    /// applied. 0 = nothing applied (WAL sequences start at 0).
    high_water: AtomicU64,
    poisoned: AtomicBool,
}

#[derive(Debug, thiserror::Error)]
pub enum FollowerError {
    #[error("follower poisoned by an earlier mid-batch failure; rejoin via state transfer")]
    Poisoned,
    #[error("read at seq {requested} is ahead of applied_seq {applied_seq}")]
    Stale { requested: u64, applied_seq: u64 },
    #[error(
        "tick regression at seq {seq}: record tick {record_tick} is below world tick {world_tick}"
    )]
    TickRegression {
        seq: u64,
        record_tick: u64,
        world_tick: u64,
    },
    #[error("gap in log: expected seq {expected}, got {got}")]
    Gap { expected: u64, got: u64 },
    #[error("apply failed: {0}")]
    Apply(#[from] WalError),
}

impl Default for Follower {
    fn default() -> Self {
        Self::new()
    }
}

impl Follower {
    pub fn new() -> Self {
        Self::with_baseline(0)
    }

    /// A follower resuming onto a world bootstrapped from a storage baseline
    /// (LSM sorted runs via `recover_world`). `baseline_seq` is the first
    /// sequence NOT covered by the baseline: records below it are already in
    /// the restored world, so the first expected log record is
    /// `baseline_seq`. For a manifest whose runs cover through `seq_hi`,
    /// pass the replay floor the recovery used.
    pub fn with_baseline(baseline_seq: u64) -> Self {
        Self {
            high_water: AtomicU64::new(baseline_seq),
            poisoned: AtomicBool::new(false),
        }
    }

    /// The next sequence this follower expects; all records below it are
    /// applied. 0 means nothing applied yet. `read_at(seq)` is valid for
    /// `seq < applied_seq()`.
    pub fn applied_seq(&self) -> u64 {
        self.high_water.load(Ordering::Acquire)
    }

    /// True once a mid-batch failure has poisoned this follower.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    pub(crate) fn poison(&self) {
        self.poisoned.store(true, Ordering::SeqCst);
    }

    /// Apply a transport batch in log order.
    ///
    /// Per record (INV-1): a record whose `tick_after` is below the world's
    /// current tick is a foreign or corrupt log — poison. Otherwise set the
    /// world tick to the record's commit-boundary tick, then apply, so the
    /// record's column marks carry the leader's ticks.
    ///
    /// Returns the last sequence applied (== `applied_seq` after the call).
    pub fn advance(
        &self,
        batch: &ReplicationBatch,
        world: &mut World,
        codecs: &CodecRegistry,
    ) -> Result<u64, FollowerError> {
        if self.is_poisoned() {
            return Err(FollowerError::Poisoned);
        }
        // Schema-scoped remap, built once before any mutation (§2.7
        // pre-flight: id-translation failures cost zero mutation).
        let remap = if batch.schema.components.is_empty() {
            None
        } else {
            match build_apply_remap(Some(&batch.schema.components), world, codecs) {
                Ok(r) => Some(Arc::new(r)),
                Err(e) => {
                    self.poison();
                    return Err(FollowerError::Apply(WalError::Format(e.to_string())));
                }
            }
        };
        let remap_ref = remap.as_deref();

        let mut last = self.applied_seq();
        for record in &batch.records {
            last = self.apply_next(record.seq, || {
                apply_record(record, world, codecs, remap_ref, None)
            })?;
        }
        Ok(last)
    }

    /// Shared positional/poison boundary for decoded and journaled ingestion.
    pub(crate) fn apply_next(
        &self,
        seq: u64,
        apply: impl FnOnce() -> Result<(), WalError>,
    ) -> Result<u64, FollowerError> {
        if self.is_poisoned() {
            return Err(FollowerError::Poisoned);
        }
        let last = self.applied_seq();
        if seq < last {
            return Ok(last);
        }
        if seq > last {
            self.poison();
            return Err(FollowerError::Gap {
                expected: last,
                got: seq,
            });
        }
        let Some(next) = seq.checked_add(1) else {
            self.poison();
            return Err(WalError::Format("follower sequence overflow".into()).into());
        };
        if let Err(error) = apply() {
            self.poison();
            return Err(match error {
                WalError::TickRegression {
                    seq,
                    record_tick,
                    world_tick,
                } => FollowerError::TickRegression {
                    seq,
                    record_tick,
                    world_tick,
                },
                other => FollowerError::Apply(other),
            });
        }
        self.high_water.store(next, Ordering::Release);
        Ok(next)
    }

    /// Bounded-staleness read at a logged prefix (INV-4).
    ///
    /// Runs `f` against the world only if `applied_seq >= seq`. The closure
    /// gets a shared reference — use `query_raw`-style reads; ticks do not
    /// advance. This is the only surface the stage 3.75 query language may
    /// hook for replicated reads.
    pub fn read_at<R>(
        &self,
        seq: u64,
        world: &World,
        f: impl FnOnce(&World) -> R,
    ) -> Result<R, FollowerError> {
        if self.is_poisoned() {
            return Err(FollowerError::Poisoned);
        }
        // Valid reads are through seq < high_water (the applied prefix).
        let applied = self.applied_seq();
        if seq >= applied {
            return Err(FollowerError::Stale {
                requested: seq,
                applied_seq: applied,
            });
        }
        Ok(f(world))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{ComponentSchema, SerializedMutation, WalRecord, WalSchema};
    use crate::wal::{Wal, WalConfig, WalCursor, WalError};
    use minkowski::{EnumChangeSet, World};
    use minkowski_lsm::codec::CodecRegistry;

    #[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq, Debug)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq, Debug)]
    struct Health(u32);

    fn test_schema() -> WalSchema {
        WalSchema {
            components: vec![ComponentSchema {
                id: 0,
                name: "pos".into(),
                size: 8,
                align: 4,
            }],
        }
    }

    /// Helper: create a WAL with N spawn mutations and return the dir + codecs.
    fn create_test_wal(dir: &std::path::Path, n: usize) -> (std::path::PathBuf, CodecRegistry) {
        let wal_dir = dir.join("test.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
        for i in 0..n {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        (wal_dir, codecs)
    }

    // ── ReplicationBatch tests ──────────────────────────────────────

    #[test]
    fn batch_round_trip() {
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![
                WalRecord {
                    tick_after: 0,
                    seq: 0,
                    mutations: vec![SerializedMutation::Despawn { entity: 1 }],
                },
                WalRecord {
                    tick_after: 0,
                    seq: 1,
                    mutations: vec![],
                },
            ],
        };

        let bytes = batch.to_bytes().unwrap();
        let restored = ReplicationBatch::from_bytes(&bytes).unwrap();

        assert_eq!(restored.records.len(), 2);
        assert_eq!(restored.records[0].seq, 0);
        assert_eq!(restored.records[1].seq, 1);
        assert_eq!(restored.schema.components.len(), 1);
        assert_eq!(restored.schema.components[0].name, "pos");
    }

    #[test]
    fn batch_round_trip_from_unaligned_bytes() {
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                tick_after: 7,
                seq: 3,
                mutations: vec![SerializedMutation::Despawn { entity: 42 }],
            }],
        };
        let bytes = batch.to_bytes().unwrap();
        // A transport envelope can place its payload at an arbitrary offset.
        let mut envelope = rkyv::util::AlignedVec::<16>::new();
        envelope.push(0);
        envelope.extend_from_slice(&bytes);
        let restored = ReplicationBatch::from_bytes(&envelope[1..]).unwrap();
        assert_eq!(restored.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn empty_batch_round_trip() {
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![],
        };

        let bytes = batch.to_bytes().unwrap();
        let restored = ReplicationBatch::from_bytes(&bytes).unwrap();
        assert!(restored.records.is_empty());
    }

    // ── WalCursor tests ────────────────────────────────────────────

    #[test]
    fn cursor_reads_from_seq_zero() {
        let dir = tempfile::tempdir().unwrap();
        let (wal_path, _codecs) = create_test_wal(dir.path(), 3);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        assert_eq!(batch.records.len(), 3);
        assert_eq!(batch.records[0].seq, 0);
        assert_eq!(batch.records[1].seq, 1);
        assert_eq!(batch.records[2].seq, 2);
        // Cursor advanced past the last record: nothing left to read.
        assert!(cursor.next_batch(100).unwrap().records.is_empty());

        assert_eq!(batch.schema.components.len(), 1);
        assert_eq!(batch.schema.components[0].name, "pos");
    }

    #[test]
    fn cursor_reads_from_mid_seq() {
        let dir = tempfile::tempdir().unwrap();
        let (wal_path, _codecs) = create_test_wal(dir.path(), 5);

        let mut cursor = WalCursor::open(&wal_path, 3).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        assert_eq!(batch.records.len(), 2);
        assert_eq!(batch.records[0].seq, 3);
        assert_eq!(batch.records[1].seq, 4);
        // Cursor at end: nothing left to read.
        assert!(cursor.next_batch(100).unwrap().records.is_empty());
    }

    #[test]
    fn cursor_at_end_returns_empty_batch() {
        let dir = tempfile::tempdir().unwrap();
        let (wal_path, _codecs) = create_test_wal(dir.path(), 2);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch1 = cursor.next_batch(100).unwrap();
        assert_eq!(batch1.records.len(), 2);

        let batch2 = cursor.next_batch(100).unwrap();
        assert!(batch2.records.is_empty());
    }

    #[test]
    fn cursor_respects_batch_limit() {
        let dir = tempfile::tempdir().unwrap();
        let (wal_path, _codecs) = create_test_wal(dir.path(), 5);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();

        let batch1 = cursor.next_batch(2).unwrap();
        assert_eq!(batch1.records.len(), 2);
        assert_eq!(batch1.records[0].seq, 0);
        assert_eq!(batch1.records[1].seq, 1);

        let batch2 = cursor.next_batch(2).unwrap();
        assert_eq!(batch2.records.len(), 2);
        assert_eq!(batch2.records[0].seq, 2);
        assert_eq!(batch2.records[1].seq, 3);

        let batch3 = cursor.next_batch(2).unwrap();
        assert_eq!(batch3.records.len(), 1);
        assert_eq!(batch3.records[0].seq, 4);
    }

    #[test]
    fn cursor_behind_error_display() {
        let err = WalError::CursorBehind {
            requested: 0,
            oldest: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("cursor behind"));
        assert!(msg.contains('0'));
        assert!(msg.contains('5'));
    }

    // ── apply_batch tests ──────────────────────────────────────────

    #[test]
    fn apply_batch_spawns_entities() {
        let dir = tempfile::tempdir().unwrap();
        let (wal_path, _codecs) = create_test_wal(dir.path(), 3);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        let mut replica = World::new();
        let mut replica_codecs = CodecRegistry::new();
        replica_codecs
            .register_as::<Pos>("pos", &mut replica)
            .unwrap();

        let last_seq = apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

        assert_eq!(last_seq, Some(2));
        assert_eq!(replica.query::<(&Pos,)>().count(), 3);
    }

    /// Heap-dense (`String`) component survives a replication batch round trip:
    /// WAL append → `WalCursor` batch → `apply_batch` into a fresh replica. The
    /// mutation is rkyv-encoded through the codec (resolved by `TypeId`), so the
    /// variable-length heap field is carried verbatim across the wire format.
    /// Asserts the recovered `String` value, not just a POD field.
    #[test]
    fn apply_batch_replicates_heap_component() {
        #[derive(Clone, PartialEq, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
        struct Note {
            text: String,
        }

        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("heap_repl.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Note>("note", &mut world).unwrap();

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e,
            (Note {
                text: "replicated".to_owned(),
            },),
        )
        .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        drop(wal);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        let mut replica = World::new();
        let mut replica_codecs = CodecRegistry::new();
        replica_codecs
            .register_as::<Note>("note", &mut replica)
            .unwrap();

        apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

        let notes: Vec<String> = replica
            .query::<(&Note,)>()
            .map(|n| n.0.text.clone())
            .collect();
        assert_eq!(
            notes,
            vec!["replicated".to_owned()],
            "heap String component should survive replication round trip value-exact"
        );
    }

    #[test]
    fn apply_batch_insert_remove() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Health>("health", &mut world).unwrap();

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        let mut cs2 = EnumChangeSet::new();
        cs2.insert::<Health>(&mut world, e, Health(100));
        cs2.remove::<Pos>(&mut world, e);
        wal.append(&cs2, &codecs, world.current_tick()).unwrap();
        cs2.apply(&mut world).unwrap();

        drop(wal);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        let mut replica = World::new();
        let mut replica_codecs = CodecRegistry::new();
        replica_codecs
            .register_as::<Pos>("pos", &mut replica)
            .unwrap();
        replica_codecs
            .register_as::<Health>("health", &mut replica)
            .unwrap();

        apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

        assert_eq!(replica.query::<(&Health,)>().count(), 1);
        assert_eq!(replica.query::<(&Pos,)>().count(), 0);
        let h = replica.query::<(&Health,)>().next().unwrap().0;
        assert_eq!(h.0, 100);
    }

    #[test]
    fn apply_batch_cross_process_remap() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("cross.wal");

        // Source: Pos=0, Health=1
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Health>("health", &mut world).unwrap();

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 }, Health(50)))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        drop(wal);

        // Replica: Health=0, Pos=1 (opposite order)
        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        let mut replica = World::new();
        let mut replica_codecs = CodecRegistry::new();
        replica_codecs
            .register_as::<Health>("health", &mut replica)
            .unwrap();
        replica_codecs
            .register_as::<Pos>("pos", &mut replica)
            .unwrap();

        apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

        let positions: Vec<(f32, f32)> =
            replica.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(1.0, 2.0)]);

        let health: Vec<u32> = replica.query::<(&Health,)>().map(|h| h.0.0).collect();
        assert_eq!(health, vec![50]);
    }

    #[test]
    fn apply_batch_preserves_transaction_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("boundaries.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },))
            .unwrap();
        wal.append(&cs, &codecs, world.current_tick()).unwrap();
        cs.apply(&mut world).unwrap();

        let mut cs2 = EnumChangeSet::new();
        cs2.record_despawn(e);
        wal.append(&cs2, &codecs, world.current_tick()).unwrap();
        cs2.apply(&mut world).unwrap();

        drop(wal);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();
        assert_eq!(batch.records.len(), 2);

        let mut replica = World::new();
        let mut replica_codecs = CodecRegistry::new();
        replica_codecs
            .register_as::<Pos>("pos", &mut replica)
            .unwrap();

        apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

        assert_eq!(replica.query::<(&Pos,)>().count(), 0);
    }

    #[test]
    fn apply_empty_batch() {
        let batch = ReplicationBatch {
            schema: WalSchema { components: vec![] },
            records: vec![],
        };

        let mut world = World::new();
        let codecs = CodecRegistry::new();

        let last_seq = apply_batch(&batch, &mut world, &codecs).unwrap();
        assert_eq!(last_seq, None);
    }

    #[test]
    fn cursor_skips_checkpoint_entries() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

        // Write 3 records, checkpoint, then 2 more
        for i in 0..3 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        wal.acknowledge_flush(wal.next_seq()).unwrap();
        for i in 3..5 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }

        drop(wal);

        let mut cursor = WalCursor::open(&wal_dir, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();
        // Should see all 5 mutation records, no checkpoint in batch
        assert_eq!(batch.records.len(), 5);
        assert_eq!(batch.records[0].seq, 0);
        assert_eq!(batch.records[4].seq, 4);
    }

    #[test]
    fn cursor_reads_across_segment_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        // Small segments to force rollover
        let config = WalConfig {
            max_segment_bytes: 128,
            max_bytes_between_checkpoints: None,
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        for i in 0..20 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        assert!(wal.stats().segment_count > 1);
        drop(wal);

        let mut cursor = WalCursor::open(&wal_dir, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();
        assert_eq!(batch.records.len(), 20);
        assert_eq!(batch.records[0].seq, 0);
        assert_eq!(batch.records[19].seq, 19);
        // Cursor at end: nothing left to read.
        assert!(cursor.next_batch(100).unwrap().records.is_empty());
    }

    #[test]
    fn cursor_behind_after_segment_deletion() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        let config = WalConfig {
            max_segment_bytes: 128,
            max_bytes_between_checkpoints: None,
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        for i in 0..20 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        assert!(wal.stats().segment_count > 2);
        wal.delete_segments_before(15).unwrap();
        drop(wal);

        let result = WalCursor::open(&wal_dir, 0);
        assert!(
            matches!(result, Err(WalError::CursorBehind { .. })),
            "should return CursorBehind when requesting deleted segment"
        );
    }

    // ── Error path tests ─────────────────────────────────────────

    #[test]
    fn from_bytes_corrupt_returns_error() {
        let result = ReplicationBatch::from_bytes(&[0xFF; 32]);
        assert!(matches!(result, Err(ReplicationError::Format(_))));
    }

    #[test]
    fn apply_batch_unknown_component_returns_error() {
        let batch = ReplicationBatch {
            schema: WalSchema {
                components: vec![ComponentSchema {
                    id: 0,
                    name: "nonexistent_component".into(),
                    size: 8,
                    align: 4,
                }],
            },
            records: vec![WalRecord {
                tick_after: 0,
                seq: 0,
                mutations: vec![],
            }],
        };

        let mut world = World::new();
        let codecs = CodecRegistry::new();

        let result = apply_batch(&batch, &mut world, &codecs);
        assert!(result.is_err());
    }

    // ── Integration test ───────────────────────────────────────────

    #[test]
    fn full_replication_flow() {
        use crate::recover::recover_world;
        use minkowski_lsm::manifest_log::ManifestLog;
        use minkowski_lsm::manifest_ops::flush_and_record;
        use minkowski_lsm::types::{SeqNo, SeqRange};

        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("source.wal");
        let lsm_dir = dir.path().join("lsm");
        let manifest_log = lsm_dir.join("manifest.log");
        std::fs::create_dir_all(&lsm_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();

        for i in 0..5 {
            world.spawn((Pos {
                x: i as f32,
                y: 0.0,
            },));
        }

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let flush_seq = wal.next_seq();
        let (mut manifest, mut log) = ManifestLog::recover::<4>(&manifest_log).unwrap();
        flush_and_record(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(flush_seq)).unwrap(),
            &mut manifest,
            &mut log,
            &lsm_dir,
            &codecs,
        )
        .unwrap()
        .expect("flush");

        for i in 5..8 {
            let e = world.alloc_entity();
            let mut cs = EnumChangeSet::new();
            cs.spawn_bundle(
                &mut world,
                e,
                (Pos {
                    x: i as f32,
                    y: 0.0,
                },),
            )
            .unwrap();
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }

        drop(wal);

        let mut replica_codecs = CodecRegistry::new();
        let mut tmp = World::new();
        replica_codecs.register_as::<Pos>("pos", &mut tmp).unwrap();

        let mut wal_replica = Wal::open(&wal_path, &replica_codecs, WalConfig::default()).unwrap();
        let mut replica =
            recover_world(&lsm_dir, &manifest_log, &mut wal_replica, &replica_codecs).unwrap();
        assert_eq!(replica.query::<(&Pos,)>().count(), 8);

        let mut cursor = WalCursor::open(&wal_path, flush_seq).unwrap();
        let batch = cursor.next_batch(100).unwrap();
        assert!(batch.records.is_empty() || batch.records.len() <= 3);
    }

    // ── Stage 4.0 substrate: Follower + convergence ─────────────────

    #[derive(Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq, Debug)]
    struct Name(String);

    /// Leader applies `n` mixed transactions (spawn with POD + heap
    /// components, insert, remove, despawn). Each commit appends to the WAL
    /// with the pre-apply world tick, then applies — the Durable commit
    /// order. Returns (wal_dir, leader world, codecs, last_seq).
    fn run_leader(dir: &std::path::Path, n: usize) -> (std::path::PathBuf, World, CodecRegistry) {
        use crate::wal::{Wal, WalConfig};

        let wal_dir = dir.join("leader.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world).unwrap();
        codecs.register_as::<Name>("name", &mut world).unwrap();
        codecs.register_as::<Health>("health", &mut world).unwrap();

        let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();
        let fastrand = std::cell::Cell::new(12345u64);
        let next = move || {
            // xorshift: deterministic pseudo-random without external state
            let mut x = fastrand.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            fastrand.set(x);
            x
        };

        let mut live: Vec<u64> = Vec::new();
        for i in 0..n {
            let roll = next() % 4;
            let mut cs = EnumChangeSet::new();
            if roll == 0 || live.is_empty() {
                // spawn
                let e = world.alloc_entity();
                let pos = Pos {
                    x: i as f32,
                    y: (i % 7) as f32,
                };
                if i % 2 == 0 {
                    cs.spawn_bundle(&mut world, e, (pos, Name(format!("entity-{i}"))))
                        .unwrap();
                } else {
                    cs.spawn_bundle(&mut world, e, (pos,)).unwrap();
                }
                live.push(e.to_bits());
            } else if roll == 1 {
                // insert Health on a live entity
                let idx = (next() as usize) % live.len();
                let e = minkowski::Entity::from_bits(live[idx]);
                cs.insert(&mut world, e, Health(i as u32));
            } else if roll == 2 {
                // remove Health (may be absent — try_insert style: skip if
                // the entity does not have it; remove of an absent component
                // is a no-op record we avoid by only removing when inserted)
                let idx = (next() as usize) % live.len();
                let e = minkowski::Entity::from_bits(live[idx]);
                cs.remove::<Health>(&mut world, e);
            } else {
                // despawn one live entity
                let idx = (next() as usize) % live.len();
                let bits = live.swap_remove(idx);
                let e = minkowski::Entity::from_bits(bits);
                cs.record_despawn(e);
            }
            // Durable commit order: WAL write (pre-apply tick) THEN apply.
            wal.append(&cs, &codecs, world.current_tick()).unwrap();
            cs.apply(&mut world).unwrap();
        }
        (wal_dir, world, codecs)
    }

    // These fixtures need only a few rows, including under Miri.
    fn replay_test_world() -> World {
        World::builder().memory_budget(1024 * 1024).build().unwrap()
    }

    #[test]
    fn follower_round_trip_divergent_component_ids() {
        #[derive(
            Clone, Copy, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
        )]
        struct Left(u64);
        #[derive(
            Clone, Copy, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
        )]
        struct Right(u64);

        let dir = tempfile::tempdir().unwrap();
        let mut leader = replay_test_world();
        leader.register_component::<bool>();
        let mut source_codecs = CodecRegistry::new();
        source_codecs.register::<Left>(&mut leader).unwrap();
        source_codecs.register::<Right>(&mut leader).unwrap();
        let entity = leader.spawn((Left(1), Right(2)));

        // The codec registry and destination each have their own ID space.
        // Equal-sized POD types make a wrong mapping observable without
        // interpreting a heap value as another type in the failing test.
        let mut codec_world = replay_test_world();
        let mut target_codecs = CodecRegistry::new();
        target_codecs.register::<Right>(&mut codec_world).unwrap();
        target_codecs.register::<Left>(&mut codec_world).unwrap();
        let mut replica = replay_test_world();
        assert_eq!(replica.spawn((Left(1), Right(2))), entity);
        assert_ne!(
            leader.component_id::<Left>(),
            replica.component_id::<Left>()
        );
        assert_ne!(
            codec_world.component_id::<Left>(),
            replica.component_id::<Left>()
        );

        let mut wal = Wal::create(dir.path(), &source_codecs, WalConfig::default()).unwrap();
        let mut changes = EnumChangeSet::new();
        changes.insert(&mut leader, entity, Left(10));
        changes.insert(&mut leader, entity, Right(20));
        wal.append(&changes, &source_codecs, leader.current_tick())
            .unwrap();
        changes.apply(&mut leader).unwrap();

        let mut cursor = WalCursor::open(dir.path(), 0).unwrap();
        let batch = cursor.next_batch(1).unwrap();
        let decoded = ReplicationBatch::from_bytes(&batch.to_bytes().unwrap()).unwrap();
        let follower = Follower::new();
        assert_eq!(
            follower
                .advance(&decoded, &mut replica, &target_codecs)
                .unwrap(),
            1
        );
        assert_eq!(replica.get::<Left>(entity), Some(&Left(10)));
        assert_eq!(replica.get::<Right>(entity), Some(&Right(20)));
        assert_eq!(replica.current_tick(), leader.current_tick());

        let mut recovered = replay_test_world();
        assert_eq!(recovered.spawn((Left(1), Right(2))), entity);
        wal.replay_from(0, &mut recovered, &target_codecs).unwrap();
        assert_eq!(recovered.get::<Left>(entity), Some(&Left(10)));
        assert_eq!(recovered.get::<Right>(entity), Some(&Right(20)));
        assert_eq!(recovered.current_tick(), leader.current_tick());
    }

    #[test]
    fn follower_round_trip_divergent_live_world() {
        let dir = tempfile::tempdir().unwrap();
        let mut leader = replay_test_world();
        let mut source_codecs = CodecRegistry::new();
        source_codecs
            .register_as::<Pos>("pos", &mut leader)
            .unwrap();
        source_codecs
            .register_as::<Health>("health", &mut leader)
            .unwrap();
        source_codecs
            .register_as::<Name>("name", &mut leader)
            .unwrap();
        let baseline = leader.spawn((Pos { x: 1.0, y: 2.0 }, Name("baseline".into())));
        let a = leader.spawn((Health(1),));
        let b = leader.spawn((Health(2),));
        assert!(leader.despawn(b));
        assert!(leader.despawn(a));

        let mut codec_world = replay_test_world();
        let mut target_codecs = CodecRegistry::new();
        target_codecs
            .register_as::<Name>("name", &mut codec_world)
            .unwrap();
        target_codecs
            .register_as::<Pos>("pos", &mut codec_world)
            .unwrap();
        target_codecs
            .register_as::<Health>("health", &mut codec_world)
            .unwrap();
        let mut replica = replay_test_world();
        replica.register_component::<Health>();
        replica.register_component::<Name>();
        replica.register_component::<Pos>();
        assert_eq!(
            replica.spawn((Pos { x: 1.0, y: 2.0 }, Name("baseline".into()))),
            baseline
        );
        assert_eq!(replica.spawn((Health(1),)), a);
        assert_eq!(replica.spawn((Health(2),)), b);
        // Same live state and generations, different free-list order.
        assert!(replica.despawn(a));
        assert!(replica.despawn(b));
        assert_ne!(
            leader.entity_allocator_state().1,
            replica.entity_allocator_state().1
        );
        assert_eq!(leader.current_tick(), replica.current_tick());
        assert_eq!(
            crate::world_fingerprint(&leader, &source_codecs).unwrap(),
            crate::world_fingerprint(&replica, &target_codecs).unwrap(),
        );

        let mut wal = Wal::create(dir.path(), &source_codecs, WalConfig::default()).unwrap();
        let entity = leader.alloc_entity();
        assert_eq!(entity.index(), a.index());
        let mut changes = EnumChangeSet::new();
        changes
            .spawn_bundle(
                &mut leader,
                entity,
                (Pos { x: 3.0, y: 4.0 }, Name("spawned".into())),
            )
            .unwrap();
        changes.insert(&mut leader, entity, Health(42));
        changes.insert(&mut leader, baseline, Name("updated".into()));
        wal.append(&changes, &source_codecs, leader.current_tick())
            .unwrap();
        changes.apply(&mut leader).unwrap();

        let mut cursor = WalCursor::open(dir.path(), 0).unwrap();
        let batch = cursor.next_batch(1).unwrap();
        let batch = ReplicationBatch::from_bytes(&batch.to_bytes().unwrap()).unwrap();
        let follower = Follower::new();
        assert_eq!(
            follower
                .advance(&batch, &mut replica, &target_codecs)
                .unwrap(),
            1
        );
        assert_eq!(
            follower
                .advance(&batch, &mut replica, &target_codecs)
                .unwrap(),
            1
        );
        assert_eq!(
            crate::world_fingerprint(&leader, &source_codecs).unwrap(),
            crate::world_fingerprint(&replica, &target_codecs).unwrap(),
        );
        assert_eq!(replica.current_tick(), leader.current_tick());
        assert_eq!(replica.get::<Health>(entity), Some(&Health(42)));
        assert_eq!(replica.get::<Name>(entity).unwrap().0, "spawned");
        // The adopted slot must no longer be free, and adopting it must not
        // consume the unrelated slot at the end of the local free list.
        assert_eq!(replica.alloc_entity(), leader.alloc_entity());
        assert_ne!(replica.alloc_entity(), entity);
        assert!(!follower.is_poisoned());
    }

    #[test]
    fn follower_replay_reuses_logged_entity_slot_in_mutation_order() {
        let mut replica = replay_test_world();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut replica).unwrap();
        let old = replica.spawn((Pos { x: 1.0, y: 2.0 },));
        let replacement = minkowski::Entity::from_bits(old.to_bits() + (1u64 << 32));
        let encode = |value| {
            rkyv::to_bytes::<rkyv::rancor::Error>(&value)
                .unwrap()
                .to_vec()
        };
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                seq: 0,
                tick_after: replica.current_tick(),
                mutations: vec![
                    SerializedMutation::Despawn {
                        entity: old.to_bits(),
                    },
                    SerializedMutation::Spawn {
                        entity: replacement.to_bits(),
                        components: vec![(0, encode(Pos { x: 3.0, y: 4.0 }))],
                    },
                    SerializedMutation::Insert {
                        entity: replacement.to_bits(),
                        component_id: 0,
                        data: encode(Pos { x: 5.0, y: 6.0 }),
                    },
                ],
            }],
        };
        let follower = Follower::new();
        assert_eq!(follower.advance(&batch, &mut replica, &codecs).unwrap(), 1);
        assert!(!replica.is_alive(old));
        assert!(replica.is_alive(replacement));
        assert_eq!(
            replica.get::<Pos>(replacement),
            Some(&Pos { x: 5.0, y: 6.0 })
        );
        assert_ne!(replica.alloc_entity().index(), replacement.index());
    }

    #[test]
    fn follower_replay_refuses_occupied_or_older_entity_generation() {
        for occupied in [true, false] {
            let mut replica = replay_test_world();
            let mut codecs = CodecRegistry::new();
            codecs.register_as::<Pos>("pos", &mut replica).unwrap();
            let entity = replica.spawn((Pos { x: 1.0, y: 2.0 },));
            let requested = if occupied {
                // Placement conflicts even when the incoming generation differs.
                minkowski::Entity::from_bits(entity.to_bits() + (1u64 << 32))
            } else {
                assert!(replica.despawn(entity));
                entity // The local generation is now newer than this handle.
            };
            let before = replica.entity_allocator_state();
            let before = (before.0.to_vec(), before.1.to_vec());
            let batch = ReplicationBatch {
                schema: test_schema(),
                records: vec![WalRecord {
                    seq: 0,
                    tick_after: replica.current_tick(),
                    mutations: vec![SerializedMutation::Spawn {
                        entity: requested.to_bits(),
                        components: vec![(
                            0,
                            rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: 9.0, y: 10.0 })
                                .unwrap()
                                .to_vec(),
                        )],
                    }],
                }],
            };
            let follower = Follower::new();
            let err = follower.advance(&batch, &mut replica, &codecs).unwrap_err();
            if occupied {
                assert!(matches!(err, FollowerError::Apply(WalError::Apply(
                    minkowski::ApplyError::AlreadyPlaced(e)
                )) if e == requested));
                assert_eq!(replica.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
            } else {
                assert!(matches!(err, FollowerError::Apply(WalError::Apply(
                    minkowski::ApplyError::DeadEntity(e)
                )) if e == requested));
                assert!(!replica.is_alive(entity));
            }
            assert_eq!(
                replica.entity_allocator_state(),
                (before.0.as_slice(), before.1.as_slice())
            );
            assert_eq!(follower.applied_seq(), 0);
            assert!(follower.is_poisoned());
            assert!(matches!(
                follower.read_at(0, &replica, |_| ()),
                Err(FollowerError::Poisoned)
            ));
        }
    }

    #[test]
    fn follower_replay_claims_logged_slots_beyond_allocator_tail() {
        let mut replica = replay_test_world();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut replica).unwrap();
        let entities = [
            minkowski::Entity::from_bits((7u64 << 32) | 5),
            minkowski::Entity::from_bits((2u64 << 32) | 3),
        ];
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                seq: 0,
                tick_after: replica.current_tick(),
                mutations: entities
                    .iter()
                    .map(|entity| SerializedMutation::Spawn {
                        entity: entity.to_bits(),
                        components: vec![(
                            0,
                            rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: 1.0, y: 2.0 })
                                .unwrap()
                                .to_vec(),
                        )],
                    })
                    .collect(),
            }],
        };
        let follower = Follower::new();
        assert_eq!(follower.advance(&batch, &mut replica, &codecs).unwrap(), 1);
        for entity in entities {
            assert!(replica.is_alive(entity));
            assert_eq!(replica.get::<Pos>(entity), Some(&Pos { x: 1.0, y: 2.0 }));
        }
        let mut allocated = std::collections::HashSet::new();
        for _ in 0..8 {
            let entity = replica.alloc_entity();
            assert!(
                !entities
                    .iter()
                    .any(|adopted| adopted.index() == entity.index())
            );
            assert!(allocated.insert(entity.index()));
        }
    }

    #[test]
    fn follower_remap_refuses_missing_world_type_before_mutation() {
        let mut codec_world = replay_test_world();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut codec_world).unwrap();
        let mut replica = replay_test_world();
        replica.spawn((Health(10),));
        let before = replica.entity_allocator_state();
        let before = (before.0.to_vec(), before.1.to_vec());
        let tick = replica.current_tick();
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                seq: 0,
                tick_after: tick + 10,
                mutations: vec![SerializedMutation::Spawn {
                    entity: 1,
                    components: vec![(
                        0,
                        rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: 1.0, y: 2.0 })
                            .unwrap()
                            .to_vec(),
                    )],
                }],
            }],
        };
        let follower = Follower::new();
        assert!(matches!(
            follower.advance(&batch, &mut replica, &codecs),
            Err(FollowerError::Apply(WalError::Format(_)))
        ));
        assert_eq!(replica.current_tick(), tick);
        assert_eq!(replica.entity_count(), 1);
        assert_eq!(
            replica.entity_allocator_state(),
            (before.0.as_slice(), before.1.as_slice())
        );
        assert_eq!(follower.applied_seq(), 0);
        assert!(follower.is_poisoned());
    }

    #[test]
    fn convergence_100_transactions_leader_replica() {
        let dir = tempfile::tempdir().unwrap();
        let (wal_dir, leader, codecs) = run_leader(dir.path(), 100);

        // Ship transport bytes: WalCursor reads the log, batches go over the
        // wire as bytes, replica decodes.
        let mut cursor = WalCursor::open(&wal_dir, 0).unwrap();
        let mut replica = World::new();
        // The replica's codec registry must resolve the same stable names.
        let mut reg = CodecRegistry::new();
        reg.register_as::<Pos>("pos", &mut replica).unwrap();
        reg.register_as::<Name>("name", &mut replica).unwrap();
        reg.register_as::<Health>("health", &mut replica).unwrap();

        let follower = Follower::new();
        let mut last_seq = 0u64;
        loop {
            let batch = cursor.next_batch(10).unwrap();
            if batch.records.is_empty() {
                break;
            }
            let bytes = batch.to_bytes().unwrap();
            let decoded = ReplicationBatch::from_bytes(&bytes).unwrap();
            last_seq = follower.advance(&decoded, &mut replica, &reg).unwrap();
        }

        assert_eq!(follower.applied_seq(), last_seq);
        assert!(!follower.is_poisoned());

        // The deliverable invariant: full state equality.
        let fp_leader = crate::fingerprint::world_fingerprint(&leader, &codecs).unwrap();
        let fp_replica = crate::fingerprint::world_fingerprint(&replica, &reg).unwrap();
        assert_eq!(fp_leader, fp_replica, "replica state diverged from leader");

        // The replica's tick equals the leader's post-commit tick: with no
        // interleaved leader mutations, apply advances deterministically
        // from each record's commit-boundary tick.
        assert_eq!(replica.current_tick(), leader.current_tick());
    }

    #[test]
    fn follower_skips_applied_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let (wal_dir, _leader, _codecs) = run_leader(dir.path(), 5);

        let mut replica = World::new();
        let mut reg = CodecRegistry::new();
        reg.register_as::<Pos>("pos", &mut replica).unwrap();
        reg.register_as::<Name>("name", &mut replica).unwrap();
        reg.register_as::<Health>("health", &mut replica).unwrap();

        let mut cursor = WalCursor::open(&wal_dir, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        let follower = Follower::new();
        let first = follower.advance(&batch, &mut replica, &reg).unwrap();
        // Re-advance the same batch: every record is at or below applied_seq.
        let second = follower.advance(&batch, &mut replica, &reg).unwrap();
        assert_eq!(first, second);
        assert!(!follower.is_poisoned());
    }

    #[test]
    fn follower_poisons_on_mid_batch_failure() {
        // A batch whose second record despawns an entity the first record
        // spawned references nothing invalid — instead craft a direct failure:
        // a Despawn record for an entity that was never placed fails apply.
        let schema = test_schema();
        let batch = ReplicationBatch {
            schema,
            records: vec![
                WalRecord {
                    tick_after: 1,
                    seq: 0,
                    mutations: vec![SerializedMutation::Spawn {
                        entity: 0,
                        components: vec![(
                            0usize,
                            rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: 1.0, y: 2.0 })
                                .unwrap()
                                .to_vec(),
                        )],
                    }],
                },
                WalRecord {
                    tick_after: 2,
                    seq: 1,
                    mutations: vec![SerializedMutation::Despawn { entity: 99 }],
                },
            ],
        };

        let mut replica = World::new();
        let mut reg = CodecRegistry::new();
        reg.register_as::<Pos>("pos", &mut replica).unwrap();

        let follower = Follower::new();
        let err = follower.advance(&batch, &mut replica, &reg).unwrap_err();
        assert!(matches!(err, FollowerError::Apply(_)), "got: {err:?}");
        assert!(follower.is_poisoned());
        // State is exactly the applied prefix: entity 0 exists.
        let e0 = minkowski::Entity::from_bits(0);
        assert!(replica.is_alive(e0));

        // Poisoned forever: advance and read_at refuse.
        assert!(matches!(
            follower.advance(&batch, &mut replica, &reg),
            Err(FollowerError::Poisoned)
        ));
        assert!(matches!(
            follower.read_at(1, &replica, |_| ()),
            Err(FollowerError::Poisoned)
        ));
    }

    #[test]
    fn follower_tick_regression_poisons() {
        let mut replica = World::new();
        let mut reg = CodecRegistry::new();
        reg.register_as::<Pos>("pos", &mut replica).unwrap();
        replica.set_current_tick(50);

        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                tick_after: 10, // below the world's current tick 50
                seq: 0,
                mutations: vec![SerializedMutation::Despawn { entity: 5u64 << 32 }],
            }],
        };
        let follower = Follower::new();
        assert!(matches!(
            follower.advance(&batch, &mut replica, &reg),
            Err(FollowerError::TickRegression { .. })
        ));
        assert!(follower.is_poisoned());
    }

    #[test]
    fn read_at_bounds() {
        let mut replica = World::new();
        let mut reg = CodecRegistry::new();
        reg.register_as::<Pos>("pos", &mut replica).unwrap();
        let follower = Follower::new();
        // Nothing applied: any read is stale.
        assert!(matches!(
            follower.read_at(0, &replica, |_| ()),
            Err(FollowerError::Stale {
                requested: 0,
                applied_seq: 0
            })
        ));
        // Apply one record through seq 0; now reads through 0 are valid and
        // reads beyond the high-water mark are stale.
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                tick_after: 1,
                seq: 0,
                mutations: vec![SerializedMutation::Spawn {
                    entity: 0,
                    components: vec![(
                        0usize,
                        rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: 1.0, y: 2.0 })
                            .unwrap()
                            .to_vec(),
                    )],
                }],
            }],
        };
        follower.advance(&batch, &mut replica, &reg).unwrap();
        assert_eq!(follower.applied_seq(), 1);
        assert_eq!(
            follower.read_at(0, &replica, World::entity_count).unwrap(),
            1
        );
        assert!(matches!(
            follower.read_at(1, &replica, |_| ()),
            Err(FollowerError::Stale {
                requested: 1,
                applied_seq: 1
            })
        ));
    }

    #[test]
    fn follower_gap_poisons() {
        let mut replica = World::new();
        let mut reg = CodecRegistry::new();
        reg.register_as::<Pos>("pos", &mut replica).unwrap();
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                tick_after: 1,
                seq: 3, // applied_seq is 0 — this is a gap, not the next record
                mutations: vec![SerializedMutation::Despawn { entity: 0 }],
            }],
        };
        let follower = Follower::new();
        assert!(matches!(
            follower.advance(&batch, &mut replica, &reg),
            Err(FollowerError::Gap {
                expected: 0,
                got: 3
            })
        ));
        assert!(follower.is_poisoned());
    }

    #[test]
    fn follower_baseline_seeds_high_water() {
        // A replica bootstrapped from an LSM baseline holds records below
        // the baseline seq; its first shipped record is seq == baseline.
        let mut replica = World::new();
        let mut reg = CodecRegistry::new();
        reg.register_as::<Pos>("pos", &mut replica).unwrap();
        let batch = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                tick_after: 9,
                seq: 7, // == baseline: no gap
                mutations: vec![SerializedMutation::Spawn {
                    entity: 0,
                    components: vec![(
                        0usize,
                        rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: 1.0, y: 2.0 })
                            .unwrap()
                            .to_vec(),
                    )],
                }],
            }],
        };
        let follower = Follower::with_baseline(7);
        follower.advance(&batch, &mut replica, &reg).unwrap();
        assert_eq!(follower.applied_seq(), 8);
        assert!(replica.is_alive(minkowski::Entity::from_bits(0)));

        // A record below the baseline is skipped, not gap-poisoned.
        let old = ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                tick_after: 3,
                seq: 5,
                mutations: vec![SerializedMutation::Despawn { entity: 0 }],
            }],
        };
        follower.advance(&old, &mut replica, &reg).unwrap();
        assert!(replica.is_alive(minkowski::Entity::from_bits(0)));
    }

    // ── Loom: Follower state races (issue #255) ──────────────────────
    //
    // Worlds and batches are built OUTSIDE `loom::model` where possible:
    // pool allocation is not part of the state machine under test, and loom
    // re-executes thread bodies per schedule, so world operations inside the
    // model must be idempotent — Insert records qualify (same-value re-apply
    // is a no-op); Spawn does not (AlreadyPlaced on re-execution would
    // poison spuriously). The follower's high_water/poisoned are
    // `crate::sync` atomics — loom resets those per schedule, which is
    // exactly the state machine under test.

    #[cfg(loom)]
    fn retarget_batch(batch: &ReplicationBatch, entity_bits: u64) -> ReplicationBatch {
        let mut batch = batch.clone();
        for record in &mut batch.records {
            for m in &mut record.mutations {
                if let SerializedMutation::Insert { entity, .. } = m {
                    *entity = entity_bits;
                }
            }
        }
        batch
    }

    #[cfg(loom)]
    fn pos_world() -> World {
        World::new()
    }

    #[cfg(loom)]
    fn make_insert_batch(seq: u64, _entity_bits: u64, value: f32) -> ReplicationBatch {
        ReplicationBatch {
            schema: test_schema(),
            records: vec![WalRecord {
                tick_after: seq + 1,
                seq,
                mutations: vec![SerializedMutation::Insert {
                    entity: 0, // retarget_batch rewrites to the thread's entity
                    component_id: 0usize,
                    data: rkyv::to_bytes::<rkyv::rancor::Error>(&Pos { x: value, y: 0.0 })
                        .unwrap()
                        .to_vec(),
                }],
            }],
        }
    }

    #[cfg(loom)]
    fn pos_world_with_entity() -> (World, CodecRegistry) {
        let mut world = World::new();
        let mut reg = CodecRegistry::new();
        reg.register_as::<Pos>("pos", &mut world).unwrap();
        let _e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        (world, reg)
    }

    /// Writer advances one Insert record; reader probes `read_at`
    /// concurrently. Worlds live in shared mutexes (hoisted out of the
    /// model — loom re-executes the model closure, so nothing may be moved).
    /// Invariants: `read_at` never returns Ok for a seq at or above the
    /// applied prefix at check time, high-water never decreases, and the
    /// final state is fully applied and unpoisoned.
    #[cfg(loom)]
    #[test]
    fn loom_follower_advance_vs_read_at() {
        use crate::sync::{Arc, Mutex};

        let batch_proto = make_insert_batch(0, 0, 7.0);

        loom::model(move || {
            let follower = Arc::new(Follower::new());
            let batch = batch_proto.clone();
            let writer_world = Arc::new(Mutex::new(pos_world()));
            let reader_world = Arc::new(Mutex::new(pos_world()));

            let writer_follower = follower.clone();
            let writer = loom::thread::spawn(move || {
                let mut world = pos_world();
                let mut reg = CodecRegistry::new();
                reg.register_as::<Pos>("pos", &mut world).unwrap();
                let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
                let batch = retarget_batch(&batch, e.to_bits());
                writer_follower.advance(&batch, &mut world, &reg)
            });

            let reader_follower = follower.clone();
            let reader = loom::thread::spawn(move || {
                let world = reader_world.lock();
                for seq in 0..=1 {
                    match reader_follower.read_at(seq, &world, |_| ()) {
                        Ok(()) => {
                            assert!(
                                reader_follower.applied_seq() > seq,
                                "read_at({seq}) passed but applied_seq is {}",
                                reader_follower.applied_seq()
                            );
                        }
                        Err(FollowerError::Stale { applied_seq, .. }) => {
                            assert!(
                                seq >= applied_seq,
                                "read_at({seq}) stale but applied_seq is {applied_seq}"
                            );
                        }
                        Err(e) => panic!("unexpected error: {e:?}"),
                    }
                }
            });

            writer.join().unwrap();
            reader.join().unwrap();

            assert_eq!(follower.applied_seq(), 1);
            assert!(!follower.is_poisoned());
        });
    }

    /// Two threads advance the same follower with disjoint single-record
    /// Insert streams, each against its own world. Terminal states:
    /// - A completes before B starts: both Ok, high-water 2, unpoisoned.
    /// - B runs first: B gap-poisons; A either completes first (poisoned,
    ///   high-water 1) or observes the poison (Err(Poisoned)).
    /// A never gaps (seq 0 vs high-water 0 or 1 is never a gap).
    #[cfg(loom)]
    #[test]
    fn loom_follower_concurrent_advance() {
        use crate::sync::{Arc, Mutex};

        let batch_a_proto = make_insert_batch(0, 0, 1.0);
        let batch_b_proto = make_insert_batch(1, 0, 2.0);

        loom::model(move || {
            let follower = Arc::new(Follower::new());
            let batch_a = batch_a_proto.clone();
            let batch_b = batch_b_proto.clone();
            let world_a = Arc::new(Mutex::new(pos_world()));
            let world_b = Arc::new(Mutex::new(pos_world()));
            let follower_a = follower.clone();
            let follower_b = follower.clone();
            let a = loom::thread::spawn(move || {
                let mut world = pos_world();
                let mut reg = CodecRegistry::new();
                reg.register_as::<Pos>("pos", &mut world).unwrap();
                let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
                let batch = retarget_batch(&batch_a, e.to_bits());
                follower_a.advance(&batch, &mut world, &reg)
            });
            let b = loom::thread::spawn(move || {
                let mut world = pos_world();
                let mut reg = CodecRegistry::new();
                reg.register_as::<Pos>("pos", &mut world).unwrap();
                let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
                let batch = retarget_batch(&batch_b, e.to_bits());
                follower_b.advance(&batch, &mut world, &reg)
            });

            let ra = a.join().unwrap();
            let rb = b.join().unwrap();

            match (&ra, &rb) {
                (Ok(_), Ok(_)) => {
                    assert_eq!(follower.applied_seq(), 2);
                    assert!(!follower.is_poisoned());
                }
                (Ok(_), Err(FollowerError::Gap { .. }))
                | (Err(FollowerError::Poisoned), Err(FollowerError::Gap { .. })) => {
                    assert!(follower.is_poisoned());
                    assert!(matches!(follower.applied_seq(), 0 | 1));
                }
                _ => panic!("unexpected outcome: {ra:?} / {rb:?}"),
            }
        });
    }
}
