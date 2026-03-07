use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use rkyv::{Archive, Deserialize, Serialize};

use minkowski::{ComponentId, World};

use crate::codec::CodecRegistry;
use crate::record::{WalEntry, WalRecord, WalSchema};
use crate::wal::{apply_record, list_segments, read_next_frame, WalError};

/// Read-only cursor over a segmented WAL directory. Opens its own file
/// handles so it can read concurrently with an active writer. Lazily
/// advances across segment files.
pub struct WalCursor {
    dir: PathBuf,
    file: File,
    pos: u64,
    next_seq: u64,
    schema: Option<WalSchema>,
    current_segment_start_seq: u64,
}

impl WalCursor {
    /// Open a WAL directory for reading, starting from `from_seq`.
    /// Finds the segment containing `from_seq`, parses its schema preamble,
    /// and scans forward to the first record with `seq >= from_seq`.
    /// Returns `Err(CursorBehind)` if all segments start after `from_seq`.
    pub fn open(dir: &Path, from_seq: u64) -> Result<Self, WalError> {
        let segments = list_segments(dir)?;
        if segments.is_empty() {
            return Err(WalError::Format("no WAL segments found".into()));
        }

        // Find segment containing from_seq: largest start_seq <= from_seq
        let seg_idx = match segments.iter().rposition(|(start, _)| *start <= from_seq) {
            Some(idx) => idx,
            None => {
                return Err(WalError::CursorBehind {
                    requested: from_seq,
                    oldest: segments[0].0,
                });
            }
        };

        let (start_seq, seg_path) = &segments[seg_idx];
        let file = File::open(seg_path)?;
        let mut pos: u64 = 0;
        let mut schema = None;

        // Scan forward to from_seq
        loop {
            match read_next_frame(&file, pos)? {
                Some((WalEntry::Schema(s), next_pos)) => {
                    schema = Some(s);
                    pos = next_pos;
                }
                Some((WalEntry::Mutations(record), _next_pos)) => {
                    if record.seq >= from_seq {
                        break; // Don't advance past this record
                    }
                    pos = _next_pos;
                }
                None => break,
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            file,
            pos,
            next_seq: from_seq,
            schema,
            current_segment_start_seq: *start_seq,
        })
    }

    /// Read up to `limit` records from the current position.
    /// Returns a `ReplicationBatch` with the schema and records.
    /// An empty `records` vec means the cursor has caught up.
    /// Lazily advances across segment boundaries.
    pub fn next_batch(&mut self, limit: usize) -> Result<ReplicationBatch, WalError> {
        let mut records = Vec::new();

        while records.len() < limit {
            match read_next_frame(&self.file, self.pos)? {
                Some((WalEntry::Schema(s), next_pos)) => {
                    self.schema = Some(s);
                    self.pos = next_pos;
                }
                Some((WalEntry::Mutations(record), next_pos)) => {
                    self.next_seq = record.seq + 1;
                    records.push(record);
                    self.pos = next_pos;
                }
                None => {
                    // Try to advance to next segment
                    if !self.try_advance_segment()? {
                        break; // No more segments — caught up
                    }
                }
            }
        }

        let schema = self
            .schema
            .clone()
            .unwrap_or_else(|| WalSchema { components: vec![] });
        Ok(ReplicationBatch { schema, records })
    }

    /// Try to open the next segment file. Returns true if advanced.
    fn try_advance_segment(&mut self) -> Result<bool, WalError> {
        let segments = list_segments(&self.dir)?;
        let next = segments
            .iter()
            .find(|(start, _)| *start > self.current_segment_start_seq);

        match next {
            Some((start_seq, path)) => {
                self.file = File::open(path)?;
                self.pos = 0;
                self.current_segment_start_seq = *start_seq;
                // Parse schema preamble of new segment
                if let Some((WalEntry::Schema(s), next_pos)) = read_next_frame(&self.file, 0)? {
                    self.schema = Some(s);
                    self.pos = next_pos;
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// The schema parsed from the WAL preamble, if present.
    pub fn schema(&self) -> Option<&WalSchema> {
        self.schema.as_ref()
    }

    /// Next expected sequence number. Useful for persisting cursor position.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
}

/// Self-describing replication payload. Every batch carries its own schema
/// so receivers can decode without prior handshake.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct ReplicationBatch {
    pub schema: WalSchema,
    pub records: Vec<WalRecord>,
}

impl ReplicationBatch {
    /// Serialize to bytes via rkyv.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WalError> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map(|v| v.to_vec())
            .map_err(|e| WalError::Format(e.to_string()))
    }

    /// Deserialize from bytes via rkyv.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WalError> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Format(e.to_string()))
    }
}

/// Apply a `ReplicationBatch` to a target World.
///
/// Builds a component ID remap from the batch schema, then applies each
/// record atomically (one `EnumChangeSet` per record). Returns the last
/// applied seq, or `None` if the batch is empty.
///
/// Each record is applied as its own `EnumChangeSet` — per-record atomicity,
/// not per-batch. On error, previously applied records are NOT rolled back;
/// the caller can use the cursor's `next_seq()` to determine recovery position.
pub fn apply_batch(
    batch: &ReplicationBatch,
    world: &mut World,
    codecs: &CodecRegistry,
) -> Result<Option<u64>, WalError> {
    let remap: Option<HashMap<ComponentId, ComponentId>> = if batch.schema.components.is_empty() {
        None
    } else {
        Some(codecs.build_remap(&batch.schema.components)?)
    };

    let mut last_seq = None;
    for record in &batch.records {
        apply_record(record, world, codecs, remap.as_ref())?;
        last_seq = Some(record.seq);
    }

    Ok(last_seq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use crate::record::{ComponentSchema, SerializedMutation};
    use crate::wal::{Wal, WalConfig};
    use minkowski::{EnumChangeSet, World};

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
        codecs.register_as::<Pos>("pos", &mut world);

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
            );
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world);
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
                    seq: 0,
                    mutations: vec![SerializedMutation::Despawn { entity: 1 }],
                },
                WalRecord {
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
        assert_eq!(cursor.next_seq(), 3);

        assert!(cursor.schema().is_some());
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
        assert_eq!(cursor.next_seq(), 5);
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
        replica_codecs.register_as::<Pos>("pos", &mut replica);

        let last_seq = apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

        assert_eq!(last_seq, Some(2));
        assert_eq!(replica.query::<(&Pos,)>().count(), 3);
    }

    #[test]
    fn apply_batch_insert_remove() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);
        codecs.register_as::<Health>("health", &mut world);

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);

        let mut cs2 = EnumChangeSet::new();
        cs2.insert::<Health>(&mut world, e, Health(100));
        cs2.remove::<Pos>(&mut world, e);
        wal.append(&cs2, &codecs).unwrap();
        cs2.apply(&mut world);

        drop(wal);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        let mut replica = World::new();
        let mut replica_codecs = CodecRegistry::new();
        replica_codecs.register_as::<Pos>("pos", &mut replica);
        replica_codecs.register_as::<Health>("health", &mut replica);

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
        codecs.register_as::<Pos>("pos", &mut world);
        codecs.register_as::<Health>("health", &mut world);

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 }, Health(50)));
        wal.append(&cs, &codecs).unwrap();
        drop(wal);

        // Replica: Health=0, Pos=1 (opposite order)
        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();

        let mut replica = World::new();
        let mut replica_codecs = CodecRegistry::new();
        replica_codecs.register_as::<Health>("health", &mut replica);
        replica_codecs.register_as::<Pos>("pos", &mut replica);

        apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

        let positions: Vec<(f32, f32)> =
            replica.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(1.0, 2.0)]);

        let health: Vec<u32> = replica.query::<(&Health,)>().map(|h| h.0 .0).collect();
        assert_eq!(health, vec![50]);
    }

    #[test]
    fn apply_batch_preserves_transaction_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("boundaries.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();

        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);

        let mut cs2 = EnumChangeSet::new();
        cs2.record_despawn(e);
        wal.append(&cs2, &codecs).unwrap();
        cs2.apply(&mut world);

        drop(wal);

        let mut cursor = WalCursor::open(&wal_path, 0).unwrap();
        let batch = cursor.next_batch(100).unwrap();
        assert_eq!(batch.records.len(), 2);

        let mut replica = World::new();
        let mut replica_codecs = CodecRegistry::new();
        replica_codecs.register_as::<Pos>("pos", &mut replica);

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

    // ── Error path tests ─────────────────────────────────────────

    #[test]
    fn from_bytes_corrupt_returns_error() {
        let result = ReplicationBatch::from_bytes(&[0xFF; 32]);
        assert!(matches!(result, Err(WalError::Format(_))));
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
        use crate::snapshot::Snapshot;

        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("source.wal");
        let snap_path = dir.path().join("source.snap");

        // Source: spawn 5 entities, snapshot, then 3 more via WAL
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        for i in 0..5 {
            world.spawn((Pos {
                x: i as f32,
                y: 0.0,
            },));
        }

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let snap = Snapshot::new();
        let header = snap
            .save(&snap_path, &world, &codecs, wal.next_seq())
            .unwrap();
        assert_eq!(header.entity_count, 5);

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
            );
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world);
        }

        drop(wal);

        // Replica: load snapshot + pull WAL
        let mut replica_codecs = CodecRegistry::new();
        let mut tmp = World::new();
        replica_codecs.register_as::<Pos>("pos", &mut tmp);

        let (mut replica, snap_seq) = snap.load(&snap_path, &replica_codecs).unwrap();
        assert_eq!(replica.query::<(&Pos,)>().count(), 5);

        let mut cursor = WalCursor::open(&wal_path, snap_seq).unwrap();
        let batch = cursor.next_batch(100).unwrap();
        assert_eq!(batch.records.len(), 3);

        let last = apply_batch(&batch, &mut replica, &replica_codecs).unwrap();

        assert_eq!(replica.query::<(&Pos,)>().count(), 8);
        assert_eq!(last, Some(2));

        // Cursor should be caught up
        let batch2 = cursor.next_batch(100).unwrap();
        assert!(batch2.records.is_empty());
    }
}
