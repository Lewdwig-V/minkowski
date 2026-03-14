use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufWriter, Write};
use std::path::Path;

use minkowski::component::Component;
use minkowski::{BTreeIndex, ChangeTick, Entity, HashIndex, SpatialIndex};
use rkyv::api::high::HighValidator;
use rkyv::bytecheck::CheckBytes;
use rkyv::de::Pool;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize, rancor};

/// Index file magic identifying the persistent index format.
const INDEX_MAGIC: [u8; 8] = *b"MK2INDXK";

/// Header size: magic (8) + CRC32 (4) + version (4) + length (8) = 24.
pub(crate) const INDEX_HEADER_SIZE: usize = 24;

/// Current index file format version.
const INDEX_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum IndexPersistError {
    #[error("index I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("index format error: {0}")]
    Format(String),
}

/// A secondary index that can be saved to disk and loaded on recovery.
///
/// `save` is object-safe — [`AutoCheckpoint`](crate::AutoCheckpoint) can hold
/// registered indexes and call `save` on each. `load` is on the concrete type
/// (returns `Self`, not object-safe).
///
/// Writes use atomic rename: data goes to a `.tmp` file, then is renamed
/// to the final path. A crash during write cannot corrupt the previous file.
///
/// After loading, call [`SpatialIndex::update`] to catch up with
/// mutations that occurred after the index was last saved.
pub trait PersistentIndex: SpatialIndex + Send {
    /// Serialize the index state to a file.
    fn save(&self, path: &Path) -> Result<(), IndexPersistError>;
}

/// Write an index envelope: `[magic 8B][crc32 4B][reserved 4B][len u64][payload]`.
///
/// Uses write-to-tmp + fsync + atomic rename: data is flushed to stable
/// storage before the rename, so a crash at any point cannot corrupt the
/// previous file at `path`.
pub(crate) fn write_index_file(path: &Path, payload: &[u8]) -> Result<(), IndexPersistError> {
    let tmp_path = path.with_extension("idx.tmp");
    let crc = crc32fast::hash(payload);
    let len = payload.len() as u64;

    let result = (|| -> Result<(), IndexPersistError> {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&INDEX_MAGIC)?;
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(&INDEX_VERSION.to_le_bytes())?;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(payload)?;
        writer.flush()?;
        let file = writer.into_inner().map_err(|e| {
            IndexPersistError::Io(std::io::Error::other(format!("flush failed: {e}")))
        })?;
        file.sync_data()?;
        drop(file);
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

/// Read and validate an index file. Returns the verified payload bytes.
pub(crate) fn read_index_file(path: &Path) -> Result<Vec<u8>, IndexPersistError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < INDEX_HEADER_SIZE {
        return Err(IndexPersistError::Format("index file too small".into()));
    }
    if bytes[..8] != INDEX_MAGIC {
        return Err(IndexPersistError::Format("invalid index file magic".into()));
    }
    let stored_crc = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let version = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if version != INDEX_VERSION {
        return Err(IndexPersistError::Format(format!(
            "unsupported index format version {version}, expected {INDEX_VERSION}"
        )));
    }
    let len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let end = INDEX_HEADER_SIZE
        .checked_add(len)
        .ok_or_else(|| IndexPersistError::Format("invalid payload length".into()))?;
    if bytes.len() < end {
        return Err(IndexPersistError::Format(format!(
            "index truncated: expected {len} payload bytes, got {}",
            bytes.len() - INDEX_HEADER_SIZE
        )));
    }
    let payload = &bytes[INDEX_HEADER_SIZE..end];
    let actual_crc = crc32fast::hash(payload);
    if actual_crc != stored_crc {
        return Err(IndexPersistError::Format(format!(
            "index checksum mismatch: expected {stored_crc:#010x}, got {actual_crc:#010x}"
        )));
    }
    Ok(payload.to_vec())
}

// ---------------------------------------------------------------------------
// rkyv-based save/load for BTreeIndex and HashIndex
// ---------------------------------------------------------------------------

/// Type-erased payload for index persistence. Keys are serialized individually
/// as byte blobs to avoid propagating rkyv generic bounds through the trait.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct IndexPayload {
    /// Key type name from `std::any::type_name::<T>()`. Validated on load
    /// to catch wrong-type loads (e.g. loading a Score index as Health).
    key_type_name: String,
    /// Forward map: serialized key bytes → entity bits.
    entries: Vec<(Vec<u8>, Vec<u64>)>,
    /// Reverse map: entity bits → serialized key bytes.
    reverse: Vec<(u64, Vec<u8>)>,
    /// Last sync tick (raw u64). Stored for diagnostics only —
    /// `load_*` functions require the caller to supply the actual
    /// sync tick, because the original tick timeline does not survive
    /// crash recovery.
    last_sync_tick: u64,
}

/// Serialize a single key value to bytes via rkyv.
fn serialize_key<T>(key: &T) -> Result<Vec<u8>, IndexPersistError>
where
    T: Archive
        + for<'a> RkyvSerialize<
            rkyv::api::high::HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, rancor::Error>,
        >,
{
    rkyv::to_bytes::<rancor::Error>(key)
        .map(|v| v.to_vec())
        .map_err(|e| IndexPersistError::Format(format!("key serialization failed: {e}")))
}

/// Deserialize a single key value from bytes via rkyv.
fn deserialize_key<T>(bytes: &[u8]) -> Result<T, IndexPersistError>
where
    T: Archive,
    T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
        + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
{
    rkyv::from_bytes::<T, rancor::Error>(bytes)
        .map_err(|e| IndexPersistError::Format(format!("key deserialization failed: {e}")))
}

impl<T> PersistentIndex for BTreeIndex<T>
where
    T: Component
        + Ord
        + Clone
        + Archive
        + for<'a> RkyvSerialize<
            rkyv::api::high::HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, rancor::Error>,
        >,
{
    fn save(&self, path: &Path) -> Result<(), IndexPersistError> {
        let (tree, reverse, last_sync) = self.as_raw_parts();

        let mut entries = Vec::with_capacity(tree.len());
        for (key, entities) in tree {
            let key_bytes = serialize_key(key)?;
            let entity_bits: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
            entries.push((key_bytes, entity_bits));
        }

        let mut rev = Vec::with_capacity(reverse.len());
        for (entity, key) in reverse {
            let key_bytes = serialize_key(key)?;
            rev.push((entity.to_bits(), key_bytes));
        }

        let payload = IndexPayload {
            key_type_name: std::any::type_name::<T>().to_owned(),
            entries,
            reverse: rev,
            last_sync_tick: last_sync.to_raw(),
        };

        let bytes = rkyv::to_bytes::<rancor::Error>(&payload)
            .map_err(|e| IndexPersistError::Format(format!("payload serialization failed: {e}")))?;

        write_index_file(path, &bytes)
    }
}

/// Load a `BTreeIndex<T>` from a file previously written by
/// [`PersistentIndex::save`].
///
/// `sync_tick` overrides the stored last-sync tick. After crash recovery,
/// the original world's tick timeline no longer exists — pass
/// `world.change_tick()` captured immediately after snapshot restore
/// (before WAL replay) so that a subsequent `update()` catches up with
/// exactly the WAL tail.
pub fn load_btree_index<T>(
    path: &Path,
    sync_tick: ChangeTick,
) -> Result<BTreeIndex<T>, IndexPersistError>
where
    T: Component + Ord + Clone + Archive,
    T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
        + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
{
    let bytes = read_index_file(path)?;
    let payload = rkyv::from_bytes::<IndexPayload, rancor::Error>(&bytes)
        .map_err(|e| IndexPersistError::Format(format!("payload deserialization failed: {e}")))?;

    let expected_type = std::any::type_name::<T>();
    if payload.key_type_name != expected_type {
        return Err(IndexPersistError::Format(format!(
            "index key type mismatch: file has '{}', expected '{}'",
            payload.key_type_name, expected_type
        )));
    }

    let mut tree = std::collections::BTreeMap::new();
    for (key_bytes, entity_bits) in &payload.entries {
        let key: T = deserialize_key(key_bytes)?;
        let entities: Vec<Entity> = entity_bits.iter().map(|&b| Entity::from_bits(b)).collect();
        tree.insert(key, entities);
    }

    let mut reverse = HashMap::new();
    for &(entity_bits, ref key_bytes) in &payload.reverse {
        let key: T = deserialize_key(key_bytes)?;
        reverse.insert(Entity::from_bits(entity_bits), key);
    }

    Ok(BTreeIndex::from_raw_parts(tree, reverse, sync_tick))
}

impl<T> PersistentIndex for HashIndex<T>
where
    T: Component
        + Hash
        + Eq
        + Clone
        + Archive
        + for<'a> RkyvSerialize<
            rkyv::api::high::HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, rancor::Error>,
        >,
{
    fn save(&self, path: &Path) -> Result<(), IndexPersistError> {
        let (map, reverse, last_sync) = self.as_raw_parts();

        let mut entries = Vec::with_capacity(map.len());
        for (key, entities) in map {
            let key_bytes = serialize_key(key)?;
            let entity_bits: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
            entries.push((key_bytes, entity_bits));
        }

        let mut rev = Vec::with_capacity(reverse.len());
        for (entity, key) in reverse {
            let key_bytes = serialize_key(key)?;
            rev.push((entity.to_bits(), key_bytes));
        }

        let payload = IndexPayload {
            key_type_name: std::any::type_name::<T>().to_owned(),
            entries,
            reverse: rev,
            last_sync_tick: last_sync.to_raw(),
        };

        let bytes = rkyv::to_bytes::<rancor::Error>(&payload)
            .map_err(|e| IndexPersistError::Format(format!("payload serialization failed: {e}")))?;

        write_index_file(path, &bytes)
    }
}

/// Load a `HashIndex<T>` from a file previously written by
/// [`PersistentIndex::save`].
///
/// `sync_tick` overrides the stored last-sync tick. See
/// [`load_btree_index`] for rationale.
pub fn load_hash_index<T>(
    path: &Path,
    sync_tick: ChangeTick,
) -> Result<HashIndex<T>, IndexPersistError>
where
    T: Component + Hash + Eq + Clone + Archive,
    T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
        + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
{
    let bytes = read_index_file(path)?;
    let payload = rkyv::from_bytes::<IndexPayload, rancor::Error>(&bytes)
        .map_err(|e| IndexPersistError::Format(format!("payload deserialization failed: {e}")))?;

    let expected_type = std::any::type_name::<T>();
    if payload.key_type_name != expected_type {
        return Err(IndexPersistError::Format(format!(
            "index key type mismatch: file has '{}', expected '{}'",
            payload.key_type_name, expected_type
        )));
    }

    let mut map = HashMap::new();
    for (key_bytes, entity_bits) in &payload.entries {
        let key: T = deserialize_key(key_bytes)?;
        let entities: Vec<Entity> = entity_bits.iter().map(|&b| Entity::from_bits(b)).collect();
        map.insert(key, entities);
    }

    let mut reverse = HashMap::new();
    for &(entity_bits, ref key_bytes) in &payload.reverse {
        let key: T = deserialize_key(key_bytes)?;
        reverse.insert(Entity::from_bits(entity_bits), key);
    }

    Ok(HashIndex::from_raw_parts(map, reverse, sync_tick))
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski::World;

    #[derive(
        Clone,
        Copy,
        Debug,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    #[repr(C)]
    struct Score(u32);

    #[test]
    fn btree_save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("score.idx");

        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));
        let e3 = world.spawn((Score(10),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        let loaded = load_btree_index::<Score>(&path, world.change_tick()).unwrap();
        assert_eq!(loaded.get(&Score(10)).len(), 2);
        assert!(loaded.get(&Score(10)).contains(&e1));
        assert!(loaded.get(&Score(10)).contains(&e3));
        assert_eq!(loaded.get(&Score(20)).len(), 1);
        assert!(loaded.get(&Score(20)).contains(&e2));
    }

    #[test]
    fn btree_empty_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.idx");

        let mut world = World::new();
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        let loaded = load_btree_index::<Score>(&path, world.change_tick()).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn hash_save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("score_hash.idx");

        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));

        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        let loaded = load_hash_index::<Score>(&path, world.change_tick()).unwrap();
        assert_eq!(loaded.get(&Score(10)).len(), 1);
        assert!(loaded.get(&Score(10)).contains(&e1));
        assert_eq!(loaded.get(&Score(20)).len(), 1);
        assert!(loaded.get(&Score(20)).contains(&e2));
    }

    #[test]
    fn write_read_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let payload = b"hello world";
        write_index_file(&path, payload).unwrap();
        let loaded = read_index_file(&path).unwrap();
        assert_eq!(loaded, payload);
    }

    #[test]
    fn read_missing_file() {
        let result = read_index_file(Path::new("/nonexistent/test.idx"));
        assert!(matches!(result, Err(IndexPersistError::Io(_))));
    }

    #[test]
    fn read_too_small() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.idx");
        std::fs::write(&path, [0u8; 10]).unwrap();
        let result = read_index_file(&path);
        assert!(matches!(result, Err(IndexPersistError::Format(_))));
    }

    #[test]
    fn read_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("badmagic.idx");
        let mut data = vec![0u8; INDEX_HEADER_SIZE + 4];
        data[..8].copy_from_slice(b"NOTMAGIC");
        std::fs::write(&path, &data).unwrap();
        let result = read_index_file(&path);
        let err = result.err().unwrap();
        assert!(format!("{err}").contains("magic"));
    }

    #[test]
    fn read_crc_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crc.idx");
        let payload = b"test data";
        write_index_file(&path, payload).unwrap();

        let mut data = std::fs::read(&path).unwrap();
        data[INDEX_HEADER_SIZE] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = read_index_file(&path);
        let err = result.err().unwrap();
        assert!(format!("{err}").contains("checksum"));
    }

    #[test]
    fn btree_stale_catch_up_after_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stale.idx");

        let mut world = World::new();
        world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        // Capture tick at save time — this is what the caller would
        // pass after snapshot restore in a real recovery scenario.
        let saved_tick = world.change_tick();

        // Mutate after save
        *world.get_mut::<Score>(e2).unwrap() = Score(30);
        world.spawn((Score(40),));

        // Load stale index with the save-time tick, catch up
        let mut loaded = load_btree_index::<Score>(&path, saved_tick).unwrap();
        loaded.update(&mut world);

        // Verify: Score(20) gone, Score(30) and Score(40) present
        assert!(loaded.get(&Score(20)).is_empty());
        assert_eq!(loaded.get(&Score(30)).len(), 1);
        assert_eq!(loaded.get(&Score(40)).len(), 1);
        assert_eq!(loaded.get(&Score(10)).len(), 1);
    }

    #[test]
    fn full_recovery_with_persistent_index() {
        use crate::codec::CodecRegistry;
        use crate::snapshot::Snapshot;
        use crate::wal::{Wal, WalConfig};

        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("recovery.wal");
        let snap_path = dir.path().join("checkpoint.snap");
        let idx_path = dir.path().join("score.idx");

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: None,
        };

        // Phase 1: create world, spawn entities via WAL, build & save index
        let checkpoint_seq;
        {
            let mut world = World::new();
            let mut codecs = CodecRegistry::new();
            codecs.register_as::<Score>("score", &mut world).unwrap();

            let mut wal = Wal::create(&wal_dir, &codecs, config.clone()).unwrap();

            // Spawn 10 entities with Score components via WAL
            for i in 0..10 {
                let e = world.alloc_entity();
                let mut cs = minkowski::EnumChangeSet::new();
                cs.spawn_bundle(&mut world, e, (Score(i),)).unwrap();
                wal.append(&cs, &codecs).unwrap();
                cs.apply(&mut world).unwrap();
            }

            // Build and save index
            let mut idx = BTreeIndex::<Score>::new();
            idx.rebuild(&mut world);
            idx.save(&idx_path).unwrap();

            // Save snapshot
            checkpoint_seq = wal.next_seq();
            let snap = Snapshot::new();
            snap.save(&snap_path, &world, &codecs, checkpoint_seq)
                .unwrap();
            wal.acknowledge_snapshot(checkpoint_seq).unwrap();

            // Phase 2: more WAL mutations AFTER checkpoint+index save
            let e = world.alloc_entity();
            let mut cs = minkowski::EnumChangeSet::new();
            cs.spawn_bundle(&mut world, e, (Score(99),)).unwrap();
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world).unwrap();

            // Spawn another post-checkpoint entity
            let e2 = world.alloc_entity();
            let mut cs2 = minkowski::EnumChangeSet::new();
            cs2.spawn_bundle(&mut world, e2, (Score(88),)).unwrap();
            wal.append(&cs2, &codecs).unwrap();
            cs2.apply(&mut world).unwrap();

            // "Crash" — drop everything
        }

        // Phase 3: recover
        {
            let mut codecs = CodecRegistry::new();
            // Must re-register codecs before loading
            let mut tmp_world = World::new();
            codecs
                .register_as::<Score>("score", &mut tmp_world)
                .unwrap();
            drop(tmp_world);

            // Load snapshot
            let snap = Snapshot::new();
            let (mut world, _snap_seq) = snap.load(&snap_path, &codecs).unwrap();

            // Capture tick AFTER snapshot restore, BEFORE WAL replay.
            // This is the sync point for the persistent index — the index
            // was saved alongside the snapshot, so it's consistent with
            // post-restore state. WAL replay will advance ticks beyond
            // this point, and update() will catch exactly that delta.
            let post_restore_tick = world.change_tick();

            // Replay WAL from checkpoint onwards
            let mut wal = Wal::open(&wal_dir, &codecs, config.clone()).unwrap();
            wal.replay_from(checkpoint_seq, &mut world, &codecs)
                .unwrap();

            // Load persistent index and catch up from post-restore tick
            let mut idx = load_btree_index::<Score>(&idx_path, post_restore_tick).unwrap();
            idx.update(&mut world);

            // Verify: all 12 entities present (10 original + Score(99) + Score(88))
            let mut total = 0;
            for i in 0..100 {
                total += idx.get(&Score(i)).len();
            }
            assert_eq!(total, 12, "expected 12 entities in index after recovery");

            // Specifically check the post-checkpoint entities
            assert_eq!(
                idx.get(&Score(99)).len(),
                1,
                "post-checkpoint Score(99) missing"
            );
            assert_eq!(
                idx.get(&Score(88)).len(),
                1,
                "post-checkpoint Score(88) missing"
            );

            // Check original entities survived
            for i in 0..10 {
                assert_eq!(idx.get(&Score(i)).len(), 1, "original Score({i}) missing");
            }
        }
    }

    #[test]
    fn atomic_rename_leaves_no_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atomic.idx");
        write_index_file(&path, b"data").unwrap();
        assert!(path.exists());
        assert!(!path.with_extension("idx.tmp").exists());
    }
}
