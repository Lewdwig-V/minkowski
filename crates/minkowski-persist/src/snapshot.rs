use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use minkowski::{ChangeTick, ComponentId, Entity, EnumChangeSet, World};

use crate::codec::{CodecError, CodecRegistry};
use crate::record::*;

/// Snapshot file magic identifying the v2 format with CRC32 checksums.
///
/// 8 bytes long so that format detection is unambiguous against legacy v1
/// snapshots whose first 8 bytes are a u64 LE payload length.  For this
/// magic to collide with a valid v1 length, the payload would have to be
/// ~3.6 exabytes (`0x4B53_4E41_5032_4B4D` LE), which is not a real file.
/// A 4-byte magic (`b"MKS1"`) was insufficient: payload length
/// `0x31534B4D` (~790 MB) is plausible and would misclassify v1 as v2.
const SNAPSHOT_MAGIC: [u8; 8] = *b"MK2SNAPK";

/// Size of the snapshot envelope header: magic (8) + CRC32 (4) + reserved (4) + length (8).
/// Padded to 24 bytes so the rkyv payload starts at 8-byte alignment.
const SNAPSHOT_HEADER_SIZE: usize = 24;

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("snapshot I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("snapshot codec error: {0}")]
    Codec(#[from] CodecError),
    #[error("snapshot format error: {0}")]
    Format(String),
}

/// Full-world snapshot: serialize all archetype data to disk and reconstruct on load.
///
/// V2 file format: `[magic: 8B "MK2SNAPK"][crc32: 4B LE][reserved: 4B][len: u64 LE][rkyv payload]`.
/// Legacy v1 (`[len: u64 LE][payload]`) is accepted on load but no longer written.
pub struct Snapshot;

impl Snapshot {
    pub fn new() -> Self {
        Self
    }

    /// Save a full world snapshot to disk.
    ///
    /// Writes the v2 envelope (magic, CRC32, length) and rkyv payload to a
    /// temporary file, then atomically renames to the final path. A crash
    /// during the write cannot corrupt an existing snapshot at `path`.
    /// Use `save_to_bytes` when you need the framed bytes in memory
    /// (e.g. for network transfer).
    pub fn save(
        &self,
        path: &Path,
        world: &World,
        codecs: &CodecRegistry,
        wal_seq: u64,
    ) -> Result<SnapshotHeader, SnapshotError> {
        let data = self.build_snapshot_data(world, codecs, wal_seq)?;
        let header = SnapshotHeader {
            wal_seq,
            archetype_count: data.archetypes.len(),
            entity_count: data.archetypes.iter().map(|a| a.entities.len()).sum(),
        };

        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let tmp_path = path.with_extension("snap.tmp");
        let result = (|| -> Result<(), SnapshotError> {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);
            let crc = crc32fast::hash(&payload);
            let len = payload.len() as u64;
            writer.write_all(&SNAPSHOT_MAGIC)?;
            writer.write_all(&crc.to_le_bytes())?;
            writer.write_all(&[0u8; 4])?; // reserved padding for 8-byte alignment
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&payload)?;
            writer.flush()?;
            let file = writer.into_inner().map_err(|e| {
                SnapshotError::Io(std::io::Error::other(format!("flush failed: {e}")))
            })?;
            file.sync_data()?;
            drop(file);
            std::fs::rename(&tmp_path, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp_path);
            result?;
        }

        Ok(header)
    }

    /// Serialize a full world snapshot to bytes.
    ///
    /// Returns `(header, wire_bytes)`. The wire format is
    /// `[magic: 8B][crc32: 4B LE][reserved: 4B][len: u64 LE][rkyv payload: len bytes]`
    /// — identical to the on-disk v2 format. Pass the bytes to
    /// `load_from_bytes` on the receiving side.
    pub fn save_to_bytes(
        &self,
        world: &World,
        codecs: &CodecRegistry,
        wal_seq: u64,
    ) -> Result<(SnapshotHeader, Vec<u8>), SnapshotError> {
        let data = self.build_snapshot_data(world, codecs, wal_seq)?;
        let header = SnapshotHeader {
            wal_seq,
            archetype_count: data.archetypes.len(),
            entity_count: data.archetypes.iter().map(|a| a.entities.len()).sum(),
        };

        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let crc = crc32fast::hash(&payload);
        let len = payload.len() as u64;
        let mut bytes = Vec::with_capacity(SNAPSHOT_HEADER_SIZE + payload.len());
        bytes.extend_from_slice(&SNAPSHOT_MAGIC);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]); // reserved padding for 8-byte alignment
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&payload);

        Ok((header, bytes))
    }

    /// Load a world from a snapshot file.
    /// Returns `(world, wal_seq)` — the `wal_seq` is the sequence number at snapshot time.
    ///
    /// mmaps the snapshot file and validates the archived structure in-place
    /// via `rkyv::access` (no allocation for the envelope). For components
    /// whose archived layout matches native (`#[repr(C)]` POD types on LE),
    /// archived bytes are copied directly into BlobVec without typed
    /// deserialization. Other components fall back to `rkyv::from_bytes`.
    /// The mmap is dropped after load completes.
    ///
    /// Component types are resolved by stable name — registration order does not
    /// need to match the original world. Components in the snapshot schema whose
    /// stable name resolves in the receiver's `CodecRegistry` are remapped
    /// automatically. Unresolved components are filled as raw placeholders.
    pub fn load(&self, path: &Path, codecs: &CodecRegistry) -> Result<(World, u64), SnapshotError> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        self.load_from_bytes(&mmap, codecs)
    }

    /// Reconstruct a world from snapshot bytes received over the wire.
    ///
    /// Accepts both the v2 format (`[magic][crc32][reserved][len][payload]`)
    /// produced by `save_to_bytes` and the legacy v1 format (`[len: u64 LE][payload]`).
    /// Returns `(world, wal_seq)`.
    pub fn load_from_bytes(
        &self,
        bytes: &[u8],
        codecs: &CodecRegistry,
    ) -> Result<(World, u64), SnapshotError> {
        if bytes.len() < SNAPSHOT_HEADER_SIZE {
            return Err(SnapshotError::Format("snapshot too small".to_string()));
        }

        // Detect format: v2 starts with 8-byte SNAPSHOT_MAGIC, v1 starts with a u64 length.
        // The 8-byte magic is unambiguous: interpreting it as a u64 LE yields ~3.6 EB,
        // which is not a valid v1 payload length.
        let (payload_offset, stored_crc) =
            if bytes.len() >= SNAPSHOT_HEADER_SIZE && bytes[..8] == SNAPSHOT_MAGIC {
                let stored_crc = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
                // bytes[12..16] are reserved padding
                (SNAPSHOT_HEADER_SIZE, Some(stored_crc))
            } else {
                // Legacy v1 format (length-only, no magic/CRC). Accept for
                // backward compatibility but skip checksum verification.
                (8, None)
            };

        let len_bytes = if stored_crc.is_some() {
            &bytes[16..24]
        } else {
            &bytes[..8]
        };
        let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
        let end = payload_offset
            .checked_add(len)
            .ok_or_else(|| SnapshotError::Format(format!("invalid payload length: {len}")))?;
        if bytes.len() < end {
            return Err(SnapshotError::Format(format!(
                "snapshot truncated: expected {} payload bytes, got {}",
                len,
                bytes.len() - payload_offset
            )));
        }
        let payload = &bytes[payload_offset..end];

        // Verify CRC32 if present (v2 format).
        if let Some(stored_crc) = stored_crc {
            let actual_crc = crc32fast::hash(payload);
            if actual_crc != stored_crc {
                return Err(SnapshotError::Format(format!(
                    "snapshot checksum mismatch: expected {stored_crc:#010x}, got {actual_crc:#010x}"
                )));
            }
        }

        let archived = rkyv::access::<ArchivedSnapshotData, rkyv::rancor::Error>(payload)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let world = self.restore_world(archived, codecs)?;
        let wal_seq: u64 = archived.wal_seq.into();
        Ok((world, wal_seq))
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn build_snapshot_data(
        &self,
        world: &World,
        codecs: &CodecRegistry,
        wal_seq: u64,
    ) -> Result<SnapshotData, SnapshotError> {
        // Schema — one entry per registered component in the world (not just codecs).
        // This preserves the full ComponentId space so that restore can fill gaps
        // for non-persisted components, preventing ID shifts.
        // Use stable names from codecs when available; fall back to world names.
        let schema: Vec<ComponentSchema> = (0..world.component_count())
            .map(|id| {
                let name = codecs
                    .stable_name(id)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        world
                            .component_name(id)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("__unnamed_{id}"))
                    });
                ComponentSchema {
                    id,
                    name,
                    size: world.component_layout(id).map(|l| l.size()).unwrap_or(0),
                    align: world.component_layout(id).map(|l| l.align()).unwrap_or(1),
                }
            })
            .collect();

        // Allocator state
        let (gens, free) = world.entity_allocator_state();
        let allocator = AllocatorState {
            generations: gens.to_vec(),
            free_list: free.to_vec(),
        };

        // Archetypes
        let mut archetypes = Vec::new();
        for arch_idx in 0..world.archetype_count() {
            let comp_ids = world.archetype_component_ids(arch_idx);
            let entities = world.archetype_entities(arch_idx);

            // Skip empty archetypes (no entities) and the degenerate empty-component archetype
            if entities.is_empty() || comp_ids.is_empty() {
                continue;
            }

            // Verify all components have codecs
            for &comp_id in comp_ids {
                if !codecs.has_codec(comp_id) {
                    return Err(CodecError::UnregisteredComponent(comp_id).into());
                }
            }

            let mut columns = Vec::new();
            for &comp_id in comp_ids {
                let mut values = Vec::new();
                for row in 0..entities.len() {
                    // SAFETY: arch_idx, comp_id, row are all valid — we're iterating
                    // within bounds from the archetype's own metadata.
                    let ptr = unsafe { world.archetype_column_ptr(arch_idx, comp_id, row) };
                    // PERF: Per-row Vec::new() is unavoidable — ColumnData::values
                    // owns Vec<Vec<u8>>. The rkyv to_bytes_in optimization in codec.rs
                    // eliminates the internal double-allocation per value.
                    let mut buf = Vec::new();
                    // SAFETY: ptr points to a valid, aligned component value in the
                    // archetype column.
                    unsafe { codecs.serialize(comp_id, ptr, &mut buf)? };
                    values.push(buf);
                }
                columns.push(ColumnData {
                    component_id: comp_id,
                    values,
                });
            }

            archetypes.push(ArchetypeData {
                component_ids: comp_ids.to_vec(),
                entities: entities.iter().map(|e| e.to_bits()).collect(),
                columns,
            });
        }

        // Sparse components
        let mut sparse = Vec::new();
        for comp_id in world.sparse_component_ids() {
            if !codecs.has_codec(comp_id) {
                return Err(CodecError::UnregisteredComponent(comp_id).into());
            }
            let entries = codecs.serialize_sparse(comp_id, world)?;
            if !entries.is_empty() {
                sparse.push(SparseComponentData {
                    component_id: comp_id,
                    entries,
                });
            }
        }

        Ok(SnapshotData {
            wal_seq,
            schema,
            allocator,
            archetypes,
            sparse,
        })
    }

    fn restore_world(
        &self,
        data: &ArchivedSnapshotData,
        codecs: &CodecRegistry,
    ) -> Result<World, SnapshotError> {
        let mut world = World::new();

        // Register components in schema order, preserving ID slots.
        for entry in data.schema.iter() {
            let name_str = entry.name.as_str();
            if let Some(local_id) = codecs.resolve_name(name_str) {
                codecs.register_one(local_id, &mut world);
            } else {
                let size: usize = u32::from(entry.size) as usize;
                let align: usize = u32::from(entry.align) as usize;
                let layout = std::alloc::Layout::from_size_align(size, align).map_err(|_| {
                    SnapshotError::Format(format!(
                        "invalid layout for component: size={size}, align={align}"
                    ))
                })?;
                let name: &'static str = Box::leak(name_str.to_owned().into_boxed_str());
                world.register_component_raw(name, layout);
            }
        }

        // Build remap from archived schema: sender ComponentId → receiver ComponentId.
        let schema_defs: Vec<ComponentSchema> = data
            .schema
            .iter()
            .filter(|entry| codecs.resolve_name(entry.name.as_str()).is_some())
            .map(|entry| ComponentSchema {
                id: u32::from(entry.id) as usize,
                name: entry.name.as_str().to_owned(),
                size: u32::from(entry.size) as usize,
                align: u32::from(entry.align) as usize,
            })
            .collect();

        let remap = if !schema_defs.is_empty() {
            codecs
                .build_remap(&schema_defs)
                .map_err(|e| SnapshotError::Format(e.to_string()))?
        } else {
            HashMap::new()
        };
        let remap_id = |id: ComponentId| -> ComponentId { remap.get(&id).copied().unwrap_or(id) };

        // Restore archetypes. Components were registered in schema order, so the
        // fresh world's IDs match the sender's IDs. Use sender IDs for record_spawn,
        // but remap to receiver's codec IDs for deserialization.
        // For components where the archived layout matches native (raw_copy_size is
        // Some), copy archived bytes directly — no rkyv::from_bytes.
        for arch_data in data.archetypes.iter() {
            let entity_count = arch_data.entities.len();
            for col in arch_data.columns.iter() {
                if col.values.len() != entity_count {
                    return Err(SnapshotError::Format(format!(
                        "archetype column/entity count mismatch: column has {} values but archetype has {} entities",
                        col.values.len(),
                        entity_count,
                    )));
                }
            }

            let mut changeset = EnumChangeSet::new();

            for (row, entity_bits) in arch_data.entities.iter().enumerate() {
                let entity = Entity::from_bits(u64::from(*entity_bits));
                world.alloc_entity();

                let mut raw_components: Vec<(minkowski::ComponentId, Vec<u8>, std::alloc::Layout)> =
                    Vec::new();
                for col in arch_data.columns.iter() {
                    let sender_id: ComponentId = u32::from(col.component_id) as usize;
                    let codec_id = remap_id(sender_id);
                    let archived_bytes: &[u8] = col.values[row].as_slice();
                    let layout = codecs
                        .layout(codec_id)
                        .ok_or(CodecError::UnregisteredComponent(codec_id))?;

                    let raw = if let Some(size) = codecs.raw_copy_size(codec_id) {
                        if archived_bytes.len() == size {
                            archived_bytes.to_vec()
                        } else {
                            codecs.deserialize(codec_id, archived_bytes)?
                        }
                    } else {
                        codecs.deserialize(codec_id, archived_bytes)?
                    };
                    // Use sender_id for spawn — it matches the fresh world's ID space.
                    raw_components.push((sender_id, raw, layout));
                }

                let ptrs: Vec<_> = raw_components
                    .iter()
                    .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                    .collect();
                changeset.record_spawn(entity, &ptrs);
            }

            changeset.apply(&mut world);
        }

        // Restore sparse components: sender IDs for insertion, codec IDs for deserialization.
        for sparse_data in data.sparse.iter() {
            let sender_id: ComponentId = u32::from(sparse_data.component_id) as usize;
            let codec_id = remap_id(sender_id);
            for entry in sparse_data.entries.iter() {
                let entity_bits: u64 = entry.0.into();
                let bytes: &[u8] = entry.1.as_slice();
                let entity = Entity::from_bits(entity_bits);
                codecs.insert_sparse_raw(codec_id, &mut world, entity, bytes)?;
            }
        }

        // Restore allocator state
        let generations: Vec<u32> = data
            .allocator
            .generations
            .iter()
            .map(|v| u32::from(*v))
            .collect();
        let free_list: Vec<u32> = data
            .allocator
            .free_list
            .iter()
            .map(|v| u32::from(*v))
            .collect();
        world.restore_allocator_state(generations, free_list);

        // Validate that every entity in every archetype has a generation that
        // matches the restored allocator. A corrupt snapshot could contain
        // entities whose generation diverges from the allocator state — this
        // would make is_alive() return false for live entities, silently
        // poisoning the entity lifecycle.
        for arch_idx in 0..world.archetype_count() {
            for &entity in world.archetype_entities(arch_idx) {
                if !world.is_alive(entity) {
                    return Err(SnapshotError::Format(format!(
                        "snapshot corruption: entity (index={}, gen={}) is in an archetype \
                         but the allocator has a different generation",
                        entity.index(),
                        entity.generation(),
                    )));
                }
            }
        }

        Ok(world)
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::new()
    }
}

// ── Incremental snapshots ───────────────────────────────────────────

/// Magic for incremental snapshot files (distinct from full snapshots).
const INCREMENTAL_MAGIC: [u8; 8] = *b"MK2INCRK";

/// Parameters for saving an incremental (delta) snapshot.
///
/// Groups the arguments needed by [`Snapshot::save_incremental`] and
/// [`Snapshot::save_incremental_to_bytes`] into a single struct.
pub struct IncrementalParams<'a> {
    /// The current world to snapshot.
    pub world: &'a World,
    /// Codec registry for serialization.
    pub codecs: &'a CodecRegistry,
    /// WAL sequence number of the base (full) snapshot.
    pub base_wal_seq: u64,
    /// Current WAL sequence number.
    pub wal_seq: u64,
    /// [`ChangeTick`] captured when the base snapshot was taken.
    /// Columns with ticks newer than this are considered dirty.
    pub since_tick: ChangeTick,
    /// The world produced by loading the base (full) snapshot.
    /// Used to detect structural changes (entity count diffs).
    pub base_world: &'a World,
}

fn incremental_header(
    data: &IncrementalSnapshotData,
    params: &IncrementalParams<'_>,
) -> IncrementalSnapshotHeader {
    IncrementalSnapshotHeader {
        base_wal_seq: params.base_wal_seq,
        wal_seq: params.wal_seq,
        dirty_archetype_count: data.dirty_archetypes.len(),
        clean_archetype_count: data.clean_archetypes.len(),
        entity_count: data
            .dirty_archetypes
            .iter()
            .map(|a| a.entities.len())
            .sum::<usize>()
            + data
                .clean_archetypes
                .iter()
                .map(|a| a.entity_count as usize)
                .sum::<usize>(),
    }
}

impl Snapshot {
    /// Save an incremental snapshot containing only archetypes that changed
    /// since `params.since_tick`.
    ///
    /// An archetype is dirty if **any** of the following hold:
    /// - Any column's change tick is newer than `since_tick`
    /// - Its entity count differs from the matching archetype in `base_world`
    /// - It has no matching archetype in `base_world` (new archetype)
    ///
    /// Sparse components are always included in full (they lack per-entry
    /// change ticks). Schema and allocator state are always included.
    ///
    /// The file uses a distinct magic (`MK2INCRK`) so it cannot be confused
    /// with a full snapshot.
    pub fn save_incremental(
        &self,
        path: &Path,
        params: &IncrementalParams<'_>,
    ) -> Result<IncrementalSnapshotHeader, SnapshotError> {
        let data = self.build_incremental_data(params)?;
        let header = incremental_header(&data, params);

        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let tmp_path = path.with_extension("isnap.tmp");
        let result = (|| -> Result<(), SnapshotError> {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);
            let crc = crc32fast::hash(&payload);
            let len = payload.len() as u64;
            writer.write_all(&INCREMENTAL_MAGIC)?;
            writer.write_all(&crc.to_le_bytes())?;
            writer.write_all(&[0u8; 4])?; // reserved padding for 8-byte alignment
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&payload)?;
            writer.flush()?;
            let file = writer.into_inner().map_err(|e| {
                SnapshotError::Io(std::io::Error::other(format!("flush failed: {e}")))
            })?;
            file.sync_data()?;
            drop(file);
            std::fs::rename(&tmp_path, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp_path);
            result?;
        }

        Ok(header)
    }

    /// Serialize an incremental snapshot to bytes (e.g. for network transfer).
    ///
    /// Returns `(header, wire_bytes)` in the same envelope format as
    /// [`save_incremental`](Self::save_incremental). See that method for the
    /// dirty-detection semantics.
    pub fn save_incremental_to_bytes(
        &self,
        params: &IncrementalParams<'_>,
    ) -> Result<(IncrementalSnapshotHeader, Vec<u8>), SnapshotError> {
        let data = self.build_incremental_data(params)?;
        let header = incremental_header(&data, params);

        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let crc = crc32fast::hash(&payload);
        let len = payload.len() as u64;
        let mut bytes = Vec::with_capacity(SNAPSHOT_HEADER_SIZE + payload.len());
        bytes.extend_from_slice(&INCREMENTAL_MAGIC);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&payload);

        Ok((header, bytes))
    }

    /// Load an incremental snapshot file and apply it on top of a base world.
    ///
    /// `base_world` must be the world produced by loading the corresponding
    /// base (full) snapshot. The incremental replaces dirty archetypes and
    /// carries forward clean ones from the base.
    ///
    /// Returns `(world, wal_seq)` — a fresh world reflecting the incremental
    /// state and the WAL sequence at which the incremental was taken.
    pub fn load_incremental(
        &self,
        path: &Path,
        base_world: &World,
        codecs: &CodecRegistry,
    ) -> Result<(World, u64), SnapshotError> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        self.load_incremental_from_bytes(&mmap, base_world, codecs)
    }

    /// Apply incremental snapshot bytes on top of a base world.
    ///
    /// Accepts the wire format produced by [`save_incremental_to_bytes`](Self::save_incremental_to_bytes)
    /// or the on-disk format from [`save_incremental`](Self::save_incremental).
    /// Returns `(world, wal_seq)`.
    pub fn load_incremental_from_bytes(
        &self,
        bytes: &[u8],
        base_world: &World,
        codecs: &CodecRegistry,
    ) -> Result<(World, u64), SnapshotError> {
        if bytes.len() < SNAPSHOT_HEADER_SIZE {
            return Err(SnapshotError::Format(
                "incremental snapshot too small".to_string(),
            ));
        }

        if bytes[..8] != INCREMENTAL_MAGIC {
            return Err(SnapshotError::Format(
                "not an incremental snapshot (wrong magic)".to_string(),
            ));
        }

        let stored_crc = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let end = SNAPSHOT_HEADER_SIZE
            .checked_add(len)
            .ok_or_else(|| SnapshotError::Format(format!("invalid payload length: {len}")))?;
        if bytes.len() < end {
            return Err(SnapshotError::Format(format!(
                "incremental snapshot truncated: expected {} payload bytes, got {}",
                len,
                bytes.len() - SNAPSHOT_HEADER_SIZE
            )));
        }
        let payload = &bytes[SNAPSHOT_HEADER_SIZE..end];

        let actual_crc = crc32fast::hash(payload);
        if actual_crc != stored_crc {
            return Err(SnapshotError::Format(format!(
                "incremental snapshot checksum mismatch: expected {stored_crc:#010x}, got {actual_crc:#010x}"
            )));
        }

        let archived =
            rkyv::access::<ArchivedIncrementalSnapshotData, rkyv::rancor::Error>(payload)
                .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let world = self.restore_incremental(archived, base_world, codecs)?;
        let wal_seq: u64 = archived.wal_seq.into();
        Ok((world, wal_seq))
    }

    // ── Incremental internal helpers ─────────────────────────────────

    fn build_incremental_data(
        &self,
        params: &IncrementalParams<'_>,
    ) -> Result<IncrementalSnapshotData, SnapshotError> {
        let IncrementalParams {
            world,
            codecs,
            base_wal_seq,
            wal_seq,
            since_tick,
            base_world,
        } = params;
        // Schema (full — same as full snapshot)
        let schema: Vec<ComponentSchema> = (0..world.component_count())
            .map(|id| {
                let name = codecs
                    .stable_name(id)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        world
                            .component_name(id)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("__unnamed_{id}"))
                    });
                ComponentSchema {
                    id,
                    name,
                    size: world.component_layout(id).map(|l| l.size()).unwrap_or(0),
                    align: world.component_layout(id).map(|l| l.align()).unwrap_or(1),
                }
            })
            .collect();

        // Allocator state (full)
        let (gens, free) = world.entity_allocator_state();
        let allocator = AllocatorState {
            generations: gens.to_vec(),
            free_list: free.to_vec(),
        };

        // Build a lookup of base archetype entity counts, keyed by sorted
        // component ID set, for structural dirty detection. Structural
        // mutations (despawn, insert-migration, remove) do not bump
        // per-column change ticks, so comparing entity counts against the
        // base world is the only way to detect membership changes.
        let base_entity_counts: HashMap<&[ComponentId], usize> = (0..base_world.archetype_count())
            .filter_map(|idx| {
                let ids = base_world.archetype_component_ids(idx);
                let ents = base_world.archetype_entities(idx);
                if ids.is_empty() || ents.is_empty() {
                    None
                } else {
                    Some((ids, ents.len()))
                }
            })
            .collect();

        // Partition archetypes into dirty vs clean
        let mut dirty_archetypes = Vec::new();
        let mut clean_archetypes = Vec::new();

        for arch_idx in 0..world.archetype_count() {
            let comp_ids = world.archetype_component_ids(arch_idx);
            let entities = world.archetype_entities(arch_idx);

            // Skip empty archetypes and degenerate empty-component archetype
            if entities.is_empty() || comp_ids.is_empty() {
                continue;
            }

            // Verify all components have codecs
            for &comp_id in comp_ids {
                if !codecs.has_codec(comp_id) {
                    return Err(CodecError::UnregisteredComponent(comp_id).into());
                }
            }

            // Check column ticks for value-level changes
            let ticks_dirty = comp_ids.iter().any(|&comp_id| {
                let col_tick = world.archetype_column_changed_tick(arch_idx, comp_id);
                col_tick.to_raw() > since_tick.to_raw()
            });

            // Check entity count against base world for structural changes.
            // If the archetype doesn't exist in base (new archetype), or its
            // entity count differs, it's structurally dirty.
            let structurally_dirty = match base_entity_counts.get(comp_ids) {
                Some(&base_count) => entities.len() != base_count,
                None => true, // new archetype — not in base
            };

            let is_dirty = ticks_dirty || structurally_dirty;

            if is_dirty {
                // Serialize full archetype data (same as full snapshot)
                let mut columns = Vec::new();
                for &comp_id in comp_ids {
                    let mut values = Vec::new();
                    for row in 0..entities.len() {
                        let ptr = unsafe { world.archetype_column_ptr(arch_idx, comp_id, row) };
                        let mut buf = Vec::new();
                        unsafe { codecs.serialize(comp_id, ptr, &mut buf)? };
                        values.push(buf);
                    }
                    columns.push(ColumnData {
                        component_id: comp_id,
                        values,
                    });
                }
                dirty_archetypes.push(ArchetypeData {
                    component_ids: comp_ids.to_vec(),
                    entities: entities.iter().map(|e| e.to_bits()).collect(),
                    columns,
                });
            } else {
                clean_archetypes.push(CleanArchetypeRef {
                    component_ids: comp_ids.to_vec(),
                    entity_count: entities.len() as u32,
                });
            }
        }

        // Sparse components — always included (no per-entry change tracking)
        let mut dirty_sparse = Vec::new();
        for comp_id in world.sparse_component_ids() {
            if !codecs.has_codec(comp_id) {
                return Err(CodecError::UnregisteredComponent(comp_id).into());
            }
            let entries = codecs.serialize_sparse(comp_id, world)?;
            if !entries.is_empty() {
                dirty_sparse.push(SparseComponentData {
                    component_id: comp_id,
                    entries,
                });
            }
        }

        Ok(IncrementalSnapshotData {
            base_wal_seq: *base_wal_seq,
            wal_seq: *wal_seq,
            schema,
            allocator,
            dirty_archetypes,
            clean_archetypes,
            dirty_sparse,
            clean_sparse_ids: Vec::new(),
        })
    }

    fn restore_incremental(
        &self,
        data: &ArchivedIncrementalSnapshotData,
        base_world: &World,
        codecs: &CodecRegistry,
    ) -> Result<World, SnapshotError> {
        let mut world = World::new();

        // Register components in schema order (same as full snapshot)
        for entry in data.schema.iter() {
            let name_str = entry.name.as_str();
            if let Some(local_id) = codecs.resolve_name(name_str) {
                codecs.register_one(local_id, &mut world);
            } else {
                let size: usize = u32::from(entry.size) as usize;
                let align: usize = u32::from(entry.align) as usize;
                let layout = std::alloc::Layout::from_size_align(size, align).map_err(|_| {
                    SnapshotError::Format(format!(
                        "invalid layout for component: size={size}, align={align}"
                    ))
                })?;
                let name: &'static str = Box::leak(name_str.to_owned().into_boxed_str());
                world.register_component_raw(name, layout);
            }
        }

        // Build remap from archived schema
        let schema_defs: Vec<ComponentSchema> = data
            .schema
            .iter()
            .filter(|entry| codecs.resolve_name(entry.name.as_str()).is_some())
            .map(|entry| ComponentSchema {
                id: u32::from(entry.id) as usize,
                name: entry.name.as_str().to_owned(),
                size: u32::from(entry.size) as usize,
                align: u32::from(entry.align) as usize,
            })
            .collect();

        let remap = if !schema_defs.is_empty() {
            codecs
                .build_remap(&schema_defs)
                .map_err(|e| SnapshotError::Format(e.to_string()))?
        } else {
            HashMap::new()
        };
        let remap_id = |id: ComponentId| -> ComponentId { remap.get(&id).copied().unwrap_or(id) };

        // --- Restore clean archetypes from base world ---
        // For each clean archetype ref, find the matching archetype in the base
        // world and copy its entities/columns via raw bytes (no serialization).
        for clean_ref in data.clean_archetypes.iter() {
            let ref_comp_ids: Vec<ComponentId> = clean_ref
                .component_ids
                .iter()
                .map(|id| u32::from(*id) as usize)
                .collect();

            // Find matching archetype in base_world by component ID set
            let base_arch_idx = (0..base_world.archetype_count())
                .find(|&idx| base_world.archetype_component_ids(idx) == ref_comp_ids.as_slice())
                .ok_or_else(|| {
                    SnapshotError::Format(format!(
                        "clean archetype with components {:?} not found in base world",
                        ref_comp_ids,
                    ))
                })?;

            let base_entities = base_world.archetype_entities(base_arch_idx);
            let expected_count = u32::from(clean_ref.entity_count) as usize;
            if base_entities.len() != expected_count {
                return Err(SnapshotError::Format(format!(
                    "clean archetype entity count mismatch: expected {}, base has {}",
                    expected_count,
                    base_entities.len(),
                )));
            }

            let mut changeset = EnumChangeSet::new();
            for (row, &entity) in base_entities.iter().enumerate() {
                world.alloc_entity();

                let mut raw_components: Vec<(ComponentId, Vec<u8>, std::alloc::Layout)> =
                    Vec::new();
                for &comp_id in &ref_comp_ids {
                    // Use the restored world's layout (sender-side IDs, set up
                    // during schema registration above). Do NOT use codecs.layout()
                    // here — comp_id is a sender-side ID but the codec registry is
                    // keyed by receiver-side IDs. When registration order differs,
                    // codecs.layout(sender_id) would resolve the wrong component's
                    // layout, causing incorrect byte counts in copy_nonoverlapping.
                    let layout = world.component_layout(comp_id).ok_or_else(|| {
                        SnapshotError::Format(format!(
                            "no layout for component {comp_id} in clean archetype"
                        ))
                    })?;

                    // Copy raw bytes from base world's column
                    let ptr =
                        unsafe { base_world.archetype_column_ptr(base_arch_idx, comp_id, row) };
                    let mut raw = vec![0u8; layout.size()];
                    unsafe { std::ptr::copy_nonoverlapping(ptr, raw.as_mut_ptr(), layout.size()) };
                    raw_components.push((comp_id, raw, layout));
                }

                let ptrs: Vec<_> = raw_components
                    .iter()
                    .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                    .collect();
                changeset.record_spawn(entity, &ptrs);
            }
            changeset.apply(&mut world);
        }

        // --- Restore dirty archetypes from incremental data ---
        // Same logic as full snapshot restore.
        for arch_data in data.dirty_archetypes.iter() {
            let entity_count = arch_data.entities.len();
            for col in arch_data.columns.iter() {
                if col.values.len() != entity_count {
                    return Err(SnapshotError::Format(format!(
                        "archetype column/entity count mismatch: column has {} values but archetype has {} entities",
                        col.values.len(),
                        entity_count,
                    )));
                }
            }

            let mut changeset = EnumChangeSet::new();
            for (row, entity_bits) in arch_data.entities.iter().enumerate() {
                let entity = Entity::from_bits(u64::from(*entity_bits));
                world.alloc_entity();

                let mut raw_components: Vec<(ComponentId, Vec<u8>, std::alloc::Layout)> =
                    Vec::new();
                for col in arch_data.columns.iter() {
                    let sender_id: ComponentId = u32::from(col.component_id) as usize;
                    let codec_id = remap_id(sender_id);
                    let archived_bytes: &[u8] = col.values[row].as_slice();
                    let layout = codecs
                        .layout(codec_id)
                        .ok_or(CodecError::UnregisteredComponent(codec_id))?;

                    let raw = if let Some(size) = codecs.raw_copy_size(codec_id) {
                        if archived_bytes.len() == size {
                            archived_bytes.to_vec()
                        } else {
                            codecs.deserialize(codec_id, archived_bytes)?
                        }
                    } else {
                        codecs.deserialize(codec_id, archived_bytes)?
                    };
                    raw_components.push((sender_id, raw, layout));
                }

                let ptrs: Vec<_> = raw_components
                    .iter()
                    .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                    .collect();
                changeset.record_spawn(entity, &ptrs);
            }
            changeset.apply(&mut world);
        }

        // Restore sparse components from incremental data
        for sparse_data in data.dirty_sparse.iter() {
            let sender_id: ComponentId = u32::from(sparse_data.component_id) as usize;
            let codec_id = remap_id(sender_id);
            for entry in sparse_data.entries.iter() {
                let entity_bits: u64 = entry.0.into();
                let bytes: &[u8] = entry.1.as_slice();
                let entity = Entity::from_bits(entity_bits);
                codecs.insert_sparse_raw(codec_id, &mut world, entity, bytes)?;
            }
        }

        // Restore allocator state
        let generations: Vec<u32> = data
            .allocator
            .generations
            .iter()
            .map(|v| u32::from(*v))
            .collect();
        let free_list: Vec<u32> = data
            .allocator
            .free_list
            .iter()
            .map(|v| u32::from(*v))
            .collect();
        world.restore_allocator_state(generations, free_list);

        // Validate entity lifecycle (same as full snapshot)
        for arch_idx in 0..world.archetype_count() {
            for &entity in world.archetype_entities(arch_idx) {
                if !world.is_alive(entity) {
                    return Err(SnapshotError::Format(format!(
                        "incremental snapshot corruption: entity (index={}, gen={}) is in an \
                         archetype but the allocator has a different generation",
                        entity.index(),
                        entity.generation(),
                    )));
                }
            }
        }

        Ok(world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("test.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        world.spawn((Pos { x: 5.0, y: 6.0 }, Vel { dx: 7.0, dy: 8.0 }));

        let snap = Snapshot::new();
        let header = snap.save(&snap_path, &world, &codecs, 42).unwrap();
        assert_eq!(header.entity_count, 2);
        assert_eq!(header.wal_seq, 42);

        let (mut world2, wal_seq) = snap.load(&snap_path, &codecs).unwrap();
        assert_eq!(wal_seq, 42);

        let mut positions: Vec<(f32, f32)> =
            world2.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(positions, vec![(1.0, 2.0), (5.0, 6.0)]);

        let mut velocities: Vec<(f32, f32)> = world2
            .query::<(&Vel,)>()
            .map(|v| (v.0.dx, v.0.dy))
            .collect();
        velocities.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(velocities, vec![(3.0, 4.0), (7.0, 8.0)]);
    }

    #[test]
    fn bytes_round_trip() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        world.spawn((Pos { x: 5.0, y: 6.0 },));

        let snap = Snapshot::new();
        let (header, bytes) = snap.save_to_bytes(&world, &codecs, 99).unwrap();
        assert_eq!(header.entity_count, 2);
        assert_eq!(header.wal_seq, 99);

        let (mut world2, wal_seq) = snap.load_from_bytes(&bytes, &codecs).unwrap();
        assert_eq!(wal_seq, 99);
        assert_eq!(world2.query::<(&Pos,)>().count(), 2);
        assert_eq!(world2.query::<(&Vel,)>().count(), 1);
    }

    #[test]
    fn empty_world_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("empty.snap");

        let world = World::new();
        let codecs = CodecRegistry::new();

        let snap = Snapshot::new();
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        let (_, seq) = snap.load(&snap_path, &codecs).unwrap();
        assert_eq!(seq, 0);
    }

    #[test]
    fn multiple_archetypes() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("multi.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        // Archetype 1: (Pos, Vel)
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
        // Archetype 2: (Pos) only
        world.spawn((Pos { x: 3.0, y: 4.0 },));

        let snap = Snapshot::new();
        let header = snap.save(&snap_path, &world, &codecs, 0).unwrap();
        assert_eq!(header.entity_count, 2);
        assert_eq!(header.archetype_count, 2);

        let (mut world2, _) = snap.load(&snap_path, &codecs).unwrap();
        assert_eq!(world2.query::<(&Pos,)>().count(), 2);
        assert_eq!(world2.query::<(&Vel,)>().count(), 1);
    }

    #[test]
    fn missing_codec_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("missing.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        // Spawn entity with Vel (which has no codec registered)
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));

        let snap = Snapshot::new();
        let result = snap.save(&snap_path, &world, &codecs, 0);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("sparse.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let _e2 = world.spawn((Pos { x: 3.0, y: 4.0 },));

        world.insert_sparse::<Vel>(e1, Vel { dx: 10.0, dy: 20.0 });

        let snap = Snapshot::new();
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        let (mut world2, _) = snap.load(&snap_path, &codecs).unwrap();

        assert_eq!(world2.query::<(&Pos,)>().count(), 2);

        let vel_id = world2.component_id::<Vel>().unwrap();
        let sparse_entries: Vec<_> = world2.iter_sparse::<Vel>(vel_id).unwrap().collect();
        assert_eq!(sparse_entries.len(), 1);
        assert_eq!(sparse_entries[0].1.dx, 10.0);
        assert_eq!(sparse_entries[0].1.dy, 20.0);
    }

    #[test]
    fn sparse_only_entities() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("sparse_only.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.insert_sparse::<Vel>(e1, Vel { dx: 5.0, dy: 6.0 });

        let snap = Snapshot::new();
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        let (mut world2, _) = snap.load(&snap_path, &codecs).unwrap();

        assert_eq!(world2.query::<(&Pos,)>().count(), 1);

        let vel_id = world2.component_id::<Vel>().unwrap();
        let entries: Vec<_> = world2.iter_sparse::<Vel>(vel_id).unwrap().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1.dx, 5.0);
        assert_eq!(entries[0].1.dy, 6.0);
    }

    #[test]
    fn snapshot_plus_wal_recovery() {
        use crate::wal::{Wal, WalConfig};

        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("recovery.snap");
        let wal_path = dir.path().join("recovery.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
        world.spawn((Pos { x: 3.0, y: 4.0 }, Vel { dx: 0.3, dy: 0.4 }));

        let mut wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let snap = Snapshot::new();
        let _header = snap
            .save(&snap_path, &world, &codecs, wal.next_seq())
            .unwrap();

        // More mutations after snapshot
        let e3 = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(
            &mut world,
            e3,
            (Pos { x: 5.0, y: 6.0 }, Vel { dx: 0.5, dy: 0.6 }),
        );
        wal.append(&cs, &codecs).unwrap();
        let _reverse = cs.apply(&mut world);

        let entities: Vec<_> = world
            .query::<(minkowski::Entity, &Pos)>()
            .map(|(e, p)| {
                (
                    e,
                    Pos {
                        x: p.x + 100.0,
                        y: p.y,
                    },
                )
            })
            .collect();
        let mut cs2 = EnumChangeSet::new();
        for (e, new_pos) in &entities {
            cs2.insert::<Pos>(&mut world, *e, *new_pos);
        }
        wal.append(&cs2, &codecs).unwrap();
        let _reverse2 = cs2.apply(&mut world);

        // Recover from snapshot + WAL
        let mut load_codecs = CodecRegistry::new();
        let mut load_world_tmp = World::new();
        load_codecs.register::<Pos>(&mut load_world_tmp);
        load_codecs.register::<Vel>(&mut load_world_tmp);

        let (mut recovered, snap_seq) = snap.load(&snap_path, &load_codecs).unwrap();
        let last_seq = wal
            .replay_from(snap_seq, &mut recovered, &load_codecs)
            .unwrap();

        assert_eq!(recovered.query::<(&Pos,)>().count(), 3);
        assert!(last_seq >= 1);
    }

    #[test]
    fn empty_sparse_not_serialized() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("no_sparse.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = Snapshot::new();
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        let (mut world2, _) = snap.load(&snap_path, &codecs).unwrap();
        assert_eq!(world2.query::<(&Pos,)>().count(), 1);
        assert!(world2.sparse_component_ids().is_empty());
    }

    #[test]
    fn non_persisted_component_preserves_id_space() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("gap.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();

        #[derive(Clone, Copy)]
        #[allow(dead_code)]
        struct Hidden(u32);
        world.register_component::<Hidden>();

        codecs.register::<Pos>(&mut world);
        let pos_id = world.component_id::<Pos>().unwrap();
        assert_eq!(pos_id, 1, "Pos should have id=1 (Hidden took id=0)");

        world.spawn((Pos { x: 42.0, y: 99.0 },));

        let snap = Snapshot::new();
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        let (mut world2, _) = snap.load(&snap_path, &codecs).unwrap();

        let restored_pos_id = world2.component_id::<Pos>().unwrap();
        assert_eq!(
            restored_pos_id, pos_id,
            "Pos ComponentId must match after restore (gap filled for Hidden)"
        );

        let positions: Vec<_> = world2.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(42.0, 99.0)]);
    }

    // ── Error path tests ────────────────────────────────────────────

    #[test]
    fn load_corrupted_payload() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("corrupt.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = Snapshot::new();
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        // Corrupt the payload (flip bytes after the envelope header)
        let mut data = std::fs::read(&snap_path).unwrap();
        for byte in data[SNAPSHOT_HEADER_SIZE..].iter_mut().take(16) {
            *byte ^= 0xFF;
        }
        std::fs::write(&snap_path, &data).unwrap();

        let result = snap.load(&snap_path, &codecs);
        assert!(result.is_err());
    }

    #[test]
    fn load_file_too_small() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("tiny.snap");

        std::fs::write(&snap_path, [0u8; 4]).unwrap();

        let snap = Snapshot::new();
        let codecs = CodecRegistry::new();
        let result = snap.load(&snap_path, &codecs);
        assert!(matches!(result, Err(SnapshotError::Format(_))));
    }

    #[test]
    fn load_truncated_payload() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("truncated.snap");

        // Write a length prefix claiming 1000 bytes but only 10 bytes of payload
        let mut data = Vec::new();
        data.extend_from_slice(&1000u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 10]);
        std::fs::write(&snap_path, &data).unwrap();

        let snap = Snapshot::new();
        let codecs = CodecRegistry::new();
        let result = snap.load(&snap_path, &codecs);
        assert!(matches!(result, Err(SnapshotError::Format(_))));
    }

    #[test]
    fn load_nonexistent_file() {
        let snap = Snapshot::new();
        let codecs = CodecRegistry::new();
        let result = snap.load(std::path::Path::new("/nonexistent"), &codecs);
        assert!(matches!(result, Err(SnapshotError::Io(_))));
    }

    // ── Cross-process remap tests ────────────────────────────────────

    #[test]
    fn snapshot_cross_process_different_registration_order() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("cross.snap");

        // "Process A": Pos first, then Vel
        let mut world_a = World::new();
        let mut codecs_a = CodecRegistry::new();
        codecs_a.register_as::<Pos>("pos", &mut world_a);
        codecs_a.register_as::<Vel>("vel", &mut world_a);

        world_a.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));

        let snap = Snapshot::new();
        snap.save(&snap_path, &world_a, &codecs_a, 0).unwrap();

        // "Process B": opposite order
        let mut world_b_tmp = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b.register_as::<Vel>("vel", &mut world_b_tmp);
        codecs_b.register_as::<Pos>("pos", &mut world_b_tmp);

        let (mut world_b, _) = snap.load(&snap_path, &codecs_b).unwrap();

        let positions: Vec<(f32, f32)> =
            world_b.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(1.0, 2.0)]);

        let velocities: Vec<(f32, f32)> = world_b
            .query::<(&Vel,)>()
            .map(|v| (v.0.dx, v.0.dy))
            .collect();
        assert_eq!(velocities, vec![(3.0, 4.0)]);
    }

    #[test]
    fn snapshot_cross_process_sparse_different_registration_order() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("cross_sparse.snap");

        let mut world_a = World::new();
        let mut codecs_a = CodecRegistry::new();
        codecs_a.register_as::<Pos>("pos", &mut world_a);
        codecs_a.register_as::<Vel>("vel", &mut world_a);

        let e = world_a.spawn((Pos { x: 1.0, y: 2.0 },));
        world_a.insert_sparse::<Vel>(e, Vel { dx: 10.0, dy: 20.0 });

        let snap = Snapshot::new();
        snap.save(&snap_path, &world_a, &codecs_a, 0).unwrap();

        // Opposite order
        let mut world_b_tmp = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b.register_as::<Vel>("vel", &mut world_b_tmp);
        codecs_b.register_as::<Pos>("pos", &mut world_b_tmp);

        let (mut world_b, _) = snap.load(&snap_path, &codecs_b).unwrap();

        let positions: Vec<(f32, f32)> =
            world_b.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(1.0, 2.0)]);

        let vel_id = world_b.component_id::<Vel>().unwrap();
        let sparse: Vec<(f32, f32)> = world_b
            .iter_sparse::<Vel>(vel_id)
            .unwrap()
            .map(|(_, v)| (v.dx, v.dy))
            .collect();
        assert_eq!(sparse, vec![(10.0, 20.0)]);
    }

    // ── v2 format / CRC tests ───────────────────────────────────────

    #[test]
    fn snapshot_v2_has_magic_header() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("magic.snap");

        let world = World::new();
        let codecs = CodecRegistry::new();

        let snap = Snapshot::new();
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        let data = std::fs::read(&snap_path).unwrap();
        assert_eq!(
            &data[..8],
            b"MK2SNAPK",
            "snapshot must start with MK2SNAPK magic"
        );
    }

    #[test]
    fn snapshot_crc_mismatch_detected() {
        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("crc_bad.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = Snapshot::new();
        snap.save(&snap_path, &world, &codecs, 0).unwrap();

        // Flip a byte in the CRC field (bytes 8..12) to force mismatch
        let mut data = std::fs::read(&snap_path).unwrap();
        data[8] ^= 0xFF;
        std::fs::write(&snap_path, &data).unwrap();

        let result = snap.load(&snap_path, &codecs);
        let err = result.err().expect("should fail with CRC mismatch");
        let msg = format!("{err}");
        assert!(
            msg.contains("checksum mismatch"),
            "error should mention checksum: {msg}"
        );
    }

    #[test]
    fn snapshot_bytes_crc_mismatch_detected() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = Snapshot::new();
        let (_, mut bytes) = snap.save_to_bytes(&world, &codecs, 0).unwrap();

        // Corrupt the payload (after SNAPSHOT_HEADER_SIZE)
        bytes[SNAPSHOT_HEADER_SIZE] ^= 0xFF;

        let result = snap.load_from_bytes(&bytes, &codecs);
        let err = result.err().expect("should fail with CRC mismatch");
        let msg = format!("{err}");
        assert!(
            msg.contains("checksum mismatch"),
            "error should mention checksum: {msg}"
        );
    }

    #[test]
    fn snapshot_v1_legacy_still_loads() {
        // Construct a v1-format snapshot (no magic, no CRC — just len+payload)
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        world.spawn((Pos { x: 42.0, y: 99.0 },));

        let snap = Snapshot::new();
        let (_, v2_bytes) = snap.save_to_bytes(&world, &codecs, 7).unwrap();

        // Strip the v2 envelope and rebuild v1 format
        let payload = &v2_bytes[SNAPSHOT_HEADER_SIZE..];
        let len = payload.len() as u64;
        let mut v1_bytes = Vec::with_capacity(8 + payload.len());
        v1_bytes.extend_from_slice(&len.to_le_bytes());
        v1_bytes.extend_from_slice(payload);

        let (mut world2, seq) = snap.load_from_bytes(&v1_bytes, &codecs).unwrap();
        assert_eq!(seq, 7);
        let positions: Vec<(f32, f32)> =
            world2.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions, vec![(42.0, 99.0)]);
    }

    #[test]
    fn v1_whose_length_prefix_starts_with_old_4byte_magic_loads_correctly() {
        // Regression: the old 4-byte magic b"MKS1" = 0x31534B4D LE ≈ 790 MB.
        // A v1 snapshot with that payload length would have been misclassified
        // as v2 by the old detection logic. With 8-byte magic this is impossible
        // because no valid payload length has ASCII in its high 4 bytes.
        //
        // We can't construct an 800 MB payload, but we CAN construct a v1-format
        // buffer whose first 4 bytes happen to be b"MKS1" (by choosing an
        // appropriate length) and verify it falls through to the v1 path.
        // The payload won't be valid rkyv, but we should get a Format error
        // (not a checksum mismatch — which would mean v2 was incorrectly selected).
        let fake_len: u64 = u64::from_le_bytes(*b"MKS1\x00\x00\x00\x00");
        let mut v1_bytes = Vec::new();
        v1_bytes.extend_from_slice(&fake_len.to_le_bytes());
        // Append a small dummy payload (won't match the declared length, but
        // the detection logic runs before length validation in v1 path).
        v1_bytes.extend_from_slice(&[0u8; 64]);

        let snap = Snapshot::new();
        let codecs = CodecRegistry::new();
        let result = snap.load_from_bytes(&v1_bytes, &codecs);
        // Should fail with a truncation or format error — NOT a CRC mismatch,
        // which would indicate the v2 path was incorrectly taken.
        let err = result.err().expect("should fail on truncated v1 data");
        let msg = format!("{err}");
        assert!(
            !msg.contains("checksum mismatch"),
            "v1 snapshot with MKS1 length prefix was misclassified as v2: {msg}"
        );
    }

    #[test]
    fn snapshot_round_trip_passes_generation_validation() {
        // End-to-end: save and reload — the generation high-water-mark
        // assert should pass for a well-formed snapshot.
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        // Spawn, despawn, respawn to create non-trivial generations.
        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let _e2 = world.spawn((Pos { x: 3.0, y: 4.0 },));
        world.despawn(e1);
        let _e3 = world.spawn((Pos { x: 5.0, y: 6.0 },)); // reuses index 0, gen 1

        let snap = Snapshot::new();
        let (_, bytes) = snap.save_to_bytes(&world, &codecs, 0).unwrap();

        // Should not panic — allocator generations match archetype entities.
        let (mut world2, _) = snap.load_from_bytes(&bytes, &codecs).unwrap();

        let positions: Vec<(f32, f32)> =
            world2.query::<(&Pos,)>().map(|p| (p.0.x, p.0.y)).collect();
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn snapshot_generation_mismatch_returns_error() {
        // Construct a snapshot where the allocator generation for entity 0 is
        // 99, but the archetype data contains entity(0, gen=0). This simulates
        // a corrupt or bit-rotted snapshot file. load_from_bytes must return
        // Err(SnapshotError::Format), not panic.
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        world.spawn((Pos { x: 1.0, y: 2.0 },)); // entity(index=0, gen=0)

        let snap = Snapshot::new();
        let (_, bytes) = snap.save_to_bytes(&world, &codecs, 0).unwrap();

        // The rkyv payload contains the allocator's generations vec. The
        // generation for index 0 is 0u32 LE = [0x00, 0x00, 0x00, 0x00].
        // Corrupt it by flipping generation bytes in the payload, then
        // recompute the CRC so the checksum passes.
        // Find the generation value (0u32) in the allocator section.
        // The allocator data is near the end of the payload. Search backwards
        // for a 4-byte zero run that, when changed, causes a generation mismatch.
        let mut tampered = bytes.clone();
        let payload_start = SNAPSHOT_HEADER_SIZE;
        let payload_len = tampered.len() - payload_start;

        // Try corrupting 4-byte aligned positions from the end of the payload.
        // One of them will be the allocator's generation for index 0.
        let mut found_corruption = false;
        for offset in (0..payload_len.saturating_sub(3)).rev() {
            let abs = payload_start + offset;
            if tampered[abs..abs + 4] == [0, 0, 0, 0] {
                // Flip to gen=99 and recompute CRC.
                tampered[abs] = 99;
                let new_crc = crc32fast::hash(&tampered[payload_start..]);
                tampered[8..12].copy_from_slice(&new_crc.to_le_bytes());

                let result = snap.load_from_bytes(&tampered, &codecs);
                if let Err(e) = result {
                    let msg = format!("{e}");
                    if msg.contains("snapshot corruption") {
                        found_corruption = true;
                        break;
                    }
                }
                // Restore and try next position.
                tampered[abs] = 0;
            }
        }

        assert!(
            found_corruption,
            "failed to trigger generation mismatch by corrupting payload bytes"
        );
    }

    // ── Incremental snapshot tests ──────────────────────────────────

    #[test]
    fn incremental_round_trip_no_changes() {
        // When nothing changed since the base tick, all archetypes are clean.
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");
        let inc_path = dir.path().join("inc.isnap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        world.spawn((Pos { x: 5.0, y: 6.0 },));

        let snap = Snapshot::new();
        let since_tick = world.change_tick();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        // No mutations — incremental should have zero dirty archetypes
        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let header = snap.save_incremental(&inc_path, &params).unwrap();
        assert_eq!(header.dirty_archetype_count, 0);
        assert_eq!(header.clean_archetype_count, 2);
        assert_eq!(header.entity_count, 2);

        let (mut restored, seq) = snap
            .load_incremental(&inc_path, &base_world, &codecs)
            .unwrap();
        assert_eq!(seq, 1);

        let mut positions: Vec<(f32, f32)> = restored
            .query::<(&Pos,)>()
            .map(|p| (p.0.x, p.0.y))
            .collect();
        positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(positions, vec![(1.0, 2.0), (5.0, 6.0)]);
        assert_eq!(restored.query::<(&Vel,)>().count(), 1);
    }

    #[test]
    fn incremental_round_trip_partial_dirty() {
        // Mutate one archetype, leave the other unchanged.
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");
        let inc_path = dir.path().join("inc.isnap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        // Archetype 1: (Pos, Vel)
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        // Archetype 2: (Pos) only
        world.spawn((Pos { x: 5.0, y: 6.0 },));

        let snap = Snapshot::new();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        let since_tick = world.change_tick();

        // Mutate only the (Pos, Vel) archetype via query
        for (pos, _vel) in world.query::<(&mut Pos, &Vel)>() {
            // SAFETY: the query yields mutable refs
            let pos = pos as *const Pos as *mut Pos;
            unsafe { (*pos).x = 100.0 };
        }

        // Save incremental
        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 2,
            since_tick,
            base_world: &base_world,
        };
        let header = snap.save_incremental(&inc_path, &params).unwrap();
        // (Pos, Vel) archetype is dirty, (Pos) archetype is clean
        assert_eq!(header.dirty_archetype_count, 1);
        assert_eq!(header.clean_archetype_count, 1);

        // Load and verify
        let (mut restored, seq) = snap
            .load_incremental(&inc_path, &base_world, &codecs)
            .unwrap();
        assert_eq!(seq, 2);

        let mut positions: Vec<(f32, f32)> = restored
            .query::<(&Pos,)>()
            .map(|p| (p.0.x, p.0.y))
            .collect();
        positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Pos in (Pos, Vel) archetype was updated to x=100.0
        // Pos in (Pos) archetype is unchanged at x=5.0
        assert_eq!(positions, vec![(5.0, 6.0), (100.0, 2.0)]);
    }

    #[test]
    fn incremental_all_dirty_matches_full() {
        // When since_tick is default (0), everything is dirty — should produce
        // the same world as a full snapshot.
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");
        let inc_path = dir.path().join("inc.isnap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.spawn((Pos { x: 3.0, y: 4.0 },));

        let snap = Snapshot::new();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        // Use default tick (0) — all archetypes are dirty
        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick: minkowski::ChangeTick::default(),
            base_world: &base_world,
        };
        let header = snap.save_incremental(&inc_path, &params).unwrap();
        assert_eq!(header.dirty_archetype_count, 1);
        assert_eq!(header.clean_archetype_count, 0);

        let (mut restored, _) = snap
            .load_incremental(&inc_path, &base_world, &codecs)
            .unwrap();
        let mut positions: Vec<(f32, f32)> = restored
            .query::<(&Pos,)>()
            .map(|p| (p.0.x, p.0.y))
            .collect();
        positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(positions, vec![(1.0, 2.0), (3.0, 4.0)]);
    }

    #[test]
    fn incremental_bytes_round_trip() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = Snapshot::new();
        let since_tick = world.change_tick();

        // Full snapshot for base
        let (_, base_bytes) = snap.save_to_bytes(&world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load_from_bytes(&base_bytes, &codecs).unwrap();

        // Incremental (no changes)
        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let (header, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();
        assert_eq!(header.dirty_archetype_count, 0);

        let (mut restored, seq) = snap
            .load_incremental_from_bytes(&inc_bytes, &base_world, &codecs)
            .unwrap();
        assert_eq!(seq, 1);
        assert_eq!(restored.query::<(&Pos,)>().count(), 1);
    }

    #[test]
    fn incremental_has_distinct_magic() {
        let dir = tempfile::tempdir().unwrap();
        let inc_path = dir.path().join("inc.isnap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let base_world = World::new();
        let snap = Snapshot::new();
        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 0,
            since_tick: minkowski::ChangeTick::default(),
            base_world: &base_world,
        };
        snap.save_incremental(&inc_path, &params).unwrap();

        let data = std::fs::read(&inc_path).unwrap();
        assert_eq!(
            &data[..8],
            b"MK2INCRK",
            "incremental snapshot must start with MK2INCRK magic"
        );
    }

    #[test]
    fn incremental_crc_mismatch_detected() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let base_world = World::new();
        let snap = Snapshot::new();
        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 0,
            since_tick: minkowski::ChangeTick::default(),
            base_world: &base_world,
        };
        let (_, mut bytes) = snap.save_incremental_to_bytes(&params).unwrap();

        // Corrupt payload
        bytes[SNAPSHOT_HEADER_SIZE] ^= 0xFF;

        let base_world = World::new();
        let result = snap.load_incremental_from_bytes(&bytes, &base_world, &codecs);
        let err = result.err().expect("should fail with CRC mismatch");
        let msg = format!("{err}");
        assert!(msg.contains("checksum mismatch"), "error: {msg}");
    }

    #[test]
    fn incremental_wrong_magic_rejected() {
        // A full snapshot cannot be loaded as incremental
        let world = World::new();
        let codecs = CodecRegistry::new();

        let snap = Snapshot::new();
        let (_, full_bytes) = snap.save_to_bytes(&world, &codecs, 0).unwrap();

        let result = snap.load_incremental_from_bytes(&full_bytes, &world, &codecs);
        assert!(matches!(result, Err(SnapshotError::Format(_))));
    }

    #[test]
    fn incremental_with_new_entities() {
        // Spawn new entities after the base snapshot, then take an incremental.
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = Snapshot::new();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        let since_tick = world.change_tick();

        // Spawn a new entity in a NEW archetype (Pos, Vel)
        world.spawn((Pos { x: 10.0, y: 20.0 }, Vel { dx: 1.0, dy: 2.0 }));

        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let (header, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();
        // New archetype is dirty (it was created after since_tick)
        assert!(header.dirty_archetype_count >= 1);

        let (mut restored_world, _) = snap
            .load_incremental_from_bytes(&inc_bytes, &base_world, &codecs)
            .unwrap();
        assert_eq!(restored_world.query::<(&Pos,)>().count(), 2);
        assert_eq!(restored_world.query::<(&Vel,)>().count(), 1);
    }

    #[test]
    fn incremental_sparse_components_included() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = Snapshot::new();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        let since_tick = world.change_tick();

        // Add sparse after the base snapshot
        world.insert_sparse::<Vel>(e, Vel { dx: 10.0, dy: 20.0 });

        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let (_, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();

        let (restored, _) = snap
            .load_incremental_from_bytes(&inc_bytes, &base_world, &codecs)
            .unwrap();

        let vel_id = restored.component_id::<Vel>().unwrap();
        let sparse: Vec<_> = restored
            .iter_sparse::<Vel>(vel_id)
            .unwrap()
            .map(|(_, v)| (v.dx, v.dy))
            .collect();
        assert_eq!(sparse, vec![(10.0, 20.0)]);
    }

    #[test]
    fn incremental_header_tracks_sequences() {
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let base_world = World::new();
        let snap = Snapshot::new();
        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 42,
            wal_seq: 99,
            since_tick: minkowski::ChangeTick::default(),
            base_world: &base_world,
        };
        let (header, _) = snap.save_incremental_to_bytes(&params).unwrap();

        assert_eq!(header.base_wal_seq, 42);
        assert_eq!(header.wal_seq, 99);
    }

    #[test]
    fn incremental_size_smaller_when_partial_dirty() {
        // Verify that incremental is smaller than full when some archetypes
        // are clean.
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        // Create many entities in the (Pos) archetype
        for i in 0..100 {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32,
            },));
        }
        // One entity in (Pos, Vel)
        world.spawn((Pos { x: 0.0, y: 0.0 }, Vel { dx: 1.0, dy: 1.0 }));

        let snap = Snapshot::new();
        let since_tick = world.change_tick();

        // Full snapshot size
        let (_, full_bytes) = snap.save_to_bytes(&world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load_from_bytes(&full_bytes, &codecs).unwrap();

        // Mutate only the (Pos, Vel) entity
        for (pos, _vel) in world.query::<(&mut Pos, &Vel)>() {
            let pos = pos as *const Pos as *mut Pos;
            unsafe { (*pos).x = 999.0 };
        }

        // Incremental snapshot size
        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let (header, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();

        assert_eq!(header.dirty_archetype_count, 1);
        assert_eq!(header.clean_archetype_count, 1);
        assert!(
            inc_bytes.len() < full_bytes.len(),
            "incremental ({} bytes) should be smaller than full ({} bytes)",
            inc_bytes.len(),
            full_bytes.len(),
        );
    }

    #[test]
    fn incremental_cross_process_different_registration_order() {
        // Regression: when sender and receiver register components in different
        // order, the clean-archetype restore path used codecs.layout(sender_id)
        // which resolved the wrong component in the receiver's registry. This
        // caused incorrect byte counts in copy_nonoverlapping.
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");

        // "Process A" (sender): Pos first, then Vel
        let mut world_a = World::new();
        let mut codecs_a = CodecRegistry::new();
        codecs_a.register_as::<Pos>("pos", &mut world_a);
        codecs_a.register_as::<Vel>("vel", &mut world_a);

        // Archetype 1: (Pos, Vel) — will be dirty
        world_a.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }));
        // Archetype 2: (Pos) only — will be clean
        world_a.spawn((Pos { x: 5.0, y: 6.0 },));

        let snap = Snapshot::new();
        snap.save(&base_path, &world_a, &codecs_a, 0).unwrap();
        let (base_world_a, _) = snap.load(&base_path, &codecs_a).unwrap();

        let since_tick = world_a.change_tick();

        // Mutate only (Pos, Vel) archetype
        for (pos, _vel) in world_a.query::<(&mut Pos, &Vel)>() {
            let pos = pos as *const Pos as *mut Pos;
            unsafe { (*pos).x = 100.0 };
        }

        let params = IncrementalParams {
            world: &world_a,
            codecs: &codecs_a,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world_a,
        };
        let (_, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();

        // "Process B" (receiver): OPPOSITE registration order
        let mut world_b_tmp = World::new();
        let mut codecs_b = CodecRegistry::new();
        codecs_b.register_as::<Vel>("vel", &mut world_b_tmp);
        codecs_b.register_as::<Pos>("pos", &mut world_b_tmp);

        // Load base snapshot with receiver codecs
        let (base_world_b, _) = snap.load(&base_path, &codecs_b).unwrap();

        // Load incremental on top of base — this is the path that was buggy
        let (mut restored, _) = snap
            .load_incremental_from_bytes(&inc_bytes, &base_world_b, &codecs_b)
            .unwrap();

        // Verify clean archetype (Pos-only) data survived correctly
        let mut positions: Vec<(f32, f32)> = restored
            .query::<(&Pos,)>()
            .map(|p| (p.0.x, p.0.y))
            .collect();
        positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(positions, vec![(5.0, 6.0), (100.0, 2.0)]);

        // Verify dirty archetype (Pos, Vel) data
        let velocities: Vec<(f32, f32)> = restored
            .query::<(&Vel,)>()
            .map(|v| (v.0.dx, v.0.dy))
            .collect();
        assert_eq!(velocities, vec![(3.0, 4.0)]);
    }

    // ── Structural dirty detection tests ────────────────────────────

    #[test]
    fn incremental_despawn_marks_archetype_dirty() {
        // Despawn doesn't bump column ticks, but the entity count changes,
        // so the archetype must be classified as dirty.
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let _e2 = world.spawn((Pos { x: 3.0, y: 4.0 },));

        let snap = Snapshot::new();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        let since_tick = world.change_tick();

        // Despawn one entity — column ticks are NOT bumped
        world.despawn(e1);

        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let (header, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();

        // Must be dirty: entity count changed from 2 to 1
        assert_eq!(
            header.dirty_archetype_count, 1,
            "archetype with despawned entity must be dirty"
        );
        assert_eq!(header.clean_archetype_count, 0);

        let (mut restored, _) = snap
            .load_incremental_from_bytes(&inc_bytes, &base_world, &codecs)
            .unwrap();
        let positions: Vec<(f32, f32)> = restored
            .query::<(&Pos,)>()
            .map(|p| (p.0.x, p.0.y))
            .collect();
        assert_eq!(positions, vec![(3.0, 4.0)]);
    }

    #[test]
    fn incremental_insert_migration_marks_source_dirty() {
        // Inserting a component migrates entity from archetype A to A∪{new}.
        // The source archetype (A) loses a row without bumping column ticks,
        // so it must be detected as structurally dirty.
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let _e2 = world.spawn((Pos { x: 3.0, y: 4.0 },));

        let snap = Snapshot::new();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        let since_tick = world.change_tick();

        // Migrate e1 from (Pos) to (Pos, Vel)
        world.insert(e1, (Vel { dx: 10.0, dy: 20.0 },));

        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let (header, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();

        // Both archetypes must be dirty: source lost a row, target gained one
        assert_eq!(
            header.dirty_archetype_count, 2,
            "both source and target archetypes must be dirty after migration"
        );
        assert_eq!(header.clean_archetype_count, 0);

        let (mut restored, _) = snap
            .load_incremental_from_bytes(&inc_bytes, &base_world, &codecs)
            .unwrap();
        assert_eq!(restored.query::<(&Pos,)>().count(), 2);
        assert_eq!(restored.query::<(&Vel,)>().count(), 1);
    }

    #[test]
    fn incremental_remove_migration_marks_source_dirty() {
        // Removing a component migrates entity from archetype A to A\{removed}.
        // The source archetype loses a row without bumping column ticks.
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
        let _e2 = world.spawn((Pos { x: 3.0, y: 4.0 }, Vel { dx: 0.3, dy: 0.4 }));

        let snap = Snapshot::new();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        let since_tick = world.change_tick();

        // Migrate e1 from (Pos, Vel) to (Pos)
        world.remove::<Vel>(e1);

        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let (header, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();

        // Source archetype (Pos, Vel) shrunk, target (Pos) grew — both dirty
        assert_eq!(
            header.dirty_archetype_count, 2,
            "both archetypes must be dirty after remove-migration"
        );
        assert_eq!(header.clean_archetype_count, 0);

        let (mut restored, _) = snap
            .load_incremental_from_bytes(&inc_bytes, &base_world, &codecs)
            .unwrap();
        assert_eq!(restored.query::<(&Pos,)>().count(), 2);
        assert_eq!(restored.query::<(&Vel,)>().count(), 1);

        // Verify the correct entity kept its Vel
        let vels: Vec<(f32, f32)> = restored
            .query::<(&Vel,)>()
            .map(|v| (v.0.dx, v.0.dy))
            .collect();
        assert_eq!(vels, vec![(0.3, 0.4)]);
    }

    #[test]
    fn incremental_spawn_after_base_marks_archetype_dirty() {
        // Spawning a new entity into an existing archetype changes entity
        // count. The archetype must be dirty even if no column ticks changed
        // since the base (they will have changed due to spawn, but this test
        // validates the structural check path is present).
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.snap");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = Snapshot::new();
        snap.save(&base_path, &world, &codecs, 0).unwrap();
        let (base_world, _) = snap.load(&base_path, &codecs).unwrap();

        let since_tick = world.change_tick();

        // Spawn into same archetype
        world.spawn((Pos { x: 5.0, y: 6.0 },));

        let params = IncrementalParams {
            world: &world,
            codecs: &codecs,
            base_wal_seq: 0,
            wal_seq: 1,
            since_tick,
            base_world: &base_world,
        };
        let (header, inc_bytes) = snap.save_incremental_to_bytes(&params).unwrap();

        assert_eq!(header.dirty_archetype_count, 1);

        let (mut restored, _) = snap
            .load_incremental_from_bytes(&inc_bytes, &base_world, &codecs)
            .unwrap();
        assert_eq!(restored.query::<(&Pos,)>().count(), 2);
    }
}
