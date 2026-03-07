use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use minkowski::{ComponentId, Entity, EnumChangeSet, World};

use crate::codec::{CodecError, CodecRegistry};
use crate::record::*;

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
/// File format: `[len: u64 LE][payload: len bytes]` — one snapshot per file.
/// Uses rkyv for serialization.
pub struct Snapshot;

impl Snapshot {
    pub fn new() -> Self {
        Self
    }

    /// Save a full world snapshot to disk.
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

        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let len = bytes.len() as u64;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&bytes)?;
        writer.flush()?;

        Ok(header)
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

        if mmap.len() < 8 {
            return Err(SnapshotError::Format("file too small".to_string()));
        }

        // Validate length prefix against actual file size.
        // Use checked arithmetic to prevent overflow on crafted length prefixes.
        let len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let end = 8usize
            .checked_add(len)
            .ok_or_else(|| SnapshotError::Format(format!("invalid payload length: {len}")))?;
        if mmap.len() < end {
            return Err(SnapshotError::Format(format!(
                "file truncated: expected {} payload bytes, got {}",
                len,
                mmap.len() - 8
            )));
        }
        let payload = &mmap[8..end];

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

        Ok(world)
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::new()
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
        use crate::wal::Wal;

        let dir = tempfile::tempdir().unwrap();
        let snap_path = dir.path().join("recovery.snap");
        let wal_path = dir.path().join("recovery.wal");

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);
        codecs.register::<Vel>(&mut world);

        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
        world.spawn((Pos { x: 3.0, y: 4.0 }, Vel { dx: 0.3, dy: 0.4 }));

        let mut wal = Wal::create(&wal_path, &codecs).unwrap();
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

        // Corrupt the payload (flip bytes after the length prefix)
        let mut data = std::fs::read(&snap_path).unwrap();
        for byte in data[8..].iter_mut().take(16) {
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
}
