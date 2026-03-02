use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use minkowski::{Entity, EnumChangeSet, World};

use crate::codec::{CodecError, CodecRegistry};
use crate::format::WireFormat;
use crate::record::*;

#[derive(Debug)]
pub enum SnapshotError {
    Io(std::io::Error),
    Codec(CodecError),
    Format(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "snapshot I/O: {e}"),
            Self::Codec(e) => write!(f, "snapshot codec: {e}"),
            Self::Format(msg) => write!(f, "snapshot format: {msg}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<CodecError> for SnapshotError {
    fn from(e: CodecError) -> Self {
        Self::Codec(e)
    }
}

/// Full-world snapshot: serialize all archetype data to disk and reconstruct on load.
///
/// File format: `[len: u64 LE][payload: len bytes]` — one snapshot per file.
pub struct Snapshot<W: WireFormat> {
    format: W,
}

impl<W: WireFormat> Snapshot<W> {
    pub fn new(format: W) -> Self {
        Self { format }
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

        let bytes = self
            .format
            .serialize_snapshot(&data)
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
    /// The caller must have registered codecs (via `CodecRegistry::register`) for all
    /// component types in the same order as the original world so that `ComponentId`s
    /// match. Schema validation is a future enhancement.
    pub fn load(&self, path: &Path, codecs: &CodecRegistry) -> Result<(World, u64), SnapshotError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let len = u64::from_le_bytes(len_buf) as usize;

        let mut bytes = vec![0u8; len];
        reader.read_exact(&mut bytes)?;

        let data = self
            .format
            .deserialize_snapshot(&bytes)
            .map_err(|e| SnapshotError::Format(e.to_string()))?;

        let world = self.restore_world(&data, codecs)?;
        Ok((world, data.wal_seq))
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn build_snapshot_data(
        &self,
        world: &World,
        codecs: &CodecRegistry,
        wal_seq: u64,
    ) -> Result<SnapshotData, SnapshotError> {
        // Schema — one entry per registered codec
        let schema: Vec<ComponentSchema> = codecs
            .registered_ids()
            .iter()
            .map(|&id| ComponentSchema {
                id,
                name: codecs.name(id).unwrap_or("unknown").to_string(),
                size: codecs.layout(id).map(|l| l.size()).unwrap_or(0),
                align: codecs.layout(id).map(|l| l.align()).unwrap_or(1),
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

        Ok(SnapshotData {
            wal_seq,
            schema,
            allocator,
            archetypes,
            sparse: vec![], // Task 6 adds sparse support
        })
    }

    fn restore_world(
        &self,
        data: &SnapshotData,
        codecs: &CodecRegistry,
    ) -> Result<World, SnapshotError> {
        let mut world = World::new();

        // Register all component types into the new World so that archetype
        // creation can look up layouts and drop functions by ComponentId.
        // register_all replays the concrete register_component::<T>() calls
        // in ID order so the new World gets matching ComponentIds.
        codecs.register_all(&mut world);

        // Restore archetypes via EnumChangeSet
        for arch_data in &data.archetypes {
            let mut changeset = EnumChangeSet::new();

            for (row, &entity_bits) in arch_data.entities.iter().enumerate() {
                let entity = Entity::from_bits(entity_bits);

                // Allocate the entity slot in the new world so record_spawn can place it.
                world.alloc_entity();

                // Deserialize all components for this entity
                let mut raw_components: Vec<(minkowski::ComponentId, Vec<u8>, std::alloc::Layout)> =
                    Vec::new();
                for col in &arch_data.columns {
                    let raw = codecs.deserialize(col.component_id, &col.values[row])?;
                    let layout = codecs
                        .layout(col.component_id)
                        .ok_or(CodecError::UnregisteredComponent(col.component_id))?;
                    raw_components.push((col.component_id, raw, layout));
                }

                let ptrs: Vec<_> = raw_components
                    .iter()
                    .map(|(id, raw, layout)| (*id, raw.as_ptr(), *layout))
                    .collect();
                changeset.record_spawn(entity, &ptrs);
            }

            changeset.apply(&mut world);
        }

        // Restore allocator state AFTER all entities are placed.
        // This overwrites the sequential generation counters with the saved ones,
        // ensuring entity handles from before the snapshot are still valid.
        world.restore_allocator_state(
            data.allocator.generations.clone(),
            data.allocator.free_list.clone(),
        );

        Ok(world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use crate::format::Bincode;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
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

        let snap = Snapshot::new(Bincode);
        let header = snap.save(&snap_path, &world, &codecs, 42).unwrap();
        assert_eq!(header.entity_count, 2);
        assert_eq!(header.wal_seq, 42);

        // Load — codecs carry the register_fn so load creates a fresh World internally
        let (mut world2, wal_seq) = snap.load(&snap_path, &codecs).unwrap();
        assert_eq!(wal_seq, 42);

        // Verify entities have correct component values
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

        let snap = Snapshot::new(Bincode);
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

        let snap = Snapshot::new(Bincode);
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

        let snap = Snapshot::new(Bincode);
        let result = snap.save(&snap_path, &world, &codecs, 0);
        assert!(result.is_err());
    }
}
