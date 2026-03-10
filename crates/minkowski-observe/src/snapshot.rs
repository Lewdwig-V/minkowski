//! Point-in-time metrics capture.

use std::time::Instant;

use minkowski::world::WorldStats;
use minkowski::World;
use minkowski_persist::wal::WalStats;
use minkowski_persist::Wal;

/// Per-archetype detail.
#[derive(Clone, Debug)]
pub struct ArchetypeInfo {
    pub id: usize,
    pub entity_count: usize,
    pub component_names: Vec<&'static str>,
    pub estimated_bytes: usize,
}

/// Point-in-time capture of core engine metrics.
#[derive(Clone, Debug)]
pub struct MetricsSnapshot {
    pub world: WorldStats,
    pub wal: Option<WalStats>,
    pub archetypes: Vec<ArchetypeInfo>,
    pub timestamp: Instant,
}

impl MetricsSnapshot {
    /// Capture a point-in-time snapshot of engine metrics.
    ///
    /// Pass `Some(&wal)` to include WAL stats, or `None` for World-only metrics.
    pub fn capture(world: &World, wal: Option<&Wal>) -> Self {
        let world_stats = world.stats();
        let wal_stats = wal.map(Wal::stats);

        let mut archetypes = Vec::with_capacity(world.archetype_count());
        for arch_idx in 0..world.archetype_count() {
            let comp_ids = world.archetype_component_ids(arch_idx);
            let entity_count = world.archetype_len(arch_idx);

            let component_names: Vec<&'static str> = comp_ids
                .iter()
                .filter_map(|&id| world.component_name(id))
                .collect();

            let bytes_per_entity: usize = comp_ids
                .iter()
                .filter_map(|&id| world.component_layout(id))
                .map(|layout| layout.size())
                .sum();

            archetypes.push(ArchetypeInfo {
                id: arch_idx,
                entity_count,
                component_names,
                estimated_bytes: entity_count * bytes_per_entity,
            });
        }

        Self {
            world: world_stats,
            wal: wal_stats,
            archetypes,
            timestamp: Instant::now(),
        }
    }
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "--- World ---\n  entities: {}  archetypes: {}  components: {}",
            self.world.entity_count, self.world.archetype_count, self.world.component_count
        )?;
        writeln!(
            f,
            "  free list: {}  query cache: {}  tick: {}",
            self.world.free_list_len, self.world.query_cache_len, self.world.current_tick
        )?;

        if let Some(ref wal) = self.wal {
            writeln!(
                f,
                "--- WAL ---\n  seq: {}  segments: {}  oldest: {:?}",
                wal.next_seq, wal.segment_count, wal.oldest_seq
            )?;
            writeln!(
                f,
                "  checkpoint: needed={}  last={:?}  bytes_since={}",
                wal.checkpoint_needed, wal.last_checkpoint_seq, wal.bytes_since_checkpoint
            )?;
        }

        if !self.archetypes.is_empty() {
            writeln!(f, "--- Archetypes ---")?;
            for arch in self.archetypes.iter().filter(|a| a.entity_count > 0) {
                writeln!(
                    f,
                    "  [{}] {} entities, ~{} bytes — {:?}",
                    arch.id, arch.entity_count, arch.estimated_bytes, arch.component_names
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski_persist::{CodecRegistry, WalConfig};

    #[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn capture_without_wal() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
        world.spawn((Pos { x: 3.0, y: 4.0 },));

        let snap = MetricsSnapshot::capture(&world, None);

        assert_eq!(snap.world.entity_count, 2);
        assert!(snap.wal.is_none());
        let non_empty: Vec<_> = snap
            .archetypes
            .iter()
            .filter(|a| a.entity_count > 0)
            .collect();
        assert_eq!(non_empty.len(), 2);
    }

    #[test]
    fn capture_with_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let snap = MetricsSnapshot::capture(&world, Some(&wal));
        assert!(snap.wal.is_some());
        assert_eq!(snap.wal.unwrap().next_seq, 0);
    }

    #[test]
    fn archetype_estimated_bytes() {
        let mut world = World::new();
        for _ in 0..10 {
            world.spawn((Pos { x: 1.0, y: 2.0 },));
        }

        let snap = MetricsSnapshot::capture(&world, None);
        let pos_arch = snap
            .archetypes
            .iter()
            .find(|a| a.entity_count == 10)
            .expect("should have archetype with 10 entities");

        // Pos is 8 bytes (2 x f32), 10 entities = 80 bytes
        assert_eq!(pos_arch.estimated_bytes, 80);
    }

    #[test]
    fn snapshot_display_includes_key_info() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));

        let snap = MetricsSnapshot::capture(&world, None);
        let output = format!("{snap}");

        assert!(output.contains("entities: 1"));
        assert!(output.contains("archetypes:"));
        assert!(output.contains("tick:"));
    }
}
