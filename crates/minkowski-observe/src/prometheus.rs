//! Prometheus OpenMetrics exporter for engine metrics.

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

use crate::snapshot::MetricsSnapshot;

/// Exports engine metrics in OpenMetrics text format.
///
/// Push model: call `update()` with a `MetricsSnapshot`, then `render()` to
/// produce the text body for a `/metrics` HTTP endpoint. No threads, no
/// networking — the caller owns the serving strategy.
pub struct PrometheusExporter {
    registry: Registry,

    // World gauges
    entity_count: Gauge,
    archetype_count: Gauge,
    component_count: Gauge,
    free_list_len: Gauge,
    query_cache_len: Gauge,
    tick: Gauge,
    total_spawns: Gauge,
    total_despawns: Gauge,

    // Pool gauges
    pool_capacity_bytes: Gauge,
    pool_used_bytes: Gauge,

    // WAL gauges
    wal_seq: Gauge,
    wal_segment_count: Gauge,
    wal_bytes_since_checkpoint: Gauge,

    // Per-archetype families
    archetype_entity_count: Family<Vec<(String, String)>, Gauge>,
    archetype_estimated_bytes: Family<Vec<(String, String)>, Gauge>,
}

impl PrometheusExporter {
    /// Create an exporter with all gauges registered at zero.
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let entity_count = Gauge::default();
        let archetype_count = Gauge::default();
        let component_count = Gauge::default();
        let free_list_len = Gauge::default();
        let query_cache_len = Gauge::default();
        let tick = Gauge::default();
        let total_spawns = Gauge::default();
        let total_despawns = Gauge::default();

        registry.register(
            "minkowski_entity_count",
            "Live entity count",
            entity_count.clone(),
        );
        registry.register(
            "minkowski_archetype_count",
            "Archetype count",
            archetype_count.clone(),
        );
        registry.register(
            "minkowski_component_count",
            "Registered component types",
            component_count.clone(),
        );
        registry.register(
            "minkowski_free_list_len",
            "Entity free list length",
            free_list_len.clone(),
        );
        registry.register(
            "minkowski_query_cache_len",
            "Cached query types",
            query_cache_len.clone(),
        );
        registry.register("minkowski_tick", "Current engine tick", tick.clone());
        registry.register(
            "minkowski_total_spawns",
            "Monotonic spawn counter",
            total_spawns.clone(),
        );
        registry.register(
            "minkowski_total_despawns",
            "Monotonic despawn counter",
            total_despawns.clone(),
        );

        let pool_capacity_bytes = Gauge::default();
        let pool_used_bytes = Gauge::default();

        registry.register(
            "minkowski_pool_capacity_bytes",
            "Memory pool total capacity in bytes (0 if no pool)",
            pool_capacity_bytes.clone(),
        );
        registry.register(
            "minkowski_pool_used_bytes",
            "Memory pool bytes currently allocated (0 if no pool)",
            pool_used_bytes.clone(),
        );

        let wal_seq = Gauge::default();
        let wal_segment_count = Gauge::default();
        let wal_bytes_since_checkpoint = Gauge::default();

        registry.register(
            "minkowski_wal_seq",
            "WAL next sequence number",
            wal_seq.clone(),
        );
        registry.register(
            "minkowski_wal_segment_count",
            "WAL active segment count",
            wal_segment_count.clone(),
        );
        registry.register(
            "minkowski_wal_bytes_since_checkpoint",
            "WAL bytes since last checkpoint",
            wal_bytes_since_checkpoint.clone(),
        );

        let archetype_entity_count = Family::default();
        let archetype_estimated_bytes = Family::default();

        registry.register(
            "minkowski_archetype_entity_count",
            "Entities per archetype",
            archetype_entity_count.clone(),
        );
        registry.register(
            "minkowski_archetype_estimated_bytes",
            "Estimated bytes per archetype",
            archetype_estimated_bytes.clone(),
        );

        Self {
            registry,
            entity_count,
            archetype_count,
            component_count,
            free_list_len,
            query_cache_len,
            tick,
            total_spawns,
            total_despawns,
            pool_capacity_bytes,
            pool_used_bytes,
            wal_seq,
            wal_segment_count,
            wal_bytes_since_checkpoint,
            archetype_entity_count,
            archetype_estimated_bytes,
        }
    }

    /// Update all gauge values from a snapshot.
    #[allow(clippy::cast_possible_wrap)] // prometheus-client requires i64 gauge values
    pub fn update(&self, snapshot: &MetricsSnapshot) {
        self.entity_count.set(snapshot.world.entity_count as i64);
        self.archetype_count
            .set(snapshot.world.archetype_count as i64);
        self.component_count
            .set(snapshot.world.component_count as i64);
        self.free_list_len.set(snapshot.world.free_list_len as i64);
        self.query_cache_len
            .set(snapshot.world.query_cache_len as i64);
        self.tick.set(snapshot.world.current_tick as i64);
        self.total_spawns.set(snapshot.world.total_spawns as i64);
        self.total_despawns
            .set(snapshot.world.total_despawns as i64);

        self.pool_capacity_bytes
            .set(snapshot.world.pool_capacity.unwrap_or(0) as i64);
        self.pool_used_bytes
            .set(snapshot.world.pool_used.unwrap_or(0) as i64);

        if let Some(ref wal) = snapshot.wal {
            self.wal_seq.set(wal.next_seq as i64);
            self.wal_segment_count.set(wal.segment_count as i64);
            self.wal_bytes_since_checkpoint
                .set(wal.bytes_since_checkpoint as i64);
        } else {
            self.wal_seq.set(0);
            self.wal_segment_count.set(0);
            self.wal_bytes_since_checkpoint.set(0);
        }

        // Clear stale archetype series before repopulating. Without this,
        // archetypes that disappeared since the last update would retain
        // their old gauge values in the rendered output.
        self.archetype_entity_count.clear();
        self.archetype_estimated_bytes.clear();

        for arch in &snapshot.archetypes {
            let labels = vec![("archetype_id".to_string(), arch.id.to_string())];
            self.archetype_entity_count
                .get_or_create(&labels)
                .set(arch.entity_count as i64);
            self.archetype_estimated_bytes
                .get_or_create(&labels)
                .set(arch.estimated_bytes as i64);
        }
    }

    /// Encode the registry to OpenMetrics text format.
    pub fn render(&self) -> String {
        let mut buf = String::new();
        encode(&mut buf, &self.registry).unwrap();
        buf
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski::World;

    #[test]
    fn new_renders_default_zeros() {
        let exporter = PrometheusExporter::new();
        let output = exporter.render();
        assert!(output.contains("minkowski_entity_count"));
        assert!(output.contains("minkowski_tick"));
        assert!(output.contains("minkowski_wal_seq"));
        assert!(output.contains("minkowski_archetype_entity_count"));
    }

    #[test]
    fn update_sets_world_gauges() {
        let mut world = World::new();
        world.spawn((42_u32,));
        world.spawn((42_u32,));

        let snap = MetricsSnapshot::capture(&world, None);
        let exporter = PrometheusExporter::new();
        exporter.update(&snap);
        let output = exporter.render();

        assert!(output.contains("minkowski_entity_count 2"));
        assert!(output.contains("minkowski_total_spawns 2"));
    }

    #[test]
    fn update_sets_wal_gauges() {
        use minkowski_persist::{CodecRegistry, Wal, WalConfig};

        #[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
        struct Pos {
            x: f32,
            y: f32,
        }

        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register::<Pos>(&mut world);

        let wal = Wal::create(&wal_path, &codecs, WalConfig::default()).unwrap();
        let snap = MetricsSnapshot::capture(&world, Some(&wal));

        let exporter = PrometheusExporter::new();
        exporter.update(&snap);
        let output = exporter.render();

        assert!(output.contains("minkowski_wal_seq"));
        assert!(output.contains("minkowski_wal_segment_count"));
    }

    #[test]
    fn update_sets_archetype_labels() {
        let mut world = World::new();
        for _ in 0..5 {
            world.spawn((42_u32,));
        }

        let snap = MetricsSnapshot::capture(&world, None);
        let exporter = PrometheusExporter::new();
        exporter.update(&snap);
        let output = exporter.render();

        assert!(output.contains("minkowski_archetype_entity_count"));
        assert!(output.contains('5'));
    }

    #[test]
    fn update_sets_pool_gauges() {
        use minkowski::HugePages;
        let mut world = World::builder()
            .memory_budget(4 * 1024 * 1024)
            .hugepages(HugePages::Off)
            .build()
            .unwrap();
        world.spawn((42_u32,));

        let snap = MetricsSnapshot::capture(&world, None);
        let exporter = PrometheusExporter::new();
        exporter.update(&snap);
        let output = exporter.render();

        assert!(output.contains("minkowski_pool_capacity_bytes 4194304"));
        assert!(output.contains("minkowski_pool_used_bytes"));
        // Pool used should be > 0 after spawning
        assert!(!output.contains("minkowski_pool_used_bytes 0\n"));
    }

    #[test]
    fn update_clears_stale_archetype_series() {
        let exporter = PrometheusExporter::new();

        // First snapshot: two archetypes
        let mut world1 = World::new();
        world1.spawn((42_u32,));
        world1.spawn((1.0_f32,));
        let snap1 = MetricsSnapshot::capture(&world1, None);
        exporter.update(&snap1);
        let output1 = exporter.render();
        // Both archetypes present
        assert!(output1.contains("archetype_id=\"0\""));
        assert!(output1.contains("archetype_id=\"1\""));

        // Second snapshot: different world with only one archetype
        let mut world2 = World::new();
        world2.spawn((99_u32,));
        let snap2 = MetricsSnapshot::capture(&world2, None);
        exporter.update(&snap2);
        let output2 = exporter.render();
        // Only the one archetype should remain — stale series cleared
        assert!(output2.contains("archetype_id=\"0\""));
        assert!(!output2.contains("archetype_id=\"1\""));
    }
}
