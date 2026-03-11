//! Delta computation from consecutive snapshots.

use std::time::Duration;

use crate::snapshot::MetricsSnapshot;

/// An archetype's ID and entity count, for the top-N list in [`MetricsDiff`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArchetypeSize {
    pub id: usize,
    pub entity_count: usize,
}

/// Deltas computed from two consecutive snapshots.
#[derive(Clone, Debug)]
pub struct MetricsDiff {
    pub elapsed: Duration,
    pub entity_delta: i64,
    pub entity_churn: u64,
    pub tick_delta: u64,
    pub wal_seq_delta: Option<u64>,
    pub archetype_delta: i64,
    pub largest_archetypes: Vec<ArchetypeSize>,
    /// Pool usage delta in bytes. `None` when neither snapshot uses a pool.
    pub pool_used_delta: Option<i64>,
}

impl MetricsDiff {
    /// Compute deltas from two consecutive snapshots.
    #[allow(clippy::cast_possible_wrap)] // delta computation requires signed subtraction
    pub fn compute(before: &MetricsSnapshot, after: &MetricsSnapshot) -> Self {
        let elapsed = after
            .timestamp
            .checked_duration_since(before.timestamp)
            .unwrap_or(Duration::ZERO);

        let entity_delta = after.world.entity_count as i64 - before.world.entity_count as i64;

        let spawns = after
            .world
            .total_spawns
            .saturating_sub(before.world.total_spawns);
        let despawns = after
            .world
            .total_despawns
            .saturating_sub(before.world.total_despawns);
        let entity_churn = spawns + despawns;

        let tick_delta = after
            .world
            .current_tick
            .saturating_sub(before.world.current_tick);

        let wal_seq_delta = match (before.wal, after.wal) {
            (Some(b), Some(a)) => Some(a.next_seq.saturating_sub(b.next_seq)),
            _ => None,
        };

        let archetype_delta =
            after.world.archetype_count as i64 - before.world.archetype_count as i64;

        // Top 5 archetypes by entity count (from the `after` snapshot)
        let mut sorted: Vec<ArchetypeSize> = after
            .archetypes
            .iter()
            .filter(|a| a.entity_count > 0)
            .map(|a| ArchetypeSize {
                id: a.id,
                entity_count: a.entity_count,
            })
            .collect();
        sorted.sort_by(|a, b| b.entity_count.cmp(&a.entity_count));
        sorted.truncate(5);

        let pool_used_delta = match (before.world.pool_used, after.world.pool_used) {
            (Some(b), Some(a)) => Some(a as i64 - b as i64),
            _ => None,
        };

        Self {
            elapsed,
            entity_delta,
            entity_churn,
            tick_delta,
            wal_seq_delta,
            archetype_delta,
            largest_archetypes: sorted,
            pool_used_delta,
        }
    }
}

impl std::fmt::Display for MetricsDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--- Diff ({:.1?}) ---", self.elapsed)?;
        writeln!(
            f,
            "  entity delta: {:+}  churn: {}",
            self.entity_delta, self.entity_churn
        )?;
        match self.wal_seq_delta {
            Some(d) => writeln!(f, "  tick delta: {}  WAL seq delta: {}", self.tick_delta, d)?,
            None => writeln!(f, "  tick delta: {}  WAL: n/a", self.tick_delta)?,
        }
        writeln!(f, "  archetype delta: {:+}", self.archetype_delta)?;
        if let Some(pool_delta) = self.pool_used_delta {
            writeln!(f, "  pool used delta: {:+} bytes", pool_delta)?;
        }

        if !self.largest_archetypes.is_empty() {
            writeln!(f, "  largest archetypes:")?;
            for a in &self.largest_archetypes {
                writeln!(f, "    [{}] {} entities", a.id, a.entity_count)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski::World;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn diff_entity_delta() {
        let mut world = World::new();
        let before = MetricsSnapshot::capture(&world, None);

        world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.spawn((Pos { x: 3.0, y: 4.0 },));
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert_eq!(diff.entity_delta, 2);
        assert!(diff.tick_delta > 0);
        assert_eq!(diff.wal_seq_delta, None);
    }

    #[test]
    fn diff_churn_estimation() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 2.0 },));
        let e2 = world.spawn((Pos { x: 3.0, y: 4.0 },));
        let before = MetricsSnapshot::capture(&world, None);

        world.despawn(e1);
        world.despawn(e2);
        world.spawn((Pos { x: 5.0, y: 6.0 },));
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert_eq!(diff.entity_delta, -1);
        // Exact: 2 despawns + 1 spawn = 3
        assert_eq!(diff.entity_churn, 3);
    }

    #[test]
    fn diff_largest_archetypes() {
        let mut world = World::new();
        for _ in 0..10 {
            world.spawn((Pos { x: 1.0, y: 2.0 },));
        }
        let before = MetricsSnapshot::capture(&world, None);
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert!(!diff.largest_archetypes.is_empty());
        assert_eq!(diff.largest_archetypes[0].entity_count, 10);
    }

    #[test]
    fn diff_archetype_delta() {
        let mut world = World::new();
        let before = MetricsSnapshot::capture(&world, None);

        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.1, dy: 0.2 }));
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert!(diff.archetype_delta >= 1);
    }

    #[test]
    fn diff_display_includes_key_info() {
        let mut world = World::new();
        let before = MetricsSnapshot::capture(&world, None);
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        let output = format!("{diff}");

        assert!(output.contains("entity delta:"));
        assert!(output.contains("tick delta:"));
    }

    #[test]
    fn diff_pool_used_delta() {
        use minkowski::HugePages;
        let mut world = World::builder()
            .memory_budget(4 * 1024 * 1024)
            .hugepages(HugePages::Off)
            .build()
            .unwrap();
        let before = MetricsSnapshot::capture(&world, None);
        for _ in 0..100 {
            world.spawn((Pos { x: 1.0, y: 2.0 },));
        }
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert!(
            diff.pool_used_delta.is_some(),
            "pool_used_delta should be present with pooled world"
        );
        assert!(
            diff.pool_used_delta.unwrap() > 0,
            "pool usage should increase after spawning entities"
        );

        let output = format!("{diff}");
        assert!(output.contains("pool used delta:"));
    }

    #[test]
    fn diff_no_pool_omits_delta() {
        let mut world = World::new();
        let before = MetricsSnapshot::capture(&world, None);
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        let after = MetricsSnapshot::capture(&world, None);

        let diff = MetricsDiff::compute(&before, &after);
        assert!(diff.pool_used_delta.is_none());
        let output = format!("{diff}");
        assert!(!output.contains("pool used delta:"));
    }
}
