//! Resolving an archetype's identity across multiple sorted runs.
//!
//! Each sorted run assigns its own run-local `arch_id: u16` space. The
//! compactor needs to match archetypes across runs by their component
//! identity (set of stable names), not by arch_id.

use crate::reader::SortedRunReader;

/// Find the `arch_id` within `reader` whose schema matches the given sorted
/// component-name list exactly. Returns `None` if no archetype in the run
/// matches.
///
/// # Precondition
///
/// `components` must be sorted lexicographically. The caller is responsible
/// for sorting before calling this function.
// Used by tests now and by the compactor in Task 3. The dead_code lint fires
// on the lib target because the only current callers are in cfg(test); allow
// it until Task 3 lands.
#[allow(dead_code)]
pub(crate) fn find_archetype_by_components(
    reader: &SortedRunReader,
    components: &[&str],
) -> Option<u16> {
    for arch_id in reader.archetype_ids() {
        // Collect the component slots used by this archetype (sorted, no ENTITY_SLOT).
        let slots = reader.component_slots_for_arch(arch_id);

        // Map each slot to its stable name via the schema section.
        // If any slot is missing from the schema this archetype is malformed;
        // skip it rather than panic.
        let mut names: Vec<&str> = Vec::with_capacity(slots.len());
        let mut malformed = false;
        for slot in &slots {
            match reader.schema().entry_for_slot(*slot) {
                Some(entry) => names.push(entry.name()),
                None => {
                    malformed = true;
                    break;
                }
            }
        }
        if malformed {
            continue;
        }

        // names is already in slot order which equals lexicographic order
        // (SchemaSection assigns slots in lexicographic name order). A simple
        // slice comparison is therefore sufficient.
        if names == components {
            return Some(arch_id);
        }
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SeqNo, SeqRange};
    use crate::writer::flush;
    use minkowski::World;

    // Component types used in tests.
    #[derive(Clone, Copy)]
    #[expect(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy)]
    #[expect(dead_code)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    /// Helper: flush a world to a temp dir, return the reader.
    fn flush_to_reader(
        world: &World,
        seq_lo: u64,
        seq_hi: u64,
    ) -> (tempfile::TempDir, SortedRunReader) {
        let dir = tempfile::tempdir().unwrap();
        let path = flush(
            world,
            SeqRange::new(SeqNo::from(seq_lo), SeqNo::from(seq_hi)).unwrap(),
            dir.path(),
        )
        .unwrap()
        .unwrap();
        let reader = SortedRunReader::open(&path).unwrap();
        (dir, reader)
    }

    /// A run with a single archetype (Pos,) must be findable by its component name.
    #[test]
    fn single_archetype_found() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        let (_dir, reader) = flush_to_reader(&world, 0, 10);

        // We don't know the exact name ahead of time — check the positive path
        // by enumerating arch_ids, looking up the name from the schema, and
        // confirming find returns that same arch_id.
        let arch_ids = reader.archetype_ids();
        assert_eq!(arch_ids.len(), 1);
        let slots = reader.component_slots_for_arch(arch_ids[0]);
        assert_eq!(slots.len(), 1);
        let name = reader.schema().entry_for_slot(slots[0]).unwrap().name();
        let found = find_archetype_by_components(&reader, &[name]);
        assert_eq!(found, Some(arch_ids[0]));
    }

    /// A query for a non-existent component set returns None.
    #[test]
    fn nonexistent_returns_none() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        let (_dir, reader) = flush_to_reader(&world, 0, 10);

        let result = find_archetype_by_components(&reader, &["DoesNotExist"]);
        assert!(result.is_none());
    }

    /// Two runs with the same two archetypes spawned in different orders must
    /// resolve to the correct arch_id in each run, even if the runs assign
    /// different arch_ids to the same logical archetype.
    #[test]
    fn cross_run_archetype_match() {
        // Run A: spawn (Pos,) first, then (Pos, Vel).
        let mut world_a = World::new();
        world_a.spawn((Pos { x: 0.0, y: 0.0 },));
        world_a.spawn((Pos { x: 1.0, y: 1.0 }, Vel { dx: 1.0, dy: 0.0 }));
        let (_dir_a, reader_a) = flush_to_reader(&world_a, 0, 10);

        // Run B: spawn (Pos, Vel) first, then (Pos,).
        // A different world means different archetype registration order, so
        // arch_ids may differ between the runs.
        let mut world_b = World::new();
        world_b.spawn((Pos { x: 2.0, y: 2.0 }, Vel { dx: 2.0, dy: 0.0 }));
        world_b.spawn((Pos { x: 3.0, y: 3.0 },));
        let (_dir_b, reader_b) = flush_to_reader(&world_b, 10, 20);

        // Resolve the actual component names used in run A for each archetype.
        let arch_ids_a = reader_a.archetype_ids();
        assert_eq!(arch_ids_a.len(), 2, "run A must have two archetypes");

        // Find the arch_id in A that has exactly one component (Pos only).
        let (pos_only_arch_a, pos_only_name) = arch_ids_a
            .iter()
            .find_map(|&id| {
                let slots = reader_a.component_slots_for_arch(id);
                if slots.len() == 1 {
                    let name = reader_a.schema().entry_for_slot(slots[0]).unwrap().name();
                    Some((id, name.to_owned()))
                } else {
                    None
                }
            })
            .expect("run A must have a single-component archetype");

        // Find the arch_id in A that has two components (Pos + Vel).
        let (pos_vel_arch_a, pos_name_a, vel_name_a) = arch_ids_a
            .iter()
            .find_map(|&id| {
                let slots = reader_a.component_slots_for_arch(id);
                if slots.len() == 2 {
                    let n0 = reader_a
                        .schema()
                        .entry_for_slot(slots[0])
                        .unwrap()
                        .name()
                        .to_owned();
                    let n1 = reader_a
                        .schema()
                        .entry_for_slot(slots[1])
                        .unwrap()
                        .name()
                        .to_owned();
                    Some((id, n0, n1))
                } else {
                    None
                }
            })
            .expect("run A must have a two-component archetype");

        // Now look up those same component sets in run B.
        let found_pos_in_b = find_archetype_by_components(&reader_b, &[pos_only_name.as_str()]);
        assert!(
            found_pos_in_b.is_some(),
            "Pos-only archetype must be found in run B"
        );

        let mut two_comp_names = [pos_name_a.as_str(), vel_name_a.as_str()];
        two_comp_names.sort_unstable();
        let found_pos_vel_in_b = find_archetype_by_components(&reader_b, &two_comp_names);
        assert!(
            found_pos_vel_in_b.is_some(),
            "Pos+Vel archetype must be found in run B"
        );

        // The arch_ids in run B must be valid (present in run B's arch list).
        let arch_ids_b = reader_b.archetype_ids();
        assert!(arch_ids_b.contains(&found_pos_in_b.unwrap()));
        assert!(arch_ids_b.contains(&found_pos_vel_in_b.unwrap()));

        // And the two results in run B must be different arch_ids.
        assert_ne!(
            found_pos_in_b.unwrap(),
            found_pos_vel_in_b.unwrap(),
            "Pos-only and Pos+Vel must map to different arch_ids in run B"
        );

        // In run A, the arch_ids we resolved must also be distinct.
        assert_ne!(pos_only_arch_a, pos_vel_arch_a);

        // Cross-check: run A's arch_ids for the matching sets are correct.
        let mut two_comp_names_for_a = [pos_name_a.as_str(), vel_name_a.as_str()];
        two_comp_names_for_a.sort_unstable();
        assert_eq!(
            find_archetype_by_components(&reader_a, &[pos_only_name.as_str()]),
            Some(pos_only_arch_a)
        );
        assert_eq!(
            find_archetype_by_components(&reader_a, &two_comp_names_for_a),
            Some(pos_vel_arch_a)
        );
    }
}
