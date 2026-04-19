//! Merge-kernel support for LSM compaction.
//!
//! This module builds the emit list used by the compaction write loop (Task 3b).
//! The emit list is a pure-logic computation — it reads entity IDs from
//! existing sorted-run readers and decides, for each unique entity, which
//! source run and row to copy. No file I/O is performed here.

use std::collections::HashSet;

use crate::error::LsmError;
use crate::format::{ENTITY_SLOT, PAGE_SIZE};
use crate::reader::SortedRunReader;

// ── Public types ─────────────────────────────────────────────────────────────

/// A resolved entity emission: the entity id and where its data lives.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct EmitRow {
    /// Raw `Entity::to_bits()` value stored in the entity-slot page.
    pub entity_id: u64,
    /// Index into the input readers slice that owns this entity's data.
    pub source_input_idx: usize,
    /// Row within that input run's archetype — used to locate component
    /// pages when the write loop copies bytes.
    pub source_row: usize,
}

// ── Core function ─────────────────────────────────────────────────────────────

/// Build the emit list for a compaction job. Iterates input readers in order
/// (caller must pass newest-first); emits each `entity_id` the first time it
/// is seen. Duplicates in older runs are silently skipped — newest wins.
///
/// `arch_ids_per_input[i]` is the `arch_id` within `inputs[i]` for the target
/// archetype, or `None` if the archetype doesn't exist in that input (rare
/// but possible for archetypes that first appear in a later flush).
///
/// Returns `Err(LsmError::Format)` if `inputs.len() != arch_ids_per_input.len()`.
#[allow(dead_code)]
pub(crate) fn build_emit_list(
    inputs: &[&SortedRunReader],
    arch_ids_per_input: &[Option<u16>],
) -> Result<Vec<EmitRow>, LsmError> {
    if inputs.len() != arch_ids_per_input.len() {
        return Err(LsmError::Format(format!(
            "build_emit_list: inputs length {} != arch_ids_per_input length {}",
            inputs.len(),
            arch_ids_per_input.len(),
        )));
    }

    let mut seen: HashSet<u64> = HashSet::new();
    let mut emit_list: Vec<EmitRow> = Vec::new();

    for (input_idx, input) in inputs.iter().enumerate() {
        let Some(arch_id) = arch_ids_per_input[input_idx] else {
            continue;
        };

        // Walk entity-slot pages for this archetype from page 0 upward.
        let mut page_index: u16 = 0;
        loop {
            let page = input.get_page(arch_id, ENTITY_SLOT, page_index)?;
            let Some(page) = page else {
                break;
            };

            let row_count = page.header().row_count as usize;
            let data = page.data();

            for row_within_page in 0..row_count {
                let byte_offset = row_within_page * 8;
                // SAFETY (invariant): get_page guarantees data.len() ==
                // PAGE_SIZE * item_size (8 for ENTITY_SLOT), so any row index
                // < row_count is within bounds.
                let entity_id = u64::from_le_bytes(
                    data[byte_offset..byte_offset + 8]
                        .try_into()
                        .expect("8 bytes"),
                );

                let row_in_arch = page_index as usize * PAGE_SIZE + row_within_page;

                if seen.insert(entity_id) {
                    emit_list.push(EmitRow {
                        entity_id,
                        source_input_idx: input_idx,
                        source_row: row_in_arch,
                    });
                }
            }

            // Overflow is impossible in practice (would require 2^16 pages *
            // PAGE_SIZE rows per archetype), but handle it gracefully.
            match page_index.checked_add(1) {
                Some(next) => page_index = next,
                None => break,
            }
        }
    }

    Ok(emit_list)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_match::find_archetype_by_components;
    use crate::types::{SeqNo, SeqRange};
    use crate::writer::flush;
    use minkowski::World;

    // ── Component types ──────────────────────────────────────────────────────

    #[derive(Clone, Copy)]
    #[expect(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Flush `world` to a temp dir and open a reader. Returns the dir (kept
    /// alive by the caller) and the reader.
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

    /// Resolve the arch_id for the (Pos,) archetype in a reader. Panics if
    /// not found — test bug, not production bug.
    fn pos_arch_id(reader: &SortedRunReader) -> u16 {
        // We need to discover the actual name the schema assigned to Pos.
        let arch_ids = reader.archetype_ids();
        for &id in &arch_ids {
            let slots = reader.component_slots_for_arch(id);
            if slots.len() == 1 {
                let name = reader.schema().entry_for_slot(slots[0]).unwrap().name();
                if let Some(found) = find_archetype_by_components(reader, &[name]) {
                    return found;
                }
            }
        }
        panic!("Pos archetype not found in reader");
    }

    // ── Test 1: dedup keeps newest ───────────────────────────────────────────

    /// Two runs that both contain the same entity. Inputs supplied newest-first.
    /// The emit list must contain the entity exactly once, attributed to input 0
    /// (the newer run).
    #[test]
    fn build_emit_list_from_two_runs_dedup_keeps_newest() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        let (_dir_old, reader_old) = flush_to_reader(&world, 1, 10);

        // "Modify" the entity — in this test we just flush the same world
        // again (the entity is still dirty) to represent a newer run containing
        // the same entity ID.
        let (_dir_new, reader_new) = flush_to_reader(&world, 11, 20);

        let old_arch = pos_arch_id(&reader_old);
        let new_arch = pos_arch_id(&reader_new);

        // Inputs: newest first.
        let inputs = [&reader_new, &reader_old];
        let arch_ids = [Some(new_arch), Some(old_arch)];

        let emit = build_emit_list(&inputs, &arch_ids).unwrap();

        // Entity appears in both runs — must be emitted once, from input 0 (newest).
        let entity_bits = e.to_bits();
        let matching: Vec<_> = emit.iter().filter(|r| r.entity_id == entity_bits).collect();
        assert_eq!(matching.len(), 1, "entity must appear exactly once");
        assert_eq!(
            matching[0].source_input_idx, 0,
            "must be attributed to newest run (index 0)"
        );
    }

    // ── Test 2: entities absent from newer run come from older run ────────────

    /// Older run has E1, E2. Newer run adds E3 (and also has E1, E2 because
    /// the world wasn't snapshotted between flushes in this test — but the
    /// important case is that entities *only* in the older run still appear).
    ///
    /// We use two separate worlds to control which entities are in which run.
    /// To ensure non-overlapping entity IDs, `world_new` spawns two placeholder
    /// entities first (advancing its allocator past indices 0 and 1) so that E3
    /// lands at index 2 — distinct from E1 (index 0) and E2 (index 1).
    #[test]
    fn build_emit_list_preserves_entities_absent_from_newer_runs() {
        // Older run: E1 (index 0) + E2 (index 1).
        let mut world_old = World::new();
        let e1 = world_old.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world_old.spawn((Pos { x: 2.0, y: 0.0 },));
        let (_dir_old, reader_old) = flush_to_reader(&world_old, 1, 10);

        // Newer run: only E3 (index 2 — placeholder spawns advance past 0, 1).
        let mut world_new = World::new();
        // These two placeholders are never flushed (world_new is flushed after
        // despawning them), but they advance the entity allocator so E3 gets
        // a distinct index from E1/E2.
        let ph1 = world_new.spawn((Pos { x: 0.0, y: 0.0 },));
        let ph2 = world_new.spawn((Pos { x: 0.0, y: 0.0 },));
        world_new.despawn(ph1);
        world_new.despawn(ph2);
        let e3 = world_new.spawn((Pos { x: 3.0, y: 0.0 },));
        let (_dir_new, reader_new) = flush_to_reader(&world_new, 11, 20);

        let old_arch = pos_arch_id(&reader_old);
        let new_arch = pos_arch_id(&reader_new);

        // Inputs: newest first.
        let inputs = [&reader_new, &reader_old];
        let arch_ids = [Some(new_arch), Some(old_arch)];

        let emit = build_emit_list(&inputs, &arch_ids).unwrap();

        // All three entities must appear.
        let ids: HashSet<u64> = emit.iter().map(|r| r.entity_id).collect();
        assert!(ids.contains(&e1.to_bits()), "E1 must be in emit list");
        assert!(ids.contains(&e2.to_bits()), "E2 must be in emit list");
        assert!(ids.contains(&e3.to_bits()), "E3 must be in emit list");

        // E3 comes from input 0 (newer).
        let e3_row = emit.iter().find(|r| r.entity_id == e3.to_bits()).unwrap();
        assert_eq!(e3_row.source_input_idx, 0, "E3 must come from newer run");

        // E1 and E2 come from input 1 (older) — not present in newer run.
        let e1_row = emit.iter().find(|r| r.entity_id == e1.to_bits()).unwrap();
        let e2_row = emit.iter().find(|r| r.entity_id == e2.to_bits()).unwrap();
        assert_eq!(e1_row.source_input_idx, 1, "E1 must come from older run");
        assert_eq!(e2_row.source_input_idx, 1, "E2 must come from older run");
    }

    // ── Test 3: inputs with missing archetype are skipped ────────────────────

    /// One of the inputs has `arch_ids_per_input[i] = None`. The function must
    /// skip it and still emit entities from the other inputs.
    #[test]
    fn build_emit_list_skips_inputs_where_archetype_missing() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 0.0, y: 0.0 },));
        let (_dir, reader) = flush_to_reader(&world, 0, 10);

        let arch = pos_arch_id(&reader);

        // Three inputs: the middle one is valid; the first and last have None.
        let inputs = [&reader, &reader, &reader];
        let arch_ids = [None, Some(arch), None];

        let emit = build_emit_list(&inputs, &arch_ids).unwrap();

        assert_eq!(emit.len(), 1, "exactly one entity must be emitted");
        assert_eq!(emit[0].entity_id, e.to_bits());
        // First input with data is index 1 (the only non-None).
        assert_eq!(
            emit[0].source_input_idx, 1,
            "entity must be attributed to the only non-None input"
        );
    }

    // ── Test 4: length mismatch returns LsmError::Format ────────────────────

    #[test]
    fn build_emit_list_rejects_length_mismatch() {
        let mut world = World::new();
        world.spawn((Pos { x: 0.0, y: 0.0 },));
        let (_dir, reader) = flush_to_reader(&world, 0, 10);

        let inputs = [&reader, &reader]; // len 2
        let arch_ids = [Some(0u16)]; // len 1 — mismatch

        let result = build_emit_list(&inputs, &arch_ids);
        assert!(
            matches!(result, Err(LsmError::Format(_))),
            "expected LsmError::Format for length mismatch, got: {result:?}"
        );
    }
}
