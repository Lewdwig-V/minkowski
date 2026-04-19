use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use minkowski::World;

use crate::error::LsmError;
use crate::format::*;
use crate::schema::SchemaSection;
use crate::types::SeqRange;

/// The value passed to an [`EntryObserver`] for each entity written to an
/// entity-slot page.
///
/// Currently just the raw entity bits (`Entity::to_bits()`). If Phase 4
/// needs archetype context it can be extended — the observer API is not a
/// stability boundary yet.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct EntityKey(pub u64);

/// Per-entry observer invoked by [`flush_observed`] once for each entity
/// written to an entity-slot page. Phase 4 bloom filter uses this to build
/// a per-run filter without re-plumbing the writer. Phase 3 leaves the hook
/// in place with a no-op observer.
///
/// Not `Send`-bounded — [`flush`] / [`flush_observed`] are synchronous and
/// single-threaded; callers that need cross-thread sharing can wrap their
/// own `Arc<Mutex<_>>` before installing.
pub type EntryObserver = Box<dyn FnMut(EntityKey)>;

/// Convert a `usize` to `u16`, returning `LsmError::Format` on overflow.
fn to_u16(value: usize, label: &str) -> Result<u16, LsmError> {
    u16::try_from(value).map_err(|_| LsmError::Format(format!("{label} {value} exceeds u16")))
}

/// Flush dirty pages from the World to a new sorted run file.
///
/// Returns `Ok(Some(path))` if dirty pages were written, `Ok(None)` if there
/// were no dirty pages to flush. The file is written atomically (temp + rename).
///
/// `sequence_range` is the WAL sequence range covered by this flush — stored in
/// the header for recovery to know where to start WAL replay.
pub fn flush(
    world: &World,
    sequence_range: SeqRange,
    output_dir: &Path,
) -> Result<Option<PathBuf>, LsmError> {
    flush_observed(world, sequence_range, output_dir, None)
}

/// Like [`flush`], but invokes `observer` once per entity ID written to an
/// entity-slot page. Pass `None` for no observation (identical to [`flush`]).
///
/// The observer fires *after* the entity bytes are successfully written to
/// the buffer, so it will not fire for pages that are skipped due to zero
/// `row_count`. It fires exactly once per entity per call.
pub fn flush_observed(
    world: &World,
    sequence_range: SeqRange,
    output_dir: &Path,
    mut observer: Option<&mut dyn FnMut(EntityKey)>,
) -> Result<Option<PathBuf>, LsmError> {
    // ── 1. Collect dirty page set ───────────────────────────────────────────
    // Key: (arch_idx, comp_id, page_index)
    let mut dirty: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
    // Per-archetype union of dirty page indices (for entity pages).
    let mut entity_dirty: HashMap<usize, BTreeSet<usize>> = HashMap::new();

    for arch_idx in 0..world.archetype_count() {
        for &comp_id in world.archetype_component_ids(arch_idx) {
            if let Some(pages) = world.column_dirty_pages(arch_idx, comp_id) {
                for page in pages {
                    dirty.insert((arch_idx, comp_id, page));
                    entity_dirty.entry(arch_idx).or_default().insert(page);
                }
            }
        }
    }

    // ── 2. Early return if nothing dirty ────────────────────────────────────
    if dirty.is_empty() {
        return Ok(None);
    }

    // ── 3. Build schema section ─────────────────────────────────────────────
    let mut seen_comp_ids: BTreeSet<usize> = BTreeSet::new();
    for &(_, comp_id, _) in &dirty {
        seen_comp_ids.insert(comp_id);
    }

    let components: Vec<(String, std::alloc::Layout)> = seen_comp_ids
        .iter()
        .map(|&comp_id| {
            let name = world
                .component_name(comp_id)
                .expect("dirty component must be registered");
            let layout = world
                .component_layout(comp_id)
                .expect("dirty component must have a layout");
            (name.to_owned(), layout)
        })
        .collect();

    let schema = SchemaSection::from_components(&components)?;

    // Build comp_id → component name lookup for slot resolution.
    let comp_id_to_name: HashMap<usize, &str> = seen_comp_ids
        .iter()
        .map(|&comp_id| {
            (
                comp_id,
                world
                    .component_name(comp_id)
                    .expect("dirty component must be registered"),
            )
        })
        .collect();

    // ── 4. Write to temp file ───────────────────────────────────────────────
    let seq_lo = sequence_range.lo().get();
    let seq_hi = sequence_range.hi().get();
    let tmp_name = format!("{seq_lo}-{seq_hi}.run.tmp");
    let final_name = format!("{seq_lo}-{seq_hi}.run");
    let tmp_path = output_dir.join(&tmp_name);
    let final_path = output_dir.join(&final_name);

    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)?;

    // Drop guard to clean up the temp file on error.
    struct TmpGuard<'a> {
        path: &'a Path,
        disarmed: bool,
    }
    impl Drop for TmpGuard<'_> {
        fn drop(&mut self) {
            if !self.disarmed {
                let _ = fs::remove_file(self.path);
            }
        }
    }
    let mut guard = TmpGuard {
        path: &tmp_path,
        disarmed: false,
    };

    let mut w = BufWriter::new(file);

    // (a) Header — write with crc32 = 0, patch later.
    let page_count = dirty.len() + entity_dirty.values().map(BTreeSet::len).sum::<usize>();
    let header = Header {
        magic: MAGIC,
        version: VERSION,
        schema_count: schema.len() as u32,
        page_count: page_count as u64,
        sequence_lo: seq_lo,
        sequence_hi: seq_hi,
        header_crc32: 0,
        reserved: [0u8; 20],
    };
    w.write_all(header.as_bytes())?;

    // (b) Schema section
    let schema_offset = std::mem::size_of::<Header>() as u64;
    schema.write_to(&mut w)?;

    // (c) Component page images — sorted by (arch_id, slot, page_index)
    let mut index_entries: Vec<IndexEntry> = Vec::with_capacity(page_count);

    // Pre-sort dirty pages by (arch_idx as u16, slot, page_index as u16) for
    // deterministic file order matching the index sort key.
    struct PageJob {
        arch_idx: usize,
        comp_id: usize,
        page_index: usize,
        slot: u16,
    }

    let mut component_jobs: Vec<PageJob> = Vec::with_capacity(dirty.len());
    for &(arch_idx, comp_id, page_index) in &dirty {
        let comp_name = comp_id_to_name[&comp_id];
        let slot = schema
            .slot_for(comp_name)
            .expect("component must be in schema");
        component_jobs.push(PageJob {
            arch_idx,
            comp_id,
            page_index,
            slot,
        });
    }
    // Validate all indices fit in u16 before sorting.
    for job in &component_jobs {
        to_u16(job.arch_idx, "arch_idx")?;
        to_u16(job.page_index, "page_index")?;
    }
    component_jobs.sort_by_key(|j| (j.arch_idx as u16, j.slot, j.page_index as u16));

    for job in &component_jobs {
        let arch_id = to_u16(job.arch_idx, "arch_idx")?;
        let page_idx = to_u16(job.page_index, "page_index")?;

        let arch_len = world.archetype_len(job.arch_idx);
        let start_row = job.page_index * PAGE_SIZE;
        let row_count = PAGE_SIZE.min(arch_len.saturating_sub(start_row));
        if row_count == 0 {
            continue;
        }
        let row_count_u16 = to_u16(row_count, "row_count")?;

        let item_size = schema
            .entry_for_slot(job.slot)
            .expect("slot must exist")
            .item_size as usize;

        let bytes = world
            .column_page_bytes(job.arch_idx, job.comp_id, start_row, row_count)
            .expect("dirty page must be readable");

        let page_crc = crc32fast::hash(bytes);

        let file_offset = w.stream_position()?;

        let ph = PageHeader {
            arch_id,
            slot: job.slot,
            page_index: page_idx,
            row_count: row_count_u16,
            page_crc32: page_crc,
            _padding: 0,
        };
        w.write_all(ph.as_bytes())?;
        w.write_all(bytes)?;

        // Zero-pad partial pages.
        let full_page_bytes = PAGE_SIZE * item_size;
        if bytes.len() < full_page_bytes {
            let pad = full_page_bytes - bytes.len();
            write_zeros(&mut w, pad)?;
        }

        index_entries.push(IndexEntry {
            arch_id,
            slot: job.slot,
            page_index: page_idx,
            _pad: 0,
            file_offset,
        });
    }

    // (d) Entity pages
    let entity_item_size = std::mem::size_of::<u64>();
    let mut entity_jobs: Vec<(usize, usize)> = Vec::new(); // (arch_idx, page_index)
    for (&arch_idx, pages) in &entity_dirty {
        for &page_index in pages {
            entity_jobs.push((arch_idx, page_index));
        }
    }
    // Validate all entity job indices fit in u16 before sorting.
    for &(arch_idx, page_index) in &entity_jobs {
        to_u16(arch_idx, "arch_idx")?;
        to_u16(page_index, "page_index")?;
    }
    entity_jobs.sort_by_key(|&(arch_idx, page_index)| (arch_idx as u16, page_index as u16));

    for &(arch_idx, page_index) in &entity_jobs {
        let arch_id = to_u16(arch_idx, "arch_idx")?;
        let page_idx = to_u16(page_index, "page_index")?;

        let entities = world.archetype_entities(arch_idx);
        let start_row = page_index * PAGE_SIZE;
        let row_count = PAGE_SIZE.min(entities.len().saturating_sub(start_row));
        if row_count == 0 {
            continue;
        }
        let row_count_u16 = to_u16(row_count, "row_count")?;

        let page_entities = &entities[start_row..start_row + row_count];

        // Convert entities to LE bytes.
        let mut entity_bytes = Vec::with_capacity(row_count * entity_item_size);
        for &e in page_entities {
            entity_bytes.extend_from_slice(&e.to_bits().to_le_bytes());
        }

        let page_crc = crc32fast::hash(&entity_bytes);
        let file_offset = w.stream_position()?;

        let ph = PageHeader {
            arch_id,
            slot: ENTITY_SLOT,
            page_index: page_idx,
            row_count: row_count_u16,
            page_crc32: page_crc,
            _padding: 0,
        };
        w.write_all(ph.as_bytes())?;
        w.write_all(&entity_bytes)?;

        // Notify the observer once per successfully-written entity.
        if let Some(ref mut obs) = observer {
            for &e in page_entities {
                obs(EntityKey(e.to_bits()));
            }
        }

        // Zero-pad partial pages.
        let full_page_bytes = PAGE_SIZE * entity_item_size;
        if entity_bytes.len() < full_page_bytes {
            let pad = full_page_bytes - entity_bytes.len();
            write_zeros(&mut w, pad)?;
        }

        index_entries.push(IndexEntry {
            arch_id,
            slot: ENTITY_SLOT,
            page_index: page_idx,
            _pad: 0,
            file_offset,
        });
    }

    // (e) Sparse index — sort by (arch_id, slot, page_index) so the reader
    //     can binary-search.  Component pages are already sorted by their
    //     write order, but entity pages (slot = ENTITY_SLOT) are appended
    //     after all component pages, which breaks global sort order when
    //     multiple archetypes are present.
    index_entries.sort();
    let sparse_index_offset = w.stream_position()?;
    for entry in &index_entries {
        w.write_all(entry.as_bytes())?;
    }

    // (f) Footer
    let footer = Footer {
        sparse_index_offset,
        sparse_index_count: index_entries.len() as u64,
        schema_offset,
        bloom_filter_offset: 0,
        total_crc32: 0,
        reserved: [0u8; 28],
    };
    w.write_all(footer.as_bytes())?;

    // Flush the BufWriter so all bytes reach disk before CRC patching.
    w.flush()?;

    // ── 5. Patch CRCs ──────────────────────────────────────────────────────

    // Get the inner File for seeking.
    let mut file = w
        .into_inner()
        .map_err(std::io::IntoInnerError::into_error)?;

    // Patch header CRC: compute over the first 40 bytes (everything before
    // header_crc32 field at offset 40).
    file.seek(SeekFrom::Start(0))?;
    let mut header_bytes = [0u8; 64];
    {
        use std::io::Read;
        file.read_exact(&mut header_bytes)?;
    }
    // header_crc32 is at offset 40 (8 + 4 + 4 + 8 + 8 + 8 = 40).
    let header_crc = crc32fast::hash(&header_bytes[..40]);
    file.seek(SeekFrom::Start(40))?;
    file.write_all(&header_crc.to_le_bytes())?;

    // Patch total CRC: compute over entire file with total_crc32 field zeroed.
    // total_crc32 is at footer offset + 32 (4 * u64 = 32 bytes into footer).
    let file_len = file.seek(SeekFrom::End(0))?;
    let footer_offset = file_len - 64;
    let total_crc32_file_offset = footer_offset + 32;

    // Read entire file, zero out total_crc32 field, compute CRC.
    file.seek(SeekFrom::Start(0))?;
    let mut all_bytes = vec![0u8; file_len as usize];
    {
        use std::io::Read;
        file.read_exact(&mut all_bytes)?;
    }
    // Zero out the total_crc32 field for CRC computation.
    let tco = total_crc32_file_offset as usize;
    all_bytes[tco..tco + 4].copy_from_slice(&[0, 0, 0, 0]);
    // Also zero out the header_crc32 was already patched, which is fine — we
    // want total CRC to cover the patched header.
    let total_crc = crc32fast::hash(&all_bytes);

    // Write total_crc32 into footer.
    file.seek(SeekFrom::Start(total_crc32_file_offset))?;
    file.write_all(&total_crc.to_le_bytes())?;
    file.sync_all()?;
    drop(file);

    // ── 6. Atomic rename ────────────────────────────────────────────────────
    fs::rename(&tmp_path, &final_path)?;

    // Sync the directory to ensure the rename is durable.
    let dir = fs::File::open(output_dir)?;
    dir.sync_all()?;

    guard.disarmed = true;

    Ok(Some(final_path))
}

/// Write `n` zero bytes to `w`.
fn write_zeros(w: &mut impl Write, n: usize) -> Result<(), LsmError> {
    const BLOCK: [u8; 4096] = [0u8; 4096];
    let mut remaining = n;
    while remaining > 0 {
        let chunk = remaining.min(BLOCK.len());
        w.write_all(&BLOCK[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SeqNo, SeqRange};
    use minkowski::World;

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

    #[test]
    fn flush_no_dirty_pages_returns_none() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.clear_all_dirty_pages();
        let dir = tempfile::tempdir().unwrap();
        let result = flush(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(0u64)).unwrap(),
            dir.path(),
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn flush_dirty_pages_creates_file() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        // Pages are dirty from spawn.
        let dir = tempfile::tempdir().unwrap();
        let result = flush(
            &world,
            SeqRange::new(SeqNo::from(1u64), SeqNo::from(5u64)).unwrap(),
            dir.path(),
        )
        .unwrap();
        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.exists());
        assert!(path.file_name().unwrap().to_str().unwrap().contains("1-5"));
    }

    #[test]
    fn file_has_correct_header_magic() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        let dir = tempfile::tempdir().unwrap();
        let result = flush(
            &world,
            SeqRange::new(SeqNo::from(10u64), SeqNo::from(20u64)).unwrap(),
            dir.path(),
        )
        .unwrap();
        let path = result.unwrap();
        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[..8], &MAGIC);
    }

    #[test]
    fn flush_multi_component() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 0.5, dy: -0.5 }));
        world.spawn((Pos { x: 3.0, y: 4.0 }, Vel { dx: 1.0, dy: 1.0 }));
        let dir = tempfile::tempdir().unwrap();
        let result = flush(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(10u64)).unwrap(),
            dir.path(),
        )
        .unwrap();
        assert!(result.is_some());
        let path = result.unwrap();
        let data = std::fs::read(&path).unwrap();

        // Verify header fields.
        let header = Header::from_bytes(data[..64].try_into().unwrap());
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, VERSION);
        assert_eq!(header.sequence_lo, 0);
        assert_eq!(header.sequence_hi, 10);
        // 2 components + 1 entity page = 3 page count at minimum.
        assert!(header.page_count >= 3);
    }

    #[test]
    fn header_crc_is_valid() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        let dir = tempfile::tempdir().unwrap();
        let path = flush(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(1u64)).unwrap(),
            dir.path(),
        )
        .unwrap()
        .unwrap();
        let data = std::fs::read(&path).unwrap();

        let stored_crc = u32::from_le_bytes(data[40..44].try_into().unwrap());
        let computed_crc = crc32fast::hash(&data[..40]);
        assert_eq!(stored_crc, computed_crc);
    }

    #[test]
    fn entry_observer_fires_once_per_entity_id() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 0.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 1.0, y: 1.0 },));
        let e3 = world.spawn((Pos { x: 2.0, y: 2.0 },));

        let observed: Rc<RefCell<Vec<u64>>> = Rc::new(RefCell::new(Vec::new()));
        let observed_clone = Rc::clone(&observed);

        let dir = tempfile::tempdir().unwrap();
        let result = flush_observed(
            &world,
            SeqRange::new(SeqNo::from(1u64), SeqNo::from(5u64)).unwrap(),
            dir.path(),
            Some(&mut |key: EntityKey| {
                observed_clone.borrow_mut().push(key.0);
            }),
        )
        .unwrap();

        assert!(result.is_some(), "expected a run to be written");

        let seen = observed.borrow();
        assert_eq!(seen.len(), 3, "observer must fire exactly once per entity");
        assert!(seen.contains(&e1.to_bits()), "e1 not observed");
        assert!(seen.contains(&e2.to_bits()), "e2 not observed");
        assert!(seen.contains(&e3.to_bits()), "e3 not observed");
    }
}
