//! Merge-kernel for LSM compaction: emit-list construction and write loop.
//!
//! # Overview
//!
//! [`build_emit_list`] (Task 3a) is a pure-logic computation that decides,
//! for each unique entity, which source run and row to copy. No file I/O.
//!
//! [`CompactionWriter`] (Tasks 3b + 3c) drives the full compaction pipeline:
//! 1. Call `build_emit_list` to determine which rows to emit.
//! 2. Write those rows into a new sorted-run file whose format is byte-compatible
//!    with `SortedRunReader::open` (same header / page / index / footer layout as
//!    `FlushWriter`).
//! 3. Return a `SortedRunMeta` the caller can record via `ManifestLog`.
//!
//! # Format compatibility
//!
//! The output format mirrors `writer::flush_observed` exactly:
//! - Header (64 bytes) at offset 0
//! - Schema section immediately after the header
//! - Component pages in `(arch_id=0, slot, page_index)` order
//! - Entity pages in `(arch_id=0, ENTITY_SLOT, page_index)` order
//! - Sparse index (sorted `IndexEntry` array)
//! - Footer (64 bytes) at end of file
//! - Header CRC32 and total CRC32 patched in-place after write

use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::LsmError;
use crate::format::{
    ENTITY_SLOT, Footer, Header, IndexEntry, MAGIC, PAGE_SIZE, PageHeader, VERSION,
};
use crate::manifest::SortedRunMeta;
use crate::reader::SortedRunReader;
use crate::schema::SchemaSection;
use crate::types::{PageCount, SeqRange, SizeBytes};

// ── Public types (shared between emit-list builder and write loop) ───────────

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

// ── CompactionWriter ─────────────────────────────────────────────────────────

/// Writes a single compacted sorted-run file from a set of input runs.
///
/// ## Format contract
///
/// The output file is byte-compatible with `SortedRunReader::open`. It uses
/// `arch_id = 0` for the single output archetype regardless of the source
/// `arch_id` values in the input runs.
///
/// ## Per-input slot translation
///
/// Different input runs may assign different slot indices to the same
/// component (each run is independent). `CompactionWriter` computes a
/// per-input `(input_slot) -> output_slot` table up front from the schema
/// section of each input, keyed on the component's stable name.
// `CompactionWriter` is constructed by the compactor (Task 3d / 4). The
// dead_code lint fires now because no caller outside cfg(test) exists yet.
#[allow(dead_code)]
pub(crate) struct CompactionWriter<'a> {
    inputs: Vec<&'a SortedRunReader>,
    arch_ids_per_input: Vec<Option<u16>>,
    /// Sorted stable component names for the target archetype.
    target_components: Vec<String>,
    output_path: PathBuf,
    /// WAL sequence range for the output run: `(min(input.lo), max(input.hi))`.
    output_seq_range: SeqRange,
}

// Methods are called from the compactor (Task 3d / 4) and from cfg(test).
// The dead_code lint fires here because no non-test caller exists yet.
#[allow(dead_code)]
impl<'a> CompactionWriter<'a> {
    /// Create a new `CompactionWriter`.
    ///
    /// Returns `Err(LsmError::Format)` if:
    /// - `inputs.len() != arch_ids_per_input.len()`
    /// - `target_components` is empty
    pub(crate) fn new(
        inputs: Vec<&'a SortedRunReader>,
        arch_ids_per_input: Vec<Option<u16>>,
        target_components: Vec<String>,
        output_path: PathBuf,
        output_seq_range: SeqRange,
    ) -> Result<Self, LsmError> {
        if inputs.len() != arch_ids_per_input.len() {
            return Err(LsmError::Format(format!(
                "CompactionWriter: inputs length {} != arch_ids_per_input length {}",
                inputs.len(),
                arch_ids_per_input.len(),
            )));
        }
        if target_components.is_empty() {
            return Err(LsmError::Format(
                "CompactionWriter: target_components is empty".to_owned(),
            ));
        }
        Ok(Self {
            inputs,
            arch_ids_per_input,
            target_components,
            output_path,
            output_seq_range,
        })
    }

    /// Run the full compaction pipeline and write the output sorted-run file.
    ///
    /// Steps:
    /// 1. Build the emit list (dedup by entity ID, newest-first wins).
    /// 2. Sort the emit list by `entity_id` for deterministic output.
    /// 3. Compute the output schema (same stable names, sequential slots).
    /// 4. Write header + schema + component pages + entity pages + index + footer.
    /// 5. Patch CRCs, fsync, rename atomically, fsync directory.
    /// 6. Return `SortedRunMeta` for the caller to record.
    ///
    /// Returns `Err(LsmError::Format("cannot compact: emit list empty"))` if no
    /// rows survive dedup — the caller should not compact an empty set.
    pub(crate) fn write(self) -> Result<SortedRunMeta, LsmError> {
        self.write_observed(None)
    }

    /// Like [`write`], but invokes `observer` once per entity ID written to an
    /// entity-slot page. Pass `None` for no observation (identical to [`write`]).
    ///
    /// The observer fires *after* the entity bytes are successfully written to
    /// the buffer, so it will not fire for pages that are skipped due to zero
    /// `row_count`. It fires exactly once per entity in the output.
    pub(crate) fn write_observed(
        self,
        mut observer: Option<&mut dyn FnMut(crate::writer::EntityKey)>,
    ) -> Result<SortedRunMeta, LsmError> {
        // ── 1. Build and sort the emit list ──────────────────────────────────
        let mut emit_list = build_emit_list(self.inputs.as_slice(), &self.arch_ids_per_input)?;

        if emit_list.is_empty() {
            return Err(LsmError::Format(
                "cannot compact: emit list empty".to_owned(),
            ));
        }

        emit_list.sort_by_key(|r| r.entity_id);

        let row_count = emit_list.len();

        // ── 2. Build output schema ────────────────────────────────────────────
        // We need an item_size and item_align for each target component. Retrieve
        // these from the first input that has the archetype and the component.
        let output_schema = self.build_output_schema()?;

        // ── 3. Build per-input slot translation tables ────────────────────────
        // For each input, map output_slot -> input_slot (for reading source bytes).
        // `slot_translations[input_idx][output_slot]` = input_slot (or None if the
        // input doesn't carry that component for this archetype).
        let slot_translations = self.build_slot_translations(&output_schema);

        // ── 4. Compute page count: component pages + entity pages ─────────────
        let pages_per_column = row_count.div_ceil(PAGE_SIZE);
        let column_count = output_schema.len(); // entity-slot is separate
        let total_pages = column_count * pages_per_column + pages_per_column;
        // ^^ component columns × pages + entity column pages

        // ── 5. Open temp file ─────────────────────────────────────────────────
        let seq_lo = self.output_seq_range.lo().get();
        let seq_hi = self.output_seq_range.hi().get();
        let tmp_path = make_tmp_path(&self.output_path, seq_lo, seq_hi);

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;

        // Drop guard — cleans up tmp on error.
        struct TmpGuard<'p> {
            path: &'p Path,
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

        // ── 6. Write header (CRC patched later) ───────────────────────────────
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            schema_count: output_schema.len() as u32,
            page_count: total_pages as u64,
            sequence_lo: seq_lo,
            sequence_hi: seq_hi,
            header_crc32: 0,
            reserved: [0u8; 20],
        };
        w.write_all(header.as_bytes())?;

        // ── 7. Write schema section (immediately after header) ────────────────
        let schema_offset = std::mem::size_of::<Header>() as u64;
        output_schema.write_to(&mut w)?;

        // ── 8. Write component pages (arch_id=0, slot, page_index) ───────────
        let mut index_entries: Vec<IndexEntry> = Vec::with_capacity(total_pages);

        // One output archetype: arch_id = 0.
        const OUTPUT_ARCH_ID: u16 = 0;

        for entry in output_schema.entries() {
            let output_slot = entry.slot();
            let item_size = entry.item_size() as usize;

            for page_idx_usize in 0..pages_per_column {
                let page_idx = u16::try_from(page_idx_usize).map_err(|_| {
                    LsmError::Format(format!(
                        "page_index {page_idx_usize} exceeds u16 — too many rows"
                    ))
                })?;

                let row_start = page_idx_usize * PAGE_SIZE;
                let row_end = (row_start + PAGE_SIZE).min(row_count);
                let rows_in_page = row_end - row_start;
                let row_count_u16 = u16::try_from(rows_in_page)
                    .map_err(|_| LsmError::Format("row count per page exceeds u16".to_owned()))?;

                // Assemble page bytes by copying from source inputs.
                let mut page_bytes: Vec<u8> = Vec::with_capacity(PAGE_SIZE * item_size);

                for emit in &emit_list[row_start..row_end] {
                    let input_idx = emit.source_input_idx;
                    let source_row = emit.source_row;

                    // Find the input slot for this output_slot in this input.
                    let input_slot_opt = slot_translations[input_idx]
                        .iter()
                        .find(|t| t.output_slot == output_slot)
                        .map(|t| t.input_slot);

                    let Some(input_slot) = input_slot_opt else {
                        // Component absent in this input — emit zeros.
                        page_bytes.extend(std::iter::repeat_n(0u8, item_size));
                        continue;
                    };

                    let source_page_idx = u16::try_from(source_row / PAGE_SIZE).map_err(|_| {
                        LsmError::Format("source page index exceeds u16".to_owned())
                    })?;
                    let row_within_page = source_row % PAGE_SIZE;

                    let arch_id_in_input = self.arch_ids_per_input[input_idx]
                        .expect("emit row references input with no arch_id");

                    let page_ref = self.inputs[input_idx]
                        .get_page(arch_id_in_input, input_slot, source_page_idx)?
                        .ok_or_else(|| {
                            LsmError::Format(format!(
                                "source page ({arch_id_in_input}, {input_slot}, \
                                 {source_page_idx}) not found in input {input_idx}"
                            ))
                        })?;

                    let data = page_ref.data();
                    let byte_offset = row_within_page * item_size;
                    page_bytes.extend_from_slice(&data[byte_offset..byte_offset + item_size]);
                }

                // CRC covers the actual data bytes (not zero-padding).
                let page_crc = crc32fast::hash(&page_bytes);
                let file_offset = w.stream_position()?;

                let ph = PageHeader {
                    arch_id: OUTPUT_ARCH_ID,
                    slot: output_slot,
                    page_index: page_idx,
                    row_count: row_count_u16,
                    page_crc32: page_crc,
                    _padding: 0,
                };
                w.write_all(ph.as_bytes())?;
                w.write_all(&page_bytes)?;

                // Zero-pad to full page.
                let full_page_bytes = PAGE_SIZE * item_size;
                if page_bytes.len() < full_page_bytes {
                    write_zeros(&mut w, full_page_bytes - page_bytes.len())?;
                }

                index_entries.push(IndexEntry {
                    arch_id: OUTPUT_ARCH_ID,
                    slot: output_slot,
                    page_index: page_idx,
                    _pad: 0,
                    file_offset,
                });
            }
        }

        // ── 9. Write entity pages (arch_id=0, ENTITY_SLOT, page_index) ────────
        let entity_item_size = std::mem::size_of::<u64>();

        for page_idx_usize in 0..pages_per_column {
            let page_idx = u16::try_from(page_idx_usize).map_err(|_| {
                LsmError::Format(format!(
                    "page_index {page_idx_usize} exceeds u16 — too many rows"
                ))
            })?;

            let row_start = page_idx_usize * PAGE_SIZE;
            let row_end = (row_start + PAGE_SIZE).min(row_count);
            let rows_in_page = row_end - row_start;
            let row_count_u16 = u16::try_from(rows_in_page)
                .map_err(|_| LsmError::Format("row count per page exceeds u16".to_owned()))?;

            let mut entity_bytes: Vec<u8> = Vec::with_capacity(rows_in_page * entity_item_size);
            for emit in &emit_list[row_start..row_end] {
                entity_bytes.extend_from_slice(&emit.entity_id.to_le_bytes());
            }

            let page_crc = crc32fast::hash(&entity_bytes);
            let file_offset = w.stream_position()?;

            let ph = PageHeader {
                arch_id: OUTPUT_ARCH_ID,
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
                for emit in &emit_list[row_start..row_end] {
                    obs(crate::writer::EntityKey(emit.entity_id));
                }
            }

            // Zero-pad.
            let full_page_bytes = PAGE_SIZE * entity_item_size;
            if entity_bytes.len() < full_page_bytes {
                write_zeros(&mut w, full_page_bytes - entity_bytes.len())?;
            }

            index_entries.push(IndexEntry {
                arch_id: OUTPUT_ARCH_ID,
                slot: ENTITY_SLOT,
                page_index: page_idx,
                _pad: 0,
                file_offset,
            });
        }

        // ── 10. Write sparse index (sorted) ──────────────────────────────────
        index_entries.sort();
        let sparse_index_offset = w.stream_position()?;
        for entry in &index_entries {
            w.write_all(entry.as_bytes())?;
        }

        // ── 11. Write footer (CRCs patched later) ─────────────────────────────
        let footer = Footer {
            sparse_index_offset,
            sparse_index_count: index_entries.len() as u64,
            schema_offset,
            bloom_filter_offset: 0,
            total_crc32: 0,
            reserved: [0u8; 28],
        };
        w.write_all(footer.as_bytes())?;
        w.flush()?;

        // ── 12. Patch CRCs ────────────────────────────────────────────────────
        let mut file = w
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;

        // Header CRC: covers first 40 bytes.
        file.seek(SeekFrom::Start(0))?;
        let mut header_bytes = [0u8; 64];
        {
            use std::io::Read;
            file.read_exact(&mut header_bytes)?;
        }
        let header_crc = crc32fast::hash(&header_bytes[..40]);
        file.seek(SeekFrom::Start(40))?;
        file.write_all(&header_crc.to_le_bytes())?;

        // Total CRC: covers entire file with total_crc32 field zeroed.
        // total_crc32 is at footer_offset + 32 (4 × u64 = 32 bytes into footer).
        let file_len = file.seek(SeekFrom::End(0))?;
        let footer_offset = file_len - 64;
        let total_crc32_file_offset = footer_offset + 32;

        file.seek(SeekFrom::Start(0))?;
        let mut all_bytes = vec![0u8; file_len as usize];
        {
            use std::io::Read;
            file.read_exact(&mut all_bytes)?;
        }
        let tco = total_crc32_file_offset as usize;
        all_bytes[tco..tco + 4].copy_from_slice(&[0, 0, 0, 0]);
        let total_crc = crc32fast::hash(&all_bytes);

        file.seek(SeekFrom::Start(total_crc32_file_offset))?;
        file.write_all(&total_crc.to_le_bytes())?;
        file.sync_all()?;
        drop(file);

        // ── 13. Atomic rename + dir fsync ─────────────────────────────────────
        fs::rename(&tmp_path, &self.output_path)?;

        let parent = self
            .output_path
            .parent()
            .ok_or_else(|| LsmError::Format("output_path has no parent".to_owned()))?;
        let dir = fs::File::open(parent)?;
        dir.sync_all()?;

        guard.disarmed = true;

        // ── 14. Build and return SortedRunMeta ────────────────────────────────
        let size_bytes = fs::metadata(&self.output_path).map_or(0, |m| m.len());

        let page_count_val = u64::try_from(total_pages).unwrap_or(u64::MAX);
        let page_count = PageCount::new(page_count_val)
            .ok_or_else(|| LsmError::Format("compaction produced zero pages".to_owned()))?;

        SortedRunMeta::new(
            self.output_path.clone(),
            self.output_seq_range,
            vec![OUTPUT_ARCH_ID],
            page_count,
            SizeBytes::new(size_bytes),
        )
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Build the output schema from the target component names + layout info
    /// sourced from the first input that carries each component.
    fn build_output_schema(&self) -> Result<SchemaSection, LsmError> {
        let mut components: Vec<(String, std::alloc::Layout)> =
            Vec::with_capacity(self.target_components.len());

        'outer: for comp_name in &self.target_components {
            // Find item_size and item_align from the first input that has
            // this component (via the schema section of inputs with the arch).
            for (input_idx, &input) in self.inputs.iter().enumerate() {
                if self.arch_ids_per_input[input_idx].is_none() {
                    continue;
                }
                let schema = input.schema();
                if let Some(entry) = schema
                    .slot_for(comp_name)
                    .and_then(|s| schema.entry_for_slot(s))
                {
                    let size = entry.item_size() as usize;
                    let align = entry.item_align() as usize;
                    // Treat align == 0 as 1 (ZST convention).
                    let align_safe = if align == 0 { 1 } else { align };
                    let layout =
                        std::alloc::Layout::from_size_align(size, align_safe).map_err(|_| {
                            LsmError::Format(format!(
                                "invalid layout for component {comp_name}: \
                                 size={size} align={align}"
                            ))
                        })?;
                    components.push((comp_name.clone(), layout));
                    continue 'outer;
                }
            }
            return Err(LsmError::Format(format!(
                "component {comp_name} not found in any input schema"
            )));
        }

        SchemaSection::from_components(&components)
    }

    /// Build per-input slot translation tables.
    ///
    /// Returns `slot_translations[input_idx]` = a `Vec<SlotTranslation>` mapping
    /// output_slot → input_slot for that input run.
    fn build_slot_translations(&self, output_schema: &SchemaSection) -> Vec<Vec<SlotTranslation>> {
        let mut result: Vec<Vec<SlotTranslation>> = Vec::with_capacity(self.inputs.len());

        for (input_idx, &input) in self.inputs.iter().enumerate() {
            let mut translations: Vec<SlotTranslation> = Vec::new();

            if self.arch_ids_per_input[input_idx].is_some() {
                let input_schema = input.schema();
                for output_entry in output_schema.entries() {
                    let comp_name = output_entry.name();
                    if let Some(input_slot) = input_schema.slot_for(comp_name) {
                        translations.push(SlotTranslation {
                            output_slot: output_entry.slot(),
                            input_slot,
                        });
                    }
                    // If the input doesn't have this component, no translation
                    // entry — the write loop emits zeros for missing components.
                }
            }

            result.push(translations);
        }

        result
    }
}

/// Maps an output component slot to the corresponding input slot in one input run.
// Only constructed inside build_slot_translations which is called by
// CompactionWriter::write. Both are #[allow(dead_code)] until Task 3d/4 land.
#[allow(dead_code)]
struct SlotTranslation {
    output_slot: u16,
    input_slot: u16,
}

// ── File helpers ──────────────────────────────────────────────────────────────

/// Build the temp-file path alongside the final output path.
// Called by CompactionWriter::write — allow until external callers exist.
#[allow(dead_code)]
fn make_tmp_path(output_path: &Path, seq_lo: u64, seq_hi: u64) -> PathBuf {
    let parent = output_path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{seq_lo}-{seq_hi}.run.tmp"))
}

/// Write `n` zero bytes to `w`.
// Called by CompactionWriter::write — allow until external callers exist.
#[allow(dead_code)]
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

    // ── Helper: get component name from any single-slot archetype in reader ──

    /// Return (arch_id, sorted component-name list) for every archetype in a
    /// reader, ordered by arch_id.
    fn arch_component_names(reader: &SortedRunReader) -> Vec<(u16, Vec<String>)> {
        reader
            .archetype_ids()
            .into_iter()
            .map(|arch_id| {
                let slots = reader.component_slots_for_arch(arch_id);
                let names: Vec<String> = slots
                    .iter()
                    .map(|&s| {
                        reader
                            .schema()
                            .entry_for_slot(s)
                            .expect("slot must be in schema")
                            .name()
                            .to_owned()
                    })
                    .collect();
                (arch_id, names)
            })
            .collect()
    }

    // ── Task 3c test 1: compact two runs, all entities present ───────────────

    /// Flush World(E1,E2,E3) to run A, flush World(E4,E5) to run B,
    /// compact [A, B] into C, open C with SortedRunReader, verify all
    /// 5 entities present and their component values correct.
    ///
    /// E1–E3 come from world_a; E4–E5 from world_b (different worlds so IDs
    /// are distinct by construction: world_b spawns placeholders to advance
    /// its allocator past indices 0, 1, 2).
    #[test]
    fn compact_two_runs_preserves_all_entities() {
        // ── Build run A (E1, E2, E3) ─────────────────────────────────────────
        let mut world_a = World::new();
        let e1 = world_a.spawn((Pos { x: 1.0, y: 10.0 },));
        let e2 = world_a.spawn((Pos { x: 2.0, y: 20.0 },));
        let e3 = world_a.spawn((Pos { x: 3.0, y: 30.0 },));
        let (_dir_a, reader_a) = flush_to_reader(&world_a, 1, 10);

        // ── Build run B (E4, E5) ──────────────────────────────────────────────
        // Advance allocator past indices 0, 1, 2 (used by E1–E3) so E4/E5
        // get distinct IDs.
        let mut world_b = World::new();
        let ph1 = world_b.spawn((Pos { x: 0.0, y: 0.0 },));
        let ph2 = world_b.spawn((Pos { x: 0.0, y: 0.0 },));
        let ph3 = world_b.spawn((Pos { x: 0.0, y: 0.0 },));
        world_b.despawn(ph1);
        world_b.despawn(ph2);
        world_b.despawn(ph3);
        let e4 = world_b.spawn((Pos { x: 4.0, y: 40.0 },));
        let e5 = world_b.spawn((Pos { x: 5.0, y: 50.0 },));
        let (_dir_b, reader_b) = flush_to_reader(&world_b, 11, 20);

        // ── Discover component name ───────────────────────────────────────────
        let arch_info_a = arch_component_names(&reader_a);
        assert_eq!(arch_info_a.len(), 1);
        let (arch_a, comp_names_a) = &arch_info_a[0];
        assert_eq!(comp_names_a.len(), 1);
        let comp_name = &comp_names_a[0];

        let arch_b = find_archetype_by_components(&reader_b, &[comp_name.as_str()])
            .expect("run B must have the Pos archetype");

        // ── Compact A + B → C (newest-first: B then A) ───────────────────────
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("1-20.compact.run");
        let seq_range = SeqRange::new(SeqNo::from(1u64), SeqNo::from(20u64)).unwrap();

        let writer = CompactionWriter::new(
            vec![&reader_b, &reader_a],
            vec![Some(arch_b), Some(*arch_a)],
            vec![comp_name.clone()],
            out_path.clone(),
            seq_range,
        )
        .unwrap();

        let meta = writer.write().unwrap();

        // ── Open and validate output ──────────────────────────────────────────
        let out_reader = SortedRunReader::open(&out_path).unwrap();
        assert_eq!(out_reader.sequence_range(), seq_range);
        assert_eq!(out_reader.schema().len(), 1, "one component in schema");

        // Every reachable page CRC must be valid.  Walk entity-slot pages and
        // component-slot pages via the public API (no access to private `index`).
        let mut found_entities: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut page_idx: u16 = 0;
        while let Ok(Some(page)) = out_reader.get_page(0, ENTITY_SLOT, page_idx) {
            out_reader.validate_page_crc(&page).unwrap();
            let row_count = page.header().row_count as usize;
            let data = page.data();
            for r in 0..row_count {
                let off = r * 8;
                let id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                found_entities.insert(id);
            }
            page_idx += 1;
        }
        // Validate component-slot pages.
        for &slot in &out_reader.component_slots_for_arch(0) {
            let mut pg: u16 = 0;
            while let Ok(Some(page)) = out_reader.get_page(0, slot, pg) {
                out_reader.validate_page_crc(&page).unwrap();
                pg += 1;
            }
        }

        assert!(
            found_entities.contains(&e1.to_bits()),
            "E1 missing from output"
        );
        assert!(
            found_entities.contains(&e2.to_bits()),
            "E2 missing from output"
        );
        assert!(
            found_entities.contains(&e3.to_bits()),
            "E3 missing from output"
        );
        assert!(
            found_entities.contains(&e4.to_bits()),
            "E4 missing from output"
        );
        assert!(
            found_entities.contains(&e5.to_bits()),
            "E5 missing from output"
        );
        assert_eq!(found_entities.len(), 5, "exactly 5 unique entities");

        // Verify SortedRunMeta fields.
        assert_eq!(meta.archetype_coverage(), &[0u16]);
        assert!(meta.size_bytes().get() > 0);
    }

    // ── Task 3c test 2: newer version wins after compaction ──────────────────

    /// Flush entity E with Pos.x = 1.0 → run OLD.
    /// Flush same world again (still dirty) with Pos.x effectively the same
    /// value (we verify the entity ID is present from the newer run).
    ///
    /// For a sharper test: use two separate worlds where E exists in both but
    /// the newer world has a different x value — verify we keep the newer value.
    #[test]
    fn compact_entity_update_keeps_newest_version() {
        // ── Run OLD: entity E with Pos.x = 1.0 ───────────────────────────────
        let mut world_old = World::new();
        let e = world_old.spawn((Pos { x: 1.0, y: 0.0 },));
        let (_dir_old, reader_old) = flush_to_reader(&world_old, 1, 5);

        // ── Run NEW: same entity bits (re-created in a new world at index 0)
        //    but with Pos.x = 2.0 to represent the updated value.
        //
        //    We construct a world where the entity ID matches e.to_bits() by
        //    spawning at the same index (first spawn) and same generation (1).
        //    Since World::new() always starts at generation 1, the first spawn
        //    in world_new will produce the same bits as the first spawn in
        //    world_old.
        let mut world_new = World::new();
        let e_new = world_new.spawn((Pos { x: 2.0, y: 0.0 },));
        // Verify the IDs match (same index 0, generation 1).
        assert_eq!(
            e.to_bits(),
            e_new.to_bits(),
            "entity IDs must match for the update test to be meaningful"
        );
        let (_dir_new, reader_new) = flush_to_reader(&world_new, 6, 10);

        // ── Discover component name ───────────────────────────────────────────
        let arch_info = arch_component_names(&reader_old);
        assert_eq!(arch_info.len(), 1);
        let (arch_old, comp_names) = &arch_info[0];
        let comp_name = &comp_names[0];

        let arch_new = find_archetype_by_components(&reader_new, &[comp_name.as_str()])
            .expect("run NEW must have Pos archetype");

        // ── Compact NEW + OLD (newest first) → output ─────────────────────────
        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("1-10.compact.run");
        let seq_range = SeqRange::new(SeqNo::from(1u64), SeqNo::from(10u64)).unwrap();

        let writer = CompactionWriter::new(
            vec![&reader_new, &reader_old],
            vec![Some(arch_new), Some(*arch_old)],
            vec![comp_name.clone()],
            out_path.clone(),
            seq_range,
        )
        .unwrap();

        writer.write().unwrap();

        // ── Open output and read entity data ──────────────────────────────────
        let out_reader = SortedRunReader::open(&out_path).unwrap();

        // Validate all page CRCs via the public get_page API.
        {
            let mut pg: u16 = 0;
            while let Ok(Some(page)) = out_reader.get_page(0, ENTITY_SLOT, pg) {
                out_reader.validate_page_crc(&page).unwrap();
                pg += 1;
            }
            for &slot in &out_reader.component_slots_for_arch(0) {
                let mut pg: u16 = 0;
                while let Ok(Some(page)) = out_reader.get_page(0, slot, pg) {
                    out_reader.validate_page_crc(&page).unwrap();
                    pg += 1;
                }
            }
        }

        // Find which row in the entity-slot page corresponds to entity E.
        let entity_bits = e.to_bits();
        let mut entity_row: Option<usize> = None;
        let mut search_page: u16 = 0;
        'search: while let Ok(Some(page)) = out_reader.get_page(0, ENTITY_SLOT, search_page) {
            let row_count = page.header().row_count as usize;
            let data = page.data();
            for r in 0..row_count {
                let off = r * 8;
                let id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                if id == entity_bits {
                    entity_row = Some(search_page as usize * PAGE_SIZE + r);
                    break 'search;
                }
            }
            search_page += 1;
        }
        let entity_row = entity_row.expect("entity E must be in the output");

        // Read the Pos component for this row.
        let comp_slot = out_reader
            .schema()
            .slot_for(comp_name)
            .expect("component must be in output schema");
        let comp_page_idx = u16::try_from(entity_row / PAGE_SIZE).unwrap();
        let row_in_page = entity_row % PAGE_SIZE;
        let item_size = out_reader
            .schema()
            .entry_for_slot(comp_slot)
            .unwrap()
            .item_size() as usize;

        let comp_page = out_reader
            .get_page(0, comp_slot, comp_page_idx)
            .unwrap()
            .expect("component page must exist");

        let byte_off = row_in_page * item_size;
        let raw = &comp_page.data()[byte_off..byte_off + item_size];

        // Pos is {x: f32, y: f32} — 8 bytes. x is the first 4 bytes (LE).
        assert_eq!(item_size, 8, "Pos must be 8 bytes");
        let x = f32::from_le_bytes(raw[..4].try_into().unwrap());

        assert!(
            (x - 2.0_f32).abs() < f32::EPSILON,
            "expected Pos.x == 2.0 from newer run, got {x}"
        );
    }
}
