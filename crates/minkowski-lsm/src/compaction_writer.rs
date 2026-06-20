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

use crate::bloom;
use crate::error::LsmError;
use crate::format::{
    ALLOCATOR_SLOT, ENTITY_SLOT, Footer, Header, IndexEntry, MAGIC, META_ARCH_ID, PAGE_SIZE,
    PageHeader, VERSION,
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

        // Walk entity-slot pages for this archetype via the sparse index.
        // Using entity_pages (not sequential get_page probing) correctly
        // handles non-contiguous page indices — e.g. pages 0 and 5 flushed
        // with gaps in between. Sequential probing would break at the first
        // gap and silently drop entities in higher pages.
        for result in input.entity_pages(arch_id) {
            let (page_index, page) = result?;

            let row_count = page.header().row_count as usize;
            let data = page.data();

            for row_within_page in 0..row_count {
                let byte_offset = row_within_page * 8;
                // SAFETY (invariant): entity_pages guarantees data.len() ==
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
/// sequential `arch_id`s (0, 1, 2, …) for the output archetypes, one per
/// entry in `all_signatures`, regardless of the source `arch_id` values in
/// the input runs.
///
/// ## Multi-archetype support
///
/// When input runs contain multiple archetypes, the writer preserves **all**
/// of them in the output. This prevents data loss when the input runs are
/// removed from the manifest after compaction.
///
/// ## Per-input slot translation
///
/// Different input runs may assign different slot indices to the same
/// component (each run is independent). `CompactionWriter` computes a
/// per-input `(input_slot) -> output_slot` table up front from the schema
/// section of each input, keyed on the component's stable name.
#[allow(dead_code)]
pub(crate) struct CompactionWriter<'a> {
    inputs: Vec<&'a SortedRunReader>,
    /// For each signature, for each input, the arch_id in that input (or None).
    arch_ids_per_signature_per_input: Vec<Vec<Option<u16>>>,
    /// All archetype signatures (sorted component names) in the output.
    all_signatures: Vec<Vec<String>>,
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
    /// - `inputs.len() != arch_ids_per_signature_per_input[i].len()` for any signature
    /// - `all_signatures` is empty
    pub(crate) fn new(
        inputs: Vec<&'a SortedRunReader>,
        arch_ids_per_signature_per_input: Vec<Vec<Option<u16>>>,
        all_signatures: Vec<Vec<String>>,
        output_path: PathBuf,
        output_seq_range: SeqRange,
    ) -> Result<Self, LsmError> {
        if all_signatures.is_empty() {
            return Err(LsmError::Format(
                "CompactionWriter: all_signatures is empty".to_owned(),
            ));
        }
        for (sig_idx, per_input) in arch_ids_per_signature_per_input.iter().enumerate() {
            if per_input.len() != inputs.len() {
                return Err(LsmError::Format(format!(
                    "CompactionWriter: signature {sig_idx} has {} arch_ids but {} inputs",
                    per_input.len(),
                    inputs.len(),
                )));
            }
        }
        if arch_ids_per_signature_per_input.len() != all_signatures.len() {
            return Err(LsmError::Format(format!(
                "CompactionWriter: {} signature arch-id vectors but {} signatures",
                arch_ids_per_signature_per_input.len(),
                all_signatures.len(),
            )));
        }
        Ok(Self {
            inputs,
            arch_ids_per_signature_per_input,
            all_signatures,
            output_path,
            output_seq_range,
        })
    }

    /// Run the full compaction pipeline and write the output sorted-run file.
    ///
    /// Steps:
    /// 1. Build the emit list for each archetype (dedup by entity ID, newest-first wins).
    /// 2. Sort each emit list by `entity_id` for deterministic output.
    /// 3. Compute the output schema (union of all components across all signatures).
    /// 4. Build per-archetype slot translation tables.
    /// 5. Write header + schema + component pages + entity pages + index + footer.
    /// 6. Patch CRCs, fsync, rename atomically, fsync directory.
    /// 7. Return `SortedRunMeta` for the caller to record.
    ///
    /// Returns `Err(LsmError::Format("cannot compact: all emit lists empty"))` if no
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
        // ── 1. Build and sort emit lists for each archetype ──────────────────
        let emit_lists: Vec<Vec<EmitRow>> = self
            .all_signatures
            .iter()
            .enumerate()
            .map(|(sig_idx, _)| {
                build_emit_list(
                    self.inputs.as_slice(),
                    &self.arch_ids_per_signature_per_input[sig_idx],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        if emit_lists.iter().all(Vec::is_empty) {
            return Err(LsmError::Format(
                "cannot compact: all emit lists empty".to_owned(),
            ));
        }

        let mut emit_lists = emit_lists;
        for list in &mut emit_lists {
            list.sort_by_key(|r| r.entity_id);
        }

        // ── 2. Build output schema (union of all components) ─────────────────
        let all_components = self.collect_all_components();
        let output_schema = self.build_output_schema(&all_components)?;

        // ── 3. Build per-archetype slot translation tables ───────────────────
        let slot_translations_per_sig: Vec<Vec<Vec<SlotTranslation>>> = self
            .all_signatures
            .iter()
            .map(|sig| self.build_slot_translations_for_sig(sig, &output_schema))
            .collect();

        // ── 4. Compute page counts ────────────────────────────────────────────
        let mut total_pages: usize = 0;
        let mut row_counts: Vec<usize> = Vec::with_capacity(emit_lists.len());
        let mut pages_per_column_per_sig: Vec<usize> = Vec::with_capacity(emit_lists.len());
        for (sig_idx, emit_list) in emit_lists.iter().enumerate() {
            let row_count = emit_list.len();
            row_counts.push(row_count);
            if row_count == 0 {
                pages_per_column_per_sig.push(0);
                continue;
            }
            let column_count = self.all_signatures[sig_idx].len();
            let pages_per_column = row_count.div_ceil(PAGE_SIZE);
            pages_per_column_per_sig.push(pages_per_column);
            total_pages += column_count * pages_per_column + pages_per_column;
        }

        if total_pages == 0 {
            return Err(LsmError::Format("cannot compact: zero pages".to_owned()));
        }

        // Allocator metadata: carry forward from the input run with the highest
        // seq_hi. Compaction has no live World, so without this the free list and
        // generations are lost once every flush that wrote them is compacted away
        // (recovery would then rebuild generations from live entities only).
        let allocator_pages: Vec<Vec<u8>> =
            match self.inputs.iter().max_by_key(|r| r.sequence_range().hi()) {
                Some(newest) => {
                    let mut pages = Vec::new();
                    for result in newest.slot_pages(META_ARCH_ID, ALLOCATOR_SLOT) {
                        let (_page_index, page) = result?;
                        newest.validate_page_crc(&page)?;
                        let len = page.header().row_count as usize;
                        pages.push(page.data()[..len].to_vec());
                    }
                    pages
                }
                None => Vec::new(),
            };
        total_pages += allocator_pages.len();

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

        // ── 8. Write component + entity pages per archetype ──────────────────
        let mut index_entries: Vec<IndexEntry> = Vec::with_capacity(total_pages);

        for (sig_idx, emit_list) in emit_lists.iter().enumerate() {
            let row_count = emit_list.len();
            if row_count == 0 {
                continue;
            }
            let output_arch_id = u16::try_from(sig_idx).map_err(|_| {
                LsmError::Format(format!(
                    "signature index {sig_idx} exceeds u16 — too many archetypes"
                ))
            })?;
            let pages_per_column = pages_per_column_per_sig[sig_idx];
            let sig = &self.all_signatures[sig_idx];
            let slot_translations = &slot_translations_per_sig[sig_idx];

            // Resolve output slots for this signature's components.
            let sig_output_slots: Vec<u16> = sig
                .iter()
                .filter_map(|name| output_schema.slot_for(name))
                .collect();

            // ── Component pages ──────────────────────────────────────────────
            for &output_slot in &sig_output_slots {
                let entry = output_schema.entry_for_slot(output_slot).ok_or_else(|| {
                    LsmError::Format(format!(
                        "output slot {output_slot} not in schema (arch {output_arch_id})"
                    ))
                })?;
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
                    let row_count_u16 = u16::try_from(rows_in_page).map_err(|_| {
                        LsmError::Format("row count per page exceeds u16".to_owned())
                    })?;

                    let mut page_bytes: Vec<u8> = Vec::with_capacity(PAGE_SIZE * item_size);

                    for emit in &emit_list[row_start..row_end] {
                        let input_idx = emit.source_input_idx;
                        let source_row = emit.source_row;

                        let input_slot_opt = slot_translations[input_idx]
                            .iter()
                            .find(|t| t.output_slot == output_slot)
                            .map(|t| t.input_slot);

                        let Some(input_slot) = input_slot_opt else {
                            page_bytes.extend(std::iter::repeat_n(0u8, item_size));
                            continue;
                        };

                        let source_page_idx =
                            u16::try_from(source_row / PAGE_SIZE).map_err(|_| {
                                LsmError::Format("source page index exceeds u16".to_owned())
                            })?;
                        let row_within_page = source_row % PAGE_SIZE;

                        let arch_id_in_input = self.arch_ids_per_signature_per_input[sig_idx]
                            [input_idx]
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

                    let page_crc = crc32fast::hash(&page_bytes);
                    let file_offset = w.stream_position()?;

                    let ph = PageHeader {
                        arch_id: output_arch_id,
                        slot: output_slot,
                        page_index: page_idx,
                        row_count: row_count_u16,
                        page_crc32: page_crc,
                        _padding: 0,
                    };
                    w.write_all(ph.as_bytes())?;
                    w.write_all(&page_bytes)?;

                    let full_page_bytes = PAGE_SIZE * item_size;
                    if page_bytes.len() < full_page_bytes {
                        write_zeros(&mut w, full_page_bytes - page_bytes.len())?;
                    }

                    index_entries.push(IndexEntry {
                        arch_id: output_arch_id,
                        slot: output_slot,
                        page_index: page_idx,
                        _pad: 0,
                        file_offset,
                    });
                }
            }

            // ── Entity pages ──────────────────────────────────────────────────
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
                    arch_id: output_arch_id,
                    slot: ENTITY_SLOT,
                    page_index: page_idx,
                    row_count: row_count_u16,
                    page_crc32: page_crc,
                    _padding: 0,
                };
                w.write_all(ph.as_bytes())?;
                w.write_all(&entity_bytes)?;

                if let Some(ref mut obs) = observer {
                    for emit in &emit_list[row_start..row_end] {
                        obs(crate::writer::EntityKey(emit.entity_id));
                    }
                }

                let full_page_bytes = PAGE_SIZE * entity_item_size;
                if entity_bytes.len() < full_page_bytes {
                    write_zeros(&mut w, full_page_bytes - entity_bytes.len())?;
                }

                index_entries.push(IndexEntry {
                    arch_id: output_arch_id,
                    slot: ENTITY_SLOT,
                    page_index: page_idx,
                    _pad: 0,
                    file_offset,
                });
            }
        }

        // ── 8b. Carry forward allocator metadata pages ───────────────────────
        for (page_index, payload) in allocator_pages.iter().enumerate() {
            let page_index = u16::try_from(page_index)
                .map_err(|_| LsmError::Format("allocator page_index exceeds u16".to_owned()))?;
            let page_crc = crc32fast::hash(payload);
            let file_offset = w.stream_position()?;
            let ph = PageHeader {
                arch_id: META_ARCH_ID,
                slot: ALLOCATOR_SLOT,
                page_index,
                // payload came from a validated page whose row_count is u16.
                row_count: payload.len() as u16,
                page_crc32: page_crc,
                _padding: 0,
            };
            w.write_all(ph.as_bytes())?;
            w.write_all(payload)?;
            index_entries.push(IndexEntry {
                arch_id: META_ARCH_ID,
                slot: ALLOCATOR_SLOT,
                page_index,
                _pad: 0,
                file_offset,
            });
        }

        // ── 9. Write sparse index (sorted) ──────────────────────────────────
        index_entries.sort();
        let sparse_index_offset = w.stream_position()?;
        for entry in &index_entries {
            w.write_all(entry.as_bytes())?;
        }

        // ── 9b. Bloom filter — built from all page keys in the index.
        let bloom_filter_offset = bloom::write_bloom_section(&mut w, &index_entries, seq_lo)?;

        // ── 10. Write footer (CRCs patched later) ─────────────────────────────
        let footer = Footer {
            sparse_index_offset,
            sparse_index_count: index_entries.len() as u64,
            schema_offset,
            bloom_filter_offset,
            total_crc32: 0,
            reserved: [0u8; 28],
        };
        w.write_all(footer.as_bytes())?;
        w.flush()?;

        // ── 11. Patch CRCs ────────────────────────────────────────────────────
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

        // ── 12. Atomic rename + dir fsync ─────────────────────────────────────
        fs::rename(&tmp_path, &self.output_path)?;

        let parent = self
            .output_path
            .parent()
            .ok_or_else(|| LsmError::Format("output_path has no parent".to_owned()))?;
        let dir = fs::File::open(parent)?;
        dir.sync_all()?;

        guard.disarmed = true;

        // ── 13. Build and return SortedRunMeta ────────────────────────────────
        let size_bytes = fs::metadata(&self.output_path).map_or(0, |m| m.len());

        let page_count_val = u64::try_from(total_pages).unwrap_or(u64::MAX);
        let page_count = PageCount::new(page_count_val)
            .ok_or_else(|| LsmError::Format("compaction produced zero pages".to_owned()))?;

        // archetype_coverage: sequential arch_ids 0..num_sigs_with_data
        let archetype_coverage: Vec<u16> = emit_lists
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.is_empty())
            .map(|(idx, _)| idx as u16)
            .collect();

        SortedRunMeta::new(
            self.output_path.clone(),
            self.output_seq_range,
            archetype_coverage,
            page_count,
            SizeBytes::new(size_bytes),
        )
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Collect all unique component names across all signatures, sorted.
    fn collect_all_components(&self) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut result: Vec<String> = Vec::new();
        for sig in &self.all_signatures {
            for name in sig {
                if seen.insert(name.clone()) {
                    result.push(name.clone());
                }
            }
        }
        result.sort();
        result
    }

    /// Build the output schema from the given component names + layout info
    /// sourced from the first input that carries each component.
    fn build_output_schema(&self, all_components: &[String]) -> Result<SchemaSection, LsmError> {
        let mut components: Vec<(String, std::alloc::Layout)> =
            Vec::with_capacity(all_components.len());

        'outer: for comp_name in all_components {
            for &input in &self.inputs {
                let Some(entry) = input
                    .schema()
                    .slot_for(comp_name)
                    .and_then(|s| input.schema().entry_for_slot(s))
                else {
                    continue;
                };
                let size = entry.item_size() as usize;
                let align = entry.item_align() as usize;
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
            return Err(LsmError::Format(format!(
                "component {comp_name} not found in any input schema"
            )));
        }

        SchemaSection::from_components(&components)
    }

    /// Build per-input slot translation tables for one signature.
    ///
    /// Returns `slot_translations[input_idx]` = a `Vec<SlotTranslation>` mapping
    /// output_slot → input_slot for that input run.
    fn build_slot_translations_for_sig(
        &self,
        sig: &[String],
        output_schema: &SchemaSection,
    ) -> Vec<Vec<SlotTranslation>> {
        let mut result: Vec<Vec<SlotTranslation>> = Vec::with_capacity(self.inputs.len());

        for &input in &self.inputs {
            let mut translations: Vec<SlotTranslation> = Vec::new();
            let input_schema = input.schema();
            for comp_name in sig {
                if let (Some(input_slot), Some(output_slot)) = (
                    input_schema.slot_for(comp_name),
                    output_schema.slot_for(comp_name),
                ) {
                    translations.push(SlotTranslation {
                        output_slot,
                        input_slot,
                    });
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
        let codecs = crate::codec::CodecRegistry::new();
        let path = flush(
            world,
            SeqRange::new(SeqNo::from(seq_lo), SeqNo::from(seq_hi)).unwrap(),
            dir.path(),
            &codecs,
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
            vec![vec![Some(arch_b), Some(*arch_a)]],
            vec![comp_names_a.clone()],
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
            vec![vec![Some(arch_new), Some(*arch_old)]],
            vec![comp_names.clone()],
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

    // ── Task 4 test: compaction rebuilds bloom filter ─────────────────────────

    #[test]
    fn compaction_rebuilds_bloom_filter() {
        let mut world = World::new();
        let _e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let _e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));

        let (_dir_a, reader_a) = flush_to_reader(&world, 1, 5);
        let (_dir_b, reader_b) = flush_to_reader(&world, 6, 10);

        let arch_info = arch_component_names(&reader_a);
        assert_eq!(arch_info.len(), 1);
        let (arch_a, comp_names) = &arch_info[0];
        let arch_b = find_archetype_by_components(&reader_b, &[comp_names[0].as_str()])
            .expect("run B must have the Pos archetype");

        let out_dir = tempfile::tempdir().unwrap();
        let out_path = out_dir.path().join("1-10.compact.run");
        let seq_range = SeqRange::new(SeqNo::from(1u64), SeqNo::from(10u64)).unwrap();

        let writer = CompactionWriter::new(
            vec![&reader_b, &reader_a],
            vec![vec![Some(arch_b), Some(*arch_a)]],
            vec![comp_names.clone()],
            out_path.clone(),
            seq_range,
        )
        .unwrap();

        writer.write().unwrap();

        let data = std::fs::read(&out_path).unwrap();
        let footer_start = data.len() - 64;
        let footer = crate::format::Footer::from_bytes(
            data[footer_start..footer_start + 64].try_into().unwrap(),
        );
        assert!(
            footer.bloom_filter_offset > 0,
            "compacted run must have a bloom filter"
        );
    }
}
