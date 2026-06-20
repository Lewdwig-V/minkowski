use std::fs;
use std::path::Path;

use crate::bloom::BloomView;
use crate::codec::CrcProof;
use crate::error::LsmError;
use crate::format::*;
use crate::schema::SchemaSection;
use crate::types::{SeqNo, SeqRange};

/// A reference to a page within a sorted run.
pub struct PageRef<'a> {
    /// Header describing the page metadata (owned copy, read from file).
    header: PageHeader,
    /// Raw page data bytes (full `PAGE_SIZE * item_size`, zero-padded).
    data: &'a [u8],
    /// Absolute byte offset of this page's header within the file.
    file_offset: u64,
}

impl<'a> PageRef<'a> {
    /// Header describing the page metadata.
    pub fn header(&self) -> &PageHeader {
        &self.header
    }

    /// Raw page data bytes (full `PAGE_SIZE * item_size`, zero-padded).
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Absolute byte offset of this page's header within the file.
    pub fn file_offset(&self) -> u64 {
        self.file_offset
    }
}

// ── Internal mmap abstraction ───────────────────────────────────────────────

enum MappedData {
    #[cfg(not(miri))]
    Mmap(memmap2::Mmap),
    /// Used under Miri (which cannot mmap) and as a fallback.
    #[expect(dead_code, reason = "constructed only under cfg(miri)")]
    Vec(Vec<u8>),
}

impl MappedData {
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(miri))]
            Self::Mmap(m) => m.as_ref(),
            Self::Vec(v) => v,
        }
    }
}

/// Reader for sorted run files. Memory-maps the file for zero-copy access.
pub struct SortedRunReader {
    data: MappedData,
    schema: SchemaSection,
    index: Vec<IndexEntry>,
    bloom: Option<BloomView>,
    sequence_range: SeqRange,
    page_count: u64,
}

/// Minimum file size: Header (64) + Footer (64).
const MIN_FILE_SIZE: usize = 128;

/// Parsed metadata extracted from a file buffer during validation.
struct ParsedMetadata {
    schema: SchemaSection,
    index: Vec<IndexEntry>,
    bloom: Option<BloomView>,
    sequence_range: SeqRange,
    page_count: u64,
}

/// Validate and parse a sorted-run file buffer.
///
/// Separated from `SortedRunReader::open` so that the borrow on `buf` does not
/// prevent moving `MappedData` into `Self`.
fn validate_and_parse(buf: &[u8]) -> Result<ParsedMetadata, LsmError> {
    // 2. Validate minimum size.
    if buf.len() < MIN_FILE_SIZE {
        return Err(LsmError::Format(format!(
            "file too small: {} bytes (minimum {MIN_FILE_SIZE})",
            buf.len(),
        )));
    }

    // 3. Read header, validate magic and version.
    let header = Header::from_bytes(buf[..64].try_into().expect("slice is 64 bytes"));

    if header.magic != MAGIC {
        return Err(LsmError::Format(format!(
            "bad magic: expected {MAGIC:?}, got {:?}",
            header.magic
        )));
    }
    if header.version != VERSION {
        return Err(LsmError::Format(format!(
            "unsupported version: expected {VERSION}, got {}",
            header.version
        )));
    }

    // 4. Validate header CRC (covers first 40 bytes).
    let computed_crc = crc32fast::hash(&buf[..40]);
    if header.header_crc32 != computed_crc {
        return Err(LsmError::Crc {
            offset: 0,
            expected: header.header_crc32,
            actual: computed_crc,
        });
    }

    let schema_count = header.schema_count;
    let sequence_range = SeqRange::new(
        SeqNo::from(header.sequence_lo),
        SeqNo::from(header.sequence_hi),
    )
    .map_err(|_| {
        LsmError::Format(format!(
            "header sequence range invalid: lo={} > hi={}",
            header.sequence_lo, header.sequence_hi
        ))
    })?;
    let page_count = header.page_count;

    // 5. Read footer (last 64 bytes).
    let footer_start = buf.len() - 64;
    let footer = Footer::from_bytes(
        buf[footer_start..footer_start + 64]
            .try_into()
            .expect("64 bytes"),
    );

    // 5b. Validate total CRC (covers entire file with total_crc32 field zeroed).
    // total_crc32 is at footer_start + 32 (4 * u64 = 32 bytes into footer).
    let total_crc32_offset = footer_start + 32;
    {
        let mut check_buf = buf.to_vec();
        check_buf[total_crc32_offset..total_crc32_offset + 4].copy_from_slice(&[0, 0, 0, 0]);
        let computed_total = crc32fast::hash(&check_buf);
        if footer.total_crc32 != computed_total {
            return Err(LsmError::Crc {
                offset: total_crc32_offset as u64,
                expected: footer.total_crc32,
                actual: computed_total,
            });
        }
    }

    // 6. Read schema section.
    let schema_start = footer.schema_offset as usize;
    if schema_start > buf.len() {
        return Err(LsmError::Format(
            "schema offset extends beyond file".to_owned(),
        ));
    }
    let schema_data = &buf[schema_start..];
    let schema = SchemaSection::read_from(schema_data, schema_count)?;

    // 7. Parse sparse index.
    let idx_offset = footer.sparse_index_offset as usize;
    let idx_count = footer.sparse_index_count as usize;
    let idx_byte_len = idx_count * std::mem::size_of::<IndexEntry>();

    if idx_offset + idx_byte_len > buf.len() {
        return Err(LsmError::Format(
            "sparse index extends beyond file".to_owned(),
        ));
    }

    let mut index = Vec::with_capacity(idx_count);
    for i in 0..idx_count {
        let entry_start = idx_offset + i * 16;
        let entry_bytes: &[u8; 16] = buf[entry_start..entry_start + 16]
            .try_into()
            .expect("16 bytes");
        index.push(IndexEntry::from_bytes(entry_bytes));
    }

    let bloom = if footer.bloom_filter_offset != 0 {
        let offset = footer.bloom_filter_offset as usize;
        Some(BloomView::from_bytes(buf, offset)?)
    } else {
        None
    };

    Ok(ParsedMetadata {
        schema,
        index,
        bloom,
        sequence_range,
        page_count,
    })
}

impl SortedRunReader {
    /// Open and validate a sorted run file.
    pub fn open(path: &Path) -> Result<Self, LsmError> {
        let data = Self::map_file(path)?;
        let parsed = validate_and_parse(data.as_slice())?;

        Ok(Self {
            data,
            schema: parsed.schema,
            index: parsed.index,
            bloom: parsed.bloom,
            sequence_range: parsed.sequence_range,
            page_count: parsed.page_count,
        })
    }

    /// Look up a page by `(arch_id, slot, page_index)`.
    ///
    /// Returns `Ok(None)` if the page is not in this sorted run's index.
    /// Returns `Err(LsmError::Format(...))` if the page is indexed but the
    /// file data is corrupt or out of bounds.
    pub fn get_page(
        &self,
        arch_id: u16,
        slot: u16,
        page_index: u16,
    ) -> Result<Option<PageRef<'_>>, LsmError> {
        let key = (arch_id, slot, page_index);
        let Ok(pos) = self
            .index
            .binary_search_by_key(&key, |e| (e.arch_id, e.slot, e.page_index))
        else {
            return Ok(None);
        };

        let entry = &self.index[pos];
        let buf = self.data.as_slice();
        let offset = entry.file_offset as usize;
        let header_size = std::mem::size_of::<PageHeader>();

        // Bounds-check header.
        let header_end = offset.checked_add(header_size).ok_or_else(|| {
            LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): header offset overflow"
            ))
        })?;
        if header_end > buf.len() {
            return Err(LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): header at offset {offset} extends beyond file"
            )));
        }
        let header_bytes: &[u8; 16] = buf[offset..header_end].try_into().expect("16 bytes");
        let header = PageHeader::from_bytes(header_bytes);

        // Compute and bounds-check data.
        let data_len =
            if slot == crate::format::ALLOCATOR_SLOT || arch_id == crate::format::SPARSE_ARCH_ID {
                // Sparse/allocator pages store a raw byte chunk in `row_count`;
                // their `slot` is a grouping index, not a schema slot, so do
                // NOT resolve it via `item_size_for_slot`.
                header.row_count as usize
            } else {
                let item_size = self.item_size_for_slot(slot)?;
                PAGE_SIZE.checked_mul(item_size).ok_or_else(|| {
                    LsmError::Format(format!(
                        "page ({arch_id}, {slot}, {page_index}): data length overflow"
                    ))
                })?
            };
        let data_start = offset.checked_add(header_size).ok_or_else(|| {
            LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): data start overflow"
            ))
        })?;
        let data_end = data_start.checked_add(data_len).ok_or_else(|| {
            LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): data end overflow"
            ))
        })?;
        if data_end > buf.len() {
            return Err(LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): data region [{data_start}..{data_end}] extends beyond file"
            )));
        }
        let data = &buf[data_start..data_end];

        Ok(Some(PageRef {
            header,
            data,
            file_offset: entry.file_offset,
        }))
    }

    /// Validate the CRC of a specific page and return a [`CrcProof`] on success.
    ///
    /// The returned token feeds into [`CodecRegistry::decode`]'s `raw_copy_size`
    /// fast path (direct memcpy, skipping rkyv bytecheck). The CRC covers
    /// `row_count * item_size` bytes — the actual data, not zero-padding.
    pub fn validate_page_crc(&self, page: &PageRef<'_>) -> Result<CrcProof, LsmError> {
        // Sparse pages store `chunk.len()` in `row_count` (byte count, not row
        // count), so item_size is always 1. This mirrors `page_ref_at`'s
        // `data_len` special-case for `SPARSE_ARCH_ID`.
        let item_size = if page.header().arch_id == crate::format::SPARSE_ARCH_ID {
            1
        } else {
            self.item_size_for_slot(page.header().slot)?
        };
        let actual_len = page.header().row_count as usize * item_size;
        let payload = &page.data()[..actual_len];

        CrcProof::verify(payload, page.header().page_crc32).ok_or_else(|| LsmError::Crc {
            offset: page.file_offset(),
            expected: page.header().page_crc32,
            actual: crc32fast::hash(payload),
        })
    }

    /// Get the schema section.
    pub fn schema(&self) -> &SchemaSection {
        &self.schema
    }

    /// WAL sequence range covered by this sorted run.
    pub fn sequence_range(&self) -> SeqRange {
        self.sequence_range
    }

    /// Total number of pages in this sorted run.
    pub fn page_count(&self) -> u64 {
        self.page_count
    }

    /// Iterate entity-slot pages for a given archetype, yielding
    /// `(page_index, PageRef)` pairs in ascending page_index order.
    ///
    /// Unlike sequential `get_page` probing (which breaks on the first gap),
    /// this scans the sparse index directly and visits every entity page
    /// present — even when pages are non-contiguous (e.g. only pages 0 and 5
    /// were dirty and flushed).
    pub fn entity_pages(
        &self,
        arch_id: u16,
    ) -> impl Iterator<Item = Result<(u16, PageRef<'_>), LsmError>> + '_ {
        self.slot_pages(arch_id, ENTITY_SLOT)
    }

    /// Iterate all pages for `(arch_id, slot)` from the sparse index.
    pub fn slot_pages(
        &self,
        arch_id: u16,
        slot: u16,
    ) -> impl Iterator<Item = Result<(u16, PageRef<'_>), LsmError>> + '_ {
        let key = (arch_id, slot, 0u16);
        let start = self
            .index
            .partition_point(|e| (e.arch_id, e.slot, e.page_index) < key);

        self.index[start..]
            .iter()
            .take_while(move |e| e.arch_id == arch_id && e.slot == slot)
            .map(move |entry| {
                let page_index = entry.page_index;
                self.page_ref_at(entry.file_offset, arch_id, slot, page_index)
                    .map(|page| (page_index, page))
            })
    }

    fn page_ref_at(
        &self,
        file_offset: u64,
        arch_id: u16,
        slot: u16,
        page_index: u16,
    ) -> Result<PageRef<'_>, LsmError> {
        let buf = self.data.as_slice();
        let offset = file_offset as usize;
        let header_size = std::mem::size_of::<PageHeader>();

        let header_end = offset.checked_add(header_size).ok_or_else(|| {
            LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): header offset overflow"
            ))
        })?;
        if header_end > buf.len() {
            return Err(LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): header at offset {offset} extends beyond file"
            )));
        }
        let header_bytes: &[u8; 16] = buf[offset..header_end].try_into().expect("16 bytes");
        let header = PageHeader::from_bytes(header_bytes);

        let data_len =
            if slot == crate::format::ALLOCATOR_SLOT || arch_id == crate::format::SPARSE_ARCH_ID {
                // Sparse/allocator pages store a raw byte chunk in `row_count`;
                // their `slot` is a grouping index, not a schema slot, so do
                // NOT resolve it via `item_size_for_slot`.
                header.row_count as usize
            } else {
                let item_size = self.item_size_for_slot(slot)?;
                PAGE_SIZE.checked_mul(item_size).ok_or_else(|| {
                    LsmError::Format(format!(
                        "page ({arch_id}, {slot}, {page_index}): data length overflow"
                    ))
                })?
            };
        let data_start = offset.checked_add(header_size).ok_or_else(|| {
            LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): data start overflow"
            ))
        })?;
        let data_end = data_start.checked_add(data_len).ok_or_else(|| {
            LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): data end overflow"
            ))
        })?;
        if data_end > buf.len() {
            return Err(LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): data region [{data_start}..{data_end}] extends beyond file"
            )));
        }
        let data = &buf[data_start..data_end];

        Ok(PageRef {
            header,
            data,
            file_offset,
        })
    }

    /// Number of entries in the sparse index.
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Check if a page *might* be present in this sorted run.
    ///
    /// Returns `true` if the page might exist (could be a false positive).
    /// Returns `false` if the page is definitely not present (no false negatives).
    ///
    /// When no bloom filter exists (old files without one), returns `true`
    /// conservatively — the caller should fall through to `get_page` or
    /// binary search.
    pub fn contains_page(&self, arch_id: u16, slot: u16, page_index: u16) -> bool {
        match &self.bloom {
            Some(view) => view.contains(crate::bloom::pack_page_key(arch_id, slot, page_index)),
            None => true,
        }
    }

    /// Returns the sorted, deduplicated list of archetype IDs present in this sorted run.
    ///
    /// Excludes [`META_ARCH_ID`] metadata pages (e.g. allocator state).
    pub fn archetype_ids(&self) -> Vec<u16> {
        let mut ids: Vec<u16> = self
            .index
            .iter()
            .map(|e| e.arch_id)
            .filter(|&id| id != crate::format::META_ARCH_ID && id != crate::format::SPARSE_ARCH_ID)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Returns the sorted, deduplicated list of component slot indices used by
    /// `arch_id` in this sorted run, excluding the entity pseudo-slot
    /// (`ENTITY_SLOT = 0xFFFF`).
    ///
    /// Returns an empty `Vec` if the archetype is not present in the index.
    // Used by `schema_match` (Task 2) and will be used directly by the
    // compactor (Task 3). The dead_code lint fires on the lib target because
    // the only current callers are in cfg(test); allow it until Task 3 lands.
    #[allow(dead_code)]
    pub(crate) fn component_slots_for_arch(&self, arch_id: u16) -> Vec<u16> {
        let mut slots: Vec<u16> = self
            .index
            .iter()
            .filter(|e| e.arch_id == arch_id && e.slot != ENTITY_SLOT)
            .map(|e| e.slot)
            .collect();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// Item size in bytes for a given slot.
    fn item_size_for_slot(&self, slot: u16) -> Result<usize, LsmError> {
        if slot == crate::format::ALLOCATOR_SLOT {
            Ok(1)
        } else if slot == ENTITY_SLOT {
            Ok(std::mem::size_of::<u64>())
        } else {
            self.schema
                .entry_for_slot(slot)
                .map(|e| e.item_size as usize)
                .ok_or_else(|| LsmError::Format(format!("unknown slot {slot} not found in schema")))
        }
    }

    /// Memory-map the file (or read into Vec under Miri).
    fn map_file(path: &Path) -> Result<MappedData, LsmError> {
        #[cfg(miri)]
        {
            let bytes = fs::read(path)?;
            Ok(MappedData::Vec(bytes))
        }
        #[cfg(not(miri))]
        {
            let file = fs::File::open(path)?;
            // SAFETY: The file is opened read-only and we do not modify it.
            // The mapping is valid for the lifetime of `SortedRunReader`.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            Ok(MappedData::Mmap(mmap))
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::flush;
    use minkowski::World;
    use std::path::PathBuf;

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

    /// Helper: spawn entities and flush to a sorted run file.
    fn flush_world_with_pos(n: usize) -> (tempfile::TempDir, std::path::PathBuf, World) {
        let mut world = World::new();
        for i in 0..n {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32 * 2.0,
            },));
        }
        let dir = tempfile::tempdir().unwrap();
        let codecs = crate::codec::CodecRegistry::new();
        let path = flush(
            &world,
            SeqRange::new(SeqNo::from(10u64), SeqNo::from(20u64)).unwrap(),
            dir.path(),
            &codecs,
        )
        .unwrap()
        .unwrap();
        (dir, path, world)
    }

    #[test]
    fn open_valid_file() {
        let (_dir, path, _world) = flush_world_with_pos(5);
        let reader = SortedRunReader::open(&path).unwrap();

        assert_eq!(
            reader.sequence_range(),
            SeqRange::new(SeqNo::from(10u64), SeqNo::from(20u64)).unwrap()
        );
        assert_eq!(reader.schema().len(), 1); // Pos only
        assert!(reader.page_count() > 0);
        assert!(reader.index_len() > 0);
    }

    #[test]
    fn get_page_returns_data() {
        let (_dir, path, _world) = flush_world_with_pos(3);
        let reader = SortedRunReader::open(&path).unwrap();

        // Find the first index entry to know a valid key.
        assert!(reader.index_len() > 0);
        let entry = &reader.index[0];
        let page = reader
            .get_page(entry.arch_id, entry.slot, entry.page_index)
            .unwrap()
            .expect("page should exist");

        assert!(page.header().row_count > 0);
        assert!(!page.data().is_empty());
    }

    #[test]
    fn get_nonexistent_page_returns_none() {
        let (_dir, path, _world) = flush_world_with_pos(1);
        let reader = SortedRunReader::open(&path).unwrap();

        // Use an arch_id that cannot exist.
        let result = reader.get_page(255, 255, 255).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn validate_page_crc_succeeds() {
        let (_dir, path, _world) = flush_world_with_pos(5);
        let reader = SortedRunReader::open(&path).unwrap();

        // Validate CRC of every page in the index.
        for entry in &reader.index {
            let page = reader
                .get_page(entry.arch_id, entry.slot, entry.page_index)
                .unwrap()
                .unwrap();
            reader.validate_page_crc(&page).unwrap();
        }
    }

    #[test]
    fn corrupted_magic_returns_error() {
        let (_dir, path, _world) = flush_world_with_pos(1);

        // Corrupt the first byte.
        let mut data = std::fs::read(&path).unwrap();
        data[0] = 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = SortedRunReader::open(&path);
        assert!(
            matches!(result, Err(LsmError::Format(_))),
            "expected Format error for corrupted magic"
        );
    }

    #[test]
    fn archetype_ids_returns_sorted_unique() {
        let mut world = World::new();
        // Create two archetypes
        world.spawn((Pos { x: 1.0, y: 2.0 },));
        world.spawn((Vel { dx: 1.0, dy: 0.0 },));

        let dir = tempfile::tempdir().unwrap();
        let codecs = crate::codec::CodecRegistry::new();
        let path = flush(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(0u64)).unwrap(),
            dir.path(),
            &codecs,
        )
        .unwrap()
        .unwrap();
        let reader = SortedRunReader::open(&path).unwrap();

        let ids = reader.archetype_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids[0] < ids[1]); // sorted
        // No duplicates
        let mut deduped = ids.clone();
        deduped.dedup();
        assert_eq!(ids, deduped);
    }

    #[test]
    fn open_rejects_header_with_lo_greater_than_hi() {
        let (_dir, path, _world) = flush_world_with_pos(3);

        // Corrupt sequence_lo / sequence_hi so that lo > hi, then fix up
        // both CRCs so the file passes magic, version, and CRC checks but
        // fails the SeqRange::new validation inside validate_and_parse.
        //
        // Header layout (all fields little-endian):
        //   bytes  0..8  : magic
        //   bytes  8..12 : version
        //   bytes 12..16 : schema_count
        //   bytes 16..24 : page_count
        //   bytes 24..32 : sequence_lo  ← overwrite to 9999
        //   bytes 32..40 : sequence_hi  ← overwrite to 1 (so lo > hi)
        //   bytes 40..44 : header_crc32  (covers bytes 0..40)
        //   bytes 44..64 : reserved
        let mut data = std::fs::read(&path).unwrap();

        // Write lo = 9999, hi = 1 — clearly lo > hi.
        data[24..32].copy_from_slice(&9999u64.to_le_bytes());
        data[32..40].copy_from_slice(&1u64.to_le_bytes());

        // Recompute header CRC (covers bytes 0..40).
        let new_header_crc = crc32fast::hash(&data[..40]);
        data[40..44].copy_from_slice(&new_header_crc.to_le_bytes());

        // Recompute total CRC (entire file with total_crc32 field zeroed).
        // Footer is the last 64 bytes; total_crc32 is at footer_offset + 32.
        let file_len = data.len();
        let total_crc32_offset = (file_len - 64) + 32;
        data[total_crc32_offset..total_crc32_offset + 4].copy_from_slice(&[0, 0, 0, 0]);
        let new_total_crc = crc32fast::hash(&data);
        data[total_crc32_offset..total_crc32_offset + 4]
            .copy_from_slice(&new_total_crc.to_le_bytes());

        std::fs::write(&path, &data).unwrap();

        let result = SortedRunReader::open(&path);
        assert!(
            matches!(result, Err(LsmError::Format(_))),
            "expected Format error for header with lo > hi, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn multi_component_index_lookup() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((
                Pos {
                    x: i as f32,
                    y: 0.0,
                },
                Vel {
                    dx: 1.0,
                    dy: i as f32,
                },
            ));
        }

        let dir = tempfile::tempdir().unwrap();
        let codecs = crate::codec::CodecRegistry::new();
        let path = flush(
            &world,
            SeqRange::new(SeqNo::from(0u64), SeqNo::from(100u64)).unwrap(),
            dir.path(),
            &codecs,
        )
        .unwrap()
        .unwrap();
        let reader = SortedRunReader::open(&path).unwrap();

        // Verify schema has 2 component entries.
        assert_eq!(reader.schema().len(), 2);

        // Verify every index entry is findable.
        for entry in &reader.index {
            let page = reader
                .get_page(entry.arch_id, entry.slot, entry.page_index)
                .expect("get_page should not error")
                .expect("every indexed page must be findable");
            reader
                .validate_page_crc(&page)
                .expect("every page CRC must be valid");
        }

        // Verify entity pages are present (slot == ENTITY_SLOT).
        let has_entity_page = reader.index.iter().any(|e| e.slot == ENTITY_SLOT);
        assert!(has_entity_page, "must have at least one entity page");
    }

    #[test]
    fn validate_page_crc_returns_proof_token() {
        use crate::codec::CrcProof;

        let (_dir, path, _world) = flush_world_with_pos(5);
        let reader = SortedRunReader::open(&path).unwrap();

        // Find the first page and validate its CRC, extracting the proof.
        assert!(reader.index_len() > 0);
        let entry = &reader.index[0];
        let page = reader
            .get_page(entry.arch_id, entry.slot, entry.page_index)
            .unwrap()
            .unwrap();
        let proof: CrcProof = reader.validate_page_crc(&page).unwrap();
        let _ = proof;
    }

    /// Build a minimal sorted-run file with entity pages at non-contiguous
    /// indices (0 and 3, but not 1 or 2). This simulates a flush that only
    /// writes dirty pages — entity-slot pages can be sparse when only some
    /// pages of an archetype's entity column were modified.
    ///
    /// Before the `entity_pages` method, `build_emit_list` used sequential
    /// `get_page(arch_id, ENTITY_SLOT, page_index++)` probing which breaks
    /// at the first gap and silently drops entities in higher pages.
    fn build_sparse_entity_page_file(dir: &Path) -> PathBuf {
        use crate::format::{Footer, Header, IndexEntry, MAGIC, PageHeader, VERSION};

        let path = dir.join("sparse.run");
        let arch_id: u16 = 0;

        let entity_data_page = |page_index: u16, entity_ids: &[u64]| -> Vec<u8> {
            let row_count = entity_ids.len() as u16;
            let mut page_bytes = Vec::new();
            let mut data_bytes = Vec::new();
            for &id in entity_ids {
                data_bytes.extend_from_slice(&id.to_le_bytes());
            }
            let data_crc = crc32fast::hash(&data_bytes);
            let ph = PageHeader {
                arch_id,
                slot: ENTITY_SLOT,
                page_index,
                row_count,
                page_crc32: data_crc,
                _padding: 0,
            };
            page_bytes.extend_from_slice(ph.as_bytes());
            page_bytes.extend_from_slice(&data_bytes);
            let item_size = 8usize;
            let pad_len = PAGE_SIZE * item_size - data_bytes.len();
            page_bytes.extend(std::iter::repeat_n(0u8, pad_len));
            page_bytes
        };

        let e0: u64 = 100;
        let e1: u64 = 200;
        let e2: u64 = 300;
        let e3: u64 = 400;

        let page0 = entity_data_page(0, &[e0, e1]);
        let page3 = entity_data_page(3, &[e2, e3]);

        let page0_offset: u64 = 64;
        let page3_offset: u64 = page0_offset + page0.len() as u64;

        let mut index = vec![
            IndexEntry {
                arch_id,
                slot: ENTITY_SLOT,
                page_index: 0,
                _pad: 0,
                file_offset: page0_offset,
            },
            IndexEntry {
                arch_id,
                slot: ENTITY_SLOT,
                page_index: 3,
                _pad: 0,
                file_offset: page3_offset,
            },
        ];
        index.sort();

        let index_offset = page3_offset + page3.len() as u64;

        let schema_count: u32 = 0;
        let schema_offset = index_offset + (index.len() as u64) * 16;
        let schema_bytes: Vec<u8> = Vec::new();

        let total_page_count: u64 = 2;

        let _footer_offset = schema_offset + schema_bytes.len() as u64;
        let footer = Footer {
            sparse_index_offset: index_offset,
            sparse_index_count: index.len() as u64,
            schema_offset,
            bloom_filter_offset: 0,
            total_crc32: 0,
            reserved: [0u8; 28],
        };

        let mut file = std::fs::File::create(&path).unwrap();

        let header = Header {
            magic: MAGIC,
            version: VERSION,
            schema_count,
            page_count: total_page_count,
            sequence_lo: 1,
            sequence_hi: 10,
            header_crc32: 0,
            reserved: [0u8; 20],
        };

        let mut header_bytes = header.as_bytes().to_vec();
        let header_crc = crc32fast::hash(&header_bytes[..40]);
        header_bytes[40..44].copy_from_slice(&header_crc.to_le_bytes());

        use std::io::Write;
        file.write_all(&header_bytes).unwrap();
        file.write_all(&page0).unwrap();
        file.write_all(&page3).unwrap();
        for entry in &index {
            file.write_all(entry.as_bytes()).unwrap();
        }
        file.write_all(&schema_bytes).unwrap();

        let mut footer_bytes = footer.as_bytes().to_vec();
        let total_crc = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&header_bytes);
            buf.extend_from_slice(&page0);
            buf.extend_from_slice(&page3);
            for entry in &index {
                buf.extend_from_slice(entry.as_bytes());
            }
            buf.extend_from_slice(&schema_bytes);
            buf.extend_from_slice(&footer_bytes);
            let total_crc32_offset = buf.len() - 64 + 32;
            buf[total_crc32_offset..total_crc32_offset + 4].copy_from_slice(&[0, 0, 0, 0]);
            crc32fast::hash(&buf)
        };
        footer_bytes[32..36].copy_from_slice(&total_crc.to_le_bytes());
        file.write_all(&footer_bytes).unwrap();

        path
    }

    #[test]
    fn entity_pages_iterates_sparse_entity_pages() {
        let dir = tempfile::tempdir().unwrap();
        let path = build_sparse_entity_page_file(dir.path());
        let reader = SortedRunReader::open(&path).unwrap();

        // Sequential get_page probing stops at the first gap (page 1 missing)
        // and misses page 3 entirely — this was the bug.
        assert!(
            reader.get_page(0, ENTITY_SLOT, 0).unwrap().is_some(),
            "page 0 must exist"
        );
        assert!(
            reader.get_page(0, ENTITY_SLOT, 1).unwrap().is_none(),
            "page 1 must be absent (sparse gap)"
        );
        assert!(
            reader.get_page(0, ENTITY_SLOT, 3).unwrap().is_some(),
            "page 3 must exist despite gap at page 1"
        );

        // entity_pages must iterate both pages 0 and 3, skipping the gap.
        let pages: Vec<(u16, u64)> = reader
            .entity_pages(0)
            .map(|r| {
                let (page_index, page) = r.unwrap();
                let row_count = page.header().row_count as usize;
                let data = page.data();
                let first_entity = u64::from_le_bytes(data[..8].try_into().unwrap());
                assert_eq!(row_count, 2, "each page has 2 entities");
                (page_index, first_entity)
            })
            .collect();

        assert_eq!(pages.len(), 2, "must find both sparse pages");
        assert_eq!(pages[0].0, 0, "first page index is 0");
        assert_eq!(pages[1].0, 3, "second page index is 3 (skipping gap)");
        assert_eq!(pages[0].1, 100, "first entity in page 0");
        assert_eq!(pages[1].1, 300, "first entity in page 3");
    }

    #[test]
    fn sequential_get_page_probing_misses_sparse_pages() {
        let dir = tempfile::tempdir().unwrap();
        let path = build_sparse_entity_page_file(dir.path());
        let reader = SortedRunReader::open(&path).unwrap();

        // Demonstrate the old sequential probing approach would stop
        // at the first gap and miss page 3 entirely.
        let mut found_via_sequential = Vec::new();
        let mut page_index: u16 = 0;
        loop {
            match reader.get_page(0, ENTITY_SLOT, page_index) {
                Ok(Some(page)) => {
                    let row_count = page.header().row_count as usize;
                    let data = page.data();
                    for r in 0..row_count {
                        let off = r * 8;
                        let id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                        found_via_sequential.push(id);
                    }
                    page_index += 1;
                }
                Ok(None) => break, // stops here — misses page 3!
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Sequential probing finds page 0's entities but not page 3's.
        assert_eq!(
            found_via_sequential,
            &[100, 200],
            "sequential probing stops at gap, missing page 3"
        );

        // entity_pages finds all entities including page 3.
        let mut found_via_entity_pages = Vec::new();
        for result in reader.entity_pages(0) {
            let (_page_index, page) = result.unwrap();
            let row_count = page.header().row_count as usize;
            let data = page.data();
            for r in 0..row_count {
                let off = r * 8;
                let id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                found_via_entity_pages.push(id);
            }
        }
        assert_eq!(
            found_via_entity_pages,
            &[100, 200, 300, 400],
            "entity_pages must find all entities across sparse pages"
        );
    }

    #[test]
    fn bloom_filter_matches_index() {
        let (_dir, path, _world) = flush_world_with_pos(10);
        let reader = SortedRunReader::open(&path).unwrap();

        for entry in &reader.index {
            assert!(
                reader.contains_page(entry.arch_id, entry.slot, entry.page_index),
                "bloom filter must return true for indexed page ({}, {}, {})",
                entry.arch_id,
                entry.slot,
                entry.page_index
            );
        }
    }

    #[test]
    fn bloom_filter_rejects_absent_pages() {
        let (_dir, path, _world) = flush_world_with_pos(5);
        let reader = SortedRunReader::open(&path).unwrap();

        let mut rejected = 0;
        for arch_id in 200u16..210 {
            for page_index in 0u16..10 {
                if !reader.contains_page(arch_id, 0, page_index) {
                    rejected += 1;
                }
            }
        }
        assert!(
            rejected > 0,
            "bloom filter must reject some absent page keys"
        );
    }

    #[test]
    fn old_run_without_bloom_falls_through() {
        let dir = tempfile::tempdir().unwrap();
        let path = build_sparse_entity_page_file(dir.path());
        let reader = SortedRunReader::open(&path).unwrap();

        assert!(
            reader.contains_page(0, ENTITY_SLOT, 0),
            "contains_page must return true when no bloom filter is present"
        );
        assert!(
            reader.contains_page(255, 255, 255),
            "contains_page must return true when no bloom filter is present"
        );
    }
}
