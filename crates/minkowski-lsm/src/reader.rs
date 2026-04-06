use std::fs;
use std::path::Path;

use crate::error::LsmError;
use crate::format::*;
use crate::schema::SchemaSection;

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
    sequence_range: (u64, u64),
    page_count: u64,
}

/// Minimum file size: Header (64) + Footer (64).
const MIN_FILE_SIZE: usize = 128;

/// Parsed metadata extracted from a file buffer during validation.
struct ParsedMetadata {
    schema: SchemaSection,
    index: Vec<IndexEntry>,
    sequence_range: (u64, u64),
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
    let sequence_range = (header.sequence_lo, header.sequence_hi);
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

    Ok(ParsedMetadata {
        schema,
        index,
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
        let item_size = self.item_size_for_slot(slot)?;
        let data_len = PAGE_SIZE.checked_mul(item_size).ok_or_else(|| {
            LsmError::Format(format!(
                "page ({arch_id}, {slot}, {page_index}): data length overflow"
            ))
        })?;
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

    /// Validate the CRC of a specific page.
    ///
    /// The CRC covers `row_count * item_size` bytes (the actual data, not
    /// zero-padding).
    pub fn validate_page_crc(&self, page: &PageRef<'_>) -> Result<(), LsmError> {
        let item_size = self.item_size_for_slot(page.header().slot)?;
        let actual_len = page.header().row_count as usize * item_size;
        let computed = crc32fast::hash(&page.data()[..actual_len]);

        if computed != page.header().page_crc32 {
            return Err(LsmError::Crc {
                offset: page.file_offset(),
                expected: page.header().page_crc32,
                actual: computed,
            });
        }
        Ok(())
    }

    /// Get the schema section.
    pub fn schema(&self) -> &SchemaSection {
        &self.schema
    }

    /// WAL sequence range covered by this sorted run.
    pub fn sequence_range(&self) -> (u64, u64) {
        self.sequence_range
    }

    /// Total number of pages in this sorted run.
    pub fn page_count(&self) -> u64 {
        self.page_count
    }

    /// Number of entries in the sparse index.
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// Item size in bytes for a given slot.
    fn item_size_for_slot(&self, slot: u16) -> Result<usize, LsmError> {
        if slot == ENTITY_SLOT {
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
        let path = flush(&world, (10, 20), dir.path()).unwrap().unwrap();
        (dir, path, world)
    }

    #[test]
    fn open_valid_file() {
        let (_dir, path, _world) = flush_world_with_pos(5);
        let reader = SortedRunReader::open(&path).unwrap();

        assert_eq!(reader.sequence_range(), (10, 20));
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
        let path = flush(&world, (0, 100), dir.path()).unwrap().unwrap();
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
}
