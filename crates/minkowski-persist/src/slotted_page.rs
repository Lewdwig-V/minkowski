//! Slotted-page storage layer for the WAL.
//!
//! A **slotted page** is a fixed-size block that packs multiple variable-length
//! records into page-aligned units. Each page has a header (magic, page
//! sequence, slot count, free/data offsets, CRC32) and a slot directory
//! pointing to records packed from the end of the page backward.
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ PageHeader (20 bytes)                       │
//! │   magic: u32    = 0x4D4B5750 ("MKWP")       │
//! │   page_seq: u32 = monotonic page number      │
//! │   slot_count: u16                            │
//! │   free_offset: u16 = end of slot directory    │
//! │   data_offset: u16 = start of record data     │
//! │   _padding: u16 = 0                           │
//! │   checksum: u32 = CRC32 of full page          │
//! ├─────────────────────────────────────────────┤
//! │ Slot Directory (slot_count × 4 bytes)       │
//! │   slot[0]: offset: u16, length: u16          │
//! │   slot[1]: ...                               │
//! ├─────────────────────────────────────────────┤
//! │ Free Space                                   │
//! ├─────────────────────────────────────────────┤
//! │ Record Data (packed from end, growing down)  │
//! │   record[N-1]: [payload bytes]               │
//! │   ...                                        │
//! │   record[0]: [payload bytes]                 │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! The page header ends with a CRC32 that covers the entire page content
//! (everything from byte 0 through `page_size`, with the CRC32 field itself
//! zeroed during computation). This is a second level of integrity on top of
//! the per-frame CRC32 in each WAL record.
//!
//! Records larger than a single page are stored as **overflow pages** with
//! magic `0x4D4B574F` ("MKWO"). The first overflow page has a single slot
//! entry whose length covers the full record; continuation pages are pure
//! data with `slot_count = 0`.

use std::io::{self, Read, Write};

/// Regular slotted page magic: "MKWP" in little-endian.
pub const PAGE_MAGIC: u32 = 0x4D4B5750;

/// Overflow page magic: "MKWO" in little-endian.
pub const OVERFLOW_MAGIC: u32 = 0x4D4B574F;

/// Page header size in bytes.
pub const PAGE_HEADER_SIZE: usize = 20;

/// Slot directory entry size in bytes (offset: u16 + length: u16).
pub const SLOT_ENTRY_SIZE: usize = 4;

/// Default page size (4 KiB).
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// Minimum useful record size — a record must be at least 1 byte.
const MIN_RECORD_SIZE: usize = 1;

/// Page header stored at the start of every slotted page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageHeader {
    /// Magic number identifying page type.
    pub magic: u32,
    /// Monotonic page sequence number within the segment.
    pub page_seq: u32,
    /// Number of slots in the directory.
    pub slot_count: u16,
    /// Byte offset of the end of the slot directory (start of free space).
    pub free_offset: u16,
    /// Byte offset of the start of record data (end of free space).
    pub data_offset: u16,
    /// Padding for alignment; must be zero.
    pub _padding: u16,
    /// CRC32 of the full page content (with this field zeroed during computation).
    pub checksum: u32,
}

/// Byte range of the checksum field within the page header.
const CHECKSUM_OFFSET: usize = 16;
const CHECKSUM_SIZE: usize = 4;

impl PageHeader {
    fn to_bytes(self) -> [u8; PAGE_HEADER_SIZE] {
        let mut buf = [0u8; PAGE_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.page_seq.to_le_bytes());
        buf[8..10].copy_from_slice(&self.slot_count.to_le_bytes());
        buf[10..12].copy_from_slice(&self.free_offset.to_le_bytes());
        buf[12..14].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[14..16].copy_from_slice(&self._padding.to_le_bytes());
        buf[16..20].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8; PAGE_HEADER_SIZE]) -> Self {
        Self {
            magic: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            page_seq: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            slot_count: u16::from_le_bytes([buf[8], buf[9]]),
            free_offset: u16::from_le_bytes([buf[10], buf[11]]),
            data_offset: u16::from_le_bytes([buf[12], buf[13]]),
            _padding: u16::from_le_bytes([buf[14], buf[15]]),
            checksum: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
        }
    }
}

/// A single entry in the slot directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotEntry {
    /// Byte offset of the record within the page.
    pub offset: u16,
    /// Length of the record in bytes.
    pub length: u16,
}

impl SlotEntry {
    fn to_bytes(self) -> [u8; SLOT_ENTRY_SIZE] {
        let mut buf = [0u8; SLOT_ENTRY_SIZE];
        buf[0..2].copy_from_slice(&self.offset.to_le_bytes());
        buf[2..4].copy_from_slice(&self.length.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8; SLOT_ENTRY_SIZE]) -> Self {
        Self {
            offset: u16::from_le_bytes([buf[0], buf[1]]),
            length: u16::from_le_bytes([buf[2], buf[3]]),
        }
    }
}

/// In-memory representation of a single slotted page.
///
/// Records are inserted from the end of the page backward, while the slot
/// directory grows forward from after the header. Free space sits between
/// the two.
pub struct SlottedPage {
    /// Raw page buffer, always `page_size` bytes.
    data: Vec<u8>,
    /// Cached header (kept in sync with `data[0..PAGE_HEADER_SIZE]`).
    header: PageHeader,
    /// Page size in bytes.
    page_size: usize,
}

impl SlottedPage {
    /// Create a new empty slotted page.
    pub fn new(page_seq: u32, page_size: usize) -> Self {
        assert!(
            page_size >= PAGE_HEADER_SIZE + SLOT_ENTRY_SIZE + MIN_RECORD_SIZE,
            "page_size {page_size} too small for slotted page"
        );
        assert!(
            page_size <= u16::MAX as usize,
            "page_size {page_size} exceeds u16 addressable range"
        );

        let header = PageHeader {
            magic: PAGE_MAGIC,
            page_seq,
            slot_count: 0,
            free_offset: PAGE_HEADER_SIZE as u16,
            data_offset: page_size as u16,
            _padding: 0,
            checksum: 0,
        };

        let mut data = vec![0u8; page_size];
        data[0..PAGE_HEADER_SIZE].copy_from_slice(&header.to_bytes());

        Self {
            data,
            header,
            page_size,
        }
    }

    /// Try to insert a record into the page. Returns the slot index if
    /// there is enough space, or `None` if the record doesn't fit.
    pub fn try_insert(&mut self, payload: &[u8]) -> Option<u16> {
        let needed = SLOT_ENTRY_SIZE + payload.len();
        if self.free_space() < needed {
            return None;
        }
        if payload.len() > u16::MAX as usize {
            return None;
        }

        // Record grows downward from data_offset.
        let new_data_offset = self.header.data_offset as usize - payload.len();
        self.data[new_data_offset..new_data_offset + payload.len()].copy_from_slice(payload);

        // Slot entry grows upward from free_offset.
        let slot_idx = self.header.slot_count;
        let entry = SlotEntry {
            offset: new_data_offset as u16,
            length: payload.len() as u16,
        };
        let slot_pos = self.header.free_offset as usize;
        self.data[slot_pos..slot_pos + SLOT_ENTRY_SIZE].copy_from_slice(&entry.to_bytes());

        // Update header.
        self.header.slot_count += 1;
        self.header.free_offset += SLOT_ENTRY_SIZE as u16;
        self.header.data_offset = new_data_offset as u16;
        self.flush_header();

        Some(slot_idx)
    }

    /// Read a record by slot index. Returns `None` if the index is out of
    /// range or the slot entry references data outside the page (corruption).
    pub fn get(&self, slot_index: u16) -> Option<&[u8]> {
        if slot_index >= self.header.slot_count {
            return None;
        }
        let entry = self.slot_entry(slot_index)?;
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        if end > self.page_size {
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Number of records stored in this page.
    pub fn slot_count(&self) -> u16 {
        self.header.slot_count
    }

    /// Available free space in bytes (for both a new slot entry and record data).
    pub fn free_space(&self) -> usize {
        let free_start = self.header.free_offset as usize;
        let data_start = self.header.data_offset as usize;
        data_start.saturating_sub(free_start)
    }

    /// Returns true if there is enough space for a record of `payload_len` bytes.
    pub fn can_fit(&self, payload_len: usize) -> bool {
        self.free_space() >= SLOT_ENTRY_SIZE + payload_len
    }

    /// Page sequence number.
    pub fn page_seq(&self) -> u32 {
        self.header.page_seq
    }

    /// Compute the CRC32 of the full page content (with the checksum field
    /// zeroed during computation).
    pub fn compute_checksum(&self) -> u32 {
        let mut buf = self.data.clone();
        // Zero out the checksum field (bytes 16..20) for computation.
        buf[CHECKSUM_OFFSET..CHECKSUM_OFFSET + CHECKSUM_SIZE].fill(0);
        crc32fast::hash(&buf)
    }

    /// Compute and store the full CRC32 checksum into the page header.
    pub fn seal(&mut self) {
        self.header.checksum = 0;
        self.flush_header();
        let crc = self.compute_checksum();
        self.header.checksum = crc;
        self.data[CHECKSUM_OFFSET..CHECKSUM_OFFSET + CHECKSUM_SIZE]
            .copy_from_slice(&crc.to_le_bytes());
    }

    /// Validate the page-level CRC32 checksum. Returns true if valid.
    pub fn validate_checksum(&self) -> bool {
        let stored = self.header.checksum;
        let mut buf = self.data.clone();
        buf[CHECKSUM_OFFSET..CHECKSUM_OFFSET + CHECKSUM_SIZE].fill(0);
        let computed = crc32fast::hash(&buf);
        stored == computed
    }

    /// Write the full page to a writer.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.data)
    }

    /// Read a page from a reader. Returns an error if the page header
    /// contains values that are inconsistent with `page_size` (corrupt
    /// or crafted data). Individual slot entries are validated lazily
    /// by `get()` — this method only checks structural invariants.
    pub fn read_from<R: Read>(reader: &mut R, page_size: usize) -> io::Result<Self> {
        if page_size < PAGE_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("page too small ({page_size}) to contain header"),
            ));
        }

        let mut data = vec![0u8; page_size];
        reader.read_exact(&mut data)?;

        let mut header_buf = [0u8; PAGE_HEADER_SIZE];
        header_buf.copy_from_slice(&data[0..PAGE_HEADER_SIZE]);
        let header = PageHeader::from_bytes(&header_buf);

        // Validate page magic before trusting any other header fields.
        if header.magic != PAGE_MAGIC && header.magic != OVERFLOW_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid page magic: expected {PAGE_MAGIC:#010x} (MKWP) or \
                     {OVERFLOW_MAGIC:#010x} (MKWO), got {:#010x}",
                    header.magic
                ),
            ));
        }

        // Validate structural invariants against page_size.
        let slot_dir_end = PAGE_HEADER_SIZE + (header.slot_count as usize) * SLOT_ENTRY_SIZE;
        if slot_dir_end > page_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "corrupt page header: slot directory end ({slot_dir_end}) \
                     exceeds page size ({page_size})"
                ),
            ));
        }
        if slot_dir_end > header.free_offset as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "corrupt page header: slot directory end ({slot_dir_end}) \
                     exceeds free_offset ({})",
                    header.free_offset
                ),
            ));
        }
        if (header.free_offset as usize) < PAGE_HEADER_SIZE
            || (header.free_offset as usize) > page_size
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "corrupt page header: free_offset ({}) out of range",
                    header.free_offset
                ),
            ));
        }
        if (header.data_offset as usize) > page_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "corrupt page header: data_offset ({}) exceeds page size ({page_size})",
                    header.data_offset
                ),
            ));
        }
        if header.free_offset > header.data_offset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "corrupt page header: free_offset ({}) > data_offset ({})",
                    header.free_offset, header.data_offset
                ),
            ));
        }

        Ok(Self {
            data,
            header,
            page_size,
        })
    }

    /// Iterate over all records in slot order.
    pub fn iter_slots(&self) -> SlotIter<'_> {
        SlotIter {
            page: self,
            index: 0,
        }
    }

    /// Access the raw page header.
    pub fn header(&self) -> &PageHeader {
        &self.header
    }

    /// Read a slot entry at the given index. Returns `None` if the slot
    /// directory entry falls outside the page buffer (corrupt header).
    fn slot_entry(&self, index: u16) -> Option<SlotEntry> {
        let pos = PAGE_HEADER_SIZE + (index as usize) * SLOT_ENTRY_SIZE;
        let end = pos + SLOT_ENTRY_SIZE;
        if end > self.page_size {
            return None;
        }
        let mut buf = [0u8; SLOT_ENTRY_SIZE];
        buf.copy_from_slice(&self.data[pos..end]);
        Some(SlotEntry::from_bytes(&buf))
    }

    /// Flush the cached header back into the data buffer.
    fn flush_header(&mut self) {
        self.data[0..PAGE_HEADER_SIZE].copy_from_slice(&self.header.to_bytes());
    }
}

impl Clone for SlottedPage {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            header: self.header,
            page_size: self.page_size,
        }
    }
}

/// Iterator over records in a slotted page.
pub struct SlotIter<'a> {
    page: &'a SlottedPage,
    index: u16,
}

impl<'a> Iterator for SlotIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.page.header.slot_count {
            return None;
        }
        let data = self.page.get(self.index)?;
        self.index += 1;
        Some(data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.page.header.slot_count.saturating_sub(self.index) as usize;
        // Lower bound is 0 because corrupt slots may terminate iteration early.
        (0, Some(remaining))
    }
}

/// Availability list tracking pages with free space.
///
/// This is an in-memory structure rebuilt on open by scanning page headers.
/// For append-only WAL usage, only the current active page typically has
/// free space. For future compaction scenarios, sealed pages with
/// invalidated records could be added back to the availability list.
pub struct AvailabilityList {
    /// Page indices with free space, sorted by available bytes descending
    /// for best-fit allocation.
    entries: Vec<AvailabilityEntry>,
}

/// A single entry in the availability list.
#[derive(Debug, Clone, Copy)]
pub struct AvailabilityEntry {
    /// Page sequence number.
    pub page_seq: u32,
    /// Available free space in bytes.
    pub free_bytes: usize,
}

impl AvailabilityList {
    /// Create an empty availability list.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add or update a page entry. Maintains descending sort by free bytes.
    pub fn update(&mut self, page_seq: u32, free_bytes: usize) {
        // Remove existing entry for this page_seq, if any.
        self.entries.retain(|e| e.page_seq != page_seq);

        if free_bytes >= SLOT_ENTRY_SIZE + MIN_RECORD_SIZE {
            let entry = AvailabilityEntry {
                page_seq,
                free_bytes,
            };
            // Insert maintaining descending sort.
            let pos = self.entries.partition_point(|e| e.free_bytes >= free_bytes);
            self.entries.insert(pos, entry);
        }
    }

    /// Remove a page from the availability list.
    pub fn remove(&mut self, page_seq: u32) {
        self.entries.retain(|e| e.page_seq != page_seq);
    }

    /// Find the best-fit page that can hold `needed_bytes` (payload + slot entry).
    /// Returns the page_seq of the best candidate, or `None`.
    pub fn find_fit(&self, needed_bytes: usize) -> Option<u32> {
        // Best-fit: find the page with the smallest free space that still fits.
        self.entries
            .iter()
            .rev()
            .find(|e| e.free_bytes >= needed_bytes)
            .map(|e| e.page_seq)
    }

    /// Number of pages in the availability list.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the availability list is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over entries.
    pub fn iter(&self) -> impl Iterator<Item = &AvailabilityEntry> {
        self.entries.iter()
    }
}

impl Default for AvailabilityList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_page_has_correct_header() {
        let page = SlottedPage::new(0, DEFAULT_PAGE_SIZE);
        assert_eq!(page.header().magic, PAGE_MAGIC);
        assert_eq!(page.header().page_seq, 0);
        assert_eq!(page.slot_count(), 0);
        assert_eq!(page.header().free_offset, PAGE_HEADER_SIZE as u16);
        assert_eq!(page.header().data_offset, DEFAULT_PAGE_SIZE as u16);
    }

    #[test]
    fn insert_and_read_single_record() {
        let mut page = SlottedPage::new(1, DEFAULT_PAGE_SIZE);
        let payload = b"hello world";
        let slot = page.try_insert(payload).expect("should fit");
        assert_eq!(slot, 0);
        assert_eq!(page.slot_count(), 1);
        assert_eq!(page.get(0), Some(payload.as_slice()));
    }

    #[test]
    fn insert_multiple_records() {
        let mut page = SlottedPage::new(0, DEFAULT_PAGE_SIZE);
        let payloads: Vec<Vec<u8>> = (0..10).map(|i| vec![i; 50]).collect();

        for (i, p) in payloads.iter().enumerate() {
            let slot = page.try_insert(p).expect("should fit");
            assert_eq!(slot, i as u16);
        }

        assert_eq!(page.slot_count(), 10);

        for (i, p) in payloads.iter().enumerate() {
            assert_eq!(page.get(i as u16), Some(p.as_slice()));
        }
    }

    #[test]
    fn insert_fills_page() {
        // Use a small page to test fill behavior.
        let page_size = 64;
        let mut page = SlottedPage::new(0, page_size);
        // Available: 64 - 20 (header) = 44 bytes for slots + data.
        // Each record needs SLOT_ENTRY_SIZE (4) + payload_len bytes.

        // 44 / (4 + 7) = 4 records of 7 bytes
        for _ in 0..4 {
            assert!(page.try_insert(&[0xAA; 7]).is_some());
        }
        // Page should be full now.
        assert!(page.try_insert(&[0xBB; 1]).is_none());
    }

    #[test]
    fn get_out_of_bounds_returns_none() {
        let page = SlottedPage::new(0, DEFAULT_PAGE_SIZE);
        assert_eq!(page.get(0), None);
        assert_eq!(page.get(100), None);
    }

    #[test]
    fn free_space_decreases_on_insert() {
        let mut page = SlottedPage::new(0, DEFAULT_PAGE_SIZE);
        let initial_free = page.free_space();

        page.try_insert(&[0u8; 100]).unwrap();
        let after_insert = page.free_space();

        assert_eq!(initial_free - after_insert, SLOT_ENTRY_SIZE + 100);
    }

    #[test]
    fn can_fit_checks_correctly() {
        let mut page = SlottedPage::new(0, 64);
        // 44 bytes total free (64 - 20 header).
        assert!(page.can_fit(40)); // 4 (slot) + 40 = 44
        assert!(!page.can_fit(41)); // 4 + 41 = 45 > 44

        page.try_insert(&[0u8; 20]).unwrap(); // Uses 4 + 20 = 24
        assert!(page.can_fit(16)); // 4 + 16 = 20 remaining
        assert!(!page.can_fit(17));
    }

    #[test]
    fn write_and_read_roundtrip() {
        let mut page = SlottedPage::new(42, DEFAULT_PAGE_SIZE);
        page.try_insert(b"first record").unwrap();
        page.try_insert(b"second record").unwrap();
        page.try_insert(b"third record").unwrap();

        let mut buf = Vec::new();
        page.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), DEFAULT_PAGE_SIZE);

        let restored = SlottedPage::read_from(&mut buf.as_slice(), DEFAULT_PAGE_SIZE).unwrap();
        assert_eq!(restored.header().magic, PAGE_MAGIC);
        assert_eq!(restored.header().page_seq, 42);
        assert_eq!(restored.slot_count(), 3);
        assert_eq!(restored.get(0), Some(b"first record".as_slice()));
        assert_eq!(restored.get(1), Some(b"second record".as_slice()));
        assert_eq!(restored.get(2), Some(b"third record".as_slice()));
    }

    #[test]
    fn iter_slots_yields_all_records() {
        let mut page = SlottedPage::new(0, DEFAULT_PAGE_SIZE);
        page.try_insert(b"aaa").unwrap();
        page.try_insert(b"bbb").unwrap();
        page.try_insert(b"ccc").unwrap();

        let records: Vec<&[u8]> = page.iter_slots().collect();
        assert_eq!(records, vec![b"aaa".as_slice(), b"bbb", b"ccc"]);
        assert_eq!(page.iter_slots().count(), 3);
    }

    #[test]
    fn seal_and_validate_checksum() {
        let mut page = SlottedPage::new(7, DEFAULT_PAGE_SIZE);
        page.try_insert(b"some data").unwrap();
        page.seal();
        assert!(page.validate_checksum());

        // Corrupt a data byte.
        let mut corrupted = page.clone();
        corrupted.data[DEFAULT_PAGE_SIZE - 1] ^= 0xFF;
        assert!(!corrupted.validate_checksum());
    }

    #[test]
    fn header_round_trip() {
        let header = PageHeader {
            magic: PAGE_MAGIC,
            page_seq: 12345,
            slot_count: 42,
            free_offset: 200,
            data_offset: 3000,
            _padding: 0,
            checksum: 0xDEADBEEF,
        };
        let bytes = header.to_bytes();
        let restored = PageHeader::from_bytes(&bytes);
        assert_eq!(header, restored);
    }

    #[test]
    fn slot_entry_round_trip() {
        let entry = SlotEntry {
            offset: 1234,
            length: 567,
        };
        let bytes = entry.to_bytes();
        let restored = SlotEntry::from_bytes(&bytes);
        assert_eq!(entry, restored);
    }

    #[test]
    fn availability_list_update_and_find() {
        let mut list = AvailabilityList::new();
        assert!(list.is_empty());

        list.update(0, 1000);
        list.update(1, 500);
        list.update(2, 2000);

        assert_eq!(list.len(), 3);

        // Best-fit: smallest page that fits 600 bytes.
        assert_eq!(list.find_fit(600), Some(0)); // 1000 bytes fits, 500 doesn't
                                                 // Best-fit: smallest page that fits 100 bytes.
        assert_eq!(list.find_fit(100), Some(1)); // 500 is smallest that fits
                                                 // Best-fit: needs 1500 bytes.
        assert_eq!(list.find_fit(1500), Some(2)); // Only 2000 fits
                                                  // Nothing fits 3000.
        assert_eq!(list.find_fit(3000), None);
    }

    #[test]
    fn availability_list_remove() {
        let mut list = AvailabilityList::new();
        list.update(0, 1000);
        list.update(1, 500);
        assert_eq!(list.len(), 2);

        list.remove(0);
        assert_eq!(list.len(), 1);
        assert_eq!(list.find_fit(1000), None);
        assert_eq!(list.find_fit(500), Some(1));
    }

    #[test]
    fn availability_list_update_replaces_existing() {
        let mut list = AvailabilityList::new();
        list.update(0, 1000);
        assert_eq!(list.len(), 1);

        list.update(0, 500);
        assert_eq!(list.len(), 1);
        // Should reflect the updated free bytes.
        assert_eq!(list.find_fit(600), None);
        assert_eq!(list.find_fit(500), Some(0));
    }

    #[test]
    fn availability_list_ignores_tiny_pages() {
        let mut list = AvailabilityList::new();
        // Too small to hold even a slot entry + 1 byte record.
        list.update(0, SLOT_ENTRY_SIZE);
        assert!(list.is_empty());

        list.update(1, SLOT_ENTRY_SIZE + MIN_RECORD_SIZE);
        assert_eq!(list.len(), 1);
    }

    #[test]
    #[should_panic(expected = "too small")]
    fn page_size_too_small_panics() {
        SlottedPage::new(0, PAGE_HEADER_SIZE); // No room for even one slot.
    }

    #[test]
    fn empty_payload_insert() {
        let mut page = SlottedPage::new(0, DEFAULT_PAGE_SIZE);
        // Empty payloads are technically valid (a "tombstone" slot).
        let slot = page.try_insert(&[]).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(page.get(0), Some([].as_slice()));
    }

    // ── Corruption resilience tests ─────────────────────────────────

    #[test]
    fn corrupt_slot_offset_beyond_page_returns_none() {
        let mut page = SlottedPage::new(0, 64);
        page.try_insert(b"hello").unwrap();

        // Corrupt the slot entry: set offset to beyond page_size.
        let slot_pos = PAGE_HEADER_SIZE;
        page.data[slot_pos] = 0xFF;
        page.data[slot_pos + 1] = 0xFF; // offset = 65535, way beyond 64

        assert_eq!(page.get(0), None, "corrupt offset must not panic");
    }

    #[test]
    fn corrupt_slot_length_beyond_page_returns_none() {
        let mut page = SlottedPage::new(0, 64);
        page.try_insert(b"hello").unwrap();

        // Corrupt the slot entry: set length to a huge value.
        let slot_pos = PAGE_HEADER_SIZE;
        page.data[slot_pos + 2] = 0xFF;
        page.data[slot_pos + 3] = 0xFF; // length = 65535

        assert_eq!(page.get(0), None, "corrupt length must not panic");
    }

    #[test]
    fn corrupt_slot_count_beyond_page_returns_none_on_get() {
        let mut page = SlottedPage::new(0, 64);
        page.try_insert(b"hi").unwrap();

        // Corrupt slot_count to a huge value — slot_entry will try to
        // read directory entries beyond the page buffer.
        page.header.slot_count = 10000;
        page.flush_header();

        // Slots 0 might still be in bounds, but slot 9999 definitely isn't.
        // get must return None, not panic.
        assert_eq!(page.get(9999), None);
    }

    #[test]
    fn corrupt_iter_stops_on_bad_slot() {
        let mut page = SlottedPage::new(0, 64);
        page.try_insert(b"aaa").unwrap();
        page.try_insert(b"bbb").unwrap();

        // Corrupt second slot's offset to go out of bounds.
        let slot1_pos = PAGE_HEADER_SIZE + SLOT_ENTRY_SIZE;
        page.data[slot1_pos] = 0xFF;
        page.data[slot1_pos + 1] = 0xFF;

        let records: Vec<&[u8]> = page.iter_slots().collect();
        // First slot is valid, second is corrupt — iterator yields 1 then stops.
        assert_eq!(records.len(), 1);
        assert_eq!(records[0], b"aaa");
    }

    #[test]
    fn read_from_corrupt_slot_count_errors() {
        // Create a page buffer with slot_count that overflows the page.
        let page_size = 64;
        let mut data = vec![0u8; page_size];
        let header = PageHeader {
            magic: PAGE_MAGIC,
            page_seq: 0,
            slot_count: 1000, // would need 4000+ bytes for slot directory
            free_offset: PAGE_HEADER_SIZE as u16,
            data_offset: page_size as u16,
            _padding: 0,
            checksum: 0,
        };
        data[0..PAGE_HEADER_SIZE].copy_from_slice(&header.to_bytes());

        let result = SlottedPage::read_from(&mut data.as_slice(), page_size);
        assert!(
            result.is_err(),
            "corrupt slot_count should produce an error"
        );
        let msg = result.err().expect("should be Err").to_string();
        assert!(msg.contains("slot directory end"), "{msg}");
    }

    #[test]
    fn read_from_corrupt_free_offset_errors() {
        let page_size = 64;
        let mut data = vec![0u8; page_size];
        let header = PageHeader {
            magic: PAGE_MAGIC,
            page_seq: 0,
            slot_count: 0,
            free_offset: 5, // below PAGE_HEADER_SIZE
            data_offset: page_size as u16,
            _padding: 0,
            checksum: 0,
        };
        data[0..PAGE_HEADER_SIZE].copy_from_slice(&header.to_bytes());

        let result = SlottedPage::read_from(&mut data.as_slice(), page_size);
        assert!(result.is_err());
        let msg = result.err().expect("should be Err").to_string();
        assert!(msg.contains("free_offset"), "{msg}");
    }

    #[test]
    fn read_from_free_offset_exceeds_data_offset_errors() {
        let page_size = 64;
        let mut data = vec![0u8; page_size];
        let header = PageHeader {
            magic: PAGE_MAGIC,
            page_seq: 0,
            slot_count: 0,
            free_offset: 50,
            data_offset: 40, // free > data = overlapping
            _padding: 0,
            checksum: 0,
        };
        data[0..PAGE_HEADER_SIZE].copy_from_slice(&header.to_bytes());

        let result = SlottedPage::read_from(&mut data.as_slice(), page_size);
        assert!(result.is_err());
        let msg = result.err().expect("should be Err").to_string();
        assert!(msg.contains("free_offset"), "{msg}");
    }

    #[test]
    fn read_from_data_offset_exceeds_page_size_errors() {
        let page_size = 64;
        let mut data = vec![0u8; page_size];
        let header = PageHeader {
            magic: PAGE_MAGIC,
            page_seq: 0,
            slot_count: 0,
            free_offset: PAGE_HEADER_SIZE as u16,
            data_offset: 100, // beyond 64-byte page
            _padding: 0,
            checksum: 0,
        };
        data[0..PAGE_HEADER_SIZE].copy_from_slice(&header.to_bytes());

        let result = SlottedPage::read_from(&mut data.as_slice(), page_size);
        assert!(result.is_err());
        let msg = result.err().expect("should be Err").to_string();
        assert!(msg.contains("data_offset"), "{msg}");
    }

    #[test]
    fn read_from_undersized_page_returns_error() {
        // page_size < PAGE_HEADER_SIZE must return an error, not panic.
        let tiny = vec![0u8; 4];
        let result = SlottedPage::read_from(&mut tiny.as_slice(), 4);
        assert!(result.is_err());
        let msg = result.err().expect("should be Err").to_string();
        assert!(msg.contains("too small"), "{msg}");
    }

    #[test]
    fn read_from_bad_magic_returns_error() {
        let page_size = 64;
        let data = vec![0u8; page_size]; // all zeros — magic is 0x00000000
        let result = SlottedPage::read_from(&mut data.as_slice(), page_size);
        assert!(result.is_err());
        let msg = result.err().expect("should be Err").to_string();
        assert!(msg.contains("invalid page magic"), "{msg}");
    }

    #[test]
    fn read_from_accepts_overflow_magic() {
        let page_size = 64;
        let mut page = SlottedPage::new(0, page_size);
        // Overwrite magic with OVERFLOW_MAGIC.
        page.data[0..4].copy_from_slice(&OVERFLOW_MAGIC.to_le_bytes());
        // Re-parse header so it's consistent.
        page.header.magic = OVERFLOW_MAGIC;
        // Update free_offset/data_offset in raw bytes to match header.
        page.seal();
        let result = SlottedPage::read_from(&mut page.data.as_slice(), page_size);
        assert!(result.is_ok(), "overflow magic should be accepted");
    }

    #[test]
    fn read_from_slot_dir_exceeds_free_offset_errors() {
        let page_size = 64;
        let mut data = vec![0u8; page_size];
        // Claim 5 slots: slot_dir_end = 20 + 5*4 = 40.
        // Set free_offset = 24 (< 40), so the slot directory overruns free space.
        let header = PageHeader {
            magic: PAGE_MAGIC,
            page_seq: 0,
            slot_count: 5,
            free_offset: 24,
            data_offset: page_size as u16,
            _padding: 0,
            checksum: 0,
        };
        data[0..PAGE_HEADER_SIZE].copy_from_slice(&header.to_bytes());

        let result = SlottedPage::read_from(&mut data.as_slice(), page_size);
        assert!(result.is_err());
        let msg = result.err().expect("should be Err").to_string();
        assert!(
            msg.contains("slot directory end") && msg.contains("free_offset"),
            "{msg}"
        );
    }
}
