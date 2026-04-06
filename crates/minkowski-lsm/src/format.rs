/// On-disk magic bytes identifying a Minkowski LSM sorted-run file.
pub const MAGIC: [u8; 8] = *b"MKLSM01\0";

/// File format version.
pub const VERSION: u32 = 1;

/// Special `slot` value used in [`PageHeader`] and [`IndexEntry`] to identify
/// entity-ID pages (as opposed to component-data pages).
pub const ENTITY_SLOT: u16 = 0xFFFF;

/// Number of rows per page — re-exported from the minkowski crate for
/// convenience so callers do not need to depend on both crates.
pub const PAGE_SIZE: usize = minkowski::PAGE_SIZE;

// ── compile-time size assertions ────────────────────────────────────────────

const _: () = assert!(std::mem::size_of::<Header>() == 64);
const _: () = assert!(std::mem::size_of::<PageHeader>() == 16);
const _: () = assert!(std::mem::size_of::<IndexEntry>() == 16);
const _: () = assert!(std::mem::size_of::<Footer>() == 64);

// ── File Header (64 bytes) ───────────────────────────────────────────────────

/// Fixed-size file header written at byte offset 0.
///
/// The header is protected by `header_crc32`, computed over the first 60 bytes
/// (i.e., everything before the CRC field itself and the reserved tail).
#[repr(C)]
pub struct Header {
    /// Must equal [`MAGIC`].
    pub magic: [u8; 8],
    /// Must equal [`VERSION`].
    pub version: u32,
    /// Number of schema entries that follow the page data.
    pub schema_count: u32,
    /// Total number of data pages in this run.
    pub page_count: u64,
    /// Lower 64 bits of the WAL sequence range covered by this run.
    pub sequence_lo: u64,
    /// Upper 64 bits of the WAL sequence range covered by this run.
    pub sequence_hi: u64,
    /// CRC32 of the preceding 60 bytes of this header.
    pub header_crc32: u32,
    /// Reserved for future use; must be zero on write.
    pub reserved: [u8; 20],
}

impl Header {
    /// View this header as a 64-byte slice for I/O.
    ///
    /// # Safety
    /// `Header` is `#[repr(C)]` with a compile-time-verified size of 64 bytes.
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Header is #[repr(C)] and the compile-time assert above
        // guarantees its size is exactly 64 bytes.
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// Interpret a 64-byte slice as a `Header` reference.
    ///
    /// # Safety
    /// The caller must ensure `bytes` is properly aligned for `Header`.
    /// For unaligned reads (e.g., from a file buffer), copy into an aligned
    /// value first or use the `as_bytes` + `clone`-by-value pattern.
    pub fn from_bytes(bytes: &[u8; 64]) -> &Self {
        // SAFETY: Header is #[repr(C)]; any 64-byte bit pattern is a valid
        // representation (individual field semantics are validated separately).
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

// ── Page Header (16 bytes) ───────────────────────────────────────────────────

/// Header prefixing each data page.
///
/// A data page contains `PAGE_SIZE` rows of a single component column for
/// one archetype.  When `slot == ENTITY_SLOT` the page stores raw entity IDs.
#[repr(C)]
pub struct PageHeader {
    /// Archetype index within this run.
    pub arch_id: u16,
    /// Component slot within the archetype, or [`ENTITY_SLOT`].
    pub slot: u16,
    /// Which page of the column (0-based).
    pub page_index: u16,
    /// Number of rows in this page (≤ [`PAGE_SIZE`]).
    pub row_count: u16,
    /// CRC32 of the page data bytes that immediately follow this header.
    pub page_crc32: u32,
    /// Explicit padding to reach 16 bytes; must be zero on write.
    #[allow(clippy::pub_underscore_fields)]
    pub _padding: u32,
}

impl PageHeader {
    /// View this header as a 16-byte slice for I/O.
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: PageHeader is #[repr(C)] with compile-time-verified size 16.
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// Interpret a 16-byte slice as a `PageHeader` reference.
    pub fn from_bytes(bytes: &[u8; 16]) -> &Self {
        // SAFETY: PageHeader is #[repr(C)]; any 16-byte bit pattern is valid.
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

// ── Index Entry (16 bytes) ───────────────────────────────────────────────────

/// Sparse index entry mapping `(arch_id, slot, page_index)` to a file offset.
///
/// Entries are stored in sorted order so readers can binary-search for a page.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IndexEntry {
    /// Archetype index.
    pub arch_id: u16,
    /// Component slot, or [`ENTITY_SLOT`].
    pub slot: u16,
    /// Page index within the column.
    pub page_index: u16,
    /// Explicit padding; must be zero on write.
    #[allow(clippy::pub_underscore_fields)]
    pub _pad: u16,
    /// Absolute byte offset of the [`PageHeader`] for this page.
    pub file_offset: u64,
}

impl IndexEntry {
    /// View this entry as a 16-byte slice for I/O.
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: IndexEntry is #[repr(C)] with compile-time-verified size 16.
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// Interpret a 16-byte slice as an `IndexEntry` reference.
    pub fn from_bytes(bytes: &[u8; 16]) -> &Self {
        // SAFETY: IndexEntry is #[repr(C)]; any 16-byte bit pattern is valid.
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }

    /// The sort key used for binary search.
    #[inline]
    fn sort_key(&self) -> (u16, u16, u16) {
        (self.arch_id, self.slot, self.page_index)
    }
}

impl PartialEq for IndexEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key() == other.sort_key()
    }
}

impl Eq for IndexEntry {}

impl PartialOrd for IndexEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

// ── File Footer (64 bytes) ───────────────────────────────────────────────────

/// Fixed-size footer written at the end of the file, immediately before the
/// final 8 bytes used to store the footer's own offset (not included here).
#[repr(C)]
pub struct Footer {
    /// Absolute byte offset of the first [`IndexEntry`].
    pub sparse_index_offset: u64,
    /// Number of [`IndexEntry`] records.
    pub sparse_index_count: u64,
    /// Absolute byte offset of the schema section.
    pub schema_offset: u64,
    /// Absolute byte offset of the Bloom filter section (0 if absent).
    pub bloom_filter_offset: u64,
    /// CRC32 over all page data, headers, and the schema section.
    pub total_crc32: u32,
    /// Reserved for future use; must be zero on write.
    pub reserved: [u8; 28],
}

impl Footer {
    /// View this footer as a 64-byte slice for I/O.
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Footer is #[repr(C)] with compile-time-verified size 64.
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// Interpret a 64-byte slice as a `Footer` reference.
    pub fn from_bytes(bytes: &[u8; 64]) -> &Self {
        // SAFETY: Footer is #[repr(C)]; any 64-byte bit pattern is valid.
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_bytes_are_correct() {
        assert_eq!(&MAGIC, b"MKLSM01\0");
    }

    #[test]
    fn struct_sizes_are_correct() {
        assert_eq!(std::mem::size_of::<Header>(), 64);
        assert_eq!(std::mem::size_of::<PageHeader>(), 16);
        assert_eq!(std::mem::size_of::<IndexEntry>(), 16);
        assert_eq!(std::mem::size_of::<Footer>(), 64);
    }

    #[test]
    fn index_entry_ordering() {
        let make = |arch_id, slot, page_index| IndexEntry {
            arch_id,
            slot,
            page_index,
            _pad: 0,
            file_offset: 0,
        };

        let a = make(1, 2, 3);
        let b = make(1, 2, 4);
        let c = make(1, 3, 0);
        let d = make(2, 0, 0);

        assert!(a < b);
        assert!(b < c);
        assert!(c < d);
        assert!(a < d);
    }

    #[test]
    fn header_round_trip() {
        let original = Header {
            magic: MAGIC,
            version: VERSION,
            schema_count: 3,
            page_count: 42,
            sequence_lo: 100,
            sequence_hi: 200,
            header_crc32: 0xDEAD_BEEF,
            reserved: [0u8; 20],
        };

        let bytes = original.as_bytes();
        let recovered = Header::from_bytes(bytes);

        assert_eq!(recovered.magic, MAGIC);
        assert_eq!(recovered.version, VERSION);
        assert_eq!(recovered.schema_count, 3);
        assert_eq!(recovered.page_count, 42);
        assert_eq!(recovered.sequence_lo, 100);
        assert_eq!(recovered.sequence_hi, 200);
        assert_eq!(recovered.header_crc32, 0xDEAD_BEEF);
    }

    #[test]
    fn page_header_round_trip() {
        let original = PageHeader {
            arch_id: 7,
            slot: ENTITY_SLOT,
            page_index: 15,
            row_count: 256,
            page_crc32: 0x1234_5678,
            _padding: 0,
        };

        let bytes = original.as_bytes();
        let recovered = PageHeader::from_bytes(bytes);

        assert_eq!(recovered.arch_id, 7);
        assert_eq!(recovered.slot, ENTITY_SLOT);
        assert_eq!(recovered.page_index, 15);
        assert_eq!(recovered.row_count, 256);
        assert_eq!(recovered.page_crc32, 0x1234_5678);
        assert_eq!(recovered._padding, 0);
    }
}
