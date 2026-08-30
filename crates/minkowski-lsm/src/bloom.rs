//! Cache-line-blocked bloom filter for sorted-run page membership.
//!
//! One `BlockedBloomFilter` per sorted run. Each 64-byte block holds 8 u64
//! words; each key sets 1 bit per word (8 bits total). Probing touches
//! exactly one cache line regardless of the number of hash functions.

use crate::error::LsmError;
use crate::format::IndexEntry;
use std::io::{Seek, Write};

const BITS_PER_KEY: usize = 10;
const BLOCK_BITS: usize = 512;
const BLOCK_BYTES: usize = 64;
const WORDS_PER_BLOCK: usize = 8;
const NUM_HASHES: usize = 8;

#[repr(C, align(64))]
#[derive(Clone)]
struct Block {
    words: [u64; WORDS_PER_BLOCK],
}

impl Block {
    const ZERO: Block = Block {
        words: [0u64; WORDS_PER_BLOCK],
    };
}

#[repr(C)]
struct BloomHeader {
    num_blocks: [u8; 4],
    seed: [u8; 8],
    _padding: [u8; 4],
}

impl BloomHeader {
    fn to_bytes(&self) -> &[u8; 16] {
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    fn from_bytes(bytes: &[u8; 16]) -> Self {
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Self) }
    }
}

pub struct BlockedBloomFilter {
    blocks: Vec<Block>,
    num_blocks: u32,
    seed: u64,
}

fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb13311177);
    z ^ (z >> 31)
}

pub fn pack_page_key(arch_id: u16, slot: u16, page_index: u16) -> u64 {
    ((arch_id as u64) << 32) | ((slot as u64) << 16) | (page_index as u64)
}

impl BlockedBloomFilter {
    pub fn new(expected_keys: usize, seed: u64) -> Self {
        if expected_keys == 0 {
            return Self {
                blocks: Vec::new(),
                num_blocks: 0,
                seed,
            };
        }
        let num_blocks = (expected_keys * BITS_PER_KEY).div_ceil(BLOCK_BITS);
        let num_blocks = num_blocks.max(1) as u32;
        let blocks = vec![Block::ZERO; num_blocks as usize];
        Self {
            blocks,
            num_blocks,
            seed,
        }
    }

    pub fn insert(&mut self, key: u64) {
        if self.num_blocks == 0 {
            return;
        }
        let h1 = splitmix64(key ^ self.seed);
        let block_idx = (h1 % self.num_blocks as u64) as usize;
        let h2 = splitmix64(h1);
        let h3 = splitmix64(h2);

        let block = &mut self.blocks[block_idx];
        for i in 0..NUM_HASHES {
            let bit = ((h2 >> (i * 6)) ^ (h3 >> ((7 - i) * 6))) & 0x3F;
            block.words[i] |= 1u64 << bit;
        }
    }

    pub fn contains(&self, key: u64) -> bool {
        if self.num_blocks == 0 {
            return true;
        }
        let h1 = splitmix64(key ^ self.seed);
        let block_idx = (h1 % self.num_blocks as u64) as usize;
        let h2 = splitmix64(h1);
        let h3 = splitmix64(h2);

        let block = &self.blocks[block_idx];
        for i in 0..NUM_HASHES {
            let bit = ((h2 >> (i * 6)) ^ (h3 >> ((7 - i) * 6))) & 0x3F;
            if block.words[i] & (1u64 << bit) == 0 {
                return false;
            }
        }
        true
    }

    pub fn write_to(&self, w: &mut impl Write) -> Result<(), LsmError> {
        if self.num_blocks == 0 {
            return Err(LsmError::Format(
                "cannot serialize empty bloom filter".to_owned(),
            ));
        }
        let header = BloomHeader {
            num_blocks: self.num_blocks.to_le_bytes(),
            seed: self.seed.to_le_bytes(),
            _padding: [0u8; 4],
        };
        w.write_all(header.to_bytes())?;
        for block in &self.blocks {
            for &word in &block.words {
                w.write_all(&word.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

pub(crate) struct BloomView {
    blocks_ptr: *const Block,
    num_blocks: u32,
    seed: u64,
    _owned: Option<Vec<Block>>,
}

// SAFETY: BloomView holds a raw pointer into either the mmap (immutable after
// open, thread-safe like the rest of SortedRunReader) or an owned Vec<Block>
// (also thread-safe for reads). No &mut access exists after construction.
unsafe impl Send for BloomView {}
unsafe impl Sync for BloomView {}

impl BloomView {
    pub(crate) fn from_bytes(buf: &[u8], offset: usize) -> Result<Self, LsmError> {
        let header_size = 16;
        let header_end = offset
            .checked_add(header_size)
            .ok_or_else(|| LsmError::Format("bloom header offset overflow".to_owned()))?;
        if header_end > buf.len() {
            return Err(LsmError::Format(
                "bloom header extends beyond file".to_owned(),
            ));
        }
        let header_bytes: &[u8; 16] = buf[offset..header_end].try_into().expect("16 bytes");
        let header = BloomHeader::from_bytes(header_bytes);

        let num_blocks = u32::from_le_bytes(header.num_blocks);
        let seed = u64::from_le_bytes(header.seed);

        if num_blocks == 0 {
            return Ok(Self {
                blocks_ptr: std::ptr::null(),
                num_blocks: 0,
                seed,
                _owned: None,
            });
        }

        let blocks_offset = header_end;
        let blocks_len = (num_blocks as usize)
            .checked_mul(BLOCK_BYTES)
            .ok_or_else(|| LsmError::Format("bloom blocks length overflow".to_owned()))?;
        let blocks_end = blocks_offset
            .checked_add(blocks_len)
            .ok_or_else(|| LsmError::Format("bloom blocks offset overflow".to_owned()))?;
        if blocks_end > buf.len() {
            return Err(LsmError::Format(
                "bloom blocks extend beyond file".to_owned(),
            ));
        }

        let blocks_ptr = buf[blocks_offset..].as_ptr() as *const Block;
        let aligned = (blocks_ptr as usize).is_multiple_of(64);

        if aligned {
            // NOTE: zero-copy path reads Block words as native-endian u64.
            // `write_to` serializes words in little-endian, so this is
            // correct only on LE targets (x86-64, AArch64 in LE mode).
            // The unaligned fallback below does explicit LE decoding.
            Ok(Self {
                blocks_ptr,
                num_blocks,
                seed,
                _owned: None,
            })
        } else {
            // ponytail: BLOCK_BYTES is 64 (a full cache line); chunks_exact
            // batch counts in BLOCK_BYTES units, so each chunk is one block.
            let owned: Vec<Block> = buf[blocks_offset..blocks_end]
                .as_chunks::<BLOCK_BYTES>()
                .0
                .iter()
                .map(|chunk| {
                    let mut block = Block::ZERO;
                    for i in 0..WORDS_PER_BLOCK {
                        let start = i * 8;
                        block.words[i] = u64::from_le_bytes(
                            chunk[start..start + 8].try_into().expect("8 bytes"),
                        );
                    }
                    block
                })
                .collect();
            let ptr = owned.as_ptr();
            Ok(Self {
                blocks_ptr: ptr,
                num_blocks,
                seed,
                _owned: Some(owned),
            })
        }
    }

    pub fn contains(&self, key: u64) -> bool {
        if self.num_blocks == 0 {
            return true;
        }
        let h1 = splitmix64(key ^ self.seed);
        let block_idx = (h1 % self.num_blocks as u64) as usize;
        let h2 = splitmix64(h1);
        let h3 = splitmix64(h2);

        unsafe {
            let block = &*self.blocks_ptr.add(block_idx);
            for i in 0..NUM_HASHES {
                let bit = ((h2 >> (i * 6)) ^ (h3 >> ((7 - i) * 6))) & 0x3F;
                if block.words[i] & (1u64 << bit) == 0 {
                    return false;
                }
            }
        }
        true
    }
}

pub(crate) fn write_bloom_section(
    w: &mut (impl Write + Seek),
    index_entries: &[IndexEntry],
    seed: u64,
) -> Result<u64, LsmError> {
    if index_entries.is_empty() {
        return Ok(0);
    }
    let mut bloom = BlockedBloomFilter::new(index_entries.len(), seed);
    for entry in index_entries {
        bloom.insert(pack_page_key(entry.arch_id, entry.slot, entry.page_index));
    }
    let mut offset = w.stream_position()?;
    if offset % 64 != 0 {
        let pad = 64 - (offset as usize % 64);
        const ZEROS: [u8; 64] = [0u8; 64];
        w.write_all(&ZEROS[..pad])?;
        offset = w.stream_position()?;
    }
    assert!(
        offset % 64 == 0,
        "bloom filter section must be 64-byte aligned"
    );
    bloom.write_to(w)?;
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn insert_and_contains_zero_false_negatives() {
        let mut filter = BlockedBloomFilter::new(1000, 42);
        let mut keys = HashSet::new();
        for i in 0..1000u64 {
            filter.insert(i);
            keys.insert(i);
        }
        for key in &keys {
            assert!(filter.contains(*key), "false negative for key {key}");
        }
    }

    #[test]
    fn empty_filter_returns_true_for_all_probes() {
        let filter = BlockedBloomFilter::new(0, 42);
        assert!(filter.contains(0), "empty filter must return true");
        assert!(filter.contains(u64::MAX), "empty filter must return true");
    }

    #[test]
    fn seed_determinism() {
        let mut a = BlockedBloomFilter::new(100, 99);
        let mut b = BlockedBloomFilter::new(100, 99);
        for i in 0..100u64 {
            a.insert(i);
            b.insert(i);
        }
        for i in 0..200u64 {
            assert_eq!(a.contains(i), b.contains(i), "mismatch at key {i}");
        }
    }

    #[test]
    fn pack_page_key_layout() {
        let key = pack_page_key(1, 2, 3);
        assert_eq!((key >> 32) as u16, 1, "arch_id");
        assert_eq!(((key >> 16) & 0xFFFF) as u16, 2, "slot");
        assert_eq!((key & 0xFFFF) as u16, 3, "page_index");
    }

    #[test]
    fn pack_page_key_distinguishes_slots() {
        let a = pack_page_key(0, 1, 5);
        let b = pack_page_key(0, 2, 5);
        assert_ne!(a, b, "different slots must produce different keys");
    }

    #[test]
    fn false_positive_rate_within_bounds() {
        let n = 10_000usize;
        let mut filter = BlockedBloomFilter::new(n, 12345);
        let inserted: HashSet<u64> = (0..n as u64).collect();
        for &k in &inserted {
            filter.insert(k);
        }

        let mut fp = 0usize;
        let probes = 100_000u64;
        for p in (n as u64)..((n as u64) + probes) {
            if filter.contains(p) {
                fp += 1;
            }
        }
        let fpr = fp as f64 / probes as f64;
        assert!(fpr < 0.02, "FPR {fpr:.4} exceeds 2% threshold");
    }

    #[test]
    fn serialization_round_trip() {
        let mut filter = BlockedBloomFilter::new(500, 77);
        let keys: Vec<u64> = (0..500).collect();
        for &k in &keys {
            filter.insert(k);
        }

        let mut buf = Vec::new();
        filter.write_to(&mut buf).unwrap();

        let view = BloomView::from_bytes(&buf, 0).unwrap();
        for &k in &keys {
            assert!(
                view.contains(k),
                "false negative for key {k} after round-trip"
            );
        }

        let mut fp = 0;
        for p in 500..1500u64 {
            if view.contains(p) {
                fp += 1;
            }
        }
        let fpr = fp as f64 / 1000.0;
        assert!(fpr < 0.02, "FPR after round-trip {fpr:.4} exceeds 2%");
    }
}
