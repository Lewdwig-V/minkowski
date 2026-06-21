//! Canonical fingerprint binding a run's Serialized-column layout to the binary
//! that may decode it via the unchecked (bytecheck-skipping) recovery path.
//! Computed IDENTICALLY by the flush writer, compaction, and recovery — the
//! single source of truth (spec §2.1). A divergence here silently mis-gates the
//! unsafe path, so all sites call `run_fingerprint`.
//!
//! The hash is deterministic FNV-1a — it is persisted to disk and compared
//! across binary invocations and platforms. NEVER use a randomly-seeded hasher.

use crate::codec::CodecRegistry;
use crate::schema::{SchemaSection, StorageKind};

/// Bump on any rkyv upgrade that may change archived layouts WITHOUT changing
/// native sizes (size-preserving internal layout changes that the per-component
/// sizes below cannot detect). Bumping forces every prior run to fail the gate →
/// checked decode. Fails closed.
pub const RKYV_DECODE_EPOCH: u64 = 1;

/// One Serialized component's layout identity. Callers pass these sorted by name.
#[derive(Clone, Copy, Debug)]
pub struct FingerprintEntry<'a> {
    pub name: &'a str,
    pub native_size: u64,
    pub native_align: u64,
    pub archived_size: u64,
}

#[inline]
fn fnv1a(h: &mut u64, bytes: &[u8]) {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

/// FNV-1a over the epoch then each entry. `entries` MUST be sorted by name.
#[must_use]
pub fn decode_fingerprint(entries: &[FingerprintEntry<'_>]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h = FNV_OFFSET;
    fnv1a(&mut h, &RKYV_DECODE_EPOCH.to_le_bytes());
    for e in entries {
        fnv1a(&mut h, &(e.name.len() as u64).to_le_bytes()); // length-prefix: unambiguous
        fnv1a(&mut h, e.name.as_bytes());
        fnv1a(&mut h, &e.native_size.to_le_bytes());
        fnv1a(&mut h, &e.native_align.to_le_bytes());
        fnv1a(&mut h, &e.archived_size.to_le_bytes());
    }
    h
}

/// The fingerprint for a run, computed over its Serialized schema entries with
/// archived sizes looked up in `codecs`. THE shared entry point: writer,
/// compaction, and recovery all call this so their hashes agree by construction.
///
/// A Serialized component whose codec is missing (e.g. recovery without it
/// registered) gets `u64::MAX` as a poison archived size, guaranteeing a
/// mismatch → checked decode (the run cannot be decoded unchecked anyway).
#[must_use]
pub fn run_fingerprint(schema: &SchemaSection, codecs: &CodecRegistry) -> u64 {
    let mut entries: Vec<FingerprintEntry<'_>> = schema
        .entries()
        .iter()
        .filter(|e| e.storage_kind() == StorageKind::Serialized)
        .map(|e| {
            // Source native AND archived layout from the LIVE codecs (this
            // binary), NOT the on-disk schema. If recovery read native size/align
            // back from `reader.schema()` (the writer's stamped values), the
            // native terms would be identical on both sides and detect nothing —
            // only `archived_size` would discriminate (soundness audit N2). A
            // missing codec poisons to u64::MAX → guaranteed mismatch → checked.
            let (native_size, native_align) = codecs
                .native_layout_by_name(e.name())
                .map_or((u64::MAX, u64::MAX), |(s, a)| (s as u64, a as u64));
            let archived_size = codecs
                .archived_size_by_name(e.name())
                .map_or(u64::MAX, |s| s as u64);
            FingerprintEntry {
                name: e.name(),
                native_size,
                native_align,
                archived_size,
            }
        })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(b.name)); // schema is already name-sorted; explicit for safety
    decode_fingerprint(&entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(name: &str, ns: u64, na: u64, asz: u64) -> FingerprintEntry<'_> {
        FingerprintEntry {
            name,
            native_size: ns,
            native_align: na,
            archived_size: asz,
        }
    }

    #[test]
    fn deterministic_same_input_same_hash() {
        let v = [e("a", 8, 4, 8), e("b", 24, 8, 16)];
        assert_eq!(decode_fingerprint(&v), decode_fingerprint(&v));
    }

    #[test]
    fn archived_size_change_changes_hash() {
        let a = [e("name", 24, 8, 16)];
        let b = [e("name", 24, 8, 20)];
        assert_ne!(decode_fingerprint(&a), decode_fingerprint(&b));
    }

    #[test]
    fn epoch_is_mixed_in() {
        // Empty entry set still depends on the epoch constant (non-zero hash).
        assert_ne!(decode_fingerprint(&[]), 0);
    }

    #[test]
    fn name_boundaries_are_unambiguous() {
        // length-prefixing prevents "ab"+"c" colliding with "a"+"bc".
        let x = [e("ab", 1, 1, 1), e("c", 1, 1, 1)];
        let y = [e("a", 1, 1, 1), e("bc", 1, 1, 1)];
        assert_ne!(decode_fingerprint(&x), decode_fingerprint(&y));
    }

    #[derive(Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    struct HeapC {
        s: String,
    }

    #[test]
    fn run_fingerprint_uses_live_codec_native_layout_not_schema() {
        use crate::schema::{SchemaSection, StorageKind};
        use minkowski::World;
        let mut world = World::new();
        let mut codecs = crate::codec::CodecRegistry::new();
        codecs.register_as::<HeapC>("hc", &mut world).unwrap();
        // Two schemas for "hc" differing ONLY in the (bogus) native item_size.
        let bogus = SchemaSection::from_components(&[(
            "hc".to_owned(),
            std::alloc::Layout::from_size_align(999, 1).unwrap(),
            StorageKind::Serialized,
        )])
        .unwrap();
        let real = SchemaSection::from_components(&[(
            "hc".to_owned(),
            std::alloc::Layout::new::<HeapC>(),
            StorageKind::Serialized,
        )])
        .unwrap();
        // Native layout comes from the codec, so the bogus schema size is ignored:
        // both fingerprints are equal. (Pre-fix, they would DIFFER.)
        assert_eq!(
            run_fingerprint(&bogus, &codecs),
            run_fingerprint(&real, &codecs)
        );
    }

    #[test]
    fn run_fingerprint_poisons_missing_codec() {
        use crate::schema::{SchemaSection, StorageKind};
        use minkowski::World;
        let mut world = World::new();
        let mut with = crate::codec::CodecRegistry::new();
        with.register_as::<HeapC>("hc", &mut world).unwrap();
        let without = crate::codec::CodecRegistry::new(); // "hc" unregistered
        let schema = SchemaSection::from_components(&[(
            "hc".to_owned(),
            std::alloc::Layout::new::<HeapC>(),
            StorageKind::Serialized,
        )])
        .unwrap();
        // Missing codec ⇒ u64::MAX poison ⇒ a different fingerprint ⇒ mismatch ⇒
        // recovery falls back to checked decode. (Closes a Task-2 review gap too.)
        assert_ne!(
            run_fingerprint(&schema, &with),
            run_fingerprint(&schema, &without)
        );
    }
}
