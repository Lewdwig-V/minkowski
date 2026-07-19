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
/// native + archived layout resolved from the live `codecs` BY THE SCHEMA NAME
/// (the Rust `type_name`). THE shared entry point: writer, compaction, and
/// recovery all call this so their hashes agree by construction.
///
/// Returns `0` (fail CLOSED) if ANY Serialized component's layout cannot be
/// resolved — recovery's `decode_fingerprint() != 0` guard then forces checked
/// decode. A `u64::MAX` poison sentinel would be WRONG here: the same poison is
/// hashed at flush and recovery, so it would self-match and wrongly ENABLE the
/// unchecked path without proving the layout.
#[must_use]
pub fn run_fingerprint(schema: &SchemaSection, codecs: &CodecRegistry) -> u64 {
    let mut entries: Vec<FingerprintEntry<'_>> = Vec::new();
    for e in schema
        .entries()
        .iter()
        .filter(|e| e.storage_kind() == StorageKind::Serialized)
    {
        let Some((native_size, native_align)) = codecs.native_layout_by_type_name(e.name()) else {
            return 0;
        };
        let Some(archived_size) = codecs.archived_size_by_type_name(e.name()) else {
            return 0;
        };
        entries.push(FingerprintEntry {
            name: e.name(),
            native_size: native_size as u64,
            native_align: native_align as u64,
            archived_size: archived_size as u64,
        });
    }
    entries.sort_by(|a, b| a.name.cmp(b.name));
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
    fn native_size_change_changes_hash() {
        let a = [e("name", 24, 8, 16)];
        let b = [e("name", 32, 8, 16)];
        assert_ne!(decode_fingerprint(&a), decode_fingerprint(&b));
    }

    #[test]
    fn native_align_change_changes_hash() {
        let a = [e("name", 24, 8, 16)];
        let b = [e("name", 24, 4, 16)];
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

    fn heap_schema() -> crate::schema::SchemaSection {
        use crate::schema::{SchemaSection, StorageKind};
        // Production stores the Rust type_name as the dense schema name.
        SchemaSection::from_components(&[(
            std::any::type_name::<HeapC>().to_owned(),
            std::alloc::Layout::new::<HeapC>(),
            StorageKind::Serialized,
        )])
        .unwrap()
    }

    #[test]
    fn run_fingerprint_resolves_aliased_codec_by_type_name() {
        use minkowski::World;
        let mut world = World::new();
        let mut codecs = crate::codec::CodecRegistry::new();
        // register_as with an ALIAS != type_name — the case that used to poison.
        codecs.register_as::<HeapC>("hc-alias", &mut world).unwrap();
        // Resolves via type_name despite the alias ⇒ a real, non-zero fingerprint.
        assert_ne!(run_fingerprint(&heap_schema(), &codecs), 0);
    }

    #[test]
    fn run_fingerprint_fails_closed_to_zero_on_missing_codec() {
        // No codec registered for HeapC's type_name ⇒ fail CLOSED to 0 (NOT a
        // self-matching poison). This is the P1#2 regression guard.
        let codecs = crate::codec::CodecRegistry::new();
        assert_eq!(run_fingerprint(&heap_schema(), &codecs), 0);
    }

    #[test]
    fn run_fingerprint_present_vs_missing_differ() {
        use minkowski::World;
        let mut world = World::new();
        let mut with = crate::codec::CodecRegistry::new();
        with.register_as::<HeapC>("hc-alias", &mut world).unwrap();
        let without = crate::codec::CodecRegistry::new();
        // Present ⇒ real non-zero hash; missing ⇒ 0. They MUST differ (pre-fix,
        // both produced the same self-matching poison hash).
        assert_ne!(
            run_fingerprint(&heap_schema(), &with),
            run_fingerprint(&heap_schema(), &without)
        );
    }
}
