//! Length-prefixed, self-delimiting encoding for a sparse component's baseline
//! pages. One blob per sparse component, carrying its own stable component
//! NAME so recovery can resolve the component without consulting the run's
//! `SchemaSection`: `[name_len u32][name_utf8][count u32][(entity_bits u64,
//! value_len u32, value_bytes)…]`.
//!
//! Keeping the name in the blob deliberately decouples sparse storage from the
//! archetype schema: sparse components are NOT added to `SchemaSection`, so
//! they cannot perturb archetype component slot assignments. The blob is
//! chunked at `u16::MAX` bytes across `page_index` by the writer and
//! concatenated back by recovery before decode. Mirrors `allocator_meta`
//! (self-describing, trailing-byte-checked).

use crate::error::LsmError;

/// A decoded sparse blob: the component's stable name and its
/// `(entity_bits, value_bytes)` entries.
pub type DecodedSparse = (String, Vec<(u64, Vec<u8>)>);

/// Encode a sparse component's name + entries into a self-delimiting blob.
pub fn encode(name: &str, entries: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let total: usize =
        4 + name.len() + 4 + entries.iter().map(|(_, v)| 8 + 4 + v.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(
        &u32::try_from(name.len())
            .expect("name length exceeds u32")
            .to_le_bytes(),
    );
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(
        &u32::try_from(entries.len())
            .expect("entry count exceeds u32")
            .to_le_bytes(),
    );
    for (entity_bits, value) in entries {
        out.extend_from_slice(&entity_bits.to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(value.len())
                .expect("value length exceeds u32")
                .to_le_bytes(),
        );
        out.extend_from_slice(value);
    }
    out
}

/// Decode a blob produced by [`encode`], returning the component name and its
/// entries. Errors on truncation, invalid UTF-8 name, or trailing bytes.
pub fn decode(bytes: &[u8]) -> Result<DecodedSparse, LsmError> {
    let mut pos = 0usize;
    let name_len = read_u32(bytes, &mut pos)? as usize;
    if name_len == 0 {
        return Err(LsmError::Format(
            "sparse_page: component name is empty".to_owned(),
        ));
    }
    let name_end = pos
        .checked_add(name_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| LsmError::Format("sparse_page: name region truncated".to_owned()))?;
    let name = std::str::from_utf8(&bytes[pos..name_end])
        .map_err(|e| LsmError::Format(format!("sparse_page: invalid component name: {e}")))?
        .to_owned();
    pos = name_end;

    let count = read_u32(bytes, &mut pos)? as usize;
    // Cap the pre-allocation against the remaining bytes to avoid an OOM
    // panic on a corrupt or attacker-crafted blob with a huge `count`. Each
    // entry is at least 8 (entity_bits) + 4 (value_len) + 0 (empty value) =
    // 12 bytes, so `remaining / 12` is an upper bound on the number of
    // complete entries that could possibly follow. The loop's per-entry
    // truncation check still fires on a short body; this just prevents the
    // `Vec::with_capacity(count)` from requesting ~96 GB for count=u32::MAX.
    let max_entries = (bytes.len() - pos) / 12;
    let mut entries = Vec::with_capacity(count.min(max_entries));
    for _ in 0..count {
        let entity_bits = read_u64(bytes, &mut pos)?;
        let value_len = read_u32(bytes, &mut pos)? as usize;
        let end = pos
            .checked_add(value_len)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| LsmError::Format("sparse_page: value region truncated".to_owned()))?;
        entries.push((entity_bits, bytes[pos..end].to_vec()));
        pos = end;
    }
    if pos != bytes.len() {
        return Err(LsmError::Format(format!(
            "sparse_page: {} trailing bytes after decode",
            bytes.len() - pos
        )));
    }
    Ok((name, entries))
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, LsmError> {
    let end = *pos + 4;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| LsmError::Format("sparse_page: truncated u32".to_owned()))?;
    *pos = end;
    Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64, LsmError> {
    let end = *pos + 8;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| LsmError::Format("sparse_page: truncated u64".to_owned()))?;
    *pos = end;
    Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_entries() {
        let entries = vec![
            (1u64, vec![10u8, 11, 12]),
            (2u64, vec![]),
            (0xFFFF_FFFF_0000_0001u64, vec![42u8; 300]),
        ];
        let blob = encode("my_component", &entries);
        let (name, decoded) = decode(&blob).unwrap();
        assert_eq!(name, "my_component");
        assert_eq!(decoded, entries);
    }

    #[test]
    fn round_trips_empty() {
        let entries: Vec<(u64, Vec<u8>)> = Vec::new();
        let blob = encode("empty_component", &entries);
        let (name, decoded) = decode(&blob).unwrap();
        assert_eq!(name, "empty_component");
        assert_eq!(decoded, entries);
    }

    #[test]
    fn rejects_truncated() {
        let blob = encode("c", &[(1u64, vec![1, 2, 3])]);
        let truncated = &blob[..blob.len() - 1];
        assert!(decode(truncated).is_err());
    }

    #[test]
    fn rejects_trailing_garbage() {
        let mut blob = encode("c", &[(1u64, vec![1, 2, 3])]);
        blob.push(0xAB);
        assert!(decode(&blob).is_err());
    }

    /// A corrupt or attacker-crafted blob with `count = u32::MAX` must NOT
    /// trigger an OOM panic in `Vec::with_capacity`. The pre-allocation is
    /// capped against the remaining bytes (min entry = 12 bytes), so the loop
    /// runs at most `remaining / 12` iterations before hitting the truncation
    /// check and returning `Err`.
    #[test]
    fn decode_huge_count_does_not_oom() {
        // Hand-craft: name_len=1, name="c", count=0xFFFF_FFFF, then a short body
        // (no real entries). Without the cap this would attempt
        // Vec::with_capacity(4_294_967_295) → OOM panic.
        let mut blob = Vec::new();
        blob.extend_from_slice(&1u32.to_le_bytes()); // name_len
        blob.extend_from_slice(b"c");
        blob.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // count
        blob.extend_from_slice(&[0u8; 8]); // a few trailing bytes, far short of 4B entries
        let result = decode(&blob);
        assert!(
            result.is_err(),
            "a huge count with a short body must error, not OOM"
        );
    }
}
