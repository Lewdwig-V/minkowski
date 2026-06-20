//! Length-prefixed, self-delimiting encoding for a sparse component's baseline
//! pages. One blob per sparse component: `[count u32][(entity_bits u64,
//! value_len u32, value_bytes)…]`. Mirrors `allocator_meta` (self-describing,
//! trailing-byte-checked). The blob is chunked at `u16::MAX` bytes across
//! `page_index` by the writer and concatenated back by recovery before decode.

use crate::error::LsmError;

/// Encode a sparse component's entries into a self-delimiting blob.
pub fn encode(entries: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let total: usize = 4 + entries.iter().map(|(_, v)| 8 + 4 + v.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
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

/// Decode a blob produced by [`encode`]. Errors on truncation or trailing bytes.
pub fn decode(bytes: &[u8]) -> Result<Vec<(u64, Vec<u8>)>, LsmError> {
    let mut pos = 0usize;
    let count = read_u32(bytes, &mut pos)? as usize;
    let mut entries = Vec::with_capacity(count);
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
    Ok(entries)
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
        let blob = encode(&entries);
        let decoded = decode(&blob).unwrap();
        assert_eq!(decoded, entries);
    }

    #[test]
    fn round_trips_empty() {
        let entries: Vec<(u64, Vec<u8>)> = Vec::new();
        let blob = encode(&entries);
        assert_eq!(decode(&blob).unwrap(), entries);
    }

    #[test]
    fn rejects_truncated() {
        let blob = encode(&[(1u64, vec![1, 2, 3])]);
        let truncated = &blob[..blob.len() - 1];
        assert!(decode(truncated).is_err());
    }

    #[test]
    fn rejects_trailing_garbage() {
        let mut blob = encode(&[(1u64, vec![1, 2, 3])]);
        blob.push(0xAB);
        assert!(decode(&blob).is_err());
    }
}
