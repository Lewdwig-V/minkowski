//! Variable-length dense column page body (Arrow-style).
//!
//! A Serialized dense column page stores `n` rows of rkyv-serialized component
//! bytes as `[offsets: (n+1) × u32 LE][values]`, where `offsets[0] == 0`,
//! `offsets[i+1] - offsets[i]` is row `i`'s byte length, and `offsets[n]` is the
//! total values length. The page's `row_count` (in its `PageHeader`) is the true
//! row count `n`; the byte length is derived from the offset table, so a reader
//! that knows `n` can both bound the page and split it into rows.

use crate::error::LsmError;

/// Encode `rows` (each already rkyv-serialized) into one page body.
pub fn encode(rows: &[Vec<u8>]) -> Vec<u8> {
    let n = rows.len();
    let total_values: usize = rows.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(4 * (n + 1) + total_values);
    let mut acc: u32 = 0;
    out.extend_from_slice(&acc.to_le_bytes());
    for row in rows {
        acc = acc
            .checked_add(u32::try_from(row.len()).expect("row exceeds u32 bytes"))
            .expect("serialized page values exceed u32 bytes");
        out.extend_from_slice(&acc.to_le_bytes());
    }
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

/// Total encoded byte length, given `row_count` and a `data` slice that contains
/// at least the full `(row_count + 1)` offset table. Used by the reader to bound
/// the page slice before reading the values.
pub fn encoded_len(data: &[u8], row_count: usize) -> Result<usize, LsmError> {
    let table_len = 4 * (row_count + 1);
    if data.len() < table_len {
        return Err(LsmError::Format(format!(
            "serialized page: offset table needs {table_len} bytes, have {}",
            data.len()
        )));
    }
    let last = u32::from_le_bytes(
        data[4 * row_count..4 * row_count + 4]
            .try_into()
            .expect("4 bytes"),
    ) as usize;
    Ok(table_len + last)
}

/// Split a page body into `row_count` borrowed row slices. Validates the offset
/// table is monotonic and in range.
pub fn decode(data: &[u8], row_count: usize) -> Result<Vec<&[u8]>, LsmError> {
    let table_len = 4 * (row_count + 1);
    if data.len() < table_len {
        return Err(LsmError::Format(format!(
            "serialized page: offset table needs {table_len} bytes, have {}",
            data.len()
        )));
    }
    let off = |i: usize| -> usize {
        u32::from_le_bytes(data[4 * i..4 * i + 4].try_into().expect("4 bytes")) as usize
    };
    if off(0) != 0 {
        return Err(LsmError::Format(format!(
            "serialized page: offsets[0] = {}, expected 0",
            off(0)
        )));
    }
    let values = &data[table_len..];
    // The values region must be EXACTLY consumed by the offset table. A `values`
    // region longer than `offsets[row_count]` implies a row-count under-count
    // (decoding `n` rows from a body that actually holds more), which would
    // silently return a truncated row set. Reject it.
    let declared = off(row_count);
    if declared != values.len() {
        return Err(LsmError::Format(format!(
            "serialized page: values region is {} bytes, offset table accounts for {declared} \
             (row_count {row_count} under-counts the page)",
            values.len()
        )));
    }
    let mut rows = Vec::with_capacity(row_count);
    for i in 0..row_count {
        let (start, end) = (off(i), off(i + 1));
        if end < start || end > values.len() {
            return Err(LsmError::Format(format!(
                "serialized page: row {i} offsets [{start}..{end}] out of range \
                 (values len {})",
                values.len()
            )));
        }
        rows.push(&values[start..end]);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_multiple_rows() {
        let rows = vec![b"alpha".to_vec(), b"".to_vec(), b"gamma!!".to_vec()];
        let body = encode(&rows);
        assert_eq!(encoded_len(&body, rows.len()).unwrap(), body.len());
        let decoded = decode(&body, rows.len()).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], b"alpha");
        assert_eq!(decoded[1], b"");
        assert_eq!(decoded[2], b"gamma!!");
    }

    #[test]
    fn round_trip_zero_rows() {
        let body = encode(&[]);
        // Just the single offsets[0] = 0 entry.
        assert_eq!(body, 0u32.to_le_bytes());
        assert_eq!(encoded_len(&body, 0).unwrap(), 4);
        assert!(decode(&body, 0).unwrap().is_empty());
    }

    #[test]
    fn round_trip_single_row() {
        let rows = vec![b"only".to_vec()];
        let body = encode(&rows);
        let decoded = decode(&body, 1).unwrap();
        assert_eq!(decoded[0], b"only");
    }

    #[test]
    fn decode_rejects_truncated_offset_table() {
        let body = encode(&[b"x".to_vec(), b"y".to_vec()]);
        // Claim 2 rows but hand decode only 4 bytes (needs 12 for the table).
        assert!(matches!(decode(&body[..4], 2), Err(LsmError::Format(_))));
    }

    #[test]
    fn decode_rejects_truncated_values() {
        let mut body = encode(&[b"abcd".to_vec()]);
        body.truncate(body.len() - 2); // chop the values tail
        assert!(matches!(decode(&body, 1), Err(LsmError::Format(_))));
    }

    #[test]
    fn decode_rejects_nonzero_first_offset() {
        let mut body = encode(&[b"ab".to_vec()]);
        body[0..4].copy_from_slice(&1u32.to_le_bytes()); // offsets[0] = 1
        assert!(matches!(decode(&body, 1), Err(LsmError::Format(_))));
    }

    #[test]
    fn decode_rejects_non_monotonic_offsets() {
        // Two rows of 5 bytes each → offsets [0, 5, 10]. Corrupt the middle
        // offset so the table descends ([0, 8, 10] keeps it monotonic; use
        // [0, 8, 6] so row 1 has end < start) and confirm the `end < start`
        // guard fires rather than slicing backwards.
        let mut body = encode(&[b"aaaaa".to_vec(), b"bbbbb".to_vec()]);
        body[4..8].copy_from_slice(&8u32.to_le_bytes()); // offsets[1] = 8
        body[8..12].copy_from_slice(&6u32.to_le_bytes()); // offsets[2] = 6 (< 8)
        assert!(matches!(decode(&body, 2), Err(LsmError::Format(_))));
    }

    #[test]
    fn decode_rejects_trailing_data() {
        // A well-formed body for two rows, with an extra byte appended to the
        // values region. `offsets[n]` no longer equals `values.len()`, so decode
        // must reject it rather than silently returning a truncated row set (the
        // under-count footgun: `decode(body, n)` with `n` smaller than the true
        // count).
        let mut body = encode(&[b"aa".to_vec(), b"bbb".to_vec()]);
        body.push(0xFF); // trailing byte past the values region
        assert!(matches!(decode(&body, 2), Err(LsmError::Format(_))));
    }

    #[test]
    fn decode_rejects_offset_past_values_with_full_table() {
        // Full, well-formed offset table but offsets[n] claims more value bytes
        // than the page actually carries. The `end > values.len()` guard must
        // catch this before any slice executes.
        let mut body = encode(&[b"xy".to_vec()]);
        body[4..8].copy_from_slice(&999u32.to_le_bytes()); // offsets[1] = 999
        assert!(matches!(decode(&body, 1), Err(LsmError::Format(_))));
    }
}
