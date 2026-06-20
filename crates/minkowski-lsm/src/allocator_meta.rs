//! Entity allocator snapshot encoding for sorted-run metadata pages.

use crate::error::LsmError;

/// Encode `(generations, free_list)` into a byte blob for an allocator page.
pub fn encode(generations: &[u32], free_list: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + generations.len() * 4 + free_list.len() * 4);
    out.extend_from_slice(&(generations.len() as u64).to_le_bytes());
    for &g in generations {
        out.extend_from_slice(&g.to_le_bytes());
    }
    out.extend_from_slice(&(free_list.len() as u64).to_le_bytes());
    for &f in free_list {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Decode allocator metadata written by [`encode`].
pub fn decode(bytes: &[u8]) -> Result<(Vec<u32>, Vec<u32>), LsmError> {
    let mut pos = 0;
    let read_u64 = |bytes: &[u8], pos: &mut usize| -> Result<u64, LsmError> {
        let end = pos
            .checked_add(8)
            .ok_or_else(|| LsmError::Format("allocator meta truncated".to_owned()))?;
        if end > bytes.len() {
            return Err(LsmError::Format("allocator meta truncated".to_owned()));
        }
        let val = u64::from_le_bytes(bytes[*pos..end].try_into().expect("8 bytes"));
        *pos = end;
        Ok(val)
    };
    let read_u32_slice =
        |bytes: &[u8], pos: &mut usize, count: u64| -> Result<Vec<u32>, LsmError> {
            let count = usize::try_from(count).map_err(|_| {
                LsmError::Format(format!(
                    "allocator meta count {count} exceeds address space"
                ))
            })?;
            let byte_len = count
                .checked_mul(4)
                .ok_or_else(|| LsmError::Format("allocator meta length overflow".to_owned()))?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| LsmError::Format("allocator meta truncated".to_owned()))?;
            if end > bytes.len() {
                return Err(LsmError::Format("allocator meta truncated".to_owned()));
            }
            let mut out = Vec::with_capacity(count);
            for chunk in bytes[*pos..end].chunks_exact(4) {
                out.push(u32::from_le_bytes(chunk.try_into().expect("4 bytes")));
            }
            *pos = end;
            Ok(out)
        };

    let gen_count = read_u64(bytes, &mut pos)?;
    let generations = read_u32_slice(bytes, &mut pos, gen_count)?;
    let free_count = read_u64(bytes, &mut pos)?;
    let free_list = read_u32_slice(bytes, &mut pos, free_count)?;
    if pos != bytes.len() {
        return Err(LsmError::Format(format!(
            "allocator meta trailing bytes: {} leftover",
            bytes.len() - pos
        )));
    }
    Ok((generations, free_list))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty() {
        let bytes = encode(&[], &[]);
        let (gens, free) = decode(&bytes).unwrap();
        assert!(gens.is_empty());
        assert!(free.is_empty());
    }

    #[test]
    fn round_trip_nonempty() {
        let bytes = encode(&[0, 1, 2], &[3, 4]);
        let (gens, free) = decode(&bytes).unwrap();
        assert_eq!(gens, vec![0, 1, 2]);
        assert_eq!(free, vec![3, 4]);
    }
}
