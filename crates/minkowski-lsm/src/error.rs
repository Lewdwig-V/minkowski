use std::io;

/// Errors that can occur during LSM file operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum LsmError {
    /// Underlying I/O error.
    Io(io::Error),
    /// File format violation (bad magic, unsupported version, truncated data, etc.).
    Format(String),
    /// CRC mismatch at a given file offset.
    Crc {
        /// Byte offset of the record whose checksum failed.
        offset: u64,
        /// Checksum stored in the file.
        expected: u32,
        /// Checksum computed over the actual bytes.
        actual: u32,
    },
    /// The recovered state does not fit the target world's memory pool.
    PoolExhausted(minkowski::PoolExhausted),
}

impl std::fmt::Display for LsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
            Self::Crc {
                offset,
                expected,
                actual,
            } => write!(
                f,
                "CRC mismatch at offset {offset}: expected {expected:#010x}, got {actual:#010x}"
            ),
            Self::PoolExhausted(e) => write!(f, "recovery does not fit the world's pool: {e}"),
        }
    }
}

impl std::error::Error for LsmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::PoolExhausted(e) => Some(e),
            Self::Format(_) | Self::Crc { .. } => None,
        }
    }
}

impl From<io::Error> for LsmError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<minkowski::PoolExhausted> for LsmError {
    fn from(e: minkowski::PoolExhausted) -> Self {
        Self::PoolExhausted(e)
    }
}
