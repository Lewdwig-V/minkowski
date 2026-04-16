//! Invariant-carrying newtype primitives shared across the manifest.

use std::fmt;

use crate::error::LsmError;
use crate::manifest::NUM_LEVELS;

/// A WAL sequence number.
///
/// Raw `u64` arithmetic on sequence numbers is disallowed by the type
/// (no `Add`/`Sub` impls) because sequences are identities, not sizes.
/// Callers that need "next seq" do so explicitly: `SeqNo(x.0 + 1)`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SeqNo(pub u64);

impl From<u64> for SeqNo {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<SeqNo> for u64 {
    fn from(s: SeqNo) -> Self {
        s.0
    }
}

impl fmt::Display for SeqNo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A half-open sequence range `[lo, hi)`.
///
/// Construction enforces `lo <= hi`. `hi == lo` represents an empty
/// range (syntactically allowed, not currently produced by any code path).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SeqRange {
    pub(crate) lo: SeqNo,
    pub(crate) hi: SeqNo,
}

impl SeqRange {
    pub fn new(lo: SeqNo, hi: SeqNo) -> Result<Self, LsmError> {
        if lo > hi {
            return Err(LsmError::Format(format!("SeqRange: lo ({lo}) > hi ({hi})")));
        }
        Ok(Self { lo, hi })
    }

    /// Lower bound of the range (inclusive).
    pub fn lo(self) -> SeqNo {
        self.lo
    }

    /// Upper bound of the range (exclusive).
    pub fn hi(self) -> SeqNo {
        self.hi
    }
}

/// An LSM level index. Construction enforces `< NUM_LEVELS`.
///
/// The bounds check lives in exactly one place (`Level::new`); all
/// other code sites trust the invariant once they hold a `Level`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Level(u8);

impl Level {
    pub const L0: Level = Level(0);
    pub const L1: Level = Level(1);
    pub const L2: Level = Level(2);
    pub const L3: Level = Level(3);

    pub fn new(level: u8) -> Option<Self> {
        if (level as usize) < NUM_LEVELS {
            Some(Self(level))
        } else {
            None
        }
    }

    pub fn as_u8(self) -> u8 {
        self.0
    }

    pub fn as_index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seqno_display_matches_inner_u64() {
        assert_eq!(SeqNo(42).to_string(), "42");
    }

    #[test]
    fn seqno_ordering_matches_u64() {
        assert!(SeqNo(1) < SeqNo(2));
        assert_eq!(SeqNo(5), SeqNo(5));
    }

    #[test]
    fn seqrange_rejects_lo_greater_than_hi() {
        let result = SeqRange::new(SeqNo(10), SeqNo(5));
        assert!(matches!(result, Err(LsmError::Format(_))));
    }

    #[test]
    fn seqrange_accepts_lo_equal_to_hi() {
        let r = SeqRange::new(SeqNo(5), SeqNo(5)).unwrap();
        assert_eq!(r.lo, SeqNo(5));
        assert_eq!(r.hi, SeqNo(5));
    }

    #[test]
    fn seqrange_accepts_lo_less_than_hi() {
        let r = SeqRange::new(SeqNo(0), SeqNo(10)).unwrap();
        assert_eq!(r.lo, SeqNo(0));
        assert_eq!(r.hi, SeqNo(10));
    }

    #[test]
    fn level_rejects_values_at_or_above_num_levels() {
        assert!(Level::new(NUM_LEVELS as u8).is_none());
        assert!(Level::new(255).is_none());
    }

    #[test]
    fn level_accepts_values_below_num_levels() {
        for i in 0..NUM_LEVELS {
            assert!(Level::new(i as u8).is_some());
        }
    }

    #[test]
    fn level_consts_match_new() {
        assert_eq!(Level::L0, Level::new(0).unwrap());
        assert_eq!(Level::L1, Level::new(1).unwrap());
        assert_eq!(Level::L2, Level::new(2).unwrap());
        assert_eq!(Level::L3, Level::new(3).unwrap());
    }

    #[test]
    fn level_as_u8_and_as_index_agree() {
        let l = Level::L2;
        assert_eq!(l.as_u8(), 2);
        assert_eq!(l.as_index(), 2);
    }
}
