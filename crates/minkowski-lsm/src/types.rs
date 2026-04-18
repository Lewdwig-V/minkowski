//! Newtype primitives for the LSM manifest subsystem.
//!
//! Construction idiom reflects the validity story:
//! - `From<u64>` — total, interchangeable with `u64` (e.g., `SeqNo`).
//! - `new(u64) -> Option<Self>` — fallible with a precondition (e.g., `PageCount::new` rejects zero, `Level::new` rejects out-of-range).
//! - `new(u64) -> Self` without `From<u64>` — infallible but nominally distinct; `From` is deliberately omitted to block silent `.into()` conversions at adjacent-arg call sites (e.g., `SizeBytes`).
//!
//! Invariant-carrying newtype primitives shared across the manifest.

use std::fmt;
use std::num::NonZeroU64;

use crate::error::LsmError;
use crate::manifest::NUM_LEVELS;

/// A WAL sequence number.
///
/// Raw `u64` arithmetic on sequence numbers is disallowed by the type
/// (no `Add`/`Sub` impls) because sequences are identities, not sizes.
/// Use `.next()` for "advance by one"; use `.get()` to extract the raw value.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SeqNo(u64);

impl SeqNo {
    /// Extract the underlying `u64`.
    pub fn get(self) -> u64 {
        self.0
    }

    /// The next sequence number. Panics on `u64::MAX + 1` — an internal
    /// invariant violation, since the WAL is the only `SeqNo` producer
    /// and a 64-bit sequence space exhausts long after any realistic
    /// process lifetime.
    pub fn next(self) -> Self {
        Self(self.0.checked_add(1).expect("SeqNo overflow"))
    }
}

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
/// Construction enforces `lo <= hi`. `hi == lo` represents an empty range.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SeqRange {
    lo: SeqNo,
    hi: SeqNo,
}

impl SeqRange {
    /// Build a half-open range. Returns `Err(LsmError::Format)` if `lo > hi`.
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

    /// Construct a level index. Returns `None` if `level >= NUM_LEVELS`.
    pub fn new(level: u8) -> Option<Self> {
        if (level as usize) < NUM_LEVELS {
            Some(Self(level))
        } else {
            None
        }
    }

    /// The underlying level byte (always `< NUM_LEVELS`).
    pub fn as_u8(self) -> u8 {
        self.0
    }

    /// Convert to a `usize` for indexing `[T; NUM_LEVELS]` arrays.
    pub fn as_index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

/// A page count — guaranteed non-zero at construction.
///
/// Layout-compatible with `u64` via `std::num::NonZeroU64`, so
/// `Option<PageCount>` has the same size as `u64` (niche optimization).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PageCount(NonZeroU64);

impl PageCount {
    /// Returns `None` if `value` is zero.
    pub fn new(value: u64) -> Option<Self> {
        NonZeroU64::new(value).map(Self)
    }

    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for PageCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

/// Size in bytes of an on-disk artifact.
///
/// Infallible newtype — zero is permitted (matches the semantics of
/// `fs::metadata(...).len()`, which returns `0` for empty files).
/// Type-level distinction from `PageCount` prevents arg-swap bugs at
/// any call site taking both.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SizeBytes(u64);

impl SizeBytes {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SizeBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tombstones: sequences are identities, not sizes — arithmetic must
    // stay unimplemented so `seq + 1` or `seq - other` fails to compile.
    // See the SeqNo type doc comment above for the rationale.
    use static_assertions::{assert_eq_size, assert_not_impl_all};
    use std::ops::{Add, AddAssign, Div, Mul, Neg, Rem, Sub, SubAssign};

    assert_not_impl_all!(SeqNo: Add<SeqNo>, Sub<SeqNo>, AddAssign<SeqNo>, SubAssign<SeqNo>);
    assert_not_impl_all!(SeqNo: Add<u64>, Sub<u64>, AddAssign<u64>, SubAssign<u64>);
    assert_not_impl_all!(SeqNo: Mul<u64>, Div<u64>, Rem<u64>, Neg);

    // Pin the PageCount layout claims documented on the type:
    // - PageCount itself is 8 bytes (same as u64).
    // - Option<PageCount> is 8 bytes via NonZeroU64's niche.
    assert_eq_size!(PageCount, u64);
    assert_eq_size!(Option<PageCount>, u64);

    // SizeBytes is a plain u64 wrapper — no niche, so only the direct size holds.
    // Option<SizeBytes> is 16 bytes, not 8.
    assert_eq_size!(SizeBytes, u64);

    #[test]
    fn seqno_get_returns_inner_u64() {
        let s = SeqNo::from(42u64);
        assert_eq!(s.get(), 42);
    }

    #[test]
    fn seqno_next_advances_by_one() {
        let s = SeqNo::from(5u64);
        let n = s.next();
        assert_eq!(n.get(), 6);
    }

    #[test]
    #[should_panic(expected = "SeqNo overflow")]
    fn seqno_next_panics_on_overflow() {
        let s = SeqNo::from(u64::MAX);
        let _ = s.next();
    }

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
        assert_eq!(r.lo(), SeqNo(5));
        assert_eq!(r.hi(), SeqNo(5));
    }

    #[test]
    fn seqrange_accepts_lo_less_than_hi() {
        let r = SeqRange::new(SeqNo(0), SeqNo(10)).unwrap();
        assert_eq!(r.lo(), SeqNo(0));
        assert_eq!(r.hi(), SeqNo(10));
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

    #[test]
    fn pagecount_rejects_zero() {
        assert!(PageCount::new(0).is_none());
    }

    #[test]
    fn pagecount_accepts_one() {
        let pc = PageCount::new(1).unwrap();
        assert_eq!(pc.get(), 1);
    }

    #[test]
    fn pagecount_accepts_large_values() {
        let pc = PageCount::new(u64::MAX).unwrap();
        assert_eq!(pc.get(), u64::MAX);
    }

    #[test]
    fn pagecount_roundtrip() {
        let pc = PageCount::new(42).unwrap();
        let raw: u64 = pc.get();
        let restored = PageCount::new(raw).unwrap();
        assert_eq!(pc, restored);
    }

    #[test]
    fn sizebytes_get_returns_inner_u64() {
        let s = SizeBytes::new(1024);
        assert_eq!(s.get(), 1024);
    }

    #[test]
    fn sizebytes_allows_zero() {
        let s = SizeBytes::new(0);
        assert_eq!(s.get(), 0);
    }

    #[test]
    fn sizebytes_display() {
        assert_eq!(SizeBytes::new(42).to_string(), "42");
    }
}
