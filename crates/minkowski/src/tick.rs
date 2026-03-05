/// Monotonic event counter for change detection.
///
/// Each BlobVec column stores the tick at which it was last mutably accessed.
/// Queries compare column ticks against their last-read tick to skip unchanged
/// archetypes. The tick is an implementation detail of the storage engine —
/// it is not a frame counter, simulation clock, or user-facing concept.
///
/// u64 gives ~584,000 years at 1M events/second. No wrapping needed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub(crate) struct Tick(u64);

impl Tick {
    #[allow(dead_code)]
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    #[allow(dead_code)]
    pub(crate) fn raw(self) -> u64 {
        self.0
    }

    /// Returns true if `self` is strictly more recent than `other`.
    pub fn is_newer_than(self, other: Tick) -> bool {
        self.0 > other.0
    }

    /// Advance the tick by one. Returns the new tick value.
    pub fn advance(&mut self) -> Tick {
        self.0 += 1;
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_zero() {
        assert_eq!(Tick::default(), Tick::new(0));
    }

    #[test]
    fn advance_increments() {
        let mut t = Tick::new(0);
        let new = t.advance();
        assert_eq!(new, Tick::new(1));
        assert_eq!(t, Tick::new(1));
    }

    #[test]
    fn newer_than_basic() {
        let a = Tick::new(5);
        let b = Tick::new(3);
        assert!(a.is_newer_than(b));
        assert!(!b.is_newer_than(a));
    }

    #[test]
    fn newer_than_equal_is_false() {
        let a = Tick::new(5);
        assert!(!a.is_newer_than(a));
    }

    #[test]
    fn ordering() {
        let a = Tick::new(1);
        let b = Tick::new(2);
        let c = Tick::new(2);
        assert!(a < b);
        assert_eq!(b, c);
        assert!(b > a);
    }
}
