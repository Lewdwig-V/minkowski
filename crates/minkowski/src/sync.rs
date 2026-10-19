//! Conditional sync primitives — routes to parking_lot (`Mutex`) and std
//! (`Arc`, atomics, `yield_now`) in production, loom equivalents under
//! `cfg(loom)` for deterministic schedule testing. The loom `Mutex` wrapper
//! converts `Result`-returning `lock()` to match parking_lot's infallible API.
//! Under Miri, `Mutex` is a `std::sync::Mutex` wrapper with the same API:
//! Miri rejects the raw `syscall(SYS_futex, ...)` that `parking_lot_core`
//! issues when a thread parks on a contended lock (#300), while std's futex
//! use is modelled natively.

#[cfg(all(not(loom), not(miri)))]
pub(crate) use parking_lot::Mutex;

#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

#[cfg(not(loom))]
pub(crate) use std::sync::Arc;

#[cfg(loom)]
pub(crate) use loom::sync::Arc;

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

// loom::sync::Mutex::lock() returns Result — wrap to match parking_lot's
// infallible API so call sites don't change.
#[cfg(loom)]
mod loom_mutex {
    pub(crate) struct Mutex<T>(loom::sync::Mutex<T>);

    impl<T> Mutex<T> {
        #[inline]
        pub fn new(val: T) -> Self {
            Self(loom::sync::Mutex::new(val))
        }

        #[inline]
        pub fn lock(&self) -> loom::sync::MutexGuard<'_, T> {
            self.0.lock().unwrap()
        }
    }
}

#[cfg(loom)]
pub(crate) use loom_mutex::Mutex;

// std::sync::Mutex::lock() returns Result (poisoning) — wrap to match
// parking_lot's infallible, non-poisoning API so call sites don't change.
#[cfg(all(miri, not(loom)))]
mod miri_mutex {
    pub(crate) struct Mutex<T>(std::sync::Mutex<T>);

    impl<T> Mutex<T> {
        #[inline]
        pub fn new(val: T) -> Self {
            Self(std::sync::Mutex::new(val))
        }

        #[inline]
        pub fn lock(&self) -> std::sync::MutexGuard<'_, T> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }
    }
}

#[cfg(all(miri, not(loom)))]
pub(crate) use miri_mutex::Mutex;

// Thread operations: yield_now must route through loom for schedule control.
#[cfg(not(loom))]
#[inline]
pub(crate) fn yield_now() {
    std::thread::yield_now();
}

#[cfg(loom)]
#[inline]
pub(crate) fn yield_now() {
    loom::thread::yield_now();
}
