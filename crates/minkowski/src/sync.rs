//! Conditional sync primitives — routes to parking_lot (`Mutex`) and std
//! (`Arc`, atomics, `yield_now`) in production, loom equivalents under
//! `cfg(loom)` for deterministic schedule testing. The loom `Mutex` wrapper
//! converts `Result`-returning `lock()` to match parking_lot's infallible API.

#[cfg(not(loom))]
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
