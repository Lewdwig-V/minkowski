//! Loom-compatible sync primitives for the replication state machine.
//! Active only under `cfg(loom)` (feature `loom`); production builds never
//! include this module.

#[cfg(loom)]
pub(crate) use loom::sync::Arc;

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// loom::sync::Mutex::lock() returns Result — wrap to match the infallible
// API call sites use.
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
