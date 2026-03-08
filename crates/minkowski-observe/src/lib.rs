//! Observability companion for Minkowski ECS.
//!
//! Pure consumer crate: captures read-only stats from `World` and `Wal`,
//! diffs consecutive snapshots, and computes rates. No changes to engine
//! semantics.

pub mod diff;
pub mod snapshot;

pub use diff::MetricsDiff;
pub use snapshot::{ArchetypeInfo, MetricsSnapshot};
