//! Observability companion for Minkowski ECS.
//!
//! Pure consumer crate: captures read-only stats from `World` and `Wal`,
//! diffs consecutive snapshots, and computes deltas. No changes to engine
//! semantics.

pub mod diff;
pub mod prometheus;
pub mod snapshot;

pub use diff::{ArchetypeSize, MetricsDiff};
pub use prometheus::PrometheusExporter;
pub use snapshot::{ArchetypeInfo, MetricsSnapshot};
