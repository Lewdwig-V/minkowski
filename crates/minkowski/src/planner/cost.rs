use crate::index::SpatialCost;

/// Cost estimate for a plan node. All values are dimensionless relative units
/// tuned for in-memory access patterns (L1/L2 cache locality assumed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    /// Estimated number of rows this node will produce.
    pub(super) rows: f64,
    /// Estimated CPU cost (comparison/hash operations).
    pub(super) cpu: f64,
}

impl Cost {
    /// Estimated number of rows this node will produce.
    #[inline]
    pub fn rows(&self) -> f64 {
        self.rows
    }

    /// Estimated CPU cost (comparison/hash operations).
    #[inline]
    pub fn cpu(&self) -> f64 {
        self.cpu
    }
}

impl Cost {
    const SCAN_PER_ROW: f64 = 1.0;
    const INDEX_LOOKUP: f64 = 5.0; // BTree/Hash lookup base cost
    const HASH_BUILD_PER_ROW: f64 = 2.0;
    const HASH_PROBE_PER_ROW: f64 = 1.5;
    const NESTED_LOOP_PER_PAIR: f64 = 1.0;
    const FILTER_PER_ROW: f64 = 0.5;

    pub(super) fn scan(rows: usize) -> Self {
        let r = rows as f64;
        Cost {
            rows: r,
            cpu: r * Self::SCAN_PER_ROW,
        }
    }

    pub(super) fn index_lookup(selectivity: f64, total_rows: usize) -> Self {
        let r = (total_rows as f64 * selectivity).max(1.0);
        Cost {
            rows: r,
            cpu: Self::INDEX_LOOKUP + r * Self::SCAN_PER_ROW,
        }
    }

    pub(super) fn spatial_lookup(spatial_cost: &SpatialCost) -> Self {
        let rows = if spatial_cost.estimated_rows.is_finite() && spatial_cost.estimated_rows >= 0.0
        {
            spatial_cost.estimated_rows
        } else {
            debug_assert!(
                false,
                "SpatialCost::estimated_rows must be finite and non-negative, got {}",
                spatial_cost.estimated_rows
            );
            1.0
        };
        let cpu = if spatial_cost.cpu.is_finite() && spatial_cost.cpu >= 0.0 {
            spatial_cost.cpu
        } else {
            debug_assert!(
                false,
                "SpatialCost::cpu must be finite and non-negative, got {}",
                spatial_cost.cpu
            );
            10.0
        };
        Cost { rows, cpu }
    }

    pub(super) fn filter(input: Cost, selectivity: f64) -> Self {
        Self::filter_with_branchless(input, selectivity, false)
    }

    pub(super) fn filter_with_branchless(input: Cost, selectivity: f64, branchless: bool) -> Self {
        let speedup = if branchless { 0.5 } else { 0.85 };
        Cost {
            rows: (input.rows * selectivity).max(0.0),
            cpu: input.cpu + input.rows * Self::FILTER_PER_ROW * speedup,
        }
    }

    pub(super) fn hash_join(left: Cost, right: Cost) -> Self {
        Cost {
            rows: (left.rows * right.rows).sqrt().max(1.0), // conservative estimate
            cpu: left.cpu
                + right.cpu
                + left.rows * Self::HASH_BUILD_PER_ROW
                + right.rows * Self::HASH_PROBE_PER_ROW,
        }
    }

    pub(super) fn nested_loop_join(left: Cost, right: Cost) -> Self {
        Cost {
            rows: (left.rows * right.rows).sqrt().max(1.0),
            cpu: left.cpu + right.cpu + left.rows * right.rows * Self::NESTED_LOOP_PER_PAIR,
        }
    }

    /// Total cost for comparison (cpu-dominated since everything is in memory).
    #[inline]
    pub fn total(&self) -> f64 {
        self.cpu
    }
}

// ── Constraint-based optimization utilities ──────────────────────────

/// Constraint on the expected cardinality of a query result.
///
/// Used for constraint-based optimization: the planner can verify that a
/// plan satisfies cardinality bounds and choose strategies accordingly.
#[derive(Clone, Copy, Debug)]
pub enum CardinalityConstraint {
    /// Expect exactly one result (unique lookup).
    ExactlyOne,
    /// Expect at most N results.
    AtMost(usize),
    /// Expect at least N results.
    AtLeast(usize),
    /// Expect between lo and hi results (inclusive).
    ///
    /// Use [`CardinalityConstraint::between`] for validated construction.
    Between(usize, usize),
}

impl CardinalityConstraint {
    /// Create a `Between` constraint with validation that `lo <= hi`.
    ///
    /// # Panics
    /// Panics if `lo > hi`.
    pub fn between(lo: usize, hi: usize) -> Self {
        assert!(
            lo <= hi,
            "CardinalityConstraint::between: lo ({lo}) > hi ({hi})"
        );
        Self::Between(lo, hi)
    }

    /// Check whether an estimated row count satisfies this constraint.
    pub fn satisfied_by(&self, estimated_rows: f64) -> bool {
        match self {
            CardinalityConstraint::ExactlyOne => (0.5..1.5).contains(&estimated_rows),
            CardinalityConstraint::AtMost(n) => estimated_rows <= *n as f64 + 0.5,
            CardinalityConstraint::AtLeast(n) => estimated_rows >= *n as f64 - 0.5,
            CardinalityConstraint::Between(lo, hi) => {
                estimated_rows >= *lo as f64 - 0.5 && estimated_rows <= *hi as f64 + 0.5
            }
        }
    }
}

/// Options for plan cost estimation (internal).
#[derive(Clone, Copy, Debug)]
pub(crate) struct VectorizeOpts {
    /// L2 cache size in bytes. Used to partition hash join build tables
    /// so each partition fits in cache. Default: 256 KiB.
    pub(crate) l2_cache_bytes: usize,

    /// Average component size in bytes. Used to estimate how many rows
    /// fit in a cache line / partition. Default: 16 bytes.
    pub(crate) avg_component_bytes: usize,

    /// Target archetype chunk size. Scans that produce chunks larger than
    /// this will note it in the plan for monitoring. Default: 4096 rows.
    pub(crate) target_chunk_rows: usize,
}

impl Default for VectorizeOpts {
    fn default() -> Self {
        VectorizeOpts {
            l2_cache_bytes: 256 * 1024, // 256 KiB
            avg_component_bytes: 16,    // typical f32x4 or (f32, f32, f32, f32)
            target_chunk_rows: 4096,
        }
    }
}
