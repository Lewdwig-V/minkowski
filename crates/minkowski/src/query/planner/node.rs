use std::fmt;

use super::cost::Cost;
use super::predicate::IndexKind;

/// Join strategy selected by the optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinKind {
    /// Standard inner join — only entities present in both sides.
    Inner,
    /// Left join — all entities from the left, matched from the right.
    Left,
}

/// A node in the query execution plan tree.
///
/// All plans execute via chunked, slice-based iteration over 64-byte-aligned
/// columns. LLVM auto-vectorizes loops over these contiguous slices — there
/// is no separate "scalar" execution path. Cost estimates reflect the
/// vectorized execution model (batch amortization, branchless filters,
/// cache-partitioned joins).
#[derive(Debug)]
#[non_exhaustive]
pub enum PlanNode {
    /// Chunked archetype scan: yields one batch per archetype.
    /// Each batch is a contiguous column slice (64-byte aligned).
    Scan {
        query_name: &'static str,
        estimated_rows: usize,
        /// Average rows per chunk (= archetype size). Tuned for L1 residency.
        avg_chunk_size: usize,
        cost: Cost,
    },
    /// Index-driven gather: lookup entities via index, then batch-fetch
    /// components. Entities are sorted by archetype to maximize sequential
    /// access.
    IndexLookup {
        index_kind: IndexKind,
        component_name: &'static str,
        predicate: String,
        estimated_rows: usize,
        cost: Cost,
    },
    /// Spatial index gather: lookup entities via a spatial index.
    SpatialLookup {
        component_name: &'static str,
        predicate: String,
        estimated_rows: usize,
        cost: Cost,
    },
    /// Filter applied to contiguous slices from the child node.
    /// Branchless filters (Eq/Range on numeric types) are ~2x faster
    /// than branched filters (Custom predicates) on aligned data.
    Filter {
        child: Box<PlanNode>,
        predicate: String,
        selectivity: f64,
        /// Whether this filter can be applied branchlessly on aligned data.
        branchless: bool,
        cost: Cost,
    },
    /// Partitioned hash join: build side is partitioned into L2-cache-sized
    /// segments. Probe side streams through chunks, probing the partition
    /// that fits in cache.
    HashJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_kind: JoinKind,
        /// Number of partitions (tuned for L2 cache residency).
        partitions: usize,
        cost: Cost,
    },
    /// Batch nested-loop join on small materialized sides.
    NestedLoopJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_kind: JoinKind,
        cost: Cost,
    },
    /// ER (Entity-Relationship) join: follow entity references from the
    /// left side to probe a hash set built from right-side entities.
    ///
    /// For each left entity, the join reads component `R: AsEntityRef`,
    /// extracts the referenced `Entity`, and checks membership in the
    /// right set. Inner joins emit only left entities whose reference
    /// target is in the right set; left joins emit all left entities.
    ErJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_kind: JoinKind,
        /// Name of the entity-reference component (e.g. `"Parent"`).
        ref_component: &'static str,
        cost: Cost,
    },
    /// Stream aggregate: compute aggregate functions in a single pass
    /// over child output.
    Aggregate {
        child: Box<PlanNode>,
        /// Human-readable labels for each aggregate (e.g. `["COUNT(*)", "SUM(Score)"]`).
        aggregates: Vec<String>,
        cost: Cost,
    },
}

impl PlanNode {
    /// Cost of this node (including children).
    #[inline]
    pub fn cost(&self) -> Cost {
        match self {
            PlanNode::Scan { cost, .. }
            | PlanNode::IndexLookup { cost, .. }
            | PlanNode::SpatialLookup { cost, .. }
            | PlanNode::Filter { cost, .. }
            | PlanNode::HashJoin { cost, .. }
            | PlanNode::NestedLoopJoin { cost, .. }
            | PlanNode::ErJoin { cost, .. }
            | PlanNode::Aggregate { cost, .. } => *cost,
        }
    }

    /// Estimated output row count.
    #[inline]
    pub fn estimated_rows(&self) -> f64 {
        self.cost().rows
    }

    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        // Write indentation without heap allocation: two spaces per level.
        for _ in 0..indent {
            f.write_str("  ")?;
        }
        match self {
            PlanNode::Scan {
                query_name,
                estimated_rows,
                avg_chunk_size,
                cost,
            } => {
                writeln!(
                    f,
                    "ChunkedScan [{query_name}] rows={estimated_rows} \
                     chunk_size={avg_chunk_size} cpu={:.1}",
                    cost.cpu
                )
            }
            PlanNode::IndexLookup {
                index_kind,
                component_name,
                predicate,
                estimated_rows,
                cost,
            } => {
                writeln!(
                    f,
                    "IndexGather [{index_kind:?} on {component_name}] \
                     {predicate} rows={estimated_rows} cpu={:.1}",
                    cost.cpu
                )
            }
            PlanNode::SpatialLookup {
                component_name,
                predicate,
                estimated_rows,
                cost,
            } => {
                writeln!(
                    f,
                    "SpatialGather [Spatial on {component_name}] \
                     {predicate} rows={estimated_rows} cpu={:.1}",
                    cost.cpu
                )
            }
            PlanNode::Filter {
                child,
                predicate,
                selectivity,
                branchless,
                cost,
            } => {
                let mode = if *branchless {
                    "branchless"
                } else {
                    "branched"
                };
                writeln!(
                    f,
                    "Filter [{predicate}] sel={selectivity:.2} \
                     mode={mode} rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                child.fmt_indent(f, indent + 1)
            }
            PlanNode::HashJoin {
                left,
                right,
                join_kind,
                partitions,
                cost,
            } => {
                writeln!(
                    f,
                    "PartitionedHashJoin [{join_kind:?}] partitions={partitions} \
                     rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                left.fmt_indent(f, indent + 1)?;
                right.fmt_indent(f, indent + 1)
            }
            PlanNode::NestedLoopJoin {
                left,
                right,
                join_kind,
                cost,
            } => {
                writeln!(
                    f,
                    "BatchNestedLoopJoin [{join_kind:?}] rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                left.fmt_indent(f, indent + 1)?;
                right.fmt_indent(f, indent + 1)
            }
            PlanNode::ErJoin {
                left,
                right,
                join_kind,
                ref_component,
                cost,
            } => {
                writeln!(
                    f,
                    "ErJoin [{join_kind:?} on {ref_component}] \
                     rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                left.fmt_indent(f, indent + 1)?;
                right.fmt_indent(f, indent + 1)
            }
            PlanNode::Aggregate {
                child,
                aggregates,
                cost,
            } => {
                let agg_list = aggregates.join(", ");
                writeln!(
                    f,
                    "StreamAggregate [{agg_list}] rows={:.0} cpu={:.1}",
                    cost.rows, cost.cpu
                )?;
                child.fmt_indent(f, indent + 1)
            }
        }
    }
}

impl fmt::Display for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}

// ── Warnings ─────────────────────────────────────────────────────────

/// Diagnostic produced during plan compilation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanWarning {
    /// A predicate exists for this component but no matching index was registered.
    MissingIndex {
        component_name: &'static str,
        predicate_kind: &'static str,
        suggestion: &'static str,
    },
    /// A hash index was registered but the predicate needs range support.
    IndexKindMismatch {
        component_name: &'static str,
        have: &'static str,
        need: &'static str,
    },
    /// A spatial index is registered but declined the expression
    /// ([`SpatialIndex::supports`] returned `None`). The predicate falls
    /// back to a scan + post-filter.
    SpatialIndexDeclined {
        component_name: &'static str,
        expression: String,
    },
    /// Join has no index on either side — will use nested loop.
    UnindexedJoin {
        left_name: &'static str,
        right_name: &'static str,
    },
    /// Multiple aggregate expressions share the same label. `get_by_label`
    /// returns the first match — the duplicate is silently hidden.
    DuplicateAggregateLabel { label: String },
    /// An inner join was eliminated at build time — its required components
    /// were merged into the scan's bitset. The plan executes as a pure
    /// archetype scan instead of materializing a join.
    JoinEliminated { right_name: &'static str },
}

impl fmt::Display for PlanWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanWarning::MissingIndex {
                component_name,
                predicate_kind,
                suggestion,
            } => {
                write!(
                    f,
                    "no index for {predicate_kind} predicate on `{component_name}` — \
                     full scan required. Suggestion: {suggestion}"
                )
            }
            PlanWarning::IndexKindMismatch {
                component_name,
                have,
                need,
            } => {
                write!(
                    f,
                    "`{component_name}` has {have} index but predicate needs {need} — \
                     falling back to scan + filter"
                )
            }
            PlanWarning::SpatialIndexDeclined {
                component_name,
                expression,
            } => {
                write!(
                    f,
                    "spatial index for `{component_name}` does not support {expression} — \
                     falling back to scan + filter"
                )
            }
            PlanWarning::UnindexedJoin {
                left_name,
                right_name,
            } => {
                write!(
                    f,
                    "join between `{left_name}` and `{right_name}` has no index on either side — \
                     using nested loop join"
                )
            }
            PlanWarning::DuplicateAggregateLabel { label } => {
                write!(
                    f,
                    "duplicate aggregate label `{label}` — get_by_label() returns the first match"
                )
            }
            PlanWarning::JoinEliminated { right_name } => {
                write!(
                    f,
                    "inner join with `{right_name}` eliminated — merged into scan \
                     (use scan::<(Left, Right)>() directly to avoid this)"
                )
            }
        }
    }
}
