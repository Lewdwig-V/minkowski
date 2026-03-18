//! Query engine — bitset matching, incremental caching, parallel and chunk-based iteration.
//!
//! - [`fetch`] — `WorldQuery` trait, fetch types, `Changed<T>` filter
//! - [`iter`] — query iterators (`QueryIter`, `for_each_chunk`, `par_for_each`)
//! - `planner` — query planner for composing index lookups, joins, and scans

pub mod fetch;
pub mod iter;
pub mod planner;
