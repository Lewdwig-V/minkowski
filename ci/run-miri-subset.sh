#!/usr/bin/env bash
# Run Miri on the curated unsafe-code test subset (~85 tests).
# Usage: ci/run-miri-subset.sh
#
# Runs cargo miri test with Tree Borrows on module-level filters
# covering all unsafe-code-exercising test paths.

set -euo pipefail

export MIRIFLAGS="${MIRIFLAGS:--Zmiri-tree-borrows}"

echo "=== Miri subset: ~85 tests targeting unsafe code paths ==="

# Tier 1: ALL tests in unsafe-heavy modules
FILTERS=(
    "storage::blob_vec::tests"       # 16 tests — raw pointer column storage
    "storage::sparse::tests"         # 13 tests — paged indices + dense BlobVec
    "storage::archetype::tests"      # 4 tests — BlobVec composition
    "query::fetch::tests"            # 12 tests — raw pointer to slice cast
)

# Tier 2: Selected tests from large modules (by exact name)
EXACT_TESTS=(
    # Pool: mmap + CAS + side table + UnsafeCell TCache (12)
    "pool::tests::slab_pool_allocate_and_deallocate"
    "pool::tests::slab_pool_returns_error_on_exhaustion"
    "pool::tests::slab_pool_overflow_dealloc_returns_to_correct_class"
    "pool::tests::slab_pool_overflow_telemetry"
    "pool::tests::slab_pool_multi_class_used_bytes_tracking"
    "pool::tests::slab_pool_zero_size_allocation"
    "pool::tests::tcache_refill_and_hit"
    "pool::tests::tcache_spill_on_full"
    "pool::tests::tcache_epoch_flush"
    "pool::tests::tcache_thread_exit_flushes"
    "pool::tests::tcache_overflow_refill_correct_bin"
    "pool::tests::tcache_cross_thread_dealloc"
    # Query iter (6)
    "query::iter::tests::iterate_single_archetype"
    "query::iter::tests::iterate_multiple_archetypes"
    "query::iter::tests::mutate_during_iteration"
    "query::iter::tests::for_each_chunk_yields_correct_data"
    "query::iter::tests::for_each_chunk_mutation"
    "query::iter::tests::par_for_each_mutation"
    # World entity lifecycle (11)
    "world::tests::spawn_and_get"
    "world::tests::get_mut"
    "world::tests::get_mut_marks_column_tick"
    "world::tests::despawn_and_is_alive"
    "world::tests::entity_recycling"
    "world::tests::insert_new_component"
    "world::tests::remove_component"
    "world::tests::remove_last_component_swap_fixup"
    "world::tests::despawn_batch_multiple_archetypes"
    "world::tests::get_batch_mut_basic"
    "world::tests::get_batch_mut_marks_changed"
    # Changeset arena + drop safety (6)
    "changeset::tests::arena_alloc_and_read_back"
    "changeset::tests::arena_alignment"
    "changeset::tests::apply_does_not_double_drop"
    "changeset::tests::drop_runs_destructor_for_unapplied_insert"
    "changeset::tests::insert_raw_and_apply"
    "changeset::tests::insert_raw_drop_on_abort"
    # Table derive macro type erasure (3)
    "table::tests::derive_table_register"
    "table::tests::query_table_yields_correct_data"
    "table::tests::typed_query_table_mut"
    # Bundle unsafe put (2)
    "bundle::tests::put_writes_correct_data"
    "bundle::tests::pair_component_ids_sorted"
)

TOTAL=0
FAILED=0

# Run module-level filters (full modules)
for filter in "${FILTERS[@]}"; do
    echo ""
    echo "--- $filter ---"
    if cargo +nightly miri test -p minkowski --lib -- "$filter" 2>&1; then
        COUNT=$(cargo test -p minkowski --lib -- "$filter" --list 2>/dev/null | grep -c ": test" || echo 0)
        TOTAL=$((TOTAL + COUNT))
    else
        FAILED=$((FAILED + 1))
    fi
done

# Run exact tests one at a time (for selected tests from large modules)
for test in "${EXACT_TESTS[@]}"; do
    echo ""
    echo "--- $test ---"
    if cargo +nightly miri test -p minkowski --lib -- "$test" --exact 2>&1; then
        TOTAL=$((TOTAL + 1))
    else
        echo "FAILED: $test"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Miri subset complete: $TOTAL tests, $FAILED failures ==="
[ "$FAILED" -eq 0 ] || exit 1
