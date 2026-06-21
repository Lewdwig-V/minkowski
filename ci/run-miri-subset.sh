#!/usr/bin/env bash
# Run Miri over the minkowski core crate's lib tests via cargo-nextest.
# Scope is deliberately `-p minkowski --lib` (the unsafe-heavy core: storage,
# pool, world), NOT the whole workspace — higher crates (minkowski-lsm, -persist)
# are covered by their normal test suites + TSan/Loom, and their rkyv/mmap paths
# are impractical under Miri.
# Usage: ci/run-miri-subset.sh
#
# Exclusions (defined in .config/nextest.toml [profile.default-miri]):
#   - par_for_each: rayon thread pool unsupported by Miri (covered by TSan)
#   - concurrent/contention pool tests: too slow under Miri (covered by TSan + Loom)

set -euo pipefail

export MIRIFLAGS="${MIRIFLAGS:--Zmiri-tree-borrows}"

echo "=== Miri: minkowski core lib tests (nextest, parallel) ==="

# --no-fail-fast: run all tests even if some fail, for full diagnostics.
cargo +nightly miri nextest run -p minkowski --lib --no-fail-fast
