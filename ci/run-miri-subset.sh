#!/usr/bin/env bash
# Run Miri on the full test suite via cargo-nextest.
# Usage: ci/run-miri-subset.sh
#
# Exclusions (defined in .config/nextest.toml [profile.default-miri]):
#   - par_for_each: rayon thread pool unsupported by Miri (covered by TSan)
#   - concurrent/contention pool tests: too slow under Miri (covered by TSan + Loom)

set -euo pipefail

export MIRIFLAGS="${MIRIFLAGS:--Zmiri-tree-borrows}"

echo "=== Miri full suite (nextest, parallel) ==="

# --no-fail-fast: run all tests even if some fail, for full diagnostics.
cargo +nightly miri nextest run -p minkowski --lib --no-fail-fast
