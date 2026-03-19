#!/usr/bin/env bash
# Run Miri on the curated unsafe-code test subset via cargo-nextest.
# Usage: ci/run-miri-subset.sh
#
# Uses the [profile.default-miri] filterset in .config/nextest.toml to select
# tests. Nextest runs each test in its own process, enabling parallel Miri
# execution (Miri itself is single-threaded).

set -euo pipefail

export MIRIFLAGS="${MIRIFLAGS:--Zmiri-tree-borrows}"

echo "=== Miri subset (nextest, parallel) ==="

# --no-fail-fast: run all tests even if some fail, for full diagnostics.
cargo +nightly miri nextest run -p minkowski --lib --no-fail-fast
