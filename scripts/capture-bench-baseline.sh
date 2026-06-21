#!/usr/bin/env bash
# Capture a reproducible LSM + ECS benchmark baseline for regression comparison.
#
# Usage: scripts/capture-bench-baseline.sh [output_dir]   (default: target/bench-baseline)
#
# iai-callgrind benches (instruction counts — the machine-independent regression
# signal) require valgrind + `cargo install iai-callgrind-runner` (matching the
# `iai-callgrind` dev-dependency version, currently 0.16.1). The repo's
# `.cargo/config.toml` sets `target-cpu=native`, which makes valgrind SIGILL, so the
# iai runs override RUSTFLAGS to a valgrind-safe baseline.
#
# Prefer the iai instruction counts for regression detection; criterion wall-clock
# is host-relative and only comparable on the same machine.
set -euo pipefail

OUT="${1:-target/bench-baseline}"
mkdir -p "$OUT"
SAFE_FLAGS="-C target-cpu=x86-64-v2"

echo "=== iai-callgrind: ECS hot paths (instruction counts) ==="
if command -v valgrind >/dev/null 2>&1; then
  RUSTFLAGS="$SAFE_FLAGS" cargo bench -p minkowski-bench --bench ecs_icount \
    2>&1 | tee "$OUT/ecs_icount.txt"
  echo "=== iai-callgrind: LSM page codec ==="
  RUSTFLAGS="$SAFE_FLAGS" cargo bench -p minkowski-lsm --features bench-support --bench page_codec \
    2>&1 | tee "$OUT/page_codec.txt"
else
  echo "valgrind absent — skipping iai-callgrind benches (install valgrind for instruction counts)" \
    | tee "$OUT/iai-skipped.txt"
fi

echo "=== criterion: LSM throughput / compaction (wall-clock, host-relative) ==="
cargo bench -p minkowski-persist --bench lsm_throughput 2>&1 | tee "$OUT/lsm_throughput.txt"
cargo bench -p minkowski-persist --bench lsm_compaction  2>&1 | tee "$OUT/lsm_compaction.txt" | grep -E "WRITEAMP|time:" || true

echo "=== level sweep (write-amp + recovery vs N) ==="
cargo bench -p minkowski-persist --bench lsm_level_sweep 2>&1 | tee "$OUT/lsm_level_sweep.txt" | grep SWEEP || true

echo "baseline written to $OUT"
