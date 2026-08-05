#!/usr/bin/env bash
# Runs the ripple benchmark, Rust DSL vs hand-written CUDA, 30x at each of
# the GPU-capacity tiers (100% / 85% / 72.25% of usable device memory) from
# ../../BENCHMARK_GPU_CAPACITY.md, then prints the mean/stdev/min/max/median
# per (impl, size) plus the Rust-vs-CUDA overhead table.
#
# Usage: ./run_ripple.sh
# (safe to run from anywhere -- paths are resolved relative to this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"

DEMO_BIN="$REPO_ROOT/target/debug/demo"
CUDA_BIN="$EXAMPLES_DIR/bin/ripple"
CSV="$SCRIPT_DIR/ripple_runs.csv"
SUMMARY="$SCRIPT_DIR/ripple_summary.txt"
RUNS=30
SIZES=(20330 17280 14688)

echo "Building Rust demo binary..."
(cd "$EXAMPLES_DIR" && cargo build)

echo "Building CUDA ripple binary..."
mkdir -p "$EXAMPLES_DIR/bin"
nvcc -lm "$EXAMPLES_DIR/src/benchmarks/cuda/ripple.cu" -o "$CUDA_BIN"

rm -f "$CSV"

for n in "${SIZES[@]}"; do
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k ripple -i rust -s "$n" -n "$RUNS" -- "$DEMO_BIN" ripple "$n"
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k ripple -i cuda -s "$n" -n "$RUNS" -- "$CUDA_BIN" "$n"
done

echo
echo "=== Summary ($CSV) ==="
"$EXAMPLES_DIR/scripts/analyze_runs.py" "$CSV" -o "$SUMMARY"
