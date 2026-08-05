#!/usr/bin/env bash
# Runs the nearest_neighbor benchmark, Rust DSL vs hand-written CUDA, 30x at
# each of the reference sizes (100M/200M/300M -- smaller than the
# GPU-capacity tiers in ../../BENCHMARK_GPU_CAPACITY.md, sized for a fast
# overhead-comparison run), then prints the mean/stdev/min/max/median per
# (impl, size) plus the Rust-vs-CUDA overhead table.
#
# Usage: ./run_nearest_neighbor.sh
# (safe to run from anywhere -- paths are resolved relative to this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"

DEMO_BIN="$REPO_ROOT/target/debug/demo"
CUDA_BIN="$EXAMPLES_DIR/bin/nearest_neighbor"
CSV="$SCRIPT_DIR/nearest_neighbor_runs.csv"
SUMMARY="$SCRIPT_DIR/nearest_neighbor_summary.txt"
RUNS=30
SIZES=(100000000 200000000 300000000)

echo "Building Rust demo binary..."
(cd "$EXAMPLES_DIR" && cargo build)

echo "Building CUDA nearest_neighbor binary..."
mkdir -p "$EXAMPLES_DIR/bin"
nvcc -lm "$EXAMPLES_DIR/src/benchmarks/cuda/nearest_neighbor.cu" -o "$CUDA_BIN"

rm -f "$CSV"

for n in "${SIZES[@]}"; do
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k nearest_neighbor -i rust -s "$n" -n "$RUNS" -- "$DEMO_BIN" nearest_neighbor "$n"
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k nearest_neighbor -i cuda -s "$n" -n "$RUNS" -- "$CUDA_BIN" "$n"
done

echo
echo "=== Summary ($CSV) ==="
"$EXAMPLES_DIR/scripts/analyze_runs.py" "$CSV" -o "$SUMMARY"
