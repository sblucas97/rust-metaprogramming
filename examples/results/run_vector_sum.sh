#!/usr/bin/env bash
# Runs the vector_sum benchmark, Rust DSL vs hand-written CUDA, 30x at each of
# the GPU-capacity tiers (100% / 85% / 72.25% of usable device memory) from
# ../../BENCHMARK_GPU_CAPACITY.md, then prints the mean/stdev/min/max/median
# per (impl, size) plus the Rust-vs-CUDA overhead table.
#
# Usage: ./run_vector_sum.sh
# (safe to run from anywhere -- paths are resolved relative to this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"

DEMO_BIN="$REPO_ROOT/target/debug/demo"
CUDA_BIN="$EXAMPLES_DIR/bin/vector_sum"
CSV="$SCRIPT_DIR/vector_sum_runs.csv"
SUMMARY="$SCRIPT_DIR/vector_sum_summary.txt"
RUNS=30
SIZES=(551121715 468453457 398185439)

echo "Building Rust demo binary..."
(cd "$EXAMPLES_DIR" && cargo build)

echo "Building CUDA vector_sum binary..."
mkdir -p "$EXAMPLES_DIR/bin"
nvcc -lm "$EXAMPLES_DIR/src/benchmarks/cuda/vector_sum.cu" -o "$CUDA_BIN"

rm -f "$CSV"

for n in "${SIZES[@]}"; do
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k vector_sum -i rust -s "$n" -n "$RUNS" -- "$DEMO_BIN" vector_sum "$n"
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k vector_sum -i cuda -s "$n" -n "$RUNS" -- "$CUDA_BIN" "$n"
done

echo
echo "=== Summary ($CSV) ==="
"$EXAMPLES_DIR/scripts/analyze_runs.py" "$CSV" -o "$SUMMARY"
