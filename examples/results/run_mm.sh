#!/usr/bin/env bash
# Runs the mm (matrix multiply) benchmark, Rust DSL vs hand-written CUDA, 30x
# at each of the reference sizes (5000/7000/9000 -- not the GPU-capacity
# tiers in ../../BENCHMARK_GPU_CAPACITY.md, which are O(dim^3) and take tens
# of minutes for a 30x sweep; these are sized for a fast overhead-comparison
# run instead), then prints the mean/stdev/min/max/median per (impl, size)
# plus the Rust-vs-CUDA overhead table.
#
# Usage: ./run_mm.sh
# (safe to run from anywhere -- paths are resolved relative to this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"

DEMO_BIN="$REPO_ROOT/target/debug/demo"
CUDA_BIN="$EXAMPLES_DIR/bin/mm"
CSV="$SCRIPT_DIR/mm_runs.csv"
SUMMARY="$SCRIPT_DIR/mm_summary.txt"
RUNS=30
SIZES=(5000 7000 9000)

echo "Building Rust demo binary..."
(cd "$EXAMPLES_DIR" && cargo build)

echo "Building CUDA mm binary..."
mkdir -p "$EXAMPLES_DIR/bin"
nvcc -lm "$EXAMPLES_DIR/src/benchmarks/cuda/mm.cu" -o "$CUDA_BIN"

rm -f "$CSV"

for n in "${SIZES[@]}"; do
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k mm -i rust -s "$n" -n "$RUNS" -- "$DEMO_BIN" mm "$n"
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k mm -i cuda -s "$n" -n "$RUNS" -- "$CUDA_BIN" "$n"
done

echo
echo "=== Summary ($CSV) ==="
"$EXAMPLES_DIR/scripts/analyze_runs.py" "$CSV" -o "$SUMMARY"
