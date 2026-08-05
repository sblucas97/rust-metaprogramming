#!/usr/bin/env bash
# Runs the julia benchmark, Rust DSL vs hand-written CUDA, 30x at each of the
# reference sizes (7168/9216/11264, matching raytracer), then prints the
# mean/stdev/min/max/median per (impl, size) plus the Rust-vs-CUDA overhead
# table.
#
# Usage: ./run_julia.sh
# (safe to run from anywhere -- paths are resolved relative to this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"

DEMO_BIN="$REPO_ROOT/target/debug/demo"
CUDA_BIN="$EXAMPLES_DIR/bin/julia"
CSV="$SCRIPT_DIR/julia_runs.csv"
SUMMARY="$SCRIPT_DIR/julia_summary.txt"
RUNS=30
SIZES=(7168 9216 11264)

echo "Building Rust demo binary..."
(cd "$EXAMPLES_DIR" && cargo build)

echo "Building CUDA julia binary..."
mkdir -p "$EXAMPLES_DIR/bin"
nvcc -lm "$EXAMPLES_DIR/src/benchmarks/cuda/julia.cu" -o "$CUDA_BIN"

rm -f "$CSV"

for n in "${SIZES[@]}"; do
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k julia -i rust -s "$n" -n "$RUNS" -- "$DEMO_BIN" julia "$n"
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k julia -i cuda -s "$n" -n "$RUNS" -- "$CUDA_BIN" "$n"
done

echo
echo "=== Summary ($CSV) ==="
"$EXAMPLES_DIR/scripts/analyze_runs.py" "$CSV" -o "$SUMMARY"
