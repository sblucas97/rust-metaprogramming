#!/usr/bin/env bash
# Runs the raytracer benchmark, Rust DSL vs hand-written CUDA, 30x at each of
# the reference sizes (7168/9216/11264 -- not the GPU-capacity tiers in
# ../../BENCHMARK_GPU_CAPACITY.md, sized for a fast overhead-comparison run),
# then prints the mean/stdev/min/max/median per (impl, size) plus the
# Rust-vs-CUDA overhead table.
#
# Usage: ./run_raytracer.sh
# (safe to run from anywhere -- paths are resolved relative to this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"

DEMO_BIN="$REPO_ROOT/target/debug/demo"
CUDA_BIN="$EXAMPLES_DIR/bin/raytracer"
CSV="$SCRIPT_DIR/raytracer_runs.csv"
SUMMARY="$SCRIPT_DIR/raytracer_summary.txt"
RUNS=30
SIZES=(7168 9216 11264)

echo "Building Rust demo binary..."
(cd "$EXAMPLES_DIR" && cargo build)

echo "Building CUDA raytracer binary..."
mkdir -p "$EXAMPLES_DIR/bin"
nvcc -lm "$EXAMPLES_DIR/src/benchmarks/cuda/raytracer.cu" -o "$CUDA_BIN"

rm -f "$CSV"

for n in "${SIZES[@]}"; do
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k raytracer -i rust -s "$n" -n "$RUNS" -- "$DEMO_BIN" raytracer "$n"
    "$EXAMPLES_DIR/scripts/bench_runs.sh" -o "$CSV" -k raytracer -i cuda -s "$n" -n "$RUNS" -- "$CUDA_BIN" "$n"
done

echo
echo "=== Summary ($CSV) ==="
"$EXAMPLES_DIR/scripts/analyze_runs.py" "$CSV" -o "$SUMMARY"
