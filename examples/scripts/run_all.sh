#!/usr/bin/env bash
# Sweep every kernel in kernels.conf into a single new version directory.
#
# Usage:
#   run_all.sh [-i <impls>] [-n <runs>] [--profile <p>] [--arch <sm_XX>]
#              [--limit-<impl> <n>] [--version <vN>] [--only <k1,k2>] [--dry-run]
#
# All flags are passed through to run_bench.sh, except --only, which restricts
# the sweep to a subset of kernels. The version number is resolved once here so
# every kernel lands in the same batch.
#
# --sizes / --sizes-<impl> are per-kernel by nature, so they're only worth
# passing here alongside --only for a single kernel; --limit-<impl> is per-kernel
# safe, since it just slices whatever sizes that kernel resolved.
#
# Examples:
#   run_all.sh                              # every kernel, all three impls
#   run_all.sh -i rust-gpu,cuda -n 30       # DSL vs CUDA across the board
#   run_all.sh --only julia,raytracer -n 5  # quick subset
#   run_all.sh -i rust,rust-gpu --limit-rust 1   # CPU only on the shared tier

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$EXAMPLES_DIR/results"

# shellcheck source=kernels.conf
source "$SCRIPT_DIR/kernels.conf"

version=""
only=""
passthrough=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) version="$2"; shift 2 ;;
        --only)    only="$2"; shift 2 ;;
        -i|--impls|-n|--runs|-s|--sizes|--order|--profile|--arch|--sizes-*|--limit-*)
                   passthrough+=("$1" "$2"); shift 2 ;;
        --dry-run|--no-build)
                   passthrough+=("$1"); shift ;;
        *)         echo "Error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

selected=("${KERNELS[@]}")
if [[ -n "$only" ]]; then
    IFS=',' read -ra selected <<< "$only"
    for k in "${selected[@]}"; do
        found=0
        for known in "${KERNELS[@]}"; do [[ "$known" == "$k" ]] && found=1; done
        [[ "$found" -eq 1 ]] || { echo "Error: unknown kernel '$k'" >&2; exit 1; }
    done
fi

# Resolve the version once so the whole sweep shares one directory.
if [[ -z "$version" ]]; then
    max=-1
    for dir in "$RESULTS_DIR"/v*; do
        [[ -d "$dir" ]] || continue
        n="${dir##*/v}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        (( n > max )) && max="$n"
    done
    version="v$((max + 1))"
fi

echo "Sweeping ${#selected[@]} kernel(s) into $RESULTS_DIR/$version"
echo

for kernel in "${selected[@]}"; do
    echo "──────────────────────────────────────────────────────────────"
    "$SCRIPT_DIR/run_bench.sh" -k "$kernel" --version "$version" "${passthrough[@]}"
    echo
done

echo "Sweep complete: $RESULTS_DIR/$version"
