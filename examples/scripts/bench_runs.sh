#!/usr/bin/env bash
# Run one benchmark command N times and append each "[kernel] elapsed: X ms"
# line it prints to a CSV, one row per run. Works for both the Rust DSL demo
# binary and the hand-written CUDA binaries -- both print that same format,
# so this script doesn't know or care what kernel it's running.
#
# Usage:
#   bench_runs.sh -o <csv_file> -k <kernel> -i rust|cuda -s <size> [-n <runs>] -- <command> [args...]
#
#   -o   CSV file to append to (created with a header if it doesn't exist)
#   -k   kernel name, e.g. nbodies / nearest_neighbor (just a label, doesn't have
#        to match the binary's own "[name]" tag, though it usually will)
#   -i   which implementation this run is: "rust" (the DSL, via demo) or "cuda"
#        (the hand-written .cu binary) -- this is the column you filter/group
#        on to compare the two
#   -s   problem size N you're passing to the command (recorded for grouping,
#        not used to build the command -- put it in the command yourself)
#   -n   number of repetitions (default 30)
#
# Examples:
#   bench_runs.sh -o runs.csv -k nbodies -i rust -s 1000000 -n 30 -- \
#       ../target/debug/demo nbodies 1000000
#   bench_runs.sh -o runs.csv -k nbodies -i cuda -s 1000000 -n 30 -- \
#       /tmp/bench_nbodies 1000000

set -euo pipefail

usage() {
    echo "Usage: $0 -o <csv_file> -k <kernel> -i rust|cuda -s <size> [-n <runs>] -- <command> [args...]" >&2
    exit 1
}

csv_file=""
kernel=""
impl=""
size=""
runs=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o) csv_file="$2"; shift 2 ;;
        -k) kernel="$2"; shift 2 ;;
        -i) impl="$2"; shift 2 ;;
        -s) size="$2"; shift 2 ;;
        -n) runs="$2"; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done

[[ -z "$csv_file" || -z "$kernel" || -z "$size" || $# -eq 0 ]] && usage
if [[ "$impl" != "rust" && "$impl" != "cuda" ]]; then
    echo "Error: -i must be 'rust' or 'cuda' (got '${impl}')" >&2
    usage
fi

if [[ ! -f "$csv_file" ]]; then
    echo "timestamp,kernel,impl,size,run_index,elapsed_ms,command" > "$csv_file"
fi

echo "Running '$*' x${runs} -> kernel=${kernel} impl=${impl} size=${size}, appending to ${csv_file}"
for ((i = 1; i <= runs; i++)); do
    output="$("$@" 2>&1)" || { echo "run ${i} FAILED:"; echo "$output"; exit 1; }
    matched=0
    while IFS= read -r line; do
        if [[ "$line" =~ \[([a-zA-Z0-9_]+)\]\ elapsed:\ ([0-9.]+)\ ms ]]; then
            ms="${BASH_REMATCH[2]}"
            ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "${ts},${kernel},${impl},${size},${i},${ms},\"$*\"" >> "$csv_file"
            matched=1
        fi
    done <<< "$output"
    if [[ "$matched" -eq 0 ]]; then
        echo "run ${i}: no '[kernel] elapsed: X ms' line found in output:" >&2
        echo "$output" >&2
        exit 1
    fi
    printf "  [%d/%d] done\n" "$i" "$runs"
done

echo "Done: ${runs} runs appended to ${csv_file}"
