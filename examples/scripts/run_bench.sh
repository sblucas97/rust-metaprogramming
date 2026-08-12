#!/usr/bin/env bash
# Run one kernel across any combination of implementations and record the
# results into a versioned directory.
#
# Implementations:
#   rust-gpu  the Rust DSL / custom compiler  (demo <kernel> <n>)
#   rust      pure Rust on the CPU, rayon     (demo <kernel>_cpu <n>)
#   cuda      hand-written .cu                (bin/<kernel> <n>)
#
# Usage:
#   run_bench.sh -k <kernel> [options]
#
#   -k, --kernel <name>     kernel to run (see scripts/kernels.conf)
#   -i, --impls <list>      comma-separated, default rust-gpu,rust,cuda
#   -n, --runs <n>          repetitions per (impl, size), default from kernels.conf
#   -s, --sizes "<a b c>"   sizes to sweep, default from kernels.conf
#       --version <vN>      write into results/<vN> (created if absent), or
#                           "latest" to follow the symlink; default: next free vN
#       --profile <p>       release (default) or debug
#       --arch <sm_XX>      nvcc target, default from kernels.conf
#       --no-build          skip cargo/nvcc, use whatever is already built
#       --dry-run           print the resolved plan and exit
#
# Examples:
#   run_bench.sh -k julia                          # all three impls, defaults
#   run_bench.sh -k julia -i rust-gpu,cuda         # DSL vs hand-written CUDA
#   run_bench.sh -k julia -i rust-gpu,rust -n 10   # DSL vs CPU, 10 reps
#   run_bench.sh -k mm -s "512 1024" --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"
RESULTS_DIR="$EXAMPLES_DIR/results"

# shellcheck source=kernels.conf
source "$SCRIPT_DIR/kernels.conf"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 1
}

die() { echo "Error: $*" >&2; exit 1; }

kernel=""
impls="rust-gpu,rust,cuda"
runs=""
sizes=""
version=""
profile="release"
arch="$NVCC_ARCH"
do_build=1
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--kernel)  kernel="$2"; shift 2 ;;
        -i|--impls)   impls="$2"; shift 2 ;;
        -n|--runs)    runs="$2"; shift 2 ;;
        -s|--sizes)   sizes="$2"; shift 2 ;;
        --version)    version="$2"; shift 2 ;;
        --profile)    profile="$2"; shift 2 ;;
        --arch)       arch="$2"; shift 2 ;;
        --no-build)   do_build=0; shift ;;
        --dry-run)    dry_run=1; shift ;;
        -h|--help)    usage ;;
        *)            die "unknown argument '$1'" ;;
    esac
done

[[ -n "$kernel" ]] || usage

# ---------------------------------------------------------------- validation
known=0
for k in "${KERNELS[@]}"; do [[ "$k" == "$kernel" ]] && known=1; done
[[ "$known" -eq 1 ]] || die "unknown kernel '$kernel' (known: ${KERNELS[*]})"

IFS=',' read -ra impl_list <<< "$impls"
[[ ${#impl_list[@]} -gt 0 ]] || die "--impls must name at least one implementation"
has_cpu=0
has_rust=0
has_cuda=0
for impl in "${impl_list[@]}"; do
    case "$impl" in
        rust-gpu) has_rust=1 ;;
        rust)     has_cpu=1; has_rust=1 ;;
        cuda)     has_cuda=1 ;;
        *)        die "unknown impl '$impl' (expected rust-gpu, rust or cuda)" ;;
    esac
done

case "$profile" in
    release|debug) ;;
    *) die "--profile must be 'release' or 'debug' (got '$profile')" ;;
esac
if [[ "$profile" == "debug" && "$has_cpu" -eq 1 ]]; then
    echo "Warning: the 'rust' (CPU) impl in a debug build runs at opt-level 0 and" >&2
    echo "         will be roughly 10-50x slower than it should be." >&2
fi

# ------------------------------------------------------------------ defaults
# A run that includes the CPU impl uses CPU_SIZES for *every* impl, so each size
# yields a comparison row -- analyze_runs.py only compares within a shared size.
if [[ -z "$sizes" ]]; then
    if [[ "$has_cpu" -eq 1 ]]; then
        sizes="${CPU_SIZES[$kernel]:-${GPU_SIZES[$kernel]}}"
    else
        sizes="${GPU_SIZES[$kernel]}"
    fi
fi
[[ -n "$runs" ]] || runs="${RUNS_DEFAULT[$kernel]:-$DEFAULT_RUNS}"

read -ra size_list <<< "$sizes"

# ------------------------------------------------------------------- version
next_version() {
    local max=-1 dir n
    for dir in "$RESULTS_DIR"/v*; do
        [[ -d "$dir" ]] || continue
        n="${dir##*/v}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        (( n > max )) && max="$n"
    done
    echo "v$((max + 1))"
}

if [[ -z "$version" ]]; then
    version="$(next_version)"
elif [[ "$version" == "latest" ]]; then
    [[ -e "$RESULTS_DIR/latest" ]] || die "results/latest does not exist yet"
    version="$(basename "$(readlink -f "$RESULTS_DIR/latest")")"
fi
[[ "$version" =~ ^v[0-9]+$ ]] || die "--version must look like v3 or be 'latest' (got '$version')"

VERSION_DIR="$RESULTS_DIR/$version"
KERNEL_DIR="$VERSION_DIR/$kernel"
CSV="$KERNEL_DIR/runs.csv"

DEMO_BIN="$REPO_ROOT/target/$profile/demo"
CUDA_BIN="$EXAMPLES_DIR/bin/$kernel"
CUDA_SRC="$EXAMPLES_DIR/src/benchmarks/cuda/$kernel.cu"

total=$(( ${#size_list[@]} * runs * ${#impl_list[@]} ))
cat <<EOF
kernel   : $kernel
impls    : ${impl_list[*]}
sizes    : ${size_list[*]}
runs     : $runs per (impl, size)  ->  $total executions
profile  : $profile
output   : $KERNEL_DIR
EOF

if [[ "$dry_run" -eq 1 ]]; then
    echo
    echo "Commands that would run (once per repetition, interleaved):"
    for n in "${size_list[@]}"; do
        for impl in "${impl_list[@]}"; do
            case "$impl" in
                rust-gpu) echo "  [$impl] $DEMO_BIN $kernel $n" ;;
                rust)     echo "  [$impl] $DEMO_BIN ${kernel}_cpu $n" ;;
                cuda)     echo "  [$impl] $CUDA_BIN $n" ;;
            esac
        done
    done
    exit 0
fi

# --------------------------------------------------------------------- build
if [[ "$do_build" -eq 1 ]]; then
    if [[ "$has_rust" -eq 1 ]]; then
        echo "Building demo binary ($profile)..."
        if [[ "$profile" == "release" ]]; then
            (cd "$EXAMPLES_DIR" && RUSTFLAGS="-A warnings" cargo build --release)
        else
            (cd "$EXAMPLES_DIR" && RUSTFLAGS="-A warnings" cargo build)
        fi
    fi
    if [[ "$has_cuda" -eq 1 ]]; then
        [[ -f "$CUDA_SRC" ]] || die "no CUDA source at $CUDA_SRC"
        mkdir -p "$EXAMPLES_DIR/bin"
        if [[ ! -f "$CUDA_BIN" || "$CUDA_SRC" -nt "$CUDA_BIN" ]]; then
            echo "Building CUDA $kernel binary ($arch)..."
            nvcc -lm -arch="$arch" "$CUDA_SRC" -o "$CUDA_BIN"
        else
            echo "CUDA $kernel binary is up to date."
        fi
    fi
fi

[[ "$has_rust" -eq 0 || -x "$DEMO_BIN" ]] || die "$DEMO_BIN not found (drop --no-build?)"
[[ "$has_cuda" -eq 0 || -x "$CUDA_BIN" ]] || die "$CUDA_BIN not found (drop --no-build?)"

# ----------------------------------------------------------------------- run
mkdir -p "$KERNEL_DIR"
rm -f "$CSV"

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Interleave implementations inside the repetition loop so GPU boost-clock and
# machine-load drift over a long session hit all of them equally, instead of
# penalizing whichever block happens to run last.
for n in "${size_list[@]}"; do
    for ((i = 1; i <= runs; i++)); do
        for impl in "${impl_list[@]}"; do
            case "$impl" in
                rust-gpu) cmd=("$DEMO_BIN" "$kernel" "$n") ;;
                rust)     cmd=("$DEMO_BIN" "${kernel}_cpu" "$n") ;;
                cuda)     cmd=("$CUDA_BIN" "$n") ;;
            esac
            "$SCRIPT_DIR/bench_runs.sh" -o "$CSV" -k "$kernel" -i "$impl" \
                -s "$n" -n 1 -v "$version" -- "${cmd[@]}"
        done
    done
done

finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$SCRIPT_DIR/record_meta.py" "$VERSION_DIR/meta.json" \
    --kernel "$kernel" --impls "$impls" --runs "$runs" --sizes "$sizes" \
    --profile "$profile" --started "$started" --finished "$finished"

ln -sfn "$version" "$RESULTS_DIR/latest"

# Per-kernel report, then refresh the whole version's combined report so it
# always reflects every kernel recorded under it.
echo
echo "=== $kernel ($version) ==="
"$SCRIPT_DIR/analyze_runs.py" "$KERNEL_DIR" -o "$KERNEL_DIR/summary.txt" >/dev/null
"$SCRIPT_DIR/analyze_runs.py" "$VERSION_DIR" -o "$VERSION_DIR/summary.txt"
