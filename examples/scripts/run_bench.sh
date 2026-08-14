#!/usr/bin/env bash
# Run one kernel across any combination of implementations and record the
# results into a versioned directory.
#
# Implementations:
#   rust-gpu  the Rust DSL / custom compiler  (demo <kernel> <n>)
#   rust      pure Rust on the CPU, sequential   (demo <kernel>_cpu <n>)
#   cuda      hand-written .cu                (bin/<kernel> <n>)
#
# Usage:
#   run_bench.sh -k <kernel> [options]
#
#   -k, --kernel <name>     kernel to run (see scripts/kernels.conf)
#   -i, --impls <list>      comma-separated, default rust-gpu,rust,cuda
#   -n, --runs <n>          repetitions per (impl, size), default from kernels.conf
#   -s, --sizes "<a b c>"   sizes to sweep, default from kernels.conf
#       --sizes-<impl> "<a b>"  sizes for just that impl, e.g. --sizes-rust "1024"
#       --limit-<impl> <n>  that impl only runs the first <n> of the swept sizes
#                           (0 = all); default for "rust" in a mixed run comes
#                           from CPU_SIZE_LIMIT in kernels.conf
#       --order <o>         blocked (default): run one impl's whole sweep before
#                           starting the next. interleaved: rotate impls inside
#                           the repetition loop, so boost-clock and machine-load
#                           drift over a long session hit each impl equally.
#       --version <vN>      write into results/<vN> (created if absent), or
#                           "latest" to follow the symlink; default: next free vN
#       --profile <p>       release (default) or debug
#       --arch <sm_XX>      nvcc target, default from kernels.conf
#       --no-build          skip cargo/nvcc, use whatever is already built
#       --dry-run           print the resolved plan and exit
#
# Impls do not have to share a size list. Sizes they do share are the ones
# analyze_runs.py turns into overhead rows; sizes only one impl ran still show up
# in the per-(kernel, impl, size) table. That is what lets the sequential CPU
# impl sit out the big tiers instead of holding up the whole sweep.
#
# Examples:
#   run_bench.sh -k julia                          # all three impls, defaults
#   run_bench.sh -k julia -i rust-gpu,cuda         # DSL vs hand-written CUDA
#   run_bench.sh -k julia -i rust-gpu,rust -n 10   # DSL vs CPU, 10 reps
#   run_bench.sh -k julia -i rust-gpu,rust --limit-rust 2   # CPU: first 2 tiers
#   run_bench.sh -k mm -i rust-gpu,rust --sizes-rust "1024" # CPU: only 1024
#   run_bench.sh -k julia --order interleaved      # rotate impls run by run
#   run_bench.sh -k mm -s "512 1024" --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"
RESULTS_DIR="$EXAMPLES_DIR/results"

# shellcheck source=kernels.conf
source "$SCRIPT_DIR/kernels.conf"

usage() {
    # Everything from the shebang to the first non-comment line.
    sed -n '2,${/^#/!q;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}" >&2
    exit 1
}

die() { echo "Error: $*" >&2; exit 1; }

kernel=""
impls="rust-gpu,rust,cuda"
runs=""
sizes=""
version=""
order="blocked"
profile="release"
arch="$NVCC_ARCH"
do_build=1
dry_run=0
declare -A sizes_for=()   # impl -> explicit size list  (--sizes-<impl>)
declare -A limit_for=()   # impl -> "first N sizes"     (--limit-<impl>)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--kernel)  kernel="$2"; shift 2 ;;
        -i|--impls)   impls="$2"; shift 2 ;;
        -n|--runs)    runs="$2"; shift 2 ;;
        -s|--sizes)   sizes="$2"; shift 2 ;;
        --sizes-*)    sizes_for["${1#--sizes-}"]="$2"; shift 2 ;;
        --limit-*)    limit_for["${1#--limit-}"]="$2"; shift 2 ;;
        --order)      order="$2"; shift 2 ;;
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
has_gpu=0
for impl in "${impl_list[@]}"; do
    case "$impl" in
        rust-gpu) has_rust=1; has_gpu=1 ;;
        rust)     has_cpu=1; has_rust=1 ;;
        cuda)     has_cuda=1; has_gpu=1 ;;
        *)        die "unknown impl '$impl' (expected rust-gpu, rust or cuda)" ;;
    esac
done

selected() {
    local impl
    for impl in "${impl_list[@]}"; do [[ "$impl" == "$1" ]] && return 0; done
    return 1
}

for impl in "${!sizes_for[@]}"; do
    selected "$impl" || die "--sizes-$impl given but '$impl' is not in --impls ($impls)"
done
for impl in "${!limit_for[@]}"; do
    selected "$impl" || die "--limit-$impl given but '$impl' is not in --impls ($impls)"
    [[ "${limit_for[$impl]}" =~ ^[0-9]+$ ]] || die "--limit-$impl must be a number (0 = all sizes)"
done

case "$order" in
    blocked|interleaved) ;;
    *) die "--order must be 'blocked' or 'interleaved' (got '$order')" ;;
esac

case "$profile" in
    release|debug) ;;
    *) die "--profile must be 'release' or 'debug' (got '$profile')" ;;
esac
if [[ "$profile" == "debug" && "$has_cpu" -eq 1 ]]; then
    echo "Warning: the 'rust' (CPU) impl in a debug build runs at opt-level 0 and" >&2
    echo "         will be roughly 10-50x slower than it should be." >&2
fi

# ------------------------------------------------------------------ defaults
# A run that includes the CPU impl sweeps CPU_SIZES rather than GPU_SIZES, so the
# tiers stay ones the CPU can actually reach -- analyze_runs.py only compares
# within a shared size, and a size no impl shares yields no overhead row.
if [[ -z "$sizes" ]]; then
    if [[ "$has_cpu" -eq 1 ]]; then
        sizes="${CPU_SIZES[$kernel]:-${GPU_SIZES[$kernel]}}"
    else
        sizes="${GPU_SIZES[$kernel]}"
    fi
fi
[[ -n "$runs" ]] || runs="${RUNS_DEFAULT[$kernel]:-$DEFAULT_RUNS}"

read -ra size_list <<< "$sizes"
[[ ${#size_list[@]} -gt 0 ]] || die "--sizes is empty"

# Each impl then gets its own slice of that sweep: an explicit --sizes-<impl>
# list, or the first N of it. The sequential CPU impl defaults to a short slice
# when it shares a run with a GPU impl (CPU_SIZE_LIMIT), because the big tiers
# cost hours on one core and only the shared leading tier feeds a comparison.
declare -A run_sizes=()
for impl in "${impl_list[@]}"; do
    if [[ -n "${sizes_for[$impl]:-}" ]]; then
        run_sizes["$impl"]="${sizes_for[$impl]}"
        continue
    fi
    limit="${limit_for[$impl]:-}"
    if [[ -z "$limit" && "$impl" == "rust" && "$has_gpu" -eq 1 ]]; then
        limit="${CPU_SIZE_LIMIT[$kernel]:-$DEFAULT_CPU_SIZE_LIMIT}"
    fi
    if [[ -z "$limit" || "$limit" -eq 0 || "$limit" -ge ${#size_list[@]} ]]; then
        run_sizes["$impl"]="${size_list[*]}"
    else
        run_sizes["$impl"]="${size_list[*]:0:$limit}"
    fi
done

runs_size() {  # runs_size <impl> <size> -> true if that impl runs that size
    local n
    for n in ${run_sizes[$1]}; do [[ "$n" == "$2" ]] && return 0; done
    return 1
}

# Sweep order: the union of the per-impl lists, in the configured order (ripple's
# descend on purpose, so don't sort), with sizes only an override named last.
all_sizes=()
add_size() {
    local seen
    for seen in ${all_sizes[@]+"${all_sizes[@]}"}; do [[ "$seen" == "$1" ]] && return; done
    all_sizes+=("$1")
}
for n in "${size_list[@]}"; do
    for impl in "${impl_list[@]}"; do
        runs_size "$impl" "$n" && { add_size "$n"; break; }
    done
done
for impl in "${impl_list[@]}"; do
    for n in ${run_sizes[$impl]}; do add_size "$n"; done
done
[[ ${#all_sizes[@]} -gt 0 ]] || die "no impl has any size to run"

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

build_cmd() {  # build_cmd <impl> <size> -> fills the `cmd` array
    case "$1" in
        rust-gpu) cmd=("$DEMO_BIN" "$kernel" "$2") ;;
        rust)     cmd=("$DEMO_BIN" "${kernel}_cpu" "$2") ;;
        cuda)     cmd=("$CUDA_BIN" "$2") ;;
    esac
}
dry_cmd() { build_cmd "$1" "$2"; echo "${cmd[*]}"; }

total=0
for impl in "${impl_list[@]}"; do
    read -ra impl_size_list <<< "${run_sizes[$impl]}"
    total=$(( total + ${#impl_size_list[@]} * runs ))
done

# Sizes more than one impl runs are the ones that become overhead rows.
shared_sizes=()
for n in "${all_sizes[@]}"; do
    count=0
    for impl in "${impl_list[@]}"; do runs_size "$impl" "$n" && count=$((count + 1)); done
    (( count > 1 )) && shared_sizes+=("$n")
done

cat <<EOF
kernel   : $kernel
impls    : ${impl_list[*]}
sizes    : ${all_sizes[*]}
EOF
for impl in "${impl_list[@]}"; do
    printf '  %-9s: %s\n' "$impl" "${run_sizes[$impl]}"
done
cat <<EOF
compared : ${shared_sizes[*]:-(none -- no size is run by two impls)}
runs     : $runs per (impl, size)  ->  $total executions
order    : $order
profile  : $profile
output   : $KERNEL_DIR
EOF

if [[ "$dry_run" -eq 1 ]]; then
    echo
    if [[ "$order" == "blocked" ]]; then
        echo "Commands that would run (x$runs each, in this order):"
        for impl in "${impl_list[@]}"; do
            for n in ${run_sizes[$impl]}; do echo "  [$impl] $(dry_cmd "$impl" "$n")"; done
        done
    else
        echo "Commands that would run (once per repetition, interleaved):"
        for n in "${all_sizes[@]}"; do
            for impl in "${impl_list[@]}"; do
                runs_size "$impl" "$n" || continue
                echo "  [$impl] $(dry_cmd "$impl" "$n")"
            done
        done
    fi
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

if [[ "$order" == "blocked" ]]; then
    # One impl at a time, start to finish: the whole sweep for rust-gpu, then the
    # whole sweep for rust. Each (impl, size) is a single bench_runs.sh call.
    for impl in "${impl_list[@]}"; do
        for n in ${run_sizes[$impl]}; do
            build_cmd "$impl" "$n"
            "$SCRIPT_DIR/bench_runs.sh" -o "$CSV" -k "$kernel" -i "$impl" \
                -s "$n" -n "$runs" -v "$version" -- "${cmd[@]}"
        done
    done
else
    # Interleave implementations inside the repetition loop so GPU boost-clock and
    # machine-load drift over a long session hit all of them equally, instead of
    # penalizing whichever block happens to run last.
    for n in "${all_sizes[@]}"; do
        for ((i = 1; i <= runs; i++)); do
            for impl in "${impl_list[@]}"; do
                runs_size "$impl" "$n" || continue
                build_cmd "$impl" "$n"
                "$SCRIPT_DIR/bench_runs.sh" -o "$CSV" -k "$kernel" -i "$impl" \
                    -s "$n" -n 1 -v "$version" -- "${cmd[@]}"
            done
        done
    done
fi

finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

meta_args=()
for impl in "${impl_list[@]}"; do
    meta_args+=(--impl-sizes "$impl=${run_sizes[$impl]}")
done

"$SCRIPT_DIR/record_meta.py" "$VERSION_DIR/meta.json" \
    --kernel "$kernel" --impls "$impls" --runs "$runs" --sizes "${all_sizes[*]}" \
    "${meta_args[@]}" --order "$order" \
    --profile "$profile" --started "$started" --finished "$finished"

ln -sfn "$version" "$RESULTS_DIR/latest"

# Per-kernel report, then refresh the whole version's combined report so it
# always reflects every kernel recorded under it.
echo
echo "=== $kernel ($version) ==="
"$SCRIPT_DIR/analyze_runs.py" "$KERNEL_DIR" -o "$KERNEL_DIR/summary.txt" >/dev/null
"$SCRIPT_DIR/analyze_runs.py" "$VERSION_DIR" -o "$VERSION_DIR/summary.txt"
