# GPU Capacity & Benchmark Sizing

Findings and sizing decisions for running each benchmark kernel at "max
device capacity", then two successive 15% step-downs.

> **Note on re-measurement:** the device section and the `nbodies` /
> `nearest_neighbor` tiers below were captured in an earlier session (free
> memory 7.421 GB at that time). The `mm` / `julia` / `ripple` / `raytracer`
> / `vector_sum` tiers were added in a later session, re-querying free memory
> at that time (7.348 GB — slightly less headroom, other things were using
> the GPU) and reapplying the same 90%-of-free / 15%-step-down methodology.
> The tier sizes below reflect whichever free-memory reading was current
> when each kernel's numbers were computed; re-run the query in
> `## Methodology` if you want everything on one consistent snapshot.

## Device

`nvidia-smi` was unusable at query time (`Failed to initialize NVML: Driver/library
version mismatch` — loaded kernel module `595.84` vs. userspace lib `595.71.05`;
a reboot resolves this). Queried the CUDA runtime API directly instead
(`cudaGetDeviceProperties` / `cudaMemGetInfo`), which doesn't go through NVML.

**NVIDIA GeForce RTX 3070** — compute capability 8.6

| Property | Value |
|---|---|
| Total global memory | 8.188 GB (8,187,674,624 B) |
| Free global memory (at query time) | 7.421 GB (7,421,231,104 B) |
| SMs | 46 |
| Max threads / block | 1024 |
| Max threads / SM | 1536 (→ 70,656 concurrent threads across the device) |
| Max grid dim | (2,147,483,647, 65535, 65535) |
| Warp size | 32 |
| Shared mem / block / SM | 48 KB / 100 KB |
| Registers / block / SM | 65536 / 65536 |

Thread/grid/register limits are not binding for either benchmark below — both
use flat 1D grids of 128-thread blocks, and `num_blocks` stays well under the
`2,147,483,647`-wide max grid dim for any N considered here. **Global memory is
the actual constraint.**

## Methodology

1. Use **free** memory (7.421 GB), not total (8.188 GB) — ~0.77 GB is already
   held by the desktop/display, and that's the memory actually available to a
   benchmark run right now.
2. Apply a **10% safety margin** on top of that (usable budget = 90% of free =
   **6.679 GB**) to leave headroom for the CUDA context itself, allocator
   fragmentation, and anything else that starts on the GPU mid-run.
3. Tier 1 ("max capacity") = floor(usable budget / bytes-per-element).
   Tier 2 = Tier 1 × 0.85. Tier 3 = Tier 1 × 0.85² (≈ 0.7225 of Tier 1).

Each `CudaVec::new` allocates one device buffer sized to the host `Vec` it
wraps (`runtime/src/cuda_vec.rs`); there's no double-buffering, so per-element
device footprint is just the sum of each benchmark's `CudaVec` element sizes.

## nearest_neighbor — memory-bound, single pass (O(n))

Device allocations: `d_locations` (2 × f32/record) + `d_distances` (1 × f32/record)
= **12 bytes/record**.

| Tier | N (records) | Device memory | Measured duration |
|---|---|---|---|
| 100% (max) | **556,592,332** | 6.679 GB | **1.81 s** (h2d 393ms + kernel 17ms + d2h 1.40s) |
| 85% | **473,103,482** | 5.677 GB | **1.32 s** (h2d 336ms + kernel 14ms + d2h 967ms) |
| 72.25% | **402,137,959** | 4.826 GB | **1.11 s** (h2d 311ms + kernel 12ms + d2h 786ms) |

The kernel does O(n) work total, so these are directly runnable — memory is
the real ceiling. Runtime is dominated by the H2D/D2H PCIe transfers, not
compute (kernel itself is 12-17ms even at 556M records); it scales linearly
with N as expected. **All three tiers finish in under 2 seconds.**

## nbodies — memory-bound number is not practically runnable (O(n²))

Device allocation: `p` (6 × f32/body: pos + vel) = **24 bytes/body**.

| Tier | N (bodies), memory-bound |
|---|---|
| 100% (max) | 278,296,166 |
| 85% | 236,551,741 |
| 72.25% | 201,068,979 |

**Decision: don't use these directly.** `gpu_n_bodies` is all-pairs (O(n²))
and `STEPS = 3` runs the whole thing three times — a rough FLOP estimate put
the 100% memory tier at ~640 hours. Instead of trusting that estimate, picked
a candidate N and **measured it directly**: compiled a standalone CUDA program
with the same kernel math as `gpu_n_bodies`/`gpu_integrate` (`bench_nb.cu`),
timed via `cudaEvent`, run on the same idle GPU at the exact candidate sizes.

| Tier | N (bodies) | Measured duration (6 kernel launches: 3 steps × 2 kernels) |
|---|---|---|
| 100% (max) | **1,000,000** | **13.6 s** |
| 85% | **850,000** | **10.0 s** |
| 72.25% | **722,500** | **7.2 s** |

The FLOP estimate was pessimistic — actual GPU throughput on this kernel is
higher than the crude model assumed. All three tiers are well within a
reasonable benchmark run.

## vector_sum — memory-bound, single pass (O(n))

Re-queried free memory for this and the remaining kernels: **7.348 GB**
free at query time → usable budget (90%) = **6.614 GB**.

Device allocations: `a` + `b` + `result` (3 × f32/element) = **12
bytes/element** — same footprint and same O(n) shape as `nearest_neighbor`,
so the memory-bound tiers are directly runnable.

| Tier | N (elements) | Device memory | Measured duration (h2d+kernel+d2h) |
|---|---|---|---|
| 100% (max) | **551,121,715** | 6.613 GB | **4.03 s** (kernel 17.7 ms) |
| 85% | **468,453,457** | 5.621 GB | **3.36 s** (kernel 14.0 ms) |
| 72.25% | **398,185,439** | 4.778 GB | **2.91 s** (kernel 11.9 ms) |

Measured directly with the standalone CUDA binary (`examples/bin/vector_sum`,
built from `examples/src/benchmarks/cuda/vector_sum.cu`) via `/usr/bin/time`
for wall clock plus the binary's own `cudaEvent`-timed kernel line. Kernel
time is negligible (single add per element); total time is dominated by
host-side vector generation and H2D/D2H transfer, scaling linearly with N as
expected.

## mm (matrix multiply) — memory-bound size *is* practically runnable (O(n³))

Device allocations: `a` (m×n) + `b` (n×k) + `c` (m×k), all f32, with
`m = n = k = dim` (see `examples/src/benchmarks/mm.rs`) = **12 bytes ×
dim²**, i.e. `dim = floor(sqrt(usable_budget / 12))`.

| Tier | dim | Device memory | Elements (dim²) |
|---|---|---|---|
| 100% (max) | **23,475** | 6.613 GB | 551,075,625 |
| 85% | **19,953** | 4.777 GB | 398,122,209 |
| 72.25% | **16,960** | 3.452 GB | 287,641,600 |

Unlike `nbodies`, this one *is* practically runnable at the memory-bound
size — checked by measuring `bin/mm` directly at 512/1024/2048/4096/8192/
16384 first (confirms clean O(dim³) scaling — each doubling costs ~8x), then
running the exact tier sizes above rather than trusting the extrapolation:

| Tier | dim | Measured kernel time | Measured wall time |
|---|---|---|---|
| 100% (max) | 23,475 | **26.93 s** | 31.94 s |
| 85% | 19,953 | **16.01 s** | 19.63 s |
| 72.25% | 16,960 | **9.14 s** | 11.88 s |

This is a naive (non-tiled, no shared memory) matmul kernel, so it's much
more compute-bound than the other kernels here — tier 1 takes ~27s of actual
kernel time, longer than any other benchmark in this doc, but still finishes
in well under a minute and doesn't need the nbodies-style "pick a smaller
time-budgeted N" treatment. Host RAM was checked before running tier 1
(`m*n*4` bytes × 3 arrays ≈ 6.6 GB) — fine against the ~10 GB that was
available.

## julia / ripple / raytracer — pixel-parallel image kernels, O(dim²) total work

All three write a `dim × dim` RGBA `f32` image = **16 bytes/pixel**
(`raytracer` also uploads a fixed 20-sphere array, 560 bytes — negligible
next to the image). Same formula as `mm`: `dim = floor(sqrt(usable_budget /
16))`. Because all three share this footprint, **the tier sizes are
identical** across all three kernels:

| Tier | dim | Device memory | Pixels (dim²) |
|---|---|---|---|
| 100% (max) | **20,330** | 6.613 GB | 413,308,900 |
| 85% | **17,280** | 4.778 GB | 298,598,400 |
| 72.25% | **14,688** | 3.452 GB | 215,737,344 |

`julia` and `ripple` launch one thread *per block* (`spawn!(..., (dim, dim,
1), (1, 1, 1), ...)` — grid dims (`dim`, `dim`, 1), block dims (1,1,1)),
which is a very unusual (and inefficient — 31/32 lanes idle per warp)
launch shape. This also means `dim` is bound by **max grid dim y = 65535**,
not just memory — but the memory-bound `dim` (20,330) is well under that, so
memory is still the binding constraint, same conclusion as the other
kernels. `raytracer` uses a conventional 16×16 2D block instead.

Verified all three scale as O(dim²) (quadratic in dim = linear in pixel
count) by measuring at dim = 256/1024/4096/8192 first — confirms the
1-thread-per-block launch shape for `julia`/`ripple` doesn't blow up despite
being wasteful, then measured the exact tier sizes directly:

| Tier | dim | julia (kernel) | ripple (kernel) | raytracer (kernel) |
|---|---|---|---|---|
| 100% (max) | 20,330 | **3.64 s** | **367 ms** | **35 ms** |
| 85% | 17,280 | **2.66 s** | **265 ms** | **22 ms** |
| 72.25% | 14,688 | **1.87 s** | **185 ms** | **17 ms** |

`julia`'s 200-iteration escape loop per pixel makes it noticeably heavier
than `ripple` (closed-form trig) or `raytracer` (20-sphere loop, but only
~4 pixels in 8192² actually iterate the full sphere list at these test
sphere counts). All three finish in well under 4 seconds even at 100%
capacity — comfortably the fastest-per-tier kernels in this doc.

## Vector sizes to pass at run time

Every benchmark takes a single `size` arg (`N` or `dim`), forwarded straight
into `run(n)` — that's the value to pass on the command line / as the
Makefile size var.

Invocation shapes (from `examples/src/main.rs` and `examples/Makefile`):

```
../target/debug/demo nearest_neighbor <N>
../target/debug/demo nbodies <N>
../target/debug/demo vector_sum <N>
../target/debug/demo mm <dim>
../target/debug/demo julia <dim>
../target/debug/demo ripple <dim>
../target/debug/demo raytracer <dim>

# or, from examples/
NEAREST_NEIGHBOR_SIZE=<N> make nearest_neighbor
NBODIES_SIZE=<N> make nbodies
```

### nearest_neighbor

| Tier | N to pass | Expected duration |
|---|---|---|
| 100% (max) | `556592332` | ~1.8 s |
| 85% | `473103482` | ~1.3 s |
| 72.25% | `402137959` | ~1.1 s |

```
../target/debug/demo nearest_neighbor 556592332
../target/debug/demo nearest_neighbor 473103482
../target/debug/demo nearest_neighbor 402137959
```

### nbodies

| Tier | N to pass | Expected duration |
|---|---|---|
| 100% (max, time-budgeted) | `1000000` | ~13.6 s |
| 85% | `850000` | ~10.0 s |
| 72.25% | `722500` | ~7.2 s |

```
../target/debug/demo nbodies 1000000
../target/debug/demo nbodies 850000
../target/debug/demo nbodies 722500
```

### vector_sum

| Tier | N to pass | Expected duration |
|---|---|---|
| 100% (max) | `551121715` | ~4.0 s |
| 85% | `468453457` | ~3.4 s |
| 72.25% | `398185439` | ~2.9 s |

### mm

| Tier | dim to pass | Expected duration |
|---|---|---|
| 100% (max) | `23475` | ~27 s |
| 85% | `19953` | ~16 s |
| 72.25% | `16960` | ~9 s |

### julia / ripple / raytracer

Same `dim` values for all three (identical 16-byte/pixel footprint):

| Tier | dim to pass | julia | ripple | raytracer |
|---|---|---|---|---|
| 100% (max) | `20330` | ~3.6 s | ~0.4 s | ~0.04 s |
| 85% | `17280` | ~2.7 s | ~0.3 s | ~0.02 s |
| 72.25% | `14688` | ~1.9 s | ~0.2 s | ~0.02 s |

## Running each kernel N times for averaging/analysis

Goal: measure how much slower the Rust DSL is than hand-written CUDA, per
kernel, per size. Both the Rust DSL binary (`demo <bench> <N>`) and the
hand-written CUDA binaries (`examples/bin/<bench> <N>`) print the same line
format: `[<kernel>] elapsed: <ms> ms`. `bench_runs.sh` runs one command N
times, pulls that line out of each run, and appends one CSV row per run —
tagged with which kernel it was, which size, and **whether it was the Rust
DSL or pure CUDA** (that's an explicit flag you pass, `-i rust` or `-i cuda`,
not something it guesses).

```
examples/scripts/bench_runs.sh -o <csv_file> -k <kernel> -i rust|cuda -s <size> [-n <runs>] -- <command> [args...]
```

- `-o` CSV file to append to (created automatically with a header)
- `-k` kernel name, e.g. `nbodies` or `nearest_neighbor` — just a label for grouping
- `-i` `rust` or `cuda` — **this is the column that lets you tell them apart later**
- `-s` the size N you're passing to the command (record it here; you still
  have to put the actual number in the command yourself)
- `-n` how many repetitions (default 30)
- everything after `--` is the actual command to run

**Different kernels** = just change `-k` and the command. **Different sizes**
= change `-s` and the number in the command (they should always match — `-s`
is only used for grouping/reporting, it doesn't get passed to the binary for
you). Same CSV file can accumulate results from every kernel/size/impl
combination; `analyze_runs.py` groups them back apart.

Concretely, to get 30 runs of nbodies at N=1,000,000 for both versions:

```
cd examples
cargo build                                                    # builds ../target/debug/demo
mkdir -p bin && nvcc -lm src/benchmarks/cuda/nbodies.cu -o bin/nbodies  # builds the CUDA version

scripts/bench_runs.sh -o runs.csv -k nbodies -i rust -s 1000000 -n 30 -- \
    ../target/debug/demo nbodies 1000000

scripts/bench_runs.sh -o runs.csv -k nbodies -i cuda -s 1000000 -n 30 -- \
    bin/nbodies 1000000
```

Full sweep over all 6 candidate sizes from this doc, both implementations:

```
cd examples
cargo build
mkdir -p bin
nvcc -lm src/benchmarks/cuda/nbodies.cu -o bin/nbodies
nvcc -lm src/benchmarks/cuda/nearest_neighbor.cu -o bin/nearest_neighbor

CSV=runs.csv

for n in 1000000 850000 722500; do
  scripts/bench_runs.sh -o $CSV -k nbodies -i rust -s $n -n 30 -- ../target/debug/demo nbodies $n
  scripts/bench_runs.sh -o $CSV -k nbodies -i cuda -s $n -n 30 -- bin/nbodies $n
done

for n in 556592332 473103482 402137959; do
  scripts/bench_runs.sh -o $CSV -k nearest_neighbor -i rust -s $n -n 30 -- ../target/debug/demo nearest_neighbor $n
  scripts/bench_runs.sh -o $CSV -k nearest_neighbor -i cuda -s $n -n 30 -- bin/nearest_neighbor $n
done

scripts/analyze_runs.py $CSV -o results/summary.txt
```

`analyze_runs.py <csv_file> [-k KERNEL ...] [-o OUT_FILE]` prints mean/stdev/
min/max/median grouped by `(kernel, impl, size)`, then — for every
`(kernel, size)` where both a `rust` and a `cuda` row exist — a second table
with `rust_ms`, `cuda_ms`, `overhead_ms`, and `overhead_%`, which is the
actual "how much more time does Rust cost" number. `-k` filters to specific
kernels (repeatable); `-o` saves the same report to a file in addition to
printing it, so it's there for later instead of only living in the terminal
scrollback. Example output (smoke test, nearest_neighbor, N=5000, 3 runs each):

```
kernel              impl  size          n   mean_ms     stdev_ms  min_ms    max_ms    median_ms
--------------------------------------------------------------------------------------------------
nearest_neighbor    cuda  5000          3   0.122       0.007     0.118     0.130     0.118
nearest_neighbor    rust  5000          3   0.242       0.024     0.214     0.260     0.251

Rust vs CUDA overhead
kernel              size          rust_ms     cuda_ms     overhead_ms overhead_%
------------------------------------------------------------------------------
nearest_neighbor    5000          0.242       0.122       0.120       98.1
```

Each row in `runs.csv` is `timestamp,kernel,impl,size,run_index,elapsed_ms,command`
— raw enough to redo the aggregation in anything else (pandas, a spreadsheet,
etc.) if the built-in summary isn't enough.

## Summary of decisions

- Sized off **free** device memory at run time, not total, with a **10%**
  safety margin.
- **nearest_neighbor**: use the memory-bound tiers as-is (556,592,332 /
  473,103,482 / 402,137,959) — the kernel is O(n), so they're runnable.
  Measured **~1.1-1.8 s** total across all three tiers.
- **nbodies**: memory-bound tiers are not usable due to O(n²) cost × 3 steps.
  Used a time-budgeted N instead (1,000,000 / 850,000 / 722,500), verified by
  directly timing the actual kernel math at those exact sizes rather than
  trusting a FLOP estimate. Measured **~7.2-13.6 s** total across all three
  tiers.
- All six proposed runs (both benchmarks × 3 tiers) complete in **under 15
  seconds each** — no run should feel "SO long."
- **vector_sum**: same 12-bytes/element, O(n) profile as `nearest_neighbor`
  — memory-bound tiers used as-is (551,121,715 / 468,453,457 / 398,185,439).
  Measured **~2.9-4.0 s** total across all three tiers.
- **mm**: O(dim³) like `nbodies` is O(n²), but unlike `nbodies` the
  memory-bound `dim` (23,475 / 19,953 / 16,960) turned out to be directly
  measurable and practically runnable — no need to time-budget down to a
  smaller N. Measured **~9-27 s** kernel time across the three tiers (the
  slowest kernel in this doc, but still well under a minute).
- **julia / ripple / raytracer**: all three are O(dim²) total work (linear
  in pixel count) with an identical 16-bytes/pixel footprint, so they share
  one set of tiers (20,330 / 17,280 / 14,688). `julia` and `ripple` use an
  unusual 1-thread-per-block launch (`grid=(dim,dim,1)`, `block=(1,1,1)`),
  which is also grid-dim-y-bound (≤ 65535) — not binding here since memory
  caps `dim` well below that. Measured **~17 ms to ~3.6 s** across all nine
  (kernel × tier) combinations — the fastest kernels in this doc.
- Per-kernel `results/run_<kernel>.sh` scripts (mirroring the existing
  `run_julia.sh`) build both implementations and run all three tiers × 30
  reps each, writing `results/<kernel>_runs.csv` and
  `results/<kernel>_summary.txt`.
