# Why the Rust DSL sometimes beats hand-written CUDA (and whether the data is valid)

Analysis of the anomaly in the `*_summary.txt` results where the Rust DSL is
sometimes *faster* than the pure CUDA binaries — most strikingly `mm`, where
Rust wins by 11-13% at every size. Since the DSL ultimately runs CUDA under
the hood, a consistent negative overhead should be impossible. This document
explains what actually happened, backed by controlled experiments run on the
same machine/GPU as the sweeps.

## TL;DR

**The measurements are real, but the mm comparison is unfair: the two
binaries do not run the same GPU code.** The DSL's code generator silently
rewrites 64-bit loop counters (`u64` in the Rust source) to 32-bit `int` in
the generated CUDA, and 32-bit index arithmetic is measurably ~12% faster in
mm's inner loop. When the hand-written kernel is rebuilt with the *same*
loop type, the entire "Rust is faster" gap disappears and the DSL shows the
expected small positive overhead (~1% at these kernel durations).

Nothing needs refactoring in `examples/src/benchmarks/*.rs`. The fixes are
three defects in the DSL compiler's CUDA codegen (loop counter types,
literal suffixes, dropped `else` branches — see "What to change") plus
build flags in the benchmark scripts.

## What the data actually shows

The five summaries fall into three distinct patterns:

| Kernel | Overhead (rust vs cuda) | Pattern |
|---|---|---|
| mm | **-11.8 / -11.3 / -13.1%** | consistent, all sizes → *real anomaly* |
| nearest_neighbor | +9.1 / +7.7 / +2.2% | consistent positive → expected behavior |
| nbodies | +2.5 / -0.9 / +1.4% | sign flips → measurement noise |
| julia | +0.5 / +3.2 / -0.6% | sign flips → measurement noise |
| raytracer | +4.7 / +3.9 / -8.3% | sign flips → measurement noise |

Only `mm` needs a causal explanation. The others are within run-to-run
variance (compare the stdev columns — e.g. raytracer's "-8.3%" is a 0.63 ms
difference on a ~7 ms kernel with stdev ~0.6 ms).

## Root cause: the generated kernel is not the same code

### 1. Loop counters: `u64` in the source → `int` in the generated CUDA (dominant effect)

`compiler/src/codegen/cuda.rs:359` hardcodes the loop variable type:

```rust
"{indent}for (int {var} = {start}; {var} < {end}; {var} += {step}) {{\n{body}{end_indent}}}",
```

So `mm.rs`'s `for i in (0u64, n).step_by(1u64)` becomes
`for (int i = 0; i < n; i += 1)` in `generated_mm.cu`, while the
hand-written `src/benchmarks/cuda/mm.cu` faithfully uses
`for (uint64_t i = 0; ...)`. In mm's inner loop every iteration computes
`a[row * n + i]` and `b[i * k + col]` — with a 32-bit counter the compiler
strength-reduces the 64-bit address arithmetic far better.

**Controlled experiment** (same GPU, interleaved runs so thermal drift can't
bias a variant, 5 runs each, size 5000):

| Variant | Mean kernel time |
|---|---|
| hand-written mm.cu, default nvcc arch (what the sweep ran) | 208.2 ms |
| hand-written mm.cu, `-arch=sm_86` | 206.3 ms |
| hand-written mm.cu, `-arch=sm_86` **+ `int` loop counter** | **182.7 ms** |

The `int` counter alone accounts for a ~12% speedup — exactly the size and
direction of the "anomaly". A second interleaved experiment comparing the
Rust demo binary directly against the int-counter build (identical device
code) showed **rust ≈ cuda within noise (~+1%)** — i.e. once the device code
matches, the DSL behaves exactly as expected.

Supporting evidence from the other kernels:
- `julia`: the hand-written `julia.cu` *also* uses `int iter` for its
  200-step escape loop — both sides run 32-bit counters, and julia shows ~0
  gap. Consistent.
- `nbodies`: generated code uses `int j` vs hand-written `uint64_t j`, but
  its inner loop is dominated by `sqrtf`/FLOPs rather than indexing, and the
  gap is ~0. Consistent.
- `nearest_neighbor` / `raytracer` / `ripple`: no data-dependent loops over
  `n` (or trivially small fixed loops) → no counter-type advantage → normal
  positive overhead or noise. Consistent.

### 1b. Literal suffixes stripped → hidden double-precision math (hurts the DSL side)

`compiler/src/codegen/cuda.rs:418-419` emits literals via `base10_digits()`,
which discards the Rust suffix: `255.0f32` → `255.0`. In C++ an unsuffixed
`255.0` is a **double**, so expressions like `r * 255.0` or
`128.0 + 127.0 * cosf(...)` promote to FP64 and only narrow back to float on
assignment. Counting `.f64` instructions in the generated PTX:

| Kernel | FP64 instructions in generated PTX |
|---|---|
| ripple | **26** |
| raytracing | **11** |
| all others (mm, julia, nbodies, euclid, add_vectors, integrate) | 0 |

nvcc managed to fold the harmless patterns back to float (e.g. julia's
`> 1000.0` comparison, nbodies' `1.0 / sqrtf(...)`), which is why those
kernels show no FP64 — but ripple's and raytracer's arithmetic chains
survive as real double math. On the RTX 3070, FP64 throughput is **1/64**
of FP32, so this bug makes those DSL kernels *slower* than the handwritten
CUDA. The unfairness therefore cuts both ways: mm was unfairly fast
(32-bit counters), ripple/raytracer unfairly slow (FP64 contamination).

### 1c. `else` branches are silently dropped (correctness, not perf)

`gen_if` (`compiler/src/codegen/cuda.rs:282`) never reads
`expr_if.else_branch` — any `else` block in a kernel is discarded without
error. `raytracer.rs` has one (`else { t = -99999.0f32; n = 0.0f32; }`) and
it is absent from `generated_raytracing.cu`; the benchmark only produces
correct output by coincidence, because that else re-assigns the variables'
initial values. Any kernel whose else branch does real work would be
silently wrong. This doesn't skew today's timings, but it must be fixed
before trusting the DSL's output generally.

### 2. Architecture flag mismatch (minor, ~1%)

The DSL compiles kernels with `nvcc -arch=sm_86 -ptx`
(`macros/src/cuda_module.rs:104`). The benchmark scripts build the
hand-written binaries with plain `nvcc -lm` — CUDA 12.0's default arch is
**sm_52**, so at runtime the driver JIT-recompiles old compute_52 PTX for
the RTX 3070. Measured cost: ~1% (208.2 → 206.3 ms). Real but small; worth
fixing for cleanliness.

### 3. Run ordering / thermal drift (affects noise, not the mm anomaly)

`run_*.sh` executes all 30 rust reps, then all 30 cuda reps per size. GPU
boost clocks sag as the card heats up, so whichever implementation runs
later in a long sweep is systematically penalized. Observed directly: the
same `mm` binary measured 182.7 ms mean on a cool GPU and ~207 ms after
minutes of sustained load — a ~13% swing from clocks alone. This is the
likely source of the sign-flipping "wins" in julia/raytracer/nbodies, and it
adds variance everywhere. (It cannot explain mm, where rust ran *first* and
won consistently across all sizes — but it means small overheads like
raytracer's -8.3% should not be trusted.)

## Is the data incorrect?

- The *numbers* are genuine: both sides time real kernel executions. The
  Rust side even measures slightly more work per rep (PTX load + module JIT
  + launch + full stream synchronize vs the .cu binaries' pure `cudaEvent`
  kernel window).
- The *comparison* is invalid for `mm` (different device code) and
  noise-limited for julia/raytracer/nbodies (thermal ordering + short
  kernels). `nearest_neighbor` is the only sweep that is fair as-is.
- Latent correctness hazard, separate from benchmarking: an `int` counter
  compared against a `uint64_t` bound is undefined behavior / wrong past
  `n > 2^31 - 1`. The DSL currently generates code that does not honor the
  source's declared `u64` semantics. None of the current benchmark sizes
  trip this, but the GPU-capacity tiers in `BENCHMARK_GPU_CAPACITY.md`
  (e.g. nearest_neighbor at 556M records) are in the range where a
  data-dependent `int` loop *would* break.

## What to change

Note: `-arch=sm_86` alone is **not** sufficient to make comparisons fair —
it was only the ~1% effect. Fairness requires the generated device code to
be semantically identical to the handwritten kernels, i.e. the three codegen
fixes below.

1. **Fix loop counter types** — `compiler/src/codegen/cuda.rs:359`: emit
   the loop counter with the type declared in the Rust source (`0u64` →
   `uint64_t`) instead of hardcoded `int`. Expect mm's rust numbers to
   *rise* to match the hand-written kernel — that is the correct outcome,
   not a regression.
2. **Fix literal suffixes** — `compiler/src/codegen/cuda.rs:418-419`: map
   the Rust literal suffix to the C++ one (`1.5f32` → `1.5f`, `1.5f64` →
   `1.5`, integer suffixes → `ULL`/`U`/etc. as appropriate) instead of
   stripping it. This removes the hidden FP64 math from ripple/raytracer.
3. **Fix dropped `else` branches** — `gen_if` in
   `compiler/src/codegen/cuda.rs:282`: emit `else { ... }` (and handle
   `else if` chains, which syn represents as a nested `Expr::If` in
   `else_branch`). Pure correctness fix.
4. **Match nvcc flags in the benchmark scripts** — add `-arch=sm_86` (or
   `-arch=native`) to every `nvcc` invocation in `results/run_*.sh` so both
   sides are compiled for the actual GPU.
5. **Interleave reps (recommended)** — alternate rust/cuda within each size
   in `bench_runs.sh` sweeps (or add a warm-up period) so boost-clock drift
   hits both implementations equally, instead of 30-rust-then-30-cuda
   blocks.
6. **Re-run the sweeps** after 1-4. Expected result: small positive overhead
   everywhere — roughly +1-10% on short kernels (launch/sync/JIT amortizes
   poorly) shrinking toward ~0-2% on long kernels like mm 9000.

No changes are needed to the benchmark Rust sources
(`examples/src/benchmarks/*.rs`) — they were never the problem. The
defects are all in the DSL compiler's CUDA codegen.
