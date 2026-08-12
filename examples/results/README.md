# Benchmark results

Every batch of runs lands in a numbered version directory. `latest` is a symlink
to the newest one.

```
results/
  v0/
    meta.json          # provenance for the whole batch
    summary.txt        # combined report across every kernel in the batch
    julia/
      runs.csv         # one row per individual run
      summary.txt      # report for this kernel alone
    mm/ ...
  v1/ ...
  latest -> v1
```

## Producing a batch

Nothing here is written by hand. The runner lives in `../scripts`:

```bash
../scripts/run_bench.sh -k julia                        # all three impls
../scripts/run_bench.sh -k julia -i rust-gpu,cuda       # DSL vs hand-written CUDA
../scripts/run_bench.sh -k julia -i rust-gpu,rust -n 10 # DSL vs CPU, 10 reps
../scripts/run_all.sh -i rust-gpu,cuda                  # every kernel, one batch
```

Each invocation allocates the next free `vN` unless `--version vN` (or
`--version latest`) says otherwise, updates `latest`, and regenerates both the
per-kernel and the combined `summary.txt`. `--dry-run` prints the resolved plan
without running anything. Sizes and repetition counts default from
`../scripts/kernels.conf`, which is the only place to change them.

## Reading a batch

```bash
../scripts/analyze_runs.py results/latest              # combined report
../scripts/analyze_runs.py results/latest/julia        # one kernel
../scripts/analyze_runs.py results/v2 results/v3 -k julia   # across versions
```

## Implementation labels

| label      | what it is                            | how it's invoked        |
| ---------- | ------------------------------------- | ----------------------- |
| `rust-gpu` | the Rust DSL, via the custom compiler | `demo <kernel> <n>`     |
| `rust`     | pure Rust on the CPU, rayon           | `demo <kernel>_cpu <n>` |
| `cuda`     | the hand-written `.cu`                | `bin/<kernel> <n>`      |

## Conventions these numbers depend on

- **Timing scope.** Every implementation times the compute only. On the GPU side
  that is the `spawn!`, which blocks until the kernel finishes; host allocation
  and the H2D/D2H transfers are outside the measurement. On the CPU side it is
  the rayon loop, with input generation outside.
- **Build profile.** `--release`. In a debug build the CPU implementations run at
  `opt-level = 0` and are 10-50x slower than they should be, which makes any
  comparison against them meaningless.
- **Interleaving.** Implementations alternate inside the repetition loop, so
  clock and load drift over a long session hits all of them equally.
- **Sizes.** A run that includes the `rust` (CPU) implementation uses the
  `CPU_SIZES` list for *every* implementation in that run, because the analyzer
  only compares implementations that ran at the same size.

## Caveats on v0 and v1

Both predate this layout and were migrated into it:

- The `impl` column originally said `rust` for the DSL. It has been rewritten to
  `rust-gpu`, since `rust` now means the CPU implementation.
- Neither has the `version` column that newer CSVs carry.
- Both were collected from a debug build, so they carry no usable CPU numbers
  (there were no CPU implementations yet).
- v1's julia timings include the H2D/D2H transfers; from v2 onward julia times
  the kernel alone like every other benchmark. v1 julia numbers are therefore
  not comparable with later versions.
