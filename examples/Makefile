VECTOR_SUM_SIZE  ?= 8000
MM_SIZE          ?= 8000
JULIA_SIZE       ?= 8000
RAYTRACER_SIZE   ?= 8000
RIPPLE_SIZE      ?= 8000
NBODIES_SIZE     ?= 4096
NEAREST_NEIGHBOR_SIZE ?= 100000

NVCC_ARCH ?= sm_86

# Debug build, for iterating on the compiler. Benchmarks use --release instead
# (see `bench` below) -- the CPU implementations are meaningless at opt-level 0.
build:
	RUSTFLAGS="-A warnings" cargo build

release:
	RUSTFLAGS="-A warnings" cargo build --release

CUDA_KERNELS = vector_sum mm julia raytracer ripple nbodies nearest_neighbor
CUDA_BINS = $(addprefix bin/,$(CUDA_KERNELS))

bin/%: src/benchmarks/cuda/%.cu
	@mkdir -p bin
	nvcc -lm -arch=$(NVCC_ARCH) $< -o $@

cuda: $(CUDA_BINS)

# One-off correctness/smoke pass: run each kernel once through both the DSL and
# the hand-written CUDA. For timings use `make bench` instead -- a single run
# is noise, and these sizes are not the benchmark tiers.
smoke: build $(CUDA_BINS)
	../target/debug/demo $(VECTOR_SUM_SIZE) $(MM_SIZE) $(JULIA_SIZE) $(RAYTRACER_SIZE) $(RIPPLE_SIZE)
	./bin/vector_sum $(VECTOR_SUM_SIZE)
	./bin/mm $(MM_SIZE)
	./bin/julia $(JULIA_SIZE)
	./bin/raytracer $(RAYTRACER_SIZE)
	./bin/ripple $(RIPPLE_SIZE)

run: build
	../target/debug/demo $(VECTOR_SUM_SIZE) $(MM_SIZE) $(JULIA_SIZE)

# Benchmarking entry points. Everything they do is configured in
# scripts/kernels.conf; pass IMPLS/RUNS/SIZES to override.
#
#   make bench                              every kernel, all three impls
#   make bench IMPLS=rust-gpu,cuda          DSL vs hand-written CUDA
#   make bench-julia IMPLS=rust-gpu,rust    DSL vs CPU, julia only
#   make bench-mm RUNS=5 SIZES="512 1024"
BENCH_FLAGS = $(if $(IMPLS),-i $(IMPLS)) $(if $(RUNS),-n $(RUNS)) $(if $(SIZES),-s "$(SIZES)")

bench:
	./scripts/run_all.sh $(BENCH_FLAGS)

bench-%:
	./scripts/run_bench.sh -k $* $(BENCH_FLAGS)

clean:
	rm -f *.cu *.h
	rm -rf bin

.PHONY: build release cuda smoke run bench clean
