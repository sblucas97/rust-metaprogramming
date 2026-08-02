VECTOR_SUM_SIZE  ?= 8000
MM_SIZE          ?= 8000
JULIA_SIZE       ?= 8000
RAYTRACER_SIZE   ?= 8000
RIPPLE_SIZE      ?= 8000
NBODIES_SIZE     ?= 4096
NEAREST_NEIGHBOR_SIZE ?= 100000

build:
	RUSTFLAGS="-A warnings" cargo build

CUDA_BINS = /tmp/bench_vector_sum /tmp/bench_mm /tmp/bench_julia /tmp/bench_raytracer /tmp/bench_ripple /tmp/bench_nbodies /tmp/bench_nearest_neighbor

/tmp/bench_%: src/benchmarks/cuda/%.cu
	nvcc -lm $< -o $@

benchmark: build $(CUDA_BINS)
	../target/debug/demo $(VECTOR_SUM_SIZE) $(MM_SIZE) $(JULIA_SIZE) $(RAYTRACER_SIZE) $(RIPPLE_SIZE)
	/tmp/bench_vector_sum $(VECTOR_SUM_SIZE)
	/tmp/bench_mm $(MM_SIZE)
	/tmp/bench_julia $(JULIA_SIZE)
	/tmp/bench_raytracer $(RAYTRACER_SIZE)
	/tmp/bench_ripple $(RIPPLE_SIZE)

clean:
	rm -f *.cu *.h

run: build
	../target/debug/demo $(VECTOR_SUM_SIZE) $(MM_SIZE) $(JULIA_SIZE)

# Run only the nbodies benchmark (Rust DSL) and its hand-written CUDA
# equivalent at the same N, for a direct timing comparison.
nbodies: build /tmp/bench_nbodies
	../target/debug/demo nbodies $(NBODIES_SIZE)
	/tmp/bench_nbodies $(NBODIES_SIZE)

# Run only the nearest_neighbor benchmark (Rust DSL) and its hand-written CUDA
# equivalent at the same N, for a direct timing comparison.
nearest_neighbor: build /tmp/bench_nearest_neighbor
	../target/debug/demo nearest_neighbor $(NEAREST_NEIGHBOR_SIZE)
	/tmp/bench_nearest_neighbor $(NEAREST_NEIGHBOR_SIZE)

