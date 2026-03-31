VECTOR_SUM_SIZE  ?= 8000
MM_SIZE          ?= 8000
JULIA_SIZE       ?= 8000
RAYTRACER_SIZE   ?= 8000
RIPPLE_SIZE      ?= 8000

build:
	RUSTFLAGS="-A warnings" cargo build

CUDA_BINS = /tmp/bench_vector_sum /tmp/bench_mm /tmp/bench_julia /tmp/bench_raytracer /tmp/bench_ripple

/tmp/bench_%: src/benchmarks/cuda/%.cu
	nvcc -lm $< -o $@

benchmark: build $(CUDA_BINS)
	./target/debug/rust-metaprogramming $(VECTOR_SUM_SIZE) $(MM_SIZE) $(JULIA_SIZE) $(RAYTRACER_SIZE) $(RIPPLE_SIZE)
	/tmp/bench_vector_sum $(VECTOR_SUM_SIZE)
	/tmp/bench_mm $(MM_SIZE)
	/tmp/bench_julia $(JULIA_SIZE)
	/tmp/bench_raytracer $(RAYTRACER_SIZE)
	/tmp/bench_ripple $(RIPPLE_SIZE)

clean:
	rm -f *.cu *.h

run: build
	./target/debug/rust-metaprogramming $(VECTOR_SUM_SIZE) $(MM_SIZE) $(JULIA_SIZE)

