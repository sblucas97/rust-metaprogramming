VECTOR_SUM_SIZE ?= 8000
MM_SIZE         ?= 8000
JULIA_SIZE      ?= 8000

build:
	RUSTFLAGS="-A warnings" cargo build

CUDA_BINS = /tmp/bench_vector_sum /tmp/bench_mm /tmp/bench_julia

/tmp/bench_%: src/benchmarks/cuda/%.cu
	nvcc $< -o $@

benchmark: build $(CUDA_BINS)
	./target/debug/rust-metaprogramming $(VECTOR_SUM_SIZE) $(MM_SIZE) $(JULIA_SIZE)
	/tmp/bench_vector_sum $(VECTOR_SUM_SIZE)
	/tmp/bench_mm $(MM_SIZE)
	/tmp/bench_julia $(JULIA_SIZE)

clean:
	rm -f *.cu *.h

run: build
	./target/debug/rust-metaprogramming $(VECTOR_SUM_SIZE) $(MM_SIZE) $(JULIA_SIZE)

