mod benchmarks;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);

    // benchmarks::run_timed("vector_sum", || benchmarks::vector_sum::run(size));
    // benchmarks::run_timed("matrix_mult", || benchmarks::mm::run(size));
    benchmarks::vector_sum::run(size);
    benchmarks::mm::run(size);
    benchmarks::julia::run(size);
    benchmarks::raytracer::run(size);
    benchmarks::ripple::run(size);
    // benchmarks::run_timed("raytracer", || benchmarks::raytracer::run(size));
    // benchmarks::run_timed("ripple", || benchmarks::ripple::run(size));
}