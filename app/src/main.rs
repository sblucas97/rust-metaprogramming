mod benchmarks;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vs_size: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);
    let mm_size: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);
    let julia_size: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);

    benchmarks::run_timed("vector_sum", || benchmarks::vector_sum::run(vs_size));
    benchmarks::run_timed("matrix_mult", || benchmarks::mm::run(mm_size));
    benchmarks::run_timed("julia", || benchmarks::julia::run(julia_size));
}