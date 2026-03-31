mod benchmarks;

fn main() {
    benchmarks::run_timed("vector_sum", || benchmarks::vector_sum_k::run(8000));
}