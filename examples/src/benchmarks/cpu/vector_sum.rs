use std::time::Instant;

// Mirrors the DSL kernel in benchmarks::vector_sum: same inputs (1..=n as f32),
// same elementwise add. Single threaded, no parallelism. Only the compute is
// timed, matching the DSL side which times just the spawn!.
pub fn run(n: usize) -> Vec<f32> {
    let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let mut result = vec![0.0_f32; n];

    let start = Instant::now();
    result
        .iter_mut()
        .enumerate()
        .for_each(|(i, out)| *out = a[i] + b[i]);
    let elapsed = start.elapsed();
    println!(
        "[vector_sum_cpu] elapsed: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    result
}
