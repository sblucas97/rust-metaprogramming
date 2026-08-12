use rayon::prelude::*;

use std::time::Instant;

// Mirrors the DSL kernel in benchmarks::mm: square m x m matrices filled with
// (i % 100 + 1), naive triple loop with no tiling or blocking so the CPU does
// the same arithmetic the kernel does. One rayon task per output row. Only the
// compute is timed, matching the DSL side which times just the spawn!.
pub fn run(m: usize) -> Vec<f32> {
    let n = m;
    let k = m;

    let a: Vec<f32> = (0..m * n).map(|i| (i % 100 + 1) as f32).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i % 100 + 1) as f32).collect();
    let mut c = vec![0.0_f32; m * k];

    let start = Instant::now();
    c.par_chunks_exact_mut(k)
        .enumerate()
        .for_each(|(row, c_row)| {
            for (col, out) in c_row.iter_mut().enumerate() {
                let mut sum = 0.0_f32;
                for i in 0..n {
                    sum += a[row * n + i] * b[i * k + col];
                }
                *out = sum;
            }
        });
    let elapsed = start.elapsed();
    println!("[mm_cpu] elapsed: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    c
}
