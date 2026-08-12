use rayon::prelude::*;

use std::time::Instant;

// Mirrors the DSL kernel in benchmarks::ripple exactly (same constants, same
// greyscale formula, same RGBA layout) so timings are directly comparable. Only
// the compute is timed, matching the DSL side which times just the spawn!.
pub fn run(dim: usize) -> Vec<f32> {
    let mut pixels = vec![0.0_f32; dim * dim * 4];
    let ticks = 10.0_f32;

    let start = Instant::now();
    pixels
        .par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(offset, pixel)| {
            let x = offset % dim;
            let y = offset / dim;

            let dim_f = dim as f32;
            let fx = 0.5_f32 * x as f32 - dim_f / 15.0_f32;
            let fy = 0.5_f32 * y as f32 - dim_f / 15.0_f32;
            let d = (fx * fx + fy * fy).sqrt();
            let grey =
                (128.0_f32 + 127.0_f32 * (d / 10.0_f32 - ticks / 7.0_f32).cos() / (d / 10.0_f32 + 1.0_f32))
                    .floor();

            pixel[0] = grey;
            pixel[1] = grey;
            pixel[2] = grey;
            pixel[3] = 255.0_f32;
        });
    let elapsed = start.elapsed();
    println!(
        "[ripple_cpu] elapsed: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    pixels
}
