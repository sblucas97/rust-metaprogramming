use rayon::prelude::*;

use std::time::Instant;

// Mirrors the DSL kernel in benchmarks::julia exactly (same constants, same
// escape logic, same RGBA layout) so timings are directly comparable. Only the
// compute is timed, matching the DSL side which times just the spawn!.
pub fn run(dim: usize) -> Vec<f32> {
    let mut pixels = vec![0.0_f32; dim * dim * 4];

    let start = Instant::now();
    pixels
        .par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(offset, pixel)| {
            let x = offset % dim;
            let y = offset / dim;

            let scale = 0.1_f32;

            let jx = scale * (dim - x) as f32 / dim as f32;
            let jy = scale * (dim - y) as f32 / dim as f32;

            let cr = -0.8_f32;
            let ci = 0.156_f32;

            let mut ar = jx;
            let mut ai = jy;

            let mut julia_value = 1.0_f32;
            let mut escaped = false;

            for _ in 0..200 {
                if !escaped {
                    let nar = ar * ar - ai * ai + cr;
                    let nai = 2.0_f32 * ar * ai + ci;

                    if nar * nar + nai * nai > 1000.0_f32 {
                        julia_value = 0.0_f32;
                        escaped = true;
                    }

                    if !escaped {
                        ar = nar;
                        ai = nai;
                    }
                }
            }

            pixel[0] = 255.0_f32 * julia_value; // Red
            pixel[1] = 0.0_f32; // Green
            pixel[2] = 0.0_f32; // Blue
            pixel[3] = 255.0_f32; // Alpha
        });
    let elapsed = start.elapsed();
    println!(
        "[julia_cpu] elapsed: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    pixels
}
