use std::time::Instant;

use crate::benchmarks::nbodies::{DT, FLOATS_PER_BODY, SOFTENING, STEPS, generate_bodies};

// Mirrors the DSL kernels in benchmarks::nbodies: same bodies from the shared
// generator, same STEPS iterations of (force pass, integrate pass), same
// interleaved pos/vel layout. Single threaded, one body after another. Only the
// compute is timed, matching the DSL side which times just the spawn!s.
//
// The force pass reads every body's position while writing its own velocity.
// On the GPU that aliasing is fine because the two halves of the record never
// overlap; in safe Rust it needs a read-only snapshot of the buffer, which
// costs an O(n) copy per step against O(n^2) of force computation.
pub fn run(n: usize) -> Vec<f32> {
    let start = Instant::now();

    let mut p = generate_bodies(n);
    
    for _ in 0..STEPS {
        let snapshot = p.clone();
        for i in 0..n {
            let mut fx = 0.0_f32;
            let mut fy = 0.0_f32;
            let mut fz = 0.0_f32;
            for j in 0..n {
                let dx = snapshot[6 * j] - snapshot[6 * i];
                let dy = snapshot[6 * j + 1] - snapshot[6 * i + 1];
                let dz = snapshot[6 * j + 2] - snapshot[6 * i + 2];
                let dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                let inv_dist = 1.0_f32 / dist_sqr.sqrt();
                let inv_dist3 = inv_dist * inv_dist * inv_dist;
                fx += dx * inv_dist3;
                fy += dy * inv_dist3;
                fz += dz * inv_dist3;
            }
            p[6 * i + 3] += DT * fx;
            p[6 * i + 4] += DT * fy;
            p[6 * i + 5] += DT * fz;
        }

        for i in 0..n {
            p[6 * i] += p[6 * i + 3] * DT;
            p[6 * i + 1] += p[6 * i + 4] * DT;
            p[6 * i + 2] += p[6 * i + 5] * DT;
        }
    }
    let elapsed = start.elapsed();
    println!(
        "[nbodies_cpu] elapsed: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    p
}
