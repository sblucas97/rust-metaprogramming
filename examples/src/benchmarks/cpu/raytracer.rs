use std::time::Instant;

use crate::benchmarks::raytracer::generate_spheres;

// Mirrors the DSL kernel in benchmarks::raytracer exactly (same 20 spheres from
// the shared generator, same hit test, same RGBA layout) so timings are directly
// comparable. Single threaded: one pixel after another, no parallelism. Only the
// compute is timed, matching the DSL side which times just the spawn!.
pub fn run(dim: usize) -> Vec<f32> {
    let spheres = generate_spheres(dim);
    
    let start = Instant::now();

    let mut image = vec![0.0_f32; dim * dim * 4];

    let width = dim;
    let height = dim;

    for offset in 0..(width * height) {
        let x = offset % width;
        let y = offset / width;

        let ox = x as f32 - width as f32 / 2.0_f32;
        let oy = y as f32 - height as f32 / 2.0_f32;

        let mut r = 0.0_f32;
        let mut g = 0.0_f32;
        let mut b = 0.0_f32;
        let mut maxz = -99999.0_f32;

        for i in 0..20 {
            let sphere_radius = spheres[i * 7 + 3];
            let dx = ox - spheres[i * 7 + 4];
            let dy = oy - spheres[i * 7 + 5];
            let n;
            let t;

            if (dx * dx + dy * dy) < (sphere_radius * sphere_radius) {
                let dz = (sphere_radius * sphere_radius - dx * dx - dy * dy).sqrt();
                n = dz / (sphere_radius * sphere_radius).sqrt();
                t = dz + spheres[i * 7 + 6];
            } else {
                t = -99999.0_f32;
                n = 0.0_f32;
            }

            if t > maxz {
                let fscale = n;
                r = spheres[i * 7] * fscale;
                g = spheres[i * 7 + 1] * fscale;
                b = spheres[i * 7 + 2] * fscale;
                maxz = t;
            }
        }

        image[offset * 4] = r * 255.0_f32;
        image[offset * 4 + 1] = g * 255.0_f32;
        image[offset * 4 + 2] = b * 255.0_f32;
        image[offset * 4 + 3] = 255.0_f32;
    }
    let elapsed = start.elapsed();
    println!(
        "[raytracer_cpu] elapsed: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    image
}
