use rayon::prelude::*;

use std::time::Instant;

use crate::benchmarks::nearest_neighbor::generate_locations;

// Mirrors the DSL kernel in benchmarks::nearest_neighbor: same lat/lng, same
// locations from the shared generator, same euclidean distance per record. This
// is the parallel form of `nearest_neighbor::cpu_reference`, which stays serial
// because it is the correctness oracle rather than a timed implementation. Only
// the compute is timed, matching the DSL side which times just the spawn!.
pub fn run(num_records: usize) -> Vec<f32> {
    let lat: f32 = 30.0;
    let lng: f32 = 90.0;

    let locations = generate_locations(num_records);
    let mut distances = vec![0.0_f32; num_records];

    let start = Instant::now();
    distances
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, distance)| {
            let dlat = lat - locations[2 * i];
            let dlng = lng - locations[2 * i + 1];
            *distance = (dlat * dlat + dlng * dlng).sqrt();
        });
    let elapsed = start.elapsed();
    println!(
        "[nearest_neighbor_cpu] elapsed: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    distances
}
