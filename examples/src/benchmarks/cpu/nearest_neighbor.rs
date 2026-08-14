use std::time::Instant;

use crate::benchmarks::nearest_neighbor::generate_locations;

// Mirrors the DSL kernel in benchmarks::nearest_neighbor: same lat/lng, same
// locations from the shared generator, same euclidean distance per record.
// Single threaded, no parallelism -- same arithmetic as the correctness oracle
// `nearest_neighbor::cpu_reference`, but timed. Only the compute is timed,
// matching the DSL side which times just the spawn!.
pub fn run(num_records: usize) -> Vec<f32> {
    let lat: f32 = 30.0;
    let lng: f32 = 90.0;
    let locations = generate_locations(num_records);
    
    let start = Instant::now();

    let mut distances = vec![0.0_f32; num_records];

    for i in 0..num_records {
        let dlat = lat - locations[2 * i];
        let dlng = lng - locations[2 * i + 1];
        distances[i] = (dlat * dlat + dlng * dlng).sqrt();
    }
    let elapsed = start.elapsed();
    println!(
        "[nearest_neighbor_cpu] elapsed: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    distances
}
