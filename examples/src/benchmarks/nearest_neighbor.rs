use std::time::Instant;

use macros::{cuda_module, spawn};
use runtime::CudaVec;

/// Deterministic pseudo-random coordinates so the CPU reference is reproducible.
fn generate_locations(num_records: usize) -> Vec<f32> {
    let mut state: u32 = 7919;
    let mut locations = Vec::with_capacity(num_records * 2);
    for _ in 0..num_records {
        for range in [180.0f32, 360.0f32] {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = (state >> 8) % 32767;
            locations.push(v as f32 / 32767.0 * range - range / 2.0);
        }
    }
    locations
}

fn cpu_reference(locations: &[f32], lat: f32, lng: f32) -> Vec<f32> {
    locations
        .chunks_exact(2)
        .map(|loc| ((lat - loc[0]) * (lat - loc[0]) + (lng - loc[1]) * (lng - loc[1])).sqrt())
        .collect()
}

pub fn run(num_records: usize) -> CudaVec<f32> {
    let lat: f32 = 30.0;
    let lng: f32 = 90.0;

    let locations = generate_locations(num_records);
    let d_locations: CudaVec<f32> = CudaVec::new(locations.clone());
    let mut d_distances: CudaVec<f32> = CudaVec::new(vec![0.0f32; num_records]);

    let threads_per_block: u32 = 128;
    let num_blocks: u32 = (num_records as u32 + threads_per_block - 1) / threads_per_block;

    let start = Instant::now();
    spawn!(
        nearest_neighbor_kernel::euclid,
        (num_blocks, 1, 1),
        (threads_per_block, 1, 1),
        d_locations,
        d_distances,
        num_records as u64,
        lat,
        lng
    );
    let elapsed = start.elapsed();
    println!("[nearest_neighbor] elapsed: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    d_distances.copy_from_device();

    let expected = cpu_reference(&locations, lat, lng);
    let ok = d_distances
        .as_slice()
        .iter()
        .zip(expected.iter())
        .all(|(gpu, cpu)| (gpu - cpu).abs() <= 1e-3 * cpu.abs().max(1.0));
    println!(
        "nearest_neighbor: {} ({} records)",
        if ok { "PASS" } else { "FAIL" },
        num_records
    );

    d_distances
}

#[cuda_module]
pub mod nearest_neighbor_kernel {
    use runtime::CudaVec;
    use macros::kernel;

    #[kernel]
    pub fn euclid(
        d_locations: &CudaVec<f32>,
        d_distances: &mut CudaVec<f32>,
        num_records: u64,
        lat: f32,
        lng: f32,
    ) {
        let global_id: u64 = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
        let ilat: u64 = 2 * global_id;
        let ilng: u64 = 2 * global_id + 1;
        if global_id < num_records {
            let dlat: f32 = lat - d_locations[ilat];
            let dlng: f32 = lng - d_locations[ilng];
            d_distances[global_id] = sqrtf(dlat * dlat + dlng * dlng);
        }
    }
}
