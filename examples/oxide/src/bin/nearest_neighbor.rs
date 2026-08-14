//! Nearest-neighbor (Euclidean distance) benchmark in cuda-oxide, mirroring
//! the rust-gpu DSL version in examples/src/benchmarks/nearest_neighbor.rs:
//!
//! - same data: deterministic LCG-generated (lat, lng) pairs in
//!   [-90, 90] x [-180, 180], same seed/formula as the DSL's `generate_locations`
//! - same math: one `sqrtf` per record against a fixed (lat, lng) query point
//! - same launch geometry: 128 threads/block, ceil(n / 128) blocks, 1D
//! - same measured region: device alloc + H2D upload, kernel, D2H readback
//!
//! This is the closest of the four ports to cuda-oxide's own `vecadd` example
//! in the crate docs: one read-only input slice, one `DisjointSlice` output,
//! `get_mut_indexed()` doing the bounds-checked index-and-resolve in one call.
//!
//! Prints the `[nearest_neighbor] elapsed: X ms` line bench_runs.sh parses.
//!
//! Build with `cargo oxide build` (not plain cargo); run as
//! `nearest_neighbor <num_records>`.

use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_module;

/// Deterministic pseudo-random (lat, lng) pairs, matching the DSL's
/// `generate_locations` LCG stream exactly.
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

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel};

    /// One record per thread. `distances.get_mut_indexed()` mints the 1D
    /// thread index and resolves it to this thread's output slot in one
    /// bounds-checked call; `locations` stays a plain read-only slice since
    /// every thread reads a different pair but writes nothing there.
    #[kernel]
    pub fn euclid(locations: &[f32], mut distances: DisjointSlice<f32>, lat: f32, lng: f32) {
        if let Some((dist, idx)) = distances.get_mut_indexed() {
            let i = idx.get();
            let dlat: f32 = lat - locations[2 * i];
            let dlng: f32 = lng - locations[2 * i + 1];
            *dist = (dlat * dlat + dlng * dlng).sqrt();
        }
    }
}

fn main() {
    let num_records: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let lat: f32 = 30.0;
    let lng: f32 = 90.0;

    let locations = generate_locations(num_records);
    let zeros = vec![0.0f32; num_records];

    let threads_per_block: u32 = 128;
    let num_blocks: u32 = (num_records as u32 + threads_per_block - 1) / threads_per_block;

    let start = Instant::now();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let d_locations = DeviceBuffer::from_host(&stream, &locations).expect("H2D upload");
    let mut d_distances = DeviceBuffer::from_host(&stream, &zeros).expect("H2D upload");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // SAFETY: 1D launch of num_blocks * threads_per_block threads over
    // num_records records; each thread resolves a unique output slot through
    // `distances`'s witness, in-grid threads beyond num_records mint no
    // witness.
    unsafe {
        module.euclid(
            &stream,
            LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            },
            &d_locations,
            &mut d_distances,
            lat,
            lng,
        )
    }
    .expect("Kernel launch failed");

    let result = d_distances.to_host_vec(&stream).expect("D2H readback");

    let elapsed = start.elapsed();
    println!(
        "[nearest_neighbor] elapsed: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    // Light sanity check so a silently-failed launch can't masquerade as a
    // fast run: every distance must be finite (uninitialized/garbage reads
    // off the device tend to show up as NaN/Inf, not as a plausible value).
    let bad = result.iter().filter(|d| !d.is_finite()).count();
    if bad != 0 {
        eprintln!("ERROR: {bad} of {} distances were not finite", result.len());
        std::process::exit(1);
    }
}
