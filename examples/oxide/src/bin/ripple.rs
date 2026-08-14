//! Ripple benchmark in cuda-oxide, mirroring the rust-gpu DSL version in
//! examples/src/benchmarks/ripple.rs as closely as the API allows:
//!
//! - same math: cosine ripple pattern, one grey value per pixel, f32 throughout
//! - same launch geometry: a (dim x dim) grid of single-thread blocks, so
//!   `blockIdx.{x,y}` alone (thread 0 of a 1-thread block) is the pixel
//!   coordinate, exactly as the DSL kernel reads it
//! - same buffer: dim*dim RGBA pixels of f32 (one float4-layout `Rgba` per
//!   pixel here, as in the julia port, instead of 4 flat f32)
//! - same measured region: device alloc + H2D upload, kernel, D2H readback
//!
//! Prints the `[ripple] elapsed: X ms` line bench_runs.sh parses.
//!
//! Build with `cargo oxide build` (not plain cargo); run as `ripple <dim>`.

use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_module;

/// One RGBA pixel with CUDA `float4` layout, as in the julia port.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Rgba(pub [f32; 4]);

unsafe impl cuda_core::DeviceCopy for Rgba {}

#[cuda_module]
mod kernels {
    use super::Rgba;
    use cuda_device::thread::Runtime2DIndex;
    use cuda_device::{DisjointSlice, kernel, thread};

    /// One pixel per thread, launched as a (dim x dim) grid of 1x1 blocks:
    /// `index_2d_runtime`'s `blockIdx * blockDim + threadIdx` formula reduces
    /// to plain `blockIdx` here, matching the DSL kernel's `x = blockIdx.x`.
    /// As in the julia port, `dim` needs no separate parameter: it lives in
    /// the slice's runtime row width.
    #[kernel]
    pub fn ripple(mut out: DisjointSlice<Rgba, Runtime2DIndex>, ticks: f32) {
        if let Some(idx) = thread::index_2d_runtime(&out) {
            let x: u32 = idx.col() as u32;
            let y: u32 = idx.row() as u32;
            let dim_f: f32 = out.row_width() as f32;

            let fx: f32 = 0.5f32 * x as f32 - dim_f / 15.0f32;
            let fy: f32 = 0.5f32 * y as f32 - dim_f / 15.0f32;
            let d: f32 = (fx * fx + fy * fy).sqrt();
            let grey: f32 =
                (128.0f32 + 127.0f32 * (d / 10.0f32 - ticks / 7.0f32).cos() / (d / 10.0f32 + 1.0f32))
                    .floor();

            if let Some(px) = out.get_mut(idx) {
                *px = Rgba([grey, grey, grey, 255.0]);
            }
        }
    }
}

fn main() {
    let dim: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let ticks: f32 = 10.0;
    let data: Vec<Rgba> = vec![Rgba([0.0; 4]); dim * dim];

    let start = Instant::now();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let mut image = DeviceBuffer::from_host(&stream, &data).expect("H2D upload");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // SAFETY: 2D launch of dim x dim single-thread blocks; every thread
    // resolves its own (row, col) cell through the slice's witness, in-grid
    // threads beyond the image mint no witness.
    unsafe {
        module.ripple(
            &stream,
            LaunchConfig {
                grid_dim: (dim as u32, dim as u32, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            cuda_host::RowWidth::new(&mut image, dim as u32),
            ticks,
        )
    }
    .expect("Kernel launch failed");

    let result = image.to_host_vec(&stream).expect("D2H readback");

    let elapsed = start.elapsed();
    println!("[ripple] elapsed: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    // Light sanity check so a silently-failed launch can't masquerade as a
    // fast run: every pixel must have been written (alpha = 255).
    let unwritten = result.iter().filter(|px| px.0[3] != 255.0).count();
    if unwritten != 0 {
        eprintln!("ERROR: {unwritten} of {} pixels were never written", result.len());
        std::process::exit(1);
    }
}
