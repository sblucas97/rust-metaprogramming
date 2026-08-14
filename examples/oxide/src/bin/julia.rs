//! Julia-set benchmark in cuda-oxide, mirroring the rust-gpu DSL version in
//! examples/src/benchmarks/julia.rs as closely as the API allows:
//!
//! - same math: 200 fixed iterations with an escape flag, f32 throughout
//! - same launch geometry: a (dim x dim) grid of single-thread blocks
//! - same buffer: dim*dim RGBA pixels of f32 (here one float4-layout `Rgba`
//!   per pixel instead of 4 flat f32, so each thread owns exactly one element
//!   of the DisjointSlice)
//! - same measured region: device alloc + H2D upload, kernel, D2H readback
//!   (context/module init lands inside it, as it does for the DSL demo, whose
//!   first CUDA call is the timed CudaVec::new)
//!
//! Prints the `[julia] elapsed: X ms` line bench_runs.sh parses.
//!
//! Build with `cargo oxide build` (not plain cargo); run as `julia <dim>`.

use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_module;

/// One RGBA pixel with CUDA `float4` layout: 16-byte aligned, moved between
/// host and device as a single 128-bit value.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Rgba(pub [f32; 4]);

// Plain POD aggregate, no pointers: safe to memcpy to/from the device.
unsafe impl cuda_core::DeviceCopy for Rgba {}

#[cuda_module]
mod kernels {
    use super::Rgba;
    use cuda_device::thread::Runtime2DIndex;
    use cuda_device::{DisjointSlice, kernel, thread};

    /// One pixel per thread. The slice's runtime row width carries `dim`, so
    /// the kernel needs no separate size parameter: a witness only exists for
    /// in-grid (col < dim) threads, and the slice length bounds the rows.
    #[kernel]
    pub fn julia(mut out: DisjointSlice<Rgba, Runtime2DIndex>) {
        let dim = out.row_width();
        if let Some(idx) = thread::index_2d_runtime(&out) {
            let x = thread::index_2d_col() as u32;
            let y = thread::index_2d_row() as u32;

            let scale: f32 = 0.1;
            let jx: f32 = scale * (dim - x) as f32 / dim as f32;
            let jy: f32 = scale * (dim - y) as f32 / dim as f32;
            let cr: f32 = -0.8;
            let ci: f32 = 0.156;

            let mut ar: f32 = jx;
            let mut ai: f32 = jy;
            let mut julia_value: f32 = 1.0;
            let mut escaped = false;
            // Fixed 200 iterations with an escape flag rather than `break`,
            // matching the DSL kernel's control flow exactly.
            for _ in 0..200u32 {
                if !escaped {
                    let nar: f32 = ((ar * ar) - (ai * ai)) + cr;
                    let nai: f32 = ((ai * ar) + (ar * ai)) + ci;
                    if ((nar * nar) + (nai * nai)) > 1000.0 {
                        julia_value = 0.0;
                        escaped = true;
                    }
                    if !escaped {
                        ar = nar;
                        ai = nai;
                    }
                }
            }

            if let Some(px) = out.get_mut(idx) {
                *px = Rgba([255.0 * julia_value, 0.0, 0.0, 255.0]);
            }
        }
    }
}

fn main() {
    let dim: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    // Host-side zeroed image, same bytes as the DSL version's
    // vec![0.0f32; dim * dim * 4].
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
        module.julia(
            &stream,
            LaunchConfig {
                grid_dim: (dim as u32, dim as u32, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            cuda_host::RowWidth::new(&mut image, dim as u32),
        )
    }
    .expect("Kernel launch failed");

    let result = image.to_host_vec(&stream).expect("D2H readback");

    let elapsed = start.elapsed();
    println!("[julia] elapsed: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    // Light sanity check so a silently-failed launch can't masquerade as a
    // fast run: every pixel must have been written (alpha = 255).
    let unwritten = result.iter().filter(|px| px.0[3] != 255.0).count();
    if unwritten != 0 {
        eprintln!("ERROR: {unwritten} of {} pixels were never written", result.len());
        std::process::exit(1);
    }
}
