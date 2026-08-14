//! N-bodies benchmark in cuda-oxide, mirroring the rust-gpu DSL version in
//! examples/src/benchmarks/nbodies.rs as closely as the API allows.
//!
//! # Layout: AoS -> SoA
//!
//! The DSL version packs each body as 6 interleaved f32s (pos.xyz, vel.xyz)
//! in one `CudaVec<f32>` and indexes it by hand (`p[6*i]`, `p[6*i+3]`, ...).
//! `gpu_n_bodies` there reads *every* body's position while writing only its
//! own velocity, and `gpu_integrate` reads its own velocity while writing
//! only its own position -- both against the same interleaved buffer.
//!
//! cuda-oxide's ownership model doesn't have a way to say "this buffer is
//! read-only here except for a disjoint per-thread slice I also get to
//! write": a `DisjointSlice` claims exclusive access to the whole buffer for
//! the launch, and a `&[T]` claims shared read access to the whole buffer, and
//! the two can't overlap. The natural fix is the one SoA split already
//! implies: positions and velocities become two separate device buffers, so
//! each kernel takes the one it mutates as a `DisjointSlice` and the other as
//! a plain read-only `&[T]`. That is a genuine layout change; the physics, the
//! per-step order (force then integrate), the RNG stream, and the resulting
//! numbers are identical to the DSL version.
//!
//! - same math: same pairwise force loop, same softening, same integration,
//!   f32 throughout
//! - same launch geometry: 128 threads/block, ceil(n / 128) blocks, 1D, 3 steps
//! - same measured region: device alloc + H2D upload, kernels, D2H readback
//!
//! Prints the `[nbodies] elapsed: X ms` line bench_runs.sh parses.
//!
//! Build with `cargo oxide build` (not plain cargo); run as `nbodies <n>`.

use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_module;

const DT: f32 = 0.01;
const SOFTENING: f32 = 1e-9;
const STEPS: usize = 3;

/// A 3-vector, used for both positions and velocities.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vec3(pub [f32; 3]);

unsafe impl cuda_core::DeviceCopy for Vec3 {}

/// Deterministic pseudo-random positions in [-1, 1], matching the DSL's
/// `generate_bodies` LCG stream exactly (it advances state 3 times per body,
/// once per position component, leaving velocity untouched -- so a separate
/// zeroed velocity buffer reproduces its behavior exactly).
fn generate_positions(n: usize) -> Vec<Vec3> {
    let mut state: u32 = 42;
    let mut positions = vec![Vec3([0.0; 3]); n];
    for pos in positions.iter_mut() {
        for k in 0..3 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            pos.0[k] = ((state >> 8) % 32767) as f32 / 32767.0 * 2.0 - 1.0;
        }
    }
    positions
}

#[cuda_module]
mod kernels {
    use super::Vec3;
    use cuda_device::{DisjointSlice, kernel};

    /// Accumulates this step's pairwise gravitational force into this
    /// thread's own velocity. `pos` is read in full (every other body), `vel`
    /// is written only at this thread's own index.
    #[kernel]
    pub fn gpu_n_bodies(
        pos: &[Vec3],
        mut vel: DisjointSlice<Vec3>,
        dt: f32,
        softening: f32,
    ) {
        if let Some((v, idx)) = vel.get_mut_indexed() {
            let i: usize = idx.get();
            let pi: Vec3 = pos[i];

            let mut fx: f32 = 0.0f32;
            let mut fy: f32 = 0.0f32;
            let mut fz: f32 = 0.0f32;
            for j in 0..pos.len() {
                let pj: Vec3 = pos[j];
                let dx: f32 = pj.0[0] - pi.0[0];
                let dy: f32 = pj.0[1] - pi.0[1];
                let dz: f32 = pj.0[2] - pi.0[2];
                let dist_sqr: f32 = dx * dx + dy * dy + dz * dz + softening;
                let inv_dist: f32 = 1.0f32 / dist_sqr.sqrt();
                let inv_dist3: f32 = inv_dist * inv_dist * inv_dist;
                fx = fx + dx * inv_dist3;
                fy = fy + dy * inv_dist3;
                fz = fz + dz * inv_dist3;
            }

            v.0[0] = v.0[0] + dt * fx;
            v.0[1] = v.0[1] + dt * fy;
            v.0[2] = v.0[2] + dt * fz;
        }
    }

    /// Applies this thread's own velocity to its own position. `vel` is read
    /// only at this thread's own index, `pos` is written only there too.
    #[kernel]
    pub fn gpu_integrate(mut pos: DisjointSlice<Vec3>, vel: &[Vec3], dt: f32) {
        if let Some((p, idx)) = pos.get_mut_indexed() {
            let i: usize = idx.get();
            let vi: Vec3 = vel[i];
            p.0[0] = p.0[0] + vi.0[0] * dt;
            p.0[1] = p.0[1] + vi.0[1] * dt;
            p.0[2] = p.0[2] + vi.0[2] * dt;
        }
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let positions = generate_positions(n);
    let velocities = vec![Vec3([0.0; 3]); n];

    let threads_per_block: u32 = 128;
    let num_blocks: u32 = (n as u32 + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = Instant::now();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let mut pos_buf = DeviceBuffer::from_host(&stream, &positions).expect("H2D upload");
    let mut vel_buf = DeviceBuffer::from_host(&stream, &velocities).expect("H2D upload");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    for _ in 0..STEPS {
        // SAFETY: 1D launch of num_blocks * threads_per_block threads over n
        // bodies; each thread resolves a unique element through the written
        // slice's witness (velocity here, position in the second call),
        // reading the other buffer in full as a plain shared slice.
        unsafe {
            module
                .gpu_n_bodies(&stream, cfg, &pos_buf, &mut vel_buf, DT, SOFTENING)
                .expect("gpu_n_bodies launch failed");
            module
                .gpu_integrate(&stream, cfg, &mut pos_buf, &vel_buf, DT)
                .expect("gpu_integrate launch failed");
        }
    }

    let result = pos_buf.to_host_vec(&stream).expect("D2H readback");

    let elapsed = start.elapsed();
    println!("[nbodies] elapsed: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    // Light sanity check so a silently-failed launch can't masquerade as a
    // fast run: every body's position must still be finite after 3 steps.
    let bad = result
        .iter()
        .filter(|p| !p.0.iter().all(|c| c.is_finite()))
        .count();
    if bad != 0 {
        eprintln!("ERROR: {bad} of {} bodies had a non-finite position", result.len());
        std::process::exit(1);
    }
}
