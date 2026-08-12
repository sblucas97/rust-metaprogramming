use std::time::Instant;

use macros::{cuda_module, spawn};
use runtime::CudaVec;

pub(crate) const FLOATS_PER_BODY: usize = 6; // pos x/y/z, vel x/y/z
pub(crate) const DT: f32 = 0.01;
pub(crate) const SOFTENING: f32 = 1e-9;
pub(crate) const STEPS: usize = 3;

/// Deterministic pseudo-random bodies in [-1, 1], zero initial velocity.
pub(crate) fn generate_bodies(n: usize) -> Vec<f32> {
    let mut state: u32 = 42;
    let mut bodies = vec![0.0f32; n * FLOATS_PER_BODY];
    for body in bodies.chunks_exact_mut(FLOATS_PER_BODY) {
        for pos in body.iter_mut().take(3) {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *pos = ((state >> 8) % 32767) as f32 / 32767.0 * 2.0 - 1.0;
        }
    }
    bodies
}

pub fn run(n: usize) -> CudaVec<f32> {
    let bodies = generate_bodies(n);
    let mut p: CudaVec<f32> = CudaVec::new(bodies);

    let threads_per_block: u32 = 128;
    let num_blocks: u32 = (n as u32 + threads_per_block - 1) / threads_per_block;

    let start = Instant::now();
    for _ in 0..STEPS {
        spawn!(
            nbodies_kernel::gpu_n_bodies,
            (num_blocks, 1, 1),
            (threads_per_block, 1, 1),
            p,
            DT,
            n as u64,
            SOFTENING
        );
        spawn!(
            nbodies_kernel::gpu_integrate,
            (num_blocks, 1, 1),
            (threads_per_block, 1, 1),
            p,
            DT,
            n as u64
        );
    }
    let elapsed = start.elapsed();
    println!("[nbodies] elapsed: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    p.copy_from_device();

    p
}

#[cuda_module]
pub mod nbodies_kernel {
    use runtime::CudaVec;
    use macros::kernel;

    #[kernel]
    pub fn gpu_n_bodies(p: &mut CudaVec<f32>, dt: f32, n: u64, softening: f32) {
        let i: u64 = blockDim.x * blockIdx.x + threadIdx.x;
        if i < n {
            let mut fx: f32 = 0.0f32;
            let mut fy: f32 = 0.0f32;
            let mut fz: f32 = 0.0f32;
            for j in (0u64, n).step_by(1u64) {
                let dx: f32 = p[6 * j] - p[6 * i];
                let dy: f32 = p[6 * j + 1] - p[6 * i + 1];
                let dz: f32 = p[6 * j + 2] - p[6 * i + 2];
                let dist_sqr: f32 = dx * dx + dy * dy + dz * dz + softening;
                let inv_dist: f32 = 1.0f32 / sqrtf(dist_sqr);
                let inv_dist3: f32 = inv_dist * inv_dist * inv_dist;
                fx = fx + dx * inv_dist3;
                fy = fy + dy * inv_dist3;
                fz = fz + dz * inv_dist3;
            }
            p[6 * i + 3] = p[6 * i + 3] + dt * fx;
            p[6 * i + 4] = p[6 * i + 4] + dt * fy;
            p[6 * i + 5] = p[6 * i + 5] + dt * fz;
        }
    }

    #[kernel]
    pub fn gpu_integrate(p: &mut CudaVec<f32>, dt: f32, n: u64) {
        let i: u64 = blockDim.x * blockIdx.x + threadIdx.x;
        if i < n {
            p[6 * i] = p[6 * i] + p[6 * i + 3] * dt;
            p[6 * i + 1] = p[6 * i + 1] + p[6 * i + 4] * dt;
            p[6 * i + 2] = p[6 * i + 2] + p[6 * i + 5] * dt;
        }
    }
}
