use lib::{cuda_module, spawn};
use lib_core::CudaVec;

pub fn run(dim: usize) -> CudaVec<f32> {
    let mut ptr: CudaVec<f32> = CudaVec::new(vec![0.0f32; dim * dim * 4]);
    let ticks: f32 = 10.0f32;
    spawn!(
        ripple_kernel::ripple_kernel,
        (dim as u32, dim as u32, 1),
        (1, 1, 1),
        ptr,
        dim as u64,
        ticks
    );
    ptr.copy_from_device();
    ptr
}

#[cuda_module]
pub mod ripple_kernel {
    use lib_core::CudaVec;
    use lib::kernel;

    #[kernel]
    pub fn ripple_kernel(ptr: &mut CudaVec<f32>, dim: u64, ticks: f32) {
        let x: u64 = blockIdx.x;
        let y: u64 = blockIdx.y;
        if x < dim && y < dim {
            let offset: u64 = x + y * dim;
            let dim_f: f32 = dim as f32;
            let fx: f32 = 0.5f32 * x as f32 - dim_f / 15.0f32;
            let fy: f32 = 0.5f32 * y as f32 - dim_f / 15.0f32;
            let d: f32 = sqrtf(fx * fx + fy * fy);
            let grey: f32 = floorf(
                128.0f32
                    + 127.0f32 * cosf(d / 10.0f32 - ticks / 7.0f32) / (d / 10.0f32 + 1.0f32),
            );
            ptr[offset * 4 + 0] = grey;
            ptr[offset * 4 + 1] = grey;
            ptr[offset * 4 + 2] = grey;
            ptr[offset * 4 + 3] = 255.0f32;
        }
    }
}
