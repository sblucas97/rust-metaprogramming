use lib::cuda_module;
use lib::spawn;
use lib_core::CudaVec;

use std::time::Instant;

pub fn run(dim: usize) -> CudaVec<f32> {
    let data: Vec<f32> = vec![0.0f32; dim * dim * 4];
    let start = Instant::now();    
    let mut ptr: CudaVec<f32> = CudaVec::new(data);
    spawn!(
        julia_kernel::julia_kernel,
        (dim as u32, dim as u32, 1),
        (1, 1, 1),
        ptr,
        dim as u64
    );
    ptr.copy_from_device();
    let end = Instant::now();
    println!("julia kernel time: {}ms", end.duration_since(start).as_millis());
    ptr
}

#[cuda_module]
pub mod julia_kernel {
    use lib_core::CudaVec;
    use lib::kernel;

    #[kernel]
    pub fn julia_kernel(ptr: &mut CudaVec<f32>, dim: u64) {
        let x: u64 = blockIdx.x;
        let y: u64 = blockIdx.y;
        if x < dim && y < dim {
            let offset: u64 = x + y * dim;
            let scale: f32 = 0.1f32;
            let jx: f32 = scale * (dim - x) as f32 / dim as f32;
            let jy: f32 = scale * (dim - y) as f32 / dim as f32;
            let cr: f32 = (0.0f32 - 0.8f32);
            let ci: f32 = 0.156f32;
            let mut ar: f32 = jx;
            let mut ai: f32 = jy;
            let mut julia_value: f32 = 1.0f32;
            let mut escaped: u32 = 0;
            for _i in (0u64, 200u64).step_by(1u64) {
                if escaped == 0 {
                    let nar: f32 = ((ar * ar) - (ai * ai)) + cr;
                    let nai: f32 = ((ai * ar) + (ar * ai)) + ci;
                    if ((nar * nar) + (nai * nai)) > 1000.0f32 {
                        julia_value = 0.0f32;
                        escaped = 1;
                    }
                    if escaped == 0 {
                        ar = nar;
                        ai = nai;
                    }
                }
            }
            ptr[offset * 4 + 0] = 255.0f32 * julia_value;
            ptr[offset * 4 + 1] = 0.0f32;
            ptr[offset * 4 + 2] = 0.0f32;
            ptr[offset * 4 + 3] = 255.0f32;
        }
    }
}
