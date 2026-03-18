use lib::{cuda_module};

#[cuda_module]
pub mod mm_kernel {
    use lib_core::{CudaVec};
    use lib::{kernel, device_function};

    #[device_function] 
    pub fn normalize(val: f32) -> f32 {
        return (val - 10.0f32) / (500.0f32 - 10.0f32);
    }
    
    #[kernel]
    pub fn mm(cc: &CudaVec<f32>, dd: &CudaVec<f32>, ee: &mut CudaVec<f32>) {
        let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < ROWS * COLS) {
            let normA: f32 = normalize(cc[idx]);
            let normB: f32 = normalize(dd[idx]);
            ee[idx] = normA * normB;
        }
    }
}