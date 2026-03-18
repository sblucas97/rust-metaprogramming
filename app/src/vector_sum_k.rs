use lib::{cuda_module};

#[cuda_module]
pub mod vector_sum_kernel {
    use lib_core::CudaVec;
    use lib::kernel;
    
    #[kernel]
    pub fn add_vectors(a: &CudaVec<f32>, b: &CudaVec<f32>, result: &mut CudaVec<f32>, n: u64) {
        let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
        let stride: u64 = blockDim.x * gridDim.x;

        for i in (idx, n).step_by(stride) {
            result[i] = a[i] + b[i];
        }
    }
}