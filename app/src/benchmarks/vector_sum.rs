use lib::{cuda_module, spawn};
use lib_core::CudaVec;

pub fn run(n: usize) -> CudaVec<f32> {
    let a: CudaVec<f32> = CudaVec::new((1..=n).map(|i| i as f32).collect());
    let b: CudaVec<f32> = CudaVec::new((1..=n).map(|i| i as f32).collect());
    let mut result: CudaVec<f32> = CudaVec::new(vec![0.0f32; n]);

    let threads_per_block: u32 = 128;
    let num_blocks: u32 = (n as u32 + threads_per_block - 1) / threads_per_block;

    spawn!(
        vector_sum_kernel::add_vectors,
        (num_blocks, 1, 1),
        (threads_per_block, 1, 1),
        a,
        b,
        result,
        n as u64
    );
    result.copy_from_device();
    result
}

// #[cuda_module]
// pub mod vector_sum_kernel {
//     use lib_core::CudaVec;
//     use lib::kernel;

//     #[kernel]
//     pub fn add_vectors(a: &CudaVec<f32>, b: &CudaVec<f32>, result: &mut CudaVec<f32>, n: u64) {
//         let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
//         let stride: u64 = blockDim.x * gridDim.x;

//         for i in (idx, n).step_by(stride) {
//             result[i] = a[i] + b[i];
//         }
//     }
// }
