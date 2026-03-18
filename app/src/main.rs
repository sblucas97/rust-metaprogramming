use lib_core::{CudaVec};
use lib::spawn;

mod mult_k;
mod vector_sum_k;
use mult_k::mm_kernel;
use vector_sum_k::vector_sum_kernel;
// use mult_k::mm_kernel::mm;


fn main() {
    let size: usize = 10_000_000;
    let mut a_host_data: CudaVec<f32> = CudaVec::new(vec![1.0f32; size]);
    let mut b_host_data: CudaVec<f32> = CudaVec::new(vec![1.0f32; size]);
    let mut result_host_data: Vec<f32> = Vec::new();
    let n: u64 = size as u64;
    
    spawn!(vector_sum_kernel::add_vectors, a_host_data, b_host_data, > result_host_data: f32, n: u64);
    // multiply(10);

    // let x: Vec<f32> = vec![1.0f32];
    // x.iter().map(|x| x * 2);

    // gpufor!(a_host_data -> |x| x * 2);
    
    // spawn!(mm_kernel::mm, a_host_data, b_host_data, > result_host_data: f32);
    

    // for n in result_host_data.iter() {
    //     println!("{}", n);
    // }
}