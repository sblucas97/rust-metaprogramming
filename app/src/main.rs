use lib_core::CudaVec;
use lib::spawn;

mod mult_k;
mod vector_sum_k;
mod julia_k;
// use mult_k::mm_kernel::mm;


fn main() {
    let x = julia_k::run(8000);
    for n in x.as_slice() {
        println!("{}", n);
    }
    // let size: usize = 10_000;
    // let a_host_data: CudaVec<f32> = CudaVec::new(vec![2.0f32; size]);
    // let b_host_data: CudaVec<f32> = CudaVec::new(vec![3.0f32; size]);
    // let mut result_host_data_1: CudaVec<f32> = CudaVec::new(vec![0.0f32; size]);
    // let n: u64 = size as u64;

    // spawn!(
    //     julia_k::julia_k::julia_kernel,
    //     (size as u32, size as u32, 1),
    //     (1, 1, 1),
    //     a_host_data,
    //     b_host_data,
    //     result_host_data_1,
    //     1000
    // );
    
    // result_host_data_1.copy_from_device();
    // for n in result_host_data_1.as_slice() {
    //     println!("{}", n);
    // }
    // let config = lib_core::launch::LaunchConfig::for_num_elems(n as u32);
    // let generated_ptx = format!("{}/generated_add_vectors.ptx", env!("CARGO_MANIFEST_DIR"));
    // lib_core::launch::launch_generated_ptx(
    //     generated_ptx,
    //     (
    //         a_host_data.get_device_ptr() as u64,
    //         b_host_data.get_device_ptr() as u64,
    //         result_host_data_1.get_device_ptr() as u64,
    //         n,
    //     ),
    //     config,
    // )
    // .expect("failed to launch generated add_vectors kernel");

    // spawn!(
    //     mm_kernel::mm, 
    //     a_host_data, 
    //     b_host_data, 
    //     > result_host_data_2: f32
    // );
    // multiply(10);

    // let x: Vec<f32> = vec![1.0f32];
    // x.iter().map(|x| x * 2);

    // gpufor!(a_host_data -> |x| x * 2);
    
    // spawn!(mm_kernel::mm, a_host_data, b_host_data, > result_host_data: f32);
    
    // for n in result_host_data_1.iter() {
    //     println!("{}", n);
    // }

    // for n in result_host_data_2.iter() {
    //     println!("{}", n);
    // }
}