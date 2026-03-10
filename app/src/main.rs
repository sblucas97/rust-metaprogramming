use lib_core::{CudaVec, spawn};
use lib::{cuda_module};

#[cuda_module]
mod kernels {
    use lib_core::CudaVec;
    use lib::{kernel, device_function};

    #[device_function] 
    pub fn multiply(idx: u64) {
        let x = 10 + 5;
    }

    #[kernel]
    fn mm(cc: &CudaVec<f32>, dd: &CudaVec<f32>, ee: &mut CudaVec<f32>) {
        let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;

        // multiply(idx);

        if (idx < ROWS * COLS) {
            ee[idx] = cc[idx] * dd[idx];
        }
    }
}
    
fn main() {
    let mut a_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut b_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut result_host_data: Vec<f32> = Vec::new();
    
    // multiply(10);

    spawn::<kernels::mm::Marker>(a_host_data, b_host_data, &mut result_host_data);

    // for n in result_host_data.iter() {
    //     println!("{}", n);
    // }
}