use lib_core::{CudaVec, spawn};
use lib::{cuda_module};

#[cuda_module]
mod kernels {
    use lib_core::{CudaVec, KernelName};
    use lib::{kernel, device_function};

    #[device_function] 
    pub fn normalize(val: f32) -> f32 {
        return (val - 10.0f32) / (500.0f32 - 10.0f32);
    }
    
    #[kernel]
    fn mm(cc: &CudaVec<f32>, dd: &CudaVec<f32>, ee: &mut CudaVec<f32>) {
        let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < ROWS * COLS) {
            let normA: f32 = normalize(cc[idx]);
            let normB: f32 = normalize(dd[idx]);
            ee[idx] = normA * normB;
        }
    }
}
    
fn main() {
    let mut a_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut b_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut result_host_data: Vec<f32> = Vec::new();
    
    // multiply(10);

    spawn::<kernels::mm::Marker>(a_host_data, b_host_data, &mut result_host_data);

    for n in result_host_data.iter() {
        println!("{}", n);
    }
}