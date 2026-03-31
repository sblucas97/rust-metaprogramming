use lib::{cuda_module, spawn};
use lib_core::CudaVec;

pub fn run(m: usize) -> CudaVec<f32> {
    let n = m;
    let k = m;

    let a: CudaVec<f32> = CudaVec::new((0..m * n).map(|i| (i % 100 + 1) as f32).collect());
    let b: CudaVec<f32> = CudaVec::new((0..n * k).map(|i| (i % 100 + 1) as f32).collect());
    let mut c: CudaVec<f32> = CudaVec::new(vec![0.0f32; m * k]);

    let block_size: u32 = 16;
    let grid_rows: u32 = (m as u32 + block_size - 1) / block_size;
    let grid_cols: u32 = (k as u32 + block_size - 1) / block_size;

    spawn!(
        mm_kernel::mm,
        (grid_cols, grid_rows, 1),
        (block_size, block_size, 1),
        a,
        b,
        c,
        m as u64,
        n as u64,
        k as u64
    );
    c.copy_from_device();
    c
}

#[cuda_module]
pub mod mm_kernel {
    use lib_core::CudaVec;
    use lib::kernel;

    #[kernel]
    pub fn mm(a: &CudaVec<f32>, b: &CudaVec<f32>, c: &mut CudaVec<f32>, m: u64, n: u64, k: u64) {
        let row: u64 = blockIdx.y * blockDim.y + threadIdx.y;
        let col: u64 = blockIdx.x * blockDim.x + threadIdx.x;
        let mut sum: f32 = 0.0f32;
        if col < k && row < m {
            for i in (0u64, n).step_by(1u64) {
                sum = sum + a[row * n + i] * b[i * k + col];
            }
            c[row * k + col] = sum;
        }
    }
}
