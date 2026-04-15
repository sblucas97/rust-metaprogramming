use lib::{cuda_module, spawn};


pub fn run() {
    let n = 10;
    let threads_per_block: u32 = 128;
    let num_blocks: u32 = (n as u32 + threads_per_block - 1) / threads_per_block;
    spawn!(
        tc_kernel::add_vectors,
        (num_blocks, 1, 1),
        (threads_per_block, 1, 1),
    );
}

#[cuda_module]
pub mod tc_kernel {
    use lib::kernel;

    #[kernel]
    pub fn add_vectors() {
        let x: u64 = 10;
        let y: u64 = 12.0;

        let c: u64 = x + y;
    }
}
