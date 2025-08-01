
use lib::define_gpu_kernel;
//mod dynamic_lib;

fn main() {
    define_gpu_kernel! {
        fn my_kernel(x: &[f32], y: &[f32], result: &mut [f32]) {
            
        }
    }
}