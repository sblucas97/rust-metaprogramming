use std::ptr;
use std::fs::write;
use std::process::Command;

use lib::{spawn};
use lib_core::CudaVec;
/*
use std::path::Path;
use lib::my_macro;
*/
mod rust_macros;
mod dynamic_lib;


fn main() {

    let mut a_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut b_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut result_host_data: Vec<f32> = Vec::new();

    spawn!(a_host_data, b_host_data, result_host_data);
    
    for n in result_host_data.iter() {
        println!("{}", n);
    }

}
