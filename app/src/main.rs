use std::{alloc::handle_alloc_error, ptr};
use std::fs::write;
use std::process::Command;

use lib::{kernel};
use lib_core::CudaVec;
/*
use std::path::Path;
use lib::my_macro;
*/
mod rust_macros;
mod dynamic_lib;

pub trait KernelName {
    fn kernel_name() -> &'static str;
}

#[kernel]
pub fn hello_world_custom_2(a: CudaVec<f32>, b: CudaVec<f32>, result: CudaVec<f32>) {
    any_custom_variable_here + 1;
    any_custom_function(a);
}

fn main() {

    let mut a_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut b_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut result_host_data: Vec<f32> = Vec::new();

    spawn::<hello_world_custom_2::Marker>();

    for n in result_host_data.iter() {
        println!("{}", n);
    }

}


pub fn spawn<K: KernelName>() {
    let function_name = K::kernel_name();
    let output_lib_path_generated = format!("kernel_and_launcher_generated_for_{function_name}.so");
    let output = Command::new("nvcc")
        .arg("-arch=sm_86")  // Adjust for your GPU
        .arg(format!("generated_{function_name}.cu"))
        .arg(format!("generated_{function_name}_launcher.cu"))
        .arg("-o")
        .arg(&output_lib_path_generated)
        .arg("--shared")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-x")
        .arg("cu")
        .output()
        .expect("Failed to execute nvcc");

    // Print stdout
    if !output.stdout.is_empty() {
        eprintln!("NVCC stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    }
    
    // Print stderr (this is where most compiler messages go)
    if !output.stderr.is_empty() {
        eprintln!("NVCC stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    }
    
    // Check if compilation succeeded
    if !output.status.success() {
        panic!("nvcc compilation failed with exit code: {:?}", output.status.code());
    }
    
    
    let absolute_path = std::env::current_dir()
        .unwrap()
        .join(&output_lib_path_generated);

    eprintln!("Loading library from: {}", absolute_path.display());
    unsafe {
        type LaunchKernelFuncHelloVoid = unsafe extern "C" fn();
        
        let lib = libloading::Library::new(&absolute_path)
        .expect("Failed to load library");

        let launch_hello_fun_name = format!("launch_generated_{function_name}");
        
        // Fix: Convert the String to a null-terminated CString
        let symbol_name = std::ffi::CString::new(launch_hello_fun_name)
            .expect("Failed to create CString");
        
        let launch_kernel_symbol: libloading::Symbol<LaunchKernelFuncHelloVoid> =
            lib.get(symbol_name.as_bytes_with_nul())
                .expect("Failed to get symbol");

        launch_kernel_symbol();
    }
}