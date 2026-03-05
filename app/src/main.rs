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
fn mm(cc: &CudaVec<f32>, dd: &CudaVec<f32>, ee: &mut CudaVec<f32>) {
    let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ROWS * COLS) {
        ee[idx] = cc[idx] * dd[idx];
    }
}

    
fn main() {
    let mut a_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut b_host_data: CudaVec<f32> = CudaVec::new((1..=100*100).map(|x| x as f32).collect());
    let mut result_host_data: Vec<f32> = Vec::new();
    
    spawn::<mm::Marker>(a_host_data, b_host_data, &mut result_host_data);

    for n in result_host_data.iter() {
        println!("{}", n);
    }

}


pub fn spawn<K: KernelName>(a_h: CudaVec<f32>, b_h: CudaVec<f32>, r_h: &mut Vec<f32>) {
// pub fn spawn<K: KernelName>() {
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
        // allocating mem in GPU for result
        let mut result_device_ptr: *mut f32 = ptr::null_mut();
        lib_core::custom_allocate_gpu_mem(&mut result_device_ptr as *mut *mut f32);

        type LaunchKernelFuncFloat = unsafe extern "C" fn(*mut std::os::raw::c_float, *mut std::os::raw::c_float, *mut std::os::raw::c_float);
        
        let lib = libloading::Library::new(&absolute_path)
        .expect("Failed to load library");

        let launch_hello_fun_name = format!("launch_generated_{function_name}");
        
        // Fix: Convert the String to a null-terminated CString
        let symbol_name = std::ffi::CString::new(launch_hello_fun_name)
            .expect("Failed to create CString");
        
        let launch_kernel_symbol: libloading::Symbol<LaunchKernelFuncFloat> =
            lib.get(symbol_name.as_bytes_with_nul())
                .expect("Failed to get symbol");

        launch_kernel_symbol(
            a_h.get_device_ptr() as *mut std::os::raw::c_float,
            b_h.get_device_ptr() as *mut std::os::raw::c_float,
            result_device_ptr as *mut std::os::raw::c_float,
        );

        let result_size = a_h.len();
        r_h.resize(result_size, 0.0);
        lib_core::custom_copy_from_gpu(r_h.as_mut_ptr(), result_device_ptr);
        lib_core::custom_free_gpu_mem(result_device_ptr);
    }
}