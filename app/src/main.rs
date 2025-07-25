use std::ptr;
use std::fs::write;
use std::process::Command;
/*
use std::path::Path;
use lib::my_macro;
*/
mod rust_macros;
mod dynamic_lib;

fn main() {
    

    let new_c_function = r#"
    #include<stdio.h>
    #include<cuda_runtime.h>

    #define ROWS 10
    #define COLS 10

    __global__ void matrix_add_kernel_2(const float *a, const float *b, float *result) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < ROWS * COLS) {
            result[idx] = a[idx] + b[idx];
        }
    }

    extern "C" void launch_kernel_2(float *a_d, float *b_d, float *result_d) {
        int total = ROWS * COLS;
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;

        matrix_add_kernel_2<<<grid_size, block_size>>>(a_d, b_d, result_d);
	    cudaError_t err = cudaGetLastError();	
        if (err != cudaSuccess) {
		    fprintf(stderr, "CUDA Error: %s: %s \n", "Failed launching kernel", cudaGetErrorString(err));
		    exit(EXIT_FAILURE);
    	}

        cudaDeviceSynchronize(); // Wait for kernel to finish
    } 
    "#;

    let file_path = "dynamic.cu";

    write(file_path, new_c_function).expect("Failed to create file");

    // compile to shared object
    let so_path = "./libdyn_cu.so";

    let output = Command::new("nvcc")
        .args(["-shared", file_path, "-o", so_path, "-arch=sm_86", "-Xcompiler", "-fPIC"])
        .output()
        .expect("failed to compile cuda.cu");

    if !output.status.success() {
        eprintln!("GCC Error: {}", String::from_utf8_lossy(&output.stderr));
        return;
    }

    // device pointer
    let mut a_device_ptr: *mut f32 = ptr::null_mut();
    let mut b_device_ptr: *mut f32 = ptr::null_mut();
    let mut result_device_ptr: *mut f32 = ptr::null_mut();

    lib_core::custom_allocate_gpu_mem(&mut a_device_ptr as *mut *mut f32);
    lib_core::custom_allocate_gpu_mem(&mut b_device_ptr as *mut *mut f32);
    lib_core::custom_allocate_gpu_mem(&mut result_device_ptr as *mut *mut f32);

    // host data
    // let mut a_host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut a_host_data: Vec<f32> = (1..=10*10).map(|x| x as f32).collect();
    let mut b_host_data: Vec<f32> = (1..=10*10).map(|x| x as f32).collect();
    // let mut b_host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut result_host_data: Vec<f32> = vec![0.0; 10*10];
    lib_core::custom_copy_to_gpu(a_device_ptr, a_host_data.as_ptr());
    lib_core::custom_copy_to_gpu(b_device_ptr, b_host_data.as_ptr());
    
    unsafe {
        let dyn_lib = dynamic_lib::DynamicLib::new(so_path).expect("Failed to load dynamic lib"); 
    //     // Library::new(so_path).expect("Failed to load lib");
    //     // let mult: Symbol<unsafe extern "C" fn(i32, i32) -> i32> =
    //     //     lib.get(b"mult").expect("Failed to find symbol");
        (dyn_lib.launch_kernel_2)(a_device_ptr, b_device_ptr, result_device_ptr);
    }
    lib_core::custom_copy_from_gpu(result_host_data.as_mut_ptr(), result_device_ptr);

    lib_core::custom_free_gpu_mem(a_device_ptr);
    lib_core::custom_free_gpu_mem(b_device_ptr);
    lib_core::custom_free_gpu_mem(result_device_ptr);

    // let r = lib_core::custom_add(300, 400);
    // lib_core::custom_launch_kernel(a_device_ptr, b_device_ptr, result_device_ptr);

    for n in result_host_data.iter() {
        println!("{}", n);
    }

    /*
    custom_for!(
        let x = 0; x < 10; x += 1 => {
            println!("from outside")
        }
    );
    custom_for!(let x = 10; x > 0; x -= 1);
    */

    /*
    let total_sum = custom_reduce!([1, 2, 3, 10, 20], item => acc + item, 1555);  
    let total_product = custom_reduce!([1, 2, 3], item => acc * item, 1);  
    println!("{}", total_sum);
    println!("{}", total_product);
    */

    /*let filename = "runtime_generated.c";
    let binary = "./runtime_generated";

    my_macro! {
        xx a : 10;
        xx b : 10;
        xx r : a + b;
        print("%d", r);

        xx a1 : 20;
        xx b1 : 30;
        xx r2 : a1 * b1;
        print("%d", r2);
    };

    let compile_status = Command::new("gcc")
        .arg(filename)
        .arg("-o")
        .arg(binary)
        .status()
        .expect("Failed to run gcc");

    if !compile_status.success() {
        panic!("GCC compilation failed");
    }

    if !Path::new(binary).exists() {
        panic!("Compile binary not found");
    }

    println!("Running generated C program");
    let run_status = Command::new(binary)
        .status()
        .expect("Failed to run C binary");

    if !run_status.success() {
        panic!("C binary ran with a non-zero exit code");
    }
*/
}
