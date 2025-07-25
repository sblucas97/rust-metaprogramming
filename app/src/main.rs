use std::ptr;

/*use std::fs;
use std::process::Command;
use std::path::Path;
use lib::my_macro;
*/
mod rust_macros;

fn main() {
    

    let r = lib_core::custom_add(300, 400);
    // device pointer
    let mut a_device_ptr: *mut f32 = ptr::null_mut();
    let mut b_device_ptr: *mut f32 = ptr::null_mut();
    let mut result_device_ptr: *mut f32 = ptr::null_mut();

    lib_core::custom_allocate_gpu_mem(&mut a_device_ptr as *mut *mut f32);
    lib_core::custom_allocate_gpu_mem(&mut b_device_ptr as *mut *mut f32);
    lib_core::custom_allocate_gpu_mem(&mut result_device_ptr as *mut *mut f32);

    // host data
    // let mut a_host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut a_host_data: Vec<f32> = (1..=5000*5000).map(|x| x as f32).collect();
    let mut b_host_data: Vec<f32> = (1..=5000*5000).map(|x| x as f32).collect();
    // let mut b_host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut result_host_data: Vec<f32> = vec![0.0; 5000*5000];
    lib_core::custom_copy_to_gpu(a_device_ptr, a_host_data.as_ptr());
    lib_core::custom_copy_to_gpu(b_device_ptr, b_host_data.as_ptr());
    
    lib_core::custom_launch_kernel(a_device_ptr, b_device_ptr, result_device_ptr);

    lib_core::custom_copy_from_gpu(result_host_data.as_mut_ptr(), result_device_ptr);

    lib_core::custom_free_gpu_mem(a_device_ptr);
    lib_core::custom_free_gpu_mem(b_device_ptr);
    lib_core::custom_free_gpu_mem(result_device_ptr);

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
