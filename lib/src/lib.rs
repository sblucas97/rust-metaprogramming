extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{parse::Parse, parse_macro_input, Token};

mod helpers;

struct SpawnArgs {
    a_host: Ident,
    _comma1: Token![,],
    b_host: Ident,
    _comma2: Token![,],
    result_host: Ident,
}

impl Parse for SpawnArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(SpawnArgs {
            a_host: input.parse()?,
            _comma1: input.parse()?,
            b_host: input.parse()?,
            _comma2: input.parse()?,
            result_host: input.parse()?,
        })
    }
}

#[proc_macro]
pub fn spawn(input: TokenStream) -> TokenStream {
    let SpawnArgs {
        a_host,
        b_host,
        result_host,
        ..
    } = parse_macro_input!(input as SpawnArgs);
    
    let expanded = quote! {
        {
            use std::ptr;
            use std::process::Command;

            let kernel = r#"
#include<stdio.h>
#include<cuda_runtime.h>

#define ROWS 100
#define COLS 100

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s \n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    fprintf(stderr, "EVERYTHING OK\n");
}


__global__ void matrix_add_kernel(const float *a, const float *b, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ROWS * COLS) {
        result[idx] = a[idx] + b[idx];
    }
}

extern "C" void launch_kernel(float *a_d, float *b_d, float *result_d) {
    int total = ROWS * COLS;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    matrix_add_kernel<<<grid_size, block_size>>>(a_d, b_d, result_d);
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");
    cudaDeviceSynchronize();
}
"#;
            
            let output_lib_path = "./matrix_add_kernel.so";
            std::fs::write("matrix_add_kernel.cu", kernel).expect("Failed to write kernel file");

            let compile_status = Command::new("nvcc")
                .arg("matrix_add_kernel.cu")
                .arg("-o")
                .arg(output_lib_path)
                .arg("--shared")
                .arg("-Xcompiler")
                .arg("-fPIC")
                .arg("-x")
                .arg("cu") 
                .status()
                .expect("Failed to execute nvcc");

            if !compile_status.success() {
                panic!("nvcc compilation failed");
            }
            
            let mut result_device_ptr: *mut f32 = ptr::null_mut();
            lib_core::custom_allocate_gpu_mem(&mut result_device_ptr as *mut *mut f32);

            unsafe {
                type LaunchKernelFuncFloat = unsafe extern "C" fn(*mut std::os::raw::c_float, *mut std::os::raw::c_float, *mut std::os::raw::c_float);

                let lib = libloading::Library::new(output_lib_path).expect("Failed to load library");

                let launch_kernel_symbol: libloading::Symbol<LaunchKernelFuncFloat> = 
                    lib.get(b"launch_kernel\0").expect("Failed to get symbol");
                    
                launch_kernel_symbol(
                    #a_host.get_device_ptr() as *mut std::os::raw::c_float,
                    #b_host.get_device_ptr() as *mut std::os::raw::c_float,
                    result_device_ptr as *mut std::os::raw::c_float,
                );
            }
            
            // or rows * cols
            let result_size = #a_host.len(); 
            #result_host.resize(result_size, 0.0);
            lib_core::custom_copy_from_gpu(#result_host.as_mut_ptr(), result_device_ptr);
            lib_core::custom_free_gpu_mem(result_device_ptr);

            ()
        }
    };

    expanded.into()
}