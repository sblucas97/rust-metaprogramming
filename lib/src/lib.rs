extern crate proc_macro;

use std::fmt::format;
use std::process::Command;

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{parse::Parse, parse_macro_input, Token, ItemFn};

mod helpers_1;
mod gen_kernel;
mod gen_kernel_header;
mod gen_launcher;
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

#[proc_macro_attribute]
pub fn kernel(args: TokenStream, input: TokenStream) -> TokenStream {

    // 1. Parse the input TokenStream into a structured ItemFn AST node.
    // If the input is not a function, this will panic and generate a helpful error.
    let input_fn = parse_macro_input!(input as ItemFn);

    // --- Identification/Extraction ---

    // 2. Identify the function name (and convert it into a string)
    let function_name = input_fn.sig.ident.to_string();
    println!("Kernel Name: {}", function_name); // Prints at macro compilation time

    gen_kernel::gen_kernel(&function_name);
    gen_kernel_header::gen_kernel_header(&function_name);
    gen_launcher::gen_launcher(&function_name);

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
    // 3. Access function arguments
    // input_fn.sig.inputs is a Punctuated<FnArg, Comma>
    // println!("Arguments:");
    // for arg in &input_fn.sig.inputs {
    //     // You would typically use pattern matching here to destructure the argument
    //     // into its identifier, type, etc.
    //     println!("  - {:?}", arg);
    // }

    // 4. Access the function body (Block)
    // input_fn.block is a Box<Block>
    // You can iterate over the statements in the body:
    let body_statements = &input_fn.block.stmts;
    // println!("Function Body Statements Count: {}", body_statements.len());

    // NOTE: To parse the *contents* of these statements dynamically (like finding
    // `any_custom_variable_here` or `any_custom_function`), you would have to
    // traverse the tokens within each statement's AST, which is the most complex
    // part and usually requires a custom visitor pattern (syn::visit::Visit).

    // Example of just printing the raw body tokens:
    for (index, item) in body_statements.iter().enumerate() {

        println!("i {} Token {:?}", index, item);
    }
    // let body_tokens = quote! { #(#body_statements)* };
    TokenStream::new()
}
