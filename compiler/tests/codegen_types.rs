//! Pins CUDA codegen's type mapping: new element/scalar types come out as the
//! right C types, and — just as important — f32 kernel output is byte-for-byte
//! what it was before the scalar table existed.

use std::collections::HashMap;

use compiler::codegen::gen_kernel;

#[test]
fn new_scalar_types_map_to_c_types() {
    let kernel: syn::ItemFn = syn::parse_quote! {
        fn k(a: &CudaVec<f64>, b: &mut CudaVec<i32>, n: u64, flag: u8) {
            let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
            if idx < n {
                let scaled: f64 = a[idx] * 2.0;
                let small: i16 = 3i16;
                let wide: i64 = 4i64;
                b[idx] = scaled as i32;
            }
        }
    };

    let out = gen_kernel(kernel, "k".to_string(), HashMap::new(), None, None);

    for needle in [
        "const double *a",
        "int32_t *b",
        "uint64_t n",
        "uint8_t flag",
        "double scaled",
        "int16_t small",
        "int64_t wide",
        "4LL",
    ] {
        assert!(out.contains(needle), "missing `{needle}` in:\n{out}");
    }
}

#[test]
fn bool_local_generates() {
    let kernel: syn::ItemFn = syn::parse_quote! {
        fn k(n: u64) {
            let b: bool = 1u64 < n;
        }
    };
    let out = gen_kernel(kernel, "k".to_string(), HashMap::new(), None, None);
    assert!(out.contains("bool b = "), "missing bool local in:\n{out}");
}

// Byte-identical snapshot of an existing f32 kernel (add_vectors, mirrored
// from examples/src/benchmarks/vector_sum.rs): extending the scalar set must
// not change what f32 kernels compile to.
#[test]
fn f32_kernel_output_is_unchanged() {
    let kernel: syn::ItemFn = syn::parse_quote! {
        pub fn add_vectors(a: &CudaVec<f32>, b: &CudaVec<f32>, result: &mut CudaVec<f32>, n: u64) {
            let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
            result[idx] = a[idx] + b[idx];
        }
    };

    let out = gen_kernel(kernel, "add_vectors".to_string(), HashMap::new(), None, None);

    let expected = "\
#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>

extern \"C\" __global__ void add_vectors(const float *a, const float *b, float *result, uint64_t n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = a[idx] + b[idx];
}
";
    assert_eq!(out, expected);
}
