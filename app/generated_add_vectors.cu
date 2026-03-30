#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>

extern "C" __global__ void add_vectors(const float *a, const float *b, float *result, uint64_t n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        result[i] = a[i] + b[i];
    }
}
