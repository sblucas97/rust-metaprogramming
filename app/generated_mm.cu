#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>

extern "C" __global__ void mm(const float *a, const float *b, float *c, uint64_t m, uint64_t n, uint64_t k) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i += 1) {
            sum = sum + a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }

}
