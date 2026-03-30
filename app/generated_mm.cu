#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>


#define ROWS 100
#define COLS 100

extern "C" __global__ void mm(const float *cc, const float *dd, float *ee) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ROWS * COLS) {
        ee[idx] = cc[idx] * dd[idx];
    }

}
