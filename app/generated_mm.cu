#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>


#define ROWS 100
#define COLS 100

__device__ float normalize(float val) {
    return (val - 10.0) / (500.0 - 10.0);
}

__global__ void mm(const float *cc, const float *dd, float *ee) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ROWS * COLS) {
        float normA = normalize(cc[idx]);
        float normB = normalize(dd[idx]);
        ee[idx] = normA * normB;
    }

}
