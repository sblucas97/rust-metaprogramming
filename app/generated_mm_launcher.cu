#include<stdio.h>
#include<cuda_runtime.h>
#include "generated_mm.h"

#define ROWS 100
#define COLS 100

extern "C" void launch_generated_mm(const float *cc, const float *dd, float *ee) {
    int total = ROWS * COLS; 
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    mm<<<grid_size, block_size>>>(cc, dd, ee);
    cudaDeviceSynchronize();
}