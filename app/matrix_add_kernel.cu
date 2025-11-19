
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
