#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void add_vectors(const float *a, const float *b, float *result, uint64_t n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;

    for (uint64_t i = idx; i < n; i += stride) {
        result[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv) {
    uint64_t n = 1ULL << 20;
    if (argc >= 2) {
        char *end = nullptr;
        unsigned long long v = strtoull(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && v > 0) {
            n = v;
        }
    }
    size_t bytes = n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_result = (float *)malloc(bytes);

    for (uint64_t i = 0; i < n; i++) {
        h_a[i] = (float)(i + 1);
        h_b[i] = (float)(i + 1);
    }

    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_result, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    uint32_t threads_per_block = 128;
    uint32_t num_blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    add_vectors<<<num_blocks, threads_per_block>>>(d_a, d_b, d_result, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[vector_sum] elapsed: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_result);

    return 0;
}
