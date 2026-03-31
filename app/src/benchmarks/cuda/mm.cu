#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void mm(const float *a, const float *b, float *c, uint64_t m, uint64_t n, uint64_t k) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (col < k && row < m) {
        for (uint64_t i = 0; i < n; i += 1) {
            sum = sum + a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

int main(int argc, char **argv) {
    uint64_t m = 512, n = 512, k = 512;
    if (argc >= 2) {
        char *end = nullptr;
        unsigned long long v = strtoull(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && v > 0) {
            m = n = k = v;
        }
    }
    size_t bytes_a = m * n * sizeof(float);
    size_t bytes_b = n * k * sizeof(float);
    size_t bytes_c = m * k * sizeof(float);

    float *h_a = (float *)malloc(bytes_a);
    float *h_b = (float *)malloc(bytes_b);
    float *h_c = (float *)malloc(bytes_c);

    for (uint64_t i = 0; i < m * n; i++) h_a[i] = (float)(i % 100 + 1);
    for (uint64_t i = 0; i < n * k; i++) h_b[i] = (float)(i % 100 + 1);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    uint32_t block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((k + block_size - 1) / block_size, (m + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mm<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[matrix_mult] elapsed: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
