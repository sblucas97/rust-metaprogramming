#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>

__global__ void ripple_kernel(float *ptr, uint64_t dim, float ticks) {
    uint64_t x = blockIdx.x;
    uint64_t y = blockIdx.y;
    if (x < dim && y < dim) {
        uint64_t offset = x + y * dim;
        float dim_f = (float)dim;
        float fx = 0.5f * (float)x - dim_f / 15.0f;
        float fy = 0.5f * (float)y - dim_f / 15.0f;
        float d = sqrtf(fx * fx + fy * fy);
        float grey = floorf(
            128.0f + 127.0f * cosf(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
        ptr[offset * 4 + 0] = grey;
        ptr[offset * 4 + 1] = grey;
        ptr[offset * 4 + 2] = grey;
        ptr[offset * 4 + 3] = 255.0f;
    }
}

int main(int argc, char **argv) {
    uint64_t dim = 256;
    if (argc >= 2) {
        char *end = nullptr;
        unsigned long long v = strtoull(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && v > 0) {
            dim = v;
        }
    }
    const float ticks = 10.0f;

    uint64_t count = dim * dim * 4;
    size_t bytes = count * sizeof(float);

    float *h_ptr = (float *)calloc(count, sizeof(float));

    float *d_ptr;
    cudaMalloc(&d_ptr, bytes);
    cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);

    dim3 grid((unsigned)dim, (unsigned)dim, 1);
    dim3 block(1, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ripple_kernel<<<grid, block>>>(d_ptr, dim, ticks);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[ripple] elapsed: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_ptr);
    free(h_ptr);

    return 0;
}
