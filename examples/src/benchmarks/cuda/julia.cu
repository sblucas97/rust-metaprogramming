#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void julia_kernel(float *ptr, uint64_t dim) {
    uint64_t x = blockIdx.x;
    uint64_t y = blockIdx.y;
    if (x < dim && y < dim) {
        uint64_t offset = x + y * dim;
        float scale = 0.1f;
        float jx = scale * (float)(dim - x) / (float)dim;
        float jy = scale * (float)(dim - y) / (float)dim;
        float cr = 0.0f - 0.8f;
        float ci = 0.156f;
        float ar = jx;
        float ai = jy;
        float julia_value = 1.0f;
        uint32_t escaped = 0;
        for (int iter = 0; iter < 200; iter++) {
            if (escaped == 0) {
                float nar = (ar * ar - ai * ai) + cr;
                float nai = (ai * ar + ar * ai) + ci;
                if ((nar * nar + nai * nai) > 1000.0f) {
                    julia_value = 0.0f;
                    escaped = 1;
                }
                if (escaped == 0) {
                    ar = nar;
                    ai = nai;
                }
            }
        }
        ptr[offset * 4 + 0] = 255.0f * julia_value;
        ptr[offset * 4 + 1] = 0.0f;
        ptr[offset * 4 + 2] = 0.0f;
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    uint64_t count = dim * dim * 4;
    size_t bytes = count * sizeof(float);

    float *h_ptr = (float *)calloc(count, sizeof(float));

    float *d_ptr;
    cudaMalloc(&d_ptr, bytes);
    cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);

    dim3 grid(dim, dim, 1);
    dim3 block(1, 1, 1);

    julia_kernel<<<grid, block>>>(d_ptr, dim);

    float ms = 0.0f;

    cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("[julia] elapsed: %.3f ms\n", ms);

    cudaFree(d_ptr);
    free(h_ptr);

    return 0;
}
