#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define DT 0.01f
#define SOFTENING 1e-9f
#define STEPS 3

__global__ void gpu_n_bodies(float *p, float dt, uint64_t n, float softening) {
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;
        for (uint64_t j = 0; j < n; j += 1) {
            float dx = p[6 * j] - p[6 * i];
            float dy = p[6 * j + 1] - p[6 * i + 1];
            float dz = p[6 * j + 2] - p[6 * i + 2];
            float dist_sqr = dx * dx + dy * dy + dz * dz + softening;
            float inv_dist = 1.0f / sqrtf(dist_sqr);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            fx = fx + dx * inv_dist3;
            fy = fy + dy * inv_dist3;
            fz = fz + dz * inv_dist3;
        }
        p[6 * i + 3] = p[6 * i + 3] + dt * fx;
        p[6 * i + 4] = p[6 * i + 4] + dt * fy;
        p[6 * i + 5] = p[6 * i + 5] + dt * fz;
    }
}

__global__ void gpu_integrate(float *p, float dt, uint64_t n) {
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        p[6 * i] = p[6 * i] + p[6 * i + 3] * dt;
        p[6 * i + 1] = p[6 * i + 1] + p[6 * i + 4] * dt;
        p[6 * i + 2] = p[6 * i + 2] + p[6 * i + 5] * dt;
    }
}

// Same LCG as examples/src/benchmarks/nbodies.rs::generate_bodies, so both
// binaries start from the same body layout at a given n.
static void generate_bodies(float *p, uint64_t n) {
    uint32_t state = 42;
    for (uint64_t body = 0; body < n; body++) {
        for (int axis = 0; axis < 3; axis++) {
            state = state * 1664525u + 1013904223u;
            p[6 * body + axis] = ((state >> 8) % 32767) / 32767.0f * 2.0f - 1.0f;
        }
        p[6 * body + 3] = 0.0f;
        p[6 * body + 4] = 0.0f;
        p[6 * body + 5] = 0.0f;
    }
}

int main(int argc, char **argv) {
    uint64_t n = 4096;
    if (argc >= 2) {
        char *end = nullptr;
        unsigned long long v = strtoull(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && v > 0) {
            n = v;
        }
    }

    size_t bytes = n * 6 * sizeof(float);
    float *h_p = (float *)malloc(bytes);
    generate_bodies(h_p, n);

    float *d_p;
    cudaMalloc(&d_p, bytes);
    cudaMemcpy(d_p, h_p, bytes, cudaMemcpyHostToDevice);

    uint32_t block_size = 128;
    uint32_t num_blocks = (uint32_t)((n + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int step = 0; step < STEPS; step++) {
        gpu_n_bodies<<<num_blocks, block_size>>>(d_p, DT, n, SOFTENING);
        gpu_integrate<<<num_blocks, block_size>>>(d_p, DT, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[nbodies] elapsed: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_p);
    free(h_p);

    return 0;
}
