#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_SPHERES 20

__global__ void raytracing(const float *spheres, float *image, uint64_t width, uint64_t height) {
    uint64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    uint64_t offset = x + y * width;

    float ox = (float)x - (float)width  / 2.0f;
    float oy = (float)y - (float)height / 2.0f;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    float maxz = -99999.0f;

    for (int i = 0; i < NUM_SPHERES; i++) {
        float sphereRadius = spheres[i * 7 + 3];
        float dx = ox - spheres[i * 7 + 4];
        float dy = oy - spheres[i * 7 + 5];
        float n = 0.0f, t = -99999.0f, dz = 0.0f;

        if ((dx * dx + dy * dy) < (sphereRadius * sphereRadius)) {
            dz = sqrtf(sphereRadius * sphereRadius - dx * dx - dy * dy);
            n  = dz / sqrtf(sphereRadius * sphereRadius);
            t  = dz + spheres[i * 7 + 6];
        } else {
            t = -99999.0f;
            n = 0.0f;
        }

        if (t > maxz) {
            float fscale = n;
            r    = spheres[i * 7 + 0] * fscale;
            g    = spheres[i * 7 + 1] * fscale;
            b    = spheres[i * 7 + 2] * fscale;
            maxz = t;
        }
    }

    image[offset * 4 + 0] = r * 255.0f;
    image[offset * 4 + 1] = g * 255.0f;
    image[offset * 4 + 2] = b * 255.0f;
    image[offset * 4 + 3] = 255.0f;
}

/* Simple LCG matching the seed/range used in the Elixir reference:
   rnd(x) = x * randint(1,32767) / 32767                             */
static uint32_t rng_state = 313;
static float rnd(float x) {
    rng_state = rng_state * 1664525u + 1013904223u;
    uint32_t v = ((rng_state >> 1) % 32767u) + 1u;
    return x * (float)v / 32767.0f;
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

    float radius, sum;
    if      (dim == 256)  { radius = 20.0f; sum =  5.0f; }
    else if (dim == 1024) { radius = 80.0f; sum = 20.0f; }
    else if (dim == 2048) { radius = 120.0f; sum = 20.0f; }
    else                  { radius = 160.0f; sum = 20.0f; }

    /* Build sphere list on the host (20 * 7 floats) */
    float h_spheres[NUM_SPHERES * 7];
    for (int i = 0; i < NUM_SPHERES; i++) {
        h_spheres[i * 7 + 0] = rnd(1.0f);                          /* r      */
        h_spheres[i * 7 + 1] = rnd(1.0f);                          /* g      */
        h_spheres[i * 7 + 2] = rnd(1.0f);                          /* b      */
        h_spheres[i * 7 + 3] = rnd(radius) + sum;                  /* radius */
        h_spheres[i * 7 + 4] = rnd((float)dim) - (float)dim / 2.0f; /* x    */
        h_spheres[i * 7 + 5] = rnd((float)dim) - (float)dim / 2.0f; /* y    */
        h_spheres[i * 7 + 6] = rnd(256.0f) - 128.0f;               /* z      */
    }

    uint64_t img_count = dim * dim * 4;
    size_t bytes_spheres = NUM_SPHERES * 7 * sizeof(float);
    size_t bytes_image   = img_count * sizeof(float);

    float *h_image = (float *)calloc(img_count, sizeof(float));

    float *d_spheres, *d_image;
    cudaMalloc(&d_spheres, bytes_spheres);
    cudaMalloc(&d_image,   bytes_image);

    cudaMemcpy(d_spheres, h_spheres, bytes_spheres, cudaMemcpyHostToDevice);
    cudaMemset(d_image, 0, bytes_image);

    uint32_t block_size = 16;
    dim3 block(block_size, block_size, 1);
    dim3 grid((dim + block_size - 1) / block_size,
              (dim + block_size - 1) / block_size, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    raytracing<<<grid, block>>>(d_spheres, d_image, dim, dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[raytracer] elapsed: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_image, d_image, bytes_image, cudaMemcpyDeviceToHost);

    cudaFree(d_spheres);
    cudaFree(d_image);
    free(h_image);

    return 0;
}
