#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void euclid(const float *d_locations, float *d_distances, uint64_t num_records, float lat, float lng) {
    uint64_t global_id = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
    uint64_t ilat = 2 * global_id;
    uint64_t ilng = 2 * global_id + 1;
    if (global_id < num_records) {
        float dlat = lat - d_locations[ilat];
        float dlng = lng - d_locations[ilng];
        d_distances[global_id] = sqrtf(dlat * dlat + dlng * dlng);
    }
}

// Same LCG as examples/src/benchmarks/nearest_neighbor.rs::generate_locations,
// so both binaries start from the same coordinates at a given num_records.
static void generate_locations(float *locations, uint64_t num_records) {
    uint32_t state = 7919;
    float ranges[2] = {180.0f, 360.0f};
    for (uint64_t i = 0; i < num_records; i++) {
        for (int axis = 0; axis < 2; axis++) {
            state = state * 1664525u + 1013904223u;
            uint32_t v = (state >> 8) % 32767;
            locations[2 * i + axis] = v / 32767.0f * ranges[axis] - ranges[axis] / 2.0f;
        }
    }
}

int main(int argc, char **argv) {
    uint64_t num_records = 100000;
    if (argc >= 2) {
        char *end = nullptr;
        unsigned long long v = strtoull(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && v > 0) {
            num_records = v;
        }
    }

    const float lat = 30.0f;
    const float lng = 90.0f;

    size_t loc_bytes = num_records * 2 * sizeof(float);
    size_t dist_bytes = num_records * sizeof(float);

    float *h_locations = (float *)malloc(loc_bytes);
    float *h_distances = (float *)malloc(dist_bytes);
    generate_locations(h_locations, num_records);

    float *d_locations, *d_distances;
    cudaMalloc(&d_locations, loc_bytes);
    cudaMalloc(&d_distances, dist_bytes);
    cudaMemcpy(d_locations, h_locations, loc_bytes, cudaMemcpyHostToDevice);

    uint32_t threads_per_block = 128;
    uint32_t num_blocks = (uint32_t)((num_records + threads_per_block - 1) / threads_per_block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    euclid<<<num_blocks, threads_per_block>>>(d_locations, d_distances, num_records, lat, lng);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[nearest_neighbor] elapsed: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_distances, d_distances, dist_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_locations);
    cudaFree(d_distances);
    free(h_locations);
    free(h_distances);

    return 0;
}
