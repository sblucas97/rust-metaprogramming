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


// __global__ void matrix_add_kernel(const float *a, const float *b, float *result) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < ROWS * COLS) {
//         result[idx] = a[idx] + b[idx];
//     }
// }

extern "C" void allocate_gpu_mem(float **a_d) {
	size_t size = ROWS * COLS * sizeof(float);
	cudaError_t err = cudaMalloc((void**)a_d, size);
	checkCudaError(err, "Failed to cuda malloc");
}

extern "C" void copy_to_gpu(float *a_d, const float *a_h) {
	size_t size = ROWS * COLS * sizeof(float);
	cudaError_t err = cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	checkCudaError(err, "Failed to allocate device memory");
}

extern "C" void copy_from_gpu(float *result_h, float *result_d) {
	size_t size = ROWS * COLS * sizeof(float);
	cudaError_t err = cudaMemcpy(result_h, result_d, size, cudaMemcpyDeviceToHost);
	checkCudaError(err, "Failed to copy result to device memory");
}

// extern "C" void launch_kernel(float *a_d, float *b_d, float *result_d) {
//     int total = ROWS * COLS;
//     int block_size = 256;
//     int grid_size = (total + block_size - 1) / block_size;

//     matrix_add_kernel<<<grid_size, block_size>>>(a_d, b_d, result_d);
// 	cudaError_t err = cudaGetLastError();
//     checkCudaError(err, "Kernel launch failed");
//     cudaDeviceSynchronize(); // Wait for kernel to finish
// }

extern "C" void free_gpu_mem(float *data_device) {
	cudaFree(data_device);
}