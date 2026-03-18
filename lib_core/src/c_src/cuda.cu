#include<stdio.h>
#include<cuda_runtime.h>

void checkCudaError(cudaError_t err, const char *msg) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s: %s \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

extern "C" void allocate_gpu_mem(float **a_d, size_t n) {
	size_t size = n * sizeof(float);
	cudaError_t err = cudaMalloc((void**)a_d, size);
	checkCudaError(err, "Failed to cuda malloc");
}

extern "C" void copy_to_gpu(float *a_d, const float *a_h, size_t n) {
	size_t size = n * sizeof(float);
	cudaError_t err = cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	checkCudaError(err, "Failed to allocate device memory");
}

extern "C" void copy_from_gpu(float *result_h, float *result_d, size_t n) {
	size_t size = n * sizeof(float);
	cudaError_t err = cudaMemcpy(result_h, result_d, size, cudaMemcpyDeviceToHost);
	checkCudaError(err, "Failed to copy result to device memory");
}

extern "C" void free_gpu_mem(float *data_device) {
	cudaFree(data_device);
}