#include<stdio.h>
#include<cuda_runtime.h>

// All four entry points are element-type-agnostic: the Rust side computes the
// byte count (len * size_of::<T>()), so `nbytes` here is bytes, not elements.

void checkCudaError(cudaError_t err, const char *msg) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s: %s \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

extern "C" void allocate_gpu_mem(void **p, size_t nbytes) {
	if (nbytes == 0) {
		*p = nullptr;
		return;
	}
	cudaError_t err = cudaMalloc(p, nbytes);
	checkCudaError(err, "Failed to cuda malloc");
}

extern "C" void copy_to_gpu(void *dst_d, const void *src_h, size_t nbytes) {
	if (nbytes == 0) return;
	cudaError_t err = cudaMemcpy(dst_d, src_h, nbytes, cudaMemcpyHostToDevice);
	checkCudaError(err, "Failed to copy to device memory");
}

extern "C" void copy_from_gpu(void *dst_h, const void *src_d, size_t nbytes) {
	if (nbytes == 0) return;
	cudaError_t err = cudaMemcpy(dst_h, src_d, nbytes, cudaMemcpyDeviceToHost);
	checkCudaError(err, "Failed to copy result to host memory");
}

extern "C" void free_gpu_mem(void *p) {
	// cudaFree(nullptr) is a documented no-op, so zero-length vecs are fine.
	cudaFree(p);
}
