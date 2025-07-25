
    #include<stdio.h>
    #include<cuda_runtime.h>

    #define ROWS 10
    #define COLS 10

    __global__ void matrix_add_kernel_2(const float *a, const float *b, float *result) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < ROWS * COLS) {
            result[idx] = a[idx] + b[idx];
        }
    }

    extern "C" void launch_kernel_2(float *a_d, float *b_d, float *result_d) {
        int total = ROWS * COLS;
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;

        matrix_add_kernel_2<<<grid_size, block_size>>>(a_d, b_d, result_d);
	    cudaError_t err = cudaGetLastError();	
        if (err != cudaSuccess) {
		    fprintf(stderr, "CUDA Error: %s: %s \n", "Failed launching kernel", cudaGetErrorString(err));
		    exit(EXIT_FAILURE);
    	}

        cudaDeviceSynchronize(); // Wait for kernel to finish
    } 
    