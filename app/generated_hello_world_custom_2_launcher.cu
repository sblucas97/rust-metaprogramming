
    #include <cuda_runtime.h>
    #include <iostream>
    
#include "generated_hello_world_custom_2.h"

        #define ROWS 100
        #define COLS 100
    

extern "C" void launch_generated_hello_world_custom_2() {
    int total = ROWS * COLS;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    hello_world_custom_2<<<grid_size, block_size>>>();
    // cudaError_t err = cudaGetLastError();
    // checkCudaError(err, "Kernel launch failed");
    cudaDeviceSynchronize();
}
    
