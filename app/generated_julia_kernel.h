#ifndef KERNEL_H
#define KERNEL_H

#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>


extern "C" __global__ void julia_kernel(float *ptr, uint64_t dim);

#endif