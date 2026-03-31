#ifndef KERNEL_H
#define KERNEL_H

#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>


extern "C" __global__ void mm(const float *a, const float *b, float *c, uint64_t m, uint64_t n, uint64_t k);

#endif