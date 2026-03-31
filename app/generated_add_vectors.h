#ifndef KERNEL_H
#define KERNEL_H

#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>


extern "C" __global__ void add_vectors(const float *a, const float *b, float *result, uint64_t n);

#endif