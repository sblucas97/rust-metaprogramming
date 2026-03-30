#ifndef KERNEL_H
#define KERNEL_H

#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>


extern "C" __global__ void mm(const float *cc, const float *dd, float *ee);

#endif