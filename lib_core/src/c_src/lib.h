#ifndef LIB_H
#define LIB_H

int add(int a, int b);
void allocate_gpu_mem(float **a_d);
void free_gpu_mem(float *data_device);
void copy_to_gpu(float *a_d, const float *a_h);
void copy_from_gpu(float *result_h, float *result_d);
void launch_kernel(float *a_d, float *b_d, float *result_d);
#endif
