#include<stdio.h>
#include<cstdint>
#include<cuda_runtime.h>

extern "C" __global__ void julia_kernel(float *ptr, uint64_t dim) {
    uint64_t x = blockIdx.x;
    uint64_t y = blockIdx.y;
    if (x < dim && y < dim) {
        uint64_t offset = x + y * dim;
        float scale = 0.1;
        float jx = scale * ((float)((dim - x))) / ((float)(dim));
        float jy = scale * ((float)((dim - y))) / ((float)(dim));
        float cr = (0.0 - 0.8);
        float ci = 0.156;
        float ar = jx;
        float ai = jy;
        float julia_value = 1.0;
        uint32_t escaped = 0;
        for (int _i = 0; _i < 200; _i += 1) {
            if (escaped == 0) {
                float nar = ((ar * ar) - (ai * ai)) + cr;
                float nai = ((ai * ar) + (ar * ai)) + ci;
                if (((nar * nar) + (nai * nai)) > 1000.0) {
                    julia_value = 0.0;
                    escaped = 1;
                }

                if (escaped == 0) {
                    ar = nar;
                    ai = nai;
                }

            }

        }
        ptr[offset * 4 + 0] = 255.0 * julia_value;
        ptr[offset * 4 + 1] = 0.0;
        ptr[offset * 4 + 2] = 0.0;
        ptr[offset * 4 + 3] = 255.0;
    }

}
