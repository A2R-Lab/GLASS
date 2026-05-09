#pragma once
#include <cstdint>

template <typename T>
__device__ void infnorm(uint32_t n, T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    uint32_t left = n;
    bool odd;
    while (left > 3) {
        odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = ind; i < left; i += stride)
            x[i] = max(abs(x[i]), abs(x[i + left]));
        if (ind == 0 && odd) x[0] = max(abs(x[0]), abs(x[2*left]));
        __syncthreads();
    }
    if (ind == 0) { for (uint32_t i = 1; i < left; i++) x[0] = max(abs(x[0]), abs(x[i])); }
}

template <typename T, uint32_t N>
__device__ void infnorm(T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    uint32_t left = N;
    bool odd;
    while (left > 3) {
        odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = ind; i < left; i += stride)
            x[i] = max(abs(x[i]), abs(x[i + left]));
        if (ind == 0 && odd) x[0] = max(abs(x[0]), abs(x[2*left]));
        __syncthreads();
    }
    if (ind == 0) { for (uint32_t i = 1; i < left; i++) x[0] = max(abs(x[0]), abs(x[i])); }
}
