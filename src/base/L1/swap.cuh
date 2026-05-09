#pragma once
#include <cstdint>

template <typename T>
__device__ void swap(uint32_t n, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}

template <typename T, uint32_t N>
__device__ void swap(T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}
