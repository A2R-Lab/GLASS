#pragma once
#include <cstdint>

template <typename T>
__device__ void clip(uint32_t n, T *x, T *l, T *u)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size)
        x[i] = max(l[i], min(x[i], u[i]));
}

template <typename T, uint32_t N>
__device__ void clip(T *x, T *l, T *u)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size)
        x[i] = max(l[i], min(x[i], u[i]));
}
