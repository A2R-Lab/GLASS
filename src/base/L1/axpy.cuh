#pragma once
#include <cstdint>

template <typename T>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) y[i] = alpha*x[i] + y[i];
}

template <typename T>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y, T *z)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) z[i] = alpha*x[i] + y[i];
}

template <typename T>
__device__ void axpby(uint32_t n, T alpha, T *x, T beta, T *y, T *z)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) z[i] = alpha*x[i] + beta*y[i];
}

// compile-time size overloads
template <typename T, uint32_t N>
__device__ void axpy(T alpha, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) y[i] = alpha*x[i] + y[i];
}

template <typename T, uint32_t N>
__device__ void axpy(T alpha, T *x, T *y, T *z)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) z[i] = alpha*x[i] + y[i];
}

template <typename T, uint32_t N>
__device__ void axpby(T alpha, T *x, T beta, T *y, T *z)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) z[i] = alpha*x[i] + beta*y[i];
}
