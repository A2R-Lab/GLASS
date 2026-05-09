#pragma once
#include <cstdint>

template <typename T>
__device__ void loadIdentity(uint32_t n, T *A)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n*n; i += size) {
        uint32_t r = i % n, c = i / n;
        A[i] = static_cast<T>(r == c);
    }
}

template <typename T>
__device__ void addI(uint32_t n, T *A, T alpha)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n*n; i += size)
        if (i % n == i / n) A[i] += alpha;
}

template <typename T, uint32_t N>
__device__ void loadIdentity(T *A)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N*N; i += size) {
        uint32_t r = i % N, c = i / N;
        A[i] = static_cast<T>(r == c);
    }
}

template <typename T, uint32_t N>
__device__ void addI(T *A, T alpha)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N*N; i += size)
        if (i % N == i / N) A[i] += alpha;
}
