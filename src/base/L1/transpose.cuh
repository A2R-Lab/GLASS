#pragma once
#include <cstdint>

// out-of-place: NxM column-major a → b
template <typename T>
__device__ void transpose(uint32_t N, uint32_t M, T *a, T *b)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N*M; i += size) {
        uint32_t col = i / N, row = i % N;
        b[col + M*row] = a[row + N*col];
    }
    __syncthreads();
}

// in-place: NxN column-major
template <typename T>
__device__ void transpose(uint32_t N, T *a)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t idx = rank; idx < N*N; idx += size) {
        uint32_t i = idx % N, j = idx / N;
        if (i < j) {
            uint32_t swap = i*N + j;
            T tmp = a[idx]; a[idx] = a[swap]; a[swap] = tmp;
        }
    }
    __syncthreads();
}

// compile-time out-of-place
template <typename T, uint32_t N, uint32_t M>
__device__ void transpose(T *a, T *b)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N*M; i += size) {
        uint32_t col = i / N, row = i % N;
        b[col + M*row] = a[row + N*col];
    }
    __syncthreads();
}

// compile-time in-place NxN
template <typename T, uint32_t N>
__device__ void transpose(T *a)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t idx = rank; idx < N*N; idx += size) {
        uint32_t i = idx % N, j = idx / N;
        if (i < j) {
            uint32_t sw = i*N + j;
            T tmp = a[idx]; a[idx] = a[sw]; a[sw] = tmp;
        }
    }
    __syncthreads();
}
