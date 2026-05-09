#pragma once
#include <cstdint>

// A += alpha * x * y^T  (A is m×n column-major)
template <typename T>
__device__ void ger(uint32_t m, uint32_t n, T alpha, T *x, T *y, T *A)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t col = 0; col < n; col++) {
        T ay = alpha * y[col];
        for (uint32_t row = rank; row < m; row += size)
            A[row + col*m] += ay * x[row];
    }
}

template <typename T, uint32_t M, uint32_t N>
__device__ void ger(T alpha, T *x, T *y, T *A)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t col = 0; col < N; col++) {
        T ay = alpha * y[col];
        for (uint32_t row = rank; row < M; row += size)
            A[row + col*M] += ay * x[row];
    }
}
