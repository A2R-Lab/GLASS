#pragma once
#include <cstdint>

// Solve lower-triangular Lx=b in-place (column-major L, result overwrites b)
template <typename T>
__device__ void trsm(uint32_t n, T *L, T *b)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t col = 0; col < n; col++) {
        if (rank == 0) b[col] /= L[col*n + col];
        __syncthreads();
        T factor = b[col];
        for (uint32_t row = rank + col + 1; row < n; row += size)
            b[row] -= L[col*n + row] * factor;
        __syncthreads();
    }
}

template <typename T, uint32_t N>
__device__ void trsm(T *L, T *b)
{
    trsm<T>(N, L, b);
}
