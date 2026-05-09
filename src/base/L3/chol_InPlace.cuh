#pragma once
#include <cstdint>
#include <math.h>

template <typename T>
__device__ void cholDecomp_InPlace(uint32_t n, T *s_A)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t row = 0; row < n; row++) {
        if (rank == 0) {
            T sum = static_cast<T>(0);
            T val = s_A[n*row + row];
            for (int32_t rl = 0; rl < (int32_t)row; rl++) sum += s_A[rl*n + row]*s_A[rl*n + row];
            s_A[row*n + row] = sqrtf(val - sum);
        }
        __syncthreads();
        for (uint32_t col = rank + row + 1; col < n; col += size) {
            T sum = static_cast<T>(0);
            for (uint32_t kk = 0; kk < row; kk++) sum += s_A[kk*n + col]*s_A[kk*n + row];
            s_A[row*n + col] = (static_cast<T>(1)/s_A[row*n + row])*(s_A[row*n + col] - sum);
        }
        __syncthreads();
    }
}

template <typename T, uint32_t N>
__device__ void cholDecomp_InPlace(T *s_A)
{
    cholDecomp_InPlace<T>(N, s_A);
}
