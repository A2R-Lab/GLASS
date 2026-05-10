#pragma once
#include <cstdint>

// Per-thread compile-time strided dot product.
// Computes sum(x[i*SX] * y[i*SY]) for i in 0..N-1.
// No block-wide reduction — intended for use inside already-parallelized loops
// (e.g., GRiD generated kernels where the outer loop is already thread-parallel).
// With SX=SY=1 this is a plain scalar dot.  The inner loop is fully unrolled
// by the compiler since N, SX, SY are all compile-time constants.

template <typename T, uint32_t N, uint32_t SX = 1, uint32_t SY = 1>
__device__ T dot_strided(const T* x, const T* y)
{
    T res = static_cast<T>(0);
    for (uint32_t i = 0; i < N; i++)
        res += x[i * SX] * y[i * SY];
    return res;
}

template <typename T, uint32_t N, uint32_t SX = 1, uint32_t SY = 1>
__device__ void dot_strided(const T* x, const T* y, T* out)
{
    *out = dot_strided<T, N, SX, SY>(x, y);
}
