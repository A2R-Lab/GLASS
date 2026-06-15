#pragma once
#include <cstdint>

/**
 * @brief Swap two vectors element-wise: `x ↔ y` (SWAP).
 *
 * Exchanges the contents of `x` and `y`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n`.
 * @param y  In/out vector of length `n`.
 */
template <typename T>
__device__ void swap(uint32_t n, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}

/**
 * @brief Swap two vectors element-wise: `x ↔ y` (SWAP), compile-time size.
 *
 * Compile-time-`N` overload of swap.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  In/out vector of length `N`.
 * @param y  In/out vector of length `N`.
 */
template <typename T, uint32_t N>
__device__ void swap(T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}
