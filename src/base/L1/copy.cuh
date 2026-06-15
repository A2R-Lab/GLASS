#pragma once
#include <cstdint>

/**
 * @brief Vector copy: `y = x` (COPY).
 *
 * Each thread in the block strides over the `n` elements. NumPy equivalent:
 * `y = x.copy()`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  Input vector of length `n`.
 * @param y  Output vector of length `n` (overwritten with a copy of `x`).
 */
template <typename T>
__device__ void copy(uint32_t n, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) y[i] = x[i];
}

/**
 * @brief Scaled vector copy: `y = alpha * x`.
 *
 * Copies `x` into `y` while scaling by `alpha`. NumPy equivalent:
 * `y = alpha * x`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Number of elements.
 * @param alpha  Scalar multiplier.
 * @param x      Input vector of length `n`.
 * @param y      Output vector of length `n` (overwritten with the result).
 */
template <typename T>
__device__ void copy(uint32_t n, T alpha, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) y[i] = alpha*x[i];
}

/**
 * @brief Vector copy: `y = x` (COPY), compile-time size.
 *
 * Compile-time-`N` overload of copy. NumPy equivalent: `y = x.copy()`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  Input vector of length `N`.
 * @param y  Output vector of length `N` (overwritten with a copy of `x`).
 */
template <typename T, uint32_t N>
__device__ void copy(T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) y[i] = x[i];
}

/**
 * @brief Scaled vector copy: `y = alpha * x`, compile-time size.
 *
 * Compile-time-`N` overload of scaled copy. NumPy equivalent: `y = alpha * x`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param alpha  Scalar multiplier.
 * @param x      Input vector of length `N`.
 * @param y      Output vector of length `N` (overwritten with the result).
 */
template <typename T, uint32_t N>
__device__ void copy(T alpha, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) y[i] = alpha*x[i];
}
