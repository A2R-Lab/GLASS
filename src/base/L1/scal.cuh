#pragma once
#include <cstdint>

/**
 * @brief Scale a vector in place: `x = alpha * x` (SCAL).
 *
 * NumPy equivalent: `x *= alpha`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Number of elements.
 * @param alpha  Scalar multiplier.
 * @param x      In/out vector of length `n` (overwritten with the result).
 */
template <typename T>
__device__ void scal(uint32_t n, T alpha, T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) x[i] = alpha*x[i];
}

/**
 * @brief Scale a vector into a second buffer: `y = alpha * x` (SCAL).
 *
 * Out-of-place variant that leaves `x` untouched. NumPy equivalent: `y = alpha * x`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Number of elements.
 * @param alpha  Scalar multiplier.
 * @param x      Input vector of length `n`.
 * @param y      Output vector of length `n` (overwritten with the result).
 */
template <typename T>
__device__ void scal(uint32_t n, T alpha, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) y[i] = alpha*x[i];
}

/**
 * @brief Scale a vector in place: `x = alpha * x` (SCAL), compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `x *= alpha`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param alpha  Scalar multiplier.
 * @param x      In/out vector of length `N` (overwritten with the result).
 */
template <typename T, uint32_t N>
__device__ void scal(T alpha, T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) x[i] = alpha*x[i];
}

/**
 * @brief Scale a vector into a second buffer: `y = alpha * x` (SCAL), compile-time size.
 *
 * Compile-time-`N` out-of-place overload. NumPy equivalent: `y = alpha * x`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param alpha  Scalar multiplier.
 * @param x      Input vector of length `N`.
 * @param y      Output vector of length `N` (overwritten with the result).
 */
template <typename T, uint32_t N>
__device__ void scal(T alpha, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) y[i] = alpha*x[i];
}
