#pragma once
#include <cstdint>

/**
 * @brief Fill a vector with a constant: `x[i] = alpha`.
 *
 * NumPy equivalent: `x = np.full(n, alpha)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Number of elements.
 * @param alpha  Value to broadcast into every element.
 * @param x      Output vector of length `n`.
 */
template <typename T>
__device__ void set_const(uint32_t n, T alpha, T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) x[i] = alpha;
}

/**
 * @brief Fill a vector with a constant: `x[i] = alpha`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `x = np.full(N, alpha)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param alpha  Value to broadcast into every element.
 * @param x      Output vector of length `N`.
 */
template <typename T, uint32_t N>
__device__ void set_const(T alpha, T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) x[i] = alpha;
}
