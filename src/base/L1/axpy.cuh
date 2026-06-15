#pragma once
#include <cstdint>

/**
 * @brief Scaled vector sum: `y = alpha * x + y` (AXPY).
 *
 * Each thread in the block strides over the `n` elements. NumPy equivalent:
 * `y += alpha * x`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Number of elements.
 * @param alpha  Scalar multiplier.
 * @param x      Input vector of length `n`.
 * @param y      In/out vector of length `n` (overwritten with the result).
 */
template <typename T>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) y[i] = alpha*x[i] + y[i];
}

/**
 * @brief Scaled vector sum into a third buffer: `z = alpha * x + y` (AXPY).
 *
 * Out-of-place variant that leaves `x` and `y` untouched. NumPy equivalent:
 * `z = alpha * x + y`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Number of elements.
 * @param alpha  Scalar multiplier.
 * @param x      Input vector of length `n`.
 * @param y      Input vector of length `n`.
 * @param z      Output vector of length `n` (overwritten with the result).
 */
template <typename T>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y, T *z)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) z[i] = alpha*x[i] + y[i];
}

/**
 * @brief Doubly-scaled vector sum: `z = alpha * x + beta * y` (AXPBY).
 *
 * Out-of-place generalization of AXPY with an independent scale on `y`. NumPy
 * equivalent: `z = alpha * x + beta * y`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Number of elements.
 * @param alpha  Scalar multiplier on `x`.
 * @param x      Input vector of length `n`.
 * @param beta   Scalar multiplier on `y`.
 * @param y      Input vector of length `n`.
 * @param z      Output vector of length `n` (overwritten with the result).
 */
template <typename T>
__device__ void axpby(uint32_t n, T alpha, T *x, T beta, T *y, T *z)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) z[i] = alpha*x[i] + beta*y[i];
}

// compile-time size overloads
/**
 * @brief Scaled vector sum: `y = alpha * x + y` (AXPY), compile-time size.
 *
 * Compile-time-`N` overload of AXPY. NumPy equivalent: `y += alpha * x`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param alpha  Scalar multiplier.
 * @param x      Input vector of length `N`.
 * @param y      In/out vector of length `N` (overwritten with the result).
 */
template <typename T, uint32_t N>
__device__ void axpy(T alpha, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) y[i] = alpha*x[i] + y[i];
}

/**
 * @brief Scaled vector sum into a third buffer: `z = alpha * x + y` (AXPY), compile-time size.
 *
 * Compile-time-`N` out-of-place overload. NumPy equivalent: `z = alpha * x + y`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param alpha  Scalar multiplier.
 * @param x      Input vector of length `N`.
 * @param y      Input vector of length `N`.
 * @param z      Output vector of length `N` (overwritten with the result).
 */
template <typename T, uint32_t N>
__device__ void axpy(T alpha, T *x, T *y, T *z)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) z[i] = alpha*x[i] + y[i];
}

/**
 * @brief Doubly-scaled vector sum: `z = alpha * x + beta * y` (AXPBY), compile-time size.
 *
 * Compile-time-`N` overload of AXPBY. NumPy equivalent: `z = alpha * x + beta * y`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param alpha  Scalar multiplier on `x`.
 * @param x      Input vector of length `N`.
 * @param beta   Scalar multiplier on `y`.
 * @param y      Input vector of length `N`.
 * @param z      Output vector of length `N` (overwritten with the result).
 */
template <typename T, uint32_t N>
__device__ void axpby(T alpha, T *x, T beta, T *y, T *z)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) z[i] = alpha*x[i] + beta*y[i];
}
