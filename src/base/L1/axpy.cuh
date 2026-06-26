#pragma once
#include "../barrier.cuh"
#include <cstdint>

// shared body: AXPY in place `y = alpha*x + y`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void axpy_impl(Bar bar, uint32_t n, T alpha, T *x, T *y)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) y[i] = alpha*x[i] + y[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}

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
template <typename T, bool TRAILING_SYNC = true>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y)
{
    axpy_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, alpha, x, y);
}

// shared body: out-of-place AXPY `z = alpha*x + y`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void axpy_impl(Bar bar, uint32_t n, T alpha, T *x, T *y, T *z)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) z[i] = alpha*x[i] + y[i];
    if constexpr (TRAILING_SYNC) bar.sync();
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
template <typename T, bool TRAILING_SYNC = true>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y, T *z)
{
    axpy_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, alpha, x, y, z);
}

// shared body: AXPBY `z = alpha*x + beta*y`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void axpby_impl(Bar bar, uint32_t n, T alpha, T *x, T beta, T *y, T *z)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) z[i] = alpha*x[i] + beta*y[i];
    if constexpr (TRAILING_SYNC) bar.sync();
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
template <typename T, bool TRAILING_SYNC = true>
__device__ void axpby(uint32_t n, T alpha, T *x, T beta, T *y, T *z)
{
    axpby_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, alpha, x, beta, y, z);
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
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void axpy(T alpha, T *x, T *y)
{
    axpy_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, alpha, x, y);
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
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void axpy(T alpha, T *x, T *y, T *z)
{
    axpy_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, alpha, x, y, z);
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
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void axpby(T alpha, T *x, T beta, T *y, T *z)
{
    axpby_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, alpha, x, beta, y, z);
}

namespace warp {
    // Single-warp AXPY: one 32-lane warp strides over the vector (lane i handles
    // elements i, i+32, …). Elementwise, no cross-lane communication, no shared
    // scratch, no `__syncthreads`. For warp-per-problem kernels packing many small
    // updates into one block via independent warps. Full 32 lanes required.

    /**
     * @brief Scaled vector sum within one warp: `y = alpha * x + y` (AXPY), single-warp.
     *
     * One 32-lane warp computes the update with lanes striding over the `n` elements.
     * Elementwise, no inter-lane comms, no shared scratch, no `__syncthreads`.
     * Independent warps may run distinct problems concurrently. Full 32 lanes
     * required. NumPy equivalent: `y += alpha * x`.
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
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t i = lane; i < n; i += 32) y[i] = alpha*x[i] + y[i];
        __syncwarp();
    }

    /**
     * @brief Scaled vector sum within one warp: `y = alpha * x + y` (AXPY), single-warp, compile-time size.
     *
     * Compile-time-`N` overload of the single-warp AXPY. Elementwise, no shared
     * scratch, no `__syncthreads`. NumPy equivalent: `y += alpha * x`.
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
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t i = lane; i < N; i += 32) y[i] = alpha*x[i] + y[i];
        __syncwarp();
    }
}
