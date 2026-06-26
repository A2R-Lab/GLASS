#pragma once
#include "../barrier.cuh"
#include <cstdint>

// shared body: in-place scale `x = alpha*x`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void scal_impl(Bar bar, uint32_t n, T alpha, T *x)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) x[i] = alpha*x[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}

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
template <typename T, bool TRAILING_SYNC = true>
__device__ void scal(uint32_t n, T alpha, T *x)
{
    scal_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, alpha, x);
}

// shared body: out-of-place scale `y = alpha*x`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void scal_impl(Bar bar, uint32_t n, T alpha, T *x, T *y)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) y[i] = alpha*x[i];
    if constexpr (TRAILING_SYNC) bar.sync();
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
template <typename T, bool TRAILING_SYNC = true>
__device__ void scal(uint32_t n, T alpha, T *x, T *y)
{
    scal_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, alpha, x, y);
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
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void scal(T alpha, T *x)
{
    scal_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, alpha, x);
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
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void scal(T alpha, T *x, T *y)
{
    scal_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, alpha, x, y);
}

namespace warp {
    // Single-warp SCAL: one 32-lane warp strides over the vector (lane i handles
    // elements i, i+32, …). Elementwise, no cross-lane communication, no shared
    // scratch, no `__syncthreads`. For warp-per-problem kernels packing many small
    // scalings into one block via independent warps. Full 32 lanes required.

    /**
     * @brief Scale a vector in place within one warp: `x = alpha * x` (SCAL), single-warp.
     *
     * One 32-lane warp scales the vector with lanes striding over the `n` elements.
     * Elementwise, no inter-lane comms, no shared scratch, no `__syncthreads`.
     * Independent warps may run distinct problems concurrently. Full 32 lanes
     * required. NumPy equivalent: `x *= alpha`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n      Number of elements.
     * @param alpha  Scalar multiplier.
     * @param x      In/out vector of length `n` (overwritten with the result).
     */
    template <typename T>
    __device__ void scal(uint32_t n, T alpha, T *x)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t i = lane; i < n; i += 32) x[i] = alpha*x[i];
        __syncwarp();
    }

    /**
     * @brief Scale a vector in place within one warp: `x = alpha * x` (SCAL), single-warp, compile-time size.
     *
     * Compile-time-`N` overload of the single-warp scale. Elementwise, no shared
     * scratch, no `__syncthreads`. NumPy equivalent: `x *= alpha`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param alpha  Scalar multiplier.
     * @param x      In/out vector of length `N` (overwritten with the result).
     */
    template <typename T, uint32_t N>
    __device__ void scal(T alpha, T *x)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t i = lane; i < N; i += 32) x[i] = alpha*x[i];
        __syncwarp();
    }
}
