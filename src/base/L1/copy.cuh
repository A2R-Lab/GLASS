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

namespace warp {
    // Single-warp COPY: one 32-lane warp strides over the vector (lane i handles
    // elements i, i+32, …). Elementwise, no cross-lane communication, no shared
    // scratch, no `__syncthreads`. For warp-per-problem kernels packing many small
    // copies into one block via independent warps. Full 32 lanes required.

    /**
     * @brief Vector copy within one warp: `y = x` (COPY), single-warp.
     *
     * One 32-lane warp copies the vector with lanes striding over the `n` elements.
     * Elementwise, no inter-lane comms, no shared scratch, no `__syncthreads`.
     * Independent warps may run distinct problems concurrently. Full 32 lanes
     * required. NumPy equivalent: `y = x.copy()`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  Input vector of length `n`.
     * @param y  Output vector of length `n` (overwritten with a copy of `x`).
     */
    template <typename T>
    __device__ void copy(uint32_t n, T *x, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t i = lane; i < n; i += 32) y[i] = x[i];
        __syncwarp();
    }

    /**
     * @brief Vector copy within one warp: `y = x` (COPY), single-warp, compile-time size.
     *
     * Compile-time-`N` overload of the single-warp copy. Elementwise, no shared
     * scratch, no `__syncthreads`. NumPy equivalent: `y = x.copy()`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  Input vector of length `N`.
     * @param y  Output vector of length `N` (overwritten with a copy of `x`).
     */
    template <typename T, uint32_t N>
    __device__ void copy(T *x, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t i = lane; i < N; i += 32) y[i] = x[i];
        __syncwarp();
    }
}
