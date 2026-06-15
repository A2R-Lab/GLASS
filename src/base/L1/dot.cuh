#pragma once
#include <cstdint>
#include "reduce.cuh"

/**
 * @brief Inner product: `y[0] = x · y` (DOT), in-place.
 *
 * Multiplies the vectors element-wise into `y`, then runs a block-wide halving
 * reduce so the scalar result lands in `y[0]` (uses `y` as scratch — it is
 * overwritten). NumPy equivalent: `np.dot(x, y)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  Input vector of length `n`.
 * @param y  In/out vector of length `n`; the dot product lands in `y[0]`.
 */
// in-place: y = x·y (result in y[0]); uses halving reduce on y
template <typename T>
__device__ void dot(uint32_t n, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) y[i] *= x[i];
    __syncthreads();
    reduce<T>(n, y);
}

/**
 * @brief Inner product: `y[0] = x · y` (DOT), in-place, compile-time size.
 *
 * Compile-time-`N` overload; the scalar result lands in `y[0]` (uses `y` as
 * scratch). NumPy equivalent: `np.dot(x, y)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  Input vector of length `N`.
 * @param y  In/out vector of length `N`; the dot product lands in `y[0]`.
 */
template <typename T, uint32_t N>
__device__ void dot(T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) y[i] *= x[i];
    __syncthreads();
    reduce<T, N>(y);
}

namespace low_memory {
    /**
     * @brief Inner product: `out[0] = x · y` (DOT), low-memory variant.
     *
     * Writes the element-wise products into `out`, then thread 0 serially
     * accumulates them into `out[0]`, leaving `x` and `y` untouched. NumPy
     * equivalent: `np.dot(x, y)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n    Number of elements.
     * @param x    Input vector of length `n`.
     * @param y    Input vector of length `n`.
     * @param out  Length-`n` scratch/output buffer; the result lands in `out[0]`.
     */
    // out: length-n scratch; result in out[0]
    template <typename T>
    __device__ void dot(uint32_t n, T *x, T *y, T *out)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t i = rank; i < n; i += size) out[i] = x[i]*y[i];
        __syncthreads();
        if (rank == 0) { for (uint32_t i = 1; i < n; i++) out[0] += out[i]; }
        __syncthreads();
    }
}

namespace high_speed {
    /**
     * @brief Inner product: `out[0] = x · y` (DOT), warp-shuffle variant.
     *
     * Accumulates the element-wise products with a warp-shuffle reduction plus
     * an inter-warp reduction through shared scratch, leaving `x` and `y`
     * untouched. NumPy equivalent: `np.dot(x, y)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n          Number of elements.
     * @param x          Input vector of length `n`.
     * @param y          Input vector of length `n`.
     * @param out        Output buffer; the result lands in `out[0]`.
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
     */
    // s_scratch: ceil(blockDim/32)*sizeof(T); result in out[0]
    template <typename T>
    __device__ void dot(uint32_t n, T *x, T *y, T *out, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < n; i += size) val += x[i]*y[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) out[0] = val;
        }
        __syncthreads();
    }

    /**
     * @brief Inner product: `out[0] = x · y` (DOT), warp-shuffle, compile-time size.
     *
     * Compile-time-`N` overload of the warp-shuffle dot product. NumPy
     * equivalent: `np.dot(x, y)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x          Input vector of length `N`.
     * @param y          Input vector of length `N`.
     * @param out        Output buffer; the result lands in `out[0]`.
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
     */
    template <typename T, uint32_t N>
    __device__ void dot(T *x, T *y, T *out, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < N; i += size) val += x[i]*y[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) out[0] = val;
        }
        __syncthreads();
    }
}
