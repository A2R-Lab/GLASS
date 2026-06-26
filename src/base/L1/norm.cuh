#pragma once
#include <cstdint>
#include <math.h>
#include "reduce.cuh"

/**
 * @brief Euclidean (L2) norm into a separate buffer: `out[0] = ‖a‖₂`, low-memory variant.
 *
 * Writes the per-element squares into `out`, then thread 0 serially sums
 * them and takes the square root, leaving `a` untouched. NumPy equivalent:
 * `np.linalg.norm(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N    Number of elements.
 * @param a    Input vector of length `N`.
 * @param out  Length-`N` scratch/output buffer; the result lands in `out[0]`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void vector_norm_lowmem(uint32_t N, T *a, T *out)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size) out[i] = a[i]*a[i];
    __syncthreads();
    if (rank == 0) {
        for (uint32_t i = 1; i < N; i++) out[0] += out[i];
        out[0] = sqrtf(out[0]);
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
/**
 * @brief Euclidean (L2) norm into a separate buffer: `out[0] = ‖a‖₂`, warp-shuffle variant.
 *
 * Accumulates the sum of squares with a warp-shuffle reduction plus an
 * inter-warp reduction through shared scratch, then takes the square root,
 * leaving `a` untouched. NumPy equivalent: `np.linalg.norm(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N          Number of elements.
 * @param a          Input vector of length `N`.
 * @param out        Output buffer; the result lands in `out[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void vector_norm_fast(uint32_t N, T *a, T *out, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < N; i += size) val += a[i]*a[i];
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
    uint32_t lane = rank & 31, warp = rank >> 5;
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        if (rank == 0) out[0] = sqrtf(val);
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Euclidean (L2) norm into a separate buffer: `out[0] = ‖a‖₂`, warp-shuffle, compile-time size.
 *
 * Compile-time-`N` overload of the warp-shuffle vector norm; leaves `a`
 * untouched. NumPy equivalent: `np.linalg.norm(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a          Input vector of length `N`.
 * @param out        Output buffer; the result lands in `out[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void vector_norm_fast(T *a, T *out, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < N; i += size) val += a[i]*a[i];
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
    uint32_t lane = rank & 31, warp = rank >> 5;
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        if (rank == 0) out[0] = sqrtf(val);
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
