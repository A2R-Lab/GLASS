#pragma once
#include <cstdint>
#include <math.h>
#include "reduce.cuh"

// Shared body: ‖x‖₂ via square-in-place + halving reduce + sqrt; result in x[0].
// The shared engine for the cgrps:: plain nrm2 (the block surface intentionally
// exposes only the low_memory/high_speed/warp variants). Barrier policy supplies
// rank/size + the internal/trailing sync. The inner reduce skips its trailing
// barrier (rank 0 already holds the sum for the sqrt); the one trailing barrier
// rides on TRAILING_SYNC.
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void nrm2_impl(Bar bar, uint32_t n, T *x)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) x[i] *= x[i];
    bar.sync();
    reduce_impl<Bar, T, false>(bar, n, x);
    if (rank == 0) x[0] = sqrtf(x[0]);
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Euclidean (L2) norm: `x[0] = ‖x‖₂` (in-place, destructive), low-memory variant.
 *
 * Squares each element in place, then thread 0 serially sums them and takes
 * the square root, leaving the result in `x[0]` (the input is overwritten).
 * NumPy equivalent: `np.linalg.norm(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n`; the result lands in `x[0]`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void nrm2_lowmem(uint32_t n, T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) x[i] *= x[i];
    __syncthreads();
    if (rank == 0) {
        for (uint32_t i = 1; i < n; i++) x[0] += x[i];
        x[0] = sqrtf(x[0]);
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
/**
 * @brief Euclidean (L2) norm: `x[0] = ‖x‖₂` (in-place), warp-shuffle variant.
 *
 * Accumulates the sum of squares with a warp-shuffle reduction plus an
 * inter-warp reduction through shared scratch, then takes the square root;
 * the result lands in `x[0]`. NumPy equivalent: `np.linalg.norm(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n          Number of elements.
 * @param x          In/out vector of length `n`; the result lands in `x[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void nrm2_fast(uint32_t n, T *x, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < n; i += size) val += x[i]*x[i];
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
    uint32_t lane = rank & 31, warp = rank >> 5;
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        if (rank == 0) x[0] = sqrtf(val);
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Euclidean (L2) norm: `x[0] = ‖x‖₂`, warp-shuffle, compile-time size.
 *
 * Compile-time-`N` overload of the warp-shuffle L2 norm. NumPy equivalent:
 * `np.linalg.norm(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x          In/out vector of length `N`; the result lands in `x[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void nrm2_fast(T *x, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < N; i += size) val += x[i]*x[i];
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
    uint32_t lane = rank & 31, warp = rank >> 5;
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        if (rank == 0) x[0] = sqrtf(val);
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
namespace warp {
    /**
     * @brief Euclidean (L2) norm within one warp: returns `‖x‖₂` on every lane.
     *
     * Single-warp L2 norm, mirroring `warp::dot`: one 32-lane warp reduces the
     * per-lane sum-of-squares with `__shfl_down_sync`, BROADCASTS the total to all
     * 32 lanes via `__shfl_sync` (from a register — immune to the `__restrict__`
     * stale-cache miscompile), and each lane takes the square root. Non-destructive
     * (`x` untouched), no shared scratch, no `__syncthreads`; uses type-generic
     * `sqrt` (correct for `double`, unlike the block `sqrtf` forms). Full 32 lanes
     * required. NumPy equivalent: `np.linalg.norm(x)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  Input vector of length `n` (read-only).
     * @return `‖x‖₂`, identical on every lane.
     */
    template <typename T>
    __device__ T nrm2(uint32_t n, const T *x)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < n; i += 32) val += x[i]*x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
        return sqrt(__shfl_sync(0xffffffffu, val, 0));
    }

    /**
     * @brief Euclidean (L2) norm within one warp, compile-time size (returns it on every lane).
     *
     * Compile-time-`N` overload of the single-warp L2 norm. Non-destructive, no
     * shared scratch, no `__syncthreads`; type-generic `sqrt`. NumPy equivalent:
     * `np.linalg.norm(x)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  Input vector of length `N` (read-only).
     * @return `‖x‖₂`, identical on every lane.
     */
    template <typename T, uint32_t N>
    __device__ T nrm2(const T *x)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < N; i += 32) val += x[i]*x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
        return sqrt(__shfl_sync(0xffffffffu, val, 0));
    }
}
