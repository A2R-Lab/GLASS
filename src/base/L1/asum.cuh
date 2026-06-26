#pragma once
#include <cstdint>
#include <math.h>
#include "reduce.cuh"

// Shared body: Σ|x[i]| via |x|→out + halving reduce; result in out[0] (x intact).
// The shared engine for the cgrps:: plain asum (the block surface intentionally
// exposes only the low_memory/high_speed/warp variants). Barrier policy supplies
// rank/size + the internal/trailing sync.
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void asum_impl(Bar bar, uint32_t n, T *x, T *out)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) out[i] = abs(x[i]);
    bar.sync();
    reduce_impl<Bar, T, TRAILING_SYNC>(bar, n, out);
}

namespace low_memory {
    /**
     * @brief Sum of absolute values: `out[0] = Σ|x[i]|` (ASUM), low-memory variant.
     *
     * Writes the per-element absolute values into `out`, then thread 0 serially
     * accumulates them into `out[0]`. NumPy equivalent: `np.sum(np.abs(x))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n    Number of elements.
     * @param x    Input vector of length `n`.
     * @param out  Length-`n` scratch/output buffer; the result lands in `out[0]`.
     */
    template <typename T, bool TRAILING_SYNC = true>
    __device__ void asum(uint32_t n, T *x, T *out)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t i = rank; i < n; i += size) out[i] = abs(x[i]);
        __syncthreads();
        if (rank == 0) { for (uint32_t i = 1; i < n; i++) out[0] += out[i]; }
        if constexpr (TRAILING_SYNC) __syncthreads();
    }
}

namespace high_speed {
    /**
     * @brief Sum of absolute values: `x[0] = Σ|x[i]|` (ASUM), warp-shuffle variant.
     *
     * Computes the absolute-value sum with a warp-shuffle reduction plus an
     * inter-warp reduction through shared scratch. The result is written to
     * `x[0]` (destructive — overwrites the input). NumPy equivalent:
     * `np.sum(np.abs(x))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n          Number of elements.
     * @param x          In/out vector of length `n`; result lands in `x[0]`.
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
     */
    // s_scratch: ceil(blockDim/32)*sizeof(T); result in x[0] (overwrites input!)
    template <typename T, bool TRAILING_SYNC = true>
    __device__ void asum(uint32_t n, T *x, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < n; i += size) val += abs(x[i]);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) x[0] = val;
        }
        if constexpr (TRAILING_SYNC) __syncthreads();
    }

    /**
     * @brief Sum of absolute values: `x[0] = Σ|x[i]|` (ASUM), compile-time size.
     *
     * Compile-time-`N` overload of the warp-shuffle ASUM; the result is written
     * to `x[0]` (destructive). NumPy equivalent: `np.sum(np.abs(x))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x          In/out vector of length `N`; result lands in `x[0]`.
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
     */
    template <typename T, uint32_t N, bool TRAILING_SYNC = true>
    __device__ void asum(T *x, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < N; i += size) val += abs(x[i]);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) x[0] = val;
        }
        if constexpr (TRAILING_SYNC) __syncthreads();
    }
}

namespace warp {
    /**
     * @brief Sum of absolute values within one warp: returns `Σ|x[i]|` on every lane.
     *
     * Single-warp ASUM, mirroring `warp::dot`: one 32-lane warp reduces the
     * per-lane absolute-value partials with `__shfl_down_sync` and BROADCASTS the
     * total to all 32 lanes via `__shfl_sync` (from a register — immune to the
     * `__restrict__` stale-cache miscompile). Non-destructive (`x` untouched), no
     * shared scratch, no `__syncthreads`. Full 32 lanes required; independent warps
     * may run distinct problems concurrently. NumPy equivalent: `np.sum(np.abs(x))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  Input vector of length `n` (read-only).
     * @return `Σ|x[i]|`, identical on every lane.
     */
    template <typename T>
    __device__ T asum(uint32_t n, const T *x)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < n; i += 32) val += abs(x[i]);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
        return __shfl_sync(0xffffffffu, val, 0);
    }

    /**
     * @brief Sum of absolute values within one warp, compile-time size (returns it on every lane).
     *
     * Compile-time-`N` overload of the single-warp ASUM. Non-destructive, no shared
     * scratch, no `__syncthreads`. NumPy equivalent: `np.sum(np.abs(x))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  Input vector of length `N` (read-only).
     * @return `Σ|x[i]|`, identical on every lane.
     */
    template <typename T, uint32_t N>
    __device__ T asum(const T *x)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < N; i += 32) val += abs(x[i]);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
        return __shfl_sync(0xffffffffu, val, 0);
    }
}
