#pragma once
#include <cstdint>
#include <math.h>

/**
 * @file nrm1_diff.cuh
 * @brief 1-norm of a difference: `‖x − y‖₁ = Σ|x[i] − y[i]|`.
 *
 * The `asum` sibling for a difference — the residual/convergence check GATO's
 * Schur loop wants without materializing `x − y` first. Inputs are read-only
 * (non-destructive). Block forms mirror `asum.cuh` (`low_memory` serial-tail and
 * `high_speed` warp-shuffle); `warp::nrm1_diff` mirrors `warp::dot` (returns the
 * scalar broadcast to every lane). Thread-count invariant.
 */

namespace low_memory {
    /**
     * @brief `out[0] = Σ|x[i] − y[i]|` (‖x−y‖₁), low-memory variant.
     *
     * Writes the per-element absolute differences into `out`, then thread 0
     * serially accumulates them into `out[0]`. `x`/`y` are untouched. NumPy
     * equivalent: `np.sum(np.abs(x - y))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n    Number of elements.
     * @param x    Input vector of length `n` (read-only).
     * @param y    Input vector of length `n` (read-only).
     * @param out  Length-`n` scratch/output buffer; the result lands in `out[0]`.
     */
    template <typename T, bool TRAILING_SYNC = true>
    __device__ void nrm1_diff(uint32_t n, const T *x, const T *y, T *out)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t i = rank; i < n; i += size) out[i] = abs(x[i] - y[i]);
        __syncthreads();
        if (rank == 0) { for (uint32_t i = 1; i < n; i++) out[0] += out[i]; }
        if constexpr (TRAILING_SYNC) __syncthreads();
    }
}

namespace high_speed {
    /**
     * @brief `out[0] = Σ|x[i] − y[i]|` (‖x−y‖₁), warp-shuffle variant.
     *
     * Reduces the absolute differences with a warp-shuffle reduction plus an
     * inter-warp reduction through shared scratch; the result lands in `out[0]`.
     * `x`/`y` are untouched. NumPy equivalent: `np.sum(np.abs(x - y))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n          Number of elements.
     * @param x          Input vector of length `n` (read-only).
     * @param y          Input vector of length `n` (read-only).
     * @param out        Output; the result lands in `out[0]`.
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
     */
    template <typename T, bool TRAILING_SYNC = true>
    __device__ void nrm1_diff(uint32_t n, const T *x, const T *y, T *out, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < n; i += size) val += abs(x[i] - y[i]);
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
        if constexpr (TRAILING_SYNC) __syncthreads();
    }

    /**
     * @brief `out[0] = Σ|x[i] − y[i]|` (‖x−y‖₁), warp-shuffle, compile-time size.
     *
     * Compile-time-`N` overload of the warp-shuffle `nrm1_diff`. `x`/`y` untouched.
     * NumPy equivalent: `np.sum(np.abs(x - y))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x          Input vector of length `N` (read-only).
     * @param y          Input vector of length `N` (read-only).
     * @param out        Output; the result lands in `out[0]`.
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
     */
    template <typename T, uint32_t N, bool TRAILING_SYNC = true>
    __device__ void nrm1_diff(const T *x, const T *y, T *out, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < N; i += size) val += abs(x[i] - y[i]);
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
        if constexpr (TRAILING_SYNC) __syncthreads();
    }
}

namespace warp {
    /**
     * @brief `‖x − y‖₁` within one warp: returns `Σ|x[i] − y[i]|` on every lane.
     *
     * One 32-lane warp forms the absolute differences and reduces with
     * `__shfl_down_sync`, then BROADCASTS the scalar total back to all 32 lanes via
     * `__shfl_sync` (from a register, never a shared re-read — immune to the
     * `__restrict__` stale-cache miscompile). `x`/`y` untouched; no shared scratch,
     * no `__syncthreads`. Full 32 lanes required; independent warps may run distinct
     * problems concurrently. NumPy equivalent: `np.sum(np.abs(x - y))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  Input vector of length `n` (read-only).
     * @param y  Input vector of length `n` (read-only).
     * @return `‖x − y‖₁`, identical on every lane.
     */
    template <typename T>
    __device__ T nrm1_diff(uint32_t n, const T *x, const T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < n; i += 32) val += abs(x[i] - y[i]);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
        return __shfl_sync(0xffffffffu, val, 0);
    }

    /**
     * @brief `‖x − y‖₁` within one warp, compile-time size (returns it on every lane).
     *
     * Compile-time-`N` overload of the single-warp `nrm1_diff`. No shared scratch,
     * no `__syncthreads`. NumPy equivalent: `np.sum(np.abs(x - y))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  Input vector of length `N` (read-only).
     * @param y  Input vector of length `N` (read-only).
     * @return `‖x − y‖₁`, identical on every lane.
     */
    template <typename T, uint32_t N>
    __device__ T nrm1_diff(const T *x, const T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < N; i += 32) val += abs(x[i] - y[i]);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
        return __shfl_sync(0xffffffffu, val, 0);
    }
}
