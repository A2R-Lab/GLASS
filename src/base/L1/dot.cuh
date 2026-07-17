#pragma once
#include "../barrier.cuh"
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
// Shared body: y = x·y (result in y[0]); element-wise multiply then halving
// reduce on y. The trailing barrier rides on reduce_impl's TRAILING_SYNC.
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void dot_impl(Bar bar, uint32_t n, T *x, T *y)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) y[i] *= x[i];
    bar.sync();
    reduce_impl<Bar, T, TRAILING_SYNC>(bar, n, y);
}

// in-place: y = x·y (result in y[0]); uses halving reduce on y
template <typename T, bool TRAILING_SYNC = true>
__device__ void dot(uint32_t n, T *x, T *y)
{
    dot_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, x, y);
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
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void dot(T *x, T *y)
{
    dot_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, x, y);
}

/**
 * @brief Inner product: `out[0] = x · y` (DOT), low-memory variant.
 *
 * Writes the element-wise products into `out`, then thread 0 serially
 * accumulates them into `out[0]`, leaving `x` and `y` untouched. NumPy
 * equivalent: `np.dot_lowmem(x, y)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n    Number of elements.
 * @param x    Input vector of length `n`.
 * @param y    Input vector of length `n`.
 * @param out  Length-`n` scratch/output buffer; the result lands in `out[0]`.
 */
// out: length-n scratch; result in out[0]
template <typename T, bool TRAILING_SYNC = true>
__device__ void dot_lowmem(uint32_t n, T *x, T *y, T *out)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) out[i] = x[i]*y[i];
    __syncthreads();
    if (rank == 0) { for (uint32_t i = 1; i < n; i++) out[0] += out[i]; }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
namespace thread {

    /**
     * @brief Inner product on one thread: returns `x · y`, single-thread.
     *
     * Serially accumulates the element-wise products. Inputs untouched; no shared
     * scratch, no shuffles, no barriers. NumPy equivalent: `np.dot(x, y)`.
     * Works well on low DOF problems with small vectors that fit into a register
     * Parallelism can be leveraged by scaling thread count.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  Input vector of length `n`.
     * @param y  Input vector of length `n`.
     * @return The inner product `x · y`.
     */
    template <typename T>
    __device__ T dot(uint32_t n, const T *x, const T *y)
    {
        T val = static_cast<T>(0);
        for (uint32_t i = 0; i < n; i++) val += x[i]*y[i];
        return val;
    }

    /**
     * @brief Inner product on one thread: returns `x · y`, single-thread, compile-time size.
     *
     * Compile-time-`N` overload; the trip count folds and the loop unrolls, so `x`
     * and `y` may be thread-local register arrays. NumPy equivalent: `np.dot(x, y)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  Input vector of length `N`.
     * @param y  Input vector of length `N`.
     * @return The inner product `x · y`.
     */
    template <typename T, uint32_t N>
    __device__ T dot(const T *x, const T *y)
    {
        T val = static_cast<T>(0);
        for (uint32_t i = 0; i < N; i++) val += x[i]*y[i];
        return val;
    }
}

namespace warp {
    // Single-warp dot products: one 32-lane warp owns the reduction (raw __shfl,
    // no shared scratch). For warp-per-problem kernels packing many small dots
    // into one block via independent warps (threadIdx.y selects the warp). The
    // caller must run a full 32-lane warp (mask 0xffffffff). Distinct from
    // dot_fast, which is block-scoped (warp-shuffle + shared inter-warp
    // combine). The result is broadcast to EVERY lane, so the value is usable
    // immediately by all lanes without a follow-up read.

    /**
     * @brief Inner product within one warp: returns `x · y` on every lane, single-warp.
     *
     * One 32-lane warp forms the element-wise products and reduces them with
     * `__shfl_down_sync`, then BROADCASTS the scalar total back to all 32 lanes via
     * `__shfl_sync` (from a lane's register, never a shared re-read — immune to the
     * `__restrict__` stale-cache miscompile). Inputs are left untouched; no shared
     * scratch, no `__syncthreads`. Full 32 lanes required; independent warps may run
     * distinct problems concurrently. NumPy equivalent: `np.dot(x, y)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  Input vector of length `n`.
     * @param y  Input vector of length `n`.
     * @return The inner product `x · y`, identical on every lane.
     */
    template <typename T>
    __device__ T dot(uint32_t n, T *x, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < n; i += 32) val += x[i]*y[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
        return __shfl_sync(0xffffffffu, val, 0);
    }

    /**
     * @brief Inner product within one warp: returns `x · y` on every lane, single-warp, compile-time size.
     *
     * Compile-time-`N` overload of the single-warp dot. Reduces with
     * `__shfl_down_sync` and broadcasts the total to all 32 lanes from a register.
     * No shared scratch, no `__syncthreads`. NumPy equivalent: `np.dot(x, y)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  Input vector of length `N`.
     * @param y  Input vector of length `N`.
     * @return The inner product `x · y`, identical on every lane.
     */
    template <typename T, uint32_t N>
    __device__ T dot(T *x, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < N; i += 32) val += x[i]*y[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
        return __shfl_sync(0xffffffffu, val, 0);
    }
}

/**
 * @brief Shared-scratch size in bytes for the `dot_fast` ops.
 *
 * The warp-shuffle dot combines across warps through one scratch slot per warp:
 * `ceil(block_threads / 32)` elements of `T`. Allocate
 * `dot_fast_scratch_bytes<T>(block_threads)` for the `s_scratch` argument.
 *
 * @tparam T  Scalar type.
 * @param block_threads  Number of threads in the launching block.
 * @return Bytes to allocate for `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t dot_fast_scratch_bytes(uint32_t block_threads) { return static_cast<std::size_t>((block_threads + 31) / 32) * sizeof(T); }

/**
 * @brief Inner product: `out[0] = x · y` (DOT), warp-shuffle variant.
 *
 * Accumulates the element-wise products with a warp-shuffle reduction plus
 * an inter-warp reduction through shared scratch, leaving `x` and `y`
 * untouched. NumPy equivalent: `np.dot_fast(x, y)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n          Number of elements.
 * @param x          Input vector of length `n`.
 * @param y          Input vector of length `n`.
 * @param out        Output buffer; the result lands in `out[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
 */
// s_scratch: ceil(blockDim/32)*sizeof(T); result in out[0]
template <typename T, bool TRAILING_SYNC = true>
__device__ void dot_fast(uint32_t n, T *x, T *y, T *out, T *s_scratch)
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
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Inner product: `out[0] = x · y` (DOT), warp-shuffle, compile-time size.
 *
 * Compile-time-`N` overload of the warp-shuffle dot product. NumPy
 * equivalent: `np.dot_fast(x, y)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x          Input vector of length `N`.
 * @param y          Input vector of length `N`.
 * @param out        Output buffer; the result lands in `out[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void dot_fast(T *x, T *y, T *out, T *s_scratch)
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
    if constexpr (TRAILING_SYNC) __syncthreads();
}
