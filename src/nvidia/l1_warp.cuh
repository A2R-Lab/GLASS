#pragma once
#include <cstdint>
#include <cub/cub.cuh>

// ─── glass::nvidia::warp:: — CUB WarpReduce-backed reductions ────────────────
//
// The vendor tier below block scope: the same reduce/dot/nrm2 signatures as
// the block forms above, executed by ONE FULL 32-lane warp per problem via
// cub::WarpReduce. Each calling warp reduces its own operands, so a
// multi-warp block packs one independent problem per warp (the glass::warp::
// packing model, vendor-backed). Requirements mirror glass::argmax_fast:
// FULL warps only — a partial trailing warp would deadlock the shuffle
// ladder — and compile-time sizes only. Sync semantics are warp-scope:
// TRAILING_SYNC emits __syncwarp() (not __syncthreads()); the leading
// __syncwarp() before the CUB call is not gated (TempStorage quiescence).
// Scratch is per-warp: warp_reduce_scratch_bytes<T>() bytes at a per-warp
// offset (typically 0/empty for the full-warp shuffle specialization, but
// the API keeps CUB's TempStorage contract honest).

namespace warp {

/**
 * @brief Warp-level sum reduction backed by CUB WarpReduce.
 *
 * Sums all N elements of `x` across the calling (full) warp; lane 0 writes
 * the total to `x[0]` (in place). One problem per warp — pass per-warp `x`
 * and `s_scratch` in multi-warp blocks. Compile-time sizes only. NumPy
 * equivalent: `x[0] = np.sum(x)`.
 *
 * @tparam T             Scalar type.
 * @tparam N             Number of elements.
 * @tparam TRAILING_SYNC Emit a trailing __syncwarp() before return (default true).
 * @param  x             Input array of length N; result lands in x[0].
 * @param  s_scratch     Per-warp scratch >= warp_reduce_scratch_bytes<T>() bytes.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void reduce(T *x, T *s_scratch)
{
    using WarpReduce = cub::WarpReduce<T>;
    const uint32_t lane = threadIdx.x & 31u;
    T lane_sum = static_cast<T>(0);
    for (uint32_t i = lane; i < N; i += 32u)
        lane_sum += x[i];
    __syncwarp();
    T warp_sum = WarpReduce(*reinterpret_cast<typename WarpReduce::TempStorage*>(s_scratch))
                     .Sum(lane_sum);
    if (lane == 0) x[0] = warp_sum;
    if constexpr (TRAILING_SYNC) {
        __syncwarp();
    }
}

/**
 * @brief Warp-level dot product backed by CUB WarpReduce.
 *
 * Computes the inner product of the N-element vectors `x` and `y`; lane 0
 * writes the scalar result to `*out`. One problem per (full) warp.
 * Compile-time sizes only. NumPy equivalent: `*out = np.dot(x, y)`.
 *
 * @tparam T             Scalar type.
 * @tparam N             Number of elements.
 * @tparam TRAILING_SYNC Emit a trailing __syncwarp() before return (default true).
 * @param  x             First input vector (length N).
 * @param  y             Second input vector (length N).
 * @param  out           Output pointer for the resulting scalar.
 * @param  s_scratch     Per-warp scratch >= warp_reduce_scratch_bytes<T>() bytes.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void dot(T *x, T *y, T *out, T *s_scratch)
{
    using WarpReduce = cub::WarpReduce<T>;
    const uint32_t lane = threadIdx.x & 31u;
    T lane_sum = static_cast<T>(0);
    for (uint32_t i = lane; i < N; i += 32u)
        lane_sum += x[i] * y[i];
    __syncwarp();
    T warp_sum = WarpReduce(*reinterpret_cast<typename WarpReduce::TempStorage*>(s_scratch))
                     .Sum(lane_sum);
    if (lane == 0) *out = warp_sum;
    if constexpr (TRAILING_SYNC) {
        __syncwarp();
    }
}

/**
 * @brief Warp-level Euclidean (L2) norm backed by CUB WarpReduce.
 *
 * Sums the squares of the N elements of `x` and writes the square root to
 * `*out` from lane 0. One problem per (full) warp. Compile-time sizes only.
 * NumPy equivalent: `*out = np.linalg.norm(x)`.
 *
 * @tparam T             Scalar type.
 * @tparam N             Number of elements.
 * @tparam TRAILING_SYNC Emit a trailing __syncwarp() before return (default true).
 * @param  x             Input vector (length N).
 * @param  out           Output pointer for the resulting scalar norm.
 * @param  s_scratch     Per-warp scratch >= warp_reduce_scratch_bytes<T>() bytes.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void nrm2(T *x, T *out, T *s_scratch)
{
    using WarpReduce = cub::WarpReduce<T>;
    const uint32_t lane = threadIdx.x & 31u;
    T lane_sum = static_cast<T>(0);
    for (uint32_t i = lane; i < N; i += 32u)
        lane_sum += x[i] * x[i];
    __syncwarp();
    T warp_sum = WarpReduce(*reinterpret_cast<typename WarpReduce::TempStorage*>(s_scratch))
                     .Sum(lane_sum);
    if (lane == 0) *out = sqrt(warp_sum);
    if constexpr (TRAILING_SYNC) {
        __syncwarp();
    }
}

/**
 * @brief Per-warp scratch bytes for the warp reduce/dot/nrm2 (host-callable).
 *
 * Returns `sizeof(cub::WarpReduce<T>::TempStorage)`. For the full-warp
 * shuffle specialization this is trivially small, but callers must still
 * honor it (CUB's TempStorage contract). constexpr.
 *
 * @tparam T Scalar type.
 * @return Required per-warp scratch size in bytes.
 */
template <typename T>
inline constexpr std::size_t warp_reduce_scratch_bytes()
{
    return sizeof(typename cub::WarpReduce<T>::TempStorage);
}

}  // namespace warp
