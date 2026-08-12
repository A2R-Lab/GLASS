#pragma once
#include <cstdint>
#include <math.h>
#include "reduce.cuh"

/**
 * @brief Euclidean (L2) norm into a separate buffer: `out[0] = ‖a‖₂` (default).
 *
 * Default halving-tree variant, completing the reduction family's bare surface
 * (mirrors how the plain `reduce`/`dot`/`nrm2` engines are built): the block
 * writes the per-element squares into `out`, tree-reduces them with the shared
 * `reduce` machinery, and thread 0 takes the square root. Non-destructive —
 * `a` is left untouched; `out` doubles as the length-`N` scratch (no shared
 * scratch needed, unlike `vector_norm_fast`). Thread-count invariant: the
 * pairwise tree gives byte-identical output at any block size. NumPy
 * equivalent: `np.linalg.norm(a)`.
 *
 * Range contract: naive sum of squares (no LAPACK-style `snrm2` scaling),
 * so the intermediate sum overflows to `inf` for `‖x‖ ≳ 1e19` (f32) /
 * `‖x‖ ≳ 1e154` (f64) and loses tiny components to underflow.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N    Number of elements.
 * @param a    Input vector of length `N` (read-only).
 * @param out  Length-`N` scratch/output buffer; the result lands in `out[0]`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void vector_norm(uint32_t N, T *a, T *out)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    for (uint32_t i = rank; i < N; i += size) out[i] = a[i]*a[i];
    __syncthreads();
    // Halving-tree total in out[0]; skip its trailing barrier — rank 0 wrote
    // out[0] last and immediately owns the sqrt (same pattern as nrm2_impl).
    reduce<T, false>(N, out);
    if (rank == 0) out[0] = sqrt(out[0]);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Euclidean (L2) norm into a separate buffer: `out[0] = ‖a‖₂`, compile-time size.
 *
 * Compile-time-`N` overload of the default halving-tree vector norm; leaves
 * `a` untouched. NumPy equivalent: `np.linalg.norm(a)`.
 *
 * Range contract: naive sum of squares (no LAPACK-style `snrm2` scaling),
 * so the intermediate sum overflows to `inf` for `‖x‖ ≳ 1e19` (f32) /
 * `‖x‖ ≳ 1e154` (f64) and loses tiny components to underflow.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a    Input vector of length `N` (read-only).
 * @param out  Length-`N` scratch/output buffer; the result lands in `out[0]`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void vector_norm(T *a, T *out)
{
    vector_norm<T, TRAILING_SYNC>(N, a, out);
}

/**
 * @brief Euclidean (L2) norm into a separate buffer: `out[0] = ‖a‖₂`, low-memory variant.
 *
 * Writes the per-element squares into `out`, then thread 0 serially sums
 * them and takes the square root, leaving `a` untouched. NumPy equivalent:
 * `np.linalg.norm(a)`.
 *
 * Range contract: naive sum of squares (no LAPACK-style `snrm2` scaling),
 * so the intermediate sum overflows to `inf` for `‖x‖ ≳ 1e19` (f32) /
 * `‖x‖ ≳ 1e154` (f64) and loses tiny components to underflow.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N    Number of elements.
 * @param a    Input vector of length `N`.
 * @param out  Length-`N` scratch/output buffer; the result lands in `out[0]`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void vector_norm_lowmem(uint32_t N, T *a, T *out)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    for (uint32_t i = rank; i < N; i += size) out[i] = a[i]*a[i];
    __syncthreads();
    if (rank == 0) {
        for (uint32_t i = 1; i < N; i++) out[0] += out[i];
        out[0] = sqrt(out[0]);
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
 * Range contract: naive sum of squares (no LAPACK-style `snrm2` scaling),
 * so the intermediate sum overflows to `inf` for `‖x‖ ≳ 1e19` (f32) /
 * `‖x‖ ≳ 1e154` (f64) and loses tiny components to underflow.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N          Number of elements.
 * @param a          Input vector of length `N`.
 * @param out        Output buffer; the result lands in `out[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per
 *                   warp) — size with `reduce_fast_scratch_bytes<T>(blockDim)`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void vector_norm_fast(uint32_t N, T *a, T *out, T *s_scratch)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < N; i += size) val += a[i]*a[i];
    val = shfl_detail::fold_sum(val);
    uint32_t lane = rank & 31, warp = rank >> 5;
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        val = shfl_detail::fold_sum(val);
        if (rank == 0) out[0] = sqrt(val);
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Euclidean (L2) norm into a separate buffer: `out[0] = ‖a‖₂`, warp-shuffle, compile-time size.
 *
 * Compile-time-`N` overload of the warp-shuffle vector norm; leaves `a`
 * untouched. NumPy equivalent: `np.linalg.norm(a)`.
 *
 * Range contract: naive sum of squares (no LAPACK-style `snrm2` scaling),
 * so the intermediate sum overflows to `inf` for `‖x‖ ≳ 1e19` (f32) /
 * `‖x‖ ≳ 1e154` (f64) and loses tiny components to underflow.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a          Input vector of length `N`.
 * @param out        Output buffer; the result lands in `out[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per
 *                   warp) — size with `reduce_fast_scratch_bytes<T>(blockDim)`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void vector_norm_fast(T *a, T *out, T *s_scratch)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < N; i += size) val += a[i]*a[i];
    val = shfl_detail::fold_sum(val);
    uint32_t lane = rank & 31, warp = rank >> 5;
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        val = shfl_detail::fold_sum(val);
        if (rank == 0) out[0] = sqrt(val);
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
