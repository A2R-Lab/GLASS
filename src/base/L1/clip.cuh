#pragma once
#include "../barrier.cuh"
#include <cstdint>

// shared body: element-wise clamp `x = clamp(x, l, u)`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void clip_impl(Bar bar, uint32_t n, T *x, T *l, T *u)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size)
        x[i] = max(l[i], min(x[i], u[i]));
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Element-wise clamp in place: `x = clamp(x, l, u)`.
 *
 * Each element is bounded below by the corresponding `l[i]` and above by
 * `u[i]`. NumPy equivalent: `np.clip(x, l, u)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n` (overwritten with the clamped values).
 * @param l  Per-element lower bounds, length `n`.
 * @param u  Per-element upper bounds, length `n`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void clip(uint32_t n, T *x, T *l, T *u)
{
    clip_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, x, l, u);
}

/**
 * @brief Element-wise clamp in place: `x = clamp(x, l, u)`, compile-time size.
 *
 * Compile-time-`N` overload of clip. NumPy equivalent: `np.clip(x, l, u)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  In/out vector of length `N` (overwritten with the clamped values).
 * @param l  Per-element lower bounds, length `N`.
 * @param u  Per-element upper bounds, length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void clip(T *x, T *l, T *u)
{
    clip_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, x, l, u);
}
