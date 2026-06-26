#pragma once
#include "../barrier.cuh"
#include <cstdint>

// shared body: broadcast a constant `x[i] = alpha`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void set_const_impl(Bar bar, uint32_t n, T alpha, T *x)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) x[i] = alpha;
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Fill a vector with a constant: `x[i] = alpha`.
 *
 * NumPy equivalent: `x = np.full(n, alpha)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Number of elements.
 * @param alpha  Value to broadcast into every element.
 * @param x      Output vector of length `n`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void set_const(uint32_t n, T alpha, T *x)
{
    set_const_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, alpha, x);
}

/**
 * @brief Fill a vector with a constant: `x[i] = alpha`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `x = np.full(N, alpha)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param alpha  Value to broadcast into every element.
 * @param x      Output vector of length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void set_const(T alpha, T *x)
{
    set_const_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, alpha, x);
}
