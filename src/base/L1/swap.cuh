#pragma once
#include "../barrier.cuh"
#include <cstdint>

// shared body: element-wise swap `x ↔ y`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void swap_impl(Bar bar, uint32_t n, T *x, T *y)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Swap two vectors element-wise: `x ↔ y` (SWAP).
 *
 * Exchanges the contents of `x` and `y`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n`.
 * @param y  In/out vector of length `n`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void swap(uint32_t n, T *x, T *y)
{
    swap_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, x, y);
}

/**
 * @brief Swap two vectors element-wise: `x ↔ y` (SWAP), compile-time size.
 *
 * Compile-time-`N` overload of swap.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  In/out vector of length `N`.
 * @param y  In/out vector of length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void swap(T *x, T *y)
{
    swap_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, x, y);
}
