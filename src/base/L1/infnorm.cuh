#pragma once
#include "../barrier.cuh"
#include <cstdint>

/**
 * @brief Infinity norm: `x[0] = ‖x‖∞ = max|x[i]|` (in-place, destructive).
 *
 * Computes the maximum absolute value via a block-wide halving reduction; the
 * result lands in `x[0]` (the input is overwritten). NumPy equivalent:
 * `np.max(np.abs(x))`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n`; the result lands in `x[0]`.
 */
// Shared body: max|x[i]| via halving reduce; result in x[0]. Barrier policy
// supplies rank/size + internal/trailing sync, shared by glass:: and cgrps::.
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void infnorm_impl(Bar bar, uint32_t n, T *x)
{
    uint32_t ind = bar.rank();
    uint32_t stride = bar.size();
    uint32_t left = n;
    bool odd;
    while (left > 3) {
        odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = ind; i < left; i += stride)
            x[i] = max(abs(x[i]), abs(x[i + left]));
        if (ind == 0 && odd) x[0] = max(abs(x[0]), abs(x[2*left]));
        bar.sync();
    }
    if (ind == 0) { for (uint32_t i = 1; i < left; i++) x[0] = max(abs(x[0]), abs(x[i])); }
    if constexpr (TRAILING_SYNC) bar.sync();
}

template <typename T, bool TRAILING_SYNC = true>
__device__ void infnorm(uint32_t n, T *x)
{
    infnorm_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, x);
}

/**
 * @brief Infinity norm: `x[0] = ‖x‖∞ = max|x[i]|`, compile-time size.
 *
 * Compile-time-`N` overload (in-place, destructive). NumPy equivalent:
 * `np.max(np.abs(x))`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  In/out vector of length `N`; the result lands in `x[0]`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void infnorm(T *x)
{
    infnorm_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, x);
}
