#pragma once
#include "../barrier.cuh"
#include <cstdint>

/**
 * @brief Element-wise maximum: `c = max(a, b)`.
 *
 * NumPy equivalent: `np.maximum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_max_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b[i]);
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_max(uint32_t N, T *a, T *b, T *c)
{ elementwise_max_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise minimum: `c = min(a, b)`.
 *
 * NumPy equivalent: `np.minimum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_min_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b[i]);
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_min(uint32_t N, T *a, T *b, T *c)
{ elementwise_min_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise less-than: `c = (a < b)`.
 *
 * NumPy equivalent: `c = (a < b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where `a[i] < b[i]`, else 0).
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_less_than_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i] < b[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_less_than(uint32_t N, T *a, T *b, T *c)
{ elementwise_less_than_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise greater-than: `c = (a > b)`.
 *
 * NumPy equivalent: `c = (a > b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where `a[i] > b[i]`, else 0).
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_more_than_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i] > b[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_more_than(uint32_t N, T *a, T *b, T *c)
{ elementwise_more_than_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise less-than-or-equal: `c = (a <= b)`.
 *
 * NumPy equivalent: `c = (a <= b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where `a[i] <= b[i]`, else 0).
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_less_than_or_eq_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i] <= b[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_less_than_or_eq(uint32_t N, T *a, T *b, T *c)
{ elementwise_less_than_or_eq_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise less-than-scalar: `c = (a < b)`.
 *
 * Compares every element of `a` against the scalar `b`. NumPy equivalent:
 * `c = (a < b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Scalar threshold.
 * @param c  Output vector of length `N` (1 where `a[i] < b`, else 0).
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_less_than_scalar_impl(Bar bar, uint32_t N, T *a, T b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i] < b;
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_less_than_scalar(uint32_t N, T *a, T b, T *c)
{ elementwise_less_than_scalar_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise logical AND: `c = (a && b)`.
 *
 * NumPy equivalent: `c = np.logical_and(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where both are nonzero, else 0).
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_and_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i] && b[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_and(uint32_t N, T *a, T *b, T *c)
{ elementwise_and_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise logical NOT: `c = !a`.
 *
 * NumPy equivalent: `c = np.logical_not(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where `a[i]` is zero, else 0).
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_not_impl(Bar bar, uint32_t N, T *a, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = !a[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_not(uint32_t N, T *a, T *c)
{ elementwise_not_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, c); }

/**
 * @brief Element-wise absolute value: `b = |a|`.
 *
 * NumPy equivalent: `b = np.abs(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_abs_impl(Bar bar, uint32_t N, T *a, T *b)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) b[i] = abs(a[i]);
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_abs(uint32_t N, T *a, T *b)
{ elementwise_abs_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b); }

/**
 * @brief Element-wise (Hadamard) product: `c = a ⊙ b`.
 *
 * NumPy equivalent: `c = a * b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_mult_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i]*b[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_mult(uint32_t N, T *a, T *b, T *c)
{ elementwise_mult_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise subtraction: `c = a - b`.
 *
 * NumPy equivalent: `c = a - b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_sub_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i] - b[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_sub(uint32_t N, T *a, T *b, T *c)
{ elementwise_sub_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise addition: `c = a + b`.
 *
 * NumPy equivalent: `c = a + b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_add_impl(Bar bar, uint32_t N, T *a, T *b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i] + b[i];
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_add(uint32_t N, T *a, T *b, T *c)
{ elementwise_add_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise scalar multiply: `c = a * b`.
 *
 * Scales every element of `a` by the scalar `b`. NumPy equivalent: `c = a * b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Scalar multiplier.
 * @param c  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_mult_scalar_impl(Bar bar, uint32_t N, T *a, T b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = a[i]*b;
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_mult_scalar(uint32_t N, T *a, T b, T *c)
{ elementwise_mult_scalar_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise maximum against a scalar: `c = max(a, b)`.
 *
 * NumPy equivalent: `c = np.maximum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Scalar lower bound.
 * @param c  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_max_scalar_impl(Bar bar, uint32_t N, T *a, T b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b);
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_max_scalar(uint32_t N, T *a, T b, T *c)
{ elementwise_max_scalar_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise minimum against a scalar: `c = min(a, b)`.
 *
 * NumPy equivalent: `c = np.minimum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Scalar upper bound.
 * @param c  Output vector of length `N`.
 */
// shared body
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_min_scalar_impl(Bar bar, uint32_t N, T *a, T b, T *c)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b);
    if constexpr (TRAILING_SYNC) bar.sync();
}
template <typename T, bool TRAILING_SYNC = true> __device__ void elementwise_min_scalar(uint32_t N, T *a, T b, T *c)
{ elementwise_min_scalar_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

// compile-time size overloads
/**
 * @brief Element-wise maximum: `c = max(a, b)`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `np.maximum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true> __device__ void elementwise_max(T *a, T *b, T *c)
{ elementwise_max_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise minimum: `c = min(a, b)`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `np.minimum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true> __device__ void elementwise_min(T *a, T *b, T *c)
{ elementwise_min_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise absolute value: `b = |a|`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `b = np.abs(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Output vector of length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true> __device__ void elementwise_abs(T *a, T *b)
{ elementwise_abs_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b); }

/**
 * @brief Element-wise (Hadamard) product: `c = a ⊙ b`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `c = a * b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true> __device__ void elementwise_mult(T *a, T *b, T *c)
{ elementwise_mult_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise subtraction: `c = a - b`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `c = a - b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true> __device__ void elementwise_sub(T *a, T *b, T *c)
{ elementwise_sub_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }

/**
 * @brief Element-wise addition: `c = a + b`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `c = a + b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true> __device__ void elementwise_add(T *a, T *b, T *c)
{ elementwise_add_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a, b, c); }
