#pragma once
#include "../barrier.cuh"
#include <cstdint>

/**
 * @brief Out-of-place matrix transpose: `b = aᵀ` (column-major).
 *
 * Transposes the `N×M` column-major matrix `a` into the `M×N` column-major
 * matrix `b`. NumPy equivalent: `b = a.T`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of rows of `a` (columns of `b`).
 * @param M  Number of columns of `a` (rows of `b`).
 * @param a  Input matrix of `N*M` elements (column-major).
 * @param b  Output matrix of `M*N` elements (column-major).
 */
// shared body: out-of-place transpose NxM column-major a → b
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void transpose_impl(Bar bar, uint32_t N, uint32_t M, T *a, T *b)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < N*M; i += size) {
        uint32_t col = i / N, row = i % N;
        b[col + M*row] = a[row + N*col];
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

// out-of-place: NxM column-major a → b
template <typename T, bool TRAILING_SYNC = true>
__device__ void transpose(uint32_t N, uint32_t M, T *a, T *b)
{
    transpose_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, M, a, b);
}

/**
 * @brief In-place square matrix transpose: `a = aᵀ` (column-major).
 *
 * Transposes the `N×N` column-major matrix `a` in place by swapping symmetric
 * off-diagonal entries. NumPy equivalent: `a = a.T`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Matrix dimension (number of rows/columns).
 * @param a  In/out matrix of `N*N` elements (column-major).
 */
// shared body: in-place transpose NxN column-major
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void transpose_impl(Bar bar, uint32_t N, T *a)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t idx = rank; idx < N*N; idx += size) {
        uint32_t i = idx % N, j = idx / N;
        if (i < j) {
            uint32_t swap = i*N + j;
            T tmp = a[idx]; a[idx] = a[swap]; a[swap] = tmp;
        }
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

// in-place: NxN column-major
template <typename T, bool TRAILING_SYNC = true>
__device__ void transpose(uint32_t N, T *a)
{
    transpose_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a);
}

/**
 * @brief Out-of-place matrix transpose: `b = aᵀ`, compile-time size.
 *
 * Compile-time-`N`,`M` overload; transposes the `N×M` column-major matrix `a`
 * into the `M×N` column-major matrix `b`. NumPy equivalent: `b = a.T`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of rows of `a` (columns of `b`), compile-time constant.
 * @tparam M  Number of columns of `a` (rows of `b`), compile-time constant.
 * @param a  Input matrix of `N*M` elements (column-major).
 * @param b  Output matrix of `M*N` elements (column-major).
 */
// compile-time out-of-place
template <typename T, uint32_t N, uint32_t M, bool TRAILING_SYNC = true>
__device__ void transpose(T *a, T *b)
{
    transpose_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, M, a, b);
}

/**
 * @brief In-place square matrix transpose: `a = aᵀ`, compile-time size.
 *
 * Compile-time-`N` overload; transposes the `N×N` column-major matrix `a` in
 * place. NumPy equivalent: `a = a.T`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Matrix dimension (compile-time constant).
 * @param a  In/out matrix of `N*N` elements (column-major).
 */
// compile-time in-place NxN
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void transpose(T *a)
{
    transpose_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, a);
}
