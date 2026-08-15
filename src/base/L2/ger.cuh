#pragma once
#include <cstdint>

/**
 * @brief Rank-1 update: `A += alpha * x * yᵀ` (GER).
 *
 * Adds the scaled outer product of `x` and `y` to the `m×n` column-major matrix
 * `A`. NumPy equivalent: `A += alpha * np.outer(x, y)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param m      Number of rows of `A` (length of `x`).
 * @param n      Number of columns of `A` (length of `y`).
 * @param alpha  Scalar multiplier on the outer product.
 * @param x      Input vector of length `m`.
 * @param y      Input vector of length `n`.
 * @param A      In/out matrix of `m*n` elements (column-major).
 */
// A += alpha * x * y^T  (A is m×n column-major)
template <typename T, bool TRAILING_SYNC = true>
__device__ void ger(uint32_t m, uint32_t n, T alpha, const T *x, const T *y, T *A)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    for (uint32_t col = 0; col < n; col++) {
        T ay = alpha * y[col];
        for (uint32_t row = rank; row < m; row += size)
            A[row + col*m] += ay * x[row];
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Rank-1 update: `A += alpha * x * yᵀ` (GER), compile-time size.
 *
 * Compile-time-`M`,`N` overload of the rank-1 update. NumPy equivalent:
 * `A += alpha * np.outer(x, y)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam M  Number of rows of `A` / length of `x` (compile-time constant).
 * @tparam N  Number of columns of `A` / length of `y` (compile-time constant).
 * @param alpha  Scalar multiplier on the outer product.
 * @param x      Input vector of length `M`.
 * @param y      Input vector of length `N`.
 * @param A      In/out matrix of `M*N` elements (column-major).
 */
template <typename T, uint32_t M, uint32_t N, bool TRAILING_SYNC = true>
__device__ void ger(T alpha, const T *x, const T *y, T *A)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    // Map across the complete output, so short-wide matrices use the whole
    // block instead of serializing every column over only M active lanes.
    // M and N are compile-time constants, so row/column recovery strength
    // reduces to constant-divisor arithmetic.
    for (uint32_t elem = rank; elem < M * N; elem += size) {
        uint32_t row = elem % M;
        uint32_t col = elem / M;
        A[elem] += alpha * x[row] * y[col];
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
