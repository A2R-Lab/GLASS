#pragma once
#include "../barrier.cuh"
#include <cstdint>

// shared body: write the n×n identity (column-major)
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void set_identity_impl(Bar bar, uint32_t n, T *A)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n*n; i += size) {
        uint32_t r = i % n, c = i / n;
        A[i] = static_cast<T>(r == c);
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Load the identity matrix: `A = I_n` (column-major).
 *
 * Writes the `n×n` identity into `A` in column-major order. NumPy equivalent:
 * `A = np.eye(n)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Matrix dimension (number of rows/columns).
 * @param A  Output matrix of `n*n` elements (column-major).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void set_identity(uint32_t n, T *A)
{
    set_identity_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, A);
}

// shared body: add a scaled identity to the diagonal `A += alpha*I`
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void add_identity_impl(Bar bar, uint32_t n, T *A, T alpha)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n*n; i += size)
        if (i % n == i / n) A[i] += alpha;
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Add a scaled identity to a matrix in place: `A += alpha * I`.
 *
 * Adds `alpha` to the diagonal of the `n×n` (column-major) matrix `A`. NumPy
 * equivalent: `A += alpha * np.eye(n)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Matrix dimension (number of rows/columns).
 * @param A      In/out matrix of `n*n` elements (column-major).
 * @param alpha  Scalar added to each diagonal entry.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void add_identity(uint32_t n, T *A, T alpha)
{
    add_identity_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, A, alpha);
}

/**
 * @brief Load the identity matrix: `A = I_N` (column-major), compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `A = np.eye(N)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Matrix dimension (compile-time constant).
 * @param A  Output matrix of `N*N` elements (column-major).
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void set_identity(T *A)
{
    set_identity_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, A);
}

/**
 * @brief Add a scaled identity in place: `A += alpha * I`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `A += alpha * np.eye(N)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Matrix dimension (compile-time constant).
 * @param A      In/out matrix of `N*N` elements (column-major).
 * @param alpha  Scalar added to each diagonal entry.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void add_identity(T *A, T alpha)
{
    add_identity_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, A, alpha);
}

// shared body: add a scaled identity to the leading diagonal block
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void add_identity_partial_impl(Bar bar, uint32_t n, T *A, T alpha, uint32_t diag_count)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n*n; i += size) {
        uint32_t r = i % n, c = i / n;
        if (r == c && r < diag_count) A[i] += alpha;
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Add a scaled identity to the leading diagonal block: `A[:d,:d] += alpha*I`.
 *
 * Adds `alpha` to only the first `diag_count` diagonal entries of the `n×n`
 * (column-major) matrix `A`, leaving the trailing `n-diag_count` diagonal
 * entries untouched. Use to regularize a leading sub-block — e.g. the position
 * block of a stacked `[q; v]` state, where `diag_count = n/2`. NumPy equivalent:
 * `A[:d, :d] += alpha * np.eye(d)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n           Matrix dimension (number of rows/columns).
 * @param A           In/out matrix of `n*n` elements (column-major).
 * @param alpha       Scalar added to each of the leading diagonal entries.
 * @param diag_count  Number of leading diagonal entries to bump (`<= n`).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void add_identity_partial(uint32_t n, T *A, T alpha, uint32_t diag_count)
{
    add_identity_partial_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, A, alpha, diag_count);
}

/**
 * @brief Add a scaled identity to the leading diagonal block, compile-time size.
 *
 * Compile-time `<N, DIAG_COUNT>` overload of add_identity_partial. NumPy equivalent:
 * `A[:DIAG_COUNT, :DIAG_COUNT] += alpha * np.eye(DIAG_COUNT)`.
 *
 * @tparam T           Scalar type (e.g. `float`, `double`).
 * @tparam N           Matrix dimension (compile-time constant).
 * @tparam DIAG_COUNT  Number of leading diagonal entries to bump (`<= N`).
 * @param A      In/out matrix of `N*N` elements (column-major).
 * @param alpha  Scalar added to each of the leading diagonal entries.
 */
template <typename T, uint32_t N, uint32_t DIAG_COUNT, bool TRAILING_SYNC = true>
__device__ void add_identity_partial(T *A, T alpha)
{
    add_identity_partial_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, A, alpha, DIAG_COUNT);
}
