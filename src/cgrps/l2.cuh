#pragma once
#include <cstdint>
#include <cooperative_groups.h>
#include "../base/L2/gemv.cuh"
namespace cgrps = cooperative_groups;

// glass::cgrps::gemv — delegates to shared gemv_impl with g.thread_rank()/g.size()

/**
 * @brief Matrix-vector multiply: `y = alpha * op(A) * x + beta * y` (GEMV, cooperative-groups variant).
 *
 * Runtime-size, single-block; thread rank/size come from the cooperative group.
 * NumPy equivalent: `y = alpha * A @ x + beta * y` (or `A.T @ x` when TRANSPOSE).
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE  If true, computes `A^T * x`.
 * @tparam ROW_MAJOR  Storage order of A (false = column-major).
 * @param m,n   A is m x n.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix.
 * @param x      Input vector.
 * @param beta   Scalar multiplier on the existing y (y is read; caller must initialize it).
 * @param y      In/out result vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(g.thread_rank(), g.size(), m, n, alpha, A, x, beta, y);
}

/**
 * @brief GEMV with implicit `beta = 0`: `y = alpha * op(A) * x` (cooperative-groups variant).
 *
 * Runtime-size overload that overwrites y (the existing y is not read). NumPy
 * equivalent: `y = alpha * A @ x` (or `A.T @ x` when TRANSPOSE).
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE  If true, computes `A^T * x`.
 * @tparam ROW_MAJOR  Storage order of A (false = column-major).
 * @param m,n   A is m x n.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix.
 * @param x      Input vector.
 * @param y      Output result vector (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(g.thread_rank(), g.size(), m, n, alpha, A, x, y);
}

/**
 * @brief GEMV with explicit layout control: `y = alpha * op(A) * x + beta * y` (cooperative-groups variant).
 *
 * Like `gemv` but exposes the A storage order as a named template parameter.
 * NumPy equivalent: `y = alpha * A @ x + beta * y` (or `A.T @ x` when TRANSPOSE).
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE  If true, computes `A^T * x`.
 * @tparam ROW_MAJOR_A  Storage order of A (false = column-major).
 * @param m,n   A is m x n.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix.
 * @param x      Input vector.
 * @param beta   Scalar multiplier on the existing y (y is read; caller must initialize it).
 * @param y      In/out result vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(g.thread_rank(), g.size(), m, n, alpha, A, x, beta, y);
}

/**
 * @brief GEMV with explicit layout control and implicit `beta = 0`: `y = alpha * op(A) * x`.
 *
 * Overwrites y (the existing y is not read). NumPy equivalent:
 * `y = alpha * A @ x` (or `A.T @ x` when TRANSPOSE).
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE  If true, computes `A^T * x`.
 * @tparam ROW_MAJOR_A  Storage order of A (false = column-major).
 * @param m,n   A is m x n.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix.
 * @param x      Input vector.
 * @param y      Output result vector (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(g.thread_rank(), g.size(), m, n, alpha, A, x, y);
}

/**
 * @brief Compile-time-size GEMV: `y = alpha * op(A) * x + beta * y` (cooperative-groups variant).
 *
 * Dimensions baked in as template parameters. NumPy equivalent:
 * `y = alpha * A @ x + beta * y` (or `A.T @ x` when TRANSPOSE).
 *
 * @tparam T  Scalar type.
 * @tparam M,N  Compile-time dimensions (A is M x N).
 * @tparam TRANSPOSE  If true, computes `A^T * x`.
 * @tparam ROW_MAJOR  Storage order of A (false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix.
 * @param x      Input vector.
 * @param beta   Scalar multiplier on the existing y (y is read; caller must initialize it).
 * @param y      In/out result vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(T alpha, T *A, T *x, T beta, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(g.thread_rank(), g.size(), M, N, alpha, A, x, beta, y);
}

/**
 * @brief Compile-time-size GEMV with implicit `beta = 0`: `y = alpha * op(A) * x`.
 *
 * Overwrites y (the existing y is not read). NumPy equivalent:
 * `y = alpha * A @ x` (or `A.T @ x` when TRANSPOSE).
 *
 * @tparam T  Scalar type.
 * @tparam M,N  Compile-time dimensions (A is M x N).
 * @tparam TRANSPOSE  If true, computes `A^T * x`.
 * @tparam ROW_MAJOR  Storage order of A (false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix.
 * @param x      Input vector.
 * @param y      Output result vector (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(T alpha, T *A, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(g.thread_rank(), g.size(), M, N, alpha, A, x, y);
}

/**
 * @brief Rank-1 update: `A += alpha * x * y^T` (GER, cooperative-groups variant).
 *
 * Adds the scaled outer product of `x` and `y` into the column-major matrix A.
 * NumPy equivalent: `A += alpha * np.outer(x, y)`.
 *
 * @tparam T  Scalar type.
 * @param m,n   A is m x n (x length m, y length n).
 * @param alpha  Scalar multiplier on the outer product.
 * @param x,y    Input vectors.
 * @param A      In/out matrix (column-major).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
// ger: A += alpha * x * y^T (column-major)
template <typename T>
__device__ void ger(uint32_t m, uint32_t n, T alpha, T *x, T *y, T *A,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t col = 0; col < n; col++) {
        T ay = alpha * y[col];
        for (uint32_t row = g.thread_rank(); row < m; row += g.size())
            A[row + col*m] += ay * x[row];
    }
}

/**
 * @brief Compile-time-size rank-1 update: `A += alpha * x * y^T` (GER, cooperative-groups variant).
 *
 * NumPy equivalent: `A += alpha * np.outer(x, y)`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N  Compile-time dimensions (A is M x N; x length M, y length N).
 * @param alpha  Scalar multiplier on the outer product.
 * @param x,y    Input vectors.
 * @param A      In/out matrix (column-major).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N>
__device__ void ger(T alpha, T *x, T *y, T *A,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t col = 0; col < N; col++) {
        T ay = alpha * y[col];
        for (uint32_t row = g.thread_rank(); row < M; row += g.size())
            A[row + col*M] += ay * x[row];
    }
}

// ─── contraction-parallel GEMV (cooperative-groups variant) ──────────────────
// Delegates to the shared glass::detail engine; pass a warp-multiple group so
// every output is owned by a full warp.

/**
 * @brief Contraction-parallel GEMV: `y = alpha * op(A) * x + beta * y` (cooperative-groups variant).
 *
 * Cooperative-groups form of `glass::gemv_reduced`. See it for semantics.
 *
 * @tparam T,M,N,TRANS  See glass::gemv_reduced.
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha,A,x,beta,y  See glass::gemv_reduced.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANS = false, bool TRAILING_SYNC = true>
__device__ void gemv_reduced(T alpha, const T* A, const T* x, T beta, T* y,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::detail::gemv_reduced_impl<T, M, N, TRANS, true>(g.thread_rank(), g.size(), alpha, A, x, beta, y);
    if constexpr (TRAILING_SYNC) g.sync();
}

/**
 * @brief Contraction-parallel GEMV with implicit `beta = 0`: `y = alpha * op(A) * x` (cooperative-groups variant).
 *
 * @tparam T,M,N,TRANS,TRAILING_SYNC  See the beta overload.
 * @param alpha,A,x,y,g  See the beta overload.
 */
template <typename T, uint32_t M, uint32_t N, bool TRANS = false, bool TRAILING_SYNC = true>
__device__ void gemv_reduced(T alpha, const T* A, const T* x, T* y,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::detail::gemv_reduced_impl<T, M, N, TRANS, false>(g.thread_rank(), g.size(), alpha, A, x, static_cast<T>(0), y);
    if constexpr (TRAILING_SYNC) g.sync();
}
