#pragma once
#include <cstdint>
#include <math.h>
#include <cooperative_groups.h>
#include "../base/L3/gemm.cuh"
namespace cgrps = cooperative_groups;

// glass::cgrps::gemm — delegates to shared gemm_impl

/**
 * @brief GEMM: `C = alpha * A * op(B) + beta * C` (cooperative-groups variant).
 *
 * Runtime-size, single-block; thread rank/size come from the cooperative group.
 * Storage order is uniform across A, B, C (`ROW_MAJOR`; false = column-major).
 * NumPy equivalent: `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (pure-SIMT path requires B square).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major).
 * @param m,n,k  Dimensions: A is m x n, B is n x k (or k x n when TRANSPOSE_B), C is m x (TRANSPOSE_B ? n : k).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, T *A, T *B, T beta, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(
        g.thread_rank(), g.size(), m, n, k, alpha, A, B, beta, C);
}

/**
 * @brief GEMM with implicit `beta = 0`: `C = alpha * A * op(B)` (cooperative-groups variant).
 *
 * Runtime-size overload that overwrites C (the existing C is not read). NumPy
 * equivalent: `C = alpha * A @ B`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (pure-SIMT path requires B square).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major).
 * @param m,n,k  Dimensions: A is m x n, B is n x k (or k x n when TRANSPOSE_B), C is m x (TRANSPOSE_B ? n : k).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, T *A, T *B, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(
        g.thread_rank(), g.size(), m, n, k, alpha, A, B, C);
}

/**
 * @brief GEMM with per-matrix layout control: `C = alpha * A * op(B) + beta * C` (cooperative-groups variant).
 *
 * Like `gemm` but the storage order of A, B and C is set independently. NumPy
 * equivalent: `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T`.
 * @tparam ROW_MAJOR_A/ROW_MAJOR_B/ROW_MAJOR_C  Storage order per matrix (false = column-major).
 * @param m,n,k  Dimensions: A is m x n, B is n x k (or k x n when TRANSPOSE_B), C is m x (TRANSPOSE_B ? n : k).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_ex(uint32_t m, uint32_t n, uint32_t k,
                         T alpha, T *A, T *B, T beta, T *C,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(
        g.thread_rank(), g.size(), m, n, k, alpha, A, B, beta, C);
}

/**
 * @brief Compile-time-size GEMM: `C = alpha * A * op(B) + beta * C` (cooperative-groups variant).
 *
 * Dimensions baked in as template parameters; uniform storage order. NumPy
 * equivalent: `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T`.
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(T alpha, T *A, T *B, T beta, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(
        g.thread_rank(), g.size(), M, N, K, alpha, A, B, beta, C);
}

/**
 * @brief Compile-time-size GEMM with implicit `beta = 0`: `C = alpha * A * op(B)` (cooperative-groups variant).
 *
 * Overwrites C (the existing C is not read). NumPy equivalent: `C = alpha * A @ B`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T`.
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(T alpha, T *A, T *B, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(
        g.thread_rank(), g.size(), M, N, K, alpha, A, B, C);
}

/**
 * @brief In-place matrix inverse via Gauss-Jordan on `[A | I]` (cooperative-groups variant).
 *
 * Reduces a column-major augmented `dimA x (2*dimA)` matrix `[A | I]` so columns
 * `dimA..2*dimA-1` hold `A^-1` on return. Serial pivot loop, block-parallel cell
 * updates. NumPy equivalent: `Ainv = np.linalg.inv(A)`.
 *
 * @tparam T  Scalar type.
 * @param dimA    Matrix dimension (A is dimA x dimA).
 * @param A       In/out augmented `[A | I]` buffer (column-major, dimA x 2*dimA);
 *                on return its right half holds `A^-1`.
 * @param s_temp  Shared scratch of `(2*dimA + 1) * sizeof(T)` bytes.
 * @param g       Cooperative thread group (defaults to the whole block).
 */
// invertMatrix — no cgrps variant needed (already uses __syncthreads internally)
// Exposed here for API completeness via glass::cgrps::invertMatrix
template <typename T>
__device__ void invertMatrix(uint32_t dimA, T *A, T *s_temp,
                              cgrps::thread_group g = cgrps::this_thread_block())
{
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
        unsigned pivOff = pivRC * dimA;
        T pvInv = static_cast<T>(1) / A[pivRC + pivOff];
        for (unsigned ind = g.thread_rank(); ind < 2*dimA+1; ind++) {
            unsigned AInd = (ind < dimA) ? (ind + pivOff) : (pivRC + pivOff + (ind-dimA)*dimA);
            s_temp[ind] = A[AInd];
        }
        g.sync();
        for (unsigned ind = g.thread_rank(); ind < dimA*(dimA+1); ind += g.size()) {
            unsigned row = ind % dimA, col = ind / dimA, coff = ind - row;
            if (row == pivRC) A[row + pivOff + coff] *= pvInv;
            else A[row + pivOff + coff] -= s_temp[row]*pvInv*s_temp[dimA+col];
        }
        g.sync();
    }
}

/**
 * @brief In-place Cholesky factorization of an SPD matrix (cooperative-groups variant).
 *
 * Factors `A = L * L^T` and overwrites `A` with the lower-triangular factor `L`
 * (only the lower triangle is written; the upper triangle keeps its input
 * values). `A` must be symmetric positive-definite, column-major. NumPy
 * equivalent: `L = np.linalg.cholesky(A)`.
 *
 * @tparam T  Scalar type.
 * @param n    Matrix dimension (A is n x n).
 * @param s_A  In/out n x n matrix (column-major); on return its lower triangle holds L.
 * @param g    Cooperative thread group (defaults to the whole block).
 */
// cholDecomp_InPlace
template <typename T>
__device__ void cholDecomp_InPlace(uint32_t n, T *s_A,
                                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t row = 0; row < n; row++) {
        if (g.thread_rank() == 0) {
            T sum = static_cast<T>(0), val = s_A[n*row + row];
            for (int32_t rl = 0; rl < (int32_t)row; rl++) sum += s_A[rl*n+row]*s_A[rl*n+row];
            s_A[row*n + row] = sqrtf(val - sum);
        }
        g.sync();
        for (uint32_t col = g.thread_rank() + row + 1; col < n; col += g.size()) {
            T sum = static_cast<T>(0);
            for (uint32_t kk = 0; kk < row; kk++) sum += s_A[kk*n+col]*s_A[kk*n+row];
            s_A[row*n+col] = (static_cast<T>(1)/s_A[row*n+row])*(s_A[row*n+col] - sum);
        }
        g.sync();
    }
}

/**
 * @brief Lower-triangular solve `L x = b` in place via forward substitution (cooperative-groups variant).
 *
 * Solves for `x` given lower-triangular `L` (column-major) and right-hand side
 * `b`, overwriting `b` with the solution. SciPy equivalent:
 * `x = scipy.linalg.solve_triangular(L, b, lower=True)`.
 *
 * @tparam T  Scalar type.
 * @param n  Dimension (L is n x n, b has length n).
 * @param L  Lower-triangular matrix (column-major).
 * @param b  In/out right-hand side; on return holds the solution x.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
// trsm
template <typename T>
__device__ void trsm(uint32_t n, T *L, T *b,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t col = 0; col < n; col++) {
        if (g.thread_rank() == 0) b[col] /= L[col*n + col];
        g.sync();
        T factor = b[col];
        for (uint32_t row = g.thread_rank() + col + 1; row < n; row += g.size())
            b[row] -= L[col*n + row] * factor;
        g.sync();
    }
}
