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
 * When `CHECK` is true and `s_fail` is non-null, rank 0 sets `*s_fail = 1` on a
 * non-PD / NaN pivot (else 0). `CHECK` defaults false and compiles out.
 *
 * @tparam T  Scalar type.
 * @tparam CHECK  If true, detect a non-PD pivot and report it via `s_fail` (default false, compiles out).
 * @param n      Matrix dimension (A is n x n).
 * @param s_A    In/out n x n matrix (column-major); on return its lower triangle holds L.
 * @param g      Cooperative thread group (defaults to the whole block).
 * @param s_fail Optional flag (CHECK only): set to 1 on a non-PD / NaN pivot, else 0. Ignored when null.
 */
// cholDecomp_InPlace
// CHECK (compile-out, default false): when true and s_fail non-null, rank 0 sets
// *s_fail=1 on a non-PD / NaN pivot (else 0) — mirrors the base block overload.
// s_fail is placed after g so existing (n, s_A[, g]) callers are unaffected.
template <typename T, bool CHECK = false>
__device__ void cholDecomp_InPlace(uint32_t n, T *s_A,
                                    cgrps::thread_group g = cgrps::this_thread_block(),
                                    int *s_fail = nullptr)
{
    if constexpr (CHECK) { if (g.thread_rank() == 0 && s_fail) *s_fail = 0; }
    for (uint32_t row = 0; row < n; row++) {
        if (g.thread_rank() == 0) {
            T sum = static_cast<T>(0), val = s_A[n*row + row];
            for (int32_t rl = 0; rl < (int32_t)row; rl++) sum += s_A[rl*n+row]*s_A[rl*n+row];
            T d = val - sum;
            if constexpr (CHECK) { if (s_fail && (d <= static_cast<T>(0) || isnan(d))) *s_fail = 1; }
            s_A[row*n + row] = sqrtf(d);
        }
        g.sync();
        // NOTE: the pivot s_A[row*n+row] is broadcast by re-reading shared after g.sync().
        // Safe for a block group (the default — g.sync() is a full block fence). If this is
        // ever called with a warp-tiled group, prefer g.shfl(pivot, 0) — a shared re-read at
        // warp scope can be cached stale by nvcc under caller __restrict__ (see the
        // glass::warp::cholDecomp_InPlace shfl fix in base/L3/chol_InPlace.cuh).
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
        // broadcast via shared re-read; safe for a block group (full fence). For a warp-tiled
        // group prefer g.shfl(b[col], 0) — see the glass::warp::trsm shfl fix (base/L3/trsm.cuh).
        T factor = b[col];
        for (uint32_t row = g.thread_rank() + col + 1; row < n; row += g.size())
            b[row] -= L[col*n + row] * factor;
        g.sync();
    }
}

// ─── contraction-parallel GEMM (cooperative-groups variant) ──────────────────
// Delegates to the shared glass::gemm_reduced_impl_ct engine. The warp-shuffle
// reduce assumes the group's lanes align with hardware warps (true for
// this_thread_block() on a 1D block and for tiled_partition<32>); pass a
// warp-multiple group so every output is owned by a full warp.

/**
 * @brief Contraction-parallel GEMM: `C = alpha * A * op(B) + beta * C` (cooperative-groups variant).
 *
 * The cooperative-groups form of `glass::gemm_reduced`: one warp owns each
 * output and its lanes split the contraction. Pass a warp-multiple group.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (requires B square: N == K).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major).
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemm_reduced(T alpha, T *A, T *B, T beta, T *C,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::gemm_reduced_impl_ct<T, M, N, K, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR, true>(
        g.thread_rank(), g.size(), alpha, A, B, beta, C);
    if constexpr (TRAILING_SYNC) g.sync();
}

/**
 * @brief Contraction-parallel GEMM with implicit `beta = 0`: `C = alpha * A * op(B)` (cooperative-groups variant).
 *
 * Overwrites C (the existing C is not read). Otherwise identical to the beta
 * overload above; pass a warp-multiple group.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (requires B square: N == K).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major).
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemm_reduced(T alpha, T *A, T *B, T *C,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::gemm_reduced_impl_ct<T, M, N, K, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR, false>(
        g.thread_rank(), g.size(), alpha, A, B, static_cast<T>(0), C);
    if constexpr (TRAILING_SYNC) g.sync();
}

// ─── tensor ⊗ vector contractions (cooperative-groups variants) ──────────────
// Delegate to the shared glass::detail engine; pass a warp-multiple group so
// every output is owned by a full warp (the warp-shuffle reduce assumes it).

/**
 * @brief Tensor ⊗ vector contraction: `Mout (+)= Σ_c v[c] · T[..c..]` (cooperative-groups variant).
 *
 * Cooperative-groups form of `glass::tensor_vec_contract`. See it for semantics.
 *
 * @tparam T,K,A,B,CONTRACT,SYMMETRIC,ACCUMULATE,TIN_ROW_MAJOR  See glass::tensor_vec_contract.
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param Tns,v,Mout  See glass::tensor_vec_contract.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t K, uint32_t A, uint32_t B,
          TensorAxis CONTRACT = TensorAxis::K, bool SYMMETRIC = false,
          bool ACCUMULATE = true, bool TIN_ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void tensor_vec_contract(const T* Tns, const T* v, T* Mout,
                                    cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::detail::tvc_impl<T, CONTRACT, K, A, B, SYMMETRIC, ACCUMULATE, TIN_ROW_MAJOR>(
        g.thread_rank(), g.size(), Tns, v, Mout);
    if constexpr (TRAILING_SYNC) g.sync();
}

/**
 * @brief Vector–tensor–vector triple product: `s[k] (+)= u^T · T_k · w` (cooperative-groups variant).
 *
 * Cooperative-groups form of `glass::vec_tensor_vec`. See it for semantics.
 *
 * @tparam T,K,A,B,ACCUMULATE,TIN_ROW_MAJOR  See glass::vec_tensor_vec.
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param Tns,u,w,s  See glass::vec_tensor_vec.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t K, uint32_t A, uint32_t B,
          bool ACCUMULATE = false, bool TIN_ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void vec_tensor_vec(const T* Tns, const T* u, const T* w, T* s,
                               cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::detail::vtv_impl<T, K, A, B, ACCUMULATE, TIN_ROW_MAJOR>(
        g.thread_rank(), g.size(), Tns, u, w, s);
    if constexpr (TRAILING_SYNC) g.sync();
}

// ─── congruence / bilinear forms (cooperative-groups variants) ───────────────
// Step 1 (M·X) uses the cgrps runtime gemm; step 2 (Xᵀ·MX) the shared engine.
// Pass a warp-multiple group so every step-2 output is owned by a full warp.

/**
 * @brief Symmetric congruence: `Q = alpha * Xᵀ·M·X + beta * Q` (cooperative-groups variant).
 *
 * Cooperative-groups form of `glass::congruence_sym`. See it for semantics.
 *
 * @tparam T,N,Kdim,ACCUMULATE  See glass::congruence_sym.
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha,X,M,beta,Q,s_temp  See glass::congruence_sym.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t N, uint32_t Kdim,
          bool ACCUMULATE = false, bool TRAILING_SYNC = true>
__device__ void congruence_sym(T alpha, const T* X, const T* M, T beta, T* Q, T* s_temp,
                               cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm(N, N, Kdim, static_cast<T>(1), const_cast<T*>(M), const_cast<T*>(X), s_temp, g);
    g.sync();
    glass::detail::xtY_impl<T, N, Kdim, Kdim, true, ACCUMULATE>(
        g.thread_rank(), g.size(), alpha, X, s_temp, beta, Q);
    if constexpr (TRAILING_SYNC) g.sync();
}

/**
 * @brief General bilinear form: `R = alpha * Xᵀ·M·Y + beta * R` (cooperative-groups variant).
 *
 * Cooperative-groups form of `glass::bilinear`. See it for semantics.
 *
 * @tparam T,N,P,Qd,ACCUMULATE  See glass::bilinear.
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha,X,M,Y,beta,R,s_temp  See glass::bilinear.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t N, uint32_t P, uint32_t Qd,
          bool ACCUMULATE = false, bool TRAILING_SYNC = true>
__device__ void bilinear(T alpha, const T* X, const T* M, const T* Y, T beta, T* R, T* s_temp,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm(N, N, Qd, static_cast<T>(1), const_cast<T*>(M), const_cast<T*>(Y), s_temp, g);
    g.sync();
    glass::detail::xtY_impl<T, N, P, Qd, false, ACCUMULATE>(
        g.thread_rank(), g.size(), alpha, X, s_temp, beta, R);
    if constexpr (TRAILING_SYNC) g.sync();
}

// ─── contraction-parallel SYRK (cooperative-groups variant) ──────────────────

/**
 * @brief Contraction-parallel symmetric rank-k update: `C = alpha * A·op(A) + beta * C` (cooperative-groups variant).
 *
 * Cooperative-groups form of `glass::syrk_reduced`. Pass a warp-multiple group.
 *
 * @tparam T,ROWS,COLS,TRANS  See glass::syrk_reduced.
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha,A,beta,C  See glass::syrk_reduced.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANS = false, bool TRAILING_SYNC = true>
__device__ void syrk_reduced(T alpha, const T* A, T beta, T* C,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::detail::syrk_reduced_impl<T, ROWS, COLS, TRANS, true>(g.thread_rank(), g.size(), alpha, A, beta, C);
    if constexpr (TRAILING_SYNC) g.sync();
}

/**
 * @brief Contraction-parallel SYRK with implicit `beta = 0`: `C = alpha * A·op(A)` (cooperative-groups variant).
 *
 * @tparam T,ROWS,COLS,TRANS,TRAILING_SYNC  See the beta overload.
 * @param alpha,A,C,g  See the beta overload.
 */
template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANS = false, bool TRAILING_SYNC = true>
__device__ void syrk_reduced(T alpha, const T* A, T* C,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::detail::syrk_reduced_impl<T, ROWS, COLS, TRANS, false>(g.thread_rank(), g.size(), alpha, A, static_cast<T>(0), C);
    if constexpr (TRAILING_SYNC) g.sync();
}
