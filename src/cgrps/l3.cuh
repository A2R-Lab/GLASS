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
 * @tparam TRANSPOSE_A  If true, `A` is `k×m` and `op(A)=Aᵀ` (else `A` is `m×k`).
 * @tparam TRANSPOSE_B  If true, `B` is `n×k` and `op(B)=Bᵀ` (else `B` is `k×n`).
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
 * @param m,n,k  Dimensions: `C` is `m×n`, contraction `k`.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, const T *A, const T *B, T beta, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(
        g.thread_rank(), g.size(), m, n, k, alpha, A, B, beta, C);
    if constexpr (TRAILING_SYNC) cgrps::sync(g);
}

/**
 * @brief GEMM with implicit `beta = 0`: `C = alpha * A * op(B)` (cooperative-groups variant).
 *
 * Runtime-size overload that overwrites C (the existing C is not read). NumPy
 * equivalent: `C = alpha * A @ B`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_A  If true, `A` is `k×m` and `op(A)=Aᵀ` (else `A` is `m×k`).
 * @tparam TRANSPOSE_B  If true, `B` is `n×k` and `op(B)=Bᵀ` (else `B` is `k×n`).
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
 * @param m,n,k  Dimensions: `C` is `m×n`, contraction `k`.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, const T *A, const T *B, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(
        g.thread_rank(), g.size(), m, n, k, alpha, A, B, C);
    if constexpr (TRAILING_SYNC) cgrps::sync(g);
}

/**
 * @brief Compile-time-size GEMM: `C = alpha * op(A) * op(B) + beta * C` (cooperative-groups variant).
 *
 * Dimensions baked in as template parameters; standard convention (C is M×N,
 * contraction K). NumPy: `C = alpha * opA(A) @ opB(B) + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  `C` is `M×N`, contraction `K` (see gemm.cuh).
 * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
 * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(T alpha, const T *A, const T *B, T beta, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(
        g.thread_rank(), g.size(), alpha, A, B, beta, C);
    if constexpr (TRAILING_SYNC) cgrps::sync(g);
}

/**
 * @brief Compile-time-size GEMM with implicit `beta = 0`: `C = alpha * op(A) * op(B)` (cooperative-groups variant).
 *
 * Overwrites C (the existing C is not read). NumPy: `C = alpha * opA(A) @ opB(B)`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  `C` is `M×N`, contraction `K` (see gemm.cuh).
 * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
 * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(T alpha, const T *A, const T *B, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(
        g.thread_rank(), g.size(), alpha, A, B, C);
    if constexpr (TRAILING_SYNC) cgrps::sync(g);
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
 * @param s_scratch  Shared scratch of `(2*dimA + 1) * sizeof(T)` bytes.
 * @param g       Cooperative thread group (defaults to the whole block).
 */
// Delegates to the shared glass::invertMatrix_impl with a GroupBarrier.
template <typename T>
__device__ void invertMatrix(uint32_t dimA, T *A, T *s_scratch,
                              cgrps::thread_group g = cgrps::this_thread_block())
{
    invertMatrix_impl<GroupBarrier, T>(GroupBarrier{g}, dimA, A, s_scratch);
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
// Delegates to the shared glass::cholDecomp_InPlace_impl with a GroupBarrier.
// NOTE on warp-tiled groups: the pivot is broadcast by re-reading shared after
// the barrier. Safe for a block group (the default — a full block fence). A
// shared re-read at warp scope can be cached stale by nvcc under caller
// __restrict__ — prefer glass::warp::cholDecomp_InPlace there (it shfl-broadcasts).
template <typename T, bool CHECK = false>
__device__ void cholDecomp_InPlace(uint32_t n, T *s_A,
                                    cgrps::thread_group g = cgrps::this_thread_block(),
                                    int *s_fail = nullptr)
{
    cholDecomp_InPlace_impl<GroupBarrier, T, CHECK>(GroupBarrier{g}, n, s_A, s_fail);
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
// Delegates to the shared glass::trsm_impl with a GroupBarrier. For a warp-tiled
// group prefer glass::warp::trsm (register-broadcast pivot) — see base/L3/trsm.cuh.
template <typename T>
__device__ void trsm(uint32_t n, T *L, T *b,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    trsm_impl<GroupBarrier, T>(GroupBarrier{g}, n, L, b);
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
 * @tparam M,N,K  `C` is `M×N`, contraction `K` (see gemm.cuh).
 * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
 * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm_reduced(T alpha, T *A, T *B, T beta, T *C,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::gemm_reduced_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C, true>(
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
 * @tparam M,N,K  `C` is `M×N`, contraction `K` (see gemm.cuh).
 * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
 * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm_reduced(T alpha, T *A, T *B, T *C,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::gemm_reduced_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C, false>(
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
 * @param alpha,X,M,beta,Q,s_scratch  See glass::congruence_sym.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t N, uint32_t Kdim,
          bool ACCUMULATE = false, bool TRAILING_SYNC = true>
__device__ void congruence_sym(T alpha, const T* X, const T* M, T beta, T* Q, T* s_scratch,
                               cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm(N, Kdim, N, static_cast<T>(1), const_cast<T*>(M), const_cast<T*>(X), s_scratch, g);
    g.sync();
    glass::detail::xtY_impl<T, N, Kdim, Kdim, true, ACCUMULATE>(
        g.thread_rank(), g.size(), alpha, X, s_scratch, beta, Q);
    if constexpr (TRAILING_SYNC) g.sync();
}

/**
 * @brief General bilinear form: `R = alpha * Xᵀ·M·Y + beta * R` (cooperative-groups variant).
 *
 * Cooperative-groups form of `glass::bilinear`. See it for semantics.
 *
 * @tparam T,N,P,Qd,ACCUMULATE  See glass::bilinear.
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha,X,M,Y,beta,R,s_scratch  See glass::bilinear.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t N, uint32_t P, uint32_t Qd,
          bool ACCUMULATE = false, bool TRAILING_SYNC = true>
__device__ void bilinear(T alpha, const T* X, const T* M, const T* Y, T beta, T* R, T* s_scratch,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm(N, Qd, N, static_cast<T>(1), const_cast<T*>(M), const_cast<T*>(Y), s_scratch, g);
    g.sync();
    glass::detail::xtY_impl<T, N, P, Qd, false, ACCUMULATE>(
        g.thread_rank(), g.size(), alpha, X, s_scratch, beta, R);
    if constexpr (TRAILING_SYNC) g.sync();
}

// ─── contraction-parallel SYRK (cooperative-groups variant) ──────────────────

/**
 * @brief Contraction-parallel symmetric rank-k update: `C = alpha * A·op(A) + beta * C` (cooperative-groups variant).
 *
 * Cooperative-groups form of `glass::syrk_reduced`. Pass a warp-multiple group.
 *
 * @tparam T,ROWS,COLS,TRANSPOSE  See glass::syrk_reduced.
 * @tparam TRAILING_SYNC  Emit a trailing `g.sync()` (default true).
 * @param alpha,A,beta,C  See glass::syrk_reduced.
 * @param g  Cooperative thread group (defaults to the whole block; pass a warp-multiple group).
 */
template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void syrk_reduced(T alpha, const T* A, T beta, T* C,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::detail::syrk_reduced_impl<T, ROWS, COLS, TRANSPOSE, true>(g.thread_rank(), g.size(), alpha, A, beta, C);
    if constexpr (TRAILING_SYNC) g.sync();
}

/**
 * @brief Contraction-parallel SYRK with implicit `beta = 0`: `C = alpha * A·op(A)` (cooperative-groups variant).
 *
 * @tparam T,ROWS,COLS,TRANSPOSE,TRAILING_SYNC  See the beta overload.
 * @param alpha,A,C,g  See the beta overload.
 */
template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void syrk_reduced(T alpha, const T* A, T* C,
                             cgrps::thread_group g = cgrps::this_thread_block())
{
    glass::detail::syrk_reduced_impl<T, ROWS, COLS, TRANSPOSE, false>(g.thread_rank(), g.size(), alpha, A, static_cast<T>(0), C);
    if constexpr (TRAILING_SYNC) g.sync();
}
