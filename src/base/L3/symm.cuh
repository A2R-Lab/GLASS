#pragma once
#include "../barrier.cuh"
#include "../flags.cuh"   // FillMode / Diag (shared with trsv/trmv/trsm/syrk)
#include <cstdint>

/**
 * @file symm.cuh
 * @brief SYMM (symmetric matrix-matrix multiply, left side) and TRMM
 *        (triangular matrix-matrix multiply, left side, out-of-place).
 *
 * `symm`: `C = alpha * A * B + beta * C` where `A` is `n x n` SYMMETRIC with
 * only the `FILL` triangle stored — the other triangle is never read, it is
 * reconstructed by mirroring (`A[i,k]` reads `A[k + i*n]` when `(i,k)` falls
 * outside the stored triangle). `B` and `C` are `n x m` column-major.
 *
 * `trmm`: `C = alpha * op(A) * B` where `A` is `n x n` triangular (only the
 * `FILL` triangle read; `DIAG=Diag::Unit` means an implicit unit diagonal that
 * is never read) and `op(A) = Aᵀ` when `TRANSPOSE`. Out-of-place into `C`
 * (deliberately NOT the BLAS in-place `B := op(A) B` — a separate output keeps
 * the flat one-output-per-thread loop race-free with no interior barrier).
 * `C` must not alias `A` or `B`.
 *
 * Both use gemm's plain-path parallelism: each thread owns disjoint output
 * elements of the flat `n*m` space (`el += size` stride) with a serial
 * ascending-k inner chain per element, so results are bit-identical at any
 * thread count and NO interior barrier is needed (guide §1a counter-note).
 * Block only (no `warp::` variants yet — pack via `warp::gemm` on a
 * materialized matrix in the meantime; future work).
 */

// ─── symm shared body ─────────────────────────────────────────────────────────
// C = alpha * A_sym * B (+ beta * C).  A: n x n, FILL triangle stored, read
// mirrored.  B, C: n x m column-major.  HAS_BETA=false never reads C.
// SizeT/SizeU deduced: uint32_t (runtime) or ct_size<N>/ct_size<M> (compile
// time — constant-folds trip counts and the %/ indexing).
template <typename Bar, typename T, FillMode FILL, bool HAS_BETA, bool TRAILING_SYNC,
          typename SizeT, typename SizeU>
__device__ void symm_impl(Bar bar, SizeT n, SizeU m, T alpha,
                          const T *__restrict__ A, const T *__restrict__ B,
                          T beta, T *__restrict__ C)
{
    static_assert(FILL != FillMode::Full, "symm: FILL must name the stored triangle (Lower or Upper)");
    uint32_t rank = bar.rank(), size = bar.size();
    const uint32_t maxel = n * m;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t i = el % n, j = el / n;
        T res = static_cast<T>(0);
        for (uint32_t k = 0; k < n; k++) {
            // stored triangle: Lower keeps row >= col, Upper keeps row <= col;
            // the mirrored read A[k + i*n] supplies the other half.
            const bool in_tri = (FILL == FillMode::Lower) ? (i >= k) : (i <= k);
            const T a = in_tri ? A[i + k*n] : A[k + i*n];
            res += a * B[k + j*n];
        }
        if constexpr (HAS_BETA) C[el] = beta_blend(alpha*res, beta, C[el]);
        else                    C[el] = alpha*res;
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Symmetric matrix-matrix multiply (left side): `C = alpha * A * B + beta * C` (SYMM).
 *
 * `A` is `n x n` symmetric with only the `FILL` triangle stored (column-major;
 * the other triangle is never read — reconstructed by mirroring). `B` and `C`
 * are `n x m` column-major. Single-block, flat one-output-per-thread
 * parallelism (serial ascending-k chain per element ⇒ bit-identical at any
 * thread count, no interior barrier). BLAS: `SSYMM('L', uplo, ...)`. NumPy
 * (Lower): `S = np.tril(A) + np.tril(A, -1).T; C = alpha * S @ B + beta * C`.
 *
 * @tparam T     Scalar type.
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param n      Dimension of `A` (n x n) and rows of `B`/`C`.
 * @param m      Columns of `B`/`C`.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Symmetric matrix, `FILL` triangle stored (column-major; read-only).
 * @param B      Input matrix (n x m, column-major).
 * @param beta   Scalar multiplier on the existing C (read only when `beta != 0`).
 * @param C      In/out result matrix (n x m, column-major); must not alias A/B.
 */
template <typename T, FillMode FILL = FillMode::Lower, bool TRAILING_SYNC = true>
__device__ void symm(uint32_t n, uint32_t m, T alpha,
                     const T *__restrict__ A, const T *__restrict__ B,
                     T beta, T *__restrict__ C)
{
    symm_impl<BlockBarrier, T, FILL, true, TRAILING_SYNC>(BlockBarrier{}, n, m, alpha, A, B, beta, C);
}

/**
 * @brief SYMM with implicit `beta = 0`: `C = alpha * A * B` (overwrite).
 *
 * Runtime-size overload that overwrites `C` (the existing C is never read) —
 * safe to write into uninitialized scratch. NumPy (Lower):
 * `C = alpha * (np.tril(A) + np.tril(A, -1).T) @ B`.
 *
 * @tparam T     Scalar type.
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param n,m    `A` is n x n; `B`/`C` are n x m.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Symmetric matrix, `FILL` triangle stored (read-only).
 * @param B      Input matrix (n x m, column-major).
 * @param C      Output result matrix (overwritten); must not alias A/B.
 */
template <typename T, FillMode FILL = FillMode::Lower, bool TRAILING_SYNC = true>
__device__ void symm(uint32_t n, uint32_t m, T alpha,
                     const T *__restrict__ A, const T *__restrict__ B,
                     T *__restrict__ C)
{
    symm_impl<BlockBarrier, T, FILL, false, TRAILING_SYNC>(BlockBarrier{}, n, m, alpha, A, B, static_cast<T>(0), C);
}

/**
 * @brief Compile-time-size SYMM: `C = alpha * A * B + beta * C`.
 *
 * Same as the runtime `symm` but with the dimensions as template parameters
 * (unrolled inner loop, magic-number `%`/`/` indexing). BLAS:
 * `SSYMM('L', uplo, ...)`. NumPy (Lower):
 * `C = alpha * (np.tril(A) + np.tril(A, -1).T) @ B + beta * C`.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension of `A` (N x N) and rows of `B`/`C`.
 * @tparam M     Columns of `B`/`C`.
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Symmetric matrix, `FILL` triangle stored (read-only).
 * @param B      Input matrix (N x M, column-major).
 * @param beta   Scalar multiplier on the existing C (read only when `beta != 0`).
 * @param C      In/out result matrix (N x M, column-major); must not alias A/B.
 */
template <typename T, uint32_t N, uint32_t M,
          FillMode FILL = FillMode::Lower, bool TRAILING_SYNC = true>
__device__ void symm(T alpha, const T *__restrict__ A, const T *__restrict__ B,
                     T beta, T *__restrict__ C)
{
    symm_impl<BlockBarrier, T, FILL, true, TRAILING_SYNC>(BlockBarrier{}, ct_size<N>{}, ct_size<M>{}, alpha, A, B, beta, C);
}

/**
 * @brief Compile-time-size SYMM with implicit `beta = 0`: `C = alpha * A * B` (overwrite).
 *
 * Compile-time-size overload that overwrites `C` (never read) — safe into
 * uninitialized scratch. NumPy (Lower):
 * `C = alpha * (np.tril(A) + np.tril(A, -1).T) @ B`.
 *
 * @tparam T     Scalar type.
 * @tparam N,M   `A` is N x N; `B`/`C` are N x M.
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Symmetric matrix, `FILL` triangle stored (read-only).
 * @param B      Input matrix (N x M, column-major).
 * @param C      Output result matrix (overwritten); must not alias A/B.
 */
template <typename T, uint32_t N, uint32_t M,
          FillMode FILL = FillMode::Lower, bool TRAILING_SYNC = true>
__device__ void symm(T alpha, const T *__restrict__ A, const T *__restrict__ B,
                     T *__restrict__ C)
{
    symm_impl<BlockBarrier, T, FILL, false, TRAILING_SYNC>(BlockBarrier{}, ct_size<N>{}, ct_size<M>{}, alpha, A, B, static_cast<T>(0), C);
}

// ─── trmm shared body ─────────────────────────────────────────────────────────
// C = alpha * op(A_tri) * B, out-of-place.  op(A) is lower-triangular when
// (FILL==Lower) != TRANSPOSE (the trsv/trsm rule); the k-chain is restricted to
// op(A)'s structural nonzeros (k <= i for lower, k >= i for upper), ascending —
// deterministic and thread-count invariant.  Diag::Unit never reads A's diagonal.
template <typename Bar, typename T, FillMode FILL, Diag DIAG, bool TRANSPOSE, bool TRAILING_SYNC,
          typename SizeT, typename SizeU>
__device__ void trmm_impl(Bar bar, SizeT n, SizeU m, T alpha,
                          const T *__restrict__ A, const T *__restrict__ B,
                          T *__restrict__ C)
{
    static_assert(FILL != FillMode::Full, "trmm: FILL must name a triangle (Lower or Upper)");
    constexpr bool LOWER_OP = ((FILL == FillMode::Lower) != TRANSPOSE);  // op(A) lower-triangular
    constexpr bool UNIT     = (DIAG == Diag::Unit);
    uint32_t rank = bar.rank(), size = bar.size();
    const uint32_t maxel = n * m;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t i = el % n, j = el / n;
        // op(A)[i,k] is structurally nonzero for k in [k0, k1).
        const uint32_t k0 = LOWER_OP ? 0u : i;
        const uint32_t k1 = LOWER_OP ? (i + 1u) : static_cast<uint32_t>(n);
        T res = static_cast<T>(0);
        for (uint32_t k = k0; k < k1; k++) {
            // op(A)[i][k] = TRANSPOSE ? A[k + i*n] : A[i + k*n] (trsv/trsm rule);
            // the diagonal is A[i + i*n] either way, or implicit 1 for Diag::Unit.
            const T a = (k == i) ? (UNIT ? static_cast<T>(1) : A[i + i*n])
                                 : (TRANSPOSE ? A[k + i*n] : A[i + k*n]);
            res += a * B[k + j*n];
        }
        C[el] = alpha * res;
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Triangular matrix-matrix multiply (left side): `C = alpha * op(A) * B` (TRMM, out-of-place).
 *
 * `A` is `n x n` triangular, column-major; only the `FILL` triangle is read.
 * `TRANSPOSE=true` multiplies by `Aᵀ` against that same stored triangle;
 * `DIAG=Diag::Unit` means an implicit unit diagonal (`A`'s diagonal is never
 * read). Unlike BLAS TRMM this writes a SEPARATE output `C` (n x m,
 * column-major) instead of overwriting `B` in place — out-of-place keeps the
 * flat one-output-per-thread loop race-free with no interior barrier. `C`
 * must not alias `A` or `B`. The per-element k-chain covers only `op(A)`'s
 * structural nonzeros, ascending ⇒ bit-identical at any thread count.
 * BLAS: `STRMM('L', uplo, transa, diag, ...)` (plus the copy). NumPy (Lower,
 * NonUnit, no transpose): `C = alpha * np.tril(A) @ B`.
 *
 * @tparam T     Scalar type.
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true multiply by `Aᵀ` (default false).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param n      Dimension of `A` (n x n) and rows of `B`/`C`.
 * @param m      Columns of `B`/`C`.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Triangular matrix (column-major; only the `FILL` triangle read).
 * @param B      Input matrix (n x m, column-major; read-only).
 * @param C      Output result matrix (n x m, column-major, overwritten); no aliasing.
 */
template <typename T, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit,
          bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void trmm(uint32_t n, uint32_t m, T alpha,
                     const T *__restrict__ A, const T *__restrict__ B,
                     T *__restrict__ C)
{
    trmm_impl<BlockBarrier, T, FILL, DIAG, TRANSPOSE, TRAILING_SYNC>(BlockBarrier{}, n, m, alpha, A, B, C);
}

/**
 * @brief Compile-time-size TRMM: `C = alpha * op(A) * B` (out-of-place).
 *
 * Same as the runtime `trmm` but with the dimensions as template parameters
 * (unrolled inner loop, magic-number `%`/`/` indexing). NumPy (Lower, NonUnit,
 * no transpose): `C = alpha * np.tril(A) @ B`.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension of `A` (N x N) and rows of `B`/`C`.
 * @tparam M     Columns of `B`/`C`.
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true multiply by `Aᵀ` (default false).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Triangular matrix (column-major; only the `FILL` triangle read).
 * @param B      Input matrix (N x M, column-major; read-only).
 * @param C      Output result matrix (N x M, column-major, overwritten); no aliasing.
 */
template <typename T, uint32_t N, uint32_t M,
          FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit,
          bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void trmm(T alpha, const T *__restrict__ A, const T *__restrict__ B,
                     T *__restrict__ C)
{
    trmm_impl<BlockBarrier, T, FILL, DIAG, TRANSPOSE, TRAILING_SYNC>(BlockBarrier{}, ct_size<N>{}, ct_size<M>{}, alpha, A, B, C);
}
