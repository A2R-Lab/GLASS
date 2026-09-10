#pragma once
#include "../barrier.cuh"
#include "../flags.cuh"   // FillMode / Diag
#include <cstdint>
#include <math.h>
#include "trsm.cuh"       // trsm_impl, composed by getrs

// ─────────────────────────────────────────────────────────────────────────────
// getrf / getrs / gesv — LU factorization with PARTIAL (row) PIVOTING and the
// matching solves, plus the LAPACK-style row-interchange helper laswp. This is
// the robustness path for GENERAL (non-SPD, non-symmetric) systems: potrf/posv
// require SPD, ldlt handles symmetric-indefinite, and the nvidia:: leg only
// wraps the NO-pivot cuSOLVERDx getrf — this SIMT path is the one that
// survives a zero/small leading pivot on an arbitrary invertible matrix.
//
// Conventions (LAPACK/SciPy `lu_factor` exactly, 0-based):
//   • A (n×n, column-major) is factored in place as P·A = L·U with L unit-lower
//     (multipliers below the diagonal, implicit unit diagonal) and U on/above.
//   • piv[k] = the row that was swapped with row k at step k (ipiv, 0-based),
//     so piv[k] >= k always. Applying the swaps SEQUENTIALLY in k reproduces P.
//     The (LU, piv) pair is drop-in for `scipy.linalg.lu_solve((lu, piv), b)`.
//   • Pivot choice is argmax |A[i,k]| over i >= k with ties going to the
//     SMALLEST index — deterministic, so the factorization is thread-count
//     invariant by construction.
//
// Parallel structure per elimination step k (barriers between every dependent
// phase): rank-0 serial pivot scan (length n−k, deterministic) → barrier →
// full-row swap thread-strided over all n columns → barrier → multiplier
// column divide thread-strided over i > k → barrier → trailing (n−k−1)²
// update flat-strided over the rectangle → barrier. The pivot scan at step
// k+1 reads column k+1 AFTER the trailing update's barrier.
// ─────────────────────────────────────────────────────────────────────────────

// Shared body: apply row interchanges col[k] ↔ col[piv[k]] for k in [k0,k1)
// to each of `ncols` columns (leading dimension n). The swaps for a given
// column compose SEQUENTIALLY in k, but distinct columns never interact — so
// each thread owns whole columns (outer stride) and applies the full swap
// sequence serially within them. No interior barrier needed; trivially
// thread-count invariant. REVERSE applies the swaps in the opposite order
// (k1−1 down to k0), i.e. the inverse permutation (LAPACK laswp INCX = −1).
template <typename Bar, typename T, bool REVERSE, bool TRAILING_SYNC>
__device__ void laswp_impl(Bar bar, uint32_t n, uint32_t ncols, T *A,
                           const uint32_t *piv, uint32_t k0, uint32_t k1)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t j = rank; j < ncols; j += size) {
        T *col = A + j * n;
        for (uint32_t s = k0; s < k1; s++) {
            uint32_t k = REVERSE ? (k0 + k1 - 1 - s) : s;
            uint32_t p = piv[k];
            if (p != k) { T tmp = col[k]; col[k] = col[p]; col[p] = tmp; }
        }
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Apply LAPACK-style row interchanges to a rectangular matrix (LASWP).
 *
 * Applies `A[k,:] ↔ A[piv[k],:]` sequentially for `k` in `[k0, k1)` to the
 * `n×ncols` column-major matrix `A` (leading dimension `n`). This is how a
 * `getrf` pivot vector is applied to right-hand sides before the triangular
 * solves. `REVERSE=true` applies the swaps in the opposite order (`k1−1` down
 * to `k0`) — the INVERSE permutation, LAPACK `laswp` with `INCX = −1`.
 * Each thread owns whole columns and applies the swap sequence serially
 * within them (columns are independent), so the result is thread-count
 * invariant with no interior barrier. LAPACK equivalent: `?laswp`; NumPy:
 * `for k in range(k0, k1): A[[k, piv[k]], :] = A[[piv[k], k], :]`.
 *
 * @tparam T        Scalar type.
 * @tparam REVERSE  Apply the swaps in reverse order (inverse permutation; default false).
 * @tparam TRAILING_SYNC  End on a barrier so the permuted A is block-visible (default true).
 * @param n      Leading dimension / number of rows of `A`.
 * @param ncols  Number of columns swaps are applied across.
 * @param A      In/out `n×ncols` matrix (column-major).
 * @param piv    Pivot indices (`piv[k]` = row swapped with row `k`; 0-based).
 * @param k0,k1  Half-open range of swap steps to apply.
 */
template <typename T, bool REVERSE = false, bool TRAILING_SYNC = true>
__device__ void laswp(uint32_t n, uint32_t ncols, T *A,
                      const uint32_t *piv, uint32_t k0, uint32_t k1)
{
    laswp_impl<BlockBarrier, T, REVERSE, TRAILING_SYNC>(BlockBarrier{}, n, ncols, A, piv, k0, k1);
}

/**
 * @brief Apply LAPACK-style row interchanges to a square n-column matrix (LASWP).
 *
 * Square convenience form of the rectangular `laswp`: applies
 * `A[k,:] ↔ A[piv[k],:]` sequentially for `k` in `[k0, k1)` to the `n×n`
 * column-major matrix `A`. Swaps compose sequentially in `k` but are
 * thread-strided across columns (each thread owns whole columns).
 * NumPy: `for k in range(k0, k1): A[[k, piv[k]], :] = A[[piv[k], k], :]`.
 *
 * @tparam T        Scalar type.
 * @tparam REVERSE  Apply the swaps in reverse order (inverse permutation; default false).
 * @tparam TRAILING_SYNC  End on a barrier (default true).
 * @param n      Matrix dimension (`A` is `n×n`).
 * @param A      In/out `n×n` matrix (column-major).
 * @param piv    Pivot indices (`piv[k]` = row swapped with row `k`; 0-based).
 * @param k0,k1  Half-open range of swap steps to apply.
 */
template <typename T, bool REVERSE = false, bool TRAILING_SYNC = true>
__device__ void laswp(uint32_t n, T *A, const uint32_t *piv, uint32_t k0, uint32_t k1)
{
    laswp_impl<BlockBarrier, T, REVERSE, TRAILING_SYNC>(BlockBarrier{}, n, n, A, piv, k0, k1);
}

/**
 * @brief Apply LAPACK-style row interchanges to a vector (LASWP, single column).
 *
 * Vector form: applies `x[k] ↔ x[piv[k]]` sequentially for `k` in `[k0, k1)`.
 * The swaps compose (they must run in order), so one thread applies them
 * serially; the routine ends on a barrier so the permuted `x` is
 * block-visible. `REVERSE=true` undoes a forward application. NumPy:
 * `for k in range(k0, k1): x[[k, piv[k]]] = x[[piv[k], k]]`.
 *
 * @tparam T        Scalar type.
 * @tparam REVERSE  Apply the swaps in reverse order (inverse permutation; default false).
 * @tparam TRAILING_SYNC  End on a barrier (default true).
 * @param piv    Pivot indices (`piv[k]` = element swapped with element `k`; 0-based).
 * @param k0,k1  Half-open range of swap steps to apply.
 * @param x      In/out vector.
 */
template <typename T, bool REVERSE = false, bool TRAILING_SYNC = true>
__device__ void laswp(const uint32_t *piv, uint32_t k0, uint32_t k1, T *x)
{
    laswp_impl<BlockBarrier, T, REVERSE, TRAILING_SYNC>(BlockBarrier{}, /*n=*/1, /*ncols=*/1, x, piv, k0, k1);
}

// Shared body: in-place LU with partial pivoting. piv[k] doubles as the shared
// slot broadcasting rank 0's pivot choice (written, barrier, read by all).
template <typename Bar, typename T, bool CHECK>
__device__ void getrf_impl(Bar bar, uint32_t n, T *A, uint32_t *piv, int *s_fail)
{
    uint32_t rank = bar.rank(), size = bar.size();
    if constexpr (CHECK) { if (rank == 0 && s_fail) *s_fail = 0; }   // only rank 0 writes s_fail
    for (uint32_t k = 0; k < n; k++) {
        // (a) partial-pivot search: rank 0 serially scans |A[i,k]|, i >= k.
        // Strict `>` keeps the SMALLEST index on ties — deterministic, so the
        // result cannot depend on the block size (same rule as iamax).
        if (rank == 0) {
            T best = abs(A[k + k * n]);
            uint32_t p = k;
            for (uint32_t i = k + 1; i < n; i++) {
                T v = abs(A[i + k * n]);
                if (v > best) { best = v; p = i; }
            }
            piv[k] = p;
        }
        bar.sync();
        // (c) swap FULL rows k ↔ piv[k] (including the L columns j < k, per
        // LAPACK), thread-strided over all n columns.
        uint32_t p = piv[k];
        if (p != k) {
            for (uint32_t j = rank; j < n; j += size) {
                T tmp = A[k + j * n]; A[k + j * n] = A[p + j * n]; A[p + j * n] = tmp;
            }
        }
        bar.sync();
        // (e) multipliers: A[i,k] /= A[k,k] for i > k. Every thread reads the
        // same post-swap pivot, so the CHECK verdict is uniform across the block.
        T pivval = A[k + k * n];
        bool div = true;
        if constexpr (CHECK) {
            if (pivval == static_cast<T>(0) || isnan(pivval) || isinf(pivval)) {
                div = false;                                // skip the divide (LAPACK: INFO=k)
                if (rank == 0 && s_fail) *s_fail = 1;
            }
        }
        if (div) {
            for (uint32_t i = rank + k + 1; i < n; i += size)
                A[i + k * n] /= pivval;
        }
        bar.sync();
        // (g) trailing update: A[i,j] -= A[i,k]*A[k,j] for i,j > k, flat-strided
        // over the (n−k−1)² rectangle. Leaves column k+1 ready for the next
        // pivot scan (barrier below).
        uint32_t rem = n - 1 - k;
        for (uint32_t flat = rank; flat < rem * rem; flat += size) {
            uint32_t i = k + 1 + flat % rem;
            uint32_t j = k + 1 + flat / rem;
            A[i + j * n] -= A[i + k * n] * A[k + j * n];
        }
        bar.sync();
    }
}

/**
 * @brief In-place LU factorization with partial pivoting (LAPACK getrf).
 *
 * Factors `P·A = L·U`, overwriting the `n×n` column-major `A` with `L\\U`
 * (unit-lower `L`'s multipliers strictly below the diagonal, `U` on and above)
 * and recording the row interchanges in `piv` (`piv[k]` = row swapped with row
 * `k` at step `k`; LAPACK ipiv convention, 0-based, so `piv[k] >= k`). Partial
 * pivoting (argmax `|A[i,k]|`, ties to the smallest index) makes this the
 * robust factorization for GENERAL non-SPD matrices — it succeeds on any
 * invertible `A`, including a zero leading pivot where no-pivot LU fails.
 * SciPy equivalent: `lu, piv = scipy.linalg.lu_factor(A)` — the output pair is
 * drop-in for `scipy.linalg.lu_solve((lu, piv), b)`.
 *
 * When `CHECK` is true and `s_fail` is non-null, a zero or non-finite pivot
 * sets `*s_fail = 1` and skips that column's divide (`A` is singular to
 * working precision; the factor is not usable), leaving `*s_fail = 0`
 * otherwise. `CHECK` defaults false and compiles out entirely
 * (`if constexpr`), so the unchecked instantiation is byte-identical.
 *
 * Ends on a barrier; deterministic pivot choice + barriers between every
 * dependent phase make the output thread-count invariant.
 *
 * @tparam T      Scalar type.
 * @tparam CHECK  If true, detect a zero/non-finite pivot and report it via `s_fail` (default false, compiles out).
 * @param n       Matrix dimension (`A` is `n×n`).
 * @param A       In/out `n×n` matrix (column-major); on return holds `L\\U`.
 * @param piv     Output pivot indices, length `n` (`piv[k]` = row swapped with row `k`; 0-based).
 * @param s_fail  Optional flag (CHECK only): set to 1 on a zero/non-finite pivot, else 0. Ignored when null.
 */
// TRAILING_SYNC: accepted for uniformity, documented NO-OP here (the pivot
// broadcast/elimination tail is fused into the final column step).
template <typename T, bool CHECK = false, bool TRAILING_SYNC = true>
__device__ void getrf(uint32_t n, T *A, uint32_t *piv, int *s_fail = nullptr)
{
    getrf_impl<BlockBarrier, T, CHECK>(BlockBarrier{}, n, A, piv, s_fail);
}

/**
 * @brief In-place LU factorization with partial pivoting, compile-time size (LAPACK getrf).
 *
 * Compile-time-`N` overload of `getrf`, forwarding to the runtime form (same
 * `L\\U` layout, 0-based LAPACK ipiv `piv`, and `CHECK` semantics). SciPy
 * equivalent: `lu, piv = scipy.linalg.lu_factor(A)`.
 *
 * @tparam T      Scalar type.
 * @tparam N      Matrix dimension (`A` is `N×N`).
 * @tparam CHECK  If true, detect a zero/non-finite pivot and report it via `s_fail` (default false, compiles out).
 * @param A       In/out `N×N` matrix (column-major); on return holds `L\\U`.
 * @param piv     Output pivot indices, length `N` (0-based LAPACK ipiv).
 * @param s_fail  Optional flag (CHECK only): set to 1 on a zero/non-finite pivot, else 0. Ignored when null.
 */
template <typename T, uint32_t N, bool CHECK = false, bool TRAILING_SYNC = true>
__device__ void getrf(T *A, uint32_t *piv, int *s_fail = nullptr)
{
    getrf_impl<BlockBarrier, T, CHECK>(BlockBarrier{}, N, A, piv, s_fail);
}

// Shared body: standard LAPACK getrs, both transpose modes.
//   P·A = L·U  ⇒  A = Pᵀ·L·U  and  Aᵀ = Uᵀ·Lᵀ·P.
//   TRANSPOSE=false: apply P to B, then L·Y = P·B (unit forward), U·X = Y (backward).
//   TRANSPOSE=true : Uᵀ·Z = B (forward), Lᵀ·W = Z (unit backward), X = Pᵀ·W
//                    (piv applied LAST, in REVERSE order = inverse permutation).
template <typename Bar, typename T, bool TRANSPOSE, bool TRAILING_SYNC = true>
__device__ void getrs_impl(Bar bar, uint32_t n, uint32_t nrhs,
                           const T *LU, const uint32_t *piv, T *B)
{
    if constexpr (!TRANSPOSE) {
        laswp_impl<Bar, T, /*REVERSE=*/false, /*TRAILING_SYNC=*/true>(bar, n, nrhs, B, piv, 0, n);
        trsm_impl<Bar, T, FillMode::Lower, Diag::Unit,    /*TRANSPOSE=*/false>(bar, n, nrhs, LU, B);
        trsm_impl<Bar, T, FillMode::Upper, Diag::NonUnit, /*TRANSPOSE=*/false, TRAILING_SYNC>(bar, n, nrhs, LU, B);
    } else {
        trsm_impl<Bar, T, FillMode::Upper, Diag::NonUnit, /*TRANSPOSE=*/true>(bar, n, nrhs, LU, B);
        trsm_impl<Bar, T, FillMode::Lower, Diag::Unit,    /*TRANSPOSE=*/true>(bar, n, nrhs, LU, B);
        laswp_impl<Bar, T, /*REVERSE=*/true, TRAILING_SYNC>(bar, n, nrhs, B, piv, 0, n);
    }
}

/**
 * @brief Solve `op(A) X = B` from a pivoted LU factorization, in place (LAPACK getrs).
 *
 * Uses the `(LU, piv)` pair produced by `getrf` to solve for all `nrhs`
 * columns of `B` (`n×nrhs`, column-major), overwriting `B` with `X`.
 * `TRANSPOSE=false` solves `A X = B`: apply the row interchanges to `B`
 * (`laswp`), forward-solve `L Y = P·B` (unit-lower `trsm`), back-solve
 * `U X = Y` (upper `trsm`). `TRANSPOSE=true` solves `Aᵀ X = B` in the reverse
 * order with transposed flags — `Uᵀ Z = B`, then `Lᵀ W = Z`, then the
 * interchanges applied LAST in reverse order (the inverse permutation).
 * Ends on a barrier. SciPy equivalent:
 * `X = scipy.linalg.lu_solve((lu, piv), B, trans=(1 if TRANSPOSE else 0))`.
 *
 * @tparam T          Scalar type.
 * @tparam TRANSPOSE  When true solve `Aᵀ X = B` (default false).
 * @param n     Dimension (`LU` is `n×n`; each column of `B` has length `n`).
 * @param nrhs  Number of right-hand sides (columns of `B`).
 * @param LU    Factorization from `getrf` (`L\\U`, column-major; read-only).
 * @param piv   Pivot indices from `getrf` (0-based LAPACK ipiv; read-only).
 * @param B     In/out right-hand sides (`n×nrhs`, column-major); on return holds `X`.
 */
template <typename T, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void getrs(uint32_t n, uint32_t nrhs, const T *LU, const uint32_t *piv, T *B)
{
    getrs_impl<BlockBarrier, T, TRANSPOSE, TRAILING_SYNC>(BlockBarrier{}, n, nrhs, LU, piv, B);
}

/**
 * @brief Solve `op(A) X = B` from a pivoted LU factorization, compile-time size (LAPACK getrs).
 *
 * Compile-time-`N`/`NRHS` overload of `getrs`, forwarding to the runtime form.
 * SciPy equivalent:
 * `X = scipy.linalg.lu_solve((lu, piv), B, trans=(1 if TRANSPOSE else 0))`.
 *
 * @tparam T          Scalar type.
 * @tparam N          Dimension (`LU` is `N×N`; each column of `B` has length `N`).
 * @tparam NRHS       Number of right-hand sides (columns of `B`).
 * @tparam TRANSPOSE  When true solve `Aᵀ X = B` (default false).
 * @param LU   Factorization from `getrf` (`L\\U`, column-major; read-only).
 * @param piv  Pivot indices from `getrf` (0-based LAPACK ipiv; read-only).
 * @param B    In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
 */
template <typename T, uint32_t N, uint32_t NRHS, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void getrs(const T *LU, const uint32_t *piv, T *B)
{
    getrs_impl<BlockBarrier, T, TRANSPOSE, TRAILING_SYNC>(BlockBarrier{}, N, NRHS, LU, piv, B);
}

/**
 * @brief Solve the general system `A X = B` via pivoted LU, in place (LAPACK gesv).
 *
 * The composed general dense solve: `getrf` (in-place pivoted LU of `A`) then
 * `getrs` (permute + two triangular solves on `B`). On return `A` holds `L\\U`,
 * `piv` the interchanges, and `B` the solution `X` (`n×nrhs`, column-major).
 * This is the robust path for GENERAL non-symmetric matrices — where
 * `posv`/`ldlt_solve` require SPD/symmetry, `gesv` only requires
 * invertibility (partial pivoting handles zero/small leading pivots). NumPy
 * equivalent: `X = np.linalg.solve(A, B)`.
 *
 * When `CHECK` is true and `s_fail` is non-null, a zero/non-finite pivot in
 * the factorization sets `*s_fail = 1` (the "solution" is then meaningless —
 * callers must test the flag), else `0`. `CHECK` defaults false and compiles
 * out. Ends on a barrier.
 *
 * @tparam T      Scalar type.
 * @tparam CHECK  If true, report a zero/non-finite pivot via `s_fail` (default false, compiles out).
 * @param n       Dimension (`A` is `n×n`; each column of `B` has length `n`).
 * @param nrhs    Number of right-hand sides (columns of `B`).
 * @param A       In/out `n×n` matrix (column-major); on return holds `L\\U`.
 * @param piv     Output pivot indices, length `n` (0-based LAPACK ipiv).
 * @param B       In/out right-hand sides (`n×nrhs`, column-major); on return holds `X`.
 * @param s_fail  Optional flag (CHECK only): set to 1 on a zero/non-finite pivot, else 0. Ignored when null.
 */
template <typename T, bool CHECK = false, bool TRAILING_SYNC = true>
__device__ void gesv(uint32_t n, uint32_t nrhs, T *A, uint32_t *piv, T *B, int *s_fail = nullptr)
{
    getrf_impl<BlockBarrier, T, CHECK>(BlockBarrier{}, n, A, piv, s_fail);
    getrs_impl<BlockBarrier, T, /*TRANSPOSE=*/false, TRAILING_SYNC>(BlockBarrier{}, n, nrhs, A, piv, B);
}

/**
 * @brief Solve the general system `A X = B` via pivoted LU, compile-time size (LAPACK gesv).
 *
 * Compile-time-`N`/`NRHS` overload of `gesv`, forwarding to the runtime form
 * (same in-place `L\\U`/`piv`/`X` outputs and `CHECK` semantics). NumPy
 * equivalent: `X = np.linalg.solve(A, B)`.
 *
 * @tparam T      Scalar type.
 * @tparam N      Dimension (`A` is `N×N`; each column of `B` has length `N`).
 * @tparam NRHS   Number of right-hand sides (columns of `B`).
 * @tparam CHECK  If true, report a zero/non-finite pivot via `s_fail` (default false, compiles out).
 * @param A       In/out `N×N` matrix (column-major); on return holds `L\\U`.
 * @param piv     Output pivot indices, length `N` (0-based LAPACK ipiv).
 * @param B       In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
 * @param s_fail  Optional flag (CHECK only): set to 1 on a zero/non-finite pivot, else 0. Ignored when null.
 */
template <typename T, uint32_t N, uint32_t NRHS, bool CHECK = false, bool TRAILING_SYNC = true>
__device__ void gesv(T *A, uint32_t *piv, T *B, int *s_fail = nullptr)
{
    gesv<T, CHECK, TRAILING_SYNC>(N, NRHS, A, piv, B, s_fail);
}
