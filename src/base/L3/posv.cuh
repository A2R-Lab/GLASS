#pragma once
#include <cstdint>

/**
 * @file posv.cuh
 * @brief SPD linear solve via Cholesky + two triangular solves (pure SIMT).
 *
 * `posv` / `potrs` are thin single-block compositions of `cholDecomp_InPlace`
 * (`chol_InPlace.cuh`) and `trsv` (`trsv.cuh`). Both callees end with a trailing
 * `__syncthreads()`, so the factor and the two solves compose with NO inter-call
 * barrier. Pure-SIMT companion to `glass::nvidia::posv`. Column-major throughout.
 */

/**
 * @brief Solve the SPD system `A x = b` in one block (LAPACK posv).
 *
 * Factors `A = L Lᵀ` in place via Cholesky, then forward-solves `L y = b` and
 * back-solves `Lᵀ x = y`. On return `A` holds its lower Cholesky factor `L` and
 * `b` holds the solution `x`. `A` must be symmetric positive-definite; behaviour
 * on non-SPD input is undefined (the Cholesky step produces NaN, no info flag).
 * Thread-count invariant. NumPy equivalent: `x = np.linalg.solve(A, b)` (A SPD).
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Dimension (`A` is `n×n`, `b` has length `n`).
 * @param A  In/out SPD matrix (column-major); overwritten with its factor `L`.
 * @param b  In/out right-hand side; on return holds the solution `x`.
 */
template <typename T>
__device__ void posv(uint32_t n, T *A, T *b)
{
    cholDecomp_InPlace<T>(n, A);            // A -> L (lower); trailing __syncthreads
    trsv<T, true, false, false>(n, A, b);  // forward: L y = b
    trsv<T, true, false, true>(n, A, b);   // back:    Lᵀ x = y
}

/**
 * @brief Compile-time-size SPD solve `A x = b` (LAPACK posv).
 *
 * Same as the runtime `posv` with the dimension as a template parameter.
 * NumPy equivalent: `x = np.linalg.solve(A, b)` (A SPD).
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension (`A` is `N×N`, `b` has length `N`).
 * @param A  In/out SPD matrix (column-major); overwritten with its factor `L`.
 * @param b  In/out right-hand side; on return holds the solution `x`.
 */
template <typename T, uint32_t N>
__device__ void posv(T *A, T *b) { posv<T>(N, A, b); }

/**
 * @brief Solve the SPD system `A x = b` from a precomputed Cholesky factor (LAPACK potrs).
 *
 * Given the lower factor `L` (e.g. from `cholDecomp_InPlace`), solves
 * `L Lᵀ x = b` by forward then back substitution — the reusable-factor /
 * multi-solve path (no re-factor). `L` is read-only; `b` is overwritten with `x`.
 * Thread-count invariant. SciPy equivalent: `x = scipy.linalg.cho_solve((L, True), b)`.
 *
 * @tparam T  Scalar type.
 * @param n  Dimension (`L` is `n×n`, `b` has length `n`).
 * @param L  Lower Cholesky factor (column-major, `n*n`; read-only).
 * @param b  In/out right-hand side; on return holds the solution `x`.
 */
template <typename T>
__device__ void potrs(uint32_t n, const T *L, T *b)
{
    trsv<T, true, false, false>(n, L, b);  // forward: L y = b
    trsv<T, true, false, true>(n, L, b);   // back:    Lᵀ x = y
}

/**
 * @brief Compile-time-size SPD solve from a precomputed Cholesky factor (LAPACK potrs).
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension.
 * @param L  Lower Cholesky factor (column-major, `N*N`; read-only).
 * @param b  In/out right-hand side; on return holds the solution `x`.
 */
template <typename T, uint32_t N>
__device__ void potrs(const T *L, T *b) { potrs<T>(N, L, b); }

// ─── multi-RHS overloads (column-major B, factor once / solve per column) ─────

/**
 * @brief Solve the SPD system `A X = B` with multiple right-hand sides (LAPACK posv).
 *
 * Factors `A = L Lᵀ` in place **once** via Cholesky, then solves each of the
 * `nrhs` columns of `B` by a forward (`L y = b`) and back (`Lᵀ x = y`)
 * substitution. On return `A` holds its lower Cholesky factor `L` and `B` holds
 * the solution `X`. `A` must be symmetric positive-definite; behaviour on non-SPD
 * input is undefined (the Cholesky step produces NaN, no info flag).
 *
 * `B` (and `X`) is `n × nrhs` stored **column-major**: column `c` begins at
 * `B + c*n` and occupies `n` contiguous elements. The Cholesky factor completes
 * before the first column's solve (its trailing `__syncthreads()`), and each
 * `trsv` self-syncs, so no extra barrier is needed between columns. Thread-count
 * invariant. NumPy equivalent: `X = np.linalg.solve(A, B)` (A SPD, B `n×nrhs`).
 *
 * @tparam T     Scalar type (e.g. `float`, `double`).
 * @param n      Dimension (`A` is `n×n`, each column of `B` has length `n`).
 * @param nrhs   Number of right-hand sides (columns of `B`).
 * @param A      In/out SPD matrix (column-major); overwritten with its factor `L`.
 * @param B      In/out right-hand sides (`n×nrhs`, column-major); on return holds `X`.
 */
template <typename T>
__device__ void posv(uint32_t n, uint32_t nrhs, T *A, T *B)
{
    cholDecomp_InPlace<T>(n, A);                  // A -> L (lower); trailing __syncthreads
    for (uint32_t c = 0; c < nrhs; c++) {
        T *Bc = B + c * n;                        // column c (column-major)
        trsv<T, true, false, false>(n, A, Bc);    // forward: L y = b
        trsv<T, true, false, true>(n, A, Bc);     // back:    Lᵀ x = y
    }
}

/**
 * @brief Compile-time-size multi-RHS SPD solve `A X = B` (LAPACK posv).
 *
 * Same as the runtime multi-RHS `posv` with the dimension and right-hand-side
 * count as template parameters. `B` is `N × NRHS` column-major (column `c` at
 * `B + c*N`). Factored once, solved per column. NumPy equivalent:
 * `X = np.linalg.solve(A, B)` (A SPD).
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension (`A` is `N×N`, each column of `B` has length `N`).
 * @tparam NRHS  Number of right-hand sides (columns of `B`).
 * @param A  In/out SPD matrix (column-major); overwritten with its factor `L`.
 * @param B  In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
 */
template <typename T, uint32_t N, uint32_t NRHS>
__device__ void posv(T *A, T *B) { posv<T>(N, NRHS, A, B); }

/**
 * @brief Multi-RHS SPD solve `A X = B` from a precomputed Cholesky factor (LAPACK potrs).
 *
 * Given the lower factor `L` (e.g. from `cholDecomp_InPlace`), solves
 * `L Lᵀ X = B` for each of the `nrhs` columns by forward then back substitution
 * — the reusable-factor / multi-solve path (no re-factor). `L` is read-only; `B`
 * is overwritten with `X`.
 *
 * `B` (and `X`) is `n × nrhs` stored **column-major**: column `c` begins at
 * `B + c*n`. Each `trsv` self-syncs, so no barrier between columns is needed.
 * Thread-count invariant. SciPy equivalent:
 * `X = scipy.linalg.cho_solve((L, True), B)`.
 *
 * @tparam T     Scalar type.
 * @param n      Dimension (`L` is `n×n`, each column of `B` has length `n`).
 * @param nrhs   Number of right-hand sides (columns of `B`).
 * @param L      Lower Cholesky factor (column-major, `n*n`; read-only).
 * @param B      In/out right-hand sides (`n×nrhs`, column-major); on return holds `X`.
 */
template <typename T>
__device__ void potrs(uint32_t n, uint32_t nrhs, const T *L, T *B)
{
    for (uint32_t c = 0; c < nrhs; c++) {
        T *Bc = B + c * n;                        // column c (column-major)
        trsv<T, true, false, false>(n, L, Bc);    // forward: L y = b
        trsv<T, true, false, true>(n, L, Bc);     // back:    Lᵀ x = y
    }
}

/**
 * @brief Compile-time-size multi-RHS SPD solve from a precomputed Cholesky factor (LAPACK potrs).
 *
 * `B` is `N × NRHS` column-major (column `c` at `B + c*N`). Solved per column,
 * no re-factor. SciPy equivalent: `X = scipy.linalg.cho_solve((L, True), B)`.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension.
 * @tparam NRHS  Number of right-hand sides (columns of `B`).
 * @param L  Lower Cholesky factor (column-major, `N*N`; read-only).
 * @param B  In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
 */
template <typename T, uint32_t N, uint32_t NRHS>
__device__ void potrs(const T *L, T *B) { potrs<T>(N, NRHS, L, B); }
