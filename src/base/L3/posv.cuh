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
