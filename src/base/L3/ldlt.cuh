#pragma once
#include <cstdint>

/**
 * @brief In-place LDLᵀ factorization of a symmetric (possibly INDEFINITE) matrix
 *        (LAPACK `sytrf` analogue, lower, non-pivoted).
 *
 * Factors `A = L * D * Lᵀ` where `L` is unit lower-triangular and `D` is
 * diagonal, overwriting `A` in place (column-major, lower triangle). On return:
 *   - the diagonal slots `A[j*n + j]` hold the pivots `D_j`,
 *   - the strict lower triangle `A[j*n + i]` (`i > j`) holds `L_ij`,
 *   - the implicit unit diagonal of `L` is NOT stored,
 *   - the upper triangle keeps its input values (untouched).
 *
 * Unlike Cholesky (`cholDecomp_InPlace`), there is **no square root**: `D_j` may
 * be negative or zero, which is exactly what lets LDLᵀ factor an indefinite
 * symmetric matrix (e.g. a KKT / saddle-point system) that has no Cholesky
 * factor. The recurrence is, for column `j`:
 * @f[ D_j = A_{jj} - \sum_{k<j} L_{jk}^2 \, D_k @f]
 * then in parallel over rows `i > j`:
 * @f[ L_{ij} = \frac{1}{D_j}\Big(A_{ij} - \sum_{k<j} L_{ik}\,D_k\,L_{jk}\Big). @f]
 *
 * Single-block, column-major, in place. The diagonal recurrence is serial
 * (pivot-to-pivot dependency, computed by rank 0); each column's sub-diagonal
 * update is parallelized across the block with the `i += size` stride. Two
 * `__syncthreads()` per column: after the diagonal write (so every thread reads
 * the finished `D_j`) and after the trailing-column update (before the next
 * column starts).
 *
 * SciPy / NumPy equivalence: `lu, d, perm = scipy.linalg.ldl(A, lower=True)`
 * returns the same `L` (here `lu`, with `perm` the identity since this variant
 * is non-pivoted) and `D = np.diag(d)`; equivalently `A == L @ D @ L.T`.
 *
 * @par Limitations
 * - **Non-pivoted.** Requires every pivot `D_j` to be nonzero. A symmetric
 *   matrix can be nonsingular yet still produce a zero pivot here (e.g. a
 *   saddle `[[0, b],[b, 0]]`): such a matrix needs the pivoted (Bunch–Kaufman)
 *   variant. The signature already reserves `bool pivot` + `uint32_t* piv` so
 *   Bunch–Kaufman can slot in later WITHOUT a signature change; `pivot=true` is
 *   not yet implemented.
 * - Thread-count invariant: identical output for any block size (1, a partial
 *   warp, or many warps).
 * - Prefer `double` for ill-conditioned / KKT systems — small pivots amplify
 *   round-off badly without pivoting.
 *
 * @tparam T  Scalar type (use `double` for ill-conditioned KKT systems).
 * @param n       Matrix dimension (A is n x n).
 * @param A       In/out n x n matrix (column-major); on return its diagonal holds
 *                `D` and its strict lower triangle holds `L`.
 * @param s_temp  Shared scratch advertised as `(n + 1)` elements, RESERVED for
 *                the pivot path. The non-pivoted path does not use it and accepts
 *                `nullptr`.
 * @param pivot   If true, request Bunch–Kaufman pivoting (NOT yet implemented).
 * @param piv     Out pivot array of `n` entries (pivot path only); may be `nullptr`.
 */
template <typename T>
__device__ void ldlt(uint32_t n, T *A, T *s_temp, bool pivot = false, uint32_t *piv = nullptr)
{
    (void)s_temp; (void)pivot; (void)piv;  // reserved for the Bunch-Kaufman pivot path
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t j = 0; j < n; j++) {
        // Serial diagonal pivot: D_j = A_jj - sum_{k<j} L_jk^2 * D_k.
        if (rank == 0) {
            T sum = static_cast<T>(0);
            for (uint32_t k = 0; k < j; k++) {
                T Ljk = A[k*n + j];          // L_jk (strict-lower, row j, col k)
                sum += Ljk * Ljk * A[k*n + k];  // * D_k (diagonal slot)
            }
            A[j*n + j] -= sum;               // overwrite diagonal with D_j
        }
        __syncthreads();                     // all threads read finished D_j
        T Dj = A[j*n + j];
        // Parallel trailing column: L_ij = (A_ij - sum_{k<j} L_ik * D_k * L_jk) / D_j.
        for (uint32_t i = j + 1 + rank; i < n; i += size) {
            T sum = static_cast<T>(0);
            for (uint32_t k = 0; k < j; k++)
                sum += A[k*n + i] * A[k*n + k] * A[k*n + j];  // L_ik * D_k * L_jk
            A[j*n + i] = (A[j*n + i] - sum) / Dj;
        }
        __syncthreads();                     // trailing column done before next col
    }
}

/**
 * @brief Compile-time-size in-place LDLᵀ factorization (LAPACK `sytrf`, lower, non-pivoted).
 *
 * Same as the runtime `ldlt` but with the dimension as a template parameter,
 * letting the compiler bake `N` in. Factors a symmetric (possibly indefinite)
 * `A = L * D * Lᵀ` in place. SciPy equivalence: `lu, d, _ = scipy.linalg.ldl(A)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N).
 * @param A       In/out N x N matrix (column-major); diagonal holds `D`, strict
 *                lower holds `L` on return.
 * @param s_temp  Shared scratch advertised as `(N + 1)` elements (reserved for the
 *                pivot path; non-pivoted path accepts `nullptr`).
 * @param pivot   If true, request Bunch–Kaufman pivoting (NOT yet implemented).
 * @param piv     Out pivot array of `N` entries (pivot path only); may be `nullptr`.
 */
template <typename T, uint32_t N>
__device__ void ldlt(T *A, T *s_temp, bool pivot = false, uint32_t *piv = nullptr)
{
    ldlt<T>(N, A, s_temp, pivot, piv);
}

/**
 * @brief Solve `A x = b` from an LDLᵀ factorization in place (LAPACK `sytrs` analogue).
 *
 * Given the in-place factor produced by `ldlt` (unit lower `L` in the strict
 * lower triangle, pivots `D` on the diagonal, column-major), solves `A x = b`
 * by three sweeps, overwriting `b` with the solution `x`:
 *   1. forward unit-lower solve `L y = b` (no divide — unit diagonal),
 *   2. diagonal scale `z = y / D` (one parallel pass + barrier),
 *   3. back unit-lower-transpose solve `Lᵀ x = z` (no divide).
 *
 * Single-block, in place. NumPy equivalence: `x = np.linalg.solve(A, b)`
 * (i.e. `x == scipy.linalg.solve(A, b, assume_a='sym')`).
 *
 * @par Limitations
 * Non-pivoted: pass `piv = nullptr` (the default). When the Bunch–Kaufman pivot
 * path lands, a non-null `piv` will apply the symmetric row/column permutation;
 * the signature is frozen so callers need not change.
 *
 * @tparam T  Scalar type.
 * @param n   Dimension (LD is n x n, b has length n).
 * @param LD  In LDLᵀ factor from `ldlt` (column-major; unit-L strict-lower, D diagonal).
 * @param b   In/out right-hand side; on return holds the solution x.
 * @param piv Pivot array from the pivoted factorization, or `nullptr` (non-pivoted).
 */
template <typename T>
__device__ void ldlt_solve(uint32_t n, const T *LD, T *b, const uint32_t *piv = nullptr)
{
    (void)piv;  // reserved for the Bunch-Kaufman pivot path
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    // 1) forward: L y = b, L unit lower => no divide. Eliminate y[col] from rows below.
    for (uint32_t col = 0; col < n; col++) {
        T factor = b[col];
        for (uint32_t row = col + 1 + rank; row < n; row += size)
            b[row] -= LD[col*n + row] * factor;   // L_{row,col}
        __syncthreads();
    }
    // 2) diagonal scale: z = y / D (parallel, independent rows).
    for (uint32_t i = rank; i < n; i += size)
        b[i] /= LD[i*n + i];                       // D_i
    __syncthreads();
    // 3) back: Lᵀ x = z, Lᵀ unit upper => no divide. Eliminate x[col] from rows above.
    for (int32_t col = (int32_t)n - 1; col >= 0; col--) {
        T factor = b[col];
        // (Lᵀ)_{i,col} = L_{col,i} for i < col
        for (uint32_t i = rank; i < (uint32_t)col; i += size)
            b[i] -= LD[i*n + col] * factor;
        __syncthreads();
    }
}

/**
 * @brief Compile-time-size LDLᵀ solve `A x = b` in place (LAPACK `sytrs` analogue).
 *
 * Same as the runtime `ldlt_solve` but with the dimension as a template
 * parameter. NumPy equivalence: `x = np.linalg.solve(A, b)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension (LD is N x N, b has length N).
 * @param LD  In LDLᵀ factor from `ldlt` (column-major; unit-L strict-lower, D diagonal).
 * @param b   In/out right-hand side; on return holds the solution x.
 * @param piv Pivot array from the pivoted factorization, or `nullptr` (non-pivoted).
 */
template <typename T, uint32_t N>
__device__ void ldlt_solve(const T *LD, T *b, const uint32_t *piv = nullptr)
{
    ldlt_solve<T>(N, LD, b, piv);
}
