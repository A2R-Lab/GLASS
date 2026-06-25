#pragma once
#include <cstdint>
// glass.cuh includes L1/iamax.cuh (glass::low_memory::iamax) before this header,
// so the pivot path below calls it unqualified — same intra-namespace dependency
// convention as posv.cuh → cholDecomp_InPlace / trsv (no local #include).

/**
 * @brief In-place LDLᵀ factorization of a symmetric (possibly INDEFINITE) matrix
 *        (LAPACK `sytrf` analogue, lower, optional symmetric 1×1 pivoting).
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
 * SciPy / NumPy equivalence: when `pivot==false`, `lu, d, perm =
 * scipy.linalg.ldl(A, lower=True)` returns the same `L` (here `lu`, with `perm`
 * the identity) and `D = np.diag(d)`; equivalently `A == L @ D @ L.T`. When
 * `pivot==true` the recorded permutation `P` (built from `piv`, see below)
 * satisfies `P @ A @ P.T == L @ D @ L.T`.
 *
 * @par Pivoting (`pivot==true`)
 * Symmetric **1×1 diagonal pivoting**: at each step `k`, among the working
 * (Schur-complement) diagonals `Dᵉᶠᶠ_i = A_ii - Σ_{m<k} L_im² D_m` for
 * `i = k..n-1`, the index `p` of largest magnitude is selected and rows/cols `k`
 * and `p` are **symmetrically swapped** in the lower-stored factor; `piv[k] = p`
 * records the swap. This moves the largest available diagonal onto the pivot
 * position, avoiding the zero/small-pivot breakdown of the non-pivoted path for
 * the common indefinite case. The permutation is applied to the right-hand side
 * by `ldlt_solve` when `piv != nullptr`. Operationally, `P b` is the forward
 * sweep `for k=0..n-1: swap(b[k], b[piv[k]])` and `Pᵀ x` the reverse sweep — the
 * two permutation passes `ldlt_solve` wraps around the triangular solves.
 *
 * @par Limitations
 * - `pivot==false` is **non-pivoted**: it requires every pivot `D_j` to be
 *   nonzero. A symmetric matrix can be nonsingular yet still produce a zero pivot
 *   here (e.g. a saddle `[[0, b],[b, 0]]`).
 * - `pivot==true` does **symmetric 1×1 diagonal pivoting** (robust for indefinite
 *   `A` with a nonzero remaining diagonal). **Full Bunch–Kaufman 2×2 pivoting is
 *   NOT implemented**, so a structurally-zero diagonal *block* — e.g.
 *   `[[0,1],[1,0]]`, whose entire remaining diagonal is zero at step 0 — still
 *   cannot be factored (it requires a 2×2 pivot). That case remains a documented
 *   known limitation.
 * - Thread-count invariant: identical output for any block size (1, a partial
 *   warp, or many warps); the chosen pivot index is broadcast via shared memory +
 *   `__syncthreads` (no racy re-read).
 * - Prefer `double` for ill-conditioned / KKT systems — small pivots amplify
 *   round-off badly; pivoting mitigates but does not eliminate this.
 *
 * When `CHECK` is true the factorization additionally reports, via two optional
 * (null-skippable) outputs written by rank 0:
 *   - `s_fail` — set to 1 if any pivot `D_j` is exactly zero or NaN (a singular /
 *     non-factorable breakdown of the non-pivoted recurrence), else 0;
 *   - `s_inertia` — three counts `{n_pos, n_neg, n_zero}` of the pivot signs (the
 *     matrix **inertia**: e.g. a well-posed KKT system has a known +/- split).
 * `CHECK` defaults false and the whole reporting path compiles out (`if
 * constexpr`), so the unchecked instantiation is byte-identical to the original.
 *
 * @tparam T  Scalar type (use `double` for ill-conditioned KKT systems).
 * @tparam CHECK  If true, report zero/NaN pivots and the inertia (default false, compiles out).
 * @param n       Matrix dimension (A is n x n).
 * @param A       In/out n x n matrix (column-major); on return its diagonal holds
 *                `D` and its strict lower triangle holds `L`.
 * @param s_temp  Shared scratch advertised as `(n + 1)` elements, used by the
 *                pivot path: slot [0] broadcasts the chosen pivot index and slots
 *                [1..n] hold the working-diagonal magnitudes fed to the no-scratch
 *                `glass::low_memory::iamax` argmax (so the scratch stays within
 *                `(n+1)` for any block size). The non-pivoted path does not use it
 *                and accepts `nullptr`.
 * @param pivot   If true, apply symmetric 1×1 diagonal pivoting (see above).
 * @param piv     Out pivot array of `n` entries (pivot path only); `piv[k]` is the
 *                index swapped into position `k`. May be `nullptr` when `!pivot`.
 * @param s_fail     Optional flag (CHECK only): 1 on a zero/NaN pivot, else 0. Ignored when null.
 * @param s_inertia  Optional 3 ints (CHECK only): `{n_pos, n_neg, n_zero}` pivot-sign counts. Ignored when null.
 */
template <typename T, bool CHECK = false>
__device__ void ldlt(uint32_t n, T *A, T *s_temp, bool pivot = false, uint32_t *piv = nullptr,
                     int *s_fail = nullptr, int *s_inertia = nullptr)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    if constexpr (CHECK) {       // only rank 0 writes the reporting outputs
        if (rank == 0) {
            if (s_fail) *s_fail = 0;
            if (s_inertia) { s_inertia[0] = 0; s_inertia[1] = 0; s_inertia[2] = 0; }
        }
    }
    // s_temp layout (pivot path only): [0] holds the broadcast pivot index (read
    // as uint32_t); [1 .. n] hold the n-j working-diagonal magnitudes argmax'd by
    // the no-scratch glass::low_memory::iamax (thread-0 serial scan — keeps the
    // scratch within the advertised (n+1) elements regardless of block size).
    if (!pivot) { (void)s_temp; (void)piv; }
    for (uint32_t j = 0; j < n; j++) {
        if (pivot) {
            // --- symmetric 1x1 pivot selection over the remaining diagonal ---
            // Working diagonal of index i (i>=j): D_eff_i = A_ii - sum_{m<j} L_im^2 D_m.
            // Each thread fills s_diag[i-j] for its strided i; barrier; argmax;
            // broadcast the winner via s_idx (shared) so every thread agrees.
            T *s_diag = s_temp + 1;          // length (n - j) working-diag scratch
            for (uint32_t i = j + rank; i < n; i += size) {
                T d = A[i*n + i];
                for (uint32_t m = 0; m < j; m++) {
                    T Lim = A[m*n + i];
                    d -= Lim * Lim * A[m*n + m];
                }
                s_diag[i - j] = d;
            }
            __syncthreads();                 // working diagonals visible to argmax
            // No-scratch argmax (thread 0 serial scan) over the working diagonals;
            // writes the winning index into s_idx[0] and ends on __syncthreads(),
            // so the pivot index is block-visible without a racy re-read.
            uint32_t *s_idx = reinterpret_cast<uint32_t *>(s_temp);
            low_memory::iamax<T>(n - j, s_diag, s_idx);
            uint32_t p = j + s_idx[0];        // absolute pivot row/col
            if (rank == 0) piv[j] = p;
            // --- symmetric swap of rows/cols j and p in the lower factor ---
            // Lower-stored symmetric layout: entry (r,c) with r>=c lives at A[c*n+r].
            // Swapping index j<->p means swapping, for the whole matrix, every
            // pair {(j,t),(p,t)} (row/col t). Strided over t for thread invariance.
            if (p != j) {
                for (uint32_t t = rank; t < n; t += size) {
                    // skip the two pivot rows/cols themselves except their
                    // diagonal handled below; handle each off-diagonal once.
                    if (t == j || t == p) continue;
                    // (j,t) and (p,t): pick lower-stored address by ordering.
                    T *a_jt = (j >= t) ? &A[t*n + j] : &A[j*n + t];
                    T *a_pt = (p >= t) ? &A[t*n + p] : &A[p*n + t];
                    T tmp = *a_jt; *a_jt = *a_pt; *a_pt = tmp;
                }
                __syncthreads();             // off-diagonal swaps done
                if (rank == 0) {
                    // diagonal entries j<->p, and the cross entry (p,j) maps to
                    // itself under the symmetric swap (stays in place).
                    T tmp = A[j*n + j]; A[j*n + j] = A[p*n + p]; A[p*n + p] = tmp;
                }
                __syncthreads();             // diagonal swap visible before elim
            }
        }
        // Serial diagonal pivot: D_j = A_jj - sum_{k<j} L_jk^2 * D_k.
        if (rank == 0) {
            T sum = static_cast<T>(0);
            for (uint32_t k = 0; k < j; k++) {
                T Ljk = A[k*n + j];          // L_jk (strict-lower, row j, col k)
                sum += Ljk * Ljk * A[k*n + k];  // * D_k (diagonal slot)
            }
            A[j*n + j] -= sum;               // overwrite diagonal with D_j
            if constexpr (CHECK) {
                T Dj_ = A[j*n + j];
                if (s_fail && (Dj_ == static_cast<T>(0) || isnan(Dj_))) *s_fail = 1;
                if (s_inertia) {
                    if (Dj_ > static_cast<T>(0)) s_inertia[0]++;
                    else if (Dj_ < static_cast<T>(0)) s_inertia[1]++;
                    else s_inertia[2]++;
                }
            }
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
 * When `CHECK` is true, reports zero/NaN pivots via `s_fail` and the inertia via
 * `s_inertia` (see the runtime overload). `CHECK` defaults false and compiles
 * out, so the unchecked instantiation is byte-identical to the original.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N).
 * @tparam CHECK  If true, report zero/NaN pivots and the inertia (default false, compiles out).
 * @param A       In/out N x N matrix (column-major); diagonal holds `D`, strict
 *                lower holds `L` on return.
 * @param s_temp  Shared scratch advertised as `(N + 1)` elements (used by the
 *                pivot path; non-pivoted path accepts `nullptr`).
 * @param pivot   If true, apply symmetric 1×1 diagonal pivoting (no 2×2 path).
 * @param piv     Out pivot array of `N` entries (pivot path only); may be `nullptr`.
 * @param s_fail     Optional flag (CHECK only): 1 on a zero/NaN pivot, else 0. Ignored when null.
 * @param s_inertia  Optional 3 ints (CHECK only): `{n_pos, n_neg, n_zero}`. Ignored when null.
 */
template <typename T, uint32_t N, bool CHECK = false>
__device__ void ldlt(T *A, T *s_temp, bool pivot = false, uint32_t *piv = nullptr,
                     int *s_fail = nullptr, int *s_inertia = nullptr)
{
    ldlt<T, CHECK>(N, A, s_temp, pivot, piv, s_fail, s_inertia);
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
 * @par Pivoting
 * When `piv != nullptr` (factor produced with `pivot=true`), the factor satisfies
 * `P A Pᵀ = L D Lᵀ`, so `A x = b` is solved as `x = Pᵀ (L D Lᵀ)⁻¹ P b`: the
 * permutation `P` is applied to `b` BEFORE the forward solve and `Pᵀ` to the
 * result AFTER the back solve. `P` is the ordered product of the recorded
 * transpositions `swap(k, piv[k])` for `k = 0..n-1`; applying it to a vector is
 * the same forward sweep of swaps (and `Pᵀ` is the reverse sweep). The swap pass
 * is done serially on rank 0 (n is small and the data is the length-n RHS), with
 * a `__syncthreads()` so the permuted vector is block-visible before the solve.
 *
 * @par Limitations
 * Pass `piv = nullptr` (the default) for a non-pivoted factor. Only symmetric 1×1
 * pivoting is supported (matching `ldlt(pivot=true)`); there is no 2×2 path.
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
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    // P b: forward sweep of the recorded transpositions swap(k, piv[k]).
    if (piv != nullptr) {
        if (rank == 0)
            for (uint32_t k = 0; k < n; k++) {
                uint32_t p = piv[k];
                T tmp = b[k]; b[k] = b[p]; b[p] = tmp;
            }
        __syncthreads();
    }
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
    // Pᵀ x: reverse sweep of the recorded transpositions (undoes P).
    if (piv != nullptr) {
        if (rank == 0)
            for (int32_t k = (int32_t)n - 1; k >= 0; k--) {
                uint32_t p = piv[k];
                T tmp = b[k]; b[k] = b[p]; b[p] = tmp;
            }
        __syncthreads();
    }
}

/**
 * @brief Compile-time-size LDLᵀ solve `A x = b` in place (LAPACK `sytrs` analogue).
 *
 * Same as the runtime `ldlt_solve` but with the dimension as a template
 * parameter. NumPy equivalence: `x = np.linalg.solve(A, b)`. A non-null `piv`
 * applies the symmetric 1×1 permutation (factor made with `pivot=true`).
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
