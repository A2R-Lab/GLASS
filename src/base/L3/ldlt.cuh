#pragma once
#include <cstdint>
// glass.cuh includes L1/iamax.cuh (glass::iamax_lowmem) before this header,
// so the pivot path below calls it unqualified — same intra-namespace dependency
// convention as posv.cuh → potrf / trsv (no local #include).

/**
 * @brief Scratch size in bytes for `ldlt`.
 *
 * The pivot path uses `n + 1` scratch elements (one broadcast slot for the
 * argmax index + up to `n` working-row magnitudes fed to the Bunch–Kaufman
 * rowmax scan); the non-pivoted path does not read it. Allocate
 * `ldlt_scratch_bytes<T>(n)` bytes for the `s_scratch` argument so it is sized
 * for both paths.
 *
 * @tparam T  Scalar type.
 * @param n  Matrix dimension (A is n x n).
 * @return Bytes to allocate for `ldlt`'s `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t ldlt_scratch_bytes(uint32_t n)
{
    return static_cast<std::size_t>(n + 1) * sizeof(T);
}

/**
 * @brief In-place LDLᵀ factorization of a symmetric (possibly INDEFINITE) matrix
 *        (LAPACK `sytrf` analogue, lower, optional Bunch–Kaufman pivoting).
 *
 * Factors `A = L * D * Lᵀ` where `L` is unit lower-triangular and `D` is
 * diagonal (non-pivoted) or block-diagonal with 1×1 and 2×2 blocks (pivoted),
 * overwriting `A` in place (column-major, lower triangle). On return:
 *   - the diagonal slots `A[j*n + j]` hold the pivots `D_j` (for a 2×2 pivot
 *     block at columns `(k, k+1)`, the subdiagonal slot `A[k*n + k + 1]`
 *     additionally holds the block's off-diagonal `D_{k+1,k}` — it is part of
 *     `D`, NOT an `L` entry; the corresponding `L_{k+1,k}` is implicitly 0),
 *   - the strict lower triangle `A[j*n + i]` (`i > j`) holds `L_ij`,
 *   - the implicit unit diagonal of `L` is NOT stored,
 *   - the upper triangle keeps its input values (untouched).
 *
 * Unlike Cholesky (`potrf`), there is **no square root**: `D_j` may
 * be negative or zero, which is exactly what lets LDLᵀ factor an indefinite
 * symmetric matrix (e.g. a KKT / saddle-point system) that has no Cholesky
 * factor. The non-pivoted recurrence is, for column `j`:
 * @f[ D_j = A_{jj} - \sum_{k<j} L_{jk}^2 \, D_k @f]
 * then in parallel over rows `i > j`:
 * @f[ L_{ij} = \frac{1}{D_j}\Big(A_{ij} - \sum_{k<j} L_{ik}\,D_k\,L_{jk}\Big). @f]
 *
 * Single-block, column-major, in place. Non-pivoted path: the diagonal
 * recurrence is serial (pivot-to-pivot dependency, computed by rank 0); each
 * column's sub-diagonal update is parallelized across the block with the
 * `i += size` stride. Pivoted path: right-looking (the trailing Schur
 * complement is updated in place after each pivot, row-parallel in two barrier-
 * separated passes), so pivot selection reads the current column directly.
 *
 * SciPy / NumPy equivalence: when `pivot==false`, `lu, d, perm =
 * scipy.linalg.ldl(A, lower=True)` returns the same `L` (here `lu`, with `perm`
 * the identity) and `D = np.diag(d)`; equivalently `A == L @ D @ L.T`. When
 * `pivot==true` the recorded permutation `P` (built from `piv`, see below)
 * satisfies `P @ A @ P.T == L @ D @ L.T` with block-diagonal `D` (scipy's `ldl`
 * is the same LAPACK Bunch–Kaufman algorithm; its returned permutation may
 * differ in representation but factors the same matrix).
 *
 * @par Pivoting (`pivot==true`) — Bunch–Kaufman partial pivoting
 * The LAPACK `sytf2` strategy (lower). At step `k`, with `absakk = |A_kk|` and
 * `colmax = |A_{imax,k}|` the largest sub-diagonal magnitude of the working
 * column and `alpha = (1+sqrt(17))/8`:
 *   1. `absakk >= alpha*colmax` → 1×1 pivot at `k` (no interchange);
 *   2. else, with `rowmax = max_{i != imax} |A_{imax,i}|` over the working row
 *      `imax`: `absakk >= alpha*colmax*(colmax/rowmax)` → 1×1 pivot at `k`;
 *   3. else `|A_{imax,imax}| >= alpha*rowmax` → 1×1 pivot, interchange
 *      `k <-> imax`;
 *   4. else → **2×2 pivot** spanning `(k, k+1)`, interchange `k+1 <-> imax`.
 * Interchanges are applied eagerly and symmetrically to the WHOLE row/col pair
 * in the lower-stored factor (including already-computed `L` columns), so the
 * clean invariant `P A Pᵀ = L D Lᵀ` holds with `P` the ordered product of the
 * recorded swaps. This bounds element growth and — unlike 1×1-only diagonal
 * pivoting — factors matrices whose remaining diagonal is entirely zero (e.g.
 * `[[0,1],[1,0]]`, handled by a 2×2 pivot).
 *
 * `piv` encoding (0-based analogue of LAPACK `ipiv`):
 *   - `piv[k] >= 0`  → 1×1 pivot; rows/cols `k` and `piv[k]` were interchanged
 *     (`piv[k] == k` means no interchange).
 *   - `piv[k] < 0`   → 2×2 pivot spanning `(k, k+1)`; rows/cols `k+1` and
 *     `-piv[k] - 1` were interchanged, and `piv[k+1] == piv[k]`.
 * The permutation is applied to the right-hand side by `ldlt_solve` when
 * `piv != nullptr`: `P b` is the forward sweep of the recorded interchanges and
 * `Pᵀ x` the reverse sweep.
 *
 * @par Limitations
 * - `pivot==false` is **non-pivoted**: it requires every pivot `D_j` to be
 *   nonzero. A symmetric matrix can be nonsingular yet still produce a zero pivot
 *   here (e.g. a saddle `[[0, b],[b, 0]]`) — use `pivot==true` for those.
 * - A singular input breaks down: if a whole working column is zero at step `k`
 *   the pivoted path records an identity 1×1 pivot, performs no elimination for
 *   that column (its `D_k` is 0), and — under `CHECK` — sets `s_fail`.
 * - Thread-count invariant: identical output for any block size (1, a partial
 *   warp, or many warps); argmax indices are broadcast via shared memory +
 *   `__syncthreads` (no racy re-read), and trailing updates write each entry
 *   from exactly one thread.
 * - Prefer `double` for ill-conditioned / KKT systems — small pivots amplify
 *   round-off badly; pivoting mitigates but does not eliminate this.
 *
 * When `CHECK` is true the factorization additionally reports, via two optional
 * (null-skippable) outputs written by rank 0:
 *   - `s_fail` — set to 1 on a zero/NaN pivot (non-pivoted), or on a zero
 *     working column / degenerate 2×2 block (pivoted) — i.e. a singular or
 *     non-factorable breakdown; else 0;
 *   - `s_inertia` — three counts `{n_pos, n_neg, n_zero}` of the eigenvalue
 *     signs of `D` (the matrix **inertia** by Sylvester's law; a Bunch–Kaufman
 *     2×2 block is indefinite and contributes one positive and one negative).
 *     Undefined on breakdown (`s_fail == 1`).
 * `CHECK` defaults false and the whole reporting path compiles out (`if
 * constexpr`), so the unchecked instantiation is byte-identical to the original.
 *
 * @tparam T  Scalar type (use `double` for ill-conditioned KKT systems).
 * @tparam CHECK  If true, report breakdowns and the inertia (default false, compiles out).
 * @param n       Matrix dimension (A is n x n).
 * @param A       In/out n x n matrix (column-major); on return its diagonal
 *                (+ 2×2 subdiagonal slots) holds `D` and its strict lower
 *                triangle holds `L`.
 * @param s_scratch  Shared scratch advertised as `(n + 1)` elements, used by the
 *                pivot path: slot [0] broadcasts argmax indices and slots
 *                [1..n] hold working magnitudes fed to the no-scratch
 *                `glass::iamax_lowmem` scans (so the scratch stays within
 *                `(n+1)` for any block size). The non-pivoted path does not use
 *                it and accepts `nullptr`.
 * @param pivot   If true, apply Bunch–Kaufman 1×1/2×2 pivoting (see above).
 * @param piv     Out pivot array of `n` int32 entries (pivot path only; encoding
 *                above). May be `nullptr` when `!pivot`.
 * @param s_fail     Optional flag (CHECK only): 1 on a factorization breakdown, else 0. Ignored when null.
 * @param s_inertia  Optional 3 ints (CHECK only): `{n_pos, n_neg, n_zero}` inertia counts. Ignored when null.
 */
// Shared body (runtime + compile-time overloads): SizeT deduced — uint32_t or
// ct_size<N> (constant-folds the trip counts and indexing).
template <typename T, bool CHECK = false, typename SizeT>
__device__ void ldlt_impl(SizeT n, T *A, T *s_scratch, bool pivot, int32_t *piv,
                          int *s_fail, int *s_inertia)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    if constexpr (CHECK) {       // only rank 0 writes the reporting outputs
        if (rank == 0) {
            if (s_fail) *s_fail = 0;
            if (s_inertia) { s_inertia[0] = 0; s_inertia[1] = 0; s_inertia[2] = 0; }
        }
    }
    if (pivot) {
        // ─── Bunch–Kaufman partial pivoting (LAPACK sytf2, lower) ───
        // Right-looking: the trailing Schur complement is updated in place after
        // each pivot, so pivot selection reads the CURRENT column directly.
        // s_scratch layout: [0] broadcasts argmax indices (read as uint32_t);
        // [1 .. n] hold row magnitudes for the rowmax scan. All branch decisions
        // read post-barrier shared/global data, so every thread agrees (uniform
        // control flow — required around the iamax_lowmem internal barriers).
        const T alpha = static_cast<T>(0.6403882032022076);  // (1 + sqrt(17)) / 8
        uint32_t *s_idx = reinterpret_cast<uint32_t *>(s_scratch);
        uint32_t k = 0;
        while (k < n) {
            // -- pivot selection on the current working column k --
            T akk = A[k*n + k];
            T absakk = (akk < static_cast<T>(0)) ? -akk : akk;
            uint32_t imax = k;
            T colmax = static_cast<T>(0);
            if (k + 1 < n) {
                // column k rows k+1..n-1 is contiguous in column-major storage
                iamax_lowmem<T>(n - k - 1, &A[k*n + k + 1], s_idx);
                imax = k + 1 + s_idx[0];
                T cv = A[k*n + imax];
                colmax = (cv < static_cast<T>(0)) ? -cv : cv;
            }
            T mx = (absakk > colmax) ? absakk : colmax;
            if (mx == static_cast<T>(0) || isnan(absakk)) {
                // zero (or NaN) working column: nothing to eliminate; record an
                // identity 1×1 pivot and flag the breakdown (LAPACK info-style).
                if (rank == 0) {
                    piv[k] = static_cast<int32_t>(k);
                    if constexpr (CHECK) {
                        if (s_fail) *s_fail = 1;
                        if (s_inertia) s_inertia[2]++;
                    }
                }
                __syncthreads();          // s_idx safe to rewrite next step
                k += 1;
                continue;
            }
            uint32_t kstep = 1, p = k;
            if (absakk < alpha * colmax) {
                // rowmax = max |w(imax, i)| over i = k..n-1, i != imax; row imax
                // straddles the diagonal in lower storage, so gather magnitudes
                // into scratch (strided, thread-invariant) and argmax those.
                T *s_row = s_scratch + 1;             // (n - k) magnitudes
                for (uint32_t i = k + rank; i < n; i += size) {
                    T v = static_cast<T>(0);
                    if (i < imax)      v = A[i*n + imax];   // entry (imax, i)
                    else if (i > imax) v = A[imax*n + i];   // entry (i, imax)
                    s_row[i - k] = (v < static_cast<T>(0)) ? -v : v;
                }
                __syncthreads();          // row magnitudes visible to argmax
                iamax_lowmem<T>(n - k, s_row, s_idx);
                T rowmax = s_row[s_idx[0]];
                T dv = A[imax*n + imax];
                T absdimax = (dv < static_cast<T>(0)) ? -dv : dv;
                if (absakk >= alpha * colmax * (colmax / rowmax)) {
                    /* kstep = 1, p = k: the diagonal is acceptable after all */
                } else if (absdimax >= alpha * rowmax) {
                    p = imax;             // 1×1 pivot, interchange k <-> imax
                } else {
                    kstep = 2; p = imax;  // 2×2 pivot, interchange k+1 <-> imax
                }
            }
            // -- eager symmetric interchange: rows/cols kk <-> p (kk = k or k+1
            //    receives the pivot row/col; includes the factored L columns, so
            //    P A Pᵀ = L D Lᵀ holds with P = product of recorded swaps) --
            uint32_t kk = k + kstep - 1;
            if (p != kk) {
                for (uint32_t t = rank; t < n; t += size) {
                    if (t == kk || t == p) continue;
                    // (kk,t) and (p,t): pick lower-stored address by ordering.
                    T *a_kt = (kk >= t) ? &A[t*n + kk] : &A[kk*n + t];
                    T *a_pt = (p >= t) ? &A[t*n + p] : &A[p*n + t];
                    T tmp = *a_kt; *a_kt = *a_pt; *a_pt = tmp;
                }
                __syncthreads();          // off-diagonal swaps done
                if (rank == 0) {
                    // diagonal entries kk<->p; the cross entry (p,kk) maps to
                    // itself under the symmetric swap (stays in place).
                    T tmp = A[kk*n + kk]; A[kk*n + kk] = A[p*n + p]; A[p*n + p] = tmp;
                }
                __syncthreads();          // diagonal swap visible before elim
            }
            if (kstep == 1) {
                if (rank == 0) {
                    piv[k] = static_cast<int32_t>(p);
                    if constexpr (CHECK) {
                        T Dk_ = A[k*n + k];
                        if (s_fail && (Dk_ == static_cast<T>(0) || isnan(Dk_))) *s_fail = 1;
                        if (s_inertia) {
                            if (Dk_ > static_cast<T>(0)) s_inertia[0]++;
                            else if (Dk_ < static_cast<T>(0)) s_inertia[1]++;
                            else s_inertia[2]++;
                        }
                    }
                }
                T r1 = static_cast<T>(1) / A[k*n + k];
                // pass 1: trailing rank-1 update from the UNDIVIDED column k
                // (row-parallel: each entry written by exactly one thread, reads
                // only pre-step data — no barrier needed inside the pass).
                for (uint32_t i = k + 1 + rank; i < n; i += size) {
                    T aik = A[k*n + i];
                    for (uint32_t j = k + 1; j <= i; j++)
                        A[j*n + i] -= r1 * aik * A[k*n + j];
                }
                __syncthreads();          // trailing update done
                // pass 2: scale column k into L
                for (uint32_t i = k + 1 + rank; i < n; i += size)
                    A[k*n + i] *= r1;
                __syncthreads();          // column k finalized
            } else {
                // 2×2 pivot block D2 = [[A_kk, d21], [d21, A_{k+1,k+1}]] (post-
                // swap). LAPACK-scaled inverse: dr = det(D2)/d21² stays O(1).
                T d21 = A[k*n + k + 1];
                T d11 = A[(k+1)*n + k + 1] / d21;
                T d22 = A[k*n + k] / d21;
                T dr = d11 * d22 - static_cast<T>(1);
                if (rank == 0) {
                    piv[k]     = -static_cast<int32_t>(p) - 1;
                    piv[k + 1] = -static_cast<int32_t>(p) - 1;
                    if constexpr (CHECK) {
                        if (s_fail && (dr == static_cast<T>(0) || isnan(dr))) *s_fail = 1;
                        if (s_inertia) {
                            // BK guarantees an indefinite 2×2 block (det < 0):
                            // one positive + one negative eigenvalue.
                            if (dr < static_cast<T>(0)) { s_inertia[0]++; s_inertia[1]++; }
                            else if (dr > static_cast<T>(0)) {
                                // not reachable for a BK-chosen block; count by trace
                                if (A[k*n + k] + A[(k+1)*n + k + 1] > static_cast<T>(0))
                                    s_inertia[0] += 2;
                                else s_inertia[1] += 2;
                            } else s_inertia[2] += 2;   // degenerate (s_fail set)
                        }
                    }
                }
                T t2 = static_cast<T>(1) / dr;
                T d21t = t2 / d21;
                // pass 1: trailing rank-2 update from the ORIGINAL block columns
                // (w-coefficients recomputed per column — deterministic, so the
                // update stays thread-count invariant with no extra scratch).
                for (uint32_t i = k + 2 + rank; i < n; i += size) {
                    T aik = A[k*n + i], aik1 = A[(k+1)*n + i];
                    for (uint32_t j = k + 2; j <= i; j++) {
                        T wj  = d21t * (d11 * A[k*n + j] - A[(k+1)*n + j]);
                        T wj1 = d21t * (d22 * A[(k+1)*n + j] - A[k*n + j]);
                        A[j*n + i] -= aik * wj + aik1 * wj1;
                    }
                }
                __syncthreads();          // trailing update done
                // pass 2: overwrite the block columns with the L entries
                // (A[k*n+k+1] keeps d21 — it is part of D, L_{k+1,k} == 0).
                for (uint32_t i = k + 2 + rank; i < n; i += size) {
                    T aik = A[k*n + i], aik1 = A[(k+1)*n + i];
                    A[k*n + i]     = d21t * (d11 * aik - aik1);
                    A[(k+1)*n + i] = d21t * (d22 * aik1 - aik);
                }
                __syncthreads();          // block columns finalized
            }
            k += kstep;
        }
        return;
    }
    (void)s_scratch; (void)piv;
    for (uint32_t j = 0; j < n; j++) {
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

// TRAILING_SYNC is accepted for interface uniformity but is a documented
// NO-OP here: the factorization's tail barrier is fused into its final
// algorithm step (both pivot paths), so nothing separable is elided.
template <typename T, bool CHECK = false, bool TRAILING_SYNC = true>
__device__ void ldlt(uint32_t n, T *A, T *s_scratch, bool pivot = false, int32_t *piv = nullptr,
                     int *s_fail = nullptr, int *s_inertia = nullptr)
{
    ldlt_impl<T, CHECK>(n, A, s_scratch, pivot, piv, s_fail, s_inertia);
}

/**
 * @brief Compile-time-size in-place LDLᵀ factorization (LAPACK `sytrf`, lower,
 *        optional Bunch–Kaufman pivoting).
 *
 * Same as the runtime `ldlt` but with the dimension as a template parameter,
 * letting the compiler bake `N` in. Factors a symmetric (possibly indefinite)
 * `A = L * D * Lᵀ` in place. SciPy equivalence: `lu, d, _ = scipy.linalg.ldl(A)`.
 *
 * When `CHECK` is true, reports breakdowns via `s_fail` and the inertia via
 * `s_inertia` (see the runtime overload). `CHECK` defaults false and compiles
 * out, so the unchecked instantiation is byte-identical to the original.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N).
 * @tparam CHECK  If true, report breakdowns and the inertia (default false, compiles out).
 * @param A       In/out N x N matrix (column-major); diagonal (+ 2×2 subdiagonal
 *                slots) holds `D`, strict lower holds `L` on return.
 * @param s_scratch  Shared scratch advertised as `(N + 1)` elements (used by the
 *                pivot path; non-pivoted path accepts `nullptr`).
 * @param pivot   If true, apply Bunch–Kaufman 1×1/2×2 pivoting (see the runtime overload).
 * @param piv     Out pivot array of `N` int32 entries (pivot path only); may be `nullptr`.
 * @param s_fail     Optional flag (CHECK only): 1 on a factorization breakdown, else 0. Ignored when null.
 * @param s_inertia  Optional 3 ints (CHECK only): `{n_pos, n_neg, n_zero}`. Ignored when null.
 */
template <typename T, uint32_t N, bool CHECK = false, bool TRAILING_SYNC = true>
__device__ void ldlt(T *A, T *s_scratch, bool pivot = false, int32_t *piv = nullptr,
                     int *s_fail = nullptr, int *s_inertia = nullptr)
{
    ldlt_impl<T, CHECK>(ct_size<N>{}, A, s_scratch, pivot, piv, s_fail, s_inertia);
}

/**
 * @brief Solve `A x = b` from an LDLᵀ factorization in place (LAPACK `sytrs` analogue).
 *
 * Given the in-place factor produced by `ldlt` (unit lower `L` in the strict
 * lower triangle, pivots `D` on the diagonal, column-major), solves `A x = b`
 * by three sweeps, overwriting `b` with the solution `x`:
 *   1. forward unit-lower solve `L y = b` (no divide — unit diagonal),
 *   2. diagonal solve `z = D⁻¹ y` (parallel scale for a non-pivoted factor;
 *      a sequential 1×1/2×2 block walk for a pivoted one),
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
 * interchanges (see the `piv` encoding on `ldlt`): ascending `k`, a
 * non-negative `piv[k]` is `swap(k, piv[k])`, a negative one is the 2×2 block
 * `(k, k+1)` with `swap(k+1, -piv[k]-1)` (skip `k+1`); `Pᵀ` is the reverse
 * sweep. The triangular sweeps are 2×2-aware: a block's two columns eliminate
 * together and its `LD[k*n + k + 1]` slot is the block off-diagonal `D_{k+1,k}`
 * (part of `D`, not an `L` entry), consumed only by the 2×2 diagonal solve.
 * The permutation and block-diagonal passes run serially on rank 0 (n is small
 * and the data is the length-n RHS), barrier-separated for block visibility.
 *
 * @tparam T  Scalar type.
 * @param n   Dimension (LD is n x n, b has length n).
 * @param LD  In LDLᵀ factor from `ldlt` (column-major; unit-L strict-lower, D diagonal).
 * @param b   In/out right-hand side; on return holds the solution x.
 * @param piv Pivot array from the pivoted factorization, or `nullptr` (non-pivoted).
 */
// Shared body (runtime + compile-time overloads): SizeT deduced — uint32_t or
// ct_size<N> (constant-folds the trip counts and indexing).
template <typename T, typename SizeT>
__device__ void ldlt_solve_impl(SizeT n, const T *LD, T *b, const int32_t *piv)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    // P b: forward sweep of the recorded interchanges.
    if (piv != nullptr) {
        if (rank == 0)
            for (uint32_t k = 0; k < n; k++) {
                int32_t pk = piv[k];
                if (pk >= 0) {
                    uint32_t p = static_cast<uint32_t>(pk);
                    T tmp = b[k]; b[k] = b[p]; b[p] = tmp;
                } else {
                    // 2×2 block (k, k+1): the interchange was at k+1
                    uint32_t p = static_cast<uint32_t>(-pk) - 1;
                    T tmp = b[k + 1]; b[k + 1] = b[p]; b[p] = tmp;
                    k++;
                }
            }
        __syncthreads();
    }
    // 1) forward: L y = b, L unit lower => no divide. Eliminate y[col] from rows
    //    below. A 2×2 block's columns eliminate together from rows col+2 down —
    //    LD[col*n + col + 1] is the block off-diagonal D21, NOT an L entry.
    for (uint32_t col = 0; col < n; col++) {
        if (piv != nullptr && piv[col] < 0) {
            T f0 = b[col], f1 = b[col + 1];
            for (uint32_t row = col + 2 + rank; row < n; row += size)
                b[row] -= LD[col*n + row] * f0 + LD[(col+1)*n + row] * f1;
            __syncthreads();
            col++;                        // second block column handled above
        } else {
            T factor = b[col];
            for (uint32_t row = col + 1 + rank; row < n; row += size)
                b[row] -= LD[col*n + row] * factor;   // L_{row,col}
            __syncthreads();
        }
    }
    // 2) diagonal solve: z = D⁻¹ y. Non-pivoted D is strictly diagonal
    //    (parallel, independent rows); a pivoted factor may hold 2×2 blocks, so
    //    walk the block structure serially on rank 0 (n is small).
    if (piv == nullptr) {
        for (uint32_t i = rank; i < n; i += size)
            b[i] /= LD[i*n + i];                       // D_i
        __syncthreads();
    } else {
        if (rank == 0)
            for (uint32_t k = 0; k < n; k++) {
                if (piv[k] >= 0) {
                    b[k] /= LD[k*n + k];
                } else {
                    // 2×2 block solve, LAPACK sytrs form (scaled by d21 for
                    // overflow robustness): D2 = [[dk*d21, d21], [d21, dk1*d21]].
                    T d21 = LD[k*n + k + 1];
                    T dk  = LD[k*n + k] / d21;
                    T dk1 = LD[(k+1)*n + k + 1] / d21;
                    T denom = dk * dk1 - static_cast<T>(1);
                    T bk = b[k] / d21, bk1 = b[k + 1] / d21;
                    b[k]     = (dk1 * bk - bk1) / denom;
                    b[k + 1] = (dk * bk1 - bk) / denom;
                    k++;
                }
            }
        __syncthreads();
    }
    // 3) back: Lᵀ x = z, Lᵀ unit upper => no divide. Descending, a negative
    //    piv[col] marks the SECOND column of its 2×2 block (blocks tile the
    //    range in order), so the pair (col-1, col) eliminates together.
    for (int32_t col = (int32_t)n - 1; col >= 0; col--) {
        if (piv != nullptr && piv[col] < 0 && col > 0) {
            uint32_t c0 = (uint32_t)col - 1;
            T f1 = b[col], f0 = b[c0];
            for (uint32_t i = rank; i < c0; i += size)
                b[i] -= LD[i*n + col] * f1 + LD[i*n + c0] * f0;
            __syncthreads();
            col--;                        // first block column handled above
        } else {
            T factor = b[col];
            // (Lᵀ)_{i,col} = L_{col,i} for i < col
            for (uint32_t i = rank; i < (uint32_t)col; i += size)
                b[i] -= LD[i*n + col] * factor;
            __syncthreads();
        }
    }
    // Pᵀ x: reverse sweep of the recorded interchanges (undoes P). Descending,
    // a negative piv[k] is the second half of its block — the position the
    // forward sweep swapped.
    if (piv != nullptr) {
        if (rank == 0)
            for (int32_t k = (int32_t)n - 1; k >= 0; k--) {
                int32_t pk = piv[k];
                if (pk >= 0) {
                    uint32_t p = static_cast<uint32_t>(pk);
                    T tmp = b[k]; b[k] = b[p]; b[p] = tmp;
                } else {
                    uint32_t p = static_cast<uint32_t>(-pk) - 1;
                    T tmp = b[k]; b[k] = b[p]; b[p] = tmp;
                    k--;                  // skip the block's first column
                }
            }
        __syncthreads();
    }
}

// TRAILING_SYNC: accepted for uniformity, documented NO-OP (the last barrier
// is the final substitution/permutation step's own, not a separable tail).
template <typename T, bool TRAILING_SYNC = true>
__device__ void ldlt_solve(uint32_t n, const T *LD, T *b, const int32_t *piv = nullptr)
{
    ldlt_solve_impl<T>(n, LD, b, piv);
}

/**
 * @brief Compile-time-size LDLᵀ solve `A x = b` in place (LAPACK `sytrs` analogue).
 *
 * Same as the runtime `ldlt_solve` but with the dimension as a template
 * parameter. NumPy equivalence: `x = np.linalg.solve(A, b)`. A non-null `piv`
 * applies the recorded Bunch–Kaufman permutation and 2×2 diagonal blocks
 * (factor made with `pivot=true`).
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension (LD is N x N, b has length N).
 * @param LD  In LDLᵀ factor from `ldlt` (column-major; unit-L strict-lower, D diagonal).
 * @param b   In/out right-hand side; on return holds the solution x.
 * @param piv Pivot array from the pivoted factorization, or `nullptr` (non-pivoted).
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void ldlt_solve(const T *LD, T *b, const int32_t *piv = nullptr)
{
    ldlt_solve_impl<T>(ct_size<N>{}, LD, b, piv);
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    /**
     * @brief Single-warp in-place LDLᵀ factorization (LAPACK `sytrf`, lower, NON-pivoted).
     *
     * Warp-per-problem parity with the block `glass::ldlt`: one 32-lane warp factors
     * the symmetric (possibly INDEFINITE) `A = L D Lᵀ` in place — lane 0 runs the
     * serial diagonal recurrence `D_j = A_jj − Σ_{k<j} L_jk² D_k` (broadcasting `D_j`
     * from its register via `__shfl_sync`, never a shared re-read — immune to the
     * `__restrict__` stale-cache miscompile), lanes fill the trailing column
     * `L_ij = (A_ij − Σ_{k<j} L_ik D_k L_jk)/D_j` strided by 32. On return the diagonal
     * slots hold `D`, the strict lower triangle holds unit-`L`. No square root, so it
     * factors KKT / saddle-point systems Cholesky cannot. No shared scratch, no
     * `__syncthreads`. **Non-pivoted** (pivoting on the warp surface is deferred — it
     * needs a `warp::iamax` over the working column; the block path covers the
     * pivoted / Bunch–Kaufman case).
     *
     * `CHECK` (compile-out) reports a zero/NaN pivot via `s_fail` and the inertia
     * `{n_pos, n_neg, n_zero}` via `s_inertia` (lane 0 writes both). NumPy:
     * `lu, d, _ = scipy.linalg.ldl(A, lower=True)` ⇒ `A == lu @ np.diag(d) @ lu.T`.
     *
     * @tparam T      Scalar type (use `double` for ill-conditioned A).
     * @tparam N      Dimension (A is N x N).
     * @tparam CHECK  If true, report zero/NaN pivot + inertia (default false, compiles out).
     * @param A         In/out N x N symmetric matrix (column-major, lower); on return holds L (strict-lower, unit) and D (diagonal).
     * @param s_fail    Optional flag (CHECK only): set to 1 on a zero/NaN pivot, else 0.
     * @param s_inertia Optional length-3 `{n_pos, n_neg, n_zero}` pivot-sign counts (CHECK only).
     */
    template <typename T, uint32_t N, bool CHECK = false>
    __device__ void ldlt(T *A, int *s_fail = nullptr, int *s_inertia = nullptr)
    {
        uint32_t lane = (flat_rank()) & 31;
        if constexpr (CHECK) {
            if (lane == 0) {
                if (s_fail) *s_fail = 0;
                if (s_inertia) { s_inertia[0] = 0; s_inertia[1] = 0; s_inertia[2] = 0; }
            }
        }
        for (uint32_t j = 0; j < N; j++) {
            T Dj = static_cast<T>(0);
            if (lane == 0) {                          // serial diagonal pivot D_j
                T sum = static_cast<T>(0);
                for (uint32_t k = 0; k < j; k++) {
                    T Ljk = A[k*N + j];               // L_jk (strict-lower, row j, col k)
                    sum += Ljk * Ljk * A[k*N + k];    // * D_k
                }
                A[j*N + j] -= sum;                    // overwrite diagonal with D_j
                Dj = A[j*N + j];
                if constexpr (CHECK) {
                    if (s_fail && (Dj == static_cast<T>(0) || isnan(Dj))) *s_fail = 1;
                    if (s_inertia) {
                        if (Dj > static_cast<T>(0)) s_inertia[0]++;
                        else if (Dj < static_cast<T>(0)) s_inertia[1]++;
                        else s_inertia[2]++;
                    }
                }
            }
            Dj = __shfl_sync(0xffffffffu, Dj, 0);     // broadcast finished D_j from lane 0's register
            for (uint32_t i = j + 1 + lane; i < N; i += 32) {   // parallel trailing column
                T sum = static_cast<T>(0);
                for (uint32_t k = 0; k < j; k++)
                    sum += A[k*N + i] * A[k*N + k] * A[k*N + j];  // L_ik * D_k * L_jk
                A[j*N + i] = (A[j*N + i] - sum) / Dj;
            }
            __syncwarp();
        }
    }

    /**
     * @brief Single-warp LDLᵀ solve `A x = b` in place from an `ldlt` factor (NON-pivoted).
     *
     * Warp parity with block `glass::ldlt_solve`: one 32-lane warp runs the three
     * sweeps — forward unit-`L` (`L y = b`), diagonal scale (`z = y / D`), back
     * unit-`Lᵀ` (`Lᵀ x = z`) — over the factor `LD` from `warp::ldlt`. `b` is
     * overwritten with `x`. No shared scratch; `__syncwarp` between dependent sweeps.
     * Non-pivoted (matches the non-pivoted `warp::ldlt`). NumPy: `x = np.linalg.solve(A, b)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (LD is N x N, b length N).
     * @param LD  LDLᵀ factor from `warp::ldlt` (column-major; unit-L strict-lower, D diagonal).
     * @param b   In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N>
    __device__ void ldlt_solve(const T *LD, T *b)
    {
        uint32_t lane = (flat_rank()) & 31;
        for (uint32_t col = 0; col < N; col++) {               // forward: L y = b (unit L)
            T factor = b[col];
            for (uint32_t row = col + 1 + lane; row < N; row += 32)
                b[row] -= LD[col*N + row] * factor;            // L_{row,col}
            __syncwarp();
        }
        for (uint32_t i = lane; i < N; i += 32)                // diagonal scale z = y / D
            b[i] /= LD[i*N + i];
        __syncwarp();
        for (int32_t col = (int32_t)N - 1; col >= 0; col--) {  // back: Lᵀ x = z (unit Lᵀ)
            T factor = b[col];
            for (uint32_t i = lane; i < (uint32_t)col; i += 32)
                b[i] -= LD[i*N + col] * factor;                // (Lᵀ)_{i,col} = L_{col,i}
            __syncwarp();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    /**
     * @brief Single-thread in-place LDLᵀ factorization (LAPACK `sytrf`, lower,
     *        NON-pivoted), compile-time size.
     *
     * ONE thread factors the symmetric (possibly INDEFINITE) `A = L D Lᵀ` in
     * place — the sequential recurrence `D_j = A_jj − Σ_{k<j} L_jk² D_k`, then
     * `L_ij = (A_ij − Σ_{k<j} L_ik D_k L_jk)/D_j` down each column — for
     * thread-per-problem solvers that pack 32 independent low-DOF problems into
     * a warp. On return the diagonal slots hold `D`, the strict lower triangle
     * holds unit-`L` (upper triangle untouched). No square root, so it factors
     * KKT / saddle-point systems Cholesky cannot. No shared scratch, no
     * barriers, no `threadIdx` read; `A` may live in a thread-local array and
     * stay register-resident.
     *
     * A FRESH serial body (not a `ThreadBarrier` instantiation of the block
     * `ldlt_impl`): that impl's runtime `bool pivot` branch would compile the
     * whole Bunch–Kaufman path — shared-scratch broadcasts, `iamax_lowmem`, raw
     * block-wide barriers — into the thread-tier function. Same algorithm and
     * operand order as the NON-pivoted `glass::ldlt<T, N>` path (and
     * `warp::ldlt`, itself a separate body), agreeing to a few ULP
     * (FMA-contraction jitter; bit-identity across tiers is NOT guaranteed).
     * **Non-pivoted only** — the tier is branch-free by contract (every lane
     * owns a different problem, so the data-dependent Bunch–Kaufman branches
     * would diverge across the warp; the block path covers the pivoted case).
     * Every pivot `D_j` must be nonzero; a saddle like `[[0,b],[b,0]]` breaks
     * down (use the block `ldlt(..., pivot=true, piv)` for those).
     *
     * `CHECK` (compile-out, default false) reports a zero/NaN pivot via
     * `s_fail` and the inertia `{n_pos, n_neg, n_zero}` via `s_inertia`.
     * SciPy: `lu, d, _ = scipy.linalg.ldl(A, lower=True)` ⇒
     * `A == lu @ np.diag(d) @ lu.T`.
     *
     * @tparam T      Scalar type (use `double` for ill-conditioned A).
     * @tparam N      Dimension (A is N x N). N<=7 keeps `A` register-resident (measured
     *                ceiling, both dtypes — see the thread-tier constraints in CLAUDE.md); larger N
     *                still computes correctly but demotes `A` to local memory, forfeiting the
     *                tier's premise.
     * @tparam CHECK  If true, report zero/NaN pivot + inertia (default false, compiles out).
     * @param A         In/out N x N symmetric matrix (column-major, lower); on return holds
     *                  L (strict-lower, unit) and D (diagonal).
     * @param s_fail    Optional flag (CHECK only): set to 1 on a zero/NaN pivot, else 0. Ignored when null.
     * @param s_inertia Optional length-3 `{n_pos, n_neg, n_zero}` pivot-sign counts (CHECK only).
     */
    template <typename T, uint32_t N, bool CHECK = false>
    __device__ void ldlt(T *A, int *s_fail = nullptr, int *s_inertia = nullptr)
    {
        if constexpr (CHECK) {
            if (s_fail) *s_fail = 0;
            if (s_inertia) { s_inertia[0] = 0; s_inertia[1] = 0; s_inertia[2] = 0; }
        }
        for (uint32_t j = 0; j < N; j++) {
            // serial diagonal pivot: D_j = A_jj - sum_{k<j} L_jk^2 * D_k
            T sum = static_cast<T>(0);
            for (uint32_t k = 0; k < j; k++) {
                T Ljk = A[k*N + j];               // L_jk (strict-lower, row j, col k)
                sum += Ljk * Ljk * A[k*N + k];    // * D_k (diagonal slot)
            }
            A[j*N + j] -= sum;                    // overwrite diagonal with D_j
            T Dj = A[j*N + j];
            if constexpr (CHECK) {
                if (s_fail && (Dj == static_cast<T>(0) || isnan(Dj))) *s_fail = 1;
                if (s_inertia) {
                    if (Dj > static_cast<T>(0)) s_inertia[0]++;
                    else if (Dj < static_cast<T>(0)) s_inertia[1]++;
                    else s_inertia[2]++;
                }
            }
            // trailing column: L_ij = (A_ij - sum_{k<j} L_ik * D_k * L_jk) / D_j
            for (uint32_t i = j + 1; i < N; i++) {
                T csum = static_cast<T>(0);
                for (uint32_t k = 0; k < j; k++)
                    csum += A[k*N + i] * A[k*N + k] * A[k*N + j];  // L_ik * D_k * L_jk
                A[j*N + i] = (A[j*N + i] - csum) / Dj;
            }
        }
    }

    /**
     * @brief Single-thread LDLᵀ solve `A x = b` in place from an `ldlt` factor
     *        (LAPACK `sytrs` analogue, NON-pivoted), compile-time size.
     *
     * ONE thread runs the three sweeps — forward unit-`L` (`L y = b`), diagonal
     * scale (`z = y / D`), back unit-`Lᵀ` (`Lᵀ x = z`) — over the factor `LD`
     * from `thread::ldlt`. `LD` is read-only; `b` is overwritten with `x`. No
     * shared scratch, no barriers, no `threadIdx` read; operands may be
     * thread-local register arrays. Non-pivoted only (matches `thread::ldlt`;
     * see there for why the pivoted path is excluded from this tier). A fresh
     * serial body for the same reason as `thread::ldlt` — the block
     * `ldlt_solve_impl` interleaves runtime `piv` branches through every sweep.
     * Same algorithm and operand order as the non-pivoted
     * `glass::ldlt_solve<T, N>` path, agreeing to a few ULP (bit-identity
     * across tiers is NOT guaranteed). NumPy: `x = np.linalg.solve(A, b)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (LD is N x N, b has length N). N<=7 keeps a `T[N*N]` factor
     *            register-resident (measured ceiling, both dtypes — see the thread-tier
     *            constraints in CLAUDE.md).
     * @param LD  LDLᵀ factor from `thread::ldlt` (column-major; unit-L strict-lower, D diagonal).
     * @param b   In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N>
    __device__ void ldlt_solve(const T *LD, T *b)
    {
        for (uint32_t col = 0; col < N; col++) {               // forward: L y = b (unit L)
            T factor = b[col];
            for (uint32_t row = col + 1; row < N; row++)
                b[row] -= LD[col*N + row] * factor;            // L_{row,col}
        }
        for (uint32_t i = 0; i < N; i++)                       // diagonal scale z = y / D
            b[i] /= LD[i*N + i];
        for (int32_t col = (int32_t)N - 1; col >= 0; col--) {  // back: Lᵀ x = z (unit Lᵀ)
            T factor = b[col];
            for (uint32_t i = 0; i < (uint32_t)col; i++)
                b[i] -= LD[i*N + col] * factor;                // (Lᵀ)_{i,col} = L_{col,i}
        }
    }
}
