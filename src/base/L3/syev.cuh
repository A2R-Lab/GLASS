#pragma once
#include <cstdint>
#include <math.h>

/**
 * @file syev.cuh
 * @brief Symmetric eigendecomposition via cyclic Jacobi (`syev`) and the
 *        eigenvalue-clamping consumer op (`eig_clamp`).
 *
 * Device-side small symmetric eigensolver (target sizes n ≤ 32 — robot
 * state/control dimensions). Kills the host round-trip solvers like PDDP
 * currently make to `Eigen::SelfAdjointEigenSolver` mid-solve just to clamp
 * Hessian eigenvalues: `eig_clamp` does decompose → clamp → reconstruct
 * entirely inside the block.
 *
 * Algorithm (classical cyclic Jacobi): work on a copy `B = A` in scratch and
 * accumulate `V = I`. For each pair `(p, q)`, `p < q`, in a FIXED cyclic order,
 * compute the Jacobi rotation `(c, s)` annihilating `B[p,q]` (standard stable
 * formulas: `theta = (B_qq - B_pp)/(2 B_pq)`,
 * `t = sign(theta)/(|theta| + sqrt(1 + theta^2))`, `c = 1/sqrt(1 + t^2)`,
 * `s = t*c`) and apply it to rows/cols `p, q` of `B` and columns `p, q` of `V`.
 * Sweeps repeat until `off(B)_F <= eps_T * ||A||_F` (checked once per sweep) or
 * a fixed cap of 15 sweeps; typical convergence is 5–8 sweeps for n ≤ 32.
 * Jacobi always converges for (finite) symmetric input, so there is no
 * CHECK-style failure flag; **NaN/Inf input propagates** into `W`/`V`
 * (garbage-in, NaN-out — the deterministic sort still terminates).
 *
 * Thread-count invariance (bit-identical at any block size): the `(p, q)` pair
 * loop is SERIAL in a fixed order; rank 0 computes `(c, s)` into shared slots
 * followed by a barrier (one FMA chain, identical for every launch); the
 * row/col rotation is thread-strided over the n affected indices, where index
 * `k` owns exactly the entries `{(k,p),(k,q),(p,k),(q,k)}` of `B` (mirror
 * writes keep `B` exactly symmetric) and row `k` of `V` — its new values read
 * only slots owned by the same `k` (the pivot 2x2 block is owned by `k == p`),
 * so the rotation phase has no cross-thread read-after-write hazard and needs
 * no staging. The sweep-end `off(B)` probe and the final ascending selection
 * sort run serially on rank 0 (deterministic), each published through a shared
 * slot + barrier; the sorted copy-out permutes `V`'s columns in parallel
 * through the no-longer-needed `B` scratch.
 */

/**
 * @brief Machine epsilon by scalar width (float / double).
 *
 * Used to scale `syev`'s deterministic convergence / rotation-skip thresholds.
 * Spelled locally (sizeof-keyed constants) so no `<limits>` lands inside
 * `namespace glass` (this header is included inside the namespace).
 *
 * @tparam T  Scalar type (4-byte -> FLT_EPSILON, 8-byte -> DBL_EPSILON).
 * @return Machine epsilon of `T`.
 */
template <typename T>
__host__ __device__ constexpr T syev_eps()
{
    return static_cast<T>(sizeof(T) >= 8 ? 2.220446049250313e-16
                                         : 1.1920928955078125e-7);
}

/**
 * @brief Scratch size in bytes for `syev`.
 *
 * Exact layout (in `T` elements): `n*n` for the working copy `B` (reused at the
 * end as the eigenvector permutation staging buffer) + `n` slots holding the
 * ascending sort permutation (stored as `uint32_t`, one per `T` slot) + 4
 * control slots (`c`, `s`, the sweep-converged flag, one pad).
 *
 * @tparam T  Scalar type.
 * @param n  Matrix dimension (A is n x n).
 * @return Bytes to allocate for `syev`'s `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t syev_scratch_bytes(uint32_t n)
{
    return static_cast<std::size_t>(n*n + n + 4) * sizeof(T);
}

/**
 * @brief Symmetric eigendecomposition `A = V diag(W) Vᵀ` (cyclic Jacobi;
 *        LAPACK `syev` analogue).
 *
 * Computes all eigenvalues and eigenvectors of the symmetric `n x n`
 * column-major matrix `A`. **`A` is preserved** (read-only — the iteration
 * works on a copy in `s_scratch`). On return `W[0..n-1]` holds the eigenvalues
 * in ASCENDING order and column `i` of the column-major `V` holds the unit
 * eigenvector matching `W[i]` (eigenvector signs are arbitrary, as with any
 * eigensolver). NumPy equivalent: `W, V = np.linalg.eigh(A)`.
 *
 * Single-block, thread-count invariant (bit-identical output at any block
 * size), deterministic: fixed cyclic pair order, rank-0 rotation coefficients
 * broadcast via shared memory + barrier, per-sweep `off(B) <= eps*||A||_F`
 * early exit decided by rank 0 into a shared flag, capped at 15 sweeps
 * (typical: 5–8 for n ≤ 32). Rotations with `|B[p,q]|` below a deterministic
 * tiny threshold (`eps*||A||_F / n²`) are skipped. Always converges for finite
 * symmetric input (no failure flag); NaN/Inf input propagates into the outputs.
 *
 * @tparam T  Scalar type (`float` / `double`; use `double` when eigenvalue
 *            clusters must be resolved tightly).
 * @param n          Matrix dimension (A is n x n; designed for n <= 32).
 * @param A          In: n x n symmetric matrix (column-major). NOT modified.
 * @param W          Out: n eigenvalues, ascending.
 * @param V          Out: n x n eigenvectors (column-major; column i ↔ W[i]).
 * @param s_scratch  Shared scratch of `syev_scratch_bytes<T>(n)` bytes.
 */
// Shared body (runtime + compile-time overloads): SizeT deduced — uint32_t or
// ct_size<N> (constant-folds the trip counts / indexing).
template <typename T, typename SizeT>
__device__ void syev_impl(SizeT n, const T *A, T *W, T *V, T *s_scratch)
{
    static_assert(sizeof(uint32_t) <= sizeof(T),
                  "syev: the permutation slots assume sizeof(T) >= 4");
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    // Scratch layout — see syev_scratch_bytes: [0, n*n) working copy B (reused
    // as the V-permutation staging buffer at the end); [n*n, n*n + n) the sort
    // permutation (as uint32_t); then 4 control slots: [0]=c, [1]=s,
    // [2]=sweep-converged flag, [3]=pad.
    T *s_B = s_scratch;
    uint32_t *s_perm = reinterpret_cast<uint32_t *>(s_scratch + n*n);
    T *s_ctrl = s_scratch + n*n + n;

    // B := A (A stays read-only), V := I. One barrier before the sweeps.
    for (uint32_t idx = rank; idx < n*n; idx += size) {
        s_B[idx] = A[idx];
        uint32_t r = idx % n, c = idx / n;
        V[idx] = (r == c) ? static_cast<T>(1) : static_cast<T>(0);
    }
    __syncthreads();

    // Deterministic thresholds, rank-0 registers only (no other thread ever
    // branches on them): convergence target off(B)_F <= eps*||A||_F, and a
    // much smaller per-rotation skip cutoff eps*||A||_F/n² (so the sum of all
    // skipped entries stays far below the convergence target).
    T tol_off = static_cast<T>(0), tol_skip = static_cast<T>(0);
    if (rank == 0) {
        T fro2 = static_cast<T>(0);
        for (uint32_t i = 0; i < n*n; i++) fro2 += s_B[i]*s_B[i];
        tol_off  = syev_eps<T>() * sqrt(fro2);   // type-generic sqrt
        tol_skip = tol_off / static_cast<T>(n*n);
    }

    constexpr uint32_t SYEV_MAX_SWEEPS = 15;   // typical convergence: 5-8 sweeps (n <= 32)
    for (uint32_t sweep = 0; sweep < SYEV_MAX_SWEEPS; sweep++) {
        // One cyclic sweep: fixed (p, q) order => deterministic.
        for (uint32_t p = 0; p + 1 < n; p++) {
            for (uint32_t q = p + 1; q < n; q++) {
                // Rank 0 computes the rotation into shared (identical FMA chain
                // every launch); a skipped rotation publishes the identity.
                if (rank == 0) {
                    T bpq = s_B[p + q*n];
                    T abpq = (bpq < static_cast<T>(0)) ? -bpq : bpq;
                    T c = static_cast<T>(1), s = static_cast<T>(0);
                    if (abpq > tol_skip) {
                        T theta = (s_B[q + q*n] - s_B[p + p*n]) / (static_cast<T>(2)*bpq);
                        T atheta = (theta < static_cast<T>(0)) ? -theta : theta;
                        T t = static_cast<T>(1) / (atheta + sqrt(static_cast<T>(1) + theta*theta));
                        if (theta < static_cast<T>(0)) t = -t;
                        c = static_cast<T>(1) / sqrt(static_cast<T>(1) + t*t);
                        s = t*c;
                    }
                    s_ctrl[0] = c; s_ctrl[1] = s;
                }
                __syncthreads();                 // (c, s) visible to the block
                T c = s_ctrl[0], s = s_ctrl[1];
                if (s != static_cast<T>(0)) {    // uniform branch: same shared values everywhere
                    // Rotate rows/cols p,q of B. Index k owns entries
                    // {(k,p),(k,q),(p,k),(q,k)} — mirror writes keep B exactly
                    // symmetric — and its new values read only k-owned slots
                    // (B[k+p*n], B[k+q*n]), so there is no cross-thread hazard.
                    // The 2x2 pivot block is owned by k == p (k == q idles);
                    // its stable update uses t = s/c: B_pp -= t*B_pq,
                    // B_qq += t*B_pq, B_pq = 0 exactly (the annihilation).
                    T t = s / c;
                    for (uint32_t k = rank; k < n; k += size) {
                        if (k == q) continue;
                        if (k == p) {
                            T bpq = s_B[p + q*n];
                            s_B[p + p*n] -= t*bpq;
                            s_B[q + q*n] += t*bpq;
                            s_B[p + q*n] = static_cast<T>(0);
                            s_B[q + p*n] = static_cast<T>(0);
                        } else {
                            T bkp = s_B[k + p*n], bkq = s_B[k + q*n];
                            T nkp = c*bkp - s*bkq;
                            T nkq = s*bkp + c*bkq;
                            s_B[k + p*n] = nkp; s_B[p + k*n] = nkp;
                            s_B[k + q*n] = nkq; s_B[q + k*n] = nkq;
                        }
                    }
                    // Accumulate the rotation into V's columns p,q (row k is
                    // k-owned; disjoint from the B loop's data, no barrier
                    // needed between the two loops).
                    for (uint32_t k = rank; k < n; k += size) {
                        T vkp = V[k + p*n], vkq = V[k + q*n];
                        V[k + p*n] = c*vkp - s*vkq;
                        V[k + q*n] = s*vkp + c*vkq;
                    }
                }
                // UNCONDITIONAL trailing barrier: rotated B/V visible before the
                // next (c, s) — and, on a skipped rotation, it keeps rank 0 from
                // overwriting s_ctrl[0..1] while another thread is still reading
                // the previous pair's values (which could diverge the skip branch).
                __syncthreads();
            }
        }
        // Sweep-end early exit: rank 0 serially sums the squared off-diagonals
        // (deterministic for any block size) and publishes the verdict.
        if (rank == 0) {
            T off2 = static_cast<T>(0);
            for (uint32_t c = 0; c < n; c++)
                for (uint32_t r = 0; r < n; r++)
                    if (r != c) off2 += s_B[r + c*n]*s_B[r + c*n];
            s_ctrl[2] = (off2 <= tol_off*tol_off) ? static_cast<T>(1) : static_cast<T>(0);
        }
        __syncthreads();                         // flag visible to the block
        bool done = (s_ctrl[2] != static_cast<T>(0));
        __syncthreads();                         // all reads done before rank 0 rewrites s_ctrl
        if (done) break;                         // uniform: same shared value everywhere
    }

    // Ascending sort: rank 0 selection-sorts the diagonal into a permutation
    // (strict '<' => ties keep the lower original index — deterministic).
    if (rank == 0) {
        for (uint32_t i = 0; i < n; i++) s_perm[i] = i;
        for (uint32_t i = 0; i + 1 < n; i++) {
            uint32_t best = i;
            for (uint32_t j = i + 1; j < n; j++)
                if (s_B[s_perm[j]*n + s_perm[j]] < s_B[s_perm[best]*n + s_perm[best]]) best = j;
            uint32_t tmp = s_perm[i]; s_perm[i] = s_perm[best]; s_perm[best] = tmp;
        }
    }
    __syncthreads();                             // permutation visible to the block
    // Permuted copy-out. Eigenvalues first (they read diag(B))...
    for (uint32_t i = rank; i < n; i += size) {
        uint32_t pi = s_perm[i];
        W[i] = s_B[pi*n + pi];
    }
    __syncthreads();                             // W extracted before B is overwritten
    // ...then permute V's columns through the B scratch (B is dead now).
    for (uint32_t idx = rank; idx < n*n; idx += size) s_B[idx] = V[idx];
    __syncthreads();                             // staging complete before the permuted write
    for (uint32_t idx = rank; idx < n*n; idx += size) {
        uint32_t r = idx % n, c = idx / n;
        V[idx] = s_B[r + s_perm[c]*n];
    }
    __syncthreads();                             // outputs valid for every thread on return
}

template <typename T>
__device__ void syev(uint32_t n, const T *A, T *W, T *V, T *s_scratch)
{
    syev_impl<T>(n, A, W, V, s_scratch);
}

/**
 * @brief Compile-time-size symmetric eigendecomposition (cyclic Jacobi).
 *
 * Same as the runtime `syev` but with the dimension as a template parameter,
 * letting the compiler bake `N` in (constant-folded trip counts / indexing).
 * `A` is preserved; `W` ascending; `V` column i ↔ `W[i]`. NumPy equivalent:
 * `W, V = np.linalg.eigh(A)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N; designed for N <= 32).
 * @param A          In: N x N symmetric matrix (column-major). NOT modified.
 * @param W          Out: N eigenvalues, ascending.
 * @param V          Out: N x N eigenvectors (column-major; column i ↔ W[i]).
 * @param s_scratch  Shared scratch of `syev_scratch_bytes<T>(N)` bytes.
 */
template <typename T, uint32_t N>
__device__ void syev(const T *A, T *W, T *V, T *s_scratch)
{
    syev_impl<T>(ct_size<N>{}, A, W, V, s_scratch);
}

/**
 * @brief Scratch size in bytes for `eig_clamp`.
 *
 * Exact layout (in `T` elements): `n` eigenvalues + `n*n` eigenvectors +
 * `syev`'s own scratch (`n*n + n + 4` — see `syev_scratch_bytes`), i.e.
 * `2*n*n + 2*n + 4` elements total.
 *
 * @tparam T  Scalar type.
 * @param n  Matrix dimension (A is n x n).
 * @return Bytes to allocate for `eig_clamp`'s `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t eig_clamp_scratch_bytes(uint32_t n)
{
    return static_cast<std::size_t>(2*n*n + 2*n + 4) * sizeof(T);
}

/**
 * @brief Clamp the eigenvalues of a symmetric matrix in place:
 *        `A := V diag(max(W, eps)) Vᵀ` where `W, V = eigh(A)`.
 *
 * The Hessian-regularization op solvers host-round-trip for: eigendecompose the
 * symmetric `n x n` column-major `A` (device-side `syev`), floor every
 * eigenvalue at `eps`, and reconstruct in place — the result is symmetric
 * positive-definite for any symmetric input when `eps > 0` (indefinite and
 * negative-definite inputs included). If `A`'s eigenvalues already all exceed
 * `eps`, the result is `A` up to the eigensolver's round-off. NumPy equivalent:
 * `W, V = np.linalg.eigh(A); A = (V * np.maximum(W, eps)) @ V.T`.
 *
 * The reconstruction is a hand-rolled thread-strided loop (each thread owns
 * disjoint output entries of `A`; it reads only the shared `W`/`V`, so no
 * barrier is needed inside the pass): entry `(r, c)` accumulates
 * `sum_k max(W[k], eps) * V[lo,k] * V[hi,k]` with `lo = min(r,c)`,
 * `hi = max(r,c)` — the canonical operand order makes the mirror entries'
 * FMA chains identical, so the output is exactly symmetric bit-for-bit.
 * Thread-count invariant end to end (inherits `syev`'s determinism).
 *
 * @tparam T  Scalar type.
 * @param n          Matrix dimension (A is n x n; designed for n <= 32).
 * @param A          In/out: n x n symmetric matrix (column-major); on return
 *                   holds the eigenvalue-clamped reconstruction (SPD).
 * @param eps        Eigenvalue floor (e.g. a small positive regularizer).
 * @param s_scratch  Shared scratch of `eig_clamp_scratch_bytes<T>(n)` bytes.
 */
// Shared body (runtime + compile-time overloads): SizeT deduced — uint32_t or
// ct_size<N> (constant-folds the trip counts / indexing).
template <typename T, typename SizeT>
__device__ void eig_clamp_impl(SizeT n, T *A, T eps, T *s_scratch)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    // Scratch layout — see eig_clamp_scratch_bytes: W (n) | V (n*n) | syev scratch.
    T *s_W = s_scratch;
    T *s_V = s_scratch + n;
    T *s_sy = s_scratch + n + n*n;
    // syev reads A only in its initial copy phase and ends on a barrier, so A
    // is safely overwritten by the reconstruction below.
    syev_impl<T>(n, A, s_W, s_V, s_sy);
    for (uint32_t idx = rank; idx < n*n; idx += size) {
        uint32_t r = idx % n, c = idx / n;
        uint32_t lo = (r < c) ? r : c, hi = (r < c) ? c : r;
        T sum = static_cast<T>(0);
        for (uint32_t k = 0; k < n; k++) {
            T w = s_W[k];
            if (w < eps) w = eps;
            T t = w * s_V[lo + k*n];       // canonical (lo, hi) order => the
            sum += t * s_V[hi + k*n];      // mirror entry is bit-identical
        }
        A[idx] = sum;
    }
    __syncthreads();                       // clamped A valid for every thread on return
}

template <typename T>
__device__ void eig_clamp(uint32_t n, T *A, T eps, T *s_scratch)
{
    eig_clamp_impl<T>(n, A, eps, s_scratch);
}

/**
 * @brief Compile-time-size eigenvalue clamp `A := V diag(max(W, eps)) Vᵀ`.
 *
 * Same as the runtime `eig_clamp` but with the dimension as a template
 * parameter. NumPy equivalent:
 * `W, V = np.linalg.eigh(A); A = (V * np.maximum(W, eps)) @ V.T`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N; designed for N <= 32).
 * @param A          In/out: N x N symmetric matrix (column-major); on return
 *                   holds the eigenvalue-clamped reconstruction (SPD).
 * @param eps        Eigenvalue floor.
 * @param s_scratch  Shared scratch of `eig_clamp_scratch_bytes<T>(N)` bytes.
 */
template <typename T, uint32_t N>
__device__ void eig_clamp(T *A, T eps, T *s_scratch)
{
    eig_clamp_impl<T>(ct_size<N>{}, A, eps, s_scratch);
}
