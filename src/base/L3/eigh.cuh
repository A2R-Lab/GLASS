#pragma once
#include <cstdint>
#include <cstddef>

/**
 * @file eigh.cuh
 * @brief Fixed-sweep cyclic Jacobi eigendecomposition (`eigh`) and the PSD
 *        projection built on it (`psd_project`).
 *
 * The DETERMINISTIC sibling of `glass::syev` (syev.cuh). `syev` is also cyclic
 * Jacobi, but adaptive: convergence-checked with early exit, rotation-skip
 * thresholds, and an ascending sort of the spectrum. `eigh` strips all of that
 * for the batched-solver hot path (GATO's stage-local PSD projection): a FIXED
 * sweep count (no data-dependent loop, no convergence check), a round-robin
 * (circle-method) pair schedule whose per-round pairs are DISJOINT — so every
 * rotation in a round applies concurrently with one barrier per phase — and an
 * UNSORTED spectrum (`W[i]` pairs with `V(:,i)`; the consumer clips, never
 * ranks). Fixed schedule + fixed sweeps + no reductions ⇒ output is
 * bit-identical across thread counts and across runs.
 *
 * Reference/parity oracle: `jacobi_study.py` (GATO so_sqp_prototype) —
 * `round_robin_rounds()` + `jacobi_eigh()`. The rotation formulas here mirror
 * it exactly (theta/t/c/s including the theta==0 → t=1 branch and the
 * `apq == 0` skip). One deliberate divergence: the oracle applies a round's
 * pairs serially, this kernel applies them phased (all row updates, barrier,
 * all col+V updates) — disjoint-pair rotations commute exactly in real
 * arithmetic, so the two agree to rounding (near-bitwise in f64, ~1e-7 rel in
 * f32), not bit-for-bit.
 *
 * Compile-time `N` only: the schedule is a compile-time constant (`N <= 64`;
 * sized by the consumer's stage blocks, n = 12..21 today).
 */

namespace detail {

/// Circle-method round-robin schedule for N indices: M-1 rounds (M = N padded
/// even) of M/2 disjoint (p<q) pairs; slots touching the pad index hold the
/// 0xFF sentinel. Mirrors jacobi_study.py's round_robin_rounds() exactly
/// (same rotation rule `idx = [idx[0], idx[-1], idx[1:-1]]`, same pair sets).
template <uint32_t N>
struct EighSchedule {
    static_assert(N >= 2 && N <= 64, "eigh: schedule is uint8-indexed, N in [2, 64]");
    static constexpr uint32_t M = N + (N & 1u);
    uint8_t p[M - 1][M / 2];
    uint8_t q[M - 1][M / 2];
};

// __host__ __device__: the call sits in a device function's static constexpr
// initializer — compile-time evaluated, but nvcc (no --expt-relaxed-constexpr)
// still requires the device annotation.
template <uint32_t N>
__host__ __device__ constexpr EighSchedule<N> eigh_schedule()
{
    constexpr uint32_t M = EighSchedule<N>::M;
    EighSchedule<N> S{};
    uint8_t idx[M] = {};
    for (uint32_t i = 0; i < M; i++) idx[i] = static_cast<uint8_t>(i);
    for (uint32_t r = 0; r < M - 1; r++) {
        for (uint32_t k = 0; k < M / 2; k++) {
            uint8_t a = idx[k], b = idx[M - 1 - k];
            if (a < N && b < N) {
                S.p[r][k] = (a < b) ? a : b;
                S.q[r][k] = (a < b) ? b : a;
            } else {
                S.p[r][k] = 0xFF; S.q[r][k] = 0xFF;   // pad-index slot: no rotation
            }
        }
        uint8_t last = idx[M - 1];                    // rotate all but idx[0]
        for (uint32_t i = M - 1; i >= 2; i--) idx[i] = idx[i - 1];
        idx[1] = last;
    }
    return S;
}

}  // namespace detail

/**
 * @brief Default sweep count for `eigh` by scalar width.
 *
 * From the de-risk study on real GATO stage Hessians (n = 12..21): p95
 * convergence is 5 sweeps, so f32 ships 6; f64 (tests/debug) ships 12. A
 * caller with harder spectra can override the SWEEPS template parameter.
 */
template <typename T>
__host__ __device__ constexpr uint32_t eigh_sweeps() { return sizeof(T) == 8 ? 12u : 6u; }

/**
 * @brief Scratch size in bytes for `eigh`.
 *
 * Exact layout (in `T` elements): `N*N` working copy of `A` + `2*ceil(N/2)`
 * per-pair rotation coefficients (c then s).
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension.
 * @return Bytes to allocate for `eigh`'s `s_scratch`.
 */
template <typename T, uint32_t N>
__host__ __device__ constexpr std::size_t eigh_scratch_bytes()
{
    return static_cast<std::size_t>(N*N + 2*((N + 1) / 2)) * sizeof(T);
}

/**
 * @brief Fixed-sweep cyclic Jacobi eigendecomposition `A = V diag(W) Vᵀ`.
 *
 * SWEEPS full round-robin sweeps (no convergence check — see the file header
 * for the design and the `glass::syev` cross-reference), each sweep M-1 rounds
 * of disjoint pairs applied phased: per round, one worker per pair computes
 * (c, s) from the current 2×2 block (`apq == 0` skips, publishing the
 * identity), barrier; all pairs' ROW rotations apply concurrently (pairs own
 * disjoint rows), barrier; all pairs' COLUMN rotations plus the V-column
 * accumulation apply concurrently (disjoint columns), barrier. The row/col
 * split is mandatory: rows p,q and cols p,q overlap at the four pivot entries
 * `(p,p),(p,q),(q,p),(q,q)`, so an unfenced one-pass update would read
 * half-updated values.
 *
 * `W` is UNSORTED (`W[i]` ↔ `V(:,i)`, the natural Jacobi order) — the PSD
 * projection consumer clips eigenvalues and never ranks them; callers needing
 * an ascending spectrum want `glass::syev`. Output is bit-identical across
 * thread counts and across runs (fixed schedule, fixed sweeps, no reductions,
 * no atomics). NumPy equivalent (up to eigenvalue order):
 * `W, V = np.linalg.eigh(A)`.
 *
 * @tparam T       Scalar type (`float` ships 6 sweeps, `double` 12 — see
 *                 `eigh_sweeps`).
 * @tparam N       Matrix dimension (compile-time; 2..64).
 * @tparam SWEEPS  Full Jacobi sweeps to run (fixed; no early exit).
 * @param A          In: `N x N` symmetric matrix (column-major; read-only).
 * @param W          Out: `N` eigenvalues, UNSORTED.
 * @param V          Out: `N x N` eigenvectors (column-major; column i ↔ W[i]).
 * @param s_scratch  Shared scratch of `eigh_scratch_bytes<T, N>()` bytes.
 */
template <typename T, uint32_t N, uint32_t SWEEPS = eigh_sweeps<T>()>
__device__ void eigh(const T *A, T *W, T *V, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    constexpr uint32_t M      = N + (N & 1u);
    constexpr uint32_t ROUNDS = M - 1u;
    constexpr uint32_t KMAX   = M / 2u;
    static constexpr detail::EighSchedule<N> sched = detail::eigh_schedule<N>();
    // Scratch layout — see eigh_scratch_bytes: [0, N*N) working copy B; then
    // KMAX c's; then KMAX s's.
    T *s_B = s_scratch;
    T *s_c = s_scratch + N*N;
    T *s_s = s_c + KMAX;

    // B := A (A stays read-only), V := I. One barrier before the sweeps.
    for (uint32_t idx = rank; idx < N*N; idx += size) {
        s_B[idx] = A[idx];
        uint32_t r = idx % N, c = idx / N;
        V[idx] = (r == c) ? static_cast<T>(1) : static_cast<T>(0);
    }
    __syncthreads();

    for (uint32_t sweep = 0; sweep < SWEEPS; sweep++) {
        for (uint32_t r = 0; r < ROUNDS; r++) {
            // Phase 0: one worker per pair computes (c, s) — the oracle's exact
            // formulas: theta = (B_qq - B_pp) / (2 B_pq); t = sign(theta) /
            // (|theta| + sqrt(1 + theta²)) with theta == 0 → t = 1;
            // c = 1/sqrt(1+t²); s = t c. `apq == 0` publishes the identity
            // (mirrors the oracle's `continue`; the update phases skip s == 0
            // rather than multiply through, so a -0 row entry can't flip sign).
            for (uint32_t k = rank; k < KMAX; k += size) {
                T c = static_cast<T>(1), s = static_cast<T>(0);
                uint32_t p = sched.p[r][k];
                if (p != 0xFFu) {
                    uint32_t q = sched.q[r][k];
                    T apq = s_B[p + q*N];
                    if (apq != static_cast<T>(0)) {
                        T theta = (s_B[q + q*N] - s_B[p + p*N]) / (static_cast<T>(2) * apq);
                        T at = (theta < static_cast<T>(0)) ? -theta : theta;
                        T t = static_cast<T>(1) / (at + sqrt(static_cast<T>(1) + theta*theta));
                        if (theta < static_cast<T>(0)) t = -t;
                        if (theta == static_cast<T>(0)) t = static_cast<T>(1);
                        c = static_cast<T>(1) / sqrt(static_cast<T>(1) + t*t);
                        s = t * c;
                    }
                }
                s_c[k] = c; s_s[k] = s;
            }
            __syncthreads();                 // (c, s) visible to the block

            // Phase 1: row rotations, all pairs concurrently — worker (k, j)
            // rewrites B(p, j) and B(q, j); pairs own DISJOINT row sets, and
            // each worker reads only the two entries it writes.
            for (uint32_t idx = rank; idx < KMAX*N; idx += size) {
                uint32_t k = idx / N, j = idx % N;
                uint32_t p = sched.p[r][k];
                T s = s_s[k];
                if (p == 0xFFu || s == static_cast<T>(0)) continue;
                uint32_t q = sched.q[r][k];
                T c = s_c[k];
                T bpj = s_B[p + j*N], bqj = s_B[q + j*N];
                s_B[p + j*N] = c*bpj - s*bqj;
                s_B[q + j*N] = s*bpj + c*bqj;
            }
            __syncthreads();                 // rows settled before columns read them

            // Phase 2: column rotations + V-column accumulation, all pairs
            // concurrently — worker (k, i) rewrites B(i, p), B(i, q) (which
            // read the phase-1 output, resolving the pivot-entry overlap) and
            // V(i, p), V(i, q) (V is untouched by phase 1; same disjointness).
            for (uint32_t idx = rank; idx < KMAX*N; idx += size) {
                uint32_t k = idx / N, i = idx % N;
                uint32_t p = sched.p[r][k];
                T s = s_s[k];
                if (p == 0xFFu || s == static_cast<T>(0)) continue;
                uint32_t q = sched.q[r][k];
                T c = s_c[k];
                T bip = s_B[i + p*N], biq = s_B[i + q*N];
                s_B[i + p*N] = c*bip - s*biq;
                s_B[i + q*N] = s*bip + c*biq;
                T vip = V[i + p*N], viq = V[i + q*N];
                V[i + p*N] = c*vip - s*viq;
                V[i + q*N] = s*vip + c*viq;
            }
            __syncthreads();                 // round complete before the next (c, s)
        }
    }

    // W := diag(B), unsorted.
    for (uint32_t i = rank; i < N; i += size) W[i] = s_B[i + i*N];
    __syncthreads();                         // outputs valid for every thread on return
}

/**
 * @brief Scratch size in bytes for `psd_project`.
 *
 * Exact layout (in `T` elements): `N` eigenvalues + `N*N` eigenvectors +
 * `eigh`'s own scratch (`N*N + 2*ceil(N/2)` — see `eigh_scratch_bytes`).
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension.
 * @return Bytes to allocate for `psd_project`'s `s_scratch`.
 */
template <typename T, uint32_t N>
__host__ __device__ constexpr std::size_t psd_project_scratch_bytes()
{
    return static_cast<std::size_t>(N + N*N) * sizeof(T) + eigh_scratch_bytes<T, N>();
}

/**
 * @brief PSD projection in place: `A := V diag(max(W, eps)) Vᵀ` with
 *        `W, V = eigh(A)` — the fixed-sweep, deterministic eigenvalue clip.
 *
 * The batched-solver counterpart of `glass::eig_clamp` (same math, different
 * engine): `eig_clamp` runs the adaptive `syev`, this runs the fixed-sweep
 * `eigh`, so its cost is compile-time constant and its output bit-identical
 * across thread counts and runs — what a batched SQP stage-Hessian projection
 * wants (GATO computes `eps = 1e-6 * (1 + max(diag))` per block). The result
 * is symmetric PSD for any symmetric input when `eps > 0`; the reconstruction
 * accumulates entry `(r, c)` in canonical `(lo, hi)` operand order, so mirror
 * entries' FMA chains are identical and the output is symmetric bit-for-bit
 * (same pattern as `eig_clamp`). NumPy equivalent:
 * `W, V = np.linalg.eigh(A); A = (V * np.maximum(W, eps)) @ V.T`.
 *
 * @tparam T       Scalar type.
 * @tparam N       Matrix dimension (compile-time; 2..64).
 * @tparam SWEEPS  Jacobi sweeps for the `eigh` call (fixed; default by dtype).
 * @param A          In/out: `N x N` symmetric matrix (column-major); on return
 *                   holds the eigenvalue-clipped reconstruction (PSD).
 * @param eps        Eigenvalue floor (runtime scalar; >= 0).
 * @param s_scratch  Shared scratch of `psd_project_scratch_bytes<T, N>()` bytes.
 */
template <typename T, uint32_t N, uint32_t SWEEPS = eigh_sweeps<T>()>
__device__ void psd_project(T *A, T eps, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    // Scratch layout — see psd_project_scratch_bytes: W (N) | V (N*N) | eigh scratch.
    T *s_W = s_scratch;
    T *s_V = s_scratch + N;
    T *s_e = s_scratch + N + N*N;
    // eigh reads A only in its initial copy phase and ends on a barrier, so A
    // is safely overwritten by the reconstruction below.
    eigh<T, N, SWEEPS>(A, s_W, s_V, s_e);
    for (uint32_t idx = rank; idx < N*N; idx += size) {
        uint32_t r = idx % N, c = idx / N;
        uint32_t lo = (r < c) ? r : c, hi = (r < c) ? c : r;
        T sum = static_cast<T>(0);
        for (uint32_t k = 0; k < N; k++) {
            T w = s_W[k];
            if (w < eps) w = eps;
            T t = w * s_V[lo + k*N];       // canonical (lo, hi) order => the
            sum += t * s_V[hi + k*N];      // mirror entry is bit-identical
        }
        A[idx] = sum;
    }
    __syncthreads();                       // projected A valid for every thread on return
}
