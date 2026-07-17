#pragma once
#include <cstdint>

// ─── congruence / bilinear forms  XᵀMX, XᵀMY ─────────────────────────────────
//
// Fused two-step products that otherwise need two gemms + a temp + a transpose:
//   congruence_sym : Q = α·XᵀMX + β·Q   (Q symmetric — lower triangle + mirror)
//   bilinear       : R = α·XᵀMY + β·R   (general)
// Drives the trajectory-optimization Q-Hessian fold AB'·Vxx·AB. Step 1 forms
// MX = M·X (or MY = M·Y) with glass::gemm into caller scratch; step 2 contracts
// Xᵀ·(MX) over the shared N dimension using the contraction-parallel engine
// (one warp owns each output, lanes split N), exploiting symmetry to compute
// only the lower triangle. Thread-count invariant at any block size (the sub-warp
// path reproduces the warp-shuffle rounding — see gemm_reduced.cuh).
//
// Note: the WITHIN-surface result is bit-identical across thread counts. ACROSS
// surfaces (block vs warp vs cgrps) results agree to floating-point tolerance,
// not bit-for-bit, because step 1's gemm is a separate function whose FMA
// contraction the compiler may fuse differently per instantiation (both correct).
//
// Requires gemm_reduced.cuh (reduced_tree32 + warp::reduce) and gemm.cuh,
// included first by glass.cuh.

namespace detail {
    // R = alpha * X^T * Y + beta*R, X is N×ROWS, Y is N×COLS (both column-major),
    // R is ROWS×COLS column-major. Contraction over the shared leading dim N. When
    // SYMM (ROWS==COLS, Y a congruence partner of X so the product is symmetric)
    // only the lower triangle is computed and mirrored. HAS_BETA reads R.
    template <typename T, uint32_t N, uint32_t ROWS, uint32_t COLS,
              bool SYMM, bool HAS_BETA>
    __device__ void xtY_impl(uint32_t rank, uint32_t size,
                             T alpha, const T* X, const T* Y, T beta, T* R)
    {
        constexpr uint32_t maxel = ROWS * COLS;
        static_assert(!SYMM || ROWS == COLS, "SYMM requires a square result");

        if (size < 32u) {
            for (uint32_t el = rank; el < maxel; el += size) {
                const uint32_t i = el % ROWS, j = el / ROWS;
                if (SYMM && i < j) continue;
                T p[32];
                #pragma unroll
                for (uint32_t vlane = 0; vlane < 32u; ++vlane) {
                    T acc = static_cast<T>(0);
                    for (uint32_t n = vlane; n < N; n += 32u)
                        acc += X[n + i*N] * Y[n + j*N];
                    p[vlane] = acc;
                }
                const T res = reduced_tree32<T>(p);
                const uint32_t idx = i + j*ROWS;
                R[idx] = HAS_BETA ? beta_blend(alpha*res, beta, R[idx]) : (alpha*res);
                if (SYMM && i != j) {
                    const uint32_t m = j + i*ROWS;
                    R[m] = HAS_BETA ? beta_blend(alpha*res, beta, R[m]) : (alpha*res);
                }
            }
            return;
        }

        const uint32_t n_warps = size >> 5;
        const uint32_t warp = rank >> 5, lane = rank & 31u;
        if (warp < n_warps) {
            for (uint32_t el = warp; el < maxel; el += n_warps) {
                const uint32_t i = el % ROWS, j = el / ROWS;
                if (SYMM && i < j) continue;
                T partial = static_cast<T>(0);
                for (uint32_t n = lane; n < N; n += 32u)
                    partial += X[n + i*N] * Y[n + j*N];
                const T res = warp::reduce<T>(partial);
                if (lane == 0) {
                    const uint32_t idx = i + j*ROWS;
                    R[idx] = HAS_BETA ? beta_blend(alpha*res, beta, R[idx]) : (alpha*res);
                    if (SYMM && i != j) {
                        const uint32_t m = j + i*ROWS;
                        R[m] = HAS_BETA ? beta_blend(alpha*res, beta, R[m]) : (alpha*res);
                    }
                }
            }
        }
    }
} // namespace detail

/**
 * @brief Shared-memory bytes needed by congruence_sym / bilinear scratch (`M·X`).
 *
 * The scratch holds the intermediate `N x Kdim` product. Host- and device-callable.
 *
 * @tparam T  Scalar type.
 * @tparam N  Rows of X / dimension of M.
 * @tparam Kdim  Columns of X (= columns of the scratch).
 * @return Bytes for the `s_scratch` buffer (these are small single-block sizes).
 */
template <typename T, uint32_t N, uint32_t Kdim>
__host__ __device__ constexpr std::size_t congruence_scratch_bytes() {
    return static_cast<std::size_t>(N) * Kdim * sizeof(T);
}

/**
 * @brief Symmetric congruence: `Q = alpha * Xᵀ·M·X + beta * Q` (Q symmetric).
 *
 * Forms `MX = M·X` into `s_scratch`, then contracts `Q = Xᵀ·MX` over the shared
 * `N` dimension, computing only the lower triangle and mirroring it (the result
 * is symmetric when `M` is). Replaces two gemms + a temp + a transpose with one
 * fused call. Column-major; single block; thread-count invariant.
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension of M (N x N) and rows of X.
 * @tparam Kdim  Columns of X — `Q` is Kdim x Kdim.
 * @tparam ACCUMULATE  Add into Q (true) vs overwrite (false, default).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha   Scalar on the product.
 * @param X       Input N x Kdim matrix (column-major).
 * @param M       Input N x N matrix (column-major; symmetric for a symmetric Q).
 * @param beta    Scalar on the existing Q (read only when ACCUMULATE).
 * @param Q       In/out Kdim x Kdim result (column-major).
 * @param s_scratch  Shared scratch of `congruence_scratch_bytes<T,N,Kdim>()` bytes (holds M·X).
 */
template <typename T, uint32_t N, uint32_t Kdim,
          bool ACCUMULATE = false, bool TRAILING_SYNC = true>
__device__ void congruence_sym(T alpha, const T* X, const T* M, T beta, T* Q, T* s_scratch)
{
    // step 1: MX = M * X  (N x N times N x Kdim -> N x Kdim), overwrite scratch.
    gemm<T, N, Kdim, N>(static_cast<T>(1), const_cast<T*>(M), const_cast<T*>(X), s_scratch);
    __syncthreads();                         // MX visible before the Xᵀ·MX contraction
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    detail::xtY_impl<T, N, Kdim, Kdim, true, ACCUMULATE>(rank, size, alpha, X, s_scratch, beta, Q);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief General bilinear form: `R = alpha * Xᵀ·M·Y + beta * R`.
 *
 * Like `congruence_sym` but with a distinct right operand `Y`, so the result is
 * not symmetric and the full `P x Qd` matrix is computed. Forms `MY = M·Y` into
 * `s_scratch`, then contracts `R = Xᵀ·MY`. Column-major; single block; invariant.
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension of M (N x N) and rows of X and Y.
 * @tparam P  Columns of X — `R` has P rows.
 * @tparam Qd  Columns of Y — `R` has Qd columns.
 * @tparam ACCUMULATE  Add into R (true) vs overwrite (false, default).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha   Scalar on the product.
 * @param X       Input N x P matrix (column-major).
 * @param M       Input N x N matrix (column-major).
 * @param Y       Input N x Qd matrix (column-major).
 * @param beta    Scalar on the existing R (read only when ACCUMULATE).
 * @param R       In/out P x Qd result (column-major).
 * @param s_scratch  Shared scratch of `congruence_scratch_bytes<T,N,Qd>()` bytes (holds M·Y).
 */
template <typename T, uint32_t N, uint32_t P, uint32_t Qd,
          bool ACCUMULATE = false, bool TRAILING_SYNC = true>
__device__ void bilinear(T alpha, const T* X, const T* M, const T* Y, T beta, T* R, T* s_scratch)
{
    gemm<T, N, Qd, N>(static_cast<T>(1), const_cast<T*>(M), const_cast<T*>(Y), s_scratch);
    __syncthreads();
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    detail::xtY_impl<T, N, P, Qd, false, ACCUMULATE>(rank, size, alpha, X, s_scratch, beta, R);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Scratch size in bytes for `congruence_accum` `s_scratch`.
 *
 * Holds the `Q×P` transpose `Gᵀ` plus the `congruence_sym` scratch (`M·Gᵀ`, also
 * `Q×P`). Total `2*P*Q` elements of `T`.
 */
template <typename T, uint32_t P, uint32_t Q>
__host__ __device__ constexpr std::size_t congruence_accum_scratch_bytes() {
    return 2u * P * Q * sizeof(T);
}

/**
 * @brief Accumulating congruence with a rectangular left factor: `C = alpha*G*M*Gᵀ + beta*C`.
 *
 * The "other orientation" of `congruence_sym`: here the rectangular factor `G` is
 * `P×Q` (the natural storage in e.g. GATO's Schur assembly, where `G = B` and
 * `M = R⁻¹`, giving `B·R⁻¹·Bᵀ`), `M` is `Q×Q` symmetric, and the symmetric result
 * `C` is `P×P`. Mathematically `G·M·Gᵀ = XᵀMX` with `X = Gᵀ`, so this transposes
 * `G` into scratch and defers to `congruence_sym<Q,P>` — inheriting its exact
 * triangle+mirror symmetry and its honest FMA-order note. `ACCUMULATE` adds into
 * `C` (the `+=` GATO wants); default overwrites. Single block, column-major,
 * thread-count invariant. NumPy: `C = alpha*(G @ M @ G.T) + beta*C`.
 *
 * @tparam T  Scalar type.
 * @tparam P  Rows of `G` — `C` is `P×P`.
 * @tparam Q  Columns of `G` and dimension of `M` (`Q×Q`).
 * @tparam ACCUMULATE  Add into `C` (true) vs overwrite (false, default).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha   Scalar on the product.
 * @param G       Input `P×Q` matrix (column-major).
 * @param M       Input `Q×Q` matrix (column-major, symmetric).
 * @param beta    Scalar on the existing `C` (read only when ACCUMULATE).
 * @param C       In/out `P×P` symmetric result (column-major).
 * @param s_scratch  Shared scratch of `congruence_accum_scratch_bytes<T,P,Q>()` elements.
 */
template <typename T, uint32_t P, uint32_t Q,
          bool ACCUMULATE = false, bool TRAILING_SYNC = true>
__device__ void congruence_accum(T alpha, const T* G, const T* M, T beta, T* C, T* s_scratch)
{
    T* Gt  = s_scratch;            // Q×P transpose of G (column-major)
    T* scr = s_scratch + Q * P;    // congruence_sym scratch (M·Gᵀ), Q×P
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t k = rank; k < P * Q; k += size) {
        uint32_t p = k % P, q = k / P;       // G col-major P×Q: G[q*P + p] = G(p,q)
        Gt[p * Q + q] = G[q * P + p];        // Gt col-major Q×P: Gt(q,p) = G(p,q)
    }
    __syncthreads();                          // Gᵀ visible before the congruence
    congruence_sym<T, Q, P, ACCUMULATE, TRAILING_SYNC>(alpha, Gt, M, beta, C, scr);
}

// ─── single-thread congruence / bilinear ─────────────────────────────────────
namespace thread {
    // Single-thread congruence forms: ONE thread runs the same two-step
    // algorithm as the block/warp surfaces — `thread::gemm` (the shared
    // `gemm_impl_ct` at rank=0/size=1) forms M·X into `scratch`, then the SAME
    // validated `detail::xtY_impl` engine (whose size=1 path is barrier-free
    // and shuffle-free, reducing through the register-only `reduced_tree32`)
    // contracts Xᵀ·(MX). The inter-step barrier the block/warp glue needs is
    // gone: on one thread, sequential program order already makes MX visible.
    //
    // SCRATCH IS ALGORITHMIC, NOT CROSS-LANE: the buffer holds the materialized
    // M·X intermediate that step 2 re-reads (it is a workspace, not a lane-
    // exchange staging area), so the thread tier keeps the caller-provided
    // pointer — intended to be a THREAD-LOCAL array (e.g. `T scr[N*Kdim]`),
    // never shared memory shared between lanes. Same sizes as the block twin:
    // `congruence_scratch_bytes` / `congruence_accum_scratch_bytes`.
    //
    // NO `TRAILING_SYNC` parameter, matching the tier's precedent (see the
    // thread:: block in tensor_contract.cuh).

    /**
     * @brief Single-thread symmetric congruence: `Q = alpha * Xᵀ·M·X + beta * Q`.
     *
     * Thread-per-problem analogue of `glass::congruence_sym`: ONE thread forms
     * `MX = M·X` into `scratch` then contracts `Q = Xᵀ·MX` (lower triangle +
     * mirror). No barriers, no shuffles, no `threadIdx` read; operands and
     * `scratch` may be thread-local register arrays (the implied `T[N*N]` M
     * stays register-resident only under the tier's N<=7 element-count ceiling
     * — see CLAUDE.md). Same algorithm and operand order as the `glass::` twin,
     * agreeing to a few ULP (cross-tier bit-identity is NOT guaranteed).
     *
     * @tparam T,N,Kdim,ACCUMULATE  See glass::congruence_sym.
     * @param alpha,X,M,beta,Q  See glass::congruence_sym (Q read only when ACCUMULATE).
     * @param scratch  Workspace of `congruence_scratch_bytes<T,N,Kdim>()` bytes
     *                 (holds M·X); a thread-local `T[N*Kdim]` is the intended form.
     */
    template <typename T, uint32_t N, uint32_t Kdim, bool ACCUMULATE = false>
    __device__ void congruence_sym(T alpha, const T* X, const T* M, T beta, T* Q, T* scratch)
    {
        thread::gemm<T, N, Kdim, N>(static_cast<T>(1), M, X, scratch);   // MX = M·X
        detail::xtY_impl<T, N, Kdim, Kdim, true, ACCUMULATE>(0u, 1u, alpha, X, scratch, beta, Q);
    }

    /**
     * @brief Single-thread general bilinear form: `R = alpha * Xᵀ·M·Y + beta * R`.
     *
     * Thread-per-problem analogue of `glass::bilinear`: ONE thread forms
     * `MY = M·Y` into `scratch` then contracts the full `P x Qd` result. No
     * barriers, no shuffles, no `threadIdx` read; operands and `scratch` may be
     * thread-local register arrays (subject to the tier's element-count ceiling
     * — see CLAUDE.md). Same algorithm and operand order as the `glass::` twin,
     * agreeing to a few ULP (cross-tier bit-identity is NOT guaranteed).
     *
     * @tparam T,N,P,Qd,ACCUMULATE  See glass::bilinear.
     * @param alpha,X,M,Y,beta,R  See glass::bilinear (R read only when ACCUMULATE).
     * @param scratch  Workspace of `congruence_scratch_bytes<T,N,Qd>()` bytes
     *                 (holds M·Y); a thread-local `T[N*Qd]` is the intended form.
     */
    template <typename T, uint32_t N, uint32_t P, uint32_t Qd, bool ACCUMULATE = false>
    __device__ void bilinear(T alpha, const T* X, const T* M, const T* Y, T beta, T* R, T* scratch)
    {
        thread::gemm<T, N, Qd, N>(static_cast<T>(1), M, Y, scratch);     // MY = M·Y
        detail::xtY_impl<T, N, P, Qd, false, ACCUMULATE>(0u, 1u, alpha, X, scratch, beta, R);
    }

    /**
     * @brief Single-thread accumulating congruence `C = alpha*G*M*Gᵀ + beta*C` (G is P×Q).
     *
     * Thread-per-problem analogue of `glass::congruence_accum`: ONE thread
     * transposes `G` into `scratch` and defers to `thread::congruence_sym<Q,P>`
     * — the same construction as the block/warp twins, minus their barrier
     * (sequential program order makes `Gᵀ` visible). No barriers, no shuffles,
     * no `threadIdx` read; `scratch` is algorithmic workspace (holds `Gᵀ` then
     * `M·Gᵀ`), intended as a thread-local `T[2*P*Q]` (see the tier's
     * element-count ceiling in CLAUDE.md). Same algorithm and operand order as
     * the `glass::` twin, agreeing to a few ULP (cross-tier bit-identity is NOT
     * guaranteed).
     *
     * @tparam T,P,Q,ACCUMULATE  See glass::congruence_accum.
     * @param alpha,G,M,beta,C  See glass::congruence_accum (C read only when ACCUMULATE).
     * @param scratch  Workspace of `congruence_accum_scratch_bytes<T,P,Q>()` bytes;
     *                 a thread-local `T[2*P*Q]` is the intended form.
     */
    template <typename T, uint32_t P, uint32_t Q, bool ACCUMULATE = false>
    __device__ void congruence_accum(T alpha, const T* G, const T* M, T beta, T* C, T* scratch)
    {
        T* Gt  = scratch;              // Q×P transpose of G (column-major)
        T* scr = scratch + Q * P;      // congruence_sym scratch (M·Gᵀ), Q×P
        for (uint32_t k = 0; k < P * Q; ++k) {
            uint32_t p = k % P, q = k / P;       // G col-major P×Q: G[q*P + p] = G(p,q)
            Gt[p * Q + q] = G[q * P + p];        // Gt col-major Q×P: Gt(q,p) = G(p,q)
        }
        thread::congruence_sym<T, Q, P, ACCUMULATE>(alpha, Gt, M, beta, C, scr);
    }
}

// ─── single-warp congruence / bilinear ───────────────────────────────────────
namespace warp {
    /**
     * @brief Single-warp symmetric congruence: `Q = alpha * Xᵀ·M·X + beta * Q`.
     *
     * Warp-per-problem analogue of `glass::congruence_sym`; one full 32-lane warp
     * forms `M·X` and the `Xᵀ·MX` contraction. See the block version for semantics.
     *
     * @tparam T,N,Kdim,ACCUMULATE  See glass::congruence_sym.
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param alpha,X,M,beta,Q,s_scratch  See glass::congruence_sym.
     */
    template <typename T, uint32_t N, uint32_t Kdim,
              bool ACCUMULATE = false, bool TRAILING_SYNC = true>
    __device__ void congruence_sym(T alpha, const T* X, const T* M, T beta, T* Q, T* s_scratch)
    {
        warp::gemm<T, N, Kdim, N>(static_cast<T>(1), const_cast<T*>(M), const_cast<T*>(X), s_scratch);
        __syncwarp();
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        detail::xtY_impl<T, N, Kdim, Kdim, true, ACCUMULATE>(lane, 32u, alpha, X, s_scratch, beta, Q);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }

    /**
     * @brief Single-warp general bilinear form: `R = alpha * Xᵀ·M·Y + beta * R`.
     *
     * Warp-per-problem analogue of `glass::bilinear`.
     *
     * @tparam T,N,P,Qd,ACCUMULATE  See glass::bilinear.
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param alpha,X,M,Y,beta,R,s_scratch  See glass::bilinear.
     */
    template <typename T, uint32_t N, uint32_t P, uint32_t Qd,
              bool ACCUMULATE = false, bool TRAILING_SYNC = true>
    __device__ void bilinear(T alpha, const T* X, const T* M, const T* Y, T beta, T* R, T* s_scratch)
    {
        warp::gemm<T, N, Qd, N>(static_cast<T>(1), const_cast<T*>(M), const_cast<T*>(Y), s_scratch);
        __syncwarp();
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        detail::xtY_impl<T, N, P, Qd, false, ACCUMULATE>(lane, 32u, alpha, X, s_scratch, beta, R);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }

    /**
     * @brief Single-warp accumulating congruence `C = alpha*G*M*Gᵀ + beta*C` (G is P×Q).
     *
     * Warp-per-problem analogue of `glass::congruence_accum`: one 32-lane warp
     * transposes `G` into scratch and defers to `warp::congruence_sym<Q,P>`.
     * See the block version for semantics.
     *
     * @tparam T,P,Q,ACCUMULATE  See glass::congruence_accum.
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param alpha,G,M,beta,C,s_scratch  See glass::congruence_accum.
     */
    template <typename T, uint32_t P, uint32_t Q,
              bool ACCUMULATE = false, bool TRAILING_SYNC = true>
    __device__ void congruence_accum(T alpha, const T* G, const T* M, T beta, T* C, T* s_scratch)
    {
        T* Gt  = s_scratch;
        T* scr = s_scratch + Q * P;
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        for (uint32_t k = lane; k < P * Q; k += 32u) {
            uint32_t p = k % P, q = k / P;
            Gt[p * Q + q] = G[q * P + p];
        }
        __syncwarp();
        warp::congruence_sym<T, Q, P, ACCUMULATE, TRAILING_SYNC>(alpha, Gt, M, beta, C, scr);
    }
}
