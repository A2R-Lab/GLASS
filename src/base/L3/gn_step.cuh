#pragma once
#include <cstdint>
#include <cstddef>

// ─── fused Gauss-Newton / Levenberg-Marquardt normal-equation step ───────────
//
// The damped-least-squares sibling of `riccati_gain` (the precedent that
// congruence + solve compositions belong in GLASS): for a tall Jacobian `J`
// (M x N, M >= N) and residual `r`, one call forms and solves the damped
// normal equations
//
//     (JᵀJ + lambda·diag(JᵀJ)) dq = Jᵀ r        (REG_DIAG = true, Marquardt)
//     (JᵀJ + lambda·I)         dq = Jᵀ r        (REG_DIAG = false, Levenberg)
//
// via the existing validated pieces — `syrk` (JᵀJ, all lanes spread over the
// N² accumulations), `gemv<TRANSPOSE>` (Jᵀr), and the flagged
// `posv<REGULARIZE, CHECK, REG_DIAG>` (shift + Cholesky + solve + non-PD
// report) — so every warp-packed GN/LM consumer (batched IK, calibration,
// small NLS) gets the exact composition HJCD-IK's solver hand-rolled,
// validated once. Every warp-packed GN/LM step is this shape; perf equals the
// hand-rolled build (the fused-vs-composed sweep's wash verdict) — the win is
// one tested entry point instead of N per-consumer helpers.
//
// SIGN CONVENTION: dq solves the normal equations exactly as written above —
// with the common residual convention `r = target − current`, `dq` is the
// UPDATE to ADD (`q += dq`); with `r = current − target`, negate. No hidden
// negation.
//
// Warp tier only for now: the warp-per-problem shape is where every named
// consumer forms it. Block/thread forms are a mechanical addition if a
// consumer materializes.

/**
 * @brief Scratch size in bytes for `warp::gn_step`'s normal-matrix buffer.
 *
 * @tparam T  Scalar type.
 * @tparam N  Parameter count (columns of J).
 * @return Bytes to allocate for `s_A`.
 */
template <typename T, uint32_t N>
__host__ __device__ constexpr std::size_t gn_step_scratch_bytes()
{
    return static_cast<std::size_t>(N) * N * sizeof(T);
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    /**
     * @brief Single-warp fused GN/LM step: form and solve
     *        `(JᵀJ + lambda·shift) dq = Jᵀ r`.
     *
     * One 32-lane warp: `s_A = JᵀJ` (`warp::syrk`, lanes spread the N²
     * accumulations), `dq = Jᵀr` (`warp::gemv<TRANSPOSE>`), then the flagged
     * `warp::posv` shifts the diagonal (Marquardt `lambda·diag(A)` when
     * `REG_DIAG`, else Levenberg `lambda·I`), factors, solves in place, and
     * reports a non-PD pivot through `s_fail` when `CHECK`. On return `dq`
     * holds the step, `s_A` holds the Cholesky factor of the shifted matrix
     * (clobbered scratch). `lambda = 0` with `CHECK = true` is plain
     * Gauss-Newton with a rank guard. Full 32 lanes required; independent
     * warps may run distinct problems concurrently.
     *
     * @tparam T  Scalar type (use `double` for ill-conditioned J).
     * @tparam M  Rows of `J` (residual length; M >= N for a full-rank system).
     * @tparam N  Columns of `J` (parameter count; A is N x N).
     * @tparam REGULARIZE  Apply the lambda shift (default true; false compiles it out).
     * @tparam CHECK  Report a non-PD pivot via `s_fail` (default true).
     * @tparam REG_DIAG  Shift by `lambda·diag(JᵀJ)` (Marquardt, default true)
     *                   instead of `lambda·I` (Levenberg).
     * @param J       Input Jacobian (M x N, column-major; read-only).
     * @param r       Input residual (M elements; read-only).
     * @param lambda  Damping factor (ignored when !REGULARIZE).
     * @param dq      Output step (N elements; warp-shared memory).
     * @param s_A     Scratch for the N x N normal matrix (warp-shared, N*N
     *                elements — see `gn_step_scratch_bytes`); clobbered.
     * @param s_fail  Non-PD flag when CHECK (set to 1 on failure, else 0).
     */
    template <typename T, uint32_t M, uint32_t N,
              bool REGULARIZE = true, bool CHECK = true, bool REG_DIAG = true>
    __device__ void gn_step(const T *J, const T *r, T lambda, T *dq,
                            T *s_A, int *s_fail = nullptr)
    {
        syrk<T, N, M, FillMode::Full, /*TRANSPOSE=*/true>(static_cast<T>(1), J, s_A);
        // gemv's trailing __syncwarp also fences the syrk lanes' A writes.
        gemv<T, M, N, /*TRANSPOSE=*/true>(static_cast<T>(1), J, r, static_cast<T>(0), dq);
        posv<T, N, 1, REGULARIZE, CHECK, REG_DIAG>(s_A, dq, lambda, s_fail);
    }
}
