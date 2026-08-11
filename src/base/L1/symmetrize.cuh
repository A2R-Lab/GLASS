#pragma once
#include "../barrier.cuh"
#include <cstdint>

/**
 * @file symmetrize.cuh
 * @brief In-place symmetrization `A := 0.5*(A + Aᵀ)` for a square matrix.
 *
 * Numerically enforces symmetry on an n×n column-major matrix — the standard
 * cleanup after a product chain (e.g. a Schur-complement gemm sequence) whose
 * result is symmetric in exact arithmetic but drifts under floating point.
 * MPCGPU hand-rolls exactly this after its Schur assembly (without it, CG
 * stagnates on the slightly-asymmetric operator); this is that loop, once.
 *
 * Each strictly-lower (i>j) pair owner reads BOTH mirror elements and writes
 * BOTH — every (i,j)/(j,i) pair is touched by exactly one thread, the diagonal
 * is untouched, so there is no cross-thread hazard and the op is trivially
 * thread-count invariant. Block + `warp::` + `cgrps::`.
 */

// shared body: in-place A := 0.5*(A + Aᵀ), n×n column-major. Threads stride the
// full n*n index space and act only on strictly-lower entries (r > c), so each
// mirror pair has a single owner — no barrier needed before the writes.
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void symmetrize_impl(Bar bar, uint32_t n, T *A)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t idx = rank; idx < n*n; idx += size) {
        uint32_t r = idx % n, c = idx / n;
        if (r > c) {
            T v = static_cast<T>(0.5) * (A[idx] + A[c + r*n]);
            A[idx] = v;
            A[c + r*n] = v;
        }
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Symmetrize a square matrix in place: `A = 0.5*(A + Aᵀ)`.
 *
 * Averages each strictly-off-diagonal mirror pair of the `n×n` column-major
 * matrix `A`; the diagonal is untouched. NumPy equivalent:
 * `A = 0.5*(A + A.T)`.
 *
 * @tparam T             Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC Emit a trailing `__syncthreads()` (default true).
 * @param n  Matrix dimension (number of rows/columns).
 * @param A  In/out matrix of `n*n` elements (column-major).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void symmetrize(uint32_t n, T *A)
{
    symmetrize_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, A);
}

/**
 * @brief Symmetrize in place: `A = 0.5*(A + Aᵀ)`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `A = 0.5*(A + A.T)`.
 *
 * @tparam T             Scalar type (e.g. `float`, `double`).
 * @tparam N             Matrix dimension (compile-time constant).
 * @tparam TRAILING_SYNC Emit a trailing `__syncthreads()` (default true).
 * @param A  In/out matrix of `N*N` elements (column-major).
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void symmetrize(T *A)
{
    symmetrize_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, A);
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    // Single-warp SYMMETRIZE: one 32-lane warp strides the n*n index space
    // (lane k handles flat indices k, k+32, …) acting on strictly-lower entries.
    // Each mirror pair has a single owning lane, so no inter-lane communication
    // is needed; a trailing __syncwarp() orders the writes for the caller.
    // Full 32 lanes required.

    /**
     * @brief Symmetrize within one warp: `A = 0.5*(A + Aᵀ)`, single-warp.
     *
     * One 32-lane warp averages the mirror pairs of the `n×n` column-major
     * matrix in place; the diagonal is untouched. Each pair owned by exactly
     * one lane; trailing `__syncwarp()`. NumPy equivalent: `A = 0.5*(A + A.T)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Matrix dimension (number of rows/columns).
     * @param A  In/out matrix of `n*n` elements (column-major).
     */
    template <typename T>
    __device__ void symmetrize(uint32_t n, T *A)
    {
        uint32_t lane = (flat_rank()) & 31;
        for (uint32_t idx = lane; idx < n*n; idx += 32) {
            uint32_t r = idx % n, c = idx / n;
            if (r > c) {
                T v = static_cast<T>(0.5) * (A[idx] + A[c + r*n]);
                A[idx] = v;
                A[c + r*n] = v;
            }
        }
        __syncwarp();
    }

    /**
     * @brief Symmetrize within one warp: `A = 0.5*(A + Aᵀ)`, compile-time size.
     *
     * Compile-time-`N` overload of the single-warp symmetrize. NumPy
     * equivalent: `A = 0.5*(A + A.T)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Matrix dimension (compile-time constant).
     * @param A  In/out matrix of `N*N` elements (column-major).
     */
    template <typename T, uint32_t N>
    __device__ void symmetrize(T *A)
    {
        symmetrize<T>(N, A);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    // Single-thread SYMMETRIZE: one THREAD averages every mirror pair, reusing
    // the shared `symmetrize_impl` body with ThreadBarrier{rank=0, size=1, no-op
    // sync} — the same algorithm as the block op degenerated to one thread.

    /**
     * @brief Symmetrize on one thread: `A = 0.5*(A + Aᵀ)`, single-thread.
     *
     * One thread averages each strictly-off-diagonal mirror pair of the `n×n`
     * column-major matrix serially; the diagonal is untouched. No shared
     * scratch, no shuffles, no barriers, no `threadIdx` read; `A` may be a
     * thread-local register array. NumPy equivalent: `A = 0.5*(A + A.T)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Matrix dimension (number of rows/columns).
     * @param A  In/out matrix of `n*n` elements (column-major).
     */
    template <typename T>
    __device__ void symmetrize(uint32_t n, T *A)
    {
        symmetrize_impl<ThreadBarrier, T>(ThreadBarrier{}, n, A);
    }

    /**
     * @brief Symmetrize on one thread: `A = 0.5*(A + Aᵀ)`, single-thread, compile-time size.
     *
     * Compile-time-`N` overload; the trip count folds and the loop unrolls, so
     * `A` may be a thread-local register array. NumPy equivalent:
     * `A = 0.5*(A + A.T)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Matrix dimension (compile-time constant).
     * @param A  In/out matrix of `N*N` elements (column-major).
     */
    template <typename T, uint32_t N>
    __device__ void symmetrize(T *A)
    {
        symmetrize_impl<ThreadBarrier, T>(ThreadBarrier{}, N, A);
    }
}
