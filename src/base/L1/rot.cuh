#pragma once
#include "../barrier.cuh"
#include <cstdint>

/**
 * @file rot.cuh
 * @brief Givens plane rotation: apply (`rot`, BLAS SROT/DROT) and generate
 *        (`rotg`, BLAS SROTG/DROTG).
 *
 * `rot` applies the rotation `[c s; -s c]` to a vector pair in place:
 * `x[i] = c*x[i] + s*y[i]`, `y[i] = c*y[i] - s*x_old[i]`. Each element index
 * is owned by exactly one thread, which reads BOTH old values into registers
 * before writing — no staging, no cross-thread hazard, trivially thread-count
 * invariant. Block + `warp::`.
 *
 * `rotg` is a `__host__ __device__` SCALAR helper (one caller, no
 * cooperation): given `(a, b)` it computes `(c, s, r)` with
 * `c*a + s*b = r` and `c*b - s*a = 0` (so applying `rot` with the returned
 * `(c, s)` to the pair `(a, b)` zeroes `b`), using the overflow-safe scaled
 * form of the reference BLAS DROTG.
 */

// shared body: in-place Givens apply on the pair (x, y). Each thread owns
// disjoint indices and reads both olds into registers first, so no barrier is
// needed before the writes.
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void rot_impl(Bar bar, uint32_t n, T *x, T *y, T c, T s)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t i = rank; i < n; i += size) {
        const T xi = x[i], yi = y[i];
        x[i] = c*xi + s*yi;
        y[i] = c*yi - s*xi;
    }
    if constexpr (TRAILING_SYNC) bar.sync();
}

/**
 * @brief Apply a Givens plane rotation to a vector pair in place (ROT).
 *
 * `x[i] = c*x[i] + s*y[i]`, `y[i] = c*y[i] - s*x_old[i]` for all `i` — the
 * BLAS SROT/DROT update. NumPy equivalent:
 * `x, y = c*x + s*y, c*y - s*x`.
 *
 * @tparam T             Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC Emit a trailing `__syncthreads()` (default true).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n`.
 * @param y  In/out vector of length `n`.
 * @param c  Rotation cosine.
 * @param s  Rotation sine.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void rot(uint32_t n, T *x, T *y, T c, T s)
{
    rot_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, x, y, c, s);
}

/**
 * @brief Apply a Givens plane rotation in place (ROT), compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `x, y = c*x + s*y, c*y - s*x`.
 *
 * @tparam T             Scalar type (e.g. `float`, `double`).
 * @tparam N             Number of elements (compile-time constant).
 * @tparam TRAILING_SYNC Emit a trailing `__syncthreads()` (default true).
 * @param x  In/out vector of length `N`.
 * @param y  In/out vector of length `N`.
 * @param c  Rotation cosine.
 * @param s  Rotation sine.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void rot(T *x, T *y, T c, T s)
{
    rot_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, x, y, c, s);
}

/**
 * @brief Generate a Givens plane rotation (ROTG) — `__host__ __device__` scalar helper.
 *
 * Given `(a, b)`, computes `(c, s, r)` such that
 * `[c s; -s c] @ [a; b] = [r; 0]` — i.e. `c = a/r`, `s = b/r`,
 * `r = ±sqrt(a² + b²)` with the reference-BLAS DROTG sign convention (the
 * sign of the larger-magnitude input) and its overflow-safe scaling
 * (`scale = |a| + |b|`; the squares are formed on `a/scale`, `b/scale`).
 * `(a, b) = (0, 0)` returns `(c, s, r) = (1, 0, 0)`. Scalar: call from one
 * thread (or redundantly from all — it is deterministic), or on the host.
 * SciPy equivalent: `(c, s), r = scipy.linalg.blas.drotg(a, b), np.hypot(a, b)`
 * (up to the sign convention).
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param a  First component.
 * @param b  Second component (the one the rotation annihilates).
 * @param c  Output rotation cosine.
 * @param s  Output rotation sine.
 * @param r  Output rotated magnitude (`c*a + s*b`).
 */
template <typename T>
__host__ __device__ void rotg(T a, T b, T &c, T &s, T &r)
{
    const T absa = (a < static_cast<T>(0)) ? -a : a;
    const T absb = (b < static_cast<T>(0)) ? -b : b;
    const T scale = absa + absb;
    if (scale == static_cast<T>(0)) {
        c = static_cast<T>(1);
        s = static_cast<T>(0);
        r = static_cast<T>(0);
        return;
    }
    const T as = a / scale, bs = b / scale;
    T rr = scale * sqrt(as*as + bs*bs);
    // reference-BLAS sign: r carries the sign of the larger-magnitude input.
    const T roe = (absa > absb) ? a : b;
    rr = (roe < static_cast<T>(0)) ? -rr : rr;
    c = a / rr;
    s = b / rr;
    r = rr;
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    /**
     * @brief Apply a Givens plane rotation within one warp (ROT), single-warp.
     *
     * One 32-lane warp applies `x[i] = c*x[i] + s*y[i]`,
     * `y[i] = c*y[i] - s*x_old[i]` (lane-strided; each index owned by one
     * lane, both olds read into registers first). Trailing `__syncwarp()`.
     * NumPy equivalent: `x, y = c*x + s*y, c*y - s*x`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  In/out vector of length `n`.
     * @param y  In/out vector of length `n`.
     * @param c  Rotation cosine.
     * @param s  Rotation sine.
     */
    template <typename T>
    __device__ void rot(uint32_t n, T *x, T *y, T c, T s)
    {
        uint32_t lane = (flat_rank()) & 31;
        for (uint32_t i = lane; i < n; i += 32) {
            const T xi = x[i], yi = y[i];
            x[i] = c*xi + s*yi;
            y[i] = c*yi - s*xi;
        }
        __syncwarp();
    }

    /**
     * @brief Apply a Givens plane rotation within one warp (ROT), compile-time size.
     *
     * Compile-time-`N` overload of the single-warp rot. NumPy equivalent:
     * `x, y = c*x + s*y, c*y - s*x`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  In/out vector of length `N`.
     * @param y  In/out vector of length `N`.
     * @param c  Rotation cosine.
     * @param s  Rotation sine.
     */
    template <typename T, uint32_t N>
    __device__ void rot(T *x, T *y, T c, T s)
    {
        rot<T>(N, x, y, c, s);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    // Single-thread ROT: one THREAD applies the whole rotation, reusing the
    // shared `rot_impl` body with ThreadBarrier{rank=0, size=1, no-op sync} —
    // the same algorithm as the block op degenerated to one thread. (`rotg` is
    // already a scalar `__host__ __device__` helper; call it directly.)

    /**
     * @brief Apply a Givens plane rotation on one thread (ROT), single-thread.
     *
     * One thread applies `x[i] = c*x[i] + s*y[i]`, `y[i] = c*y[i] - s*x_old[i]`
     * serially over the `n` elements (both olds read into registers first). No
     * shared scratch, no shuffles, no barriers, no `threadIdx` read; operands
     * may be thread-local register arrays. NumPy equivalent:
     * `x, y = c*x + s*y, c*y - s*x`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  In/out vector of length `n`.
     * @param y  In/out vector of length `n`.
     * @param c  Rotation cosine.
     * @param s  Rotation sine.
     */
    template <typename T>
    __device__ void rot(uint32_t n, T *x, T *y, T c, T s)
    {
        rot_impl<ThreadBarrier, T>(ThreadBarrier{}, n, x, y, c, s);
    }

    /**
     * @brief Apply a Givens plane rotation on one thread (ROT), single-thread, compile-time size.
     *
     * Compile-time-`N` overload; the trip count folds and the loop unrolls, so
     * `x` and `y` may be thread-local register arrays. NumPy equivalent:
     * `x, y = c*x + s*y, c*y - s*x`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  In/out vector of length `N`.
     * @param y  In/out vector of length `N`.
     * @param c  Rotation cosine.
     * @param s  Rotation sine.
     */
    template <typename T, uint32_t N>
    __device__ void rot(T *x, T *y, T c, T s)
    {
        rot_impl<ThreadBarrier, T>(ThreadBarrier{}, N, x, y, c, s);
    }
}
