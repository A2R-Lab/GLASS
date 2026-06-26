#pragma once
#include <cstdint>

// ── iamax internal combine helpers ───────────────────────────────────────────
// An argmax over |x| is carried as an (absval, index) pair. The winner is the
// element with strictly-greater |x|; on EQUAL |x| the LOWER index wins (the
// BLAS i_amax tie-break rule). Making this tie-break deterministic at EVERY
// combine step is the MECHANISM of thread-count invariance: regardless of how
// the strided ranges, warp shuffles, and scratch-tree merges interleave, the
// lexicographic-min-of (-|x|, index) winner is unique, so the answer cannot
// depend on the block size. Inactive/empty lanes seed (key=0, idx=UINT32_MAX)
// so they can never win a tie (and an all-zero vector therefore returns 0).
// NOTE: like the other base/L1 headers (reduce.cuh's high_speed::/low_memory::),
// these are written at FILE SCOPE — glass.cuh #includes them inside `namespace
// glass {`, so `iamax`, `iamax_detail`, `low_memory`, `high_speed` all land
// under `glass::`. Refer to the detail helper with a bare `iamax_detail::`.
namespace iamax_detail {

// Fold candidate (ckey,cidx) into the running best (key,idx) in place.
// Strictly-greater |x| wins; equal |x| keeps the lower index. NaN candidates
// compare false on both branches and are never selected (skip-NaN policy).
template <typename T>
__device__ __forceinline__ void combine(T &key, uint32_t &idx, T ckey, uint32_t cidx) {
    if (ckey > key || (ckey == key && cidx < idx)) { key = ckey; idx = cidx; }
}

} // namespace iamax_detail

/**
 * @brief Index of the max-absolute-value element (BLAS i_amax), into `out[0]`.
 *
 * Non-destructive: `x` is read-only and never clobbered. Computes the block-wide
 * argmax over `|x|` (default `threadIdx`-strided variant) and writes the winning
 * index (`uint32_t`) to `out[0]`. NumPy equivalent: `int(np.argmax(np.abs(x)))`.
 *
 * @par Tie-break
 * On EQUAL absolute value the LOWER index wins (the BLAS rule). This tie-break
 * is applied at every combine step, which is what makes the result identical for
 * any block size (1 thread, a partial warp, or many warps).
 *
 * @par NaN policy
 * NaN inputs are SKIPPED (IEEE compares are false, so a NaN is never selected).
 * This DIVERGES from `np.argmax(np.abs(x))`, which propagates NaN; oracle tests
 * must exclude NaN inputs. An all-zero vector returns index `0`.
 *
 * @par Scratch sizing
 * This variant uses no shared scratch (a serial pass on thread 0). For the
 * warp-shuffle `high_speed::` variant size scratch via `iamax_hs_scratch_bytes`.
 *
 * The routine ends on a trailing `__syncthreads()`, so `out[0]` is block-visible
 * on return.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n    Number of elements.
 * @param x    Read-only input vector of length `n` (not modified).
 * @param out  Output: `out[0]` receives the argmax index.
 * @param s_scratch  Shared scratch of `iamax_scratch_bytes` elements (key + index lanes).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void iamax(uint32_t n, const T *x, uint32_t *out, T *s_scratch) {
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    // s_scratch layout: [0..size) abs-keys, then index lanes packed after.
    T *s_key = s_scratch;
    uint32_t *s_idx = reinterpret_cast<uint32_t *>(s_scratch + size);

    // Per-thread local argmax over the strided range (deterministic tie-break).
    T best_key = static_cast<T>(0);
    uint32_t best_idx = UINT32_MAX;
    for (uint32_t i = rank; i < n; i += size) {
        iamax_detail::combine(best_key, best_idx, abs(x[i]), i);
    }
    s_key[rank] = best_key;
    s_idx[rank] = best_idx;
    __syncthreads();

    // Thread 0 serially folds the per-thread winners (lower-index tie-break).
    if (rank == 0) {
        T key = s_key[0];
        uint32_t idx = s_idx[0];
        uint32_t lim = (size < n) ? size : n;
        for (uint32_t i = 1; i < lim; i++) {
            iamax_detail::combine(key, idx, s_key[i], s_idx[i]);
        }
        // All-zero (or fully inactive) vector → no element beat the (0,MAX) seed.
        out[0] = (idx == UINT32_MAX) ? 0u : idx;
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Index of the max-absolute-value element (BLAS i_amax), compile-time size.
 *
 * Compile-time-`N` overload of the default i_amax; non-destructive, writes the
 * winning index to `out[0]`. NumPy equivalent: `int(np.argmax(np.abs(x)))`.
 * Tie-break (lower index wins on equal `|x|`), NaN-skip policy, and the trailing
 * `__syncthreads()` are as in the runtime-`n` overload.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x    Read-only input vector of length `N` (not modified).
 * @param out  Output: `out[0]` receives the argmax index.
 * @param s_scratch  Shared scratch of `iamax_scratch_bytes` elements.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void iamax(const T *x, uint32_t *out, T *s_scratch) {
    iamax<T, TRAILING_SYNC>(N, x, out, s_scratch);
}

/**
 * @brief i_amax with the max absolute value also returned: `out_val[0] = max|x|`.
 *
 * Like `iamax` but additionally writes the winning absolute value to `out_val[0]`
 * (`max|x|`, NumPy `np.max(np.abs(x))` over the non-NaN entries). Non-destructive;
 * lower-index tie-break; NaN skipped; trailing `__syncthreads()`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n        Number of elements.
 * @param x        Read-only input vector of length `n` (not modified).
 * @param out      Output: `out[0]` receives the argmax index.
 * @param out_val  Output: `out_val[0]` receives `max|x|`.
 * @param s_scratch   Shared scratch of `iamax_scratch_bytes` elements.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void iamax(uint32_t n, const T *x, uint32_t *out, T *out_val, T *s_scratch) {
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T *s_key = s_scratch;
    uint32_t *s_idx = reinterpret_cast<uint32_t *>(s_scratch + size);

    T best_key = static_cast<T>(0);
    uint32_t best_idx = UINT32_MAX;
    for (uint32_t i = rank; i < n; i += size) {
        iamax_detail::combine(best_key, best_idx, abs(x[i]), i);
    }
    s_key[rank] = best_key;
    s_idx[rank] = best_idx;
    __syncthreads();

    if (rank == 0) {
        T key = s_key[0];
        uint32_t idx = s_idx[0];
        uint32_t lim = (size < n) ? size : n;
        for (uint32_t i = 1; i < lim; i++) {
            iamax_detail::combine(key, idx, s_key[i], s_idx[i]);
        }
        out[0] = (idx == UINT32_MAX) ? 0u : idx;
        out_val[0] = key;
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief i_amax with `max|x|` returned, compile-time size.
 *
 * Compile-time-`N` overload of the value-returning i_amax. NumPy equivalents:
 * `out[0] = int(np.argmax(np.abs(x)))`, `out_val[0] = np.max(np.abs(x))`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x        Read-only input vector of length `N` (not modified).
 * @param out      Output: `out[0]` receives the argmax index.
 * @param out_val  Output: `out_val[0]` receives `max|x|`.
 * @param s_scratch   Shared scratch of `iamax_scratch_bytes` elements.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void iamax(const T *x, uint32_t *out, T *out_val, T *s_scratch) {
    iamax<T, TRAILING_SYNC>(N, x, out, out_val, s_scratch);
}

/**
 * @brief Shared-scratch size in bytes for the default/`low_memory` `iamax`.
 *
 * The default variant stores one abs-key (`T`) and one index (`uint32_t`) per
 * thread. Allocate `iamax_scratch_bytes<T>(block_threads)` bytes of `T` for
 * `s_scratch` (the index lanes are packed into the same buffer after the keys).
 *
 * @tparam T  Scalar type.
 * @param block_threads  Number of threads in the launching block.
 * @return Bytes to allocate for `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t iamax_scratch_bytes(uint32_t block_threads) {
    return (static_cast<std::size_t>(block_threads + (block_threads * sizeof(uint32_t) + sizeof(T) - 1) / sizeof(T))) * sizeof(T);
}

/**
 * @brief Shared-scratch size in bytes for `high_speed::iamax`.
 *
 * The warp-shuffle variant reduces within each warp in registers and combines
 * across warps through scratch, needing one (key,index) slot per warp:
 * `ceil(block_threads/32)` of each. Allocate `iamax_hs_scratch_bytes<T>(block_threads)`
 * elements of `T` for its `s_scratch`.
 *
 * @tparam T  Scalar type.
 * @param block_threads  Number of threads in the launching block.
 * @return Bytes to allocate for the `high_speed::iamax` scratch.
 */
template <typename T>
__host__ __device__ constexpr std::size_t iamax_hs_scratch_bytes(uint32_t block_threads) {
    return (static_cast<std::size_t>(((block_threads + 31) / 32)
         + (((block_threads + 31) / 32) * sizeof(uint32_t) + sizeof(T) - 1) / sizeof(T))) * sizeof(T);
}

namespace low_memory {
    /**
     * @brief i_amax, low-memory variant (no scratch).
     *
     * Thread 0 serially scans `x` for the argmax over `|x|`, writing the index to
     * `out[0]`; all other threads idle. Non-destructive. NumPy equivalent:
     * `int(np.argmax(np.abs(x)))`. Lower-index tie-break on equal `|x|`; NaN
     * skipped (diverges from `np.argmax`, exclude NaN in tests); all-zero → 0.
     * Ends on a trailing `__syncthreads()` so `out[0]` is block-visible.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n    Number of elements.
     * @param x    Read-only input vector of length `n` (not modified).
     * @param out  Output: `out[0]` receives the argmax index.
     */
    template <typename T, bool TRAILING_SYNC = true>
    __device__ void iamax(uint32_t n, const T *x, uint32_t *out) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            T key = static_cast<T>(0);
            uint32_t idx = UINT32_MAX;
            for (uint32_t i = 0; i < n; i++) {
                iamax_detail::combine(key, idx, abs(x[i]), i);
            }
            out[0] = (idx == UINT32_MAX) ? 0u : idx;
        }
        if constexpr (TRAILING_SYNC) __syncthreads();
    }

    /**
     * @brief i_amax, low-memory variant, compile-time size.
     *
     * Compile-time-`N` overload of the serial i_amax. NumPy equivalent:
     * `int(np.argmax(np.abs(x)))`. Same tie-break / NaN policy as the runtime form.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x    Read-only input vector of length `N` (not modified).
     * @param out  Output: `out[0]` receives the argmax index.
     */
    template <typename T, uint32_t N, bool TRAILING_SYNC = true>
    __device__ void iamax(const T *x, uint32_t *out) {
        iamax<T, TRAILING_SYNC>(N, x, out);
    }

    /**
     * @brief i_amax + `max|x|`, low-memory variant (no scratch).
     *
     * As above but also writes `out_val[0] = max|x|`. Non-destructive; thread 0
     * scans serially; lower-index tie-break; NaN skipped; all-zero → 0.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n        Number of elements.
     * @param x        Read-only input vector of length `n` (not modified).
     * @param out      Output: `out[0]` receives the argmax index.
     * @param out_val  Output: `out_val[0]` receives `max|x|`.
     */
    template <typename T, bool TRAILING_SYNC = true>
    __device__ void iamax(uint32_t n, const T *x, uint32_t *out, T *out_val) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            T key = static_cast<T>(0);
            uint32_t idx = UINT32_MAX;
            for (uint32_t i = 0; i < n; i++) {
                iamax_detail::combine(key, idx, abs(x[i]), i);
            }
            out[0] = (idx == UINT32_MAX) ? 0u : idx;
            out_val[0] = key;
        }
        if constexpr (TRAILING_SYNC) __syncthreads();
    }

    /**
     * @brief i_amax + `max|x|`, low-memory variant, compile-time size.
     *
     * Compile-time-`N` value-returning serial overload.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x        Read-only input vector of length `N` (not modified).
     * @param out      Output: `out[0]` receives the argmax index.
     * @param out_val  Output: `out_val[0]` receives `max|x|`.
     */
    template <typename T, uint32_t N, bool TRAILING_SYNC = true>
    __device__ void iamax(const T *x, uint32_t *out, T *out_val) {
        iamax<T, TRAILING_SYNC>(N, x, out, out_val);
    }
}

namespace high_speed {
    /**
     * @brief i_amax, warp-shuffle variant: index of `max|x|` into `out[0]`.
     *
     * Each thread forms a strided per-thread argmax in registers, the warp folds
     * it with `__shfl_down_sync` (carrying the (key,index) pair), and the per-warp
     * winners are combined through `s_scratch`. Non-destructive. NumPy equivalent:
     * `int(np.argmax(np.abs(x)))`.
     *
     * @par Tie-break / NaN
     * Lower index wins on equal `|x|` at every shuffle/scratch combine (so the
     * result is block-size invariant). NaN is skipped (diverges from `np.argmax`;
     * exclude NaN in tests). All-zero vector → index `0`.
     *
     * @par Scratch
     * `s_scratch` must hold `iamax_hs_scratch_bytes<T>(blockDim)` elements (one (key,index)
     * slot per warp). Ends on a trailing `__syncthreads()`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n       Number of elements.
     * @param x       Read-only input vector of length `n` (not modified).
     * @param out     Output: `out[0]` receives the argmax index.
     * @param s_scratch  Shared scratch sized by `iamax_hs_scratch_bytes<T>(blockDim)`.
     */
    template <typename T, bool TRAILING_SYNC = true>
    __device__ void iamax(uint32_t n, const T *x, uint32_t *out, T *s_scratch) {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        uint32_t nw = (size + 31) / 32;
        T *s_key = s_scratch;
        uint32_t *s_idx = reinterpret_cast<uint32_t *>(s_scratch + nw);

        // Per-thread strided argmax.
        T key = static_cast<T>(0);
        uint32_t idx = UINT32_MAX;
        for (uint32_t i = rank; i < n; i += size) {
            iamax_detail::combine(key, idx, abs(x[i]), i);
        }
        // Warp-shuffle fold of the (key,index) pair (lower-index tie-break).
        for (int off = 16; off > 0; off >>= 1) {
            T okey = __shfl_down_sync(0xffffffff, key, off);
            uint32_t oidx = __shfl_down_sync(0xffffffff, idx, off);
            iamax_detail::combine(key, idx, okey, oidx);
        }
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) { s_key[warp] = key; s_idx[warp] = idx; }
        __syncthreads();

        // Lane 0..nw-1 of warp 0 fold the per-warp winners.
        if (rank < 32) {
            key = (rank < nw) ? s_key[rank] : static_cast<T>(0);
            idx = (rank < nw) ? s_idx[rank] : UINT32_MAX;
            for (int off = 16; off > 0; off >>= 1) {
                T okey = __shfl_down_sync(0xffffffff, key, off);
                uint32_t oidx = __shfl_down_sync(0xffffffff, idx, off);
                iamax_detail::combine(key, idx, okey, oidx);
            }
            if (rank == 0) out[0] = (idx == UINT32_MAX) ? 0u : idx;
        }
        if constexpr (TRAILING_SYNC) __syncthreads();
    }

    /**
     * @brief i_amax, warp-shuffle variant, compile-time size.
     *
     * Compile-time-`N` overload of the warp-shuffle i_amax. NumPy equivalent:
     * `int(np.argmax(np.abs(x)))`. Scratch via `iamax_hs_scratch_bytes<T>(blockDim)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x       Read-only input vector of length `N` (not modified).
     * @param out     Output: `out[0]` receives the argmax index.
     * @param s_scratch  Shared scratch sized by `iamax_hs_scratch_bytes<T>(blockDim)`.
     */
    template <typename T, uint32_t N, bool TRAILING_SYNC = true>
    __device__ void iamax(const T *x, uint32_t *out, T *s_scratch) {
        iamax<T, TRAILING_SYNC>(N, x, out, s_scratch);
    }

    /**
     * @brief i_amax + `max|x|`, warp-shuffle variant.
     *
     * As the warp-shuffle `iamax` but also writes `out_val[0] = max|x|`.
     * Non-destructive; lower-index tie-break; NaN skipped; all-zero → 0.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n        Number of elements.
     * @param x        Read-only input vector of length `n` (not modified).
     * @param out      Output: `out[0]` receives the argmax index.
     * @param out_val  Output: `out_val[0]` receives `max|x|`.
     * @param s_scratch   Shared scratch sized by `iamax_hs_scratch_bytes<T>(blockDim)`.
     */
    template <typename T, bool TRAILING_SYNC = true>
    __device__ void iamax(uint32_t n, const T *x, uint32_t *out, T *out_val, T *s_scratch) {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        uint32_t nw = (size + 31) / 32;
        T *s_key = s_scratch;
        uint32_t *s_idx = reinterpret_cast<uint32_t *>(s_scratch + nw);

        T key = static_cast<T>(0);
        uint32_t idx = UINT32_MAX;
        for (uint32_t i = rank; i < n; i += size) {
            iamax_detail::combine(key, idx, abs(x[i]), i);
        }
        for (int off = 16; off > 0; off >>= 1) {
            T okey = __shfl_down_sync(0xffffffff, key, off);
            uint32_t oidx = __shfl_down_sync(0xffffffff, idx, off);
            iamax_detail::combine(key, idx, okey, oidx);
        }
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) { s_key[warp] = key; s_idx[warp] = idx; }
        __syncthreads();

        if (rank < 32) {
            key = (rank < nw) ? s_key[rank] : static_cast<T>(0);
            idx = (rank < nw) ? s_idx[rank] : UINT32_MAX;
            for (int off = 16; off > 0; off >>= 1) {
                T okey = __shfl_down_sync(0xffffffff, key, off);
                uint32_t oidx = __shfl_down_sync(0xffffffff, idx, off);
                iamax_detail::combine(key, idx, okey, oidx);
            }
            if (rank == 0) { out[0] = (idx == UINT32_MAX) ? 0u : idx; out_val[0] = key; }
        }
        if constexpr (TRAILING_SYNC) __syncthreads();
    }

    /**
     * @brief i_amax + `max|x|`, warp-shuffle variant, compile-time size.
     *
     * Compile-time-`N` value-returning warp-shuffle overload. NumPy equivalents:
     * `out[0] = int(np.argmax(np.abs(x)))`, `out_val[0] = np.max(np.abs(x))`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x        Read-only input vector of length `N` (not modified).
     * @param out      Output: `out[0]` receives the argmax index.
     * @param out_val  Output: `out_val[0]` receives `max|x|`.
     * @param s_scratch   Shared scratch sized by `iamax_hs_scratch_bytes<T>(blockDim)`.
     */
    template <typename T, uint32_t N, bool TRAILING_SYNC = true>
    __device__ void iamax(const T *x, uint32_t *out, T *out_val, T *s_scratch) {
        iamax<T, TRAILING_SYNC>(N, x, out, out_val, s_scratch);
    }
}

namespace warp {
    // Single-warp i_amax: one 32-lane warp owns the argmax, the winning index is
    // reduced with __shfl_down_sync (carrying the (abskey,index) pair) and
    // broadcast back to every lane with __shfl_sync — no shared scratch, no
    // __syncthreads, no inter-warp combine. For warp-per-problem kernels that pack
    // many independent vectors into one block (one 32-lane warp per vector). The
    // (key,idx) pair and the lower-index tie-break (iamax_detail::combine) are the
    // SAME mechanism as the block-scoped iamax — applied at every shuffle step so
    // the answer is lane/order independent (and identical to the block result).

    /**
     * @brief Index of `max|x|` (BLAS i_amax) within ONE warp, returned on every lane.
     *
     * A single 32-lane warp computes `argmax(|x|)` and returns it (register return,
     * broadcast to all lanes via `__shfl_sync`) — there is NO `out[]` write and NO
     * shared scratch. Each lane forms a strided per-lane `(abskey,index)` argmax over
     * `for (i = lane; i < n; i += 32)`, then the warp folds the pair with
     * `__shfl_down_sync` (lower-index tie-break at EVERY step) and broadcasts lane 0's
     * index. NumPy equivalent: `int(np.argmax(np.abs(x)))`.
     *
     * @par Full-warp requirement
     * This routine assumes a FULL 32-lane warp is active (mask `0xffffffff`); it must
     * be called by all 32 lanes of the warp. Inactive lanes (those whose `lane >= n`,
     * i.e. that never enter the strided loop) seed `key = 0, idx = UINT32_MAX`, so they
     * never win a tie. An all-zero vector therefore returns index `0`.
     *
     * @par Multi-warp independence
     * Shared-free and `__syncthreads`-free: many warps may run this concurrently in one
     * block, each on its own `x`, with no cross-warp interference. The result is also
     * identical to the block-scoped `glass::iamax` (same `(key,idx)` lower-index combine).
     *
     * @par Tie-break / NaN
     * On EQUAL `|x|` the LOWER index wins (the BLAS rule), applied at every shuffle
     * combine — so the result is lane/order independent. NaN inputs are SKIPPED (IEEE
     * compares are false, so a NaN is never selected); this DIVERGES from
     * `np.argmax(np.abs(x))`, which propagates NaN — oracle tests must exclude NaN.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  Read-only input vector of length `n` (not modified).
     * @return The argmax index (`uint32_t`), identical on every lane.
     */
    template <typename T>
    __device__ uint32_t iamax(uint32_t n, const T *x) {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        // Per-lane strided argmax over |x| (deterministic lower-index tie-break).
        T key = static_cast<T>(0);
        uint32_t idx = UINT32_MAX;
        for (uint32_t i = lane; i < n; i += 32) {
            iamax_detail::combine(key, idx, abs(x[i]), i);
        }
        // Warp-shuffle fold of the (key,index) pair (lower-index tie-break at each step).
        for (int off = 16; off > 0; off >>= 1) {
            T okey = __shfl_down_sync(0xffffffffu, key, off);
            uint32_t oidx = __shfl_down_sync(0xffffffffu, idx, off);
            iamax_detail::combine(key, idx, okey, oidx);
        }
        // All-zero / fully-inactive vector → no element beat the (0,MAX) seed.
        idx = (idx == UINT32_MAX) ? 0u : idx;
        // Broadcast lane 0's winning index to every lane (register broadcast, §1g).
        return __shfl_sync(0xffffffffu, idx, 0);
    }

    /**
     * @brief Index of `max|x|` (BLAS i_amax) within ONE warp, compile-time size.
     *
     * Compile-time-`N` overload of the single-warp i_amax. Returns `argmax(|x|)` on
     * every lane (register broadcast). NumPy equivalent: `int(np.argmax(np.abs(x)))`.
     * Same full-warp requirement, multi-warp independence, lower-index tie-break, and
     * NaN-skip policy as the runtime-`n` overload.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  Read-only input vector of length `N` (not modified).
     * @return The argmax index (`uint32_t`), identical on every lane.
     */
    template <typename T, uint32_t N>
    __device__ uint32_t iamax(const T *x) {
        return iamax<T>(N, x);
    }
}
