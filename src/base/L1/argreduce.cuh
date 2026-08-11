#pragma once
#include <cstdint>

// ─── argmax / argmin with index payload (L1) ─────────────────────────────────
//
// The SIGNED-value argreductions (`np.argmax` / `np.argmin`) that generalize
// the BLAS `iamax` (which reduces over |x|): best-rollout selection in
// sampling planners, best-cost line-search index picking, argmin-carrying
// collision reductions. Same invariance mechanism as `iamax.cuh`: the
// (key, index) pair with the LOWER-index tie-break applied at EVERY combine
// step, so the winner cannot depend on the block size.
//
// NaN policy: NaN inputs are SKIPPED (IEEE compares are false, so a NaN never
// wins — diverges from `np.argmax`, which propagates NaN; exclude NaN in
// oracles). An all-NaN (or empty) vector returns index 0. Unlike `iamax`, the
// running best seeds EMPTY (idx = UINT32_MAX) rather than key = 0, so
// all-negative (argmax) / all-positive (argmin) vectors reduce correctly.

namespace argreduce_detail {

// Fold candidate (ckey, cidx) into the running best in place. MINIMUM picks
// the comparison direction; empty slots (idx == UINT32_MAX) lose to any real
// candidate; NaN candidates are skipped; equal keys keep the lower index.
// Key policies: how a raw element becomes a comparison key, and what the
// out_val fallback is when NO candidate survived (empty/all-NaN input).
// IdKey  = signed argmax/argmin (fallback x[0], the historical behavior).
// AbsKey = BLAS i_amax over |x| (fallback 0 — iamax's documented contract).
struct IdKey {
    template <typename T> __device__ __forceinline__ T operator()(T v) const { return v; }
    template <typename T> __device__ __forceinline__ T empty(const T *x) const { return x[0]; }
};
struct AbsKey {
    template <typename T> __device__ __forceinline__ T operator()(T v) const { return abs(v); }
    template <typename T> __device__ __forceinline__ T empty(const T *) const { return static_cast<T>(0); }
};

template <typename T, bool MINIMUM>
__device__ __forceinline__ void combine(T &key, uint32_t &idx, T ckey, uint32_t cidx) {
    if (cidx == UINT32_MAX || ckey != ckey) return;           // empty or NaN candidate
    const bool better = MINIMUM ? (ckey < key) : (ckey > key);
    if (idx == UINT32_MAX || better || (ckey == key && cidx < idx)) { key = ckey; idx = cidx; }
}

// tier-shared body: default (per-thread strided scan + thread-0 serial fold
// through scratch) — the iamax default variant's shape, signed and two-sided.
template <typename T, bool MINIMUM, bool TRAILING_SYNC, typename Key = IdKey>
__device__ void argreduce(uint32_t n, const T *x, uint32_t *out, T *out_val, T *s_scratch) {
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T *s_key = s_scratch;
    uint32_t *s_idx = reinterpret_cast<uint32_t *>(s_scratch + size);

    T best_key = static_cast<T>(0);
    uint32_t best_idx = UINT32_MAX;
    for (uint32_t i = rank; i < n; i += size)
        combine<T, MINIMUM>(best_key, best_idx, Key{}(x[i]), i);
    s_key[rank] = best_key;
    s_idx[rank] = best_idx;
    __syncthreads();

    if (rank == 0) {
        T key = s_key[0];
        uint32_t idx = s_idx[0];
        uint32_t lim = (size < n) ? size : n;
        for (uint32_t i = 1; i < lim; i++)
            combine<T, MINIMUM>(key, idx, s_key[i], s_idx[i] == UINT32_MAX ? UINT32_MAX : s_idx[i]);
        out[0] = (idx == UINT32_MAX) ? 0u : idx;
        if (out_val != nullptr) out_val[0] = (idx == UINT32_MAX) ? Key{}.empty(x) : key;
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// tier-shared body, warp-shuffle variant (`_fast`): per-thread strided scan,
// in-warp `__shfl_down_sync` fold of the (key, index) pair, per-warp winners
// combined through scratch by warp 0 — the argreduce twin of `iamax_fast`.
// Same combine (lower-index tie-break at EVERY step) so the result is
// bit-identical to the default variant at any block size.
template <typename T, bool MINIMUM, bool TRAILING_SYNC, typename Key = IdKey>
__device__ void argreduce_fast(uint32_t n, const T *x, uint32_t *out, T *out_val,
                               T *s_scratch) {
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    uint32_t nw = (size + 31) / 32;
    T *s_key = s_scratch;
    uint32_t *s_idx = reinterpret_cast<uint32_t *>(s_scratch + nw);

    T key = static_cast<T>(0);
    uint32_t idx = UINT32_MAX;
    for (uint32_t i = rank; i < n; i += size)
        combine<T, MINIMUM>(key, idx, Key{}(x[i]), i);
    for (int off = 16; off > 0; off >>= 1) {
        T okey = __shfl_down_sync(0xffffffffu, key, off);
        uint32_t oidx = __shfl_down_sync(0xffffffffu, idx, off);
        combine<T, MINIMUM>(key, idx, okey, oidx);
    }
    uint32_t lane = rank & 31, warp = rank >> 5;
    if (lane == 0) { s_key[warp] = key; s_idx[warp] = idx; }
    __syncthreads();

    if (rank < 32) {
        key = (rank < nw) ? s_key[rank] : static_cast<T>(0);
        idx = (rank < nw) ? s_idx[rank] : UINT32_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            T okey = __shfl_down_sync(0xffffffffu, key, off);
            uint32_t oidx = __shfl_down_sync(0xffffffffu, idx, off);
            combine<T, MINIMUM>(key, idx, okey, oidx);
        }
        if (rank == 0) {
            out[0] = (idx == UINT32_MAX) ? 0u : idx;
            if (out_val != nullptr) out_val[0] = (idx == UINT32_MAX) ? Key{}.empty(x) : key;
        }
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// single-warp body: strided per-lane scan + shuffle fold, index broadcast.
template <typename T, bool MINIMUM, typename Key = IdKey>
__device__ uint32_t argreduce_warp(uint32_t n, const T *x) {
    uint32_t lane = (flat_rank()) & 31;
    T key = static_cast<T>(0);
    uint32_t idx = UINT32_MAX;
    for (uint32_t i = lane; i < n; i += 32u)
        combine<T, MINIMUM>(key, idx, Key{}(x[i]), i);
    for (int off = 16; off > 0; off >>= 1) {
        T okey = __shfl_down_sync(0xffffffffu, key, off);
        uint32_t oidx = __shfl_down_sync(0xffffffffu, idx, off);
        combine<T, MINIMUM>(key, idx, okey, oidx);
    }
    idx = (idx == UINT32_MAX) ? 0u : idx;
    return __shfl_sync(0xffffffffu, idx, 0);
}

// single-thread body: serial scan, register return.
template <typename T, bool MINIMUM, typename Key = IdKey>
__device__ __forceinline__ uint32_t argreduce_serial(uint32_t n, const T *x) {
    T key = static_cast<T>(0);
    uint32_t idx = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++) combine<T, MINIMUM>(key, idx, Key{}(x[i]), i);
    return (idx == UINT32_MAX) ? 0u : idx;
}


// single-warp REGISTER-pair body: fold per-lane (key, idx) with the shuffle
// ladder, broadcast the winning pair to every lane. The keyed twin of
// `warp::reduce(T partial)` — no array walk, no scratch. Empty lanes pass
// idx == UINT32_MAX (they can never win); if EVERY lane is empty the sentinel
// itself is returned (there is no x[0] to fall back to, unlike the array form).
template <typename T, bool MINIMUM>
__device__ __forceinline__ uint32_t argreduce_pair(T key, uint32_t idx, T *win_key) {
    for (int off = 16; off > 0; off >>= 1) {
        T okey = __shfl_down_sync(0xffffffffu, key, off);
        uint32_t oidx = __shfl_down_sync(0xffffffffu, idx, off);
        combine<T, MINIMUM>(key, idx, okey, oidx);
    }
    idx = __shfl_sync(0xffffffffu, idx, 0);
    if (win_key != nullptr) *win_key = __shfl_sync(0xffffffffu, key, 0);
    return idx;
}
} // namespace argreduce_detail

/**
 * @brief Shared-scratch size in bytes for `argmax` / `argmin`.
 *
 * One signed key (`T`) plus one index (`uint32_t`) per thread — the same
 * layout (and size) as `iamax_scratch_bytes`.
 *
 * @tparam T  Scalar type.
 * @param block_threads  Number of threads in the launching block.
 * @return Bytes to allocate for `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t argreduce_scratch_bytes(uint32_t block_threads) {
    return (static_cast<std::size_t>(block_threads
          + (block_threads * sizeof(uint32_t) + sizeof(T) - 1) / sizeof(T))) * sizeof(T);
}

/**
 * @brief Index of the maximum element (signed), into `out[0]`.
 *
 * `np.argmax` semantics with the LOWER-index tie-break at every combine step
 * (thread-count invariant) and NaN skipped (see the header note; all-NaN → 0).
 * Non-destructive. NumPy equivalent: `int(np.argmax(x))` (NaN-free inputs).
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param n          Number of elements.
 * @param x          Read-only input vector of length `n`.
 * @param out        Output: `out[0]` receives the argmax index.
 * @param s_scratch  Shared scratch of `argreduce_scratch_bytes<T>(blockDim)` bytes.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void argmax(uint32_t n, const T *x, uint32_t *out, T *s_scratch) {
    argreduce_detail::argreduce<T, false, TRAILING_SYNC>(n, x, out, nullptr, s_scratch);
}

/**
 * @brief `argmax` also returning the maximum value in `out_val[0]`.
 *
 * NumPy equivalents: `out[0] = int(np.argmax(x))`, `out_val[0] = np.max(x)`
 * (NaN-free inputs; all-NaN returns `x[0]`).
 *
 * @tparam T,TRAILING_SYNC  See `argmax`.
 * @param n,x,s_scratch  See `argmax`.
 * @param out      Output index slot.
 * @param out_val  Output value slot.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void argmax(uint32_t n, const T *x, uint32_t *out, T *out_val, T *s_scratch) {
    argreduce_detail::argreduce<T, false, TRAILING_SYNC>(n, x, out, out_val, s_scratch);
}

/**
 * @brief Index of the minimum element (signed), into `out[0]`.
 *
 * `np.argmin` semantics; tie-break, NaN policy, and invariance as `argmax`.
 *
 * @tparam T,TRAILING_SYNC  See `argmax`.
 * @param n,x,out,s_scratch  See `argmax`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void argmin(uint32_t n, const T *x, uint32_t *out, T *s_scratch) {
    argreduce_detail::argreduce<T, true, TRAILING_SYNC>(n, x, out, nullptr, s_scratch);
}

/**
 * @brief `argmin` also returning the minimum value in `out_val[0]`.
 *
 * NumPy equivalents: `out[0] = int(np.argmin(x))`, `out_val[0] = np.min(x)`.
 *
 * @tparam T,TRAILING_SYNC  See `argmax`.
 * @param n,x,out,out_val,s_scratch  See the value-returning `argmax`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void argmin(uint32_t n, const T *x, uint32_t *out, T *out_val, T *s_scratch) {
    argreduce_detail::argreduce<T, true, TRAILING_SYNC>(n, x, out, out_val, s_scratch);
}

/**
 * @brief Shared-scratch size in bytes for `argmax_fast` / `argmin_fast`.
 *
 * One (key, index) slot per warp — the same layout (and size) as
 * `iamax_fast_scratch_bytes`.
 *
 * @tparam T  Scalar type.
 * @param block_threads  Number of threads in the launching block.
 * @return Bytes to allocate for the `_fast` scratch.
 */
template <typename T>
__host__ __device__ constexpr std::size_t argreduce_fast_scratch_bytes(uint32_t block_threads) {
    return (static_cast<std::size_t>(((block_threads + 31) / 32)
         + (((block_threads + 31) / 32) * sizeof(uint32_t) + sizeof(T) - 1) / sizeof(T))) * sizeof(T);
}

/**
 * @brief `argmax`, warp-shuffle variant (in-register warp folds + per-warp
 *        scratch combine — the `iamax_fast` strategy, signed).
 *
 * Bit-identical result to `argmax` (same combine, same tie-break/NaN policy);
 * fewer scratch bytes and no serial thread-0 fold — the wide-block fast path.
 * Scratch via `argreduce_fast_scratch_bytes<T>(blockDim)`. REQUIRES a
 * full-warp block size (a multiple of 32): the shuffle folds use the full
 * 0xffffffff mask (the `iamax_fast` contract) — use the default `argmax` for
 * partial-warp blocks.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param n          Number of elements.
 * @param x          Read-only input vector of length `n`.
 * @param out        Output: `out[0]` receives the argmax index.
 * @param s_scratch  Shared scratch of `argreduce_fast_scratch_bytes<T>(blockDim)` bytes.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void argmax_fast(uint32_t n, const T *x, uint32_t *out, T *s_scratch) {
    argreduce_detail::argreduce_fast<T, false, TRAILING_SYNC>(n, x, out, nullptr, s_scratch);
}

/**
 * @brief `argmax_fast` also returning the maximum value in `out_val[0]`.
 *
 * @tparam T,TRAILING_SYNC  See `argmax_fast`.
 * @param n,x,s_scratch  See `argmax_fast`.
 * @param out      Output index slot.
 * @param out_val  Output value slot.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void argmax_fast(uint32_t n, const T *x, uint32_t *out, T *out_val, T *s_scratch) {
    argreduce_detail::argreduce_fast<T, false, TRAILING_SYNC>(n, x, out, out_val, s_scratch);
}

/**
 * @brief `argmin`, warp-shuffle variant. See `argmax_fast`.
 *
 * @tparam T,TRAILING_SYNC  See `argmax_fast`.
 * @param n,x,out,s_scratch  See `argmax_fast`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void argmin_fast(uint32_t n, const T *x, uint32_t *out, T *s_scratch) {
    argreduce_detail::argreduce_fast<T, true, TRAILING_SYNC>(n, x, out, nullptr, s_scratch);
}

/**
 * @brief `argmin_fast` also returning the minimum value in `out_val[0]`.
 *
 * @tparam T,TRAILING_SYNC  See `argmax_fast`.
 * @param n,x,out,out_val,s_scratch  See the value-returning `argmax_fast`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void argmin_fast(uint32_t n, const T *x, uint32_t *out, T *out_val, T *s_scratch) {
    argreduce_detail::argreduce_fast<T, true, TRAILING_SYNC>(n, x, out, out_val, s_scratch);
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    /**
     * @brief Warp argmax over PER-LANE register (key, index) pairs; winning
     *        index returned on every lane.
     *
     * The register entry point for "each lane evaluated its own candidate"
     * patterns (warp-packed samplers, coarse-search sweeps, best-rollout
     * picks): no array staging, no scratch — the pair folds through
     * `__shfl_down_sync` with the SAME lower-index tie-break and NaN-skip
     * combine as every other argreduction, so results are lane/order
     * independent. Pass `idx = UINT32_MAX` from lanes with no candidate (they
     * never win); if ALL lanes are empty the sentinel `UINT32_MAX` is
     * returned. Full 32-lane warp required (mask `0xffffffff`).
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param key  This lane's candidate key.
     * @param idx  This lane's candidate index (`UINT32_MAX` = empty lane).
     * @return The winning index, identical on every lane.
     */
    template <typename T>
    __device__ uint32_t argmax_pair(T key, uint32_t idx)
    { return argreduce_detail::argreduce_pair<T, false>(key, idx, nullptr); }

    /**
     * @brief `argmax_pair` also returning the winning key on every lane.
     *
     * @tparam T  Scalar type.
     * @param key,idx  See `argmax_pair`.
     * @param win_key  Out: the winning key (valid on every lane; unspecified
     *                 when all lanes are empty).
     * @return The winning index, identical on every lane.
     */
    template <typename T>
    __device__ uint32_t argmax_pair(T key, uint32_t idx, T &win_key)
    { return argreduce_detail::argreduce_pair<T, false>(key, idx, &win_key); }

    /**
     * @brief Warp argmin over PER-LANE register (key, index) pairs. See
     *        `argmax_pair` (same mechanism, minimum direction).
     *
     * @tparam T  Scalar type.
     * @param key,idx  See `argmax_pair`.
     * @return The winning index, identical on every lane.
     */
    template <typename T>
    __device__ uint32_t argmin_pair(T key, uint32_t idx)
    { return argreduce_detail::argreduce_pair<T, true>(key, idx, nullptr); }

    /**
     * @brief `argmin_pair` also returning the winning key on every lane.
     *
     * @tparam T  Scalar type.
     * @param key,idx,win_key  See the value-returning `argmax_pair`.
     * @return The winning index, identical on every lane.
     */
    template <typename T>
    __device__ uint32_t argmin_pair(T key, uint32_t idx, T &win_key)
    { return argreduce_detail::argreduce_pair<T, true>(key, idx, &win_key); }

    /**
     * @brief Single-warp argmax, index returned on every lane (register
     * broadcast, no scratch). Full 32 lanes required. See `glass::argmax` and
     * the `warp::iamax` notes (same mechanism, signed).
     */
    template <typename T>
    __device__ uint32_t argmax(uint32_t n, const T *x)
    { return argreduce_detail::argreduce_warp<T, false>(n, x); }

    /**
     * @brief Single-warp argmin, index returned on every lane. Full 32 lanes
     * required. See `glass::argmin`.
     */
    template <typename T>
    __device__ uint32_t argmin(uint32_t n, const T *x)
    { return argreduce_detail::argreduce_warp<T, true>(n, x); }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    /** @brief Single-thread argmax (register return). See `glass::argmax`. */
    template <typename T>
    __device__ uint32_t argmax(uint32_t n, const T *x)
    { return argreduce_detail::argreduce_serial<T, false>(n, x); }

    /** @brief Single-thread argmin (register return). See `glass::argmin`. */
    template <typename T>
    __device__ uint32_t argmin(uint32_t n, const T *x)
    { return argreduce_detail::argreduce_serial<T, true>(n, x); }
}
