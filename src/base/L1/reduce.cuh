#pragma once
#include "../barrier.cuh"
#include <cstdint>

// ═══════════════════════════════════════════════════════════════════════
// shfl_detail — THE warp shuffle ladders (shared by every reduction-family
// header; one definition replaces ~40 former inline copies, audit 2026-08-11).
// `fold*` = __shfl_down (lane 0 holds the result); `butterfly*` = __shfl_xor
// (every lane holds the result; fixed pattern → deterministic).
// ═══════════════════════════════════════════════════════════════════════
namespace shfl_detail {
    // op functors (plain structs, NOT device lambdas — extended-lambda must
    // never become a required consumer flag for a header-only library)
    struct MaxOp { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return (b > a) ? b : a; } };
    struct MinOp { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return (b < a) ? b : a; } };

    template <typename T>
    __device__ __forceinline__ T fold_sum(T v) {
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        return v;
    }
    template <typename T, typename Op>
    __device__ __forceinline__ T fold(T v, Op op) {
        for (int off = 16; off > 0; off >>= 1) {
            const T o = __shfl_down_sync(0xffffffffu, v, off);
            v = op(v, o);
        }
        return v;
    }
    template <typename T>
    __device__ __forceinline__ T butterfly_sum(T v) {
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        return v;
    }
    template <typename T, typename Op>
    __device__ __forceinline__ T butterfly(T v, Op op) {
        for (int off = 16; off > 0; off >>= 1) {
            const T o = __shfl_xor_sync(0xffffffffu, v, off);
            v = op(v, o);
        }
        return v;
    }

    // ── bounded folds: legal for a PARTIAL warp of k contiguous active lanes
    // 0..k-1 (any 1 <= k <= 32). The sync mask names exactly the active
    // lanes, and the guard discards shuffle results whose source lane is not
    // active (undefined values never enter the sum). k == 32 routes through
    // the exact full-warp ladder above, so multiple-of-32 launches remain
    // bit-identical to the historical path. These are what make the block
    // `_fast` family — and therefore `pcg` — legal at ANY thread count.
    __device__ __forceinline__ unsigned lanes_mask(uint32_t k) {
        return (k >= 32u) ? 0xffffffffu : ((1u << k) - 1u);
    }
    // active-lane count of the warp holding flat rank `rank` in a block of
    // `size` threads: 32 for every warp except a ragged last one.
    __device__ __forceinline__ uint32_t warp_active(uint32_t rank, uint32_t size) {
        const uint32_t rem = size - (rank & ~31u);
        return (rem < 32u) ? rem : 32u;
    }
    template <typename T>
    __device__ __forceinline__ T fold_sum_bounded(T v, uint32_t lane, uint32_t k) {
        if (k == 32u) return fold_sum(v);
        const unsigned mask = (1u << k) - 1u;
        for (int off = 16; off > 0; off >>= 1) {
            const T o = __shfl_down_sync(mask, v, off);
            if (lane + off < k) v += o;
        }
        return v;
    }
    template <typename T, typename Op>
    __device__ __forceinline__ T fold_bounded(T v, Op op, uint32_t lane, uint32_t k) {
        if (k == 32u) return fold(v, op);
        const unsigned mask = (1u << k) - 1u;
        for (int off = 16; off > 0; off >>= 1) {
            const T o = __shfl_down_sync(mask, v, off);
            if (lane + off < k) v = op(v, o);
        }
        return v;
    }
}

/**
 * @brief Sum reduction: `x[0] = Σ x[i]` (in-place, destructive).
 *
 * Default `threadIdx`-based halving reduce; the block cooperatively sums the
 * vector and leaves the total in `x[0]` (the input is overwritten). NumPy
 * equivalent: `np.sum(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n`; the sum lands in `x[0]`.
 */
// Shared body: default halving (tree) reduce; result in x[0]. The barrier policy
// supplies rank/size + the internal AND trailing sync, so the plain glass:: and
// the cgrps:: surfaces share this one body. TRAILING_SYNC (default true) makes
// x[0] valid for ALL threads; pass false when the caller owns the next barrier.
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void reduce_impl(Bar bar, uint32_t n, T *x)
{
    uint32_t rank = bar.rank(), size = bar.size();
    uint32_t left = n;
    while (left > 3) {
        bool odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = rank; i < left; i += size) x[i] += x[i + left];
        if (rank == 0 && odd) x[0] += x[2*left];
        bar.sync();
    }
    if (rank == 0) { for (uint32_t i = 1; i < left; i++) x[0] += x[i]; }
    if constexpr (TRAILING_SYNC) bar.sync();
}

// default threadIdx-based halving reduce; result in x[0]
template <typename T, bool TRAILING_SYNC = true>
__device__ void reduce(uint32_t n, T *x)
{
    reduce_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, n, x);
}

/**
 * @brief Sum reduction: `x[0] = Σ x[i]` (in-place), compile-time size.
 *
 * Compile-time-`N` overload of the halving reduce. NumPy equivalent: `np.sum(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  In/out vector of length `N`; the sum lands in `x[0]`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void reduce(T *x)
{
    reduce_impl<BlockBarrier, T, TRAILING_SYNC>(BlockBarrier{}, N, x);
}

/**
 * @brief Sum reduction: `x[0] = Σ x[i]` (in-place), low-memory variant.
 *
 * Thread 0 serially accumulates all elements into `x[0]`; uses no scratch.
 * NumPy equivalent: `np.sum(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n`; the sum lands in `x[0]`.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void reduce_lowmem(uint32_t n, T *x)
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        for (uint32_t i = 1; i < n; i++) x[0] += x[i];
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Sum reduction: `x[0] = Σ x[i]` (in-place), low-memory, compile-time size.
 *
 * Compile-time-`N` overload; thread 0 serially accumulates into `x[0]`.
 * NumPy equivalent: `np.sum(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  In/out vector of length `N`; the sum lands in `x[0]`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void reduce_lowmem(T *x)
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        for (uint32_t i = 1; i < N; i++) x[0] += x[i];
    if constexpr (TRAILING_SYNC) __syncthreads();
}
/**
 * @brief Shared-scratch size in bytes for the `reduce_fast` ops.
 *
 * The warp-shuffle reduce combines across warps through one scratch slot per
 * warp: `ceil(block_threads / 32)` elements of `T`. Allocate
 * `reduce_fast_scratch_bytes<T>(block_threads)` for the `s_scratch` argument.
 *
 * @tparam T  Scalar type.
 * @param block_threads  Number of threads in the launching block.
 * @return Bytes to allocate for `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t reduce_fast_scratch_bytes(uint32_t block_threads) { return static_cast<std::size_t>((block_threads + 31) / 32) * sizeof(T); }

/**
 * @brief Sum reduction: `x[0] = Σ x[i]` (in-place), warp-shuffle variant.
 *
 * Accumulates with a warp-shuffle reduction plus an inter-warp reduction
 * through shared scratch; the total lands in `x[0]`. NumPy equivalent:
 * `np.sum(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n          Number of elements.
 * @param x          In/out vector of length `n`; the sum lands in `x[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per
 *                   warp) — size with `reduce_fast_scratch_bytes<T>(blockDim)`.
 */
// warp-shuffle + inter-warp reduce; s_scratch: ceil(blockDim/32)*sizeof(T); result in x[0]
template <typename T, bool TRAILING_SYNC = true>
__device__ void reduce_fast(uint32_t n, T *x, T *s_scratch)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < n; i += size) val += x[i];
    uint32_t lane = rank & 31, warp = rank >> 5;
    val = shfl_detail::fold_sum_bounded(val, lane, shfl_detail::warp_active(rank, size));
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        val = shfl_detail::fold_sum_bounded(val, rank, (size < 32u) ? size : 32u);
        if (rank == 0) x[0] = val;
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Sum reduction: `x[0] = Σ x[i]` (in-place), warp-shuffle, compile-time size.
 *
 * Compile-time-`N` overload of the warp-shuffle reduce. NumPy equivalent:
 * `np.sum(x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x          In/out vector of length `N`; the sum lands in `x[0]`.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per
 *                   warp) — size with `reduce_fast_scratch_bytes<T>(blockDim)`.
 */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void reduce_fast(T *x, T *s_scratch)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < N; i += size) val += x[i];
    uint32_t lane = rank & 31, warp = rank >> 5;
    val = shfl_detail::fold_sum_bounded(val, lane, shfl_detail::warp_active(rank, size));
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        val = shfl_detail::fold_sum_bounded(val, rank, (size < 32u) ? size : 32u);
        if (rank == 0) x[0] = val;
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ── register-partial → block-sum overload ────────────────────────────────
// Block-reduce one PER-THREAD register value `partial` (one contribution per
// thread) and return the block total to EVERY thread, with no x[] buffer to
// materialize the partials first.  This is the entry point for fused
// "compute-a-partial-then-sum" patterns (e.g. cost/barrier kernels that form
// a per-thread term and previously did a serial thread-0 sum): each thread
// passes its own contribution directly.
//
// s_scratch must hold ceil(blockDim/32) elements (one per warp), the same
// sizing as the array overloads above.  The result is broadcast to all
// threads (s_scratch[0] holds the total on return); the routine ends on a
// __syncthreads(), so s_scratch is safe to reuse afterwards.  Threads that
// have no contribution should pass partial = 0.
/**
 * @brief Block-sum of a per-thread register value: returns `Σ partial`.
 *
 * Reduces one PER-THREAD contribution `partial` (one per thread) across the
 * block and returns the total to EVERY thread, with no intermediate `x[]`
 * buffer. This is the entry point for fused "compute-a-partial-then-sum"
 * patterns (e.g. cost/barrier kernels). The result is also broadcast through
 * `s_scratch[0]`; the routine ends on a `__syncthreads()`, so `s_scratch` is
 * safe to reuse afterwards. Threads with no contribution should pass `0`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param partial    This thread's contribution to the block sum.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per
 *                   warp) — size with `reduce_fast_scratch_bytes<T>(blockDim)`;
 *                   on return `s_scratch[0]` holds the total.
 * @return The block-wide total `Σ partial`, identical on every thread.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ T reduce_fast(T partial, T *s_scratch)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T val = partial;
    uint32_t lane = rank & 31, warp = rank >> 5;
    val = shfl_detail::fold_sum_bounded(val, lane, shfl_detail::warp_active(rank, size));
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        val = shfl_detail::fold_sum_bounded(val, rank, (size < 32u) ? size : 32u);
        if (rank == 0) s_scratch[0] = val;
    }
    __syncthreads();
    T total = s_scratch[0];
    if constexpr (TRAILING_SYNC) __syncthreads();
    return total;
}
/**
 * @brief Block-min of a per-thread register value: returns `min partial`.
 *
 * Min twin of the `reduce_fast(partial, s_scratch)` sum overload above, with the
 * same shape: one PER-THREAD contribution in, the block-wide minimum returned to
 * EVERY thread, no `x[]` buffer. The entry point for fused "compute-a-partial-then
 * -min" patterns (e.g. a softmax's max-shift, or a smooth-min collision cost).
 * Threads with no contribution should pass the identity — a large sentinel such as
 * `1e30f`, NOT zero.
 *
 * Kept separate from the sum overload rather than folded into a generic
 * `reduce_fast(partial, s_scratch, Op)`: an operator template would force every
 * existing call site through a functor, and the two identities (0 vs +inf) cannot
 * share a default.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param partial    This thread's contribution to the block min.
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per
 *                   warp) — size with `reduce_fast_scratch_bytes<T>(blockDim)`;
 *                   on return `s_scratch[0]` holds the minimum.
 * @return The block-wide minimum, identical on every thread.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ T reduce_fast_min(T partial, T *s_scratch)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    uint32_t nw = (size + 31) / 32;
    T val = partial;
    uint32_t lane = rank & 31, warp = rank >> 5;
    val = shfl_detail::fold_bounded(val, shfl_detail::MinOp{}, lane,
                                    shfl_detail::warp_active(rank, size));
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    if (rank < 32) {
        // Lanes past the warp count seed with a neighbour's value rather than a
        // sentinel: min is idempotent, so re-folding a value already in the set
        // cannot change the result, and it needs no type-specific +inf.
        val = s_scratch[(rank < nw) ? rank : 0];
        val = shfl_detail::fold_bounded(val, shfl_detail::MinOp{}, rank,
                                        (size < 32u) ? size : 32u);
        if (rank == 0) s_scratch[0] = val;
    }
    __syncthreads();
    T total = s_scratch[0];
    if constexpr (TRAILING_SYNC) __syncthreads();
    return total;
}



/**
 * @brief Fixed 32-way pairwise tree reduce of a register array (returns the sum in p[0]).
 *
 * Combines 32 partials in the EXACT pairwise grouping that `glass::warp::reduce`
 * produces in lane 0 (a `__shfl_down_sync` tree with offsets 16,8,4,2,1), so a
 * serial caller (e.g. a sub-warp fallback below 32 threads) matches the
 * full-warp caller bit-for-bit. This is the shared primitive that lets the
 * contraction-parallel `*_reduced` engines stay thread-count invariant across the
 * 32-thread boundary. NumPy equivalent: `np.sum(p)` (different rounding).
 *
 * @tparam T  Scalar type.
 * @param p  In/out array of 32 partials; on return `p[0]` holds the total (p is clobbered).
 * @return The sum of `p[0..31]`.
 */
template <typename T>
__device__ __forceinline__ T reduced_tree32(T p[32])
{
    #pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1)
        for (uint32_t l = 0; l < off; ++l) p[l] += p[l + off];
    return p[0];
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    // Single-warp reductions: raw __shfl, no shared scratch, no inter-warp combine.
    // For warp-per-problem kernels (one 32-lane warp owns the reduction). The
    // caller must run a full warp (mask 0xffffffff); partial-warp callers must
    // pass 0 from inactive lanes. Distinct from reduce_fast, which is
    // block-scoped (warp-shuffle + shared inter-warp combine).

    /**
     * @brief Sum reduction within one warp: `x[0] = Σ x[i]` (in-place), single-warp.
     *
     * One 32-lane warp sums the vector with `__shfl_down_sync`; the total lands in
     * `x[0]` (input overwritten). No shared scratch, no inter-warp combine. NumPy
     * equivalent: `np.sum(x)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  In/out vector of length `n`; the sum lands in `x[0]`.
     */
    template <typename T>
    __device__ void reduce(uint32_t n, T *x)
    {
        uint32_t lane = (flat_rank()) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < n; i += 32) val += x[i];
        val = shfl_detail::fold_sum(val);
        if (lane == 0) x[0] = val;
        __syncwarp();
    }

    /**
     * @brief Sum reduction within one warp: `x[0] = Σ x[i]` (in-place), single-warp, compile-time size.
     *
     * Compile-time-`N` overload. NumPy equivalent: `np.sum(x)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  In/out vector of length `N`; the sum lands in `x[0]`.
     */
    template <typename T, uint32_t N>
    __device__ void reduce(T *x)
    {
        uint32_t lane = (flat_rank()) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < N; i += 32) val += x[i];
        val = shfl_detail::fold_sum(val);
        if (lane == 0) x[0] = val;
        __syncwarp();
    }

    /**
     * @brief Warp-sum of a per-lane register value: returns `Σ partial` on every lane.
     *
     * Reduces one PER-LANE contribution across a single warp and broadcasts the
     * total back to all 32 lanes — no `x[]` buffer, no shared scratch. The entry
     * point for fused "compute-a-partial-then-sum" patterns inside a warp (e.g.
     * row-norm / residual accumulation). Inactive lanes should pass `0`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param partial  This lane's contribution.
     * @return The warp-wide total `Σ partial`, identical on every lane.
     */
    template <typename T>
    __device__ T reduce(T partial)
    {
        T val = partial;
        val = shfl_detail::fold_sum(val);
        return __shfl_sync(0xffffffff, val, 0);
    }

    /**
     * @brief Warp-min of a per-lane register value: returns `min(partial)` on
     *        every lane.
     *
     * The min twin of the register `warp::reduce` (same shuffle ladder +
     * broadcast; no buffer, no scratch). Inactive/idle lanes must pass the
     * identity (e.g. `+INFINITY` — there is no empty-lane sentinel here; use
     * `argmin_pair` when a lane can be excluded by index). NaN candidates
     * lose every compare, so a NaN lane never wins unless ALL lanes are NaN.
     * Full 32-lane warp required.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param partial  This lane's contribution.
     * @return The warp-wide minimum, identical on every lane.
     */
    template <typename T>
    __device__ T reduce_min(T partial)
    {
        T val = partial;
        for (int off = 16; off > 0; off >>= 1) {
            T o = __shfl_down_sync(0xffffffff, val, off);
            if (o < val) val = o;
        }
        return __shfl_sync(0xffffffff, val, 0);
    }

    /**
     * @brief Warp-max of a per-lane register value: returns `max(partial)` on
     *        every lane. See `reduce_min` (pass `-INFINITY` from idle lanes).
     *
     * @tparam T  Scalar type.
     * @param partial  This lane's contribution.
     * @return The warp-wide maximum, identical on every lane.
     */
    template <typename T>
    __device__ T reduce_max(T partial)
    {
        T val = partial;
        for (int off = 16; off > 0; off >>= 1) {
            T o = __shfl_down_sync(0xffffffff, val, off);
            if (o > val) val = o;
        }
        return __shfl_sync(0xffffffff, val, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    // Single-thread reductions: one THREAD owns the whole sum, accumulating
    // serially. The tier deliberately ships no _fast/_lowmem twins — those name
    // reduction STRATEGIES, and one thread has none (see CLAUDE.md). Serial
    // accumulation differs in summation ORDER from the block halving tree, so
    // thread::reduce and a 1-thread block reduce agree to float tolerance, not
    // ULP (same deliberate asymmetry as thread::dot).

    /**
     * @brief Sum reduction on one thread: `x[0] = Σ x[i]`, single-thread.
     *
     * Serially accumulates the elements and writes the total to `x[0]`. Only
     * `x[0]` is written — the tail is left untouched (unlike the block halving
     * reduce, which uses `x` as scratch). No shared scratch, no shuffles, no
     * barriers, no `threadIdx` read. NumPy equivalent: `np.sum(x)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param n  Number of elements.
     * @param x  In/out vector of length `n`; the sum lands in `x[0]`.
     */
    template <typename T>
    __device__ void reduce(uint32_t n, T *x)
    {
        T val = static_cast<T>(0);
        for (uint32_t i = 0; i < n; i++) val += x[i];
        x[0] = val;
    }

    /**
     * @brief Sum reduction on one thread: `x[0] = Σ x[i]`, single-thread, compile-time size.
     *
     * Compile-time-`N` overload; the trip count folds and the loop unrolls, so
     * `x` may be a thread-local register array. Only `x[0]` is written. NumPy
     * equivalent: `np.sum(x)`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @tparam N  Number of elements (compile-time constant).
     * @param x  In/out vector of length `N`; the sum lands in `x[0]`.
     */
    template <typename T, uint32_t N>
    __device__ void reduce(T *x)
    {
        T val = static_cast<T>(0);
        for (uint32_t i = 0; i < N; i++) val += x[i];
        x[0] = val;
    }

    /**
     * @brief Sum of a per-thread register value on one thread: returns `partial`.
     *
     * The single-thread mirror of `warp::reduce(partial)`: with one problem per
     * thread there is exactly ONE contribution, so the "reduction" is the
     * identity. Exists so tier-generic call sites (templated on the interface)
     * compile against every tier with the same shape. No barriers, no shuffles,
     * no `threadIdx` read. NumPy equivalent: `np.sum([partial])`.
     *
     * @tparam T  Scalar type (e.g. `float`, `double`).
     * @param partial  This thread's (sole) contribution.
     * @return `partial`, unchanged.
     */
    template <typename T>
    __device__ T reduce(T partial)
    {
        return partial;
    }
}
