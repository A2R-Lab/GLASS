#pragma once
#include <cstdint>

// ─── softmax / log-sum-exp (L1, sampling-planner reductions) ─────────────────
//
// The max-shifted exponential-normalization pair: `softmax` turns a cost/score
// vector into normalized weights and `logsumexp` is its stable log-partition.
// This is the path-integral (MPPI-style) weight update — with per-sample costs
// J and temperature λ, `softmax(n, -λ, J, w, scratch)` is exactly
// `w_i = exp(-λ(J_i - min J)) / Σ exp(-λ(J_j - min J))` (the baseline
// subtraction is the max shift) — and the same primitive serves cross-entropy
// method updates and Boltzmann exploration policies. Classic numerics, not ML:
// the NN side of such stacks stays in torch.
//
// Numerical contract: the max shift makes every exponent <= 0, so the sums
// cannot overflow and softmax is exact under input shifts
// (`softmax(alpha, x) == softmax(alpha, x + c)`); `logsumexp` returns
// `max + log(Σ exp(· − max))`.
//
// Thread-count invariance: the elementwise passes are `rank`-strided and the
// max/sum reductions run the SAME in-place halving tree as `glass::reduce`
// (combine order fixed by `n`, not by the block size), so the result is
// identical at any block size. Scratch is `n` elements (`softmax_scratch_bytes`).

namespace softmax_detail {
    // in-place halving-tree MAX reduce — the reduce_impl tree with `+` replaced
    // by max (same fixed combine order; result in x[0]).
    template <typename Bar, typename T, bool TRAILING_SYNC = true>
    __device__ void max_tree_impl(Bar bar, uint32_t n, T *x)
    {
        uint32_t rank = bar.rank(), size = bar.size();
        uint32_t left = n;
        while (left > 3) {
            bool odd = left % 2;
            left = (left - odd) / 2;
            for (uint32_t i = rank; i < left; i += size)
                x[i] = (x[i + left] > x[i]) ? x[i + left] : x[i];
            if (rank == 0 && odd) x[0] = (x[2*left] > x[0]) ? x[2*left] : x[0];
            bar.sync();
        }
        if (rank == 0) {
            for (uint32_t i = 1; i < left; i++) x[0] = (x[i] > x[0]) ? x[i] : x[0];
        }
        if constexpr (TRAILING_SYNC) bar.sync();
    }
} // namespace softmax_detail

/**
 * @brief Shared-scratch size in bytes for `softmax` / `logsumexp`.
 *
 * Both stage the scaled inputs / exponentials through an `n`-element buffer
 * that the in-place reduction trees then consume.
 *
 * @tparam T  Scalar type.
 * @param n  Number of elements.
 * @return Bytes to allocate for `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t softmax_scratch_bytes(uint32_t n) {
    return static_cast<std::size_t>(n) * sizeof(T);
}

/**
 * @brief Max-shifted softmax: `y_i = exp(αx_i − M) / Σ_j exp(αx_j − M)`,
 *        `M = max_j(αx_j)`.
 *
 * `Σ y = 1`; `α = −λ` gives the MPPI path-integral weight update on a cost
 * vector. Shift-invariant and overflow-safe by the max subtraction. `y` MAY
 * alias `x` (same-index elementwise writes); `s_scratch` must be a distinct
 * `n`-element buffer. Thread-count invariant (fixed-order reduction trees).
 * NumPy equivalent: `scipy.special.softmax(alpha * x)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param n          Number of elements.
 * @param alpha      Scale on the inputs (temperature; negate for costs).
 * @param x          Input vector of length `n`.
 * @param y          Output weights of length `n` (may alias `x`).
 * @param s_scratch  Shared scratch of `softmax_scratch_bytes<T>(n)` bytes.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void softmax(uint32_t n, T alpha, const T *x, T *y, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) s_scratch[i] = alpha*x[i];
    __syncthreads();
    softmax_detail::max_tree_impl<BlockBarrier, T>(BlockBarrier{}, n, s_scratch);
    const T m = s_scratch[0];
    __syncthreads();                       // everyone read m before scratch is rewritten
    // unroll 1: unroll-copy FMA contraction of alpha*x[i] - m would break
    // bit-identity across thread counts.
    #pragma unroll 1
    for (uint32_t i = rank; i < n; i += size) {
        const T e = exp(alpha*x[i] - m);
        y[i] = e;
        s_scratch[i] = e;
    }
    __syncthreads();
    reduce_impl<BlockBarrier, T>(BlockBarrier{}, n, s_scratch);
    const T inv = static_cast<T>(1)/s_scratch[0];
    for (uint32_t i = rank; i < n; i += size) y[i] *= inv;
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Stable log-sum-exp: `out[0] = M + log(Σ_j exp(αx_j − M))`,
 *        `M = max_j(αx_j)`.
 *
 * The log-partition of `softmax(α·x)` — free energy in path-integral control,
 * the stable normalizer everywhere else. Thread-count invariant; ends on the
 * trailing sync so `out[0]` is block-visible. NumPy equivalent:
 * `scipy.special.logsumexp(alpha * x)`.
 *
 * @tparam T,TRAILING_SYNC  See `softmax`.
 * @param n          Number of elements.
 * @param alpha      Scale on the inputs.
 * @param x          Input vector of length `n`.
 * @param out        Output: `out[0]` receives the log-sum-exp.
 * @param s_scratch  Shared scratch of `softmax_scratch_bytes<T>(n)` bytes.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void logsumexp(uint32_t n, T alpha, const T *x, T *out, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) s_scratch[i] = alpha*x[i];
    __syncthreads();
    softmax_detail::max_tree_impl<BlockBarrier, T>(BlockBarrier{}, n, s_scratch);
    const T m = s_scratch[0];
    __syncthreads();
    #pragma unroll 1
    for (uint32_t i = rank; i < n; i += size) s_scratch[i] = exp(alpha*x[i] - m);
    __syncthreads();
    reduce_impl<BlockBarrier, T>(BlockBarrier{}, n, s_scratch);
    if (rank == 0) out[0] = m + log(s_scratch[0]);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-thread softmax / logsumexp ───────────────────────────────────────
namespace thread {
    // One thread, serial two-pass, NO scratch — the register-resident tier.

    /** @brief Single-thread max-shifted softmax (no scratch). See `glass::softmax`. */
    template <typename T>
    __device__ void softmax(uint32_t n, T alpha, const T *x, T *y)
    {
        T m = alpha*x[0];
        for (uint32_t i = 1; i < n; i++) { const T v = alpha*x[i]; if (v > m) m = v; }
        T s = static_cast<T>(0);
        for (uint32_t i = 0; i < n; i++) { y[i] = exp(alpha*x[i] - m); s += y[i]; }
        const T inv = static_cast<T>(1)/s;
        for (uint32_t i = 0; i < n; i++) y[i] *= inv;
    }

    /** @brief Single-thread stable log-sum-exp (register return). See `glass::logsumexp`. */
    template <typename T>
    __device__ T logsumexp(uint32_t n, T alpha, const T *x)
    {
        T m = alpha*x[0];
        for (uint32_t i = 1; i < n; i++) { const T v = alpha*x[i]; if (v > m) m = v; }
        T s = static_cast<T>(0);
        for (uint32_t i = 0; i < n; i++) s += exp(alpha*x[i] - m);
        return m + log(s);
    }
}

// ─── single-warp softmax / logsumexp ─────────────────────────────────────────
namespace warp {
    namespace softmax_detail_warp {
        // full-warp butterfly max / sum — every lane ends with the result
        // (fixed shuffle pattern → deterministic).
        template <typename T>
        __device__ __forceinline__ T butterfly_max(T v) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const T o = __shfl_xor_sync(0xffffffffu, v, off);
                v = (o > v) ? o : v;
            }
            return v;
        }
        template <typename T>
        __device__ __forceinline__ T butterfly_sum(T v) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffffu, v, off);
            return v;
        }
    }

    /**
     * @brief Single-warp max-shifted softmax (no scratch). See `glass::softmax`.
     *
     * One full 32-lane warp; lanes stride the vector, the max and the sum fold
     * through xor-butterfly shuffles (every lane holds the result — no shared
     * memory, no `__syncthreads`). `y` may alias `x`. Full 32 lanes required.
     */
    template <typename T>
    __device__ void softmax(uint32_t n, T alpha, const T *x, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T m = static_cast<T>(-1e30);   // empty-lane seed; beaten by any real value's shift
        for (uint32_t i = lane; i < n; i += 32u) { const T v = alpha*x[i]; if (v > m) m = v; }
        m = softmax_detail_warp::butterfly_max(m);
        T partial = static_cast<T>(0);
        for (uint32_t i = lane; i < n; i += 32u) {
            const T e = exp(alpha*x[i] - m);
            y[i] = e;
            partial += e;
        }
        const T inv = static_cast<T>(1)/softmax_detail_warp::butterfly_sum(partial);
        for (uint32_t i = lane; i < n; i += 32u) y[i] *= inv;
        __syncwarp();
    }

    /**
     * @brief Single-warp stable log-sum-exp (register return on every lane).
     *        See `glass::logsumexp`. Full 32 lanes required.
     */
    template <typename T>
    __device__ T logsumexp(uint32_t n, T alpha, const T *x)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T m = static_cast<T>(-1e30);
        for (uint32_t i = lane; i < n; i += 32u) { const T v = alpha*x[i]; if (v > m) m = v; }
        m = softmax_detail_warp::butterfly_max(m);
        T partial = static_cast<T>(0);
        for (uint32_t i = lane; i < n; i += 32u) partial += exp(alpha*x[i] - m);
        return m + log(softmax_detail_warp::butterfly_sum(partial));
    }
}
