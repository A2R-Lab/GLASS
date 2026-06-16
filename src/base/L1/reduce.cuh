#pragma once
#include <cstdint>

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
// default threadIdx-based halving reduce; result in x[0]
template <typename T>
__device__ void reduce(uint32_t n, T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t left = n;
    while (left > 3) {
        bool odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = rank; i < left; i += size) x[i] += x[i + left];
        if (rank == 0 && odd) x[0] += x[2*left];
        __syncthreads();
    }
    if (rank == 0) { for (uint32_t i = 1; i < left; i++) x[0] += x[i]; }
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
template <typename T, uint32_t N>
__device__ void reduce(T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t left = N;
    while (left > 3) {
        bool odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = rank; i < left; i += size) x[i] += x[i + left];
        if (rank == 0 && odd) x[0] += x[2*left];
        __syncthreads();
    }
    if (rank == 0) { for (uint32_t i = 1; i < left; i++) x[0] += x[i]; }
}

namespace low_memory {
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
    template <typename T>
    __device__ void reduce(uint32_t n, T *x)
    {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            for (uint32_t i = 1; i < n; i++) x[0] += x[i];
        __syncthreads();
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
    template <typename T, uint32_t N>
    __device__ void reduce(T *x)
    {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            for (uint32_t i = 1; i < N; i++) x[0] += x[i];
        __syncthreads();
    }
}

namespace high_speed {
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
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
     */
    // warp-shuffle + inter-warp reduce; s_scratch: ceil(blockDim/32)*sizeof(T); result in x[0]
    template <typename T>
    __device__ void reduce(uint32_t n, T *x, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < n; i += size) val += x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) x[0] = val;
        }
        __syncthreads();
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
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
     */
    template <typename T, uint32_t N>
    __device__ void reduce(T *x, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < N; i += size) val += x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) x[0] = val;
        }
        __syncthreads();
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
     * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp);
     *                   on return `s_scratch[0]` holds the total.
     * @return The block-wide total `Σ partial`, identical on every thread.
     */
    template <typename T>
    __device__ T reduce(T partial, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = partial;
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) s_scratch[0] = val;
        }
        __syncthreads();
        T total = s_scratch[0];
        __syncthreads();
        return total;
    }
}

namespace warp {
    // Single-warp reductions: raw __shfl, no shared scratch, no inter-warp combine.
    // For warp-per-problem kernels (one 32-lane warp owns the reduction). The
    // caller must run a full warp (mask 0xffffffff); partial-warp callers must
    // pass 0 from inactive lanes. Distinct from high_speed::reduce, which is
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
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < n; i += 32) val += x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
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
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T val = static_cast<T>(0);
        for (uint32_t i = lane; i < N; i += 32) val += x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
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
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        return __shfl_sync(0xffffffff, val, 0);
    }
}
