#pragma once
#include <cstdint>

// Block-COOPERATIVE strided dot product with coalesced global loads.
//
// Computes the SAME value as dot_strided<T,N,SX,SY>:
//     sum_{i=0}^{N-1} x[i*SX] * y[i*SY]
// but is a *sibling* primitive, NOT a drop-in template swap for dot_strided:
//
//   * dot_strided<T,N,SX,SY> is PER-THREAD (no synchronization): every calling
//     thread independently walks the full N-length stride and returns T.  It is
//     meant for use inside an already-thread-parallel outer loop.  With a large
//     stride SX, thread t reads x[0], x[SX], x[2*SX], ... — addresses that are
//     SX apart, so a single thread's loads are uncoalesced (and every thread
//     redundantly reads the same data).
//
//   * dot_strided_coalesced is BLOCK-WIDE: the whole block cooperates on ONE
//     dot product.  The i-loop is distributed across threads with a block-stride
//     (thread rank handles i = rank, rank+size, ...), so at each step threads
//     0,1,2,... read x[rank*SX] for consecutive rank — i.e. consecutive threads
//     touch addresses SX apart.  When SX is itself the inner contiguous axis of
//     a 2D access pattern (the common "column of a row-major / strided matrix"
//     case in GRiD), transposing the iteration this way makes the per-warp load
//     a coalesced burst instead of a strided gather.  Partial sums are then
//     combined with a warp-shuffle + shared-scratch block reduction (same idiom
//     as glass::high_speed::reduce).
//
// Because it performs a block reduction it REQUIRES a block-wide launch and a
// __syncthreads-safe context; it writes the scalar result to *out (visible to
// all threads after the trailing barrier).  Callers in a per-thread context
// must keep using dot_strided instead.
//
// s_scratch must hold ceil(blockDim/32) elements of T.

/**
 * @brief Block-cooperative strided inner product with coalesced loads (DOT, strided).
 *
 * Computes the same value as `dot_strided<T,N,SX,SY>` (`Σ x[i*SX] * y[i*SY]`),
 * but the WHOLE block cooperates on one dot product: the `i`-loop is distributed
 * across threads with a block stride so consecutive ranks touch addresses `SX`
 * apart, turning a per-warp strided gather into a coalesced burst when `SX` is
 * the inner contiguous axis. Partial sums are combined via warp-shuffle plus a
 * shared-scratch block reduction; the scalar result is broadcast to `*out`
 * (visible to all threads after the trailing barrier). Requires a block-wide,
 * `__syncthreads`-safe launch — use `dot_strided` instead in a per-thread
 * context. NumPy equivalent: `np.dot(x[::SX], y[::SY])`.
 *
 * @tparam T   Scalar type (e.g. `float`, `double`).
 * @tparam N   Number of products to accumulate (compile-time constant).
 * @tparam SX  Element stride into `x` (compile-time, default 1).
 * @tparam SY  Element stride into `y` (compile-time, default 1).
 * @param x          Input vector, accessed at indices `0, SX, 2*SX, …`.
 * @param y          Input vector, accessed at indices `0, SY, 2*SY, …`.
 * @param out        Destination for the scalar result (broadcast to all threads).
 * @param s_scratch  Shared scratch of `ceil(blockDim/32)` elements (one per warp).
 */
template <typename T, uint32_t N, uint32_t SX = 1, uint32_t SY = 1, bool TRAILING_SYNC = true>
__device__ void dot_strided_coalesced(const T* x, const T* y, T* out, T* s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;

    // Each thread accumulates a partial over its block-strided slice of i.
    // Consecutive ranks read consecutive (i*SX, i*SY) base addresses.
    T val = static_cast<T>(0);
    for (uint32_t i = rank; i < N; i += size)
        val += x[i * SX] * y[i * SY];

    // Warp-level reduce, then inter-warp reduce via shared scratch.
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
    uint32_t lane = rank & 31, warp = rank >> 5;
    if (lane == 0) s_scratch[warp] = val;
    __syncthreads();
    uint32_t nw = (size + 31) / 32;
    if (rank < 32) {
        val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        if (rank == 0) *out = val;
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
