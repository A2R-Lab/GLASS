#pragma once
#include <cstdint>
#include <cassert>
#include <cub/cub.cuh>

// glass::nvidia L1 — CUB-backed reduce / dot / nrm2
// All functions are compile-time only: T, N, THREADS must be template parameters.
// THREADS defaults to 256; set it to match your kernel's blockDim.x.
//
// CUB contract: blockDim.x must equal THREADS exactly. Mismatch is silent
// (CUB does not assert), so we add a debug-only assertion below.
//
// TRAILING_SYNC: every function takes a `bool TRAILING_SYNC = true` template
// parameter as the last template argument. When true (default), the function
// returns with all threads at a __syncthreads() so callers can read the
// result safely. When false, the caller is responsible for syncing before
// reading. Pass false when fusing with subsequent work that already issues
// its own __syncthreads(). The LEADING sync (before the CUB BlockReduce) is
// NOT gated — CUB requires its TempStorage to be quiescent on entry.
//
// Example:
//   extern __shared__ float scratch[];   // must be >= sizeof(T) * THREADS
//   glass::nvidia::reduce<float, 64, 256>(x, scratch);                   // default sync
//   glass::nvidia::reduce<float, 64, 256, /*TRAILING_SYNC=*/false>(...); // fused

// Debug-only check that the launched blockDim.x matches CUB's THREADS template
// arg. Mismatch (in either direction) silently corrupts the BlockReduce result.
#ifdef NDEBUG
#define _GLASS_ASSERT_THREADS_EQ(THREADS) /* nothing */
#else
#define _GLASS_ASSERT_THREADS_EQ(THREADS)                                        \
    assert(blockDim.x == (THREADS) && blockDim.y == 1 && blockDim.z == 1 &&      \
           "glass::nvidia L1: blockDim must be (THREADS, 1, 1) exactly — "       \
           "CUB BlockReduce does not tolerate mismatch");
#endif

/**
 * @brief Block-level sum reduction backed by CUB BlockReduce.
 *
 * Sums all N elements of `x` across the block; thread 0 writes the total to
 * `x[0]` (in place). Requires `blockDim` == (THREADS, 1, 1) exactly — CUB
 * does not tolerate a mismatch. Compile-time sizes only. NumPy equivalent:
 * `x[0] = np.sum(x)`.
 *
 * @tparam T             Scalar type.
 * @tparam N             Number of elements.
 * @tparam THREADS       Block thread count (must equal blockDim.x).
 * @tparam TRAILING_SYNC Emit a trailing __syncthreads() before return (default true).
 * @param  x             Input array of length N; result lands in x[0].
 * @param  s_scratch     Shared scratch >= reduce_scratch_bytes<T, THREADS>() bytes.
 */
template <typename T, uint32_t N, uint32_t THREADS = 256, bool TRAILING_SYNC = true>
__device__ void reduce(T *x, T *s_scratch)
{
    _GLASS_ASSERT_THREADS_EQ(THREADS)
    using BlockReduce = cub::BlockReduce<T, THREADS>;
    T thread_sum = static_cast<T>(0);
    for (uint32_t i = threadIdx.x; i < N; i += THREADS)
        thread_sum += x[i];
    __syncthreads();
    T block_sum = BlockReduce(*reinterpret_cast<typename BlockReduce::TempStorage*>(s_scratch))
                      .Sum(thread_sum);
    if (threadIdx.x == 0) x[0] = block_sum;
    if constexpr (TRAILING_SYNC) {
        __syncthreads();
    }
}

/**
 * @brief Block-level dot product backed by CUB BlockReduce.
 *
 * Computes the inner product of the N-element vectors `x` and `y`; thread 0
 * writes the scalar result to `*out`. Requires `blockDim` == (THREADS, 1, 1)
 * exactly. Compile-time sizes only. NumPy equivalent: `*out = np.dot(x, y)`.
 *
 * @tparam T             Scalar type.
 * @tparam N             Number of elements.
 * @tparam THREADS       Block thread count (must equal blockDim.x).
 * @tparam TRAILING_SYNC Emit a trailing __syncthreads() before return (default true).
 * @param  x             First input vector (length N).
 * @param  y             Second input vector (length N).
 * @param  out           Output pointer for the resulting scalar.
 * @param  s_scratch     Shared scratch >= reduce_scratch_bytes<T, THREADS>() bytes.
 */
template <typename T, uint32_t N, uint32_t THREADS = 256, bool TRAILING_SYNC = true>
__device__ void dot(T *x, T *y, T *out, T *s_scratch)
{
    _GLASS_ASSERT_THREADS_EQ(THREADS)
    using BlockReduce = cub::BlockReduce<T, THREADS>;
    T thread_sum = static_cast<T>(0);
    for (uint32_t i = threadIdx.x; i < N; i += THREADS)
        thread_sum += x[i] * y[i];
    __syncthreads();
    T block_sum = BlockReduce(*reinterpret_cast<typename BlockReduce::TempStorage*>(s_scratch))
                      .Sum(thread_sum);
    if (threadIdx.x == 0) *out = block_sum;
    if constexpr (TRAILING_SYNC) {
        __syncthreads();
    }
}

/**
 * @brief Block-level Euclidean (L2) norm backed by CUB BlockReduce.
 *
 * Sums the squares of the N elements of `x` and writes the square root to
 * `*out` from thread 0. Requires `blockDim` == (THREADS, 1, 1) exactly.
 * Compile-time sizes only. NumPy equivalent: `*out = np.linalg.norm(x)`.
 *
 * @tparam T             Scalar type.
 * @tparam N             Number of elements.
 * @tparam THREADS       Block thread count (must equal blockDim.x).
 * @tparam TRAILING_SYNC Emit a trailing __syncthreads() before return (default true).
 * @param  x             Input vector (length N).
 * @param  out           Output pointer for the resulting scalar norm.
 * @param  s_scratch     Shared scratch >= reduce_scratch_bytes<T, THREADS>() bytes.
 */
template <typename T, uint32_t N, uint32_t THREADS = 256, bool TRAILING_SYNC = true>
__device__ void nrm2(T *x, T *out, T *s_scratch)
{
    _GLASS_ASSERT_THREADS_EQ(THREADS)
    using BlockReduce = cub::BlockReduce<T, THREADS>;
    T thread_sum = static_cast<T>(0);
    for (uint32_t i = threadIdx.x; i < N; i += THREADS)
        thread_sum += x[i] * x[i];
    __syncthreads();
    T block_sum = BlockReduce(*reinterpret_cast<typename BlockReduce::TempStorage*>(s_scratch))
                      .Sum(thread_sum);
    if (threadIdx.x == 0) *out = sqrtf(block_sum);
    if constexpr (TRAILING_SYNC) {
        __syncthreads();
    }
}

/**
 * @brief Shared-memory bytes needed for the L1 reduce/dot/nrm2 scratch (host-callable).
 *
 * Returns `sizeof(cub::BlockReduce<T, THREADS>::TempStorage)` — the size of
 * the `s_scratch` buffer these CUB-backed reductions require. constexpr.
 *
 * @tparam T       Scalar type.
 * @tparam THREADS Block thread count (must match the reduction call).
 * @return Required scratch size in bytes.
 */
template <typename T, uint32_t THREADS = 256>
inline constexpr std::size_t reduce_scratch_bytes()
{
    return sizeof(typename cub::BlockReduce<T, THREADS>::TempStorage);
}
