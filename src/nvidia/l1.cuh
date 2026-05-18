#pragma once
#include <cstdint>
#include <cassert>
#include <cub/cub.cuh>

// glass::nvidia L1 — CUB-backed reduce / dot / l2norm
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

template <typename T, uint32_t N, uint32_t THREADS = 256, bool TRAILING_SYNC = true>
__device__ void l2norm(T *x, T *out, T *s_scratch)
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

// smem size helper (host-callable): bytes needed for s_scratch
template <typename T, uint32_t THREADS = 256>
inline constexpr std::size_t reduce_smem_size()
{
    return sizeof(typename cub::BlockReduce<T, THREADS>::TempStorage);
}
