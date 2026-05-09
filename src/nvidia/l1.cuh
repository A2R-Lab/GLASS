#pragma once
#include <cstdint>
#include <cub/cub.cuh>

// glass::nvidia L1 — CUB-backed reduce / dot / l2norm
// All functions are compile-time only: T, N, THREADS must be template parameters.
// THREADS defaults to 256; set it to match your kernel's blockDim.x.
//
// Example:
//   extern __shared__ float scratch[];   // must be >= sizeof(T) * THREADS
//   glass::nvidia::reduce<float, 64, 256>(x, scratch);

template <typename T, uint32_t N, uint32_t THREADS = 256>
__device__ void reduce(T *x, T *s_scratch)
{
    using BlockReduce = cub::BlockReduce<T, THREADS>;
    T thread_sum = static_cast<T>(0);
    for (uint32_t i = threadIdx.x; i < N; i += THREADS)
        thread_sum += x[i];
    __syncthreads();
    T block_sum = BlockReduce(*reinterpret_cast<typename BlockReduce::TempStorage*>(s_scratch))
                      .Sum(thread_sum);
    if (threadIdx.x == 0) x[0] = block_sum;
    __syncthreads();
}

template <typename T, uint32_t N, uint32_t THREADS = 256>
__device__ void dot(T *x, T *y, T *out, T *s_scratch)
{
    using BlockReduce = cub::BlockReduce<T, THREADS>;
    T thread_sum = static_cast<T>(0);
    for (uint32_t i = threadIdx.x; i < N; i += THREADS)
        thread_sum += x[i] * y[i];
    __syncthreads();
    T block_sum = BlockReduce(*reinterpret_cast<typename BlockReduce::TempStorage*>(s_scratch))
                      .Sum(thread_sum);
    if (threadIdx.x == 0) *out = block_sum;
    __syncthreads();
}

template <typename T, uint32_t N, uint32_t THREADS = 256>
__device__ void l2norm(T *x, T *out, T *s_scratch)
{
    using BlockReduce = cub::BlockReduce<T, THREADS>;
    T thread_sum = static_cast<T>(0);
    for (uint32_t i = threadIdx.x; i < N; i += THREADS)
        thread_sum += x[i] * x[i];
    __syncthreads();
    T block_sum = BlockReduce(*reinterpret_cast<typename BlockReduce::TempStorage*>(s_scratch))
                      .Sum(thread_sum);
    if (threadIdx.x == 0) *out = sqrtf(block_sum);
    __syncthreads();
}

// smem size helper (host-callable): bytes needed for s_scratch
template <typename T, uint32_t THREADS = 256>
inline constexpr std::size_t reduce_smem_size()
{
    return sizeof(typename cub::BlockReduce<T, THREADS>::TempStorage);
}
