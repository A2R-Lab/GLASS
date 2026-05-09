#pragma once

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void scal(std::uint32_t n, 
          T alpha, 
          T *x, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        x[ind] = alpha * x[ind];
    }
}
template <typename T>
__device__
void scal(std::uint32_t n,
          T alpha,
          T *x,
          T *y,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = alpha * x[ind];
    }
}

// === glass::simple variants ===
namespace simple {
    // x = alpha * x (in-place)
    template <typename T>
    __device__ void scal(uint32_t n, T alpha, T *x)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t i = rank; i < n; i += size) x[i] = alpha * x[i];
    }

    // y = alpha * x
    template <typename T>
    __device__ void scal(uint32_t n, T alpha, T *x, T *y)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t i = rank; i < n; i += size) y[i] = alpha * x[i];
    }
}
// ===