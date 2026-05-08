#pragma once

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void swap(std::uint32_t n,
          T *x,
          T *y,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    T temp;
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        temp = x[ind];
        x[ind] = y[ind];
        y[ind] = temp;
    }
}

// === glass::simple variants ===
namespace simple {
    template <typename T>
    __device__ void swap(uint32_t n, T *x, T *y)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t i = rank; i < n; i += size) {
            T temp = x[i];
            x[i] = y[i];
            y[i] = temp;
        }
    }
}
// ===