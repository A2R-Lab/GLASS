#pragma once
#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// set_const: set all n elements of x to alpha
template <typename T>
__device__
void set_const(uint32_t n, T alpha, T *x,
               cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t ind = g.thread_rank(); ind < n; ind += g.size()) {
        x[ind] = alpha;
    }
}

// === glass::simple variants ===
namespace simple {
    template <typename T>
    __device__ void set_const(uint32_t n, T alpha, T *x)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t ind = rank; ind < n; ind += size) x[ind] = alpha;
    }
}
// ===
