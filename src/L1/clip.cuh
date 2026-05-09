#pragma once
#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// clip: clamp x[i] to [l[i], u[i]] element-wise
template <typename T>
__device__
void clip(uint32_t n, T *x, T *l, T *u,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t ind = g.thread_rank(); ind < n; ind += g.size()) {
        x[ind] = max(l[ind], min(x[ind], u[ind]));
    }
}

// === glass::simple variants ===
namespace simple {
    template <typename T>
    __device__ void clip(uint32_t n, T *x, T *l, T *u)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t ind = rank; ind < n; ind += size) {
            x[ind] = max(l[ind], min(x[ind], u[ind]));
        }
    }
}
// ===