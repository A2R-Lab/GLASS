#pragma once
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// infnorm: infinity norm (max absolute value)
// Computes x[0] = max(|x[i]|) for i in [0, n) in-place via binary halving.
template <typename T>
__device__  __forceinline__
void infnorm(const uint32_t n, T *x, cgrps::thread_group g = cgrps::this_thread_block())
{
    uint32_t ind = g.thread_rank();
    uint32_t stride = g.size();
    uint32_t size_left = n;
    bool odd_flag;

    while (size_left > 3) {
        odd_flag = size_left % 2;
        size_left = (size_left - odd_flag) / 2;
        for (uint32_t i = ind; i < size_left; i += stride) {
            x[i] = max(abs(x[i]), abs(x[i + size_left]));
        }
        if (ind == 0 && odd_flag) { x[0] = max(abs(x[0]), abs(x[2 * size_left])); }
        g.sync();
    }
    if (ind == 0) {
        for (uint32_t i = 1; i < size_left; i++) { x[0] = max(abs(x[0]), abs(x[i])); }
    }
}

// === glass::simple variants ===
namespace simple {
    template <typename T>
    __device__ void infnorm(const uint32_t n, T *x)
    {
        uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
        uint32_t size_left = n;
        bool odd_flag;

        while (size_left > 3) {
            odd_flag = size_left % 2;
            size_left = (size_left - odd_flag) / 2;
            for (uint32_t i = ind; i < size_left; i += stride) {
                x[i] = max(abs(x[i]), abs(x[i + size_left]));
            }
            if (ind == 0 && odd_flag) { x[0] = max(abs(x[0]), abs(x[2 * size_left])); }
            __syncthreads();
        }
        if (ind == 0) {
            for (uint32_t i = 1; i < size_left; i++) { x[0] = max(abs(x[0]), abs(x[i])); }
        }
    }
}
// ===