#pragma once
#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// ger: rank-1 update A += alpha * x * y^T
// A is m x n, stored in column-major order
// x is length m, y is length n
// Scratch: none
template <typename T>
__device__
void ger(uint32_t m, uint32_t n, T alpha, T *x, T *y, T *A,
         cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t ind = g.thread_rank(); ind < m * n; ind += g.size()) {
        uint32_t row = ind % m;
        uint32_t col = ind / m;
        A[ind] += alpha * x[row] * y[col];
    }
}

// === glass::simple variants ===
namespace simple {
    template <typename T>
    __device__
    void ger(uint32_t m, uint32_t n, T alpha, T *x, T *y, T *A)
    {
        uint32_t rank = threadIdx.x
                      + threadIdx.y * blockDim.x
                      + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t ind = rank; ind < m * n; ind += size) {
            uint32_t row = ind % m;
            uint32_t col = ind / m;
            A[ind] += alpha * x[row] * y[col];
        }
    }
}
// ===
