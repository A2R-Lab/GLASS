#pragma once
#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// trsm: solve lower-triangular system Lx = b in-place
// L is n x n lower triangular, stored in column-major order
// b is overwritten with the solution x on return
// Scratch: none (forward substitution, sequential per column)
template <typename T>
__device__
void trsm(uint32_t n, T *L, T *b,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t col = 0; col < n; col++) {
        if (g.thread_rank() == 0) {
            b[col] /= L[col * n + col];
        }
        g.sync();
        T factor = b[col];
        for (uint32_t row = g.thread_rank() + col + 1; row < n; row += g.size()) {
            b[row] -= L[col * n + row] * factor;
        }
        g.sync();
    }
}

// === glass::simple variants ===
namespace simple {
    template <typename T>
    __device__
    void trsm(uint32_t n, T *L, T *b)
    {
        uint32_t rank = threadIdx.x
                      + threadIdx.y * blockDim.x
                      + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t col = 0; col < n; col++) {
            if (rank == 0) {
                b[col] /= L[col * n + col];
            }
            __syncthreads();
            T factor = b[col];
            for (uint32_t row = rank + col + 1; row < n; row += size) {
                b[row] -= L[col * n + row] * factor;
            }
            __syncthreads();
        }
    }
}
// ===
