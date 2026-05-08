#pragma once
#include <cstdint>
#include <math.h>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// asum: sum of absolute values
// Computes out[0] = sum(|x[i]|) for i in [0, n)
// out is a scratch+output array of size n (overwritten)
// Scratch: n * sizeof(T) bytes (out array)
template <typename T>
__device__
void asum(uint32_t n, T *x, T *out, cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) {
        out[i] = abs(x[i]);
    }
    g.sync();
    reduce(n, out, g);
}

// === glass::simple variants ===
namespace simple {
    namespace low_memory {
        // Thread 0 accumulates; no scratch required.
        template <typename T>
        __device__ void asum(uint32_t n, T *x, T *out)
        {
            uint32_t rank = threadIdx.x
                          + threadIdx.y * blockDim.x
                          + threadIdx.z * blockDim.x * blockDim.y;
            uint32_t size = blockDim.x * blockDim.y * blockDim.z;
            for (uint32_t i = rank; i < n; i += size) {
                out[i] = abs(x[i]);
            }
            __syncthreads();
            if (rank == 0) {
                for (uint32_t i = 1; i < n; i++) out[0] += out[i];
            }
            __syncthreads();
        }
    }

    namespace high_speed {
        // Warp-shuffle + shared-memory reduction.
        // s_scratch must be at least ceil(blockDim/32) * sizeof(T) bytes.
        template <typename T>
        __device__ void asum(uint32_t n, T *x, T *s_scratch)
        {
            uint32_t rank = threadIdx.x
                          + threadIdx.y * blockDim.x
                          + threadIdx.z * blockDim.x * blockDim.y;
            uint32_t size = blockDim.x * blockDim.y * blockDim.z;

            T val = static_cast<T>(0);
            for (uint32_t i = rank; i < n; i += size) val += abs(x[i]);

            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xffffffff, val, offset);

            uint32_t lane = rank & 31;
            uint32_t warp = rank >> 5;
            if (lane == 0) s_scratch[warp] = val;
            __syncthreads();

            uint32_t num_warps = (size + 31) / 32;
            if (rank < 32) {
                val = (rank < num_warps) ? s_scratch[rank] : static_cast<T>(0);
                for (int offset = 16; offset > 0; offset >>= 1)
                    val += __shfl_down_sync(0xffffffff, val, offset);
                if (rank == 0) x[0] = val;
            }
            __syncthreads();
        }
    }
}
// ===
