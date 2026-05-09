#ifndef NORM_H
#define NORM_H

#include "reduce.cuh"
#include <cooperative_groups.h>
#include <cstdint>
#include <math.h>

namespace cgrps = cooperative_groups;

template <typename T> __device__ T square(T N)
{
    return N * N;
}

template <typename T>
__device__ void vector_norm(std::uint32_t N, T *a, T *out, cgrps::thread_group g = cgrps::this_thread_block())
{
    for (int i = g.thread_rank(); i < N; i += g.size())
    {
        out[i] = a[i] * a[i];
    }
    __syncthreads();
    reduce(N, out, g);
    __syncthreads();
    out[0] = sqrt(out[0]);
}

// === glass::simple variants ===
namespace simple {
    namespace low_memory {
        // Thread 0 accumulates squared elements then takes sqrt. Result in out[0].
        template <typename T>
        __device__ void vector_norm(uint32_t N, T *a, T *out)
        {
            uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
            uint32_t size = blockDim.x * blockDim.y * blockDim.z;
            for (uint32_t i = rank; i < N; i += size) out[i] = a[i] * a[i];
            __syncthreads();
            if (rank == 0) {
                for (uint32_t i = 1; i < N; i++) out[0] += out[i];
                out[0] = sqrt(out[0]);
            }
            __syncthreads();
        }
    }

    namespace high_speed {
        // Warp-shuffle norm. s_scratch: ceil(blockDim/32)*sizeof(T). Result in out[0].
        template <typename T>
        __device__ void vector_norm(uint32_t N, T *a, T *out, T *s_scratch)
        {
            uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
            uint32_t size = blockDim.x * blockDim.y * blockDim.z;

            T val = static_cast<T>(0);
            for (uint32_t i = rank; i < N; i += size) val += a[i] * a[i];

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
                if (rank == 0) out[0] = sqrt(val);
            }
            __syncthreads();
        }
    }
}
// ===

#endif