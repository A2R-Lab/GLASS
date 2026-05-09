#pragma once

#ifndef DOT_H
#define DOT_H

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

#include "reduce.cuh"


/*
    dot product of two vectors
    x and y are input vectors
    store the result in y
    n is the length of the vectors
    g is the thread group
*/
template <typename T>
__device__
void dot(uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = x[ind] * y[ind];
    }
    g.sync();
    reduce<T>(n, y, g);
}

/*
    dot product of two vectors
    x and y are input vectors
    store the result in out
    n is the length of the vectors
    g is the thread group
*/
template <typename T>
__device__
void dot(T *out,
         uint32_t n, 
         T *x, 
         T *y, 
         cgrps::thread_group g = cgrps::this_thread_block())
{
    for(uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        out[ind] = x[ind] * y[ind];
    }
    g.sync();
    reduce<T>(n, out, g);
}


template <typename T, uint32_t n>
__device__ __forceinline__
void dot(T *out,
         T *x,
         T *y)
{
    for(uint32_t ind = threadIdx.x; ind < n; ind += blockDim.x){
        out[ind] = x[ind] * y[ind];
    }
    __syncthreads();
    reduce<T>(n, out);
}

// === glass::simple variants ===
namespace simple {
    namespace low_memory {
        // Elementwise multiply into out, then thread 0 accumulates.
        // out must be length n scratch; result in out[0].
        template <typename T>
        __device__ void dot(uint32_t n, T *x, T *y, T *out)
        {
            uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
            uint32_t size = blockDim.x * blockDim.y * blockDim.z;
            for (uint32_t i = rank; i < n; i += size) out[i] = x[i] * y[i];
            __syncthreads();
            if (rank == 0) {
                for (uint32_t i = 1; i < n; i++) out[0] += out[i];
            }
            __syncthreads();
        }
    }

    namespace high_speed {
        // Warp-shuffle dot product. s_scratch: ceil(blockDim/32)*sizeof(T). Result in out[0].
        template <typename T>
        __device__ void dot(uint32_t n, T *x, T *y, T *out, T *s_scratch)
        {
            uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
            uint32_t size = blockDim.x * blockDim.y * blockDim.z;

            T val = static_cast<T>(0);
            for (uint32_t i = rank; i < n; i += size) val += x[i] * y[i];

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
                if (rank == 0) out[0] = val;
            }
            __syncthreads();
        }
    }
}
// ===


#endif