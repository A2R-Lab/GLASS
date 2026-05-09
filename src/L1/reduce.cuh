#pragma once

#ifndef REDUCE_H
#define REDUCE_H

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void reduce(uint32_t n,
            T *x,
            cgrps::thread_group g)
{
    const uint32_t rank = g.thread_rank();
    const uint32_t size = g.size(); 
    unsigned size_left = n;

    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = rank; ind < size_left; ind += size){
            x[ind] += x[ind + size_left];
        }	
        // add the odd size adjust if needed
        if (rank == 0 && odd_flag){x[0] += x[2*size_left];}
        // sync and repeat
        g.sync();
    }
    // when we get really small sum up what is left
    if (rank == 0){
        for(unsigned ind = 1; ind < size_left; ind++){x[0] += x[ind];}
    }
}

template <typename T>
__device__
void reduce(T *out,
            uint32_t n,
            T *x,
            cgrps::thread_group g)
{

    for(int i=g.thread_rank(); i < n; i += g.size()){ out[i] = x[i]; }
    g.sync();
    reduce(n, out, g);
}

// === glass::simple variants ===
namespace simple {
    namespace low_memory {
        // Thread 0 accumulates sequentially; all other threads wait at __syncthreads().
        // No scratch needed; result in x[0].
        template <typename T>
        __device__ void reduce(uint32_t n, T *x)
        {
            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
                for (uint32_t i = 1; i < n; i++) x[0] += x[i];
            }
            __syncthreads();
        }
    }

    namespace high_speed {
        // Warp-shuffle + shared-memory inter-warp reduction.
        // s_scratch: at least ceil(blockDim/32) * sizeof(T) bytes. Result in x[0].
        template <typename T>
        __device__ void reduce(uint32_t n, T *x, T *s_scratch)
        {
            uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
            uint32_t size = blockDim.x * blockDim.y * blockDim.z;

            T val = static_cast<T>(0);
            for (uint32_t i = rank; i < n; i += size) val += x[i];

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

#endif