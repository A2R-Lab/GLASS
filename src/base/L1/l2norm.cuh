#pragma once
#include <cstdint>
#include <math.h>
#include "reduce.cuh"

namespace low_memory {
    template <typename T>
    __device__ void l2norm(uint32_t n, T *x)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t i = rank; i < n; i += size) x[i] *= x[i];
        __syncthreads();
        if (rank == 0) {
            for (uint32_t i = 1; i < n; i++) x[0] += x[i];
            x[0] = sqrtf(x[0]);
        }
        __syncthreads();
    }
}

namespace high_speed {
    template <typename T>
    __device__ void l2norm(uint32_t n, T *x, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < n; i += size) val += x[i]*x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) x[0] = sqrtf(val);
        }
        __syncthreads();
    }

    template <typename T, uint32_t N>
    __device__ void l2norm(T *x, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < N; i += size) val += x[i]*x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) x[0] = sqrtf(val);
        }
        __syncthreads();
    }
}
