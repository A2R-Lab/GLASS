#pragma once
#include <cstdint>

// default threadIdx-based halving reduce; result in x[0]
template <typename T>
__device__ void reduce(uint32_t n, T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t left = n;
    while (left > 3) {
        bool odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = rank; i < left; i += size) x[i] += x[i + left];
        if (rank == 0 && odd) x[0] += x[2*left];
        __syncthreads();
    }
    if (rank == 0) { for (uint32_t i = 1; i < left; i++) x[0] += x[i]; }
}

template <typename T, uint32_t N>
__device__ void reduce(T *x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t left = N;
    while (left > 3) {
        bool odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = rank; i < left; i += size) x[i] += x[i + left];
        if (rank == 0 && odd) x[0] += x[2*left];
        __syncthreads();
    }
    if (rank == 0) { for (uint32_t i = 1; i < left; i++) x[0] += x[i]; }
}

namespace low_memory {
    template <typename T>
    __device__ void reduce(uint32_t n, T *x)
    {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            for (uint32_t i = 1; i < n; i++) x[0] += x[i];
        __syncthreads();
    }

    template <typename T, uint32_t N>
    __device__ void reduce(T *x)
    {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            for (uint32_t i = 1; i < N; i++) x[0] += x[i];
        __syncthreads();
    }
}

namespace high_speed {
    // warp-shuffle + inter-warp reduce; s_scratch: ceil(blockDim/32)*sizeof(T); result in x[0]
    template <typename T>
    __device__ void reduce(uint32_t n, T *x, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < n; i += size) val += x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) x[0] = val;
        }
        __syncthreads();
    }

    template <typename T, uint32_t N>
    __device__ void reduce(T *x, T *s_scratch)
    {
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T val = static_cast<T>(0);
        for (uint32_t i = rank; i < N; i += size) val += x[i];
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        uint32_t lane = rank & 31, warp = rank >> 5;
        if (lane == 0) s_scratch[warp] = val;
        __syncthreads();
        uint32_t nw = (size + 31) / 32;
        if (rank < 32) {
            val = (rank < nw) ? s_scratch[rank] : static_cast<T>(0);
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
            if (rank == 0) x[0] = val;
        }
        __syncthreads();
    }
}
