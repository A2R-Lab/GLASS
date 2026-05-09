#pragma once
#include <cstdint>

template <typename T>
__device__ void prefix_sum_exclusive(T *s_input, T *s_output, int n)
{
    int tid = threadIdx.x;
    s_output[tid] = (tid < n && tid > 0) ? s_input[tid-1] : static_cast<T>(0);
    __syncthreads();
    T tmp;
    for (int d = 1; d < n; d *= 2) {
        __syncthreads();
        if (tid < n && tid >= d) tmp = s_output[tid] + s_output[tid-d];
        __syncthreads();
        if (tid < n && tid >= d) s_output[tid] = tmp;
    }
}

template <typename T>
__device__ void prefix_sum_inclusive(T *s_input, T *s_output, int n)
{
    int tid = threadIdx.x;
    if (tid < n) s_output[tid] = s_input[tid];
    __syncthreads();
    T tmp;
    for (int d = 1; d < n; d *= 2) {
        __syncthreads();
        if (tid < n && tid >= d) tmp = s_output[tid-d] + s_output[tid];
        __syncthreads();
        if (tid < n && tid >= d) s_output[tid] = tmp;
    }
}
