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

#endif