#pragma once

template <typename T>
__device__  __forceinline__
void l2norm(const uint32_t n,
            T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for (uint32_t i = ind; i < n; i += stride) {
        x[i] *= x[i];
    }
    __syncthreads();
    reduce(n, x);
    if (ind == 0) {
        x[0] = sqrtf(x[0]);
    }
}