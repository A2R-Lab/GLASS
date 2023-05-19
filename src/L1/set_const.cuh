#pragma once


template <typename T>
__device__
void set_const(uint32_t n, 
          T alpha, 
          T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for(; ind < n; ind += stride){
        x[ind] = alpha;
    }
}
