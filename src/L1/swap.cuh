#pragma once


template <typename T>
__device__
void swap(uint32_t n, 
          T alpha, 
          T *x, 
          T *y)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    T temp;

    for(; ind < n; ind += stride){
        temp = x[ind];
        x[ind] = y[ind];
        y[ind] = temp;
    }
}