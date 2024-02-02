#pragma once


template <typename T>
__device__  __forceinline__
void infnorm(const uint32_t n,
            T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    uint32_t size_left = n;
    bool odd_flag;

    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (uint32_t i = ind; i < size_left; i += stride){
            x[i] = max(abs(x[i]), abs(x[i + size_left]));
        }	
        // add the odd size adjust if needed
        if (ind == 0 && odd_flag){x[0] = max(abs(x[0]), abs(x[2*size_left]));}
        // sync and repeat
        __syncthreads();
    }
    // when we get really small sum up what is left
    if (ind == 0){
        for(unsigned ind = 1; ind < size_left; ind++){x[0] = max(abs(x[0]), abs(x[ind]));}
    }
}