#pragma once


template <typename T>
__device__ inline T sqrtT(T x) {
    return sqrt(x);
}

// Specialization for float to use sqrtf
template <>
__device__ inline float sqrtT<float>(float x) {
    return sqrtf(x);
}

template <typename T>
__device__  __forceinline__
void l2norm(const uint32_t n,
            T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    unsigned size_left = n;
    bool odd_flag;

    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (uint32_t i = ind; i < size_left; i += stride){
            x[i] += x[i + size_left] * x[i + size_left];
        }	
        // add the odd size adjust if needed
        if (ind == 0 && odd_flag){x[0] += x[2*size_left] * x[2*size_left];}
        // sync and repeat
        __syncthreads();
    }
    // when we get really small sum up what is left
    if (ind == 0){
        for(uint32_t i = 1; i < size_left; i++){x[0] += x[i] * x[i];}
        x[0] = sqrtT(x[0]);
    }
}