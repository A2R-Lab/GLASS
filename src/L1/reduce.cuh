#include <cstdint>
#include <cooperative_groups.h>
#include "copy.cuh"
namespace cgrps = cooperative_groups;

template <typename T>
__device__  __forceinline__
void reduce(const uint32_t n,
            T *x)
{
    const uint32_t rank = threadIdx.x;
    const uint32_t size = blockDim.x; 
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
        __syncthreads();
    }
    // when we get really small sum up what is left
    if (rank == 0){
        for(unsigned ind = 1; ind < size_left; ind++){x[0] += x[ind];}
    }
}

template <typename T, uint32_t n>
__device__ __forceinline__
void reduce(
            T *x)
{
    const uint32_t rank = threadIdx.x;
    const uint32_t size = blockDim.x; 
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
        __syncthreads();
    }
    // when we get really small sum up what is left
    if (rank == 0){
        for(unsigned ind = 1; ind < size_left; ind++){x[0] += x[ind];}
    }
}

template <typename T>
__device__ __forceinline__
void reduce(T *out,
            const uint32_t n,
            T *x)
{

    copy<T>(n, x, out);
    __syncthreads();

    const uint32_t rank = threadIdx.x;
    const uint32_t size = blockDim.x; 
    unsigned size_left = n;

    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = rank; ind < size_left; ind += size){
            out[ind] += out[ind + size_left];
        }	
        // add the odd size adjust if needed
        if (rank == 0 && odd_flag){out[0] += out[2*size_left];}
        // sync and repeat
        __syncthreads();
    }
    // when we get really small sum up what is left
    if (rank == 0){
        for(unsigned ind = 1; ind < size_left; ind++){out[0] += out[ind];}
    }
}

template <typename T, uint32_t n>
__device__ __forceinline__
void reduce(T *out,
            T *x)
{

    copy<T>(n, x, out);
    __syncthreads();

    const uint32_t rank = threadIdx.x;
    const uint32_t size = blockDim.x; 
    unsigned size_left = n;

    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = rank; ind < size_left; ind += size){
            out[ind] += out[ind + size_left];
        }	
        // add the odd size adjust if needed
        if (rank == 0 && odd_flag){out[0] += out[2*size_left];}
        // sync and repeat
        __syncthreads();
    }
    // when we get really small sum up what is left
    if (rank == 0){
        for(unsigned ind = 1; ind < size_left; ind++){out[0] += out[ind];}
    }
}
