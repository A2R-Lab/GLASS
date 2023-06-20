#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

#include "reduce.cuh"


/*
    dot product of two vectors
    x and y are input vectors
    store the result in y
    n is the length of the vectors
    g is the thread group
*/
template <typename T>
__device__  __forceinline__
void dot(const uint32_t n, 
          T *x, 
          T *y)
{
    for(uint32_t ind = threadIdx.x; ind < n; ind += blockDim.x){
        y[ind] = x[ind] * y[ind];
    }
    __syncthreads();
    reduce<T>(n, y);
}

/*
    dot product of two vectors
    x and y are input vectors
    store the result in out
    n is the length of the vectors
    g is the thread group
*/
template <typename T>
__device__ __forceinline__
void dot(T *out,
         const uint32_t n, 
         T *x, 
         T *y)
{
    for(uint32_t ind = threadIdx.x; ind < n; ind += blockDim.x){
        out[ind] = x[ind] * y[ind];
    }
    __syncthreads();
    reduce<T>(n, out);
}

template <typename T, uint32_t n>
__device__ __forceinline__
void dot(T *out,
         T *x, 
         T *y)
{
    for(uint32_t ind = threadIdx.x; ind < n; ind += blockDim.x){
        out[ind] = x[ind] * y[ind];
    }
    __syncthreads();
    reduce<T, n>(out);
}