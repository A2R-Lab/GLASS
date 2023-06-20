#pragma once
#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

/*
    *  copy
    *  =====
    *
    *  Copies the contents of x into y.
    *
    *  Parameters
    *  ----------
    *  n : std::uint32_t
    *      The number of elements to copy.
    *  x : T*
    *      The array to copy from.
    *  y : T*
    *      The array to copy to.
    *  g : cgrps::thread_group
    *      The thread group to use.
    */
template <typename T>
__device__
void copy(const uint32_t n, 
          T *x, 
          T *y)
{
    for(std::uint32_t ind = threadIdx.x; ind < n; ind += blockDim.x){
        y[ind] = x[ind];
    }
}

/*
    * copy
    * ====
    * 
    *  Copies the scaled contents of x into y.
    *  Scale each element in x by alpha
    * 
    * Parameters
    * ----------
    * n : std::uint32_t
    *    The number of elements to copy.
    * alpha : T
    *   The scaling factor
    * x : T*
    *  The array to copy from.
    * y : T*
    * The array to copy to.
    * g : cgrps::thread_group
    * The thread group to use.
*/
template <typename T>
__device__
void copy(const std::uint32_t n,
          T alpha,
          T *x, 
          T *y)
{
    for(std::uint32_t ind = threadIdx.x; ind < n; ind += blockDim.x){
        y[ind] = alpha * x[ind];
    }
}