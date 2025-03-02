#pragma once

#ifndef COPY_H
#define COPY_H

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
void copy(std::uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
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
void copy(std::uint32_t n,
          T alpha,
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = alpha * x[ind];
    }
}

#endif