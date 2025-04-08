#ifndef DOT_H
#define DOT_H

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
__device__
void dot(uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = x[ind] * y[ind];
    }
    g.sync();
    reduce<T>(n, y, g);
}

/*
    dot product of two vectors
    x and y are input vectors
    store the result in out
    n is the length of the vectors
    g is the thread group
*/
template <typename T>
__device__
void dot(T *out,
         uint32_t n, 
         T *x, 
         T *y, 
         cgrps::thread_group g = cgrps::this_thread_block())
{
    for(uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        out[ind] = x[ind] * y[ind];
    }
    g.sync();
    reduce<T>(n, out, g);
}

/*
    dot product of two vectors
    x and y are input vectors
    store the result in out
    n is the length of vector y
    s is the shift in the length of x (s+n)
    g is the thread group
*/
template <typename T>
__device__
void dot_shifted(T *out,
         uint32_t n,
         uint32_t s, 
         T *x, 
         T *y, 
         cgrps::thread_group g = cgrps::this_thread_block())
{
    for(uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        out[ind] = x[ind+s] * y[ind];
    }
    g.sync();
    reduce<T>(n, out, g);
}

/*
    dot product of two vectors
    x and [y;z] are input vectors
    store the result in out
    m is the length of vector x
    n is the length of vector y
    g is the thread group
*/
template <typename T>
__device__
void dot_concatenate_right(T *out,
        uint32_t m, 
        uint32_t n,
         T *x, 
         T *y,
         T *z,
         cgrps::thread_group g = cgrps::this_thread_block())
{
    for(uint32_t ind = g.thread_rank(); ind < m; ind += g.size()){
        if (ind < n){
            out[ind] = x[ind] * y[ind];
        }
        else{
            out[ind] = x[ind] * z[ind-n];
        }   
    }
    g.sync();
    reduce<T>(n, out, g);
}

/*
    dot product of two vectors
    [x;y] and z are input vectors
    store the result in out
    m is the length of vector x
    n is the length of vector z
    g is the thread group
*/
template <typename T>
__device__
void dot_concatenate_left(T *out,
        uint32_t m, 
        uint32_t n,
         T *x, 
         T *y,
         T *z,
         cgrps::thread_group g = cgrps::this_thread_block())
{
    for(uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        if (ind < m){
            out[ind] = x[ind] * z[ind];
        }
        else{
            out[ind] = y[ind-m] * z[ind];
        }
    }
    g.sync();
    reduce<T>(n, out, g);
}

#endif