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