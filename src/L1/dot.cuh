#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

#include "reduce.cuh"

template <typename T>
__device__
void dot(uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g)
{
    for(uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = x[ind] * y[ind];
    }
    g.sync();
    reduce<T>(n, y, g);
}

template <typename T>
__device__
void dot(T *out,
         uint32_t n, 
         T *x, 
         T *y, 
         cgrps::thread_group g)
{
    for(uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        out[ind] = x[ind] * y[ind];
    }
    g.sync();
    reduce<T>(n, out, g);
}