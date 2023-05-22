#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void swap(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          cgrps::thread_group g)
{
    T temp;
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        temp = x[ind];
        x[ind] = y[ind];
        y[ind] = temp;
    }
}