#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void copy(std::uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g)
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = x[ind];
    }
}