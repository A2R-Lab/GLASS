#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void reduce(std::uint32_t n,
            T *x,
            cgrps::thread_block g)
{
    const std::uint32_t rank = g.thread_rank();
    const std::uint32_t size = g.size(); 
    unsigned size_left = n;

    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = rank; ind < size_left; ind += size){
            dstTemp[ind] += dstTemp[ind + size_left];
        }	
        // add the odd size adjust if needed
        if (rank == 0 && odd_flag){dstTemp[0] += dstTemp[2*size_left];}
        // sync and repeat
        g.sync();
    }
    // when we get really small sum up what is left
    if (rank == 0){
        for(unsigned ind = 1; ind < size_left; ind++){dstTemp[0] += dstTemp[ind];}
    }
}

template <typename T>
__device__
void dot(std::uint32_t n, 
          T *x, 
          T *y, 
          cgrps::thread_group g)
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = x[ind] * y[ind];
    }
    g.sync();
    reduce<T>(n, y, g);
}