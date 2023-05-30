#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void reduce(uint32_t n,
            T *x,
            cgrps::thread_group g)
{
    const uint32_t rank = g.thread_rank();
    const uint32_t size = g.size(); 
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
        g.sync();
    }
    // when we get really small sum up what is left
    if (rank == 0){
        for(unsigned ind = 1; ind < size_left; ind++){x[0] += x[ind];}
    }
}

template <typename T>
__device__
void reduce(T *out,
            uint32_t n,
            T *x,
            cgrps::thread_group g)
{

    for(int i=g.thread_rank(); i < n; i += g.size()){ out[i] = x[i]; }
    g.sync();
    reduce(n, out, g);
}
