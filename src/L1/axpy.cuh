#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

/*
    Compute the scaled sum of two vectors
    alpha * x + y
    store the result back in y
*/
template <typename T>
__device__
void axpy(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = alpha * x[ind] + y[ind];
    }
}

/*
    Compute the scaled sum of two vectors
    alpha * x + y
    store the result back in y, with a shift for y 
*/
template <typename T>
__device__
void axpy_shifted_y(std::uint32_t n,
          std::uint32_t m, 
          T alpha, 
          T *x, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    int k = n*m;
    int s = 0;
    int ind_y;
    for(std::uint32_t ind = g.thread_rank(); ind < k; ind += g.size()){
        if((s+1)%m == 0){
            s += 1;
        }
        ind_y = ind + (n-m)*s;
        y[ind_y] = alpha * x[ind] + y[ind_y];
    }
}

/*
    Compute the scaled sum of two vectors
    alpha * x + y
    store the result in z
*/
template <typename T>
__device__
void axpy(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          T *z, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        z[ind] = alpha * x[ind] + y[ind];
    }
}

// Debugging version of axpy
template <typename T>
__device__
void axpyD(std::uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          T *z, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        z[ind] = alpha * x[ind] + y[ind];
    }
    // printf("ALPHA: %f\n", alpha);
}
