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

/*
    Compute the scaled sum of two vectors
    alpha * x + beta * y
    store the result in z
*/
template <typename T>
__device__
void axpby(std::uint32_t n, 
          T alpha, 
          T *x,
		  T beta, 
          T *y, 
          T *z, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        z[ind] = alpha * x[ind] + beta * y[ind];
    }
}

/*
    Clip z between l and u
*/
template <typename T>
__device__
void clip(std::uint32_t n, 
          T *x,
          T *l, 
          T *u, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
		if ( x[ind]  < l[ind])
			x[ind] = l[ind];
		else if (x[ind > u[ind]])
			x[ind] = u[ind];
    }
}