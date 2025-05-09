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
    int k = m*m;
    int s = 0;
    int ind_y;
    for(std::uint32_t ind = g.thread_rank(); ind < k; ind += g.size()){
        ind_y = ind + (n - m) * (ind / m); // Calculate ind_y based on ind and m
        y[ind_y] += alpha * x[ind];       // Perform the update directly
    }
    // g.sync(); // Synchronize threads within the group to avoid race conditions
}

template <typename T>
__device__
void axpy_shifted_y_tmp(std::uint32_t n,
          std::uint32_t m, 
          T alpha, 
          T *x, 
          T *y)
{
    int k = m*m;
    int s = 0;
    int ind_y;
    for(std::uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x; ind < k; ind += blockDim.x * blockDim.y){
        ind_y = ind + (n - m) * (ind / m); // Calculate ind_y based on ind and m
        y[ind_y] += alpha * x[ind];       // Perform the update directly
    }
}

/*
    Compute the scaled sum of two vectors
    alpha * x + y
    store the result back in y, with a shift for y 
*/
// template <typename T>
// __device__
// void axpy_shifted_y(std::uint32_t n,
//           std::uint32_t m, 
//           T alpha, 
//           T *x, 
//           T *y)
// {
//     int k = m*m;
//     int s = 0;
//     int ind_y;
//     for(std::uint32_t ind = 0; ind < k; ind += 1){
//         ind_y = ind + (n-m)*s;
//         y[ind_y] = alpha * x[ind] + y[ind_y];
//         if((ind+1)%m == 0){
//             s += 1;
//         }
//     }
// }

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
