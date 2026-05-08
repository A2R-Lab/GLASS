#pragma once

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T, bool TRANSPOSE = false>
__device__
void gemv(std::uint32_t m,
          std::uint32_t n,
          T alpha,
          T *A,
          T *x,
          T beta, 
          T *y, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
        if(TRANSPOSE){
            T res;

            for(std::uint32_t row = g.thread_rank(); row < n; row += g.size()){
                res = static_cast<T>(0);
                for(std::uint32_t col = 0; col < m; col++){
                    res += A[row*m + col] * x[col];
                }
                y[row] = alpha * res + beta * y[row];
            }
        }
        else{
            T res;

            for(std::uint32_t row = g.thread_rank(); row < m; row += g.size()){
                res = static_cast<T>(0);
                for(std::uint32_t col = 0; col < n; col++){
                    res += A[row + col*m] * x[col];
                }
                y[row] = alpha * res + beta * y[row];
            }
        }
}


template <typename T, bool TRANSPOSE = false>
__device__
void gemv(std::uint32_t m,
          std::uint32_t n,
          T alpha,
          T *A,
          T *x,
          T *y,
          cgrps::thread_group g = cgrps::this_thread_block())
{
        if(TRANSPOSE){
            T res;

            for(std::uint32_t row = g.thread_rank(); row < n; row += g.size()){
                res = static_cast<T>(0);
                for(std::uint32_t col = 0; col < m; col++){
                    res += A[row*m + col] * x[col];
                }
                y[row] = alpha * res;
            }
        }
        else{
            T res;

            for(std::uint32_t row = g.thread_rank(); row < m; row += g.size()){
                res = static_cast<T>(0);
                for(std::uint32_t col = 0; col < n; col++){
                    res += A[row + col*m] * x[col];
                }
                y[row] = alpha * res;
            }
        }
}

// === glass::simple variants ===
namespace simple {
    // y = alpha * A * x + beta * y  (TRANSPOSE=false: m-output; TRANSPOSE=true: n-output)
    template <typename T, bool TRANSPOSE = false>
    __device__
    void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        if (TRANSPOSE) {
            for (uint32_t row = rank; row < n; row += size) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < m; col++) res += A[row * m + col] * x[col];
                y[row] = alpha * res + beta * y[row];
            }
        } else {
            for (uint32_t row = rank; row < m; row += size) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < n; col++) res += A[row + col * m] * x[col];
                y[row] = alpha * res + beta * y[row];
            }
        }
    }

    // y = alpha * A * x  (no beta)
    template <typename T, bool TRANSPOSE = false>
    __device__
    void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        if (TRANSPOSE) {
            for (uint32_t row = rank; row < n; row += size) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < m; col++) res += A[row * m + col] * x[col];
                y[row] = alpha * res;
            }
        } else {
            for (uint32_t row = rank; row < m; row += size) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < n; col++) res += A[row + col * m] * x[col];
                y[row] = alpha * res;
            }
        }
    }
}
// ===