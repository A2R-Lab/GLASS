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
          T *y)
{
        if(TRANSPOSE){
            T res;

            for(std::uint32_t row = threadIdx.x; row < n; row += blockDim.x){
                res = static_cast<T>(0);
                for(std::uint32_t col = 0; col < m; col++){
                    res += A[row*m + col] * x[col];
                }
                y[row] = alpha * res + beta * y[row];
            }
        }
        else{
            T res;

            for(std::uint32_t row = threadIdx.x; row < m; row += blockDim.x){
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
          T *y)
{
        if(TRANSPOSE){
            T res;

            for(std::uint32_t row = threadIdx.x; row < n; row += blockDim.x){
                res = static_cast<T>(0);
                for(std::uint32_t col = 0; col < m; col++){
                    res += A[row*m + col] * x[col];
                }
                y[row] = alpha * res;
            }
        }
        else{
            T res;

            for(std::uint32_t row = threadIdx.x; row < m; row += blockDim.x){
                res = static_cast<T>(0);
                for(std::uint32_t col = 0; col < n; col++){
                    res += A[row + col*m] * x[col];
                }
                y[row] = alpha * res;
            }
        }
}