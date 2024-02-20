#pragma once


template <typename T, bool TRANSPOSE = false>
__device__
void gemv(uint32_t m,
          uint32_t n,
          T alpha,
          T *A,
          T *x,
          T beta, 
          T *y)
{
    T res;
    uint32_t row, col;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    
    if(TRANSPOSE){

        for(row = ind; row < n; row += stride){
            res = static_cast<T>(0);
            for(col = 0; col < m; col++){
                res += A[row*m + col] * x[col];
            }
            y[row] = alpha * res + beta * y[row];
        }
    }
    else{

        for(row = ind; row < m; row += stride){
            res = static_cast<T>(0);
            for(col = 0; col < n; col++){
                res += A[row + col*m] * x[col];
            }
            y[row] = alpha * res + beta * y[row];
        }
    }
}


template <typename T, bool TRANSPOSE = false>
__device__
void gemv(uint32_t m,
          uint32_t n,
          T alpha,
          T *A,
          T *x,
          T *y)
{
    T res;
    uint32_t row, col;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    
    if(TRANSPOSE){
        for(row = ind; row < m; row += stride){
            res = static_cast<T>(0);
            for(col = 0; col < n; col++){
                res += A[row*n + col] * x[col];
            }
            y[row] = alpha * res;
        }
    }
    else{
        for(row = ind; row < m; row += stride){
            res = static_cast<T>(0);
            for(col = 0; col < n; col++){
                res += A[row + col*m] * x[col];
            }
            y[row] = alpha * res;
        }
    }
}