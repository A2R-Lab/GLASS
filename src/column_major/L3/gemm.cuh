#pragma once


template <typename T, bool TRANSPOSE_B>
__device__
void gemm(uint32_t m,
          uint32_t n,
          uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T beta,
          T *C)
{
    uint32_t element, row, col;
    T res;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    const unsigned max = m*n;

    if(TRANSPOSE_B){
        for(element = ind; element < max; element += stride){
            res = static_cast<T>(0);
            row = element % m;
            col = element / m;

            for(ind = 0; ind < n; ind++){
                res += A[ind*m + row] * B[ind*n + col];
            }

            C[col*m + row] = alpha * res + beta * C[col*m + row];
        }
    }
    else{
        for(element = ind; element < max; element += stride){
            res = static_cast<T>(0);
            row = element % m;
            col = element / m;
            for(ind = 0; ind < n; ind++){
                res += A[ind*m + row] * B[col*n + ind];
            }

            C[col*m + row] = alpha * res + beta * C[col*m + row];
        }
    }
}

// A = m x n
// TRANSPOSE(B) = n x k
// C = m x k
template <typename T, bool TRANSPOSE_B = false>
__device__
void gemm(uint32_t m,
          uint32_t n,
          uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T *C)
{
    uint32_t element, row, col;
    T res;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    const unsigned max = m*k;

    if(TRANSPOSE_B){
        for(element = ind; element < max; element += stride){
            res = static_cast<T>(0);
            row = element % m;
            col = element / m;

            for(ind = 0; ind < n; ind++){
                res += A[ind*m + row] * B[ind*k + col];
            }

            C[col*m + row] = alpha * res;
        }
    }
    else{
        for(element = ind; element < max; element += stride){
            res = static_cast<T>(0);
            row = element % m;
            col = element / m;

            for(ind = 0; ind < n; ind++){
                res += A[ind*m + row] * B[col*n + ind];
            }

            C[col*m + row] = alpha * res;
        }
    }
}