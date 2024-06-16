#ifndef GEMM_H
#define GEMM_H

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

///TODO: error checking
///TODO: transpose A

template <typename T, bool TRANSPOSE_B>
__device__
void gemm(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T beta,
          T *C, 
          cgrps::thread_group g)
{
    if(TRANSPOSE_B){
        const unsigned max = m*n;
        uint32_t element, ind, row, col;
        T res;

        for(element = g.thread_rank(); element < max; element += g.size()){
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
        const unsigned max = m*k;
        uint32_t element, ind, row, col;
        T res;

        for(element = g.thread_rank(); element < max; element += g.size()){
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


template <typename T, bool TRANSPOSE_B = false>
__device__
void gemm(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T *C, 
          cgrps::thread_group g = cgrps::this_thread_block())
{
    if(TRANSPOSE_B){
        const unsigned max = m*n;
        uint32_t element, ind, row, col;
        T res;

        for(element = g.thread_rank(); element < max; element += g.size()){
            res = static_cast<T>(0);
            row = element % m;
            col = element / m;

            for(ind = 0; ind < n; ind++){
                res += A[ind*m + row] * B[ind*n + col];
            }

            C[col*m + row] = alpha * res;
        }
    }
    else{
        const unsigned max = m*k;
        uint32_t element, ind, row, col;
        T res;

        for(element = g.thread_rank(); element < max; element += g.size()){
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

/*
    dot product of two vectors
    s_temp is temporary shared memory to incrementally store the result
    this method is inteded to be run by a single thread, as part of a larger matmul operation
    x and y are input vectors
    store the result in out
    n is the length of the vectors
    x_stride is the stride of x
    y_stride is the stride of y
    g is the thread group
*/
template <typename T>
__device__
void dot_single(T * out,
         T * s_temp,
         uint32_t n, 
         T *x,
         int x_stride,
         T *y, 
         int y_stride,
         cgrps::thread_group g = cgrps::this_thread_block()) {
    s_temp[threadIdx.x] = 0; 
    for(uint32_t i = 0; i < n; i++){
        s_temp[threadIdx.x] += x[i * x_stride] * y[i * y_stride];
    }
    
    *out = s_temp[threadIdx.x];
}

/*
    Function for a single dot product computation as part of a larger matrix multiply
    A and B are pointers to the start of the input matrices, stored in column major format
    ld_A and ld_B are the number of rows in A and B, since we are using column major storage
    A_vec_ind and B_vec_ind are row/column indices of the components of A and B which will be multiplied
    If TRANSPOSE_A and TRANSPOSE_B are each false, we then want the dot product of the A_vec_ind-th row of A and the B_vec_ind-th column of B
    If TRANSPOSE_A is true, we want the dot product of the A_vec_ind-th column of A and the B_vec_ind-th column of B
    If TRANSPOSE_B is true, we want the dot product of the A_vec_ind-th row of A and the B_vec_ind-th row of B

    store the result in out
*/
template <typename T, bool TRANSPOSE_A, bool TRANSPOSE_B>
__device__
void dotMM(T * out,
           T * s_temp,
           uint32_t n, 
           T * A,
           int ld_A,
           int A_vec_ind,
           T * B, 
           int ld_B,
           int B_vec_ind,
           cgrps::thread_group g = cgrps::this_thread_block()) {
    
    int a_start_ind;
    int a_stride;
    if (TRANSPOSE_A) {
        // if transpose A is true, I want to take the A_vec_ind-th column of A
        a_start_ind = ld_A * A_vec_ind;
        a_stride = 1;
    } else {
        // if transpose A is false, I want to take the A_vec_ind-th row of A
        a_start_ind = A_vec_ind;
        a_stride = ld_A;
    }

    int b_start_ind;
    int b_stride;
    if (TRANSPOSE_B){
        // if transpose B is true, I want to take the B_vec_ind-th row of B
        b_start_ind = B_vec_ind;
        b_stride = ld_B;
    } else {
        // if transpose B is false, I want to take the B_vec_ind-th column of B
        b_start_ind = ld_B * B_vec_ind;
        b_stride = 1;
    }

    dot_single(out, s_temp, n, &A[a_start_ind], a_stride, &B[b_start_ind], b_stride, g);
}

template <typename T, bool TRANSPOSE_A, bool TRANSPOSE_B>
__device__
void gemm_v2(std::uint32_t A_rows,
          std::uint32_t A_cols,
          std::uint32_t B_rows,
          std::uint32_t B_cols,
          T *A,
          int ld_A, 
          T *B,
          int ld_B,
          T *C,
          T * s_temp,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    int C_size;
    int vector_length;
    if (TRANSPOSE_A) {
        C_size = A_cols * B_cols;
        vector_length = A_rows;
        for (uint32_t i = g.thread_rank(); i < C_size; i += g.size()){
            int a_vec_ind = i % A_cols;
            int b_vec_ind = i / A_cols;
            dotMM<T, true, false>(&C[i], s_temp, vector_length, A, ld_A, a_vec_ind, B, ld_B, b_vec_ind, g);
        }
    } else if (TRANSPOSE_B)
    {
        C_size = A_rows * B_rows;
        vector_length = A_cols;
        for (uint32_t i = g.thread_rank(); i < C_size; i += g.size()){
            int a_vec_ind = i % A_rows;
            int b_vec_ind = i / A_rows;
            dotMM<T, false, true>(&C[i], s_temp, vector_length, A, ld_A, a_vec_ind, B, ld_B, b_vec_ind, g);
        }
    } else {
        C_size = A_rows * B_cols;
        vector_length = A_cols;
        for (uint32_t i = g.thread_rank(); i < C_size; i += g.size()){
            int a_vec_ind = i % A_rows;
            int b_vec_ind = i / A_rows;
            dotMM<T, false, false>(&C[i], s_temp, vector_length, A, ld_A, a_vec_ind, B, ld_B, b_vec_ind, g);
        }
    }
}

template <typename T>
__device__ void simple_submatrix_gemm(T *s_C, const T *s_A, const T *s_B, int subA_rows, int subA_cols, int subB_cols,
                                      int ld_A, int ld_B, int ld_C)
{
    int row = threadIdx.x / subB_cols; 
    int col = threadIdx.x % subB_cols; 

    if (row < subA_rows && col < subB_cols)
    {
        T sum = 0;
        for (int k = 0; k < subA_cols; ++k)
        {
            sum += s_A[row + k * ld_A] * s_B[k * ld_B + col];
        }

        s_C[row + col * ld_C] = sum;
    }
}

/*for C=αA+B
more convenient than blas routine sometimes*/
template <typename T>
__device__
void matrixAlphaAdd(T alpha, 
                    T *A, 
                    T *B, 
                    T *C, 
                    std::uint32_t rows, 
                    std::uint32_t cols,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    std::uint32_t n = rows * cols;

    for (std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()) {
        std::uint32_t row = ind % rows;
        std::uint32_t col = ind / rows;

        C[ind] = alpha * A[ind] + B[ind];
    }
}


#endif