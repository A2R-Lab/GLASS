/**
 * @brief Perform a Cholesky decomposition in place.
 *
 * Performs a Cholesky decomposition on the square matrix @p s_A, storing the result in the
 * lower triangular portion of @p s_A. .
 *
 * @param T* s_A: a square symmetric matrix , column - major order.
 * @param  int n: number of cols/rows in a square matrix s_A (n*n)
 *
 */

#include <cstdint>
#include <cooperative_groups.h>
using namespace cooperative_groups;

template <typename T> 
__device__ 
void cholDecomp_InPlace_c (uint32_t n,
                        T *s_A,
                        thread_group g = this_thread_block())
{
    bool decomposition_check = true;
    for (uint32_t row = 0; row < n; row++) {
        if (g.thread_rank() == 0){
            T sum = 0;
            T val = s_A[n*row+row]; //entry Ljj
            for(uint32_t row_l = 0 ; row_l < row; row_l++) {
                sum += pow(s_A[row_l*n+row],2);
            }
            s_A[row*n+row] = sqrt(val - sum);
        }
        g.sync(); //here we computed the diagonal entry of the Matrix
        
        // compute the rest of the row  
        for(uint32_t col = g.thread_rank()+ row +1; col < n; col += g.size()) 
        {
            T sum = 0;
            for(uint32_t k = 0; k < row; k++) {
                sum += s_A[k*n+col]*s_A[k*n+row];
            }
            s_A[row*n+col] = (1.0/s_A[row*n+row])*(s_A[row*n+col]-sum);
        }
        g.sync();
    }
}

template <typename T> 
__device__ 
void cholDecomp_check (uint32_t n, int *flag,
                        T *s_A,
                        thread_group g = this_thread_block())
{
    // Perform Cholesky decomposition with error checking
    // If any diagonal entry is non-positive, set flag to false
    // and return early.
    for (uint32_t row = 0; row < n; row++) {
        if (g.thread_rank() == 0){
            T sum = 0;
            T val = s_A[n*row+row]; //entry Ljj
            for(uint32_t row_l = 0 ; row_l < row; row_l++) {
                sum += pow(s_A[row_l*n+row],2);
            }
            s_A[row*n+row] = sqrt(val - sum);
            if (s_A[row*n+row] <= 0 || isnan(s_A[row*n+row])) {
                // decomposition failed, optionally set an error flag or return
                *flag = 1; // or set a global error flag
                return;
            }
        }
        g.sync(); //here we computed the diagonal entry of the Matrix

        // compute the rest of the row  
        for(uint32_t col = g.thread_rank()+ row +1; col < n; col += g.size()) 
        {
            T sum = 0;
            for(uint32_t k = 0; k < row; k++) {
                sum += s_A[k*n+col]*s_A[k*n+row];
            }
            s_A[row*n+col] = (1.0/s_A[row*n+row])*(s_A[row*n+col]-sum);
        }
        g.sync();
        
    }
}

/**
 * @brief Perform a Cholesky decomposition with error checking using a single thread.
 *
 * Performs a Cholesky decomposition on the square matrix @p s_A, storing the result in the
 * lower triangular portion of @p s_A. If any diagonal entry is non-positive or NaN, sets @p err to 1.
 *
 * @param uint32_t n: Number of columns/rows in the square matrix s_A (n*n).
 * @param int* err: Pointer to an integer flag that is set to 1 if decomposition fails.
 * @param T* s_A: A square symmetric matrix, column-major order.
 */
template <typename T>
__device__
void cholDecomp_check_singleThread(uint32_t n, int *err, T *s_A)
{
    // Only one thread does the work
    if (threadIdx.x == 0) {
        for (uint32_t row = 0; row < n; row++) {
            T sum = 0;
            T val = s_A[n*row+row]; // entry Ljj
            for (uint32_t row_l = 0; row_l < row; row_l++) {
                sum += pow(s_A[row_l*n+row], 2);
            }
            s_A[row*n+row] = sqrt(val - sum);
            if (s_A[row*n+row] <= 0 || isnan(s_A[row*n+row])) {
                *err = 1;
                return;
            }
        
        // Here we computed the diagonal entry of the Matrix

        // Compute the rest of the row
            for (uint32_t col = row + 1; col < n; col++) {
                T sum2 = 0;
                for (uint32_t k = 0; k < row; k++) {
                    sum2 += s_A[k*n+col] * s_A[k*n+row];
                }
                s_A[row*n+col] = (1.0 / s_A[row*n+row]) * (s_A[row*n+col] - sum2);
            }
        }
    }
}

/**
 * @brief Perform a Cholesky decomposition with error checking using multiple thread.
 *
 * Performs a Cholesky decomposition on the square matrix @p s_A, storing the result in the
 * lower triangular portion of @p s_A. If any diagonal entry is non-positive or NaN, sets @p err to 1.
 *
 * @param uint32_t n: Number of columns/rows in the square matrix s_A (n*n).
 * @param int* err: Pointer to an integer flag that is set to 1 if decomposition fails.
 * @param T* s_A: A square symmetric matrix, column-major order.
 */
template <typename T>
__device__
void cholDecomp_check_multiThread(uint32_t n, int *err, T *s_A)
{
    // Process rows sequentially (not in parallel)
    for (uint32_t row = 0; row < n; row++) {
        
        // Only thread 0 computes the diagonal element
        if (threadIdx.x == 0) {
            T sum = 0;
            T val = s_A[n*row+row]; // entry Ljj
            for (uint32_t row_l = 0; row_l < row; row_l++) {
                sum += pow(s_A[row_l*n+row], 2);
            }
            s_A[row*n+row] = sqrt(val - sum);
            if (s_A[row*n+row] <= 0 || isnan(s_A[row*n+row])) {
                *err = 1;
                // return;
            }
        }
        
        // Synchronize all threads in the block
        __syncthreads();
        
        // Check if error occurred (all threads check)
        if (*err == 1) {
            return;
        }
        
        // Multiple threads compute different columns of the same row
        // for (uint32_t col = row + 1 + threadIdx.x + blockIdx.x * blockDim.x; 
        //      col < n; 
        //      col += blockDim.x * gridDim.x) {
        for (uint32_t col = row + 1 + threadIdx.x; 
            col < n; 
            col += blockDim.x) {  // Remove blockIdx.x and gridDim.x
            T sum2 = 0;
            for (uint32_t k = 0; k < row; k++) {
                sum2 += s_A[k*n+col] * s_A[k*n+row];
            }
            s_A[row*n+col] = (1.0 / s_A[row*n+row]) * (s_A[row*n+col] - sum2);
        }
        
        // Synchronize before moving to next row
        __syncthreads();
    }
}
