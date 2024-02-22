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

template <typename T> 
__device__ 
void chol_InPlace(uint32_t n, T *s_A)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for (uint32_t row = 0; row < n-1; row++) {
        // square root
        if (ind == 0) {
            s_A[n*row+row] = pow(s_A[n*row+row], 0.5);
        }
        __syncthreads();

        // normalization
        for (uint32_t k = ind+row+1; k < n; k+= stride) {
            s_A[n*row+k] /= s_A[n*row+row];
        }
        __syncthreads();
        
        // inner prod subtraction
        for(uint32_t j = ind+row+1; j < n; j+= stride) {
            for (uint32_t k = 0; k < row+1; k++) {
                s_A[n*(row+1)+j] -= s_A[n*k+j]*s_A[n*k+row+1];
            }
        }
        __syncthreads();
    }
    if (ind == 0) {
        s_A[n*n-1] = pow(s_A[n*n-1], 0.5);
    }
}