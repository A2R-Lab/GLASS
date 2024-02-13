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
void chol_InPlace (unit32_t n,
                        T *s_A,
                        cgrps::thread_group g = cgrps::this_thread_block())
{
    for (unit32_t row = 0; row < n; row++) {
        if (g.thread_rank() == 0){
            T sum = 0;
            T val = s_A[n*row+row]; //entry Ljj
            for(std:init32_t row_l = 0 ; row_l < row; row_l++) {
                sum += pow(s_A[row_l*n+row],2);
            }
            s_A[row*n+row] = sqrt(val - sum);

        }
        g.sync(); //here we computed the diagonal entry of the Matrix
        
        // compute the rest of the row  
        for(unit32_t col = g.thread_rank()+ row +1; col < n; col += g.size()) 
        {
            T sum = 0;
            for(unit32_t k = 0; k < row; k++) {
                sum += s_A[k*n+col]*s_A[k*n+row];
            }
            s_A[row*n+col] = (1.0/s_A[row*n+row])*(s_A[row*n+col]-sum);
        }
        g.sync();
    }
}

template <typename T> 
__device__ 
void chol_InPlace_vec(uint32_t n,
                        T *s_A)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for (uint32_t row = 0; row < n-1; row++) {
        // square root
        if (ind == 0) {
            s_A[n*row+row] = pow(s_A[n*row+row], 0.5);
        }
        __syncthreads(); // don't think I need?

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