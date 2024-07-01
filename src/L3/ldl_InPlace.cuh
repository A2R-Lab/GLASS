/**
 * @brief Perform a LDL' decomposition in place.
 *
 * Note:
 * 1. LDL' decomposition is also known as square-root free Cholesky decomposition
 * 2. quasi-definite (eigen values are positive or negative, not zero) matrix admits unique LDL' decomposition
 *
 * Performs a LDL' decomposition on the square matrix @p s_A, storing the result in the
 * lower triangular portion of @p s_A.
 *
 * D = the diagonal entry of s_A.
 * L = lower triangular of s_A with diagonal entries replaced by 1.
 *
 * @param T* s_A: a square symmetric matrix , column - major order.
 * @param  int n: number of cols/rows in a square matrix s_A (n*n)
 *
 */

// by Shaohui Yang, 2024.07.01

template<typename T>
__device__
void ldl_InPlace(uint32_t n, T *s_A, T *s_D) {
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for (uint32_t row = 0; row < n - 1; row++) {
        // normalization
        for (uint32_t k = ind + row + 1; k < n; k += stride) {
            s_A[n * row + k] /= s_A[n * row + row];
        }
        __syncthreads();

        // inner prod subtraction
        for (uint32_t j = ind + row + 1; j < n; j += stride) {
            for (uint32_t k = 0; k < row + 1; k++) {
                s_A[n * (row + 1) + j] -= s_A[n * k + j] * s_A[n * k + row + 1] * s_A[n * k + k];
            }
        }
        __syncthreads();
    }
    // in place LDL' is finished

    // read the diagonal entries of s_A to s_D
    // replace the diagonal entries of s_A to be 1
    for (uint32_t row = ind; row < n; row += stride) {
        s_D[row] = s_A[n * row + row];
        s_A[n * row + row] = static_cast<T>(1);
    }
}