/**
 * @brief Perform a LDL' decomposition in place.
 *
 * Note:
 * 1. LDL' decomposition is also known as square-root free Cholesky decomposition
 * 2. quasi-definite (eigen values are positive or negative, not zero) matrix admits unique LDL' decomposition
 *
 * Performs a LDL' decomposition on the square matrix @p s_A. Entries of s_A will be changed.
 * The lower triangular matrix L is stored in the original @p S_A
 * The diagonal matrix is stored in @p s_D
 *
 *
 * @param T* s_A: a square symmetric matrix , column - major order.
 * @param int n: number of cols/rows in a square matrix s_A (n*n).
 * @param T* s_D: diagonal matrix of size n.
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

    for (uint32_t col = ind; col < n; col += stride) {
        // read the diagonal entries of s_A to s_D
        s_D[col] = s_A[n * col + col];
        // replace the diagonal entries of s_A to be 1
        s_A[n * col + col] = static_cast<T>(1);
    }
}